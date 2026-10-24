"""CPU reference checks for the actual patched functions; no CUDA or torch mocks of math."""

import hashlib
import json
import math
import os
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.source_extract import load_functions

ROOT = Path(__file__).resolve().parents[1]
ADAPTATION = ROOT / "adaptation"
FILES = ADAPTATION / "files"


def functions(path, names, namespace=None):
    return load_functions(
        path, names, {"math": math, **(namespace or {})}, anywhere=True
    )


class VisionReferenceTests(unittest.TestCase):
    def test_video_media_metadata_uses_decoder_timestamps(self):
        source = FILES / "vllm/models/deepseek_v4_1/common/mm_preprocess.py"
        functions = load_functions(
            source,
            ("video_frame_timestamps",),
            {"np": np, "Sequence": Sequence},
            anywhere=True,
        )
        timestamps = functions["video_frame_timestamps"]
        self.assertEqual(
            timestamps(3, {"frames_indices": [0, 10, 20], "fps": 10}),
            [0.0, 1.0, 2.0],
        )

    def test_reference_hash(self):
        manifest = json.loads((ADAPTATION / "manifest.json").read_text())
        self.assertEqual(
            hashlib.sha256(
                (ADAPTATION / "reference/image_processor.py").read_bytes()
            ).hexdigest(),
            manifest["image_reference"]["sha256"],
        )

    def test_geometry_matches_reference_including_full_1024_token_budget(self):
        names = ("llm_grid", "num_image_tokens", "solve_resize_ratio", "safe_resize")
        reference = functions(ADAPTATION / "reference/image_processor.py", names)
        candidate = functions(
            FILES / "vllm/models/deepseek_v4_1/common/mm_preprocess.py", names
        )
        for height, width in (
            (588, 3024),
            (1, 32768),
            (32768, 1),
            (1024, 1024),
            (100, 100),
            (4096, 8192),
            (700, 900),
            (300, 10000),
        ):
            args = (
                height,
                width,
                math.ceil(height / 14) * 14,
                math.ceil(width / 14) * 14,
                14,
                3,
                1024,
            )
            with self.subTest(height=height, width=width):
                self.assertEqual(
                    candidate["safe_resize"](*args), reference["safe_resize"](*args)
                )
        grid = candidate["safe_resize"](588, 3024, 588, 3024, 14, 3, 1024)
        self.assertEqual(candidate["num_image_tokens"](*grid[:2]), 1024)


class AtomicImageSchedulingTests(unittest.TestCase):
    @staticmethod
    def invoke(
        computed,
        new,
        cache_hit=True,
        shift=0,
        offsets=(100,),
        chunked=False,
        modality="image",
        frame_lengths=(10, 10),
        can_allocate=True,
        external_cache=False,
        identifiers=None,
        trace=None,
    ):
        trace = {} if trace is None else trace
        trace.update(cache=[], allocations=[], embed_ranges=[])

        def window(features, start, end):
            selected = [
                i for i, feature in enumerate(features)
                if feature.mm_position.offset < end
                and feature.mm_position.offset + feature.mm_position.length > start
            ]
            return (selected[0], selected[-1] + 1) if selected else (0, 0)

        def cache(request, index):
            trace["cache"].append(index)
            return cache_hit

        def allocate(request, index, *args):
            trace["allocations"].append(index)
            return can_allocate

        def embeds(start, end):
            trace["embed_ranges"].append((start, end))
            return start, end

        path = FILES / "vllm/v1/core/sched/scheduler.py"
        schedule = functions(
            path,
            ("_try_schedule_encoder_inputs",),
            {
                "get_mm_features_in_window": window
            },
        )["_try_schedule_encoder_inputs"]
        features = [
            SimpleNamespace(
                identifier=identifiers[index] if identifiers else "same-image",
                modality=modality,
                mm_position=SimpleNamespace(
                    offset=offset,
                    length=sum(frame_lengths) if modality == "video" else 20,
                    get_num_embeds=lambda: sum(frame_lengths)
                    if modality == "video"
                    else 20,
                    get_embeds_indices_in_range=embeds,
                ),
            )
            for index, offset in enumerate(offsets)
        ]
        state = SimpleNamespace(
            is_encoder_decoder=False,
            ec_connector=SimpleNamespace(has_cache_item=lambda identifier: True)
            if external_cache else None,
            scheduler_config=SimpleNamespace(
                disable_chunked_mm_input=not chunked
            ),
            encoder_cache_manager=SimpleNamespace(
                check_and_update_cache=cache,
                can_allocate=allocate,
            ),
        )
        if modality == "video":
            for feature in features:
                feature.data = {
                    "video_frame_span_lengths": SimpleNamespace(data=list(frame_lengths))
                }
        req = SimpleNamespace(has_encoder_inputs=True, mm_features=features)
        return schedule(state, req, computed, new, 1024, shift)

    def test_cached_partial_image_rolls_back_before_cache_shortcut(self):
        self.assertEqual(self.invoke(90, 20)[1], 10)
        self.assertEqual(self.invoke(90, 20, shift=7)[1], 3)

    def test_exact_start_and_cached_prefix_tail_are_atomic(self):
        self.assertEqual(self.invoke(100, 10)[1], 0)
        self.assertEqual(self.invoke(105, 5)[1], 0)
        self.assertEqual(self.invoke(100, 20)[1], 20)

    def test_duplicate_image_cannot_bypass_atomic_boundary(self):
        self.assertEqual(
            self.invoke(90, 120, cache_hit=False, offsets=(100, 200))[1], 110
        )

    def test_chunked_input_allows_span_to_cross_prefill_boundary(self):
        # With chunked multimodal input enabled, the scheduler may schedule
        # only the current token window while the encoder output is cached and
        # sliced by PlaceholderRange in the model runner.
        self.assertEqual(
            self.invoke(90, 20, cache_hit=False, chunked=True, modality="video")[1],
            20,
        )

    def test_chunked_input_keeps_image_span_atomic(self):
        self.assertEqual(
            self.invoke(90, 20, cache_hit=False, chunked=True, modality="image")[1],
            10,
        )

    def test_chunked_input_allows_video_span_to_cross_boundary(self):
        self.assertEqual(
            self.invoke(90, 20, cache_hit=False, chunked=True, modality="video")[1],
            20,
        )

    def test_chunked_video_rolls_back_inside_a_frame(self):
        self.assertEqual(
            self.invoke(90, 15, cache_hit=False, chunked=True, modality="video")[1],
            10,
        )

    def test_chunked_video_uses_actual_multi_frame_boundaries(self):
        frame_lengths = (10, 12, 10)
        self.assertEqual(
            self.invoke(
                90,
                25,
                cache_hit=False,
                chunked=True,
                modality="video",
                frame_lengths=frame_lengths,
            )[1],
            20,
        )
        self.assertEqual(
            self.invoke(
                90,
                32,
                cache_hit=False,
                chunked=True,
                modality="video",
                frame_lengths=frame_lengths,
            )[1],
            32,
        )

    def test_rolled_back_video_still_schedules_cold_encoder_input(self):
        trace = {}
        result = self.invoke(90, 25, cache_hit=False, chunked=True,
                             modality="video", trace=trace)
        self.assertEqual(result, ([0], 20, 1004, []))
        self.assertEqual(trace["embed_ranges"], [(0, 10)])

    def test_rolled_back_video_checks_local_and_external_encoder_caches(self):
        trace = {}
        self.assertEqual(
            self.invoke(90, 25, chunked=True, modality="video", trace=trace),
            ([], 20, 1024, []),
        )
        self.assertEqual(trace["cache"], [0])
        self.assertEqual(
            self.invoke(90, 25, cache_hit=False, chunked=True, modality="video",
                        external_cache=True),
            ([], 20, 1024, [0]),
        )

    def test_rolled_back_video_cannot_bypass_encoder_resource_limit(self):
        self.assertEqual(
            self.invoke(90, 25, cache_hit=False, chunked=True, modality="video",
                        can_allocate=False),
            ([], 10, 1024, []),
        )
        self.assertEqual(
            self.invoke(105, 10, cache_hit=False, chunked=True, modality="video",
                        can_allocate=False),
            ([], 0, 1024, []),
        )

    def test_disabled_chunking_keeps_the_whole_video_atomic(self):
        for end in (110, 115):
            for cache_hit in (False, True):
                with self.subTest(end=end, cache_hit=cache_hit):
                    self.assertEqual(
                        self.invoke(90, end - 90, cache_hit=cache_hit,
                                    chunked=False, modality="video"),
                        ([], 10, 1024, []),
                    )

    def test_frame_rollback_keeps_lookahead_and_embedding_window_consistent(self):
        trace = {}
        self.assertEqual(
            self.invoke(90, 18, shift=7, cache_hit=False, chunked=True,
                        modality="video", trace=trace),
            ([0], 13, 1004, []),
        )
        self.assertEqual(trace["embed_ranges"], [(0, 10)])
        self.assertEqual(
            self.invoke(90, 23, shift=7, cache_hit=False, chunked=False,
                        modality="video"),
            ([0], 23, 1004, []),
        )

    def test_frame_rollback_with_prefix_inside_a_frame(self):
        trace = {}
        self.assertEqual(
            self.invoke(105, 10, cache_hit=False, chunked=True,
                        modality="video", trace=trace),
            ([0], 5, 1004, []),
        )
        self.assertEqual(trace["embed_ranges"], [(5, 10)])
        self.assertEqual(
            self.invoke(105, 3, cache_hit=False, chunked=True, modality="video"),
            ([], 0, 1024, []),
        )

    def test_multiple_videos_schedule_the_rolled_back_last_item(self):
        trace = {}
        self.assertEqual(
            self.invoke(90, 65, cache_hit=False, chunked=True, modality="video",
                        offsets=(100, 140), identifiers=("first", "second"),
                        trace=trace),
            ([0, 1], 60, 984, []),
        )
        self.assertEqual(trace["embed_ranges"], [(0, 20), (0, 10)])
        self.assertEqual(
            self.invoke(90, 65, cache_hit=False, chunked=True, modality="video",
                        offsets=(100, 140)),
            ([0], 60, 1004, []),
        )

    def test_shrunken_window_never_touches_a_later_video(self):
        trace = {}
        # A deliberately overlapping second feature isolates the stale hi/end
        # boundary: once the first item rolls back to 110, offset 112 is out.
        self.assertEqual(
            self.invoke(90, 25, cache_hit=False, chunked=True, modality="video",
                        offsets=(100, 112), identifiers=("first", "second"),
                        trace=trace),
            ([0], 20, 1004, []),
        )
        self.assertEqual(trace["cache"], [0])
        self.assertEqual(trace["allocations"], [0])


class IndexerWorkspaceTests(unittest.TestCase):
    @staticmethod
    def capacity(context, ratio, cap, logits_mb):
        result = functions(
            FILES / "vllm/models/deepseek_v4_1/attention.py",
            ("_indexer_prefill_buffer_size",),
            {
                "os": os,
                "envs": SimpleNamespace(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=logits_mb),
                "get_max_prefill_buffer_size": lambda config: cap,
            },
        )
        return result["_indexer_prefill_buffer_size"](
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=context)), ratio
        )

    def test_valid_multirequest_chunk_fits_beyond_old_divided_cap(self):
        cap = 1048576
        size = self.capacity(262144, 2, cap, 512)
        keys, queries = 5 * 131072, 5
        self.assertLessEqual(queries * keys, 512 * 1024**2 // 4)
        self.assertLessEqual(keys, cap)
        self.assertGreater(keys, cap // 2)  # Previous allocation was too small.
        self.assertLessEqual(keys, size)

    def test_small_logits_budget_keeps_single_request_floor(self):
        size = self.capacity(262144, 2, 1048576, 1)
        self.assertGreaterEqual(size, 131072)
        self.assertLess(size, 1048576)
        for requests in range(1, 17):
            for keys_each in (1, 64, 8192, 131072):
                total = requests * keys_each
                if total <= 1048576 and (
                    requests == 1 or requests * total <= 1024**2 // 4
                ):
                    self.assertLessEqual(total, size)

    def test_bounded_profile_keeps_one_request_floor(self):
        path = FILES / "vllm/models/deepseek_v4_1/attention.py"
        namespace = {
            "os": os,
            "envs": SimpleNamespace(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512),
            "get_max_prefill_buffer_size": lambda config: 40 * config.model_config.max_model_len,
        }
        result = functions(path, ("_indexer_prefill_buffer_size",), namespace)
        with patch.dict(
            os.environ, {"VLLM_DSV41_BOUNDED_PREFILL_WORKSPACE": "1"}
        ):
            self.assertEqual(
                result["_indexer_prefill_buffer_size"](
                    SimpleNamespace(model_config=SimpleNamespace(max_model_len=1048576)),
                    2,
                ),
                524288,
            )

    @staticmethod
    def builder_capacity(context, ratio, cap, logits_mb, bounded):
        path = FILES / "vllm/v1/attention/backends/mla/indexer.py"
        namespace = {
            "os": os,
            "math": math,
            "envs": SimpleNamespace(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=logits_mb),
            "get_max_prefill_buffer_size": lambda config: cap,
        }
        result = functions(path, ("_indexer_prefill_buffer_size",), namespace)
        with patch.dict(
            os.environ,
            {"VLLM_DSV41_BOUNDED_PREFILL_WORKSPACE": "1" if bounded else "0"},
        ):
            return result["_indexer_prefill_buffer_size"](
                SimpleNamespace(model_config=SimpleNamespace(max_model_len=context)),
                ratio,
            )

    def test_builder_uses_bounded_capacity_for_multi_request_chunks(self):
        # A 1M context with ratio 4 owns 262144 compressed K rows.  The
        # metadata builder must use that same bound, otherwise two requests
        # could be packed into one chunk and overflow the op workspace.
        self.assertEqual(
            self.builder_capacity(1048576, 4, 40 * 1048576, 512, True),
            262144,
        )

    def test_chunk_split_keeps_two_long_requests_in_separate_chunks(self):
        path = FILES / "vllm/v1/attention/backends/mla/indexer.py"
        split = functions(
            path, ("_split_indexer_prefill_chunks",)
        )["_split_indexer_prefill_chunks"]
        if isinstance(split, staticmethod):
            split = split.__func__

        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        chunks = split(
            [Scalar(262144), Scalar(262144)],
            [Scalar(1), Scalar(1)],
            262144,
            1024 * 1024 * 1024,
        )
        self.assertEqual(
            [(chunk[0].start, chunk[0].stop) for chunk in chunks],
            [(0, 1), (1, 2)],
        )

    def test_compression_ratio_is_resolved_before_workspace_sizing(self):
        source = (FILES / "vllm/v1/attention/backends/mla/indexer.py").read_text()
        self.assertLess(
            source.index("self.compress_ratio = 1"),
            source.index("self.max_prefill_buffer_size = _indexer_prefill_buffer_size"),
        )


if __name__ == "__main__":
    unittest.main()
