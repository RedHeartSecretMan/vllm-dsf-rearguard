"""CPU reference checks for the actual patched functions; no CUDA or torch mocks of math."""

import ast
import hashlib
import json
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTATION = ROOT / "adaptation"
FILES = ADAPTATION / "files"


def functions(path, names, namespace=None):
    tree = ast.parse(path.read_text())
    selected = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if {n.name for n in selected} != set(names):
        raise AssertionError("Missing source function")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    result = {"math": math, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), result)
    return result


class VisionReferenceTests(unittest.TestCase):
    def test_official_reference_hash(self):
        manifest = json.loads((ADAPTATION / "manifest.json").read_text())
        self.assertEqual(
            hashlib.sha256(
                (ADAPTATION / "reference/official_image_processor.py").read_bytes()
            ).hexdigest(),
            manifest["image_reference"]["sha256"],
        )

    def test_geometry_matches_official_including_full_1024_token_budget(self):
        names = ("llm_grid", "num_image_tokens", "solve_resize_ratio", "safe_resize")
        reference = functions(ADAPTATION / "reference/official_image_processor.py", names)
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
    def invoke(computed, new, cache_hit=True, shift=0, offsets=(100,)):
        path = FILES / "vllm/v1/core/sched/scheduler.py"
        schedule = functions(
            path,
            ("_try_schedule_encoder_inputs",),
            {
                "get_mm_features_in_window": lambda features, **kwargs: (
                    0,
                    len(features),
                )
            },
        )["_try_schedule_encoder_inputs"]
        features = [
            SimpleNamespace(
                identifier="same-image",
                mm_position=SimpleNamespace(
                    offset=offset,
                    length=20,
                    get_num_embeds=lambda: 20,
                    get_embeds_indices_in_range=lambda a, b: (a, b),
                ),
            )
            for offset in offsets
        ]
        state = SimpleNamespace(
            is_encoder_decoder=False,
            ec_connector=None,
            scheduler_config=SimpleNamespace(disable_chunked_mm_input=True),
            encoder_cache_manager=SimpleNamespace(
                check_and_update_cache=lambda *a: cache_hit,
                can_allocate=lambda *a: True,
            ),
        )
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


if __name__ == "__main__":
    unittest.main()
