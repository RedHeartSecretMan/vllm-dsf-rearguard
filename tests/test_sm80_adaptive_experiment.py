"""CPU regression checks for the isolated SM80 adaptive A/B contract."""

import importlib.util
import json
import os
import sys
import time
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import test_operations

from scripts.source_extract import load_function, load_method

ROOT = Path(__file__).resolve().parents[1]
GPU_SOURCE = ROOT / "adaptation/files/vllm/v1/worker/gpu"
POLICY_PATH = GPU_SOURCE / "spec_decode/sm80_adaptive.py"


def load_policy():
    spec = importlib.util.spec_from_file_location("sm80_adaptive", POLICY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LaunchExperimentTests(unittest.TestCase):
    def test_both_arms_use_same_graphs_and_unfused_drafter(self):
        settings = (
            "--port",
            "16688",
            "--context",
            "1048576",
            "--memory",
            "0.925",
            "--chunk",
            "16384",
        )
        plan = test_operations.LaunchTests().plan
        default = plan(*settings)["command"]
        control = plan(*settings, "--sm80-adaptive", "control")["command"]
        candidate = plan(*settings, "--sm80-adaptive", "candidate")["command"]
        pos = candidate.index("--speculative-config") + 1
        flags = json.loads(candidate[pos])
        self.assertTrue(flags.pop("enable_adaptive_verification"))
        fixed = json.loads(control[pos])
        self.assertFalse(fixed.pop("enable_adaptive_verification"))
        self.assertEqual(flags, fixed)

        # Each test plan uses a distinct temporary model mount.
        def without_paths_or_spec(command):
            return [
                arg
                for i, arg in enumerate(command)
                if i != pos and not arg.startswith("type=bind,src=")
            ]

        self.assertEqual(
            without_paths_or_spec(candidate), without_paths_or_spec(control)
        )
        self.assertIn("VLLM_DSPARK_FUSED_MARKOV=0", control)
        self.assertIn("VLLM_DSV41_SM80_ADAPTIVE_EXPERIMENT=1", control)
        self.assertNotIn("VLLM_DSPARK_FUSED_MARKOV=0", default)
        graph = json.loads(candidate[candidate.index("--compilation-config") + 1])
        self.assertEqual(graph["cudagraph_mode"], "FULL_DECODE_ONLY")
        self.assertIn(640, graph["cudagraph_capture_sizes"])
        self.assertEqual(graph["max_cudagraph_capture_size"], 768)

    def test_incompatible_arms_fail_before_docker(self):
        plan = test_operations.LaunchTests().plan
        for invalid in (
            ("--dspark", "0"),
            ("--parallel", "tp8"),
            ("--eager",),
            ("--dspark-draft-sample-method", "probabilistic"),
            ("--experimental", "decode-metadata"),
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "SM80 adaptive experiment"),
            ):
                plan("--port", "16688", "--sm80-adaptive", "candidate", *invalid)
        with self.assertRaisesRegex(ValueError, "test port"):
            plan("--sm80-adaptive", "control")


class RuntimeExperimentTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()
        self.spec = NS(
            enable_adaptive_verification=True,
            method="dspark",
            num_speculative_tokens=5,
            draft_sample_method="greedy",
            rejection_sample_method="standard",
            use_local_argmax_reduction=True,
            num_speculative_tokens_per_batch_size=None,
            kv_cache_dtype=None,
            draft_tensor_parallel_size=4,
            draft_model_config=NS(
                model="/local/1017",
                architectures=["DSparkV41DraftModel"],
                hf_config=NS(dspark_block_size=5),
            ),
        )
        self.config = NS(
            speculative_config=self.spec,
            model_config=NS(
                model="/local/1017", architectures=["DeepseekV41ForCausalLM"]
            ),
            use_v2_model_runner=True,
            lora_config=None,
            kv_transfer_config=None,
            compilation_config=NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY"), mode=0),
            parallel_config=NS(
                tensor_parallel_size=4,
                data_parallel_size=2,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                use_ubatching=False,
                enable_batch_sharded_sampling=False,
                enable_expert_parallel=True,
                all2all_backend="allgather_reducescatter",
            ),
            cache_config=NS(cache_dtype="fp8_ds_mla"),
            attention_config=NS(
                use_non_causal=False, resolve_indexer_kv_dtype=lambda default: default
            ),
        )
        self.env = {
            "VLLM_DSV41_SM80_ADAPTIVE_EXPERIMENT": "1",
            "VLLM_DSPARK_FUSED_MARKOV": "0",
            "VLLM_INDEXER_QUERY_SHARD": "0",
        }

    def test_scope_and_fused_markov_requirement(self):
        validate = self.policy.require_sm80_adaptive
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(validate(self.config, (8, 0)))
        with (
            patch.dict(os.environ, self.env, clear=True),
            patch.object(self.policy, "_verify_checkpoint") as fingerprint,
        ):
            self.assertTrue(validate(self.config, (8, 0)))
            fingerprint.assert_called_once_with(Path("/local/1017"))
            with self.assertRaisesRegex(ValueError, "SM80"):
                validate(self.config, (9, 0))
            self.spec.num_speculative_tokens = 1
            with self.assertRaisesRegex(ValueError, "k5"):
                validate(self.config, (8, 0))
            self.spec.num_speculative_tokens = 5
            with (
                patch.dict(os.environ, {"VLLM_DSPARK_FUSED_MARKOV": "1"}),
                self.assertRaisesRegex(ValueError, "FUSED_MARKOV=0"),
            ):
                validate(self.config, (8, 0))

    def test_verify_budget_and_full_vote_follow_the_prefill_state(self):
        select = self.policy.sm80_adaptive_batch
        self.assertEqual(
            select(False, has_prefill=True, has_drafts=True), (True, False)
        )
        self.assertEqual(select(True, has_prefill=True, has_drafts=True), (False, True))
        self.assertEqual(
            select(True, has_prefill=False, has_drafts=True), (True, False)
        )
        self.assertEqual(
            select(True, has_prefill=False, has_drafts=False), (False, False)
        )


class QueryStartLocSyncTests(unittest.TestCase):
    def test_uneven_gpu_boundaries_replace_cpu_tail_after_reallocation(self):
        sync = load_method(
            GPU_SOURCE / "spec_decode/adaptive_verification.py",
            "AdaptiveVerificationManager",
            "get_query_start_loc_cpu",
            {"gpu_sync_allowed": nullcontext, "np": np},
        )

        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.int32)

            def __getitem__(self, index):
                return FakeTensor(self.values[index])

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        manager = NS(
            req_states=NS(max_num_reqs=7),
            query_start_loc=FakeTensor([0, 2, 5, 9, 9, 9, 9, 9]),
        )
        result = sync(manager, num_reqs=3, num_tokens=9)
        np.testing.assert_array_equal(result, [0, 2, 5, 9, 9, 9, 9, 9])


class TargetGroupsTests(unittest.TestCase):
    def test_complete_known_set_and_fail_closed_extra_cache(self):
        # Use distinct class identities, as the runtime does for its allowlist.
        types = {}
        for name in (
            "DeepseekV4ROCMAiterSparseSWAMetadataBuilder",
            "DeepseekV41ROCMAiterSparseSWABackend",
            "DeepseekV41AmpereMLASparseBackend",
            "CompressorBackend",
            "CompressorMetadataBuilder",
            "DeepseekV4SparseMLAMetadataBuilder",
            "DeepseekV32IndexerMetadataBuilder",
            "DeepseekV41IndexerBackend",
        ):
            types[name] = type(name, (), {})
        modules = dict.fromkeys(
            (
                "vllm.models.deepseek_v4_1.amd.rocm",
                "vllm.models.deepseek_v4_1.ampere.ampere_sparse",
                "vllm.models.deepseek_v4_1.compressor",
                "vllm.models.deepseek_v4_1.sparse_mla",
                "vllm.v1.attention.backends.mla.indexer",
            ),
            NS(**types),
        )
        always = object()
        validate = load_function(
            GPU_SOURCE / "spec_decode/adaptive_verification.py",
            "_validate_sm80_target_groups",
            {
                "torch": NS(uint8="uint8", float32="float32"),
                "AttentionCGSupport": NS(ALWAYS=always),
            },
        )
        cases = (
            (
                "DeepseekV41IndexerBackend",
                "DeepseekV32IndexerMetadataBuilder",
                NS(tokens_per_state=2, dtype="uint8"),
                dict(
                    use_flattening=True,
                    supports_varlen=False,
                    use_fp4_indexer_cache=False,
                    indexer_shard_size=1,
                    kernel_block_size=128,
                ),
            ),
            (
                "DeepseekV41AmpereMLASparseBackend",
                "DeepseekV4SparseMLAMetadataBuilder",
                NS(tokens_per_state=1, dtype="uint8"),
                {},
            ),
            (
                "DeepseekV41ROCMAiterSparseSWABackend",
                "DeepseekV4ROCMAiterSparseSWAMetadataBuilder",
                NS(dtype="uint8"),
                dict(window_size=128, supports_draft_decode_metadata_update=False),
            ),
            (
                "CompressorBackend",
                "CompressorMetadataBuilder",
                NS(dtype="float32", block_size=8),
                dict(get_cudagraph_support=Mock(return_value=always)),
            ),
        )
        types["CompressorBackend"].supports_device_cpu_query_lens_mismatch = Mock(
            return_value=True
        )
        groups = []
        for index, (backend, builder, spec, fields) in enumerate(cases):
            md_builder = types[builder]()
            vars(md_builder).update(kv_cache_spec=spec, **fields)
            groups.append(
                NS(
                    backend=types[backend],
                    layer_names=[str(index)],
                    get_metadata_builder=Mock(return_value=md_builder),
                )
            )
        extra = NS(
            backend=type("UnknownStorage", (), {}),
            layer_names=["extra"],
            get_metadata_builder=Mock(return_value=object()),
        )
        layer_names = {"0", "1", "2", "3"}
        with patch.dict(sys.modules, modules):
            validate([groups, [extra]], layer_names, (always, None), object())
            with self.assertRaisesRegex(ValueError, "UnknownStorage"):
                validate(
                    [groups, [extra]], layer_names | {"extra"}, (always, None), object()
                )
            with self.assertRaisesRegex(ValueError, "complete 1017"):
                validate([groups[:3]], layer_names, (always, None), object())
            groups[0].get_metadata_builder.return_value.use_flattening = False
            with self.assertRaisesRegex(ValueError, "geometry"):
                validate([groups], layer_names, (always, None), object())


class RunnerLifecycleTests(unittest.TestCase):
    def test_prefill_eager_vote_reaches_the_existing_dp_dispatch(self):
        vote = load_policy().sm80_adaptive_batch
        dispatch = Mock(return_value=(NS(num_tokens=0), None))
        execute = load_method(
            GPU_SOURCE / "model_runner.py",
            "GPUModelRunner",
            "execute_model",
            {
                "torch": NS(inference_mode=lambda: lambda fn: fn),
                "sm80_adaptive_batch": vote,
                "dispatch_cg_and_sync_dp": dispatch,
            },
        )
        runner = NS(
            sm80_adaptive=True,
            dp_size=2,
            dp_rank=0,
            decode_query_len=6,
            pcp_manager=None,
            lora_config=None,
            ubatch_runner=None,
            is_encoder_decoder=False,
            cudagraph_manager=object(),
            parallel_config=object(),
            update_pp_decode_requests=Mock(),
            finish_requests=Mock(),
            free_states=Mock(),
            add_requests=Mock(),
            update_requests=Mock(),
            block_tables=NS(apply_staged_writes=Mock()),
            kv_connector=NS(no_forward=Mock()),
            _merge_ec_connector_no_forward=Mock(),
        )
        scheduler = NS(total_num_scheduled_tokens=2, num_scheduled_tokens={"short": 2})
        for active, has_prefill, dummy, expected in (
            (True, True, False, True),
            (True, False, False, False),
            (True, True, True, False),
            (False, True, False, False),
        ):
            with self.subTest(active=active, has_prefill=has_prefill, dummy=dummy):
                runner.sm80_adaptive = active
                batch_state = (
                    None if dummy else NS(num_tokens=2, has_prefill=has_prefill)
                )
                runner.gather_batch_req_state = Mock(return_value=(batch_state, None))
                execute(runner, scheduler, dummy_run=dummy)
                self.assertEqual(dispatch.call_args.kwargs["need_eager"], expected)
                self.assertEqual(dispatch.call_args.kwargs["max_query_len"], 2)

    def test_only_the_real_capture_seeds_cost_curves(self):
        calls = []

        @contextmanager
        def collect():
            calls.append("collect")
            yield ["sampled"]

        capture = load_method(
            GPU_SOURCE / "model_runner.py",
            "GPUModelRunner",
            "capture_model",
            {
                "torch": NS(
                    inference_mode=lambda: lambda fn: fn,
                    accelerator=NS(empty_cache=Mock(), get_memory_info=lambda: (0, 0)),
                ),
                "compilation_counter": NS(num_gpu_runner_capture_triggers=0),
                "time": time,
                "freeze_gc_for_cudagraph_capture": nullcontext,
                "use_workspace_lane": lambda _: nullcontext(),
                "lock_workspace": lambda: calls.append("lock"),
                "create_lora_capture_hook": Mock(),
                "logger": Mock(),
            },
        )
        runner = NS(
            sm80_adaptive=True,
            model_state=NS(supports_mm_inputs=False),
            cudagraph_manager=NS(
                needs_capture=lambda: True,
                capture=lambda *_a, **_kw: calls.append("target"),
                captured_token_counts=lambda: [6],
            ),
            speculator=NS(capture=lambda: calls.append("draft")),
            adaptive_verification=NS(
                batches_to_profile=lambda _: [{"num_tokens": 6}],
                set_initial_cost_curves=lambda _: calls.append("seed"),
            ),
            step_timing=NS(collect=collect),
            pcp_manager=None,
            lora_config=None,
            maybe_setup_dummy_loras=lambda _: nullcontext(),
            use_aux_hidden_state_outputs=True,
            input_buffers=None,
            model=None,
            intermediate_tensors=None,
            block_tables=None,
            attn_groups=[],
            kv_cache_config=None,
            _draft_workspace_lane=1,
            _dummy_run=lambda **_: calls.append("replay"),
        )
        capture(runner, profile_only=True)
        self.assertEqual(calls, ["target", "draft"])
        calls.clear()
        capture(runner)
        self.assertEqual(
            calls, ["target", "draft", "collect", "replay", "seed", "lock"]
        )

    def test_confidence_buffer_is_reset_before_the_capture_writer(self):
        calls = []
        fake_buffer = Mock()
        fake_buffer.__getitem__ = Mock(
            return_value=NS(fill_=lambda value: calls.append(("poison", value)))
        )

        # A tiny object implements the actual tensor boundary used by this method.
        class ConfidenceBuffer:
            def __getitem__(self, index):
                return fake_buffer.__getitem__(index)

        run = load_method(
            GPU_SOURCE / "spec_decode/dspark/speculator.py",
            "DSparkSpeculator",
            "_generate_draft",
            {
                "record_function": lambda _: nullcontext(),
                "CUDAGraphMode": NS(NONE="NONE"),
            },
        )
        spec = NS(
            _check_experiment_confidence=True,
            draft_token_confidence_probs=ConfidenceBuffer(),
            _run_model=lambda *_: "hidden",
            _sample_sequential=lambda n, hidden: calls.append(("sample", n, hidden)),
        )
        run(spec, 3, 15, None, None, None)
        self.assertEqual(calls[0][0], "poison")
        self.assertNotEqual(calls[0][1], calls[0][1])
        self.assertEqual(calls[1], ("sample", 3, "hidden"))
        fake_buffer.__getitem__.assert_called_once_with(slice(None, 3))
        calls.clear()
        spec._check_experiment_confidence = False
        run(spec, 3, 15, None, None, None)
        self.assertEqual(calls, [("sample", 3, "hidden")])
