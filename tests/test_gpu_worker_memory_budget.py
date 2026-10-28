"""Exercise the actual worker budget methods with CPU-only profiling receipts.

These checks validate accounting and call counts, not CUDA graph lifetime or
runtime memory safety; those require the separately pinned GPU reproduction.
"""

import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from scripts.source_extract import load_method

SOURCE = Path(__file__).resolve().parents[1] / "adaptation/files/vllm/v1/worker/gpu_worker.py"
GIB = 1 << 30
MIB = 1 << 20


class BudgetHarness:
    def __init__(self, *, enabled=True, graph_mode="FULL", eager=False,
                 estimate=GIB, actual=3 * GIB // 2, requested=12 * GIB,
                 parallel="tp4dp2", dspark=5, manual_kv=None, init_non_kv=0):
        self.profile = NS(total_consumed=5 * GIB, transient_peak_headroom=2 * GIB,
                          non_kv_cache_memory=7 * GIB, after_profile=NS(free_memory=11 * GIB))
        compilation = NS(mode=0, backend="none", cudagraph_mode=graph_mode,
                         compilation_time=0, encoder_compilation_time=0)
        self.runner = NS(model_memory_usage=4 * GIB, profile_run=Mock(),
                         profile_cudagraph_memory=Mock(return_value=estimate),
                         kv_cache_init_non_kv_memory_bytes=init_non_kv,
                         capture_model=Mock(return_value=actual),
                         maybe_remove_all_loras=Mock(), lora_config=None)
        self.worker = NS(
            cache_config=NS(kv_cache_memory_bytes=manual_kv, gpu_memory_utilization=.6),
            model_config=NS(multimodal_config=None, enforce_eager=eager, seed=1017),
            parallel_config=NS(_api_process_count=1, tensor_parallel_size=4 if parallel == "tp4dp2" else 8,
                               data_parallel_size=2 if parallel == "tp4dp2" else 1),
            vllm_config=NS(compilation_config=compilation, speculative_config=NS(num_speculative_tokens=dspark)),
            init_snapshot=NS(free_memory=16 * GIB, total_memory=20 * GIB),
            requested_memory=requested, model_runner=self.runner,
            compilation_config=compilation, use_v2_model_runner=True,
            execute_model=Mock(), sample_tokens=Mock(), device="cpu",
            observability_config=NS(jit_monitor_mode="warn", jit_monitor_verbose=False),
        )
        self.save_plan = Mock()
        self.ipc = Mock(side_effect=lambda value, *_: value)
        self.logger = Mock()
        self.namespace = {
            "torch": NS(inference_mode=lambda: (lambda function: function)),
            "instrument": lambda **_: (lambda function: function),
            "envs": NS(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=enabled),
            "current_platform": NS(is_cuda_alike=lambda: True),
            "CUDAGraphMode": NS(NONE="NONE"), "CompilationMode": NS(NONE=0, VLLM_COMPILE=1),
            "memory_profiling": lambda *_args, **_kwargs: nullcontext(self.profile),
            "maybe_apply_startup_plan": Mock(), "maybe_save_startup_plan": self.save_plan,
            "reserve_mm_ipc_gpu_memory": self.ipc, "maybe_rocm_profiling_fallback": lambda _: None,
            "format_gib": lambda value: round(value / GIB, 2), "GiB_bytes": GIB, "logger": self.logger,
            "get_pp_group": lambda: NS(world_size=1, is_last_rank=True),
            "kernel_warmup": Mock(), "warmup_kernels": Mock(), "set_random_seed": Mock(),
            "freeze_gc_heap": Mock(), "maybe_attach_gc_debug_callback": Mock(), "enable_gpu_sync_check": Mock(),
            "set_torch_threads_for_runtime": Mock(), "CompilationTimes": lambda **kwargs: kwargs,
        }
        self.determine = load_method(SOURCE, "Worker", "determine_available_memory", self.namespace)
        self.capture = load_method(SOURCE, "Worker", "compile_or_warm_up_model", self.namespace)

    def finish_capture(self):
        modules = {name: types.ModuleType(name) for name in ("vllm", "vllm.utils", "vllm.utils.jit_monitor")}
        modules["vllm.utils.jit_monitor"].activate = Mock()
        with patch.dict(sys.modules, modules):
            return self.capture(self.worker)


class GpuWorkerMemoryBudgetTests(unittest.TestCase):
    def test_reported_equivalent_utilization_preserves_the_actual_kv_budget(self):
        h = BudgetHarness(init_non_kv=768 * MIB)
        actual = h.determine(h.worker)
        message = next(call for call in h.logger.info.call_args_list
                       if "equivalent to" in str(call.args[0]))
        equivalent_util = message.args[2]
        legacy = BudgetHarness(enabled=False, requested=round(
            equivalent_util * h.worker.init_snapshot.total_memory))
        legacy.worker.cache_config.gpu_memory_utilization = equivalent_util
        self.assertAlmostEqual(legacy.determine(legacy.worker), actual,
                               delta=h.worker.init_snapshot.total_memory * .0001)

    def test_kv_initialization_reserve_is_charged_once_outside_activation_and_graph(self):
        h = BudgetHarness(init_non_kv=768 * MIB)
        self.assertEqual(h.determine(h.worker), 3328 * MIB)
        self.assertEqual(h.worker.kv_cache_init_non_kv_memory_bytes, 768 * MIB)
        self.assertEqual(h.worker.peak_activation_memory, 2 * GIB)
        self.assertEqual(h.worker.cudagraph_memory_estimate, GIB)

    def test_saved_plan_keeps_initialization_and_retained_capture_setup_memory(self):
        h = BudgetHarness(estimate=2 * GIB, actual=GIB, init_non_kv=768 * MIB)
        self.assertEqual(h.determine(h.worker), 2304 * MIB)
        h.finish_capture()
        # The second capture delta can be smaller because setup buffers from
        # profiling remain live; it must not erase their original reserve.
        h.save_plan.assert_called_once_with(h.worker, 2154 * MIB)

    def test_saved_plan_also_keeps_init_when_actual_graph_grows(self):
        h = BudgetHarness(estimate=GIB, actual=2 * GIB, init_non_kv=768 * MIB)
        self.assertEqual(h.determine(h.worker), 3328 * MIB)
        h.finish_capture()
        h.save_plan.assert_called_once_with(h.worker, 2154 * MIB)

    def test_skipped_profiling_does_not_read_stale_runner_or_worker_init_stats(self):
        for options in ({"enabled": False}, {"graph_mode": "NONE", "eager": True},
                        {"manual_kv": 3 * GIB}):
            with self.subTest(options=options):
                h = BudgetHarness(init_non_kv=9 * GIB, **options)
                h.worker.kv_cache_init_non_kv_memory_bytes = 10 * GIB
                self.assertEqual(h.determine(h.worker), 3 * GIB if "manual_kv" in options else 5 * GIB)
                self.assertEqual(h.worker.kv_cache_init_non_kv_memory_bytes, 0)
                h.runner.profile_cudagraph_memory.assert_not_called()
                h.finish_capture()

    def test_other_runner_without_init_stat_keeps_existing_accounting(self):
        h = BudgetHarness()
        del h.runner.kv_cache_init_non_kv_memory_bytes
        self.assertEqual(h.determine(h.worker), 4 * GIB)
        self.assertEqual(h.worker.kv_cache_init_non_kv_memory_bytes, 0)

    def test_init_reserve_exhaustion_rejects_before_kv_allocation(self):
        h = BudgetHarness(init_non_kv=4 * GIB)
        with self.assertRaisesRegex(ValueError, "KV-init non-KV"):
            h.determine(h.worker)
        h.ipc.assert_not_called()
        h.save_plan.assert_not_called()

    def test_graph_estimate_is_charged_once_and_activation_keeps_its_own_unit(self):
        h = BudgetHarness()
        budget = h.determine(h.worker)
        self.assertEqual(budget, 4 * GIB)
        self.assertEqual(h.worker.peak_activation_memory, 2 * GIB)
        self.assertEqual(h.worker.cudagraph_memory_estimate, GIB)
        h.runner.profile_run.assert_called_once_with()
        h.runner.profile_cudagraph_memory.assert_called_once_with()
        self.assertEqual(h.ipc.call_args.args[0], 4 * GIB)

    def test_postcapture_proposal_and_saved_plan_charge_actual_graph_once(self):
        h = BudgetHarness(estimate=GIB, actual=3 * GIB // 2)
        h.determine(h.worker)
        h.finish_capture()
        expected = 12 * GIB - 5 * GIB - 2 * GIB - 3 * GIB // 2 - 150 * MIB
        h.save_plan.assert_called_once_with(h.worker, expected)
        h.runner.capture_model.assert_called_once_with()
        h.runner.profile_cudagraph_memory.assert_called_once_with()
        messages = [str(call.args[0]) for call in h.logger.info.call_args_list if call.args]
        self.assertTrue(any(f"--kv-cache-memory={expected}" in message for message in messages))

    def test_parallel_and_dspark_variants_use_their_measured_estimate(self):
        for parallel, k, graph in (("tp4dp2", 5, 1900 * MIB), ("tp8", 5, 3100 * MIB),
                                   ("tp4dp2", 0, 700 * MIB), ("tp8", 0, 1100 * MIB)):
            with self.subTest(parallel=parallel, dspark=k):
                h = BudgetHarness(parallel=parallel, dspark=k, estimate=graph, actual=graph)
                self.assertEqual(h.determine(h.worker), 5 * GIB - graph)
                h.finish_capture()
                h.save_plan.assert_called_once_with(h.worker, 5 * GIB - graph - 150 * MIB)
                h.runner.profile_cudagraph_memory.assert_called_once_with()

    def test_disabled_estimator_keeps_existing_opt_in_boundary(self):
        h = BudgetHarness(enabled=False, estimate=10 * GIB)
        self.assertEqual(h.determine(h.worker), 5 * GIB)
        self.assertEqual(h.worker.peak_activation_memory, 2 * GIB)
        h.runner.profile_cudagraph_memory.assert_not_called()
        h.finish_capture()
        h.save_plan.assert_called_once_with(h.worker, 5 * GIB - 3 * GIB // 2 - 150 * MIB)

    def test_graph_disabled_does_not_profile_or_charge_graph_memory(self):
        h = BudgetHarness(graph_mode="NONE", eager=True, estimate=10 * GIB)
        self.assertEqual(h.determine(h.worker), 5 * GIB)
        h.finish_capture()
        h.runner.profile_cudagraph_memory.assert_not_called()
        h.runner.capture_model.assert_not_called()
        h.save_plan.assert_called_once_with(h.worker, 5 * GIB - 150 * MIB)

    def test_none_mode_without_eager_uses_zero_actual_capture_cost(self):
        h = BudgetHarness(graph_mode="NONE", eager=False, estimate=10 * GIB, actual=0)
        self.assertEqual(h.determine(h.worker), 5 * GIB)
        h.finish_capture()
        h.runner.profile_cudagraph_memory.assert_not_called()
        h.runner.capture_model.assert_called_once_with()
        h.save_plan.assert_called_once_with(h.worker, 5 * GIB - 150 * MIB)

    def test_manual_kv_budget_remains_manual_and_does_not_trigger_graph_estimation(self):
        h = BudgetHarness(manual_kv=3 * GIB)
        self.assertEqual(h.determine(h.worker), 3 * GIB)
        h.runner.profile_run.assert_called_once_with()
        h.runner.profile_cudagraph_memory.assert_not_called()
        h.finish_capture()
        h.save_plan.assert_not_called()

    def test_exhausted_profile_budget_fails_before_returning_invalid_kv_bytes(self):
        for requested in (8 * GIB, 8 * GIB - 1):
            with self.subTest(requested=requested):
                h = BudgetHarness(requested=requested)
                with self.assertRaisesRegex(ValueError, "KV cache.*profiling"):
                    h.determine(h.worker)
                h.ipc.assert_not_called()
                h.save_plan.assert_not_called()

    def test_postcapture_exhaustion_does_not_save_an_invalid_plan(self):
        h = BudgetHarness(actual=7 * GIB)
        self.assertGreater(h.determine(h.worker), 0)
        with self.assertRaisesRegex(ValueError, "KV cache.*CUDA graph capture"):
            h.finish_capture()
        h.save_plan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
