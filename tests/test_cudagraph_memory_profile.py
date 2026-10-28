"""Run the real graph profiler/capture loop with CPU graph and allocator doubles.

These checks cover capture coverage, memory accounting and cleanup control flow.
They do not establish CUDA pointer lifetime safety or replace GPU acceptance.
"""

import ast
import itertools
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import Mock

from scripts.source_extract import load_functions, load_method

SOURCE = Path(__file__).resolve().parents[1] / "adaptation/files/vllm/v1/worker/gpu/cudagraph_utils.py"
MIB = 1 << 20


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    cg_mode: Mode = Mode.FULL


@dataclass(frozen=True)
class Device:
    type: str = "cuda"
    index: int | None = 0


class TensorStorage:
    def __init__(self, pointer, size):
        self.pointer, self.size = pointer, size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class CacheTensor:
    def __init__(self, storage, *, device=None, offset=0):
        self.storage, self.device, self.offset = storage, device or Device(), offset

    def untyped_storage(self):
        return self.storage

    def data_ptr(self):
        return self.storage.data_ptr() + self.offset


class ProfileHarness:
    def __init__(self, *, mode=Mode.FULL, fail_at=None, measured_override=None):
        self.persistent_pool, self.temporary_pool = object(), object()
        persistent = self.persistent_pool

        class Platform:
            _global_graph_pool = persistent

            def graph_pool_handle(inner):
                return self.temporary_pool

        class Manager:
            pass

        self.platform = Platform()
        self.descriptors = [Descriptor(value) for value in (768, 384, 192, 96, 48)]
        self.costs = [1700 * MIB, 90 * MIB, 2 * MIB, 2 * MIB, 2 * MIB]
        self.captured = []
        self.prepared = []
        self.free_memory = 10 * 1024 * MIB
        self.in_capture = False
        self.fail_at = fail_at
        self.measured_override = measured_override
        self.manager = Manager()
        self.manager.device = "cpu"
        self.manager.pool = self.persistent_pool
        self.manager.graphs = {}
        self.manager.use_breakable_cg = False
        self.manager._capture_descs = {Mode.FULL: self.descriptors}
        # Required only by the original implementation during the RED check.
        self.manager._max_full_descs_to_capture = None
        self.manager._capture_mem_samples = None
        self.manager.needs_capture = lambda: bool(self.manager._capture_descs)
        self.counter = NS(num_cudagraph_captured=7, num_gpu_runner_capture_triggers=3)
        self.wrapper = NS(graph_pool=object(), entries={"old": object()})
        self.breakable_wrapper = NS(graph_pool=object(), entries={"old": object()})
        self.original_wrapper_pools = (self.wrapper.graph_pool, self.breakable_wrapper.graph_pool)
        self.clear_wrappers = Mock(side_effect=self.wrapper.entries.clear)
        self.clear_breakable = Mock(side_effect=self.breakable_wrapper.entries.clear)
        self.layer = NS(kv_cache=object())
        self.encoder_clear = Mock()
        self.runner = NS(device=Device(), compilation_config=NS(cudagraph_mode=mode, static_forward_context={"layer": self.layer}),
                         vllm_config=NS(), cudagraph_manager=self.manager,
                         model=NS(modules=lambda: [self.layer]), lora_config=None,
                         model_state=NS(supports_mm_inputs=True, encoder_runner=NS(clear=self.encoder_clear),
                                        _mamba_ctx=object(), _mamba_group_ids=[1], _mamba_spec=object()),
                         cache_config=NS(num_gpu_blocks=None), maybe_remove_all_loras=Mock(),
                         speculator=NS(graph_manager=Manager(), unrelated="retain"))
        self.init = Mock(side_effect=self.initialize)
        self.empty_cache = Mock()
        self.sync = Mock()
        self.offloader = NS(sync_prev_onload=Mock(), join_after_forward=Mock())
        self.capture_calls = []
        self.namespace = {
            "torch": NS(inference_mode=lambda: (lambda function: function),
                        accelerator=NS(empty_cache=self.empty_cache, synchronize=self.sync,
                                       get_memory_info=lambda: (self.free_memory, 20 * 1024 * MIB)),
                        cuda=NS(CUDAGraph=object, graph=self.graph)),
            "gc": NS(collect=Mock()), "current_platform": self.platform,
            "set_current_vllm_config": lambda _: nullcontext(),
            "_init_minimal_kv_cache_for_profiling": self.init,
            "compilation_counter": self.counter, "CudaGraphManager": Manager,
            "CUDAGraphWrapper": NS(_all_instances=[self.wrapper], clear_all_graphs=self.clear_wrappers),
            "BreakableCUDAGraphWrapper": NS(_all_instances=[self.breakable_wrapper], clear_all_graphs=self.clear_breakable),
            "CUDAGraphMode": Mode, "graph_capture": lambda **_: nullcontext(),
            "is_global_first_rank": lambda: False, "tqdm": lambda values, **_: values,
            "logger": Mock(), "get_offloader": lambda: self.offloader,
            "set_graph_pool_id": Mock(), "current_stream": lambda: "cpu-stream",
            "itertools": itertools, "clear_layer_kv_caches": self.clear_layers,
            "_FULL_GRAPH_PROFILING_SAMPLES": 2, "_MIN_PER_GRAPH_BYTES": MIB,
        }
        self.manager.capture = MethodType(load_method(SOURCE, "CudaGraphManager", "capture", self.namespace), self.manager)
        names = ["profile_cudagraph_memory", "_teardown_profiling_state"]
        if any(isinstance(node, ast.FunctionDef) and node.name == "_kv_cache_storage_bytes"
               for node in ast.parse(SOURCE.read_text()).body):
            names.append("_kv_cache_storage_bytes")
        # Load the real old extrapolator for the baseline RED run; its removal
        # requires no test-only replacement implementation.
        if any(isinstance(node, ast.FunctionDef) and node.name == "_extrapolate_full_graph_memory"
               for node in ast.parse(SOURCE.read_text()).body):
            names.append("_extrapolate_full_graph_memory")
        self.namespace = load_functions(SOURCE, names, self.namespace)
        self.runner.capture_model = self.capture_model

    def initialize(self, runner):
        self.assert_temporary_pool()
        runner.kv_caches = [CacheTensor(TensorStorage(1000, 0))]
        runner.attn_groups = [object()]
        runner.kv_cache_config = object()
        runner.cache_config.num_gpu_blocks = 8

    def assert_temporary_pool(self):
        if type(self.platform)._global_graph_pool is not self.temporary_pool:
            raise AssertionError("profiling must use its throwaway global pool")

    def clear_layers(self, layers):
        for layer in layers:
            layer.kv_cache = None

    @contextmanager
    def graph(self, graph, pool, **kwargs):
        self.assert_temporary_pool()
        if pool is not self.temporary_pool:
            raise AssertionError("capture escaped the profiling pool")
        self.in_capture = True
        try:
            yield
        finally:
            self.in_capture = False

    def forward_factory(self, desc, warmup):
        self.prepared.append((desc.num_tokens, warmup))

        def forward(mode):
            if self.in_capture:
                if desc.num_tokens == self.fail_at:
                    raise RuntimeError("injected capture failure")
                self.captured.append(desc.num_tokens)
                self.free_memory -= self.costs[self.descriptors.index(desc)]
        return forward

    def capture_model(self, *, profile_only=False):
        self.capture_calls.append(profile_only)
        if not profile_only:
            raise AssertionError("profiling must not lock the runtime workspace")
        self.assert_temporary_pool()
        if self.wrapper.graph_pool is not self.temporary_pool or self.breakable_wrapper.graph_pool is not self.temporary_pool:
            raise AssertionError("wrapper pool was not isolated")
        self.counter.num_gpu_runner_capture_triggers += 1
        before = self.free_memory
        self.manager.capture(self.forward_factory)
        if self.measured_override is not None:
            # Model a net driver-memory change, e.g. another allocation released,
            # rather than an inconsistent return value with unchanged counters.
            self.free_memory = before - self.measured_override
        return before - self.free_memory

    def run(self):
        return self.namespace["profile_cudagraph_memory"](self.runner)


class CachedProfileHarness(ProfileHarness):
    """Native allocator contract: free ordinary cache, keep live graph storage."""

    def __init__(self, *, warmup_cache=300 * MIB, persistent=0, init_retained=0, fail_post_flush=False):
        super().__init__()
        self.warmup_cache = warmup_cache
        self.persistent = persistent
        self.init_retained = init_retained
        self.normal_cache = 0
        self.graph_bytes = 0
        self.kv_bytes = 32 * MIB
        self.kv_live = False
        self.capture_complete = False
        self.fail_post_flush = fail_post_flush
        self.events = []
        self.samples = []
        self.empty_cache.side_effect = self.flush_cache
        self.sync.side_effect = lambda: self.events.append("synchronize")
        self.namespace["torch"].accelerator.get_memory_info = self.memory_info

    def initialize(self, runner):
        super().initialize(runner)
        runner.kv_caches = [CacheTensor(TensorStorage(1000, self.kv_bytes))]
        self.kv_live = True
        self.free_memory -= self.kv_bytes + self.init_retained
        # Minimal-cache initialization can itself leave releasable ordinary
        # cache. It must be removed before the independent starting sample.
        self.normal_cache = 11 * MIB
        self.free_memory -= self.normal_cache

    def flush_cache(self):
        if self.in_capture:
            raise AssertionError("cache flushing must happen outside graph capture")
        alive = self.runner.cudagraph_manager is not None and bool(self.manager.graphs)
        if self.capture_complete and alive:
            if not self.events or self.events[-1] != "synchronize":
                raise AssertionError("capture work must finish before the measurement flush")
            if self.clear_wrappers.called or self.clear_breakable.called or not self.runner.kv_caches:
                raise AssertionError("teardown ran before the retained-memory sample")
            if self.fail_post_flush:
                raise RuntimeError("injected post-capture cache flush failure")
        self.events.append("empty_cache")
        self.free_memory += self.normal_cache
        self.normal_cache = 0
        if not alive:
            self.free_memory += self.graph_bytes
            self.graph_bytes = 0
        if self.kv_live and not self.runner.kv_caches:
            self.free_memory += self.kv_bytes
            self.kv_live = False

    def memory_info(self):
        self.events.append("memory_info")
        self.samples.append({"free": self.free_memory, "normal_cache": self.normal_cache,
                             "graph_bytes": self.graph_bytes, "kv_live": self.kv_live,
                             "graphs_alive": self.runner.cudagraph_manager is not None and bool(self.manager.graphs)})
        return self.free_memory, 20 * 1024 * MIB

    def capture_model(self, *, profile_only=False):
        # The pinned capture_model also flushes at its own entry.
        self.empty_cache()
        measured = super().capture_model(profile_only=profile_only)
        self.graph_bytes = measured
        self.normal_cache += self.warmup_cache
        self.free_memory -= self.warmup_cache + self.persistent
        self.capture_complete = True
        return measured + self.warmup_cache + self.persistent


class CudagraphMemoryProfileTests(unittest.TestCase):
    def test_minimum_kv_init_retained_memory_is_separate_from_kv_and_graphs(self):
        h = CachedProfileHarness(init_retained=97 * MIB)
        self.assertEqual(h.run(), 1796 * MIB)
        self.assertEqual(h.runner.kv_cache_init_non_kv_memory_bytes, 97 * MIB)
        self.assert_cleaned(h)

    def test_shared_offset_and_nested_kv_views_count_backing_storage_once(self):
        # The pinned allocator builds per-layer views of one flat KV buffer.
        # Its storage size, not the sum of layer view sizes or offsets, is KV.
        h = CachedProfileHarness(init_retained=97 * MIB + 256)

        def initialize(runner):
            h.initialize(runner)
            storage = TensorStorage(1000, 32 * MIB)
            runner.kv_caches = [
                CacheTensor(storage),
                [CacheTensor(storage, offset=4096), (CacheTensor(storage, offset=8192),)],
                CacheTensor(TensorStorage(2000, 3 * MIB), device=Device("cpu", None)),
                CacheTensor(TensorStorage(1000, 5 * MIB), device=Device("cuda", 1)),
            ]

        h.init.side_effect = initialize
        self.assertEqual(h.run(), 1796 * MIB)
        # The extra allocator rounding remains part of the init reserve.
        self.assertEqual(h.runner.kv_cache_init_non_kv_memory_bytes, 97 * MIB + 256)
        self.assert_cleaned(h)

    def test_distinct_kv_storages_are_all_excluded_from_init_reserve(self):
        h = CachedProfileHarness(init_retained=23 * MIB)

        def initialize(runner):
            h.initialize(runner)
            runner.kv_caches = [CacheTensor(TensorStorage(1000, 12 * MIB)),
                                CacheTensor(TensorStorage(2000, 20 * MIB))]

        h.init.side_effect = initialize
        self.assertEqual(h.run(), 1796 * MIB)
        self.assertEqual(h.runner.kv_cache_init_non_kv_memory_bytes, 23 * MIB)
        self.assert_cleaned(h)

    def test_init_stat_is_reset_and_negative_init_delta_is_not_a_credit(self):
        h = CachedProfileHarness(init_retained=-MIB)
        h.runner.kv_cache_init_non_kv_memory_bytes = 7 * MIB
        self.assertEqual(h.run(), 1796 * MIB)
        self.assertEqual(h.runner.kv_cache_init_non_kv_memory_bytes, 0)
        self.assert_cleaned(h)

    def assert_cleaned(self, h):
        self.assertIs(type(h.platform)._global_graph_pool, h.persistent_pool)
        self.assertEqual((h.wrapper.graph_pool, h.breakable_wrapper.graph_pool), h.original_wrapper_pools)
        self.assertEqual((h.counter.num_cudagraph_captured, h.counter.num_gpu_runner_capture_triggers), (7, 3))
        self.assertIsNone(h.runner.cudagraph_manager)
        self.assertIsNone(h.runner.speculator.graph_manager)
        self.assertEqual(h.runner.speculator.unrelated, "retain")
        self.assertEqual(h.runner.kv_caches, [])
        self.assertEqual(h.runner.attn_groups, [])
        self.assertFalse(hasattr(h.runner, "kv_cache_config"))
        self.assertIsNone(h.runner.cache_config.num_gpu_blocks)
        self.assertIsNone(h.runner.model_state._mamba_ctx)
        self.assertEqual(h.runner.model_state._mamba_group_ids, [])
        self.assertIsNone(h.runner.model_state._mamba_spec)
        self.assertIsNone(h.layer.kv_cache)
        h.encoder_clear.assert_called_once_with()
        h.clear_wrappers.assert_called_once_with()
        h.clear_breakable.assert_called_once_with()
        h.runner.maybe_remove_all_loras.assert_called_once_with(None)

    def test_every_full_descriptor_is_captured_and_actual_delta_is_returned(self):
        h = ProfileHarness()
        measured = h.run()
        self.assertEqual(h.captured, [desc.num_tokens for desc in h.descriptors])
        self.assertEqual(measured, sum(h.costs))
        self.assertEqual(h.capture_calls, [True])
        self.assertEqual(h.prepared, [(desc.num_tokens, flag) for desc in h.descriptors for flag in (True, False)])
        h.init.assert_called_once_with(h.runner)
        self.assert_cleaned(h)

    def test_shared_pool_tail_is_not_extrapolated_from_large_first_deltas(self):
        h = ProfileHarness()
        self.assertEqual(h.run(), 1796 * MIB)
        self.assert_cleaned(h)

    def test_none_mode_does_not_initialize_or_capture(self):
        h = ProfileHarness(mode=Mode.NONE)
        h.runner.kv_cache_init_non_kv_memory_bytes = 97 * MIB
        self.assertEqual(h.run(), 0)
        self.assertEqual(h.runner.kv_cache_init_non_kv_memory_bytes, 0)
        h.init.assert_not_called()
        self.assertEqual(h.capture_calls, [])
        self.assertIs(type(h.platform)._global_graph_pool, h.persistent_pool)
        h.empty_cache.assert_not_called()
        h.clear_wrappers.assert_not_called()

    def test_capture_exception_still_restores_pool_counters_and_tears_down(self):
        h = ProfileHarness(fail_at=384)
        with self.assertRaisesRegex(RuntimeError, "injected capture"):
            h.run()
        self.assertEqual(h.captured, [768])
        self.assert_cleaned(h)

    def test_exception_in_previously_unsampled_descriptor_also_tears_down(self):
        h = ProfileHarness(fail_at=192)
        with self.assertRaisesRegex(RuntimeError, "injected capture"):
            h.run()
        self.assertEqual(h.captured, [768, 384])
        self.assert_cleaned(h)

    def test_negative_full_measurement_is_clamped_without_extrapolation(self):
        h = ProfileHarness(measured_override=-1)
        self.assertEqual(h.run(), 0)
        self.assert_cleaned(h)

    def test_releasable_warmup_cache_is_excluded_while_live_graph_pool_is_retained(self):
        h = CachedProfileHarness()
        self.assertEqual(h.run(), sum(h.costs))
        self.assertEqual(len(h.samples), 3)
        before_init, before, after = h.samples
        self.assertFalse(before_init["kv_live"])
        self.assertTrue(before["kv_live"] and after["kv_live"])
        self.assertEqual(before["normal_cache"], 0)
        self.assertEqual(after["normal_cache"], 0)
        self.assertTrue(after["graphs_alive"])
        self.assertEqual(after["graph_bytes"], sum(h.costs))
        self.assertEqual(before["free"] - after["free"], sum(h.costs))
        self.assert_cleaned(h)

    def test_live_non_graph_allocations_are_still_counted(self):
        h = CachedProfileHarness(persistent=25 * MIB)
        self.assertEqual(h.run(), sum(h.costs) + 25 * MIB)
        self.assertEqual(h.samples[-1]["normal_cache"], 0)
        self.assertTrue(h.samples[-1]["graphs_alive"])
        self.assert_cleaned(h)

    def test_postcapture_flush_exception_still_runs_finally_cleanup(self):
        h = CachedProfileHarness(fail_post_flush=True)
        with self.assertRaisesRegex(RuntimeError, "post-capture cache flush failure"):
            h.run()
        self.assert_cleaned(h)

    def test_postinit_measurement_exception_cleans_new_profiling_state(self):
        h = CachedProfileHarness(init_retained=97 * MIB)
        failures = []

        def memory_info():
            if h.kv_live and not failures:
                failures.append(True)
                raise RuntimeError("injected post-init sample failure")
            return h.memory_info()

        h.namespace["torch"].accelerator.get_memory_info = memory_info
        with self.assertRaisesRegex(RuntimeError, "post-init sample failure"):
            h.run()
        self.assertEqual(h.capture_calls, [])
        self.assertEqual(h.runner.kv_cache_init_non_kv_memory_bytes, 0)
        self.assert_cleaned(h)


if __name__ == "__main__":
    unittest.main()
