"""Execute production attention orchestration with observable CPU buffers.

These tests cover aliasing, allocation lifetime and callback contracts. They do
not emulate CUDA kernels or replace the required pinned-runtime/GPU checks.
"""

import unittest
import weakref
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import cast

import numpy as np

from scripts.source_extract import load_method

SOURCE = Path(__file__).resolve().parents[1] / "adaptation/files/vllm/models/deepseek_v4_1/attention.py"


class Storage:
    def __init__(self, data, name):
        self.data, self.name = data, name


class Tensor:
    def __init__(self, backend, storage, dtype="bf16", data=None):
        self.backend, self.storage, self.dtype = backend, storage, dtype
        self.data = storage.data if data is None else data
        self.device = "cpu"

    @property
    def shape(self):
        return self.data.shape

    def view(self, *shape):
        return Tensor(self.backend, self.storage, self.dtype, self.data.reshape(shape))

    def clone(self):
        return self.backend.tensor(self.data.copy(), "cloned_q", self.dtype)

    def __getitem__(self, index):
        return Tensor(self.backend, self.storage, self.dtype, self.data[index])


class Buffers:
    uint8, int64 = "uint8", "int64"

    def __init__(self):
        self.live = weakref.WeakValueDictionary()
        self.events = []
        self.serial = 0
        self.native_calls = []
        self.ops = SimpleNamespace(_C=SimpleNamespace(
            fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert=self.fused))

    def tensor(self, data, name, dtype="bf16"):
        self.events.append((name, tuple(s.name for s in self.live.values())))
        storage = Storage(np.asarray(data), name)
        self.serial += 1
        self.live[self.serial] = storage
        return Tensor(self, storage, dtype)

    def empty(self, shape, *, dtype, device):
        assert device == "cpu"
        return self.tensor(np.empty(shape), "attention_out", dtype)

    def pad(self, q, padding, value):
        assert padding[:3] == (0, 0, 0)
        data = np.pad(q.data, ((0, 0), (0, padding[3]), (0, 0)), constant_values=value)
        return self.tensor(data, "padded_q", q.dtype)

    def fused(self, q, kv, cache, slots, positions, rope, heads, eps, block_size, flag):
        self.native_calls.append({"shape": q.shape, "heads": heads, "eps": eps, "block_size": block_size,
                                  "flag": flag, "slots": id(slots), "positions": id(positions),
                                  "kv_storage": id(kv.storage), "cache_storage": id(cache.storage), "rope": id(rope)})
        # A deterministic stand-in only for checking data flow, not kernel math.
        data = np.pad(q.data * 2, ((0, 0), (0, heads - q.shape[1]), (0, 0)))
        return self.tensor(data, "fused_q", q.dtype)


def static_output_break(function):
    """The pinned eager-break contract forbids newly returned output tensors."""
    def call(*args, **kwargs):
        result = function(*args, **kwargs)
        if result is not None:
            raise AssertionError("an eager-break callback must write its caller-owned output")
    return call


class AttentionHarness:
    def __init__(self, *, v2=True, profile=False, padded_heads=2, compressor=False,
                 capture_size=0, graph_mode="FULL_DECODE_ONLY", enforce_eager=False):
        self.buffers = Buffers()
        b = self.buffers
        self.n_local_heads, self.padded_heads, self.head_dim = 2, padded_heads, 3
        self.kv_cache_torch_dtype = b.uint8
        self._defer_attn_output = v2
        self._attn_compilation_config = SimpleNamespace(
            max_cudagraph_capture_size=capture_size, cudagraph_mode=graph_mode)
        self._attn_enforce_eager = enforce_eager
        self.eps = .01
        self.hidden = b.tensor(np.ones((4, 3)), "hidden")
        self.positions = b.tensor(np.arange(4), "positions", b.int64)
        self.kv = b.tensor(np.ones((4, 3)), "kv")
        self.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=b.tensor(np.zeros((4, 1, 3)), "kv_cache", b.uint8))
        self.slots = np.arange(4)
        self.metadata = None if profile else {"swa": SimpleNamespace(slot_mapping=self.slots, block_size=1)}
        self.rotary_emb = SimpleNamespace(cos_sin_cache=object())
        self.indexer_rotary_emb = object()
        self.ln_events = ["fanout", "join"]
        self.aux_stream_list = ["aux-0"] if compressor else None
        self.parallel_calls, self.operations = [], []

        def parallel(first, second, before, after, stream):
            self.parallel_calls.append((first.__name__, before, after, stream))
            return first(), second()

        namespace = {"torch": b, "F": SimpleNamespace(pad=b.pad), "cast": cast,
                     "CUDAGraphMode": SimpleNamespace(NONE="NONE"),
                     "get_forward_context": lambda: SimpleNamespace(attn_metadata=self.metadata),
                     "maybe_execute_in_parallel": parallel, "eager_break_during_capture": static_output_break}
        methods = {name: load_method(SOURCE, "DeepseekV4Attention", name, dict(namespace))
                   for name in ("forward", "_prepare_and_attn", "_prepare_and_attn_eager",
                                "_fused_qnorm_rope_kv_insert", "_sparse_indexer_and_attn")}
        for name in ("_prepare_and_attn", "_prepare_and_attn_eager", "_fused_qnorm_rope_kv_insert", "_sparse_indexer_and_attn"):
            setattr(self, name, MethodType(methods[name], self))
        self.forward = MethodType(methods["forward"], self)
        self._prepare_and_attn_fn = self._prepare_and_attn if v2 else self._prepare_and_attn_eager
        self._run_parallel_input_projections = lambda hidden: (hidden, "kv-score", "index-weights")
        self._split_qkv_and_norm = lambda qr_kv: (qr_kv, None, self.kv)
        self._wq_b_proj = lambda _qr, _scale: b.tensor(np.arange(24).reshape(4, 6), "projected_q")
        self._o_proj = lambda out, _positions: out
        self.compressor = self._compressor() if compressor else None
        self.indexer = self._indexer() if compressor else None

    def _compressor(self):
        owner = self
        class Compressor:
            def __call__(self, score, positions):
                owner.operations.append(("compress", score, id(positions)))
                return "latent"

            def insert_cache(self, latent, positions, rotary):
                owner.operations.append(("insert", latent, id(positions), id(rotary)))
        return Compressor()

    def _indexer(self):
        owner = self
        class Indexer:
            def __call__(self, qr, latent, weights, positions, rotary, scale):
                owner.operations.append(("indexer", latent, weights, id(positions), id(rotary), scale))
                return qr, None, weights

            def indexer_op(self, hidden, q, unused, weights):
                owner.operations.append(("indexer_op", unused, weights))
        return Indexer()

    def forward_mqa(self, q, kv, positions, output):
        assert output.shape == q.shape and output.dtype == q.dtype
        assert kv is self.kv and positions is self.positions
        self.operations.append(("attention", id(output.storage)))
        output.data[:] = q.data + 1


class AttentionMemoryTests(unittest.TestCase):
    def test_profile_q_preserves_values_without_aliasing_projected_q(self):
        model = AttentionHarness(profile=True)
        q = model.buffers.tensor(np.arange(24).reshape(4, 2, 3), "projected_q")
        result = model._fused_qnorm_rope_kv_insert(q, model.kv, model.positions, None)
        self.assertIsNot(result.storage, q.storage)
        np.testing.assert_array_equal(result.data, q.data)
        result.data[0, 0, 0] = -99
        self.assertEqual(q.data[0, 0, 0], 0)
        self.assertEqual(model.buffers.native_calls, [])

    def test_v2_does_not_hold_output_while_both_query_buffers_are_live(self):
        model = AttentionHarness()
        result = model.forward(model.positions, model.hidden)
        events = model.buffers.events
        names = [name for name, _ in events]
        self.assertLess(names.index("fused_q"), names.index("attention_out"))
        at_query = next(live for name, live in events if name == "fused_q")
        self.assertIn("projected_q", at_query)
        self.assertNotIn("attention_out", at_query)
        at_output = next(live for name, live in events if name == "attention_out")
        self.assertNotIn("projected_q", at_output)
        self.assertIn("fused_q", at_output)
        np.testing.assert_array_equal(result.data, np.arange(24).reshape(4, 2, 3) * 2 + 1)

    def test_v2_captured_small_shapes_keep_the_original_early_output(self):
        for capture_size in (4, 8):
            with self.subTest(capture_size=capture_size):
                model = AttentionHarness(capture_size=capture_size)
                result = model.forward(model.positions, model.hidden)
                names = [name for name, _ in model.buffers.events]
                self.assertLess(names.index("attention_out"), names.index("projected_q"))
                at_query = next(live for name, live in model.buffers.events if name == "fused_q")
                self.assertIn("attention_out", at_query)
                np.testing.assert_array_equal(result.data, np.arange(24).reshape(4, 2, 3) * 2 + 1)

    def test_v2_large_prefill_defers_only_above_current_capture_limit(self):
        model = AttentionHarness(capture_size=3)
        model.forward(model.positions, model.hidden)
        names = [name for name, _ in model.buffers.events]
        self.assertLess(names.index("fused_q"), names.index("attention_out"))
        # The runner can resolve capture sizes after model construction.
        model._attn_compilation_config.max_cudagraph_capture_size = 4
        model.buffers.events.clear()
        model.forward(model.positions, model.hidden)
        names = [name for name, _ in model.buffers.events]
        self.assertLess(names.index("attention_out"), names.index("projected_q"))

    def test_graph_disabled_or_enforced_eager_uses_zero_capture_limit(self):
        for options in ({"graph_mode": "NONE"}, {"enforce_eager": True}):
            with self.subTest(options=options):
                model = AttentionHarness(capture_size=8, **options)
                model.forward(model.positions, model.hidden)
                names = [name for name, _ in model.buffers.events]
                self.assertLess(names.index("fused_q"), names.index("attention_out"))

    def test_v1_keeps_early_output_regardless_of_capture_size_or_eager(self):
        for options in ({"capture_size": 0}, {"capture_size": 8}, {"enforce_eager": True}):
            with self.subTest(options=options):
                model = AttentionHarness(v2=False, **options)
                model.forward(model.positions, model.hidden)
                names = [name for name, _ in model.buffers.events]
                self.assertLess(names.index("attention_out"), names.index("projected_q"))

    def test_profile_padding_stays_zero_and_shape_matches_the_real_path(self):
        model = AttentionHarness(profile=True, padded_heads=4)
        q = model.buffers.tensor(np.arange(24).reshape(4, 2, 3), "projected_q")
        result = model._fused_qnorm_rope_kv_insert(q, model.kv, model.positions, None)
        self.assertEqual(result.shape, (4, 4, 3))
        np.testing.assert_array_equal(result.data[:, :2], q.data)
        np.testing.assert_array_equal(result.data[:, 2:], 0)

    def test_plain_row_bf16_profile_keeps_the_existing_inplace_contract(self):
        model = AttentionHarness(profile=True)
        model.kv_cache_torch_dtype = "bf16"
        q = model.buffers.tensor(np.arange(24).reshape(4, 2, 3), "projected_q")
        result = model._fused_qnorm_rope_kv_insert(q, model.kv, model.positions, None)
        self.assertIs(result.storage, q.storage)

    def test_v1_wide_eager_break_keeps_caller_output_and_none_return(self):
        model = AttentionHarness(v2=False)
        out = model.buffers.empty((4, 2, 3), dtype=model.hidden.dtype, device=model.hidden.device)
        storage = out.storage
        for _ in range(2):
            result = model._prepare_and_attn_eager(model.hidden, model.hidden, model.kv, None,
                                                 "kv-score", "index-weights", model.positions, out)
            self.assertIsNone(result)
            self.assertIs(out.storage, storage)
            np.testing.assert_array_equal(out.data, np.arange(24).reshape(4, 2, 3) * 2 + 1)

    def test_both_runner_paths_preserve_output_and_aux_stream_contract(self):
        for v2 in (False, True):
            for use_compressor in (False, True):
                with self.subTest(v2=v2, compressor=use_compressor):
                    model = AttentionHarness(v2=v2, compressor=use_compressor, padded_heads=4)
                    result = model.forward(model.positions, model.hidden)
                    np.testing.assert_array_equal(result.data, np.arange(24).reshape(4, 2, 3) * 2 + 1)
                    self.assertEqual(model.buffers.native_calls[0]["heads"], 4)
                    self.assertFalse(model.buffers.native_calls[0]["flag"])
                    if use_compressor:
                        self.assertEqual(model.parallel_calls, [("project_query_and_cache_kv", "fanout", "join", "aux-0"),
                                                               ("prepare_indexer", "fanout", "join", "aux-0")])
                        self.assertEqual([op[0] for op in model.operations], ["compress", "indexer", "insert", "indexer_op", "attention"])
                    else:
                        self.assertEqual(model.parallel_calls, [])


if __name__ == "__main__":
    unittest.main()
