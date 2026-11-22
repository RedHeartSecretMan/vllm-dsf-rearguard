"""CPU contract tests for the optional DSpark KV-only insertion op."""

import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from scripts.source_extract import load_functions

SOURCE = Path(__file__).resolve().parents[1] / (
    "adaptation/files/vllm/models/deepseek_v4_1/nvidia/dspark.py"
)


class FakeCache:
    def __init__(self, dtype, width):
        self.dtype = dtype
        self.shape = (2, 4, width)
        self.ndim = 3
        self.view_calls = []

    def view(self, *shape):
        self.view_calls.append(shape)
        return self


class FakeKV:
    shape = (4, 512)
    dtype = "bfloat16"
    device = "cuda"


class FakeTorch:
    uint8 = "uint8"
    bfloat16 = "bfloat16"
    float8_e4m3fn = "float8_e4m3fn"

    def __init__(self, *, native=True):
        self.calls = []
        operators = SimpleNamespace()
        if native:
            operators.fused_deepseek_v4_kv_rope_insert = self.kv_only
        operators.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert = (
            self.packed_legacy
        )
        operators.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert = (
            self.bf16_legacy
        )
        operators.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert = (
            self.fp8_legacy
        )
        self.ops = SimpleNamespace(_C=operators)

    def empty(self, *args, **kwargs):
        return object()

    def empty_like(self, *args, **kwargs):
        return object()

    def kv_only(self, *args):
        self.calls.append(("kv_only", args))

    def packed_legacy(self, *args):
        self.calls.append(("packed_legacy", args))

    def bf16_legacy(self, *args):
        self.calls.append(("bf16_legacy", args))

    def fp8_legacy(self, *args):
        self.calls.append(("fp8_legacy", args))


class DSparkContextKVTests(unittest.TestCase):
    def run_insert(self, torch, cache):
        namespace = load_functions(
            SOURCE,
            ("_get_kv_only_insert_op", "_insert_context_kv"),
            {
                "Path": Path,
                "lru_cache": lru_cache,
                "torch": torch,
                "vllm": SimpleNamespace(__file__="/tmp/vllm/__init__.py"),
            },
        )
        insert = namespace["_insert_context_kv"]
        kv = FakeKV()
        positions = object()
        slots = object()
        attn = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=32),
            rotary_emb=SimpleNamespace(cos_sin_cache=object()),
            padded_heads=8,
            n_local_heads=8,
            head_dim=512,
            eps=1e-20,
            _flashinfer_fp8_kv_scale=object(),
            _flashinfer_fp8_q_scale_inv=object(),
        )
        insert(attn, kv, positions, slots)
        return kv, positions, slots, attn

    def test_uint8_584_uses_kv_only_when_native_op_is_registered(self):
        torch = FakeTorch(native=True)
        cache = FakeCache(torch.uint8, 584)
        kv, positions, slots, attn = self.run_insert(torch, cache)

        self.assertEqual([name for name, _ in torch.calls], ["kv_only"])
        args = torch.calls[0][1]
        self.assertEqual(args[:4], (kv, cache, slots, positions))
        self.assertIs(args[4], attn.rotary_emb.cos_sin_cache)
        self.assertEqual(args[5], 32)
        self.assertEqual(cache.view_calls, [])

    def test_missing_native_op_falls_back_to_legacy_packed_insert(self):
        torch = FakeTorch(native=False)
        cache = FakeCache(torch.uint8, 584)
        self.run_insert(torch, cache)

        self.assertEqual([name for name, _ in torch.calls], ["packed_legacy"])
        self.assertEqual(cache.view_calls, [(2, -1)])

    def test_528_byte_uint8_cache_never_uses_584_byte_native_op(self):
        torch = FakeTorch(native=True)
        cache = FakeCache(torch.uint8, 528)
        self.run_insert(torch, cache)

        self.assertEqual([name for name, _ in torch.calls], ["packed_legacy"])
        self.assertEqual(cache.view_calls, [(2, -1)])

    def test_bfloat16_and_float8_keep_their_legacy_insert_ops(self):
        for dtype, expected in (
            ("bfloat16", "bf16_legacy"),
            ("float8_e4m3fn", "fp8_legacy"),
        ):
            with self.subTest(dtype=dtype):
                torch = FakeTorch(native=True)
                cache = FakeCache(dtype, 512)
                self.run_insert(torch, cache)
                self.assertEqual([name for name, _ in torch.calls], [expected])


if __name__ == "__main__":
    unittest.main()
