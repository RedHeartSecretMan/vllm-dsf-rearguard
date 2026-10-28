"""Exercise production prefill dispatch and allocation with CPU-backed buffers.

The recording kernel verifies pointers, strides and write/copy data flow. It
does not emulate Triton attention math or replace the pinned GPU regression.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.source_extract import load_functions

SOURCE = Path(__file__).resolve().parents[1] / (
    "adaptation/files/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"
)


class Tensor:
    def __init__(self, data, *, device="cuda:0", storage=None):
        self.data = np.asarray(data)
        self.device = device
        self.storage = self.data if storage is None else storage
        self.copies = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def is_cpu(self):
        return False

    def stride(self, dim):
        return self.data.strides[dim] // self.data.itemsize

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.storage.ctypes.data)

    def numel(self):
        return self.data.size

    def is_contiguous(self):
        return self.data.flags.c_contiguous

    def contiguous(self):
        return self if self.is_contiguous() else Tensor(np.ascontiguousarray(self.data))

    def squeeze(self, dim):
        return Tensor(self.data.squeeze(dim), device=self.device, storage=self.storage)

    def reshape(self, *shape):
        return Tensor(self.data.reshape(shape), device=self.device, storage=self.storage)

    def __getitem__(self, index):
        return Tensor(self.data[index], device=self.device, storage=self.storage)

    def __ge__(self, value):
        return Tensor(self.data >= value, device=self.device)

    def sum(self, dim, dtype):
        return Tensor(self.data.sum(axis=dim, dtype=dtype), device=self.device)

    def to(self, dtype):
        return self if dtype == self.dtype else Tensor(self.data.astype(dtype), device=self.device)

    def copy_(self, other):
        self.copies += 1
        self.data[...] = other.data
        return self


class Buffers:
    int32, float32 = np.dtype("int32"), np.dtype("float32")

    def __init__(self):
        self.full_allocations = []

    def empty(self, shape, *, device, dtype):
        return Tensor(np.empty(shape, dtype=dtype), device=device)

    def empty_like(self, tensor):
        out = Tensor(np.empty_like(tensor.data), device=tensor.device)
        self.full_allocations.append(out)
        return out


class RecordingKernel:
    def __init__(self):
        self.outputs = []

    @staticmethod
    def expected(q, kv, indices, indptr, sink, scale):
        # Deliberately simple data flow, not a reference attention kernel.
        values = q.data.copy() * scale
        for row in range(q.shape[0]):
            slots = indices.data[indptr.data[row]:indptr.data[row + 1]]
            values[row] += kv.data[slots].sum(axis=0)
        if sink is not None:
            values += sink.data[:q.shape[1]][None, :, None]
        return values

    def __getitem__(self, grid):
        def launch(q, kv, indices, indptr, sink, out, *args, **kwargs):
            assert grid[0] == q.shape[0]
            assert args[:3] == tuple(q.stride(i) for i in range(3))
            assert args[3:5] == tuple(kv.stride(i) for i in range(2))
            assert args[5:8] == tuple(out.stride(i) for i in range(3))
            self.outputs.append(out)
            out.data[...] = self.expected(
                q, kv, indices, indptr, sink if kwargs["HAS_ATTN_SINK"] else None, args[11])
        return launch


class PrefillHarness:
    def __init__(self, *, ragged=False):
        self.ragged = ragged
        self.buffers, self.kernel = Buffers(), RecordingKernel()
        self.q = Tensor(np.arange(4 * 2 * 8, dtype=np.float32).reshape(4, 2, 8) / 10)
        self.kv = Tensor(np.arange(6 * 8, dtype=np.float32).reshape(6, 1, 8) / 7)
        self.indices = Tensor(np.array([[0, 2, -1], [1, -1, -1], [2, 3, 4], [0, 5, -1]], dtype=np.int32))
        self.lengths = Tensor(np.array([2, 1, 3, 2], dtype=np.int32))
        self.sink = Tensor(np.array([.2, .4], dtype=np.float32))
        self.ragged_indices, self.indptr = self.build_ragged(self.indices, self.lengths, num_rows=6)
        self.output = Tensor(np.full(self.q.shape, -99, dtype=np.float32))
        self.opus = False
        self.opus_calls = 0
        namespace = {
            "torch": self.buffers,
            "envs": SimpleNamespace(VLLM_SPARSE_PREFILL_EXACT_TILE=True),
            "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                                      cdiv=lambda n, d: (n + d - 1) // d),
            "_prefill_block_h": lambda n: n,
            "_prefill_block_k": lambda _: 16,
            "_validate_sparse_dims": lambda *_: None,
            "_can_use_aiter_sparse_prefill_opus": lambda *_: self.opus,
            "_get_aiter_sparse_prefill_opus": lambda: object(),
            "_rocm_sparse_attn_prefill_ragged_aiter_opus": self.run_opus,
            "_sparse_attn_prefill_ragged_kernel": self.kernel,
            "build_ragged_indices_from_dense": self.build_ragged,
        }
        self.functions = load_functions(SOURCE, (
            "rocm_sparse_attn_prefill", "_rocm_sparse_attn_prefill_triton",
            "_rocm_sparse_attn_prefill_ragged_triton", "_as_int32_contiguous_1d",
        ), namespace)

    @staticmethod
    def build_ragged(indices, lengths, *, num_rows):
        rows = [row[:length] for row, length in zip(indices.data, lengths.data, strict=True)]
        rows = [row[(row >= 0) & (row < num_rows)] for row in rows]
        return (Tensor(np.concatenate(rows).astype(np.int32)),
                Tensor(np.array([0, *np.cumsum([len(row) for row in rows])], dtype=np.int32)))

    def run_opus(self, **kwargs):
        self.opus_calls += 1
        kwargs["output"].data.fill(7)
        return True

    def expected(self):
        indices, indptr = ((self.ragged_indices, self.indptr) if self.ragged else
                          self.build_ragged(self.indices, self.lengths, num_rows=6))
        return self.kernel.expected(self.q, self.kv.squeeze(1), indices, indptr, self.sink, .5)

    def run(self):
        return self.functions["rocm_sparse_attn_prefill"](
            q=self.q, kv=self.kv, indices=None if self.ragged else self.indices,
            topk_length=None if self.ragged else self.lengths, scale=.5,
            head_dim=8, nope_head_dim=6, rope_head_dim=2, attn_sink=self.sink,
            output=self.output, ragged_indices=self.ragged_indices if self.ragged else None,
            ragged_indptr=self.indptr if self.ragged else None)


class SparsePrefillOutputTests(unittest.TestCase):
    def test_public_fallbacks_write_caller_output_without_full_allocation_or_copy(self):
        for ragged in (False, True):
            for strided in (False, True):
                with self.subTest(ragged=ragged, strided=strided):
                    h = PrefillHarness(ragged=ragged)
                    if strided:
                        # Query offset plus head/dimension gaps, like caller-owned views.
                        base = np.full((6, 3, 16), -99, dtype=np.float32)
                        h.output = Tensor(base[1:5, :2, ::2], storage=base)
                    expected = h.expected()
                    self.assertIsNone(h.run())
                    np.testing.assert_allclose(h.output.data, expected)
                    self.assertEqual(h.buffers.full_allocations, [])
                    self.assertIs(h.kernel.outputs[0], h.output)
                    self.assertEqual(h.output.copies, 0)

    def test_legacy_shape_dtype_or_device_keeps_allocation_cast_and_slice(self):
        for ragged in (False, True):
            for mismatch in ("shape", "dtype", "device"):
                with self.subTest(ragged=ragged, mismatch=mismatch):
                    h = PrefillHarness(ragged=ragged)
                    if mismatch == "shape":
                        h.output = Tensor(np.empty((4, 2, 6), dtype=np.float32))
                    elif mismatch == "dtype":
                        h.output = Tensor(np.empty(h.q.shape, dtype=np.float16))
                    else:
                        h.output.device = "cuda:1"
                    expected = h.expected()[..., :h.output.shape[-1]].astype(h.output.dtype)
                    h.run()
                    np.testing.assert_array_equal(h.output.data, expected)
                    self.assertEqual(len(h.buffers.full_allocations), 1)
                    self.assertIsNot(h.kernel.outputs[0], h.output)
                    self.assertEqual(h.output.copies, 1)

    def test_aliases_of_each_read_input_keep_copy_after_compute(self):
        for name in ("q", "kv", "sink", "indices", "lengths", "ragged_indices", "indptr"):
            with self.subTest(input=name):
                h = PrefillHarness(ragged=name in ("ragged_indices", "indptr"))
                original = getattr(h, name)
                raw = h.output.data.view(np.uint8).reshape(-1)
                data = raw[:original.data.nbytes].view(original.dtype).reshape(original.shape)
                data[...] = original.data
                setattr(h, name, Tensor(data, storage=h.output.storage))
                expected = h.expected()
                h.run()
                np.testing.assert_allclose(h.output.data, expected)
                self.assertEqual(len(h.buffers.full_allocations), 1)
                self.assertEqual(h.output.copies, 1)

    def test_overlapping_output_layout_uses_existing_copy_path(self):
        h = PrefillHarness()
        base = np.empty((4, 1, 8), dtype=np.float32)
        h.output = Tensor(np.lib.stride_tricks.as_strided(
            base, shape=h.q.shape, strides=(base.strides[0], 0, base.strides[2])), storage=base)
        h.run()
        self.assertEqual(len(h.buffers.full_allocations), 1)
        self.assertEqual(h.output.copies, 1)

    def test_optional_output_omitted_retains_private_helper_result(self):
        h = PrefillHarness(ragged=True)
        result = h.functions["_rocm_sparse_attn_prefill_ragged_triton"](
            q=h.q, kv=h.kv.squeeze(1), indices=h.ragged_indices, indptr=h.indptr,
            scale=.5, attn_sink=h.sink, nope_head_dim=6, rope_head_dim=2)
        np.testing.assert_allclose(result.data, h.expected())
        self.assertEqual(len(h.buffers.full_allocations), 1)
        self.assertIs(result, h.kernel.outputs[0])

    def test_aiter_success_keeps_its_original_output_contract(self):
        h = PrefillHarness()
        h.opus = True
        h.run()
        self.assertEqual(h.opus_calls, 1)
        self.assertEqual(h.kernel.outputs, [])
        self.assertEqual(h.buffers.full_allocations, [])
        np.testing.assert_array_equal(h.output.data, 7)


if __name__ == "__main__":
    unittest.main()
