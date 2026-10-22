"""CPU contract checks for the query-sharded top-k exchange helper."""

import __future__

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # The repository unit suite can run without PyTorch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "adaptation/files/vllm/model_executor/layers/sparse_attn_indexer.py"


@unittest.skipIf(torch is None, "PyTorch is only available in the runtime image")
class QueryShardScratchTests(unittest.TestCase):
    @staticmethod
    def helper():
        tree = ast.parse(SOURCE.read_text())
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_all_reduce_decode_topk"
        )
        namespace = {
            "torch": torch,
            # The identity reduction models one rank and keeps this a pure
            # storage/shape test; collectives are covered by GPU acceptance.
            "tensor_model_parallel_all_reduce": lambda value: value,
        }
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]),
                str(SOURCE),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        return namespace["_all_reduce_decode_topk"]

    def test_workspace_scratch_avoids_per_layer_zero_allocation(self):
        helper = self.helper()
        topk = torch.tensor(
            [[11, 12], [13, 14], [21, 22], [23, 24]], dtype=torch.int32
        )
        scratch = torch.empty((4, 2), dtype=torch.float32)
        with patch.object(torch, "zeros", side_effect=AssertionError("fresh allocation")):
            actual = helper(topk, 4, 2, 2, 4, 100, scratch)
        torch.testing.assert_close(
            actual,
            torch.tensor([[0, 0], [0, 0], [21, 22], [23, 24]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_workspace_width_is_checked_before_collective(self):
        helper = self.helper()
        topk = torch.zeros((2, 2), dtype=torch.int32)
        scratch = torch.empty((2, 1), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "scratch width"):
            helper(topk, 2, 2, 0, 2, 100, scratch)


if __name__ == "__main__":
    unittest.main()
