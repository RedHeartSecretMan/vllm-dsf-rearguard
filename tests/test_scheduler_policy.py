"""CPU checks for the opt-in stage-aware prefill scheduler knobs."""

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "adaptation/files/vllm/v1/core/sched/scheduler.py"


def load_value_reader():
    tree = ast.parse(SCHEDULER.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prefill_throttle_value"
    )
    namespace = {"os": os}
    module = ast.Module(
        body=[ast.Import(names=[ast.alias(name="os")]), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SCHEDULER), "exec"), namespace)
    return namespace["_prefill_throttle_value"]


class SchedulerPolicyTests(unittest.TestCase):
    def test_defaults_are_stable_and_positive(self):
        read = load_value_reader()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(read("VLLM_DSV41_PREFILL_THROTTLE_CADENCE", 4), 4)
            self.assertEqual(read("VLLM_DSV41_PREFILL_THROTTLE_MIN_TOKENS", 4096), 4096)

    def test_invalid_and_non_positive_values_cannot_disable_safety(self):
        read = load_value_reader()
        with patch.dict(
            os.environ,
            {
                "VLLM_DSV41_PREFILL_THROTTLE_CADENCE": "invalid",
                "VLLM_DSV41_PREFILL_THROTTLE_MIN_TOKENS": "0",
            },
            clear=True,
        ):
            self.assertEqual(read("VLLM_DSV41_PREFILL_THROTTLE_CADENCE", 4), 4)
            self.assertEqual(read("VLLM_DSV41_PREFILL_THROTTLE_MIN_TOKENS", 4096), 1)


if __name__ == "__main__":
    unittest.main()
