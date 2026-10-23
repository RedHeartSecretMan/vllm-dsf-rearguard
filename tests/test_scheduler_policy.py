"""CPU checks for the opt-in stage-aware prefill scheduler knobs."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.source_extract import load_function

ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = ROOT / "adaptation/files/vllm/v1/core/sched/scheduler.py"


def load_value_reader():
    return load_function(
        SCHEDULER, "_prefill_throttle_value", {"os": os}, annotations=False
    )


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
