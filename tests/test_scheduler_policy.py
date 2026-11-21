"""CPU checks for the opt-in stage-aware prefill scheduler knobs."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.source_extract import load_function, load_method

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


class HybridKVLoadRecoveryTests(unittest.TestCase):
    def test_failed_swa_block_restarts_async_load_without_invalidating_other_requests(
        self,
    ):
        recover = load_method(
            SCHEDULER, "Scheduler", "_update_requests_with_invalid_blocks", {}
        )
        # A 512-token external hit uses two 256-token MLA blocks and sixteen
        # 32-token SWA blocks. The expired SWA blocks have already been released.
        blocks = {
            "failed": ([1, 2], [0] * 12 + [3, 4, 5, 6]),
            "other": ([7, 8], [0] * 12 + [9, 10, 11, 12]),
        }
        scheduler = SimpleNamespace(
            block_size=256,
            kv_cache_manager=SimpleNamespace(
                block_pool=SimpleNamespace(null_block=SimpleNamespace(block_id=0)),
                coordinator=SimpleNamespace(
                    single_type_managers=(
                        SimpleNamespace(block_size=256),
                        SimpleNamespace(block_size=32),
                    )
                ),
                get_block_ids=blocks.__getitem__,
            ),
        )
        failed = SimpleNamespace(request_id="failed", num_computed_tokens=512)
        other = SimpleNamespace(request_id="other", num_computed_tokens=512)
        for invalid_blocks in ({3}, {0, 3}):
            with self.subTest(invalid_blocks=invalid_blocks):
                failed.num_computed_tokens = other.num_computed_tokens = 512
                self.assertEqual(
                    recover(
                        scheduler,
                        [failed, other],
                        invalid_blocks,
                        {},
                        evict_blocks=False,
                    ),
                    ({"failed"}, 512, set()),
                )
                self.assertEqual(failed.num_computed_tokens, 0)
                self.assertEqual(other.num_computed_tokens, 512)


if __name__ == "__main__":
    unittest.main()
