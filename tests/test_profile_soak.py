"""A short isolated stress run exercises the real closed-loop coordinator."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmarks import profile_soak


class Observer:
    def __init__(self, *args):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def summary(self):
        return {"status": "PASS"}


class SoakTests(unittest.TestCase):
    def execute(self, directory, final_failure=False):
        def response(*args, **kwargs):
            time.sleep(.002)
            return {"status": "PASS", "input_tokens": 16, "output_tokens": 1000,
                    "cached_tokens": 0, "ttft_s": .001, "e2e_s": .002,
                    "decode_span_s": .001, "tpot_s": .000001, "itl_s": []}
        case = {"id": "stress", "isl": 16, "osl": 1000, "seed": 1017, "concurrency": 2, "duration_s": .025}
        frozen = {"run_id": "fixed", "token_pool": [10, 11], "sha256": "inputs"}
        out = Path(directory) / "run"
        effects = [{"container_id": "same"}, RuntimeError("container stopped")] if final_failure else None
        with patch.object(profile_soak, "verify_endpoint", return_value={"container_id": "same"}, side_effect=effects), \
                patch.object(profile_soak, "ServiceObserver", Observer), \
                patch.object(profile_soak, "warmup", return_value={"status": "PASS"}), \
                patch.object(profile_soak, "stream_request", side_effect=response):
            result = profile_soak.run(profile_soak.profile_workload.BASE_URL, frozen, case,
                                      {"container_id": "same"}, out)
        return result, case, out

    def test_full_duration_then_drain_keeps_every_request_and_cache_check(self):
        with tempfile.TemporaryDirectory() as directory:
            result, case, out = self.execute(directory)
            self.assertEqual(result["status"], "PASS")
            self.assertGreater(result["request_count"], 2)
            self.assertEqual(result["request_count"], len(result["request_identities"]))
            self.assertGreaterEqual(result["elapsed_s"], case["duration_s"])
            self.assertTrue((out / "request-metrics.json.gz").exists())

    def test_final_service_failure_preserves_all_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, out = self.execute(directory, final_failure=True)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("container stopped", result["endpoint_error"])
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "request-metrics.json.gz").exists())
            self.assertEqual(len((out / "completed-requests.jsonl").read_text().splitlines()), result["request_count"])


if __name__ == "__main__":
    unittest.main()
