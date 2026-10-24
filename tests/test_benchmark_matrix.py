"""Pure metric contracts for the ISL/OSL benchmark matrix runner."""

import unittest

from scripts.benchmarks.benchmark_matrix import summarize


class BenchmarkMatrixMetricTests(unittest.TestCase):
    def test_aggregate_and_per_request_metrics_keep_their_scopes(self):
        rows = [
            {
                "status": "PASS",
                "input_tokens": 100,
                "output_tokens": 10,
                "ttft_s": 2.0,
                "tpot_s": 0.1,
                "decode_span_s": 0.9,
                "itl_s": [0.1, 0.2],
            },
            {
                "status": "PASS",
                "input_tokens": 100,
                "output_tokens": 10,
                "ttft_s": 4.0,
                "tpot_s": 0.2,
                "decode_span_s": 1.8,
                "itl_s": [0.2, 0.3],
            },
        ]
        report = summarize(rows, 10.0, "nature", 100, 10, 2, 1017)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["observed_input_tokens"], 200)
        self.assertEqual(report["observed_output_tokens"], 20)
        self.assertEqual(report["prefill_tps"], 20.0)
        self.assertEqual(report["decode_tps"], 2.0)
        self.assertEqual(report["ttft_p50_s"], 2.0)
        self.assertEqual(report["tpot_p50_s"], 0.1)


if __name__ == "__main__":
    unittest.main()
