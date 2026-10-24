"""Exercise long-tail worker submission through the real stream function."""

import copy
import unittest
from unittest.mock import patch

from scripts.bench_common import load_config
from scripts.benchmarks import bench_long_tail


class LongTailExecutionTests(unittest.TestCase):
    def case(self, case_id):
        return copy.deepcopy(next(c for c in load_config("long_tail")["cases"] if c["id"] == case_id))

    def metrics(self):
        return {
            "input_tokens": 4096, "output_tokens": 1000,
            "ttft_s": 1.0, "tpot_s": 0.01, "decode_span_s": 9.99,
            "itl_s": [0.01], "output_sha256": "example", "cached_tokens": 0,
        }

    def test_tool_workers_call_the_real_stream_signature(self):
        case = self.case("tool_history_16384_c16")
        with (
            patch.object(bench_long_tail, "request") as request,
            patch.object(bench_long_tail, "stream_metrics", return_value=self.metrics()) as metrics,
        ):
            result = bench_long_tail._run_tool("http://test", case, 60)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["requests"], case["concurrency"])
        self.assertEqual(request.call_count, case["concurrency"])
        self.assertEqual(metrics.call_count, case["concurrency"])
        for call in request.call_args_list:
            base, path, body, timeout = call.args
            self.assertEqual((base, path, timeout), ("http://test", "/v1/chat/completions", 60))
            self.assertEqual(body["seed"], case["seed"])
            self.assertEqual(body["max_tokens"], case["osl"])
        for call in metrics.call_args_list:
            self.assertEqual(call.args[2:], (True, case["osl"], case["isl"], case["input_tolerance"]))

    def test_vision_submits_and_accounts_for_exactly_four_requests(self):
        case = self.case("vision_mix_4096_c4")
        with (
            patch.object(bench_long_tail, "_vision_body", autospec=True,
                         side_effect=lambda base, count, target, osl, seed: {"image_count": count, "seed": seed}) as body,
            patch.object(bench_long_tail, "request") as request,
            patch.object(bench_long_tail, "stream_metrics", return_value=self.metrics()) as metrics,
        ):
            result = bench_long_tail._run_vision("http://test", case, 60)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual((result["requests"], result["success"]), (4, 4))
        self.assertEqual(body.call_count, 4)
        self.assertEqual([call.args[1] for call in body.call_args_list], [1, 2, 4, 1])
        self.assertEqual(request.call_count, 4)
        self.assertEqual(metrics.call_count, 4)
        for call in metrics.call_args_list:
            self.assertEqual(call.args[2:], (True, case["osl"], case["isl"], case["input_tolerance"]))


if __name__ == "__main__":
    unittest.main()
