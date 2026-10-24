"""Profile measurement evidence, DP warmup proof and failure behavior."""

import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.benchmarks import profile_measure as pm

BASE = "http://127.0.0.1:6688"


def samples(p0=0, g0=0, p1=0, g1=0):
    return {f'vllm:{kind}_total{{engine="{engine}",model_name="model"}}': value
            for engine, values in ((0, (p0, g0)), (1, (p1, g1)))
            for kind, value in zip(("prompt_tokens", "generation_tokens"), values, strict=True)}


def metric_text():
    return "\n".join(f"{key} {value}" for key, value in samples(20, 5, 30, 7).items()) + "\n" + (
        'vllm:num_preemptions_total{engine="0"} 0\n'
        'vllm:num_requests_waiting{engine="0"} 3\n'
        'vllm:kv_cache_usage_perc{engine="0"} 0.25\n'
    )


def row(tokens=100, cache=0, seconds=2):
    return {"status": "PASS", "input_tokens": tokens, "output_tokens": 128,
            "e2e_s": seconds, "ttft_s": 0.5, "tpot_s": 0.01,
            "decode_span_s": 1.27, "itl_s": [0.01], "cached_tokens": cache}


class MetricsTests(unittest.TestCase):
    def test_metric_labels_and_values_are_preserved(self):
        parsed = pm.parse_metrics('# HELP test x\n# TYPE test counter\n'
                                  'test{model="a,b",engine="0"} 1.2e3 123\n'
                                  'other +Inf\nnan_metric NaN\n')
        self.assertEqual(parsed['test{engine="0",model="a,b"}'], 1200)
        self.assertEqual(parsed["other"], float("inf"))
        self.assertEqual(pm.engine_token_totals(samples(10, 2, 20, 4)), {
            "0": {"prompt_tokens": 10, "generation_tokens": 2},
            "1": {"prompt_tokens": 20, "generation_tokens": 4},
        })

    def test_missing_engine_or_counter_fails_instead_of_inventing_dp_evidence(self):
        for value in ({}, {'vllm:prompt_tokens_total': 1},
                      {'vllm:prompt_tokens_total{engine="0"}': 1},
                      {**samples(), 'vllm:prompt_tokens_total{engine="0",model_name="another"}': 1}):
            with self.assertRaises(pm.MetricsUnavailableError):
                pm.engine_token_totals(value)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pm.parse_metrics("metric 1\nmetric 2\n")
        with self.assertRaisesRegex(ValueError, "invalid Prometheus"):
            pm.parse_metrics("not a metric line")

    def test_formal_and_remote_endpoints_are_rejected(self):
        for url in ("http://127.0.0.1:6789", "http://127.0.0.1:8005",
                    "http://example.com:6688", BASE + "/v1", BASE + "?x=1"):
            with self.assertRaisesRegex(ValueError, "6688"):
                pm.require_test_endpoint(url)


class MeasurementTests(unittest.TestCase):
    def test_uses_per_request_input_lengths_and_preserves_payloads(self):
        bodies = [{"prompt": [1], "cache_salt": "first"}, {"prompt": [2], "cache_salt": "second"}]
        cell = {"id": "vision", "concurrency": 2, "osl": 128, "isl": 4096,
                "expected_input_by_index": [100, 102]}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cell"
            with patch.object(pm.bench_common, "run_burst", return_value=([row(100), row(102, seconds=4)], 4.0)) as burst:
                result = pm.measure(BASE, bodies, cell, out)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual((result["input_tps"], result["output_tps"]), (50.5, 64))
            self.assertEqual((result["e2e_p50_s"], result["e2e_p95_s"], result["e2e_p99_s"]), (2, 4, 4))
            burst.assert_called_once_with(BASE, bodies, 2, 3600, expected_output=128)
            with gzip.open(out / "request-payloads.json.gz", "rt") as stream:
                self.assertEqual(json.load(stream), bodies)
            with gzip.open(out / "request-metrics.json.gz", "rt") as stream:
                self.assertEqual(len(json.load(stream)), 2)
            self.assertEqual(json.loads((out / "report.json").read_text())["status"], "PASS")

    def test_wrong_per_request_lengths_and_missing_or_warm_cache_fail(self):
        cell = {"id": "vision", "concurrency": 1, "osl": 128, "expected_input_by_index": [100]}
        with tempfile.TemporaryDirectory() as tmp:
            for index, result_row in enumerate((row(101), row(100, None), row(100, 12))):
                with patch.object(pm.bench_common, "run_burst", return_value=([result_row], 2.0)):
                    result = pm.measure(BASE, [{}], cell, Path(tmp) / str(index))
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(result["errors"])

    def test_scalar_actual_length_and_tolerance_override_target(self):
        cell = {"id": "text", "concurrency": 1, "osl": 128, "isl": 4000,
                "actual_expected_input": 4096, "input_tolerance": 0.02}
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            pm.bench_common, "run_burst", return_value=([row(4096)], 2.0)
        ) as burst:
            pm.measure(BASE, [{}], cell, Path(tmp) / "measurement")
        self.assertEqual(burst.call_args.kwargs, {"expected_output": 128, "expected_input": 4096, "input_tolerance": 0.02})


class WarmupTests(unittest.TestCase):
    def cell(self):
        return {"id": "vision", "concurrency": 4, "osl": 1000,
                "expected_input_by_index": [100, 102, 104, 100], "warmup_metric_timeout": 0}

    def factory(self, label, index):
        return {"cache_salt": f"{label}:{index}", "max_tokens": 128, "prompt": [index]}

    def test_two_rounds_use_c2_and_prove_both_dp_engines_over_warmup(self):
        observer = MagicMock()
        observer.snapshot.side_effect = [samples(), samples(100, 128, 0, 0),
                                         samples(100, 128, 0, 0), samples(100, 128, 102, 128)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "measure", return_value={"status": "PASS"}) as measure:
            result = pm.warmup(BASE, self.factory, self.cell(), observer, Path(tmp) / "warmup")
            self.assertTrue((Path(tmp) / "warmup/round-1-metrics.json.gz").exists())
            self.assertTrue((Path(tmp) / "warmup/metrics-final.json.gz").exists())
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["both_dp_active"])
        self.assertEqual(measure.call_count, 2)
        salts = []
        for call in measure.call_args_list:
            _, bodies, cell, _, _ = call.args
            self.assertEqual((len(bodies), cell["concurrency"], cell["osl"]), (2, 2, 128))
            self.assertEqual(cell["expected_input_by_index"], [100, 102])
            salts.extend(body["cache_salt"] for body in bodies)
        self.assertEqual(len(salts), len(set(salts)))

    def test_missing_or_inactive_second_dp_fails(self):
        cases = ([{}, {}], [samples(), samples(100, 128, 0, 0), samples(100, 128, 0, 0), samples(200, 256, 0, 0)])
        with tempfile.TemporaryDirectory() as tmp:
            for index, snapshots in enumerate(cases):
                observer = MagicMock()
                observer.snapshot.side_effect = snapshots
                with patch.object(pm, "measure", return_value={"status": "PASS"}):
                    result = pm.warmup(BASE, self.factory, self.cell(), observer, Path(tmp) / str(index))
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(result["errors"])

    def test_duplicate_salts_or_output_budget_mismatch_fail_before_measurement(self):
        for factory in (lambda label, index: {"cache_salt": "same", "max_tokens": 128},
                        lambda label, index: {"cache_salt": f"{label}:{index}", "max_tokens": 1000}):
            observer = MagicMock()
            observer.snapshot.return_value = samples()
            with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "measure") as measure:
                result = pm.warmup(BASE, factory, self.cell(), observer, Path(tmp) / "warmup")
            self.assertEqual(result["status"], "FAIL")
            measure.assert_not_called()

    def test_warmup_measurement_failure_is_not_ignored(self):
        observer = MagicMock()
        observer.snapshot.side_effect = [samples(), samples(100, 128, 100, 128)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "measure", return_value={"status": "FAIL"}):
            result = pm.warmup(BASE, self.factory, self.cell(), observer, Path(tmp) / "warmup")
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("round 1 failed", result["errors"][0])


class PlanIntegrationTests(unittest.TestCase):
    def test_plan_cell_id_survives_measurement_and_warmup(self):
        from scripts.benchmarks import profile_plan, profile_workload

        cell = next(cell for cell in profile_plan.group_plan(1048576)["cells"]
                    if cell["case_id"] == "matrix_single" and cell["isl"] == 4096
                    and cell["dataset"] == "random")
        self.assertNotIn("id", cell)
        artifact = {"run_id": "integration", "token_pool": [100, 200, 300]}
        body = profile_workload.request_body(artifact, cell, "repeat-1", 0)
        result_row = {**row(4096), "output_tokens": cell["osl"]}
        observer = MagicMock()
        observer.snapshot.side_effect = [samples(), samples(4096, 128, 4096, 128),
                                         samples(4096, 128, 4096, 128), samples(8192, 256, 8192, 256)]

        def factory(label, index):
            body = profile_workload.request_body(artifact, cell, label, index)
            body["max_tokens"] = min(cell["osl"], 128)
            return body

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(pm.bench_common, "run_burst", return_value=([result_row], 2.0)):
                measured = pm.measure(BASE, [body], cell, Path(tmp) / "measurement")
            with patch.object(pm.bench_common, "run_burst", return_value=([row(4096), row(4096)], 2.0)):
                warmed = pm.warmup(BASE, factory, cell, observer, Path(tmp) / "warmup")
        for result in (measured, warmed):
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["cell_id"], cell["cell_id"])
            self.assertEqual(result["case_id"], cell["case_id"])

    def test_global_c256_warmup_accepts_cell_id_without_legacy_id(self):
        from scripts.benchmarks import profile_workload

        cell = {"cell_id": "startup-graphs", "kind": "matrix", "isl": 1024, "osl": 128,
                "concurrency": 256, "dataset": "random", "source_case": {"seed": 1017},
                "warmup_concurrency": 256, "warmup_osl": 128}
        artifact = {"run_id": "global-integration", "token_pool": [100, 200, 300]}
        observer = MagicMock()
        observer.snapshot.side_effect = [samples(), samples(131072, 16384, 131072, 16384),
                                         samples(131072, 16384, 131072, 16384), samples(262144, 32768, 262144, 32768)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "measure", return_value={"status": "PASS"}) as measured:
            result = pm.warmup(BASE, lambda label, index: profile_workload.request_body(artifact, cell, label, index),
                               cell, observer, Path(tmp) / "global-warmup")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["cell_id"], "startup-graphs")
        self.assertEqual(measured.call_count, 2)
        for call in measured.call_args_list:
            self.assertEqual(len(call.args[1]), 256)
            self.assertTrue(all(len(body["prompt"]) == 1024 for body in call.args[1]))
            self.assertEqual(call.args[2]["cell_id"], "startup-graphs")


class ObserverTests(unittest.TestCase):
    def setUp(self):
        platform_patch = patch.object(pm.platform, "uname")
        platform_patch.start()._asdict.return_value = {"system": "Linux"}
        self.addCleanup(platform_patch.stop)

    def inspect(self, image="sha256:fixed"):
        return {"Id": "container1", "Image": image, "State": {"Running": True, "StartedAt": "time"},
                "Config": {"Cmd": ["serve", "model"]},
                "NetworkSettings": {"Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "6688"}]}}}

    def test_context_preserves_raw_samples_logs_and_identity(self):
        process = MagicMock()
        process.poll.return_value = None

        def run(command, timeout=20):
            if command[0] == "docker":
                return json.dumps([self.inspect()])
            return "GPU hardware details" if command[1] == "-q" else "0,GPU-a,A100,80000,70000,99,60,300\n"

        with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "_run", side_effect=run), patch.object(
            pm.subprocess, "Popen", return_value=process
        ), patch.object(pm.bench_common, "request", side_effect=lambda *args, **kwargs: io.BytesIO(metric_text().encode())):
            directory = Path(tmp) / "observer"
            with pm.ServiceObserver(BASE, "container1", directory, interval=1000) as observer:
                self.assertEqual(set(observer.dp_totals()), {"0", "1"})
                (directory / "docker.log").write_text("Available KV cache memory: 10.50 GiB\nGPU KV cache size: 100 tokens\n")
            summary = observer.summary()
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["identity_before"], summary["identity_after"])
            self.assertEqual(len(summary["kv_log_lines"]), 2)
            self.assertEqual(summary["observed_counter_changes"]['vllm:num_preemptions_total{engine="0"}']["delta"], 0)
            self.assertFalse(summary["availability"]["oom_counter"])
            self.assertEqual(summary["observed_gauge_peaks"]['vllm:num_requests_waiting{engine="0"}'], 3)
            with gzip.open(directory / "metrics.jsonl.gz", "rt") as stream:
                captured = [json.loads(line) for line in stream]
            self.assertGreaterEqual(len(captured), 3)
            self.assertEqual(captured[0]["text"], metric_text())
            with gzip.open(directory / "gpu.jsonl.gz", "rt") as stream:
                gpu = json.loads(next(stream))
            self.assertIn("clocks.sm", gpu["fields"])
            self.assertIn("driver_version", gpu["fields"])
            self.assertTrue((directory / "environment-before.json").exists())
            self.assertTrue((directory / "environment-after.json").exists())
            self.assertTrue((directory / "observer-summary.json").exists())
        process.terminate.assert_called_once()

    def test_identity_change_is_recorded_as_failure(self):
        process = MagicMock()
        process.poll.return_value = None
        inspections = iter([self.inspect(), self.inspect("sha256:changed")])

        def run(command, timeout=20):
            return json.dumps([next(inspections)]) if command[0] == "docker" else "GPU info"

        with tempfile.TemporaryDirectory() as tmp, patch.object(pm, "_run", side_effect=run), patch.object(
            pm.subprocess, "Popen", return_value=process
        ), patch.object(pm.bench_common, "request", side_effect=lambda *args, **kwargs: io.BytesIO(metric_text().encode())):
            with pm.ServiceObserver(BASE, "container1", Path(tmp) / "observer", interval=1000) as observer:
                pass
            self.assertEqual(observer.summary()["status"], "FAIL")
            self.assertTrue(any(error["operation"] == "identity" for error in observer.summary()["observer_errors"]))


if __name__ == "__main__":
    unittest.main()
