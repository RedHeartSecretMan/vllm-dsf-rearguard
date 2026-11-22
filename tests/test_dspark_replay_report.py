"""Offline DSpark greedy/probabilistic replay comparison tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

IMAGE = "sha256:" + "a" * 64
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
MODEL_MOUNT = "/models/DeepSeek-V4.1-Flash-1017"


SPEC_COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def _raw_metrics(drafts, draft_tokens, accepted):
    values = {
        SPEC_COUNTERS[0]: drafts,
        SPEC_COUNTERS[1]: draft_tokens,
        SPEC_COUNTERS[2]: accepted,
    }
    return {
        f'{name}{{engine="{engine}"}}': float(value)
        for engine, counters in {"0": values, "1": values}.items()
        for name, value in counters.items()
    }


def _launch(method):
    config = {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "draft_sample_method": method,
        "rejection_sample_method": "standard",
        "enable_adaptive_verification": False,
        "use_local_argmax_reduction": method == "greedy",
    }
    command = [
        "docker", "run", "-d", "--name", f"dspark-{method}", "--gpus", "all",
        "--mount", f"type=bind,src=/verified/weights,dst={MODEL_MOUNT},readonly",
        "--mount", f"type=volume,src=dspark-{method}-cache,dst=/workspace",
        IMAGE,
        "serve", MODEL_MOUNT,
        "--served-model-name", MODEL,
        "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", "4",
        "--data-parallel-size", "2",
        "--data-parallel-size-local", "2",
        "--enable-expert-parallel",
        "--all2all-backend", "allgather_reducescatter",
        "--max-model-len", "1048576",
        "--max-num-seqs", "128",
        "--max-num-batched-tokens", "16384",
        "--gpu-memory-utilization", "0.925",
        "--speculative-config", json.dumps(config),
    ]
    container_id = ("b" if method == "greedy" else "c") * 64
    endpoint = {
        "container_id": container_id,
        "image_id": IMAGE,
        "started_at": "2026-11-21T11:00:00Z",
    }
    return {
        "source_lock": {
            "base_image": {
                "image": "sha256:" + "d" * 64,
                "source_commit": "source-commit",
            }
        },
        "parallel": {"tp": 4, "dp": 2, "ep": 8, "max_global_seqs": 256},
        "dspark": 5,
        "dspark_draft_sample_method": method,
        "context": 1048576,
        "lmcache_mp": None,
        "lmcache_server_command": None,
        "kv_transfer_config": None,
        "kv_transfer_config_sha256": None,
        "command": command,
        "image_id": IMAGE,
        "source_manifest_sha256": "e" * 64,
        "model_manifest_sha256": "f" * 64,
        "model_revision": "verified-model",
        "container_id": container_id,
    }, endpoint


def _report():
    request = {
        "status": "PASS",
        "input_tokens": 128,
        "output_tokens": 96,
        "ttft_s": 0.2,
        "tpot_s": 0.01,
        "e2e_s": 0.5,
        "deadline_s": 60.0,
        "partial_stream": False,
    }
    return {
        "case": "sample-c1",
        "round": 1,
        "concurrency": 1,
        "temperature": 0.7,
        "input_tokens_each": 128,
        "output_tokens_each": 96,
        "workload_sha256": "1" * 64,
        "status": "PASS",
        "request_count": 1,
        "success": 1,
        "failure_rate": 0.0,
        "elapsed_s": 2.0,
        "observed_input_tokens": 128,
        "observed_output_tokens": 96,
        "output_tps": 48.0,
        "ttft_p95_s": 0.2,
        "tpot_p95_s": 0.01,
        "e2e_p95_s": 0.5,
        "deadline_s": 60.0,
        "drain_s": 0.1,
        "requests": [request],
    }


def _arm(method, *, mutate=None):
    launch, endpoint = _launch(method)
    arm = {
        "status": "PASS",
        "mode": method,
        "launch_receipt": launch,
        "endpoint": endpoint,
        "endpoint_after": endpoint.copy(),
        "metrics_before": _raw_metrics(0, 0, 0),
        "metrics_after": _raw_metrics(10, 50, 20),
        "reports": [_report()],
        "total_output_tps": 48.0,
    }
    if mutate:
        mutate(arm)
    return arm


class ReportCLI(unittest.TestCase):
    def report(self, greedy, probabilistic):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("greedy.json", "probabilistic.json")]
            for path, data in zip(paths, (greedy, probabilistic), strict=True):
                path.write_text(json.dumps(data))
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.benchmarks.dspark_replay_report",
                    *map(str, paths),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_fixed_pair_reports_per_dp_raw_draft_metrics(self):
        result = self.report(_arm("greedy"), _arm("probabilistic"))
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["kind"], "dspark_replay_comparison")
        self.assertEqual(report["performance_comparison"]["status"], "VALID")
        self.assertEqual(report["arms"]["greedy"]["draft_metrics"]["by_dp"]["0"]["drafts"], 10)
        self.assertEqual(report["arms"]["probabilistic"]["draft_metrics"]["by_dp"]["1"]["accepted_tokens"], 20)
        self.assertEqual(
            report["arms"]["greedy"]["draft_metrics"]["by_dp"]["0"]["accepted_per_draft"],
            2.0,
        )

    def test_launch_drift_is_rejected_before_comparison(self):
        greedy, probabilistic = _arm("greedy"), _arm("probabilistic")
        command = probabilistic["launch_receipt"]["command"]
        command[command.index("--max-num-batched-tokens") + 1] = "32768"
        result = self.report(greedy, probabilistic)
        self.assertEqual(result.returncode, 2)
        self.assertIn("max-num-batched-tokens", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_missing_raw_metrics_are_declared_and_withhold_throughput(self):
        greedy, probabilistic = _arm("greedy"), _arm("probabilistic")
        for arm in (greedy, probabilistic):
            arm.pop("metrics_before")
            arm.pop("metrics_after")
            arm["reports"][0]["draft_counters"] = {
                "by_engine": {
                    "0": {"drafts": 10, "draft_tokens": 50, "accepted_tokens": 20},
                    "1": {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0},
                }
            }
        result = self.report(greedy, probabilistic)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["performance_comparison"]["status"], "WITHHELD")
        self.assertIn("raw /metrics before/after unavailable", report["arms"]["greedy"]["acceptance_issues"])
        self.assertEqual(report["arms"]["greedy"]["draft_metrics"]["source"], "reported_delta_without_raw_metrics")

    def test_failed_post_retains_partial_stream_diagnostic(self):
        greedy, probabilistic = _arm("greedy"), _arm("probabilistic")
        for arm in (greedy, probabilistic):
            row = arm["reports"][0]["requests"][0]
            row.update(status="FAIL", error="deadline exceeded", partial_stream=True)
            for field in ("input_tokens", "output_tokens", "ttft_s", "tpot_s", "e2e_s"):
                row.pop(field, None)
            arm["reports"][0].update(status="FAIL", success=0, failure_rate=1.0)
        result = self.report(greedy, probabilistic)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["arms"]["greedy"]["failed_requests"], 1)
        self.assertTrue(report["arms"]["greedy"]["failed_rows"][0]["partial_stream"])
        self.assertIsNone(report["performance_comparison"]["probabilistic_over_greedy_output_tps"])

    def test_latency_summary_separates_output_lengths_at_2405(self):
        greedy, probabilistic = _arm("greedy"), _arm("probabilistic")
        for arm in (greedy, probabilistic):
            high = _report()
            high.update(
                case="sample-c1", round=2, output_tokens_each=2405,
                output_tps=30.0, ttft_p95_s=0.8, tpot_p95_s=0.03, e2e_p95_s=2.5,
                workload_sha256="2" * 64,
            )
            arm["reports"].append(high)
            arm["total_output_tps"] = 39.0
        result = self.report(greedy, probabilistic)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        report = json.loads(result.stdout)
        high = report["arms"]["greedy"]["latency_p95_s"]["osl_ge_2405"]
        self.assertEqual(high["ttft_p95_s"], {"n": 1, "p50": 0.8, "p95": 0.8})
        self.assertEqual(high["tpot_p95_s"], {"n": 1, "p50": 0.03, "p95": 0.03})
        self.assertEqual(high["e2e_p95_s"], {"n": 1, "p50": 2.5, "p95": 2.5})

    def test_missing_output_tps_is_withheld_without_recomputing_from_elapsed(self):
        greedy, probabilistic = _arm("greedy"), _arm("probabilistic")
        for arm in (greedy, probabilistic):
            arm["reports"][0].pop("output_tps")
            arm.pop("total_output_tps")
        result = self.report(greedy, probabilistic)
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertIn("total output TPS unavailable", report["arms"]["greedy"]["acceptance_issues"])
        self.assertIsNone(report["arms"]["greedy"]["total_output_tps"])
        self.assertIsNone(report["performance_comparison"]["probabilistic_over_greedy_output_tps"])


if __name__ == "__main__":
    unittest.main()
