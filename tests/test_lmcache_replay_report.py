"""Offline A/B reports from one frozen LMCache replay workload."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

IMAGE = "sha256:" + "a" * 64
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
MODEL_MOUNT = "/models/DeepSeek-V4.1-Flash-1017"


def arm_launch(enabled):
    name = "lmcache-test-on" if enabled else "lmcache-test-off"
    docker = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--network=host",
        "--ipc=host",
        "--mount",
        f"type=bind,src=/verified/weights,dst={MODEL_MOUNT},readonly",
        "--mount",
        f"type=volume,src={name}-cache,dst=/workspace",
        "--env",
        "VLLM_WORKER_MULTIPROC_METHOD=spawn",
        IMAGE,
    ]
    serve = [
        "serve",
        MODEL_MOUNT,
        "--served-model-name",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "16688",
        "--max-model-len",
        "1048576",
        "--gpu-memory-utilization",
        "0.925",
        "--max-num-batched-tokens",
        "16384",
    ]
    connector = None
    if enabled:
        connector = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://127.0.0.1",
                "lmcache.mp.port": 15555,
                "lmcache.mp.mp_transfer_mode": "lmcache_driven",
                "lmcache.mp.mq_timeout": 300,
                "lmcache.mp.heartbeat_interval": 5,
                "lmcache.mp.lazy_offload": False,
            },
        }
        serve += ["--kv-transfer-config", json.dumps(connector)]
    container_id = ("c" if enabled else "b") * 64
    receipt = {
        "command": docker + serve,
        "image_id": IMAGE,
        "container_id": container_id,
        "source_manifest_sha256": "d" * 64,
        "weights_rechecked_at_utc": "2026-11-20T01:00:00Z",
        "model_manifest_sha256": "d" * 64,
        "model_revision": "verified-1017",
        "kv_transfer_config": connector,
    }
    if enabled:
        receipt.update(
            lmcache_container_id="e" * 64,
            lmcache_healthcheck="PASS",
            lmcache_mp={
                "l1_size_gb": 512,
                "l1_init_size_gb": 16,
                "chunk_size": 128,
                "port": 15555,
                "http_port": 19180,
            },
            lmcache_server_command=[
                "docker",
                "run",
                "-d",
                "--pull",
                "never",
                "--init",
                "--name",
                f"{name}-lmcache",
                "--label",
                "vllm-dsf-rearguard.owner=vllm-dsf-rearguard",
                "--gpus",
                "all",
                "--network=host",
                "--ipc=host",
                "--ulimit",
                "memlock=-1",
                "--log-opt",
                "max-size=100m",
                "--log-opt",
                "max-file=5",
                "--entrypoint",
                "lmcache",
                IMAGE,
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                "15555",
                "--http-host",
                "127.0.0.1",
                "--http-port",
                "19180",
                "--l1-size-gb",
                "512",
                "--l1-use-lazy",
                "--l1-init-size-gb",
                "16",
                "--eviction-policy",
                "LRU",
                "--chunk-size",
                "128",
                "--separate-object-groups",
            ],
        )
    endpoint = {
        "container_id": container_id,
        "image_id": IMAGE,
        "started_at": "2026-11-20T01:15:00Z",
        "command": serve,
        "network_mode": "host",
        "ipc_mode": "host",
        "port": 16688,
    }
    return {
        "base_url": "http://127.0.0.1:16688",
        "launch_receipt": receipt,
        "endpoint": endpoint,
        "endpoint_after": endpoint.copy(),
    }


def fixture_digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def make_arm(enabled, *, mode="fixed"):
    namespace = "1" * 32 if enabled else "0" * 32
    rows = []
    for turn, (planned, prompt) in enumerate([(0.0, 131072), (2.0, 262144)], 1):
        queue = float(turn)
        http_ttft, http_e2e = (1.0, 2.0) if enabled else (2.0, 4.0)
        start = planned + queue + (0 if turn == 1 else 3)
        rows.append(
            {
                "user_id": "user-0",
                "chain_id": "user-0/session-0",
                "cycle": 0,
                "turn": turn,
                "request_sha256": str(turn) * 64,
                "cache_salt": f"replay-{namespace}-user-0-0",
                "planned_at_s": planned,
                "actual_started_s": start,
                "completed_at_s": start + http_e2e,
                "posted": True,
                "status": "PASS",
                "client_queue_s": start - planned,
                "expected_input_tokens": prompt,
                "expected_output_tokens": 8,
                "input_tokens": prompt,
                "output_tokens": 8,
                "cached_tokens": 128 if enabled and turn == 2 else 0,
                "ttft_s": http_ttft,
                "e2e_s": http_e2e,
                "output_sha256": "e" * 64,
                "finish_reason": "length",
            }
        )
    plan = [
        [r["chain_id"], r["turn"], r["planned_at_s"], r["request_sha256"]] for r in rows
    ]
    return {
        **arm_launch(enabled),
        "schema_version": 1,
        "kind": "lmcache_replay",
        "status": "PASS",
        "workload_kind": "synthetic",
        "mode": mode,
        "model": MODEL,
        "users": 1,
        "turns": 2,
        "duration_s": 600.0,
        "elapsed_s": 9.0 if enabled else 11.0,
        "request_timeout_s": 600.0,
        "namespace": namespace,
        "workload_sha256": "f" * 64,
        "runner_sha256": "d" * 64,
        "offered_schedule_sha256": fixture_digest([mode, 600.0, plan]),
        "completion_post_attempts": len(rows),
        "completion_offered": len(rows),
        "completion_retry_limit": 0,
        "rows": rows,
        "errors": [],
    }


def make_closed_arm(enabled):
    arm = make_arm(enabled, mode="closed")
    arm.update(users=70, completion_offered=None, rows=[])
    schedule = []
    for index in range(70):
        user, planned = f"user{index:06}", index / 10
        turn_hashes = [str(turn) * 64 for turn in (1, 2)]
        schedule.append([user, planned, turn_hashes])
        for row_index, latency in enumerate((570, 15, 21) if enabled else (589, 12)):
            cycle, turn_index = divmod(row_index, 2)
            prompt = (131072, 262144)[turn_index]
            arm["rows"].append(
                {
                    "user_id": user,
                    "chain_id": f"{user}/session-{cycle}",
                    "cycle": cycle,
                    "turn": turn_index + 1,
                    "request_sha256": turn_hashes[turn_index],
                    "cache_salt": f"replay-{arm['namespace']}-{user}-{cycle}",
                    "planned_at_s": planned,
                    "actual_started_s": planned,
                    "completed_at_s": planned + latency,
                    "posted": True,
                    "status": "PASS",
                    "client_queue_s": 0.0,
                    "expected_input_tokens": prompt,
                    "expected_output_tokens": 8,
                    "input_tokens": prompt,
                    "output_tokens": 8,
                    "cached_tokens": 0,
                    "ttft_s": 1.0,
                    "e2e_s": latency,
                    "output_sha256": "e" * 64,
                    "finish_reason": "length",
                }
            )
            planned += latency
    arm.update(
        elapsed_s=max(row["completed_at_s"] for row in arm["rows"]),
        completion_post_attempts=len(arm["rows"]),
        offered_schedule_sha256=fixture_digest(["closed", 600.0, schedule]),
    )
    return with_metrics(arm, external=(0, 0) if enabled else (None, None))


def with_metrics(report, *, external=(None, None)):
    metric_names = {
        "finished_requests": "vllm:request_success_total",
        "prompt_tokens": "vllm:prompt_tokens_total",
        "generation_tokens": "vllm:generation_tokens_total",
        "local_hits": "vllm:prefix_cache_hits_total",
        "external_queries": "vllm:external_prefix_cache_queries_total",
        "external_hits": "vllm:external_prefix_cache_hits_total",
        "external_prefill_tokens": "vllm:prompt_tokens_by_source_total",
    }
    before, after, delta = {}, {}, {}
    for rank in range(2):
        success = [row for row in report["rows"][rank::2] if row["status"] == "PASS"]
        hit = external[rank]
        delta[str(rank)] = {
            "finished_requests": len(success),
            "prompt_tokens": sum(row["input_tokens"] for row in success),
            "generation_tokens": sum(row["output_tokens"] for row in success),
            "local_hits": 0,
            "external_queries": None if hit is None else 4096,
            "external_hits": None if hit is None else hit * 2,
            "external_prefill_tokens": hit,
        }
        for metric, value in delta[str(rank)].items():
            if value is None:
                continue
            key = f'{metric_names[metric]}{{engine="{rank}",model_name="{MODEL}"'
            if metric == "external_prefill_tokens":
                key += ',source="external_kv_transfer"'
            elif metric == "finished_requests":
                key += ',finished_reason="length"'
            key += "}"
            before[key] = 100.0
            after[key] = 100.0 + value
    report.update(
        metrics_before=before, metrics_after=after, metrics_delta_by_engine=delta
    )
    return report


class ReportCLI(unittest.TestCase):
    def report(self, off, on):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("off.json", "on.json")]
            for path, data in zip(paths, (off, on), strict=True):
                path.write_text(json.dumps(data))
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.benchmarks.lmcache_replay_report",
                    *map(str, paths),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_fixed_replay_keeps_http_and_arrival_latency_separate(self):
        result = self.report(
            with_metrics(make_arm(False)),
            with_metrics(make_arm(True), external=(0, 96)),
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["mode"], "fixed")
        self.assertEqual(report["workload_kind"], "synthetic")
        off = report["arms"]["off"]
        self.assertEqual(off["completion_post_attempts"], 2)
        self.assertAlmostEqual(off["output_tokens_per_s"], 16 / 11)
        self.assertEqual(off["full_round_complete_s"], 11)
        self.assertEqual(off["latency_s"]["http_ttft"], {"n": 2, "p50": 2, "p95": 2})
        self.assertEqual(off["latency_s"]["arrival_ttft"], {"n": 2, "p50": 3, "p95": 7})
        self.assertEqual(off["latency_s"]["http_e2e"], {"n": 2, "p50": 4, "p95": 4})
        self.assertEqual(off["latency_s"]["arrival_e2e"], {"n": 2, "p50": 5, "p95": 9})
        self.assertEqual(report["arms"]["on"]["cache"]["api_cached_tokens"], 128)
        self.assertAlmostEqual(
            report["performance_comparison"]["on_over_off_output_tokens_per_s"], 11 / 9
        )

    def test_sidecar_receipt_rejects_a_non_loopback_http_bind(self):
        off, on = make_arm(False), make_arm(True)
        sidecar = on["launch_receipt"]["lmcache_server_command"]
        sidecar[sidecar.index("--http-host") + 1] = "0.0.0.0"
        result = self.report(off, on)
        self.assertEqual(result.returncode, 2)
        self.assertIn("LMCache sidecar command", result.stderr)

    def test_closed_replay_compares_window_throughput_with_unequal_completion_counts(
        self,
    ):
        off, on = make_closed_arm(False), make_closed_arm(True)
        result = self.report(off, on)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["mode"], "closed")
        self.assertEqual(report["arms"]["off"]["completion_post_attempts"], 140)
        self.assertEqual(report["arms"]["off"]["completed_ok"], 140)
        self.assertEqual(report["arms"]["on"]["completion_post_attempts"], 210)
        self.assertEqual(report["arms"]["on"]["completed_during_drain"], 70)
        self.assertEqual(
            report["arms"]["on"]["output_tokens_completed_within_window"], 1120
        )
        self.assertAlmostEqual(report["arms"]["on"]["output_tokens_per_s"], 1120 / 600)
        self.assertEqual(
            report["performance_comparison"]["on_over_off_output_tokens_per_s"], 2
        )
        on["users"] = off["users"] = 69
        result = self.report(off, on)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires 70 users", result.stderr)

    def test_failed_posts_and_unsent_turns_are_retained_and_block_acceptance(self):
        off, on = make_arm(False), make_arm(True)
        on["rows"][0].update(
            status="FAIL",
            error="TimeoutError: completion deadline exceeded",
            output_tokens=None,
            input_tokens=None,
            ttft_s=None,
        )
        on["rows"][1].update(
            status="NOT_SENT",
            error="preflight stopped further turns",
            posted=False,
            actual_started_s=None,
            completed_at_s=None,
            client_queue_s=None,
            input_tokens=None,
            output_tokens=None,
            cached_tokens=None,
            ttft_s=None,
            e2e_s=None,
        )
        on.update(status="FAIL", completion_post_attempts=1)
        result = self.report(with_metrics(off), with_metrics(on, external=(0, 0)))
        self.assertEqual(result.returncode, 1, result.stderr or result.stdout)
        report = json.loads(result.stdout)
        arm = report["arms"]["on"]
        self.assertEqual(arm["failed_posts"], 1)
        self.assertEqual(arm["not_sent"], 1)
        self.assertEqual(arm["full_round_complete_s"], None)
        self.assertEqual(
            [row["error"] for row in arm["failed_rows"]],
            [
                "TimeoutError: completion deadline exceeded",
                "preflight stopped further turns",
            ],
        )
        self.assertIsNone(
            report["performance_comparison"]["on_over_off_output_tokens_per_s"]
        )

    def test_real_external_tokens_come_from_prompt_source_counters(self):
        off = with_metrics(make_arm(False))
        on = with_metrics(make_arm(True), external=(0, 96))
        result = self.report(off, on)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        report = json.loads(result.stdout)
        cache = report["arms"]["on"]["cache"]
        self.assertEqual(cache["api_cached_tokens"], 128)
        self.assertEqual(cache["external_lookup_hit_tokens"], 192)
        self.assertEqual(cache["external_prompt_tokens"], 96)
        self.assertEqual(cache["dp_counters"]["1"]["external_prefill_tokens"], 96)
        self.assertIsNone(report["arms"]["off"]["cache"]["external_prompt_tokens"])
        on["metrics_delta_by_engine"]["1"]["external_prefill_tokens"] = 512
        result = self.report(off, on)
        self.assertEqual(result.returncode, 1, result.stderr or result.stdout)
        rejected = json.loads(result.stdout)
        self.assertEqual(rejected["performance_comparison"]["status"], "WITHHELD")
        self.assertIn(
            "raw metrics", " ".join(rejected["arms"]["on"]["acceptance_issues"])
        )

    def test_incomparable_workloads_are_rejected_before_performance_comparison(self):
        cases = (
            ("fixed and closed", lambda off, on: on.update(mode="closed")),
            (
                "independent replay salt",
                lambda off, on: on.update(namespace=off["namespace"]),
            ),
            (
                "offered schedule hash",
                lambda off, on: on["rows"][1].update(
                    planned_at_s=3.0, client_queue_s=4.0
                ),
            ),
            (
                "input length, output contract",
                lambda off, on: on["rows"][1].update(
                    expected_output_tokens=9, output_tokens=9
                ),
            ),
            (
                "POST attempt count",
                lambda off, on: on.update(completion_post_attempts=3),
            ),
            (
                "model_manifest_sha256",
                lambda off, on: on["launch_receipt"].update(
                    model_manifest_sha256="9" * 64
                ),
            ),
        )
        for diagnostic, mutate in cases:
            with self.subTest(diagnostic=diagnostic):
                off, on = make_arm(False), make_arm(True)
                mutate(off, on)
                result = self.report(off, on)
                self.assertEqual(result.returncode, 2, result.stderr or result.stdout)
                self.assertIn(diagnostic, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_preflight_failure_reports_all_planned_unsent_rows(self):
        off, on = with_metrics(make_arm(False)), make_arm(True)
        on.update(
            endpoint=None,
            endpoint_after=None,
            elapsed_s=None,
            status="FAIL",
            completion_post_attempts=0,
            errors=["endpoint: refused connection"],
        )
        for row in on["rows"]:
            row.update(
                status="NOT_SENT",
                posted=False,
                error="preflight did not complete",
                actual_started_s=None,
                completed_at_s=None,
                client_queue_s=None,
            )
        result = self.report(off, on)
        self.assertEqual(result.returncode, 1, result.stderr or result.stdout)
        arm = json.loads(result.stdout)["arms"]["on"]
        self.assertEqual(arm["completion_offered"], 2)
        self.assertEqual(arm["not_sent"], 2)
        self.assertEqual(arm["errors"], ["endpoint: refused connection"])

    def test_actual_child_command_drift_blocks_comparability(self):
        off, on = arm_launch(False), arm_launch(True)
        for evidence, field in (
            (on["launch_receipt"], "command"),
            (on["endpoint"], "command"),
        ):
            evidence[field] = [
                "32768" if value == "16384" else value for value in evidence[field]
            ]
        on["endpoint_after"] = on["endpoint"].copy()
        result = self.report(off, on)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("actual launch command differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
