"""Public CLI contracts using Docker inspect and loopback HTTP fixtures."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

from scripts import bench_common as bench

MODULE = "scripts.benchmarks.lmcache_replay"
CONTAINER_ID = "a" * 64
IMAGE_ID = "sha256:" + "b" * 64


class ReplayFixture:
    def __init__(self, directory):
        self.directory = directory
        self.requests = []
        self.http_requests = []
        self.counts = [[0, 0, 0], [0, 0, 0]]
        self.ranks = {}
        self.delay = 0.02
        self.drip = False
        self.token_only_drip = False
        self.omit_token_ids = False
        self.lock = Lock()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_body(self, status, text):
                data = text.encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fixture.http_requests.append(("GET", self.path))
                if self.path != "/metrics":
                    self.send_body(404, "unexpected GET")
                    return
                with fixture.lock:
                    metrics = fixture.metrics()
                self.send_body(200, metrics)

            def do_POST(self):
                fixture.http_requests.append(("POST", self.path))
                if self.path != "/v1/completions":
                    self.send_body(404, "unexpected POST")
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stream = fixture.complete(body)
                if not fixture.drip and not fixture.token_only_drip:
                    self.send_body(200, stream)
                    return
                self.send_response(200)
                self.end_headers()
                try:
                    if fixture.token_only_drip:
                        first, *rest = stream.split("\n\n")
                        self.wfile.write((first + "\n\n").encode())
                        self.wfile.flush()
                        drip = "\n\n".join(rest)
                    else:
                        drip = stream.splitlines()[0] + "\n"
                    for byte in drip.encode():
                        self.wfile.write(bytes([byte]))
                        time.sleep(0.025)
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        command = [
            "serve", "/models/1017", "--served-model-name", bench.MODEL,
            "--host", "127.0.0.1", "--port", str(self.server.server_port),
            "--max-model-len", "1048576", "--data-parallel-size", "2",
        ]
        receipt = {
            "host_network_ipc": True,
            "command": ["docker", "run", "--network=host", "--ipc=host", IMAGE_ID, *command],
            "container_id": CONTAINER_ID,
            "image_id": IMAGE_ID,
            "context": 1048576,
        }
        self.receipt_path = directory / "launch.json"
        self.receipt_path.write_text(json.dumps(receipt))
        self.container_path = directory / "inspect.json"
        self.container = {
            "Id": CONTAINER_ID,
            "Image": IMAGE_ID,
            "Config": {"Cmd": command},
            "State": {"Running": True, "StartedAt": "2026-11-21T09:00:00Z"},
            "HostConfig": {"NetworkMode": "host", "IpcMode": "host"},
        }
        self.container_path.write_text(json.dumps([self.container]))
        docker = directory / "docker"
        docker.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "from pathlib import Path\n"
            f"assert sys.argv[1:] == ['inspect', '{CONTAINER_ID}']\n"
            "print(Path(os.environ['REPLAY_TEST_DOCKER_INSPECT']).read_text())\n"
        )
        docker.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{directory}{os.pathsep}{os.environ.get('PATH', '')}",
            "REPLAY_TEST_DOCKER_INSPECT": str(self.container_path),
            "BENCH_API_KEY": "",
        }

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def complete(self, body):
        started = time.monotonic()
        with self.lock:
            salt = body["cache_salt"]
            rank = self.ranks.setdefault(salt, len(self.ranks) % 2)
            cached = 128 if any(
                row["body"]["cache_salt"] == salt for row in self.requests
            ) else 0
            self.requests.append({"body": body, "started": started, "completed": None})
            row = self.requests[-1]
        time.sleep(self.delay)
        prompt, output = len(body["prompt"]), body["max_tokens"]
        with self.lock:
            self.counts[rank][0] += 1
            self.counts[rank][1] += prompt
            self.counts[rank][2] += output
            row["completed"] = time.monotonic()
        first = (
            'data: {"choices":[{"text":"","token_ids":[190]}]}'
            if self.token_only_drip
            else (
                'data: {"choices":[{"text":"example","token_ids":[190,191],'
                '"finish_reason":"length"}]}'
                if body.get("return_token_ids") and not self.omit_token_ids
                else 'data: {"choices":[{"text":"example","finish_reason":"length"}]}'
            )
        )
        return "\n\n".join([
            first,
            "data: " + json.dumps({
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": output,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }),
            "data: [DONE]\n\n",
        ])

    def metrics(self):
        lines = []
        for engine, (requests, prompt, output) in enumerate(self.counts):
            labels = f'model_name="{bench.MODEL}",engine="{engine}"'
            lines.extend([
                f'vllm:request_success_total{{{labels},finished_reason="length"}} {requests}',
                f"vllm:prompt_tokens_total{{{labels}}} {prompt}",
                f"vllm:generation_tokens_total{{{labels}}} {output}",
                f"vllm:prefix_cache_hits_total{{{labels}}} 0",
                f"vllm:external_prefix_cache_queries_total{{{labels}}} 0",
                f"vllm:external_prefix_cache_hits_total{{{labels}}} 0",
                f'vllm:prompt_tokens_by_source_total{{{labels},source="external_kv_transfer"}} 0',
                f"vllm:num_requests_waiting{{{labels}}} 0",
            ])
        return "\n".join(lines)

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", MODULE, *map(str, args)],
            env=self.env, capture_output=True, text=True, timeout=10,
        )

    def prepare(self, *more):
        path = self.directory / "workload.json"
        result = self.cli(
            "prepare", "--users", 2, "--turns", 2, "--seed", 7,
            "--input-tokens", 257, "--input-step", 16, "--output-tokens", 2,
            "--user-spacing", 0, "--turn-spacing", 0.01, *more, "--output", path,
        )
        if result.returncode:
            raise AssertionError(result.stderr or result.stdout)
        return path

    def run(self, workload, filename, mode="fixed", *more, duration=1):
        path = self.directory / filename
        timing = [] if duration is None else ["--duration", duration]
        result = self.cli(
            "run", "--mode", mode, "--workload", workload,
            "--base-url", self.base, "--launch-receipt", self.receipt_path,
            *timing, "--timeout", 1, *more, "--output", path,
        )
        return result, json.loads(path.read_text()) if path.exists() else None


class ReplayCliTests(unittest.TestCase):
    def test_fixed_replays_same_frozen_workload_with_fresh_session_salts(self):
        with tempfile.TemporaryDirectory() as temporary, ReplayFixture(Path(temporary)) as service:
            workload = service.prepare()
            self.assertEqual(json.loads(workload.read_text())["workload_kind"], "synthetic")
            result, first = service.run(workload, "first.json", "fixed", "--run-key", "repeat-1", duration=600)
            self.assertEqual(result.returncode, 0, first or result.stderr)
            result, second = service.run(workload, "second.json", "fixed", "--run-key", "repeat-1", duration=None)
            self.assertEqual(result.returncode, 0, second or result.stderr)

        self.assertEqual(first["mode"], "fixed")
        self.assertEqual(first["completion_offered"], 4)
        self.assertEqual(first["completion_post_attempts"], 4)
        self.assertEqual(first["endpoint"]["container_id"], CONTAINER_ID)
        self.assertEqual(first["endpoint"], first["endpoint_after"])
        self.assertEqual(first["workload_sha256"], second["workload_sha256"])
        self.assertEqual(first["offered_schedule_sha256"], second["offered_schedule_sha256"])
        self.assertNotEqual(first["namespace"], second["namespace"])
        same_fields = ("chain_id", "turn", "planned_at_s", "request_sha256", "output_sha256")
        self.assertEqual(
            [tuple(r[key] for key in same_fields) for r in first["rows"]],
            [tuple(r[key] for key in same_fields) for r in second["rows"]],
        )
        self.assertEqual({r["input_tokens"] for r in first["rows"]}, {257, 273})
        self.assertEqual({r["output_tokens"] for r in first["rows"]}, {2})
        self.assertTrue(all(r["posted"] and r["ttft_s"] >= 0 for r in first["rows"]))
        self.assertNotIn("record_progress", first)
        self.assertTrue(all("partial_stream" not in r for r in first["rows"]))
        self.assertEqual(
            [first["metrics_delta_by_engine"][str(rank)]["finished_requests"] for rank in (0, 1)],
            [2, 2],
        )
        by_salt = {}
        for request in service.requests:
            by_salt.setdefault(request["body"]["cache_salt"], []).append(request)
        self.assertEqual(len(by_salt), 4)
        for early, late in by_salt.values():
            early_body, late_body = early["body"], late["body"]
            self.assertEqual(late_body["prompt"][:257], early_body["prompt"])
            self.assertGreaterEqual(late["started"], early["completed"])
            self.assertEqual(early_body["seed"], 7)
            self.assertEqual(early_body["temperature"], 0)
            self.assertTrue(early_body["ignore_eos"])
            self.assertFalse(early_body["add_special_tokens"])

    def test_record_progress_full_stream_keeps_success_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary, ReplayFixture(Path(temporary)) as service:
            workload = service.prepare("--users", 1, "--turns", 1)
            result, baseline = service.run(workload, "baseline.json", "fixed", duration=600)
            self.assertEqual(result.returncode, 0, baseline or result.stderr)
            result, report = service.run(
                workload,
                "complete-progress.json",
                "fixed",
                "--record-progress",
                duration=600,
            )
            self.assertEqual(result.returncode, 0, report or result.stderr)
            self.assertTrue(report["record_progress"])
            self.assertFalse(service.requests[0]["body"].get("return_token_ids", False))
            self.assertTrue(service.requests[1]["body"]["return_token_ids"])

        row, = report["rows"]
        self.assertNotEqual(row["request_sha256"], baseline["rows"][0]["request_sha256"])
        self.assertNotEqual(
            report["offered_schedule_sha256"], baseline["offered_schedule_sha256"]
        )
        self.assertEqual(
            report["offered_schedule_sha256"],
            bench.workload_digest(
                ["fixed", 600.0, [[row["chain_id"], row["turn"], row["planned_at_s"], row["request_sha256"]]]]
            ),
        )
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(row["partial_stream"]["output_events"], 1)
        self.assertEqual(row["partial_stream"]["output_token_ids_seen"], 2)
        self.assertEqual(row["output_tokens"], 2)

    def test_record_progress_does_not_estimate_missing_token_ids(self):
        with tempfile.TemporaryDirectory() as temporary, ReplayFixture(Path(temporary)) as service:
            service.omit_token_ids = True
            workload = service.prepare("--users", 1, "--turns", 1)
            result, report = service.run(
                workload,
                "missing-token-ids.json",
                "fixed",
                "--record-progress",
                duration=600,
            )
            self.assertEqual(result.returncode, 0, report or result.stderr)

        row, = report["rows"]
        self.assertEqual(row["partial_stream"]["output_events"], 1)
        self.assertIsNone(row["partial_stream"]["output_token_ids_seen"])

    def test_completion_deadline_interrupts_a_trickling_sse_line_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary, ReplayFixture(Path(temporary)) as service:
            service.drip = True
            workload = service.prepare("--users", 1, "--turns", 1)
            result, report = service.run(workload, "timeout.json", "fixed", "--timeout", 0.12)
            self.assertEqual(result.returncode, 1, report or result.stderr)
            self.assertEqual(report["completion_post_attempts"], 1)
            self.assertEqual(len(service.requests), 1)
        row, = report["rows"]
        self.assertEqual(row["status"], "FAIL")
        self.assertNotIn("partial_stream", row)
        self.assertIn("TimeoutError", row["error"])
        self.assertLess(report["elapsed_s"], 0.45)

    def test_record_progress_keeps_token_only_chunk_before_deadline_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary, ReplayFixture(Path(temporary)) as service:
            service.token_only_drip = True
            workload = service.prepare("--users", 1, "--turns", 1)
            result, report = service.run(
                workload,
                "partial.json",
                "fixed",
                "--record-progress",
                "--timeout",
                0.12,
            )
            self.assertEqual(result.returncode, 1, report or result.stderr)
            self.assertEqual(report["completion_post_attempts"], 1)
            self.assertEqual(len(service.requests), 1)
            self.assertTrue(service.requests[0]["body"]["return_token_ids"])

        row, = report["rows"]
        self.assertEqual(row["status"], "FAIL")
        partial = row["partial_stream"]
        self.assertEqual(partial["output_events"], 1)
        self.assertEqual(partial["output_token_ids_seen"], 1)
        self.assertGreaterEqual(partial["first_output_s"], 0)
        self.assertGreaterEqual(partial["last_output_s"], partial["first_output_s"])
        self.assertIsNone(row["output_tokens"])
        self.assertIn("TimeoutError", row["error"])

    def test_closed_rolls_over_sessions_and_reports_completion_after_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary, ReplayFixture(Path(temporary)) as service:
            service.delay = 0.24
            workload = service.prepare("--users", 1)
            result, report = service.run(workload, "closed.json", "closed", "--duration", 0.63)
            self.assertEqual(result.returncode, 0, report or result.stderr)
            self.assertEqual(report["completion_post_attempts"], len(service.requests))
        first, second, rollover = report["rows"]
        self.assertIsNone(report["completion_offered"])
        self.assertEqual([r["turn"] for r in report["rows"]], [1, 2, 1])
        self.assertEqual([r["cycle"] for r in report["rows"]], [0, 0, 1])
        self.assertEqual(first["cache_salt"], second["cache_salt"])
        self.assertNotEqual(first["cache_salt"], rollover["cache_salt"])
        self.assertEqual(first["request_sha256"], rollover["request_sha256"])
        self.assertEqual(first["completed_at_s"], second["planned_at_s"])
        self.assertLess(rollover["actual_started_s"], 0.63)
        self.assertGreater(rollover["completed_at_s"], 0.63)
        self.assertGreaterEqual(report["elapsed_s"], rollover["completed_at_s"])
        self.assertEqual(report["success_in_window"], 2)
        self.assertEqual(report["success_in_drain"], 1)

    def test_fixed_drains_all_offered_turns_even_when_serial_delay_crosses_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary, ReplayFixture(Path(temporary)) as service:
            service.delay = 0.13
            workload = service.prepare("--users", 1)
            result, report = service.run(workload, "late.json", "fixed", duration=0.08)
            self.assertEqual(result.returncode, 0, report or result.stderr)
        first, late = report["rows"]
        self.assertLess(late["planned_at_s"], 0.08)
        self.assertGreater(late["actual_started_s"], 0.08)
        self.assertGreaterEqual(late["actual_started_s"], first["completed_at_s"])
        self.assertEqual(report["completion_post_attempts"], report["completion_offered"])
        self.assertEqual(report["success_in_drain"], 2)

    def test_refuses_production_port_and_mismatched_running_argv_before_http(self):
        with tempfile.TemporaryDirectory() as temporary, ReplayFixture(Path(temporary)) as service:
            workload = service.prepare()
            result, report = service.run(
                workload, "production.json", "fixed",
                "--base-url", "http://127.0.0.1:6789",
                "--launch-receipt", service.directory / "missing.json",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("performance test port other than 6789", result.stderr)
            self.assertIsNone(report)

            service.container["Config"]["Cmd"].append("--enforce-eager")
            service.container_path.write_text(json.dumps([service.container]))
            result, report = service.run(workload, "unverified.json")
            self.assertEqual(result.returncode, 1, report or result.stderr)
            self.assertEqual(service.http_requests, [])
        self.assertIn("argv", report["errors"][0])
        self.assertIsNone(report["endpoint"])
        self.assertEqual(report["completion_post_attempts"], 0)
        self.assertEqual(report["completion_offered"], 4)
        self.assertTrue(all(r["status"] == "NOT_SENT" and not r["posted"] for r in report["rows"]))


if __name__ == "__main__":
    unittest.main()
