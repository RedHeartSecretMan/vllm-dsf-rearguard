"""CPU contracts for the probe against loopback HTTP fixtures."""

import io
import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from scripts import bench_common as bench
from scripts.benchmarks import lmcache_mp_probe as probe

CHUNK = 128
GPU_IDS = list(range(100, 108))


class ServiceFixture:
    def __init__(self, fault):
        self.fault = fault
        self.requests = []
        self.lmcache_scrapes_after_cold = 0
        self.store_observed_before_warm = False
        self.redirects_followed = 0
        self.vllm_base = None
        self.lmcache_base = None

    def vllm_metrics(self):
        counts = {0: (0, 0, 0), 1: (0, 0, 0)}
        for rank, body in self.requests:
            hot = rank == "1"
            engine = 1 if hot and self.fault != "wrong_rank" else 0
            hits = CHUNK if hot and self.fault != "no_external_hits" else 0
            counts[engine] = (1, len(body["prompt"]), hits)
        lines = []
        for engine, (requests, tokens, hits) in counts.items():
            labels = f'model_name="{bench.MODEL}",engine="{engine}"'
            if self.fault == "missing_engine" and engine == 1:
                continue
            used = 0 if self.fault == "lookup_only" else hits
            lines.extend(
                [
                    f'vllm:request_success_total{{{labels},finished_reason="stop"}} 1',
                    f'vllm:request_success_total{{{labels},finished_reason="length"}} {requests + 3}',
                    f"vllm:prompt_tokens_total{{{labels}}} {tokens + 300}",
                    f"vllm:generation_tokens_total{{{labels}}} {requests * 8 + 32}",
                    f"vllm:prefix_cache_hits_total{{{labels}}} 60",
                    f"vllm:external_prefix_cache_queries_total{{{labels}}} {tokens + 300}",
                    f"vllm:external_prefix_cache_hits_total{{{labels}}} {hits + 40}",
                    f'vllm:prompt_tokens_by_source_total{{{labels},source="local_compute"}} {tokens - hits + 200}',
                    f'vllm:prompt_tokens_by_source_total{{{labels},source="external_kv_transfer"}} {used + 40}',
                ]
            )
        return "\n".join(lines)

    def lmcache_metrics(self):
        lines = ['lmcache_mp_l1_write_chunks_total{cache_salt="unrelated"} 500']
        if self.requests:
            self.lmcache_scrapes_after_cold += 1
            cold = self.requests[0][1]
            salt = cold["cache_salt"]
            if self.lmcache_scrapes_after_cold >= 2 and self.fault != "no_store":
                keys = (len(cold["prompt"]) - 1) // CHUNK
                lines.append(
                    f'lmcache_mp_l1_write_chunks_total{{cache_salt="{salt}"}} {keys}'
                )
            if len(self.requests) == 2 and self.fault != "no_load":
                lines.append(
                    f'lmcache_mp_num_chunks_loaded_total{{cache_salt="{salt}",worker_id="104"}} 1'
                )
        return "\n".join(lines)

    def dispatch(self, handler, lmcache):
        if handler.path == "/redirected":
            self.redirects_followed += 1
            return 404, "redirect was followed"
        if handler.command == "GET" and handler.path == "/status" and lmcache:
            ids = [100] * 8 if self.fault == "duplicate_gpus" else GPU_IDS
            return 200, json.dumps({"registered_gpu_ids": ids, "chunk_size": CHUNK})
        if handler.command == "GET" and handler.path == "/metrics":
            return 200, self.lmcache_metrics() if lmcache else self.vllm_metrics()
        if handler.command == "POST" and not lmcache:
            body = json.loads(
                handler.rfile.read(int(handler.headers["Content-Length"]))
            )
            if handler.path == "/tokenize":
                return 200, json.dumps({"tokens": [101, 102, 103]})
            if handler.path == "/v1/completions":
                rank = handler.headers["X-data-parallel-rank"]
                self.requests.append((rank, body))
                if rank == "1":
                    self.store_observed_before_warm = (
                        self.lmcache_scrapes_after_cold >= 2
                    )
                if self.fault == "redirect":
                    return 302, "redirect"
                if self.fault == "broken_sse":
                    return 200, 'data: {"choices": []}\n\n'
                text = (
                    "changed"
                    if rank == "1" and self.fault == "output_changed"
                    else "same"
                )
                return 200, "\n\n".join(
                    [
                        "data: "
                        + json.dumps(
                            {"choices": [{"text": text, "finish_reason": "length"}]}
                        ),
                        "data: "
                        + json.dumps(
                            {
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": len(body["prompt"]),
                                    "completion_tokens": body["max_tokens"],
                                    # A warm usage claim alone cannot make the probe pass.
                                    "prompt_tokens_details": {
                                        "cached_tokens": CHUNK if rank == "1" else 0
                                    },
                                },
                            }
                        ),
                        "data: [DONE]\n\n",
                    ]
                )
        return 404, "unexpected HTTP request"

    @contextmanager
    def serve(self, lmcache=False):
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def serve_request(self):
                code, text = fixture.dispatch(self, lmcache)
                self.send_response(code)
                if code == 302:
                    self.send_header("Location", fixture.vllm_base + "/redirected")
                data = text.encode()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = serve_request

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            server.timeout = 0.01
            thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
            thread.start()
            try:
                yield f"http://127.0.0.1:{server.server_port}"
            finally:
                server.shutdown()
                thread.join()


class ProbeTests(unittest.TestCase):
    def run_fixture(self, fault=None, prompt_tokens=257):
        fixture = ServiceFixture(fault)
        with ExitStack() as stack:
            fixture.vllm_base = stack.enter_context(fixture.serve())
            fixture.lmcache_base = stack.enter_context(fixture.serve(lmcache=True))
            tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.dict("os.environ", {"BENCH_API_KEY": ""}))
            # Shorten asynchronous observation windows in the local fixture.
            stack.enter_context(patch.object(probe, "GET_TIMEOUT", 0.3))
            output = tmp / "probe.json"
            result = probe.main(
                [
                    "--base-url",
                    fixture.vllm_base,
                    "--lmcache-url",
                    fixture.lmcache_base,
                    "--output",
                    str(output),
                    "--store-timeout",
                    "0.3",
                    "--prompt-tokens",
                    str(prompt_tokens),
                ]
            )
            return result, json.loads(output.read_text()), fixture

    def test_cold_then_warm_across_ranks_with_matching_sse_and_counters(self):
        for prompt_tokens in (257, 20992):
            with self.subTest(prompt_tokens=prompt_tokens):
                code, report, fixture = self.run_fixture(prompt_tokens=prompt_tokens)
                self.assertEqual(code, 0, report)
                self.assertEqual(report["registration"]["before"], GPU_IDS)
                self.assertEqual(report["registration"]["after"], GPU_IDS)
                self.assertEqual(report["completion_post_attempts"], 2)
                self.assertEqual(report["completion_retry_limit"], 0)
                self.assertEqual(
                    [row["observed_engine"] for row in report["requests"]], [0, 1]
                )
                self.assertEqual(
                    report["vllm_metrics"]["cold_delta"]["0"]["external_hits"], 0
                )
                self.assertEqual(
                    report["vllm_metrics"]["warm_delta"]["1"]["external_hits"], CHUNK
                )
                self.assertEqual(
                    report["lmcache_metrics"]["before_warm"]["l1_written_keys"],
                    (prompt_tokens - 1) // CHUNK,
                )
                self.assertEqual(
                    report["lmcache_metrics"]["after_warm"]["loaded_keys"], 1
                )
                self.assertTrue(report["output_matches"])
                self.assertTrue(fixture.store_observed_before_warm)
                self.assertEqual([r for r, _ in fixture.requests], ["0", "1"])
                cold, warm = [dict(body) for _, body in fixture.requests]
                self.assertNotEqual(cold.pop("request_id"), warm.pop("request_id"))
                self.assertEqual(cold, warm)
                self.assertEqual(cold["cache_salt"], report["cache_salt"])
                self.assertEqual(len(cold["prompt"]), prompt_tokens)
                self.assertEqual(cold["temperature"], 0)

    def test_unverified_requests_write_failure_without_resending(self):
        for fault, stage, attempts in (
            ("duplicate_gpus", "lmcache_registration", 0),
            ("missing_engine", "preflight", 0),
            ("broken_sse", "cold_completion", 1),
            ("redirect", "cold_completion", 1),
            ("no_store", "cold_l1_store", 1),
            ("wrong_rank", "warm_vllm_metrics", 2),
            ("no_external_hits", "warm_vllm_metrics", 2),
            ("lookup_only", "warm_vllm_metrics", 2),
            ("no_load", "warm_lmcache_load", 2),
            ("output_changed", "final_checks", 2),
        ):
            with self.subTest(fault=fault):
                code, report, fixture = self.run_fixture(fault)
                self.assertEqual(code, 1, report)
                self.assertEqual(report["stage"], stage, report)
                self.assertEqual(report["status"], "FAIL")
                self.assertTrue(report["error"])
                self.assertEqual(report["completion_post_attempts"], attempts)
                self.assertEqual(len(fixture.requests), attempts)
                self.assertEqual(fixture.redirects_followed, 0)

    def test_prompt_needs_at_least_one_complete_chunk_and_a_tail_token(self):
        code, report, fixture = self.run_fixture(prompt_tokens=CHUNK)
        self.assertEqual(code, 1, report)
        self.assertEqual(report["full_chunks_before_last_token"], 0)
        self.assertEqual(fixture.requests, [])

    def test_requires_http_loopback_origins_with_an_explicit_port(self):
        self.assertEqual(
            probe.loopback_url("http://localhost:16688/"), "http://localhost:16688"
        )
        self.assertEqual(probe.loopback_url("http://[::1]:16688"), "http://[::1]:16688")
        for url in (
            "http://127.0.0.1",
            "https://127.0.0.1:16688",
            "http://example.com:16688",
            "http://127.0.0.1:16688/v1",
            "http://127.0.0.1:16688?",
            "http://user@localhost:16688",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                probe.loopback_url(url)
