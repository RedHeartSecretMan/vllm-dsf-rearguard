"""Fixed offered-load counterpart of the catalog's 300-second mixed case.

The original closed loop submits different request counts as performance changes.
This explicitly named variant freezes arrivals and request counts instead.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from scripts.bench_common import (
    completion_body,
    percentile,
    stream_request,
    summarize,
    verify_cache,
    workload_digest,
)
from scripts.benchmarks.profile_workload import BASE_URL, write_gzip


def schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for batch in range(config["decode_batches"]):
        for index in range(config["decode_requests_per_batch"]):
            result.append({"stream": "decode", "index": batch * config["decode_requests_per_batch"] + index,
                           "at_s": batch * config["decode_batch_interval_s"]})
    for index in range(config["prefill_requests"]):
        result.append({"stream": "prefill", "index": index, "at_s": index * config["prefill_interval_s"]})
    return sorted(result, key=lambda row: (row["at_s"], row["stream"], row["index"]))


def run(base: str, cell: dict[str, Any], frozen: dict[str, Any], policy: dict[str, Any],
        output: Path, timeout: float = 3600) -> dict[str, Any]:
    if base != BASE_URL:
        raise ValueError("mixed profiling must use port 6688")
    output.mkdir(parents=True, exist_ok=False)
    case = cell["source_case"]
    arrivals = schedule(policy)
    bodies = []
    for item in arrivals:
        kind, index = item["stream"], item["index"]
        isl, osl = case[f"{kind}_isl"], case[f"{kind}_osl"]
        rng = random.Random((20261021 if kind == "decode" else 20261022) + index)
        prompt = [rng.choice(frozen["token_pool"]) for _ in range(isl)]
        body = completion_body(prompt, osl, case["seed"])
        body["cache_salt"] = f"profile:{frozen['run_id']}:mixed-fixed-arrivals:{kind}:{index}"
        bodies.append(body)
    write_gzip(output / "requests.json.gz", {"arrivals": arrivals, "bodies": bodies})
    started = time.perf_counter()

    def execute(item, body):
        actual_started = time.perf_counter() - started
        row = stream_request(base, body, timeout, expected_input=case[f"{item['stream']}_isl"],
                             expected_output=case[f"{item['stream']}_osl"])
        queued = max(0, actual_started - item["at_s"])
        completed = time.perf_counter() - started
        return {**row, **item, "actual_started_s": actual_started,
                "client_queue_delay_s": max(0, actual_started - item["at_s"]),
                "completed_at_s": completed,
                "arrival_to_first_token_s": queued + row["ttft_s"] if row.get("ttft_s") is not None else None,
                "arrival_to_completion_s": completed - item["at_s"]}

    futures = []
    with ThreadPoolExecutor(max_workers=case["decode_concurrency"]) as dec, \
            ThreadPoolExecutor(max_workers=case["prefill_max_inflight"]) as pre:
        for item, body in zip(arrivals, bodies, strict=True):
            delay = item["at_s"] - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
            futures.append((dec if item["stream"] == "decode" else pre).submit(execute, item, body))
        delay = policy["duration_s"] - (time.perf_counter() - started)
        if delay > 0:
            time.sleep(delay)
        rows = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    streams = {}
    for kind in ("decode", "prefill"):
        selected = [row for row in rows if row["stream"] == kind]
        report = summarize(selected, elapsed)
        for name in ("client_queue_delay_s", "arrival_to_first_token_s", "arrival_to_completion_s"):
            values = [r[name] for r in selected if r.get(name) is not None]
            report[name] = {"p50": percentile(values, .5), "p95": percentile(values, .95),
                            "p99": percentile(values, .99), "max": max(values) if values else None}
        report["cache_verified"] = verify_cache(selected, "cold")
        if not report["cache_verified"]:
            report["status"] = "FAIL"
        streams[kind] = report
    result = {"cell_id": cell["cell_id"], "case": case["id"],
              "execution_variant": "fixed_arrival_schedule", "source_duration_s": case["duration_s"],
              "arrivals_sha256": workload_digest(arrivals), "payload_sha256": workload_digest(bodies),
              "request_count": len(rows), "expected_request_count": len(arrivals),
              "measurement_window_s": policy["duration_s"], "elapsed_s": elapsed,
              "drain_elapsed_s": max(0, elapsed - policy["duration_s"]),
              "streams": streams, "status": "PASS" if all(r["status"] == "PASS" for r in streams.values()) else "FAIL"}
    write_gzip(output / "result.json.gz", result)
    return result
