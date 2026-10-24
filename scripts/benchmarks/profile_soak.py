"""Run the catalog's once-only 30-minute stress case with frozen token inputs."""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from scripts.bench_common import (
    completion_body,
    load_config,
    stream_request,
    summarize,
    verify_cache,
    verify_endpoint,
    workload_digest,
)
from scripts.benchmarks import profile_workload
from scripts.benchmarks.profile_matrix import write_json
from scripts.benchmarks.profile_measure import ServiceObserver, warmup


def run(base, frozen, case, receipt, output: Path, timeout=3600):
    if base != profile_workload.BASE_URL:
        raise ValueError("stress acceptance must use test port 6688")
    endpoint = verify_endpoint(base, receipt)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch.json", receipt)
    rows, references = [], []
    index = 0
    started = None
    observer = ServiceObserver(base, receipt["container_id"], output / "observation")

    def body(number):
        rng = random.Random(20261021 + number)
        prompt = [rng.choice(frozen["token_pool"]) for _ in range(case["isl"])]
        result = completion_body(prompt, case["osl"], case["seed"])
        result["cache_salt"] = "soak:" + workload_digest([frozen["run_id"], number])
        return result

    def send(number, request):
        row = stream_request(base, request, timeout, expected_input=case["isl"], expected_output=case["osl"])
        row.update(index=number, payload_sha256=workload_digest(request),
                   completed_at_s=time.perf_counter() - started)
        return row

    warmed, elapsed, runtime_error = None, 0, None
    try:
        with observer, ThreadPoolExecutor(max_workers=case["concurrency"]) as pool, \
                (output / "completed-requests.jsonl").open("x", buffering=1) as journal:
            warm_cell = {"cell_id": "stress-warmup", "isl": case["isl"], "osl": 128,
                         "concurrency": case["concurrency"], "warmup_concurrency": case["concurrency"]}
            def warm_factory(label, number):
                request = body(number)
                request["max_tokens"] = 128
                request["cache_salt"] = "stress-warm:" + workload_digest([frozen["run_id"], label, number])
                return request
            warmed = warmup(base, warm_factory, warm_cell, observer, output / "warmup")
            if warmed["status"] != "PASS":
                raise RuntimeError("stress DP warmup failed; no formal stress requests were sent")
            started = time.perf_counter()
            deadline = started + case["duration_s"]
            pending = {}
            for _ in range(case["concurrency"]):
                request = body(index)
                pending[pool.submit(send, index, request)] = index
                references.append({"index": index, "payload_sha256": workload_digest(request)})
                index += 1
            while pending:
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    number = pending.pop(future)
                    try:
                        row = future.result()
                    except Exception as error:
                        row = {"index": number, "status": "FAIL", "error": str(error)}
                    rows.append(row)
                    journal.write(json.dumps(row) + "\n")
                    if time.perf_counter() < deadline:
                        request = body(index)
                        pending[pool.submit(send, index, request)] = index
                        references.append({"index": index, "payload_sha256": workload_digest(request)})
                        index += 1
                write_json(output / "progress.json", {"status": "IN_PROGRESS", "duration_s": time.perf_counter() - started,
                                                       "submitted": index, "completed": len(rows),
                                                       "failed": sum(r["status"] != "PASS" for r in rows)})
            elapsed = time.perf_counter() - started
    except Exception as error:
        runtime_error = f"{type(error).__name__}: {error}"
        if started is not None and not elapsed:
            elapsed = time.perf_counter() - started
    report = summarize(sorted(rows, key=lambda r: r["index"]), elapsed,
                       case=case["id"], duration_s=case["duration_s"], concurrency=case["concurrency"])
    report.update(endpoint=endpoint, frozen_inputs_sha256=frozen["sha256"], cache_verified=verify_cache(rows, "cold"),
                  measurement_window_s=case["duration_s"], drain_elapsed_s=max(0, elapsed - case["duration_s"]),
                  observation=observer.summary(), request_identities=references, warmup=warmed,
                  runtime_error=runtime_error)
    if runtime_error or not report["cache_verified"] or report["observation"]["status"] != "PASS" or len(rows) != index:
        report["status"] = "FAIL"
    profile_workload.write_gzip(output / "request-metrics.json.gz", report.pop("requests"))
    try:
        if verify_endpoint(base, receipt) != endpoint:
            raise ValueError("container identity changed")
    except Exception as error:
        report.update(status="FAIL", endpoint_error=f"{type(error).__name__}: {error}")
    write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = next(c for c in load_config("long_tail")["cases"] if c["kind"] == "soak")
    receipt = json.loads(args.launch_receipt.read_text())
    frozen = profile_workload.load(args.inputs)
    result = run(profile_workload.BASE_URL, frozen, case, receipt, args.output)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
