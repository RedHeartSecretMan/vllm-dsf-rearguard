"""Bounded synthetic token benchmark, with server usage and client streaming times."""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def verify_endpoint(base, launch):
    """Bind a local measurement to the actual container, not just a model alias."""
    url = urllib.parse.urlsplit(base)
    command = launch["command"]
    binding = command[command.index("-p") + 1]
    port = int(binding.split(":")[1])
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost")
        or url.port != port
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Benchmark URL must match the local port in the launch receipt"
        )
    container_id = launch.get("container_id")
    if not container_id:
        raise ValueError("Launch receipt lacks actual container ID")
    container = json.loads(
        subprocess.check_output(["docker", "inspect", container_id])
    )[0]
    expected_cmd = command[command.index(launch["image_id"]) + 1 :]
    if (
        not container["State"]["Running"]
        or container["Image"] != launch["image_id"]
        or container["Config"]["Cmd"] != expected_cmd
    ):
        raise ValueError(
            "Container image/command/state differs from the launch receipt"
        )
    bindings = container["NetworkSettings"]["Ports"].get("8000/tcp") or []
    if not any(
        r["HostIp"] == "127.0.0.1" and int(r["HostPort"]) == port for r in bindings
    ):
        raise ValueError("Container published port differs from the receipt")
    return {
        "container_id": container["Id"],
        "started_at": container["State"]["StartedAt"],
        "image_id": container["Image"],
    }


def request(base, path, body=None, timeout=1800):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("BENCH_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["BENCH_API_KEY"]
    return urllib.request.urlopen(
        urllib.request.Request(
            base.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
        ),
        timeout=timeout,
    )


def json_request(base, path, body=None, timeout=1800):
    with request(base, path, body, timeout) as response:
        return json.load(response)


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def consume_sse(response, started, timeout, clock=time.perf_counter):
    first = last = None
    usage = None
    done = False
    chunks = 0
    finish = None
    for line in response:
        if clock() - started > timeout:
            raise TimeoutError("Total streaming deadline exceeded")
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            done = True
            break
        row = json.loads(payload)
        if row.get("error"):
            raise ValueError(str(row["error"]))
        if row.get("usage"):
            usage = row["usage"]
        for choice in row.get("choices", []):
            if choice.get("text"):
                now = clock()
                first = now if first is None else first
                last = now
                chunks += 1
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if not done or first is None or not usage or finish is None:
        raise ValueError(
            "Incomplete SSE, no text, missing finish reason or server token usage"
        )
    count = usage["completion_tokens"]
    if count <= 0:
        raise ValueError("No generated tokens")
    span = last - first
    return {
        "output_tokens": count,
        "input_tokens": usage["prompt_tokens"],
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
        "ttft_s": first - started,
        "e2e_s": clock() - started,
        "stream_decode_span_s": span,
        "content_chunks": chunks,
        "stream_tpot_s": span / (count - 1) if count > 1 and span > 0 else None,
        "stream_decode_tokens_s": (count - 1) / span
        if count > 1 and span > 0
        else None,
        "finish_reason": finish,
    }


def run_one(base, prompt, output, timeout, index):
    started = time.perf_counter()
    try:
        body = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": output,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        with request(base, "/v1/completions", body, timeout) as response:
            row = consume_sse(response, started, timeout)
        if row["output_tokens"] != output or row["input_tokens"] != len(prompt):
            raise ValueError("Server token counts differ from the fixed workload")
        return dict(index=index, status="PASS", **row)
    except Exception as error:
        return {
            "index": index,
            "status": "FAIL",
            "error": str(error),
            "e2e_s": time.perf_counter() - started,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8005")
    p.add_argument(
        "--input-tokens",
        type=int,
        choices=(1024, 32768, 131072, 523264, 1047552),
        required=True,
    )
    p.add_argument("--output-tokens", type=int, default=1024)
    p.add_argument(
        "--concurrency", type=int, choices=(1, 8, 32, 64, 128, 256), required=True
    )
    p.add_argument("--requests", type=int, help="Defaults to 2x concurrency, minimum 8")
    p.add_argument("--cache", choices=("cold", "warm"), required=True)
    p.add_argument("--seed", type=int, default=1017)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--launch-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    count = args.requests if args.requests is not None else max(8, 2 * args.concurrency)
    if count < args.concurrency or args.output_tokens < 2 or args.timeout <= 0:
        p.error("requests >= concurrency, output-tokens >= 2, and timeout > 0 required")
    if args.output.exists():
        p.error("Output already exists; use a fresh evidence path")
    launch = json.loads(args.launch_receipt.read_text())
    if "image_id" not in launch:
        p.error("A real launch receipt is required, not a dry run")
    if args.input_tokens + args.output_tokens > launch["context"]:
        p.error("Input + output exceeds the launched model context")
    runtime = verify_endpoint(args.base_url, launch)
    models = json_request(args.base_url, "/v1/models")
    if MODEL not in [row["id"] for row in models["data"]]:
        raise RuntimeError("1017 served model identity is missing")
    tokens = json_request(
        args.base_url,
        "/tokenize",
        {
            "model": MODEL,
            "prompt": "A careful engineer reviews the code and writes tests. The sky is blue. "
            "Numbers, functions, values, loops and data structures describe this example.",
        },
    )["tokens"]
    pool = sorted({token for token in tokens if 2 < token < 128000})
    if len(pool) < 16:
        raise RuntimeError("Insufficient ordinary token pool")
    rng = random.Random(args.seed)
    prompts = [
        [rng.choice(pool) for _ in range(args.input_tokens)] for _ in range(count)
    ]
    workload_hash = hashlib.sha256(
        json.dumps(prompts, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Prime only warm workloads. A cold run must use a fresh service/cache or seed.
    warmup = []
    if args.cache == "warm":
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            warmup = list(
                executor.map(
                    lambda pair: run_one(
                        args.base_url, pair[1], 1, args.timeout, pair[0]
                    ),
                    enumerate(prompts),
                )
            )
        if any(row["status"] != "PASS" for row in warmup):
            args.output.write_text(
                json.dumps(
                    {"status": "FAIL", "stage": "cache-prime", "requests": warmup},
                    indent=2,
                )
                + "\n"
            )
            return 1
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        rows = list(
            executor.map(
                lambda pair: run_one(
                    args.base_url, pair[1], args.output_tokens, args.timeout, pair[0]
                ),
                enumerate(prompts),
            )
        )
    elapsed = time.perf_counter() - started
    good = [r for r in rows if r["status"] == "PASS"]
    cache_verified = bool(good) and all(
        r["cached_tokens"] is not None
        and (
            r["cached_tokens"] == 0 if args.cache == "cold" else r["cached_tokens"] > 0
        )
        for r in good
    )
    report = {
        "status": "PASS" if len(good) == count and cache_verified else "FAIL",
        "workload": "synthetic random ordinary token IDs; not a quality or chat benchmark",
        "cache_requested": args.cache,
        "cache_verified": cache_verified,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "concurrency": args.concurrency,
        "request_count": count,
        "seed": args.seed,
        "workload_sha256": workload_hash,
        "launch": launch,
        "runtime": runtime,
        "elapsed_s": elapsed,
        "failure_rate": 1 - len(good) / count,
        "successful_output_tokens_s": sum(r["output_tokens"] for r in good) / elapsed,
        "p95_ttft_s": percentile([r["ttft_s"] for r in good], 0.95),
        "p95_e2e_s": percentile([r["e2e_s"] for r in good], 0.95),
        "p50_stream_tpot_s": percentile(
            [r["stream_tpot_s"] for r in good if r["stream_tpot_s"] is not None], 0.5
        ),
        "p50_stream_decode_tokens_s": percentile(
            [
                r["stream_decode_tokens_s"]
                for r in good
                if r["stream_decode_tokens_s"] is not None
            ],
            0.5,
        ),
        "timing_note": "Client chunk timestamps; multiple tokens per SSE chunk make decode/TPOT approximate.",
        "requests": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("requests", "launch")},
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
