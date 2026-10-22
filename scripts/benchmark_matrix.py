"""Measure single-stream and concurrent prefill/decode matrices.

The runner keeps the request shape explicit and reports both service-level
aggregate throughput and per-request timing. ``nature`` uses token IDs derived
from a deterministic natural-language corpus; this keeps ISL exact while
avoiding a dependency on a local tokenizer installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from scripts.benchmark_common import json_request, percentile, request
from scripts.quality_metrics import (
    aggregate_speculative,
    speculative_metrics,
    text_sha256,
)

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
NATURE_CORPUS = (
    "A careful engineer reviews the design, measures the system, and records "
    "each result before changing the implementation. Reliable services balance "
    "latency, throughput, memory, and correctness across many independent "
    "requests. The experiment stores observations, compares controlled variants, "
    "and keeps the conclusion tied to the measured workload. "
)


def sse_metrics(response, started: float) -> dict[str, Any]:
    first = last = None
    event_times: list[float] = []
    usage = None
    done = False
    finish_reason = None
    chunks = 0
    output_parts: list[str] = []
    speculative_payloads: list[dict[str, Any]] = []
    for line in response:
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            done = True
            break
        row = json.loads(payload)
        if row.get("error"):
            raise RuntimeError(str(row["error"]))
        if row.get("usage"):
            usage = row["usage"]
            speculative_payloads.append(row["usage"])
        for key in (
            "speculative",
            "speculative_metrics",
            "decode_metadata",
            "metrics",
        ):
            if isinstance(row.get(key), dict):
                speculative_payloads.append(row[key])
        for choice in row.get("choices", []):
            if choice.get("text"):
                output_parts.append(choice["text"])
                now = time.perf_counter()
                first = now if first is None else first
                last = now
                event_times.append(now)
                chunks += 1
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    if not done or first is None or last is None or not usage or finish_reason is None:
        raise RuntimeError("incomplete SSE response")
    output_tokens = int(usage["completion_tokens"])
    input_tokens = int(usage["prompt_tokens"])
    decode_span = last - first
    intervals = [b - a for a, b in zip(event_times, event_times[1:], strict=False)]
    output_text = "".join(output_parts)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_s": first - started,
        "e2e_s": time.perf_counter() - started,
        "decode_span_s": decode_span,
        "prefill_tps": input_tokens / (first - started)
        if first > started
        else None,
        "decode_tps": (output_tokens - 1) / decode_span
        if output_tokens > 1 and decode_span > 0
        else None,
        "tpot_s": decode_span / (output_tokens - 1)
        if output_tokens > 1 and decode_span > 0
        else None,
        # SSE events may contain multiple tokens. This is an inter-event
        # approximation, while TPOT uses the server token count.
        "itl_s": intervals,
        "chunks": chunks,
        "finish_reason": finish_reason,
        "output_sha256": text_sha256(output_text),
        "output_chars": len(output_text),
        "speculative": speculative_metrics(*speculative_payloads),
    }


def run_one(
    base: str,
    prompt: str | list[int],
    output_tokens: int,
    index: int,
    timeout: float,
    seed: int = 1017,
) -> dict[str, Any]:
    started = time.perf_counter()
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        with request(base, "/v1/completions", body, timeout) as response:
            metrics = sse_metrics(response, started)
        if metrics["output_tokens"] != output_tokens:
            raise RuntimeError(
                f"server returned {metrics['output_tokens']} output tokens; "
                f"expected {output_tokens}"
            )
        return {"index": index, "status": "PASS", **metrics}
    except Exception as error:
        return {
            "index": index,
            "status": "FAIL",
            "error": str(error),
            "e2e_s": time.perf_counter() - started,
        }


def token_pool(base: str) -> list[int]:
    result = json_request(
        base,
        "/tokenize",
        {
            "model": MODEL,
            "prompt": "A careful engineer measures latency and throughput.",
        },
    )
    return [int(token) for token in result["tokens"] if 2 < token < 128000]


def nature_pool(base: str, target_tokens: int) -> list[int]:
    # Tokenize a deterministic natural-language corpus and repeat it until the
    # requested target is available. The request itself uses token IDs so the
    # measured ISL is exact and independent of local tokenizer packages.
    text = (NATURE_CORPUS * max(1, target_tokens // 32 + 8)).strip()
    tokens = json_request(base, "/tokenize", {"model": MODEL, "prompt": text})[
        "tokens"
    ]
    if len(tokens) < target_tokens:
        raise RuntimeError(
            f"natural corpus produced {len(tokens)} tokens; need {target_tokens}"
        )
    return [int(token) for token in tokens[:target_tokens]]


def build_prompts(
    base: str, dataset: str, isl: int, count: int, seed: int
) -> list[str | list[int]]:
    if dataset == "random":
        pool = token_pool(base)
        rng = random.Random(seed)
        return [[rng.choice(pool) for _ in range(isl)] for _ in range(count)]
    if dataset == "nature":
        source = nature_pool(base, isl)
        # Keep each request's first token distinct so prefix caching does not
        # turn a concurrent nature run into a cache-hit benchmark.
        return [source[0:1] + [((seed + i) % 127000) + 1000] + source[2:] for i in range(count)]
    raise ValueError(f"unsupported dataset: {dataset}")


def summarize(
    rows: list[dict[str, Any]],
    elapsed: float,
    dataset: str,
    isl: int,
    osl: int,
    concurrency: int,
    seed: int,
) -> dict[str, Any]:
    good = [row for row in rows if row["status"] == "PASS"]
    total_input = sum(row.get("input_tokens", 0) for row in good)
    total_output = sum(row.get("output_tokens", 0) for row in good)
    decode_windows = [row["decode_span_s"] for row in good if row.get("decode_span_s")]
    itl = [interval for row in good for interval in row.get("itl_s", [])]
    return {
        "status": "PASS" if len(good) == len(rows) else "FAIL",
        "dataset": dataset,
        "isl_target": isl,
        "osl": osl,
        "concurrency": concurrency,
        "request_count": len(rows),
        "success": len(good),
        "failure_rate": 1 - len(good) / len(rows),
        "seed": seed,
        "elapsed_s": elapsed,
        "observed_input_tokens": total_input,
        "observed_output_tokens": total_output,
        "prefill_tps": total_input / elapsed if elapsed > 0 else None,
        "decode_tps": total_output / elapsed if elapsed > 0 else None,
        "ttft_p50_s": percentile([r["ttft_s"] for r in good], 0.50),
        "ttft_p95_s": percentile([r["ttft_s"] for r in good], 0.95),
        "tpot_p50_s": percentile([r["tpot_s"] for r in good], 0.50),
        "tpot_p95_s": percentile([r["tpot_s"] for r in good], 0.95),
        "itl_p50_s": percentile(itl, 0.50),
        "itl_p95_s": percentile(itl, 0.95),
        "decode_window_p50_s": percentile(decode_windows, 0.50),
        "speculative": aggregate_speculative(good),
        "timing_note": "ITL is SSE inter-event spacing; TPOT uses server completion token count.",
        "errors": [row["error"] for row in rows if row["status"] != "PASS"],
        "requests": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", choices=("random", "nature"), required=True)
    parser.add_argument("--isl", type=int, required=True)
    parser.add_argument("--osl", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--seed", type=int, default=1017)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = args.requests or args.concurrency
    if count < args.concurrency or args.isl <= 0 or args.osl < 2:
        parser.error("requests >= concurrency, isl > 0 and osl >= 2 are required")
    if args.output.exists():
        parser.error("output exists; use a fresh evidence path")
    models = json_request(args.base_url, "/v1/models")
    if MODEL not in [row["id"] for row in models["data"]]:
        raise RuntimeError("served model identity is missing")
    prompts = build_prompts(args.base_url, args.dataset, args.isl, count, args.seed)
    # Warm one request for the current Graph/shape. It is excluded from rows.
    warm_prompt = build_prompts(
        args.base_url, args.dataset, args.isl, 1, args.seed + 1000000
    )[0]
    run_one(
        args.base_url, warm_prompt, min(args.osl, 2), -1, args.timeout, args.seed
    )
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        rows = list(
            executor.map(
                lambda pair: run_one(
                    args.base_url,
                    pair[1],
                    args.osl,
                    pair[0],
                    args.timeout,
                    args.seed,
                ),
                enumerate(prompts),
            )
        )
    report = summarize(
        rows,
        time.perf_counter() - started,
        args.dataset,
        args.isl,
        args.osl,
        args.concurrency,
        args.seed,
    )
    report["workload_sha256"] = hashlib.sha256(
        json.dumps(prompts, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "requests"}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
