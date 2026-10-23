"""Shared helpers for the four benchmark runners.

The runners differ in what they request but share one definition of throughput:
input, output and total tokens divided by the wall clock of the whole run.
"""

from __future__ import annotations

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
from typing import Any

from scripts.quality_metrics import (
    aggregate_speculative,
    speculative_metrics,
    text_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config/benchmarks"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
# launch.py publishes the service on this host port
DEFAULT_BASE_URL = "http://127.0.0.1:8005"
NATURE_CORPUS = (
    "A careful engineer reviews the design, measures the system, and records "
    "each result before changing the implementation. Reliable services balance "
    "latency, throughput, memory, and correctness across many independent "
    "requests. The experiment stores observations, compares controlled variants, "
    "and keeps the conclusion tied to the measured workload. "
)


def request(
    base: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: float = 3600,
):
    """Open an API request with the repository's optional benchmark auth."""
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("BENCH_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    return urllib.request.urlopen(
        urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers),
        timeout=timeout,
    )


def json_request(
    base: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: float = 3600,
) -> dict[str, Any]:
    """Read one JSON response through the same request path as streaming calls."""
    with request(base, path, body, timeout) as response:
        return json.load(response)


def percentile(values: list[float], fraction: float) -> float | None:
    """Return the nearest-rank percentile used by every benchmark runner."""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def load_config(name: str) -> dict[str, Any]:
    return json.loads((CONFIG_DIR / f"{name}.json").read_text())


def launch_args(config: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    """Launch arguments of a case or variant: its preset plus its own extras."""
    return list(config["launch"][entry["launch"]]) + list(entry.get("extra_launch", []))


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


def require_model(base: str) -> None:
    models = json_request(base, "/v1/models")
    if MODEL not in [row["id"] for row in models["data"]]:
        raise RuntimeError("served model identity is missing")


_EVIDENCE_KEYS = ("speculative", "speculative_metrics", "decode_metadata", "metrics")


def stream_metrics(
    response,
    started: float,
    chat: bool,
    expected_output: int | None = None,
    expected_input: int | None = None,
    input_tolerance: float = 0.0,
    clock=time.perf_counter,
) -> dict[str, Any]:
    """Parse one SSE response into per-request timing.

    ``chat`` selects the chat delta format; otherwise the completions text
    format is read. TTFT is measured to the first chunk that carries output.
    """
    first = last = None
    events: list[float] = []
    usage = None
    done = False
    finish = None
    chunks = 0
    output_parts: list[str] = []
    evidence: list[dict[str, Any]] = []
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
            evidence.append(row["usage"])
        for key in _EVIDENCE_KEYS:
            if isinstance(row.get(key), dict):
                evidence.append(row[key])
        for choice in row.get("choices", []):
            produced = False
            if chat:
                delta = choice.get("delta", {})
                if delta.get("content"):
                    output_parts.append(delta["content"])
                if delta.get("reasoning_content"):
                    output_parts.append("<reasoning>" + delta["reasoning_content"])
                if delta.get("tool_calls"):
                    output_parts.append(
                        "<tool_calls>"
                        + json.dumps(
                            delta["tool_calls"], sort_keys=True, separators=(",", ":")
                        )
                    )
                produced = bool(
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("tool_calls")
                )
            elif choice.get("text"):
                output_parts.append(choice["text"])
                produced = True
            if produced:
                now = clock()
                first = now if first is None else first
                last = now
                events.append(now)
                chunks += 1
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if not done or first is None or last is None or not usage or finish is None:
        raise RuntimeError("incomplete SSE response")
    output_tokens = int(usage["completion_tokens"])
    input_tokens = int(usage["prompt_tokens"])
    if expected_output is not None and output_tokens != expected_output:
        raise RuntimeError(
            f"server returned {output_tokens} output tokens; expected {expected_output}"
        )
    if expected_input is not None:
        delta = abs(input_tokens - expected_input) / max(1, expected_input)
        if delta > input_tolerance:
            raise RuntimeError(
                f"server returned {input_tokens} input tokens; expected "
                f"{expected_input} +/- {input_tolerance:.1%}"
            )
    span = last - first
    text = "".join(output_parts)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "ttft_s": first - started,
        "e2e_s": clock() - started,
        "decode_span_s": span,
        "prefill_tps": input_tokens / (first - started) if first > started else None,
        "decode_tps": (output_tokens - 1) / span if output_tokens > 1 and span > 0 else None,
        "tpot_s": span / (output_tokens - 1) if output_tokens > 1 and span > 0 else None,
        # SSE events may carry several tokens, so ITL is an inter-event
        # approximation while TPOT uses the server token count.
        "itl_s": [b - a for a, b in zip(events, events[1:], strict=False)],
        "chunks": chunks,
        "finish_reason": finish,
        "output_sha256": text_sha256(text),
        "output_chars": len(text),
        "speculative": speculative_metrics(*evidence),
    }


def stream_request(
    base: str,
    body: dict[str, Any],
    timeout: float,
    expected_output: int | None = None,
    expected_input: int | None = None,
    input_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Send one streaming completions or chat request; failures become rows."""
    chat = "messages" in body
    started = time.perf_counter()
    try:
        with request(
            base,
            "/v1/chat/completions" if chat else "/v1/completions",
            body,
            timeout,
        ) as response:
            row = stream_metrics(
                response,
                started,
                chat,
                expected_output,
                expected_input,
                input_tolerance,
            )
        return {"status": "PASS", **row}
    except Exception as error:
        return {
            "status": "FAIL",
            "error": str(error),
            "e2e_s": time.perf_counter() - started,
        }


def run_burst(
    base: str, bodies: list[dict[str, Any]], concurrency: int, timeout: float, **expect
) -> tuple[list[dict[str, Any]], float]:
    """Send all bodies with ``concurrency`` workers; return rows and wall clock."""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(
            pool.map(lambda body: stream_request(base, body, timeout, **expect), bodies)
        )
    for index, row in enumerate(rows):
        row["index"] = index
    return rows, time.perf_counter() - started


def completion_body(prompt, osl: int, seed: int) -> dict[str, Any]:
    return {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": osl,
        "temperature": 0,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def chat_body(content, osl: int, seed: int | None = None, thinking: bool = False):
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": osl,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": thinking},
    }
    if seed is not None:
        body["seed"] = seed
    return body


def token_pool(
    base: str, text: str = "A careful engineer measures latency and throughput."
) -> list[int]:
    """Token IDs that random prompts are drawn from: the tokens of ``text``."""
    result = json_request(base, "/tokenize", {"model": MODEL, "prompt": text})
    return [int(token) for token in result["tokens"] if 2 < token < 128000]


def nature_pool(base: str, target_tokens: int) -> list[int]:
    """Tokenize a deterministic corpus; requests carry token IDs so ISL is exact."""
    text = (NATURE_CORPUS * max(1, target_tokens // 32 + 8)).strip()
    tokens = json_request(base, "/tokenize", {"model": MODEL, "prompt": text})["tokens"]
    if len(tokens) < target_tokens:
        raise RuntimeError(
            f"natural corpus produced {len(tokens)} tokens; need {target_tokens}"
        )
    return [int(token) for token in tokens[:target_tokens]]


def build_prompts(
    base: str, dataset: str, isl: int, count: int, seed: int
) -> list[list[int]]:
    if dataset == "random":
        pool = token_pool(base)
        rng = random.Random(seed)
        return [[rng.choice(pool) for _ in range(isl)] for _ in range(count)]
    if dataset == "nature":
        source = nature_pool(base, isl)
        # Keep each request's first token distinct so prefix caching does not
        # turn a concurrent run into a cache-hit benchmark.
        return [
            source[0:1] + [((seed + i) % 127000) + 1000] + source[2:]
            for i in range(count)
        ]
    raise ValueError(f"unsupported dataset: {dataset}")


def repeated_text(word: str, repeats: int, label: str) -> str:
    """Prompt used by the single-request prefill probes."""
    return f"unique_{label} " + (f"{word} " * repeats) + "\nReturn one token."


def summarize(
    rows: list[dict[str, Any]], elapsed: float, **meta: Any
) -> dict[str, Any]:
    """Aggregate one run. All throughputs share the same wall-clock divisor."""
    good = [row for row in rows if row["status"] == "PASS"]
    total_input = sum(row.get("input_tokens", 0) for row in good)
    total_output = sum(row.get("output_tokens", 0) for row in good)
    itl = [v for row in good for v in row.get("itl_s", [])]
    windows = [row["decode_span_s"] for row in good if row.get("decode_span_s")]
    tpot = [r["tpot_s"] for r in good if r.get("tpot_s")]
    return {
        **meta,
        "status": "PASS" if rows and len(good) == len(rows) else "FAIL",
        "request_count": len(rows),
        "success": len(good),
        "failure_rate": 1 - len(good) / len(rows) if rows else 1.0,
        "elapsed_s": elapsed,
        "observed_input_tokens": total_input,
        "observed_output_tokens": total_output,
        "input_tps": total_input / elapsed if elapsed > 0 else None,
        "output_tps": total_output / elapsed if elapsed > 0 else None,
        "total_tps": (total_input + total_output) / elapsed if elapsed > 0 else None,
        "ttft_p50_s": percentile([r["ttft_s"] for r in good], 0.50),
        "ttft_p95_s": percentile([r["ttft_s"] for r in good], 0.95),
        "tpot_p50_s": percentile(tpot, 0.50),
        "tpot_p95_s": percentile(tpot, 0.95),
        "itl_p50_s": percentile(itl, 0.50),
        "itl_p95_s": percentile(itl, 0.95),
        "decode_window_p50_s": percentile(windows, 0.50),
        "speculative": aggregate_speculative(good),
        "timing_note": "ITL is SSE inter-event spacing; TPOT uses server completion token count.",
        "errors": [row["error"] for row in rows if row["status"] != "PASS"],
        "requests": rows,
    }


def verify_cache(rows: list[dict[str, Any]], cache: str) -> bool:
    """A cold run needs every request to report cached_tokens == 0; a warm run > 0.

    Missing usage details count as unverified rather than as the wanted state.
    """
    good = [row for row in rows if row["status"] == "PASS"]
    return bool(good) and all(
        row.get("cached_tokens") is not None
        and (row["cached_tokens"] == 0 if cache == "cold" else row["cached_tokens"] > 0)
        for row in good
    )


def workload_digest(prompts) -> str:
    return hashlib.sha256(
        json.dumps(prompts, separators=(",", ":")).encode()
    ).hexdigest()


def write_report(path: Path, report: dict[str, Any], hide=("requests",)) -> int:
    if path.exists():
        raise SystemExit(f"output exists; use a fresh evidence path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in hide}, indent=2))
    return 0 if report.get("status", "PASS") == "PASS" else 1


def select_cases(config: dict[str, Any], case_id: str | None) -> list[dict[str, Any]]:
    cases = config["cases"]
    if case_id is None:
        return cases
    chosen = [case for case in cases if case["id"] == case_id]
    if not chosen:
        raise SystemExit(
            f"unknown case {case_id!r}; choose from {[c['id'] for c in cases]}"
        )
    return chosen
