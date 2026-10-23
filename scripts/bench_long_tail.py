"""Long-tail scenarios from ``config/benchmarks/long_tail.json``: tool history,
mixed image/text, long soak, and mixed prefill/decode in one service."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from scripts.bench_common import (
    DEFAULT_BASE_URL,
    MODEL,
    ROOT,
    completion_body,
    json_request,
    launch_args,
    load_config,
    percentile,
    request,
    require_model,
    select_cases,
    stream_metrics,
    token_pool,
    write_report,
)
from scripts.quality_metrics import aggregate_speculative

VISION_TEXT = (
    "An engineer compares latency, throughput, memory, and correctness while "
    "reviewing a multimodal inference result. The report keeps every observation "
    "tied to the same request shape and model configuration. "
)


def _image(number: int) -> dict[str, Any]:
    data = (ROOT / f"tests/fixtures/video_frames/{number:02}.png").read_bytes()
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(data).decode()
        },
    }


def _chat_stream(
    base: str,
    body: dict[str, Any],
    expected_output: int,
    timeout: float,
    expected_input: int | None = None,
    input_tolerance: float = 0.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    with request(base, "/v1/chat/completions", body, timeout) as response:
        return stream_metrics(
            response,
            started,
            True,
            expected_output,
            expected_input,
            input_tolerance,
        )


def _tool_messages(rounds: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in range(rounds):
        key = f"record_{index:03d}"
        messages.extend(
            [
                {"role": "user", "content": f"Look up {key} and keep the result."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{index:03d}",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": json.dumps({"key": key}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index:03d}",
                    "content": f"{index * 17 + 11}",
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": "Summarize all records as JSON with fields count and total.",
        }
    )
    return messages


def _tool_body(
    messages: list[dict[str, Any]], effort: str, osl: int, seed: int
) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "seed": seed,
        "ignore_eos": True,
        "max_tokens": osl,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": effort},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a record by key.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            }
        ],
    }


def _vision_body(
    base: str, image_count: int, target_tokens: int, osl: int, seed: int
) -> dict[str, Any]:
    text_target = max(64, target_tokens - 600 * image_count)
    text_source = VISION_TEXT * max(2, text_target // 32 + 8)
    text_tokens = json_request(
        base, "/tokenize", {"model": MODEL, "prompt": text_source}
    )["tokens"][:text_target]
    text = json_request(
        base, "/detokenize", {"model": MODEL, "tokens": text_tokens}
    )["prompt"]
    content: list[Any] = [
        {
            "type": "text",
            "text": text
            + " Read every supplied image and return JSON with the ordered numbers. "
            + "Also describe one visual detail for each image.",
        }
    ]
    content.extend(_image(number) for number in ([1, 3, 2, 1][:image_count]))
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": seed,
        "ignore_eos": True,
        "max_tokens": osl,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _completion_one(
    base: str,
    prompt: list[int],
    osl: int,
    timeout: float,
    expected_input: int | None = None,
    input_tolerance: float = 0.0,
    seed: int = 1017,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with request(
            base, "/v1/completions", completion_body(prompt, osl, seed), timeout
        ) as response:
            metrics = stream_metrics(
                response, started, False, osl, expected_input, input_tolerance
            )
        return {"status": "PASS", **metrics}
    except Exception as error:
        return {"status": "FAIL", "error": str(error), "e2e_s": time.perf_counter() - started}


def _summary(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    good = [row for row in rows if row["status"] == "PASS"]
    all_itl = [v for row in good for v in row.get("itl_s", [])]
    tpot = [r["tpot_s"] for r in good if r.get("tpot_s")]
    total_in = sum(row.get("input_tokens", 0) for row in good)
    total_out = sum(row.get("output_tokens", 0) for row in good)
    return {
        "status": "PASS" if rows and len(good) == len(rows) else "FAIL",
        "requests": len(rows),
        "success": len(good),
        "failure_rate": 1 - len(good) / len(rows) if rows else 1.0,
        "elapsed_s": elapsed,
        "observed_input_tokens": total_in,
        "observed_output_tokens": total_out,
        "input_tps": total_in / elapsed if elapsed else None,
        "output_tps": total_out / elapsed if elapsed else None,
        "total_tps": (total_in + total_out) / elapsed if elapsed else None,
        "ttft_p50_s": percentile([r["ttft_s"] for r in good], 0.5),
        "ttft_p95_s": percentile([r["ttft_s"] for r in good], 0.95),
        "tpot_p50_s": percentile(tpot, 0.5),
        "tpot_p95_s": percentile(tpot, 0.95),
        "itl_p50_s": percentile(all_itl, 0.5),
        "itl_p95_s": percentile(all_itl, 0.95),
        "output_sha256": sorted({r["output_sha256"] for r in good if r.get("output_sha256")}),
        "speculative": aggregate_speculative(good),
        "rows": rows,
    }


def _run_tool(base: str, case: dict[str, Any], timeout: float) -> dict[str, Any]:
    messages = _tool_messages(case["tool_rounds"])
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=case["concurrency"]) as pool:
        futures = [
            pool.submit(
                _chat_stream,
                base,
                _tool_body(
                    copy.deepcopy(messages),
                    case["reasoning_effort"],
                    case["osl"],
                    case.get("seed", 1017),
                ),
                case["osl"],
                timeout,
                case["isl"],
                case.get("input_tolerance", 0.08),
                case.get("seed", 1017),
            )
            for _ in range(case["concurrency"])
        ]
        for future in futures:
            try:
                rows.append({"status": "PASS", **future.result()})
            except Exception as error:
                rows.append({"status": "FAIL", "error": str(error)})
    result = _summary(rows, time.perf_counter() - started)
    result.update({"case": case["id"], "isl_target": case["isl"], "osl": case["osl"]})
    return result


def _run_vision(base: str, case: dict[str, Any], timeout: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=case["concurrency"]) as pool:
        futures = [
            pool.submit(
                _chat_stream,
                base,
                _vision_body(
                    base,
                    count,
                    case["isl"],
                    case["osl"],
                    case.get("seed", 1017),
                ),
                case["osl"],
                timeout,
                case["isl"],
                case.get("input_tolerance", 0.02),
                case.get("seed", 1017),
            )
            for count in (case["image_counts"] * case["concurrency"])
        ][: case["concurrency"]]
        for future in futures:
            try:
                rows.append({"status": "PASS", **future.result()})
            except Exception as error:
                rows.append({"status": "FAIL", "error": str(error)})
    result = _summary(rows, time.perf_counter() - started)
    result.update({"case": case["id"], "isl_target": case["isl"], "osl": case["osl"]})
    return result


def _random_prompts(base: str, isl: int, count: int, seed: int) -> list[list[int]]:
    pool = token_pool(base)
    rng = random.Random(seed)
    return [[rng.choice(pool) for _ in range(isl)] for _ in range(count)]


def _random_prompt(base: str, isl: int, seed: int) -> list[int]:
    pool = token_pool(base)
    rng = random.Random(seed)
    return [rng.choice(pool) for _ in range(isl)]


def _run_soak(base: str, case: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + case["duration_s"]
    rows: list[dict[str, Any]] = []
    prompt_index = 0
    with ThreadPoolExecutor(max_workers=case["concurrency"]) as pool:
        pending: set[Future] = {
            pool.submit(
                _completion_one,
                base,
                _random_prompt(base, case["isl"], 20261021 + prompt_index + index),
                case["osl"],
                timeout,
                case["isl"],
                case.get("input_tolerance", 0.0),
                case.get("seed", 1017),
            )
            for index in range(case["concurrency"])
        }
        prompt_index += case["concurrency"]
        while pending and time.perf_counter() < deadline:
            done, pending = wait(pending, timeout=1)
            for future in done:
                try:
                    rows.append(future.result())
                except Exception as error:
                    rows.append({"status": "FAIL", "error": str(error)})
                if time.perf_counter() < deadline:
                    prompt = _random_prompt(base, case["isl"], 20261021 + prompt_index)
                    prompt_index += 1
                    pending.add(
                        pool.submit(
                            _completion_one,
                            base,
                            prompt,
                            case["osl"],
                            timeout,
                            case["isl"],
                            case.get("input_tolerance", 0.0),
                            case.get("seed", 1017),
                        )
                    )
        done, _ = wait(pending, timeout=timeout)
        for future in done:
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({"status": "FAIL", "error": str(error)})
    result = _summary(rows, max(0.001, time.perf_counter() - started))
    result.update({"case": case["id"], "isl_target": case["isl"], "osl": case["osl"], "duration_s": case["duration_s"]})
    return result


def _run_mixed(base: str, case: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + case["duration_s"]
    decode_rows: list[dict[str, Any]] = []
    prefill_rows: list[dict[str, Any]] = []
    prefill_index = 0
    decode_index = 0
    next_prefill = time.perf_counter()
    prefill_pending: set[Future] = set()
    prefill_injected = 0
    prefill_pending_max = 0
    max_prefill_inflight = int(case.get("prefill_max_inflight", 2))
    with ThreadPoolExecutor(max_workers=case["decode_concurrency"] + 2) as pool:
        pending: set[Future] = {
            pool.submit(
                _completion_one,
                base,
                _random_prompt(base, case["decode_isl"], 20261021 + decode_index + index),
                case["decode_osl"],
                timeout,
                case["decode_isl"],
                case.get("input_tolerance", 0.0),
                case.get("seed", 1017),
            )
            for index in range(case["decode_concurrency"])
        }
        decode_index += case["decode_concurrency"]
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now >= next_prefill and len(prefill_pending) < max_prefill_inflight:
                prefill_prompt = _random_prompt(
                    base, case["prefill_isl"], 20261022 + prefill_index
                )
                prefill_index += 1
                prefill_pending.add(
                    pool.submit(
                        _completion_one,
                        base,
                        prefill_prompt,
                        case["prefill_osl"],
                        timeout,
                        case["prefill_isl"],
                        case.get("input_tolerance", 0.0),
                        case.get("seed", 1017),
                    )
                )
                prefill_injected += 1
                prefill_pending_max = max(prefill_pending_max, len(prefill_pending))
                next_prefill = now + case["prefill_interval_s"]
            done, pending = wait(pending, timeout=0.5)
            for future in done:
                try:
                    decode_rows.append(future.result())
                except Exception as error:
                    decode_rows.append({"status": "FAIL", "error": str(error)})
                if time.perf_counter() < deadline:
                    prompt = _random_prompt(
                        base, case["decode_isl"], 20261021 + decode_index
                    )
                    decode_index += 1
                    pending.add(
                        pool.submit(
                            _completion_one,
                            base,
                            prompt,
                            case["decode_osl"],
                            timeout,
                            case["decode_isl"],
                            case.get("input_tolerance", 0.0),
                            case.get("seed", 1017),
                        )
                    )
            done_prefill = {future for future in prefill_pending if future.done()}
            for future in done_prefill:
                try:
                    prefill_rows.append(future.result())
                except Exception as error:
                    prefill_rows.append({"status": "FAIL", "error": str(error)})
            prefill_pending -= done_prefill
        done, _ = wait(pending, timeout=timeout)
        for future in done:
            try:
                decode_rows.append(future.result())
            except Exception as error:
                decode_rows.append({"status": "FAIL", "error": str(error)})
        done_prefill, _ = wait(prefill_pending, timeout=timeout)
        for future in done_prefill:
            try:
                prefill_rows.append(future.result())
            except Exception as error:
                prefill_rows.append({"status": "FAIL", "error": str(error)})
    actual_elapsed = max(0.001, time.perf_counter() - started)
    prefill_completed = sum(row.get("status") == "PASS" for row in prefill_rows)
    decode_completed = sum(row.get("status") == "PASS" for row in decode_rows)
    decode_report = _summary(decode_rows, actual_elapsed)
    prefill_report = _summary(prefill_rows, actual_elapsed)
    status = "PASS" if decode_report["status"] == prefill_report["status"] == "PASS" else "FAIL"
    return {
        "status": status,
        "case": case["id"],
        "duration_s": case["duration_s"],
        "measurement_elapsed_s": case["duration_s"],
        "completion_elapsed_s": actual_elapsed,
        "drain_elapsed_s": max(0.0, actual_elapsed - case["duration_s"]),
        "decode_injected": decode_index,
        "decode_completed": decode_completed,
        "prefill_injected": prefill_injected,
        "prefill_completed": prefill_completed,
        "prefill_pending_max": prefill_pending_max,
        "decode": decode_report,
        "prefill": prefill_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--case", help="case id; omit to list the cases")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    config = load_config("long_tail")
    if args.case is None:
        for case in config["cases"]:
            print(case["id"], case["kind"], "launch:", " ".join(launch_args(config, case)))
        return 0
    if args.output is None:
        parser.error("--output is required with --case")
    (case,) = select_cases(config, args.case)
    require_model(args.base_url)
    runner = {
        "tool_history": _run_tool,
        "vision_mix": _run_vision,
        "soak": _run_soak,
        "mixed": _run_mixed,
    }.get(case["kind"])
    if runner is None:
        raise ValueError(f"unsupported case kind: {case['kind']}")
    report = runner(args.base_url, case, args.timeout)
    return write_report(args.output, report, hide=("rows", "decode", "prefill"))


if __name__ == "__main__":
    raise SystemExit(main())
