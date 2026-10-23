"""Excluded candidates defined in ``config/benchmarks/excluded.json``.

Every case has a ``kind`` that says how it was measured:

``cyclic_prefill``
    one non-streaming 260K cold prefill. The prompt is ``offset + (i % cycle)``;
    each candidate has its own offset and seed so no run can hit another's
    prefix cache.
``request_smoke``
    two sequential non-streaming requests with the same random prompt, sent
    after startup. Used for the KV transfer connectors.
``matrix_digest``
    a ``bench_regular`` matrix cell compared with the dense path by output digest.
``startup_rejected``
    a change the server refuses at startup; nothing to send.

The last two are listed by this script and run as their ``run`` field says.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from pathlib import Path
from typing import Any

from scripts.bench_common import (
    DEFAULT_BASE_URL,
    MODEL,
    json_request,
    launch_args,
    load_config,
    require_model,
    select_cases,
    token_pool,
    write_report,
)


def cyclic_prompt(isl: int, offset: int, cycle: int) -> list[int]:
    return [offset + (i % cycle) for i in range(isl)]


def probe_body(probe: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "prompt": cyclic_prompt(probe["isl"], case["offset"], probe["cycle"]),
        "max_tokens": probe["osl"],
        "temperature": 0,
        "seed": case["seed"],
        "ignore_eos": True,
        "stream": probe["stream"],
    }


def run_cyclic_prefill(base: str, config: dict[str, Any], case: dict[str, Any], timeout: float):
    probe = config["prefill_probe"]
    started = time.perf_counter()
    result = json_request(base, "/v1/completions", probe_body(probe, case), timeout)
    elapsed = time.perf_counter() - started
    usage = result["usage"]
    return {
        "status": "PASS" if usage["prompt_tokens"] == probe["isl"] else "FAIL",
        "case": case["id"],
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "prefill_seconds": elapsed,
        "input_tps": usage["prompt_tokens"] / elapsed,
    }


def smoke_body(probe: dict[str, Any], prompt: list[int]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": probe["osl"],
        "temperature": 0,
        "seed": probe["seed"],
        "ignore_eos": True,
        "stream": probe["stream"],
    }


def run_request_smoke(base: str, config: dict[str, Any], case: dict[str, Any], timeout: float):
    """Send the probe requests one after another; a failed request is recorded."""
    probe = config["smoke_probe"]
    pool = token_pool(base, probe["pool_text"])
    rng = random.Random(probe["seed"])
    body = smoke_body(probe, [rng.choice(pool) for _ in range(probe["isl"])])
    rows = []
    for index in range(probe["requests"]):
        started = time.perf_counter()
        try:
            result = json_request(base, "/v1/completions", body, timeout)
        except Exception as error:
            rows.append({"index": index, "status": "FAIL", "error": str(error)})
            break
        choice = result["choices"][0]
        usage = result["usage"]
        rows.append(
            {
                "index": index,
                "status": "PASS",
                "elapsed_s": time.perf_counter() - started,
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "finish_reason": choice["finish_reason"],
                "output_sha256": hashlib.sha256(choice.get("text", "").encode()).hexdigest(),
            }
        )
    complete = len(rows) == probe["requests"] and all(r["status"] == "PASS" for r in rows)
    return {"status": "PASS" if complete else "FAIL", "case": case["id"], "rows": rows}


RUNNERS = {"cyclic_prefill": run_cyclic_prefill, "request_smoke": run_request_smoke}


def case_plan(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    return {
        "launch_args": launch_args(config, case),
        "env": dict(case.get("env", {})),
        "image_variant": case.get("image_variant"),
    }


def describe(config: dict[str, Any], case: dict[str, Any]) -> str:
    plan = case_plan(config, case)
    text = f"{case['id']} ({case['kind']}) launch: " + " ".join(plan["launch_args"])
    if plan["env"]:
        text += "  env: " + " ".join(f"{k}={v}" for k, v in plan["env"].items())
    if plan["image_variant"]:
        text += f"  ({plan['image_variant']})"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--case", help="case id; omit to list the cases")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    config = load_config("excluded")
    if args.case is None:
        for case in config["cases"]:
            print(describe(config, case))
        return 0
    if args.output is None:
        parser.error("--output is required with --case")
    (case,) = select_cases(config, args.case)
    runner = RUNNERS.get(case["kind"])
    if runner is None:
        parser.error(f"{case['id']} is not sent by this script: {case['run']}")
    require_model(args.base_url)
    report = runner(args.base_url, config, case, args.timeout)
    report["plan"] = case_plan(config, case)
    return write_report(args.output, report, hide=())


if __name__ == "__main__":
    raise SystemExit(main())
