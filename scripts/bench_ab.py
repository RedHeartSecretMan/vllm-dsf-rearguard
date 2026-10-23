"""Configuration A/B experiments defined in ``config/benchmarks/ab.json``.

Every case changes one setting between variants. ``--plan`` prints how to start
each variant; ``--case``/``--variant`` measure one already running variant::

    python3 -m scripts.bench_ab --plan
    python3 -m scripts.bench_ab --case bounded_workspace --variant on --output out.json
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from scripts.bench_common import (
    DEFAULT_BASE_URL,
    chat_body,
    completion_body,
    launch_args,
    load_config,
    repeated_text,
    require_model,
    run_burst,
    select_cases,
    summarize,
    token_pool,
    verify_cache,
    workload_digest,
    write_report,
)


def variant_plan(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """Translate one variant into launch arguments plus the edits ``launch.py``
    has no option for: container environment overrides and dropped arguments."""
    return {
        "launch_args": launch_args(config, variant),
        "env": dict(variant.get("env", {})),
        "drop_args": list(variant.get("drop_args", [])),
    }


def plan_lines(config: dict[str, Any]) -> list[str]:
    lines = []
    for case in config["cases"]:
        lines.append(f"{case['id']}  (control: {case['control']})")
        for variant in case["variants"]:
            plan = variant_plan(config, variant)
            text = " ".join(plan["launch_args"])
            if plan["env"]:
                text += "  env: " + " ".join(f"{k}={v}" for k, v in plan["env"].items())
            if plan["drop_args"]:
                text += "  drop: " + " ".join(plan["drop_args"])
            lines.append(f"  {variant['id']}: {text}")
    return lines


def _burst_reports(base, workload, variant, timeout):
    reports = []
    kind = workload["kind"]
    seed = variant.get("seed", workload.get("seed", 1017))
    for concurrency in workload["concurrency"]:
        if kind == "random_burst":
            pool = token_pool(base)
            rng = random.Random(seed)
            prompts = [
                [rng.choice(pool) for _ in range(workload["isl"])]
                for _ in range(concurrency)
            ]
            bodies = [completion_body(p, workload["osl"], seed) for p in prompts]
        elif kind == "token_burst":
            # Identical short prompts: prefill is negligible, decode dominates.
            prompts = [list(range(1000, 1000 + workload["prompt_tokens"]))] * concurrency
            bodies = [completion_body(p, workload["osl"], seed) for p in prompts]
        elif kind == "chat_burst":
            prompts = [workload["prompt"]] * concurrency
            bodies = [chat_body(workload["prompt"], workload["osl"]) for _ in prompts]
        else:
            raise ValueError(f"unsupported burst kind: {kind}")
        rows, elapsed = run_burst(
            base, bodies, concurrency, timeout, expected_output=workload["osl"]
        )
        report = summarize(
            rows, elapsed, kind=kind, concurrency=concurrency, osl=workload["osl"], seed=seed
        )
        report["workload_sha256"] = workload_digest(prompts)
        if "cache" in workload:
            report["cache_requested"] = workload["cache"]
            report["cache_verified"] = verify_cache(rows, workload["cache"])
            if not report["cache_verified"]:
                report["status"] = "FAIL"
        reports.append(report)
    return reports


def _prefill_reports(base, workload, timeout):
    reports = []
    if workload["kind"] == "chat_prefill":
        text = repeated_text(workload["word"], workload["repeats"], workload["label"])
        rows, elapsed = run_burst(
            base, [chat_body(text, workload["osl"])], 1, timeout, expected_output=1
        )
        reports.append(summarize(rows, elapsed, kind="chat_prefill", repeats=workload["repeats"]))
    elif workload["kind"] == "random_prefill":
        pool = token_pool(base)
        for isl, seed in zip(workload["isl"], workload["seed"], strict=True):
            rng = random.Random(seed)
            prompt = [rng.choice(pool) for _ in range(isl)]
            rows, elapsed = run_burst(
                base,
                [completion_body(prompt, workload["osl"], seed)],
                1,
                timeout,
                expected_output=workload["osl"],
            )
            reports.append(
                summarize(rows, elapsed, kind="random_prefill", isl_target=isl, seed=seed)
            )
    else:
        raise ValueError(f"unsupported prefill kind: {workload['kind']}")
    return reports


def measure(base: str, case: dict[str, Any], variant: dict[str, Any], timeout: float):
    """Run the workloads this variant takes part in against the live service."""
    workloads = case["workload"]
    names = variant.get("workloads") or list(workloads)
    reports: list[dict[str, Any]] = []
    for name in names:
        workload = workloads[name]
        if "long_tail_case" in workload:
            raise SystemExit(
                f"{case['id']} runs the long-tail case {workload['long_tail_case']!r}: "
                f"python3 -m scripts.bench_long_tail --case {workload['long_tail_case']}"
            )
        if workload["kind"] in {"random_burst", "token_burst", "chat_burst"}:
            reports += _burst_reports(base, workload, variant, timeout)
        else:
            reports += _prefill_reports(base, workload, timeout)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--plan", action="store_true", help="print how to start each variant")
    parser.add_argument("--case")
    parser.add_argument("--variant")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    config = load_config("ab")
    if args.plan or args.case is None:
        print("\n".join(plan_lines(config)))
        return 0
    if args.variant is None or args.output is None:
        parser.error("--variant and --output are required with --case")
    (case,) = select_cases(config, args.case)
    variant = next((v for v in case["variants"] if v["id"] == args.variant), None)
    if variant is None:
        parser.error(f"unknown variant {args.variant!r}; choose from {[v['id'] for v in case['variants']]}")
    require_model(args.base_url)
    reports = measure(args.base_url, case, variant, args.timeout)
    status = "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL"
    return write_report(
        args.output,
        {
            "status": status,
            "case": case["id"],
            "variant": variant["id"],
            "control": case["control"],
            "plan": variant_plan(config, variant),
            "reports": reports,
        },
        hide=("reports",),
    )


if __name__ == "__main__":
    raise SystemExit(main())
