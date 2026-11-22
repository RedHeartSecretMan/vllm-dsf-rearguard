"""Regular scenarios: short-request concurrency, single-request prefill/decode,
and the ISL/OSL matrix, as defined in ``config/benchmarks/regular.json``.

Run one case against a service that was started with that case's ``launch``
arguments, for example::

    python3 -m scripts.benchmarks.bench_regular --case matrix_concurrent --output out.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.bench_common import (
    DEFAULT_PERFORMANCE_BASE_URL,
    build_prompts,
    chat_body,
    completion_body,
    launch_args,
    load_config,
    repeated_text,
    require_model,
    run_burst,
    select_cases,
    stream_request,
    summarize,
    workload_digest,
    write_report,
)


def _require_cold(report: dict[str, Any]) -> dict[str, Any]:
    """Reject prefix-cache hits in formal regular benchmark cells."""
    good = [row for row in report["requests"] if row["status"] == "PASS"]
    report["cache_requested"] = "cold"
    report["cache_verified"] = bool(good) and all(
        row.get("cached_tokens") == 0 for row in good
    )
    if not report["cache_verified"]:
        report["status"] = "FAIL"
    return report


def cell_seed(case: dict[str, Any], concurrency: int) -> int:
    """Seed for one concurrency cell.

    With ``seed_per_concurrency`` every cell draws its own prompts. Cells that
    share a seed also share their first prompts, so a later cell would hit the
    prefix cache populated by an earlier one.
    """
    if case.get("seed_per_concurrency"):
        return case["seed"] + concurrency
    return case["seed"]


def run_matrix(base: str, case: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    """One report per (dataset, isl, concurrency) cell."""
    reports = []
    timeout = case.get("timeout", timeout)
    for dataset in case["datasets"]:
        for isl in case["isl"]:
            for concurrency in case["concurrency"]:
                seed = cell_seed(case, concurrency)
                prompts = build_prompts(base, dataset, isl, concurrency, seed)
                # Warm one request for the current graph/shape; it is not reported.
                warm = build_prompts(base, dataset, isl, 1, seed + 1000000)[0]
                warm_body = completion_body(warm, min(case["osl"], 2), seed)
                warm_body["cache_salt"] = f"regular:{case['id']}:{dataset}:{isl}:{concurrency}:warm"
                stream_request(base, warm_body, timeout)
                bodies = []
                for index, prompt in enumerate(prompts):
                    body = completion_body(prompt, case["osl"], seed)
                    body["cache_salt"] = (
                        f"regular:{case['id']}:{dataset}:{isl}:{concurrency}:{index}"
                    )
                    bodies.append(body)
                rows, elapsed = run_burst(
                    base, bodies, concurrency, timeout, expected_output=case["osl"]
                )
                report = summarize(
                    rows,
                    elapsed,
                    case=case["id"],
                    dataset=dataset,
                    isl_target=isl,
                    osl=case["osl"],
                    concurrency=concurrency,
                    seed=seed,
                )
                report["workload_sha256"] = workload_digest(prompts)
                reports.append(_require_cold(report))
    return reports


def run_chat_burst(base: str, case: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    """Identical chat requests sent together; ``{n}`` in the prompt is the concurrency."""
    reports = []
    for run in case["runs"]:
        concurrency = run["concurrency"]
        prompt = case["prompt"].format(n=concurrency)
        bodies = []
        for index in range(concurrency):
            body = chat_body(prompt, case["osl"], thinking=case["thinking"])
            body["cache_salt"] = f"regular:{case['id']}:c{concurrency}:{index}"
            bodies.append(body)
        rows, elapsed = run_burst(
            base, bodies, concurrency, timeout, expected_output=case["osl"]
        )
        reports.append(
            _require_cold(
                summarize(
                    rows,
                    elapsed,
                    case=case["id"],
                    concurrency=concurrency,
                    osl=case["osl"],
                    max_seqs=run.get("max_seqs"),
                )
            )
        )
    return reports


def run_chat_prefill(base: str, case: dict[str, Any], timeout: float) -> list[dict[str, Any]]:
    """One request per size; ``max_tokens=1`` makes TTFT the prefill time."""
    reports = []
    for repeats in case["repeats"]:
        text = repeated_text(case["word"], repeats, f"prefill_{repeats // 1000}k")
        body = chat_body(text, case["osl"], thinking=case["thinking"])
        body["cache_salt"] = f"regular:{case['id']}:repeats{repeats}"
        rows, elapsed = run_burst(
            base,
            [body],
            1,
            timeout,
            expected_output=case["osl"],
        )
        reports.append(
            _require_cold(
                summarize(rows, elapsed, case=case["id"], repeats=repeats, osl=case["osl"])
            )
        )
    return reports


RUNNERS = {
    "matrix": run_matrix,
    "chat_burst": run_chat_burst,
    "chat_prefill": run_chat_prefill,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_PERFORMANCE_BASE_URL)
    parser.add_argument("--case", help="case id; omit to list the cases")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    config = load_config("regular")
    if args.case is None:
        for case in config["cases"]:
            print(case["id"], case["kind"], "launch:", " ".join(launch_args(config, case)))
        return 0
    if args.output is None:
        parser.error("--output is required with --case")
    (case,) = select_cases(config, args.case)
    require_model(args.base_url)
    reports = RUNNERS[case["kind"]](args.base_url, case, args.timeout)
    status = "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL"
    return write_report(
        args.output,
        {"status": status, "case": case["id"], "launch": launch_args(config, case), "reports": reports},
        hide=("reports",),
    )


if __name__ == "__main__":
    raise SystemExit(main())
