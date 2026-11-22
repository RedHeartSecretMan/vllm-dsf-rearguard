"""Run the complete 1M performance metric suite inside an open window."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from scripts.benchmarks.baseline_compare import BASE_URL, make_command, wait_health

REGULAR_CASES = (
    "short_concurrency",
    "single_prefill",
    "single_decode",
    "matrix_single",
    "matrix_concurrent",
)
LONG_CASES = ("tool_history_16384_c16", "vision_mix_4096_c4")


def run_case(module: str, case: str, output: Path, timeout: float) -> int:
    command = [
        "python3.10",
        "-m",
        module,
        "--base-url",
        BASE_URL,
        "--case",
        case,
        "--output",
        str(output),
        "--timeout",
        str(timeout),
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--official", action="store_true")
    args = parser.parse_args()

    command = make_command(args.image, args.name, args.volume, args.official, "probabilistic")
    container = subprocess.check_output(command, text=True).strip()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "FAIL",
        "label": args.label,
        "image": args.image,
        "official": args.official,
        "cases": {},
        "started_at": time.time(),
    }
    try:
        wait_health(timeout=1800)
        for case in REGULAR_CASES:
            timeout = 3600 if case in {"matrix_single", "matrix_concurrent"} else 1800
            report["cases"][case] = run_case(
                "scripts.benchmarks.bench_regular",
                case,
                args.result_dir / f"{case}.json",
                timeout,
            )
        for case in LONG_CASES:
            report["cases"][case] = run_case(
                "scripts.benchmarks.bench_long_tail",
                case,
                args.result_dir / f"{case}.json",
                3600,
            )
        report["status"] = "PASS" if all(code == 0 for code in report["cases"].values()) else "FAIL"
    finally:
        subprocess.run(["docker", "stop", "--time", "120", container], check=False)
        logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True, check=False)
        (args.result_dir / "server.log").write_text(logs.stdout + logs.stderr)
        report["finished_at"] = time.time()
        (args.result_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
