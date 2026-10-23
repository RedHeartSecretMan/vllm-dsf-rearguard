"""Paired CPU-only benchmark of synthetic long tool histories, with exact parity."""

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def history(turns):
    messages = [
        {
            "role": "system",
            "content": "Use lookup to check values.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                        },
                    },
                }
            ],
        },
        {"role": "user", "content": "Inspect the requested records."},
    ]
    for i in range(turns):
        messages += [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": f"Check record {i}.",
                "tool_calls": [
                    {
                        "id": str(i),
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": json.dumps({"key": str(i)}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": str(i),
                "content": f"Record {i} has value {i * 17}.",
            },
        ]
    return messages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument(
        "--candidate",
        type=Path,
        default=ROOT
        / "adaptation/files/vllm/tokenizers/deepseek_v41_encoding.py",
    )
    p.add_argument("--turns", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--owned-path",
        action="store_true",
        help="Measure candidate's already-owned message path",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.repeats < 2 or any(n < 1 for n in args.turns):
        p.error("At least two repetitions and positive history sizes are required")
    if args.output.exists():
        p.error("Use a new output path")
    base, candidate = load(args.baseline, "baseline"), load(args.candidate, "candidate")
    reference = load(ROOT / "adaptation/reference/encoder.py", "reference")
    rows = []
    for turns in args.turns:
        messages = history(turns)
        for mode in ("chat", "thinking"):
            kwargs = {"thinking_mode": mode, "reasoning_effort": "high"}
            wanted = reference.encode_messages(messages, **kwargs)
            samples = [[], []]
            for repeat in range(args.repeats):
                # Alternate A/B order. Check every result against the independent reference.
                for which in [0, 1] if repeat % 2 == 0 else [1, 0]:
                    started = time.perf_counter_ns()
                    if args.owned_path and which == 1:
                        actual = candidate.encode_messages(
                            copy.deepcopy(messages), copy_messages=False, **kwargs
                        )
                    else:
                        actual = (base, candidate)[which].encode_messages(
                            messages, **kwargs
                        )
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    if actual != wanted:
                        raise AssertionError(
                            f"Official prompt mismatch: {turns}/{mode}/{which}"
                        )
                    samples[which].append(elapsed)
            before, after = (statistics.median(v) for v in samples)
            row = {
                "tool_turns": turns,
                "messages": len(messages),
                "mode": mode,
                "encoded_chars": len(wanted),
                "output_sha256": hashlib.sha256(wanted.encode()).hexdigest(),
                "baseline_ms": samples[0],
                "candidate_ms": samples[1],
                "baseline_median_ms": before,
                "candidate_median_ms": after,
                "cpu_encoding_speedup": before / after,
                "exact_reference_match": True,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "tool_turns",
                            "mode",
                            "baseline_median_ms",
                            "candidate_median_ms",
                            "cpu_encoding_speedup",
                        )
                    }
                )
            )
    report = {
        "status": "PASS",
        "scope": "Python prompt encoding only; no tokenization, model, GPU or serving throughput measurement",
        "python": sys.version,
        "platform": platform.platform(),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
