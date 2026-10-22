"""Compare deterministic acceptance evidence before enabling an optimization.

The gate is deliberately transport independent: it consumes two JSON receipts
from ``scripts/acceptance.py`` and checks that every case passed and produced
the same assistant-visible digest.  It also reports speculative-decoding
metadata without inventing an acceptance rate when the backend does not expose
the counters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _cases(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["name"]): row for row in receipt.get("cases", [])}


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Return an auditable quality decision for two acceptance receipts."""

    expected = _cases(reference)
    actual = _cases(candidate)
    reasons: list[str] = []
    if reference.get("status") != "PASS":
        reasons.append("reference acceptance receipt is not PASS")
    if candidate.get("status") != "PASS":
        reasons.append("candidate acceptance receipt is not PASS")
    if set(expected) != set(actual):
        reasons.append("acceptance case sets differ")
    mismatches: list[dict[str, Any]] = []
    for name in sorted(set(expected) & set(actual)):
        before = expected[name]
        after = actual[name]
        before_hash = before.get("output", {}).get("sha256")
        after_hash = after.get("output", {}).get("sha256")
        if not before_hash or not after_hash:
            reasons.append(f"{name}: missing assistant output digest")
        elif before_hash != after_hash:
            mismatches.append(
                {"case": name, "reference": before_hash, "candidate": after_hash}
            )
        if after.get("status") != "PASS":
            reasons.append(f"{name}: candidate case failed")
    if mismatches:
        reasons.append("assistant output digests differ")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "cases": len(expected),
        "digest_mismatches": mismatches,
        "reasons": reasons,
        "reference_speculative": reference.get("speculative"),
        "candidate_speculative": candidate.get("speculative"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        json.loads(args.reference.read_text()),
        json.loads(args.candidate.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
