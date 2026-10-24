"""List and prepare the PPL and GPQA quality acceptance cases."""

from __future__ import annotations

import argparse
import json

from scripts.acceptance.catalog import load, print_cases, select


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list catalog cases")
    parser.add_argument("--case")
    args = parser.parse_args()
    config = load("quality")
    if args.list:
        print_cases(config)
        return 0
    cases = select(config, args.case)
    if len(cases) != 1:
        parser.error("--case is required when printing one quality plan")
    case = cases[0]
    print(json.dumps({"status": "PLAN", "case": case["id"], "command": case["command"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
