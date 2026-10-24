"""Shared catalog loading and selection for GPU acceptance runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config" / "acceptance"


def load(name: str) -> dict[str, Any]:
    """Load one acceptance catalog by stem and validate its basic shape."""
    path = CONFIG_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError(f"{path} must contain a cases list")
    if not value.get("runner"):
        raise ValueError(f"{path} must name its runner")
    return value


def select(config: dict[str, Any], case_id: str | None) -> list[dict[str, Any]]:
    cases = config["cases"]
    if case_id is None:
        return list(cases)
    selected = [case for case in cases if case.get("id") == case_id]
    if not selected:
        choices = ", ".join(case.get("id", "<unnamed>") for case in cases)
        raise ValueError(f"unknown case {case_id!r}; choose from {choices}")
    return selected


def print_cases(config: dict[str, Any]) -> None:
    for case in config["cases"]:
        details = ", ".join(
            f"{key}={value}" for key, value in case.items() if key != "id"
        )
        print(f"{case['id']}" + (f"  ({details})" if details else ""))
