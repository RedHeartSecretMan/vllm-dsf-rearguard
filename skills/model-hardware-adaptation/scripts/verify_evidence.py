#!/usr/bin/env python3
"""Validate adaptation evidence records and their artifact hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = SKILL_ROOT / "references" / "evidence-schema.json"


class EvidenceError(ValueError):
    """Raised when an evidence record fails validation."""


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise EvidenceError(f"unsupported schema type: {expected}")


def validate_schema(value: Any, schema: dict[str, Any], path: str) -> None:
    """Validate the JSON Schema subset used by evidence-schema.json."""
    if "const" in schema and value != schema["const"]:
        raise EvidenceError(f"{path}: expected {schema['const']!r}")

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(item) for item in schema["enum"])
        raise EvidenceError(f"{path}: expected one of {allowed}")

    expected_types = schema.get("type")
    if expected_types is not None:
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(_matches_type(value, item) for item in expected_types):
            names = ", ".join(expected_types)
            raise EvidenceError(f"{path}: expected type {names}, got {type(value).__name__}")

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise EvidenceError(f"{path}: missing fields: {', '.join(missing)}")

        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                raise EvidenceError(f"{path}: unexpected fields: {', '.join(unexpected)}")

        for key, child in properties.items():
            if key in value:
                validate_schema(value[key], child, f"{path}.{key}")

    if isinstance(value, list):
        minimum = schema.get("minItems")
        if minimum is not None and len(value) < minimum:
            raise EvidenceError(f"{path}: expected at least {minimum} items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate_schema(item, item_schema, f"{path}[{index}]")

    if isinstance(value, str):
        minimum = schema.get("minLength")
        if minimum is not None and len(value) < minimum:
            raise EvidenceError(f"{path}: expected at least {minimum} characters")
        pattern = schema.get("pattern")
        if pattern and re.search(pattern, value) is None:
            raise EvidenceError(f"{path}: does not match {pattern}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise EvidenceError(f"{path}: expected value >= {minimum}")


def parse_time(value: str, path: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError(f"{path}: invalid RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise EvidenceError(f"{path}: timestamp must include a timezone")
    return parsed


def relative_path(value: str, path: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EvidenceError(f"{path}: path must stay within the evidence root")
    return candidate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def present(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().upper() not in {"", "UNKNOWN", "TBD", "NONE"}


def validate_semantics(record: dict[str, Any], root: Path) -> None:
    started = parse_time(record["started_at"], "started_at")
    finished = parse_time(record["finished_at"], "finished_at")
    if finished < started:
        raise EvidenceError("finished_at must not precede started_at")

    artifacts = record["artifacts"]
    if record["status"] in {"PASS", "FAIL"} and not artifacts:
        raise EvidenceError(f"artifacts: {record['status']} requires at least one artifact")
    if record["status"] in {"PASS", "FAIL"} and not record["coverage"]["cases"]:
        raise EvidenceError(f"coverage.cases: {record['status']} requires at least one case")

    seen_paths: set[Path] = set()
    for index, artifact in enumerate(artifacts):
        artifact_path = relative_path(artifact["path"], f"artifacts[{index}].path")
        if artifact_path in seen_paths:
            raise EvidenceError(f"artifacts[{index}].path: duplicate artifact path")
        seen_paths.add(artifact_path)
        absolute = root / artifact_path
        if not absolute.is_file():
            raise EvidenceError(f"artifacts[{index}].path: file does not exist: {absolute}")
        actual = file_sha256(absolute)
        if actual != artifact["sha256"]:
            raise EvidenceError(
                f"artifacts[{index}].sha256: expected {artifact['sha256']}, got {actual}"
            )

    for index, item in enumerate(record["model_files"]):
        relative_path(item["path"], f"model_files[{index}].path")

    failure = record["failure"]
    status = record["status"]
    if record["evidence_class"] == "assumption" and status != "NOT_RUN":
        raise EvidenceError("status: assumption evidence must use NOT_RUN")

    if status == "PASS":
        if record["evidence_class"] == "assumption":
            raise EvidenceError("status: PASS cannot be assumption evidence")
        if record["workspace_state"] not in {"clean", "dirty-classified"}:
            raise EvidenceError("workspace_state: PASS requires a known, classified workspace")
        if failure["root_cause_status"] == "unresolved":
            raise EvidenceError("failure.root_cause_status: PASS cannot be unresolved")

    if status == "FAIL":
        required = ("class", "layer", "first_trusted_error", "reproduction", "next_action")
        missing = [field for field in required if not present(failure[field])]
        if missing:
            raise EvidenceError(f"failure: FAIL missing {', '.join(missing)}")

    if status == "BLOCKED":
        if not present(failure["next_action"]):
            raise EvidenceError("failure.next_action: BLOCKED requires a next action")
        if not present(failure["class"]) and not record["limitations"]:
            raise EvidenceError("BLOCKED requires failure.class or limitations")

    if status in {"SKIP", "NA"} and not record["limitations"]:
        raise EvidenceError(f"limitations: {status} requires an explicit reason")

    if status == "NOT_RUN" and not present(failure["next_action"]):
        raise EvidenceError("failure.next_action: NOT_RUN requires a next action")

    stage = int(record["stage"])
    if stage >= 6 and status == "PASS":
        if record["source_revision"] == "WORKTREE":
            raise EvidenceError("source_revision: stage 6+ PASS requires a fixed revision")
        required = {
            "image_digest": record["image_digest"],
            "model_revision": record["model_revision"],
            "model_files": record["model_files"],
            "adaptation_manifest_sha256": record["adaptation_manifest_sha256"],
            "launch_args_sha256": record["launch_args_sha256"],
            "test_config_sha256": record["test_config_sha256"],
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise EvidenceError(f"provenance: PASS missing {', '.join(missing)}")

    if stage >= 7 and status == "PASS":
        hardware = record["hardware"]
        frameworks = record["framework_versions"]
        parallelism = record["parallelism"]
        required = {
            "hardware.gpu": hardware["gpu"],
            "hardware.topology": hardware["topology"],
            "driver": record["driver"],
            "cuda": record["cuda"],
            "framework_versions.pytorch": frameworks["pytorch"],
            "framework_versions.vllm": frameworks["vllm"],
            "framework_versions.compiler": frameworks["compiler"],
            "parallelism.communication_backend": parallelism["communication_backend"],
            "parallelism.speculative": parallelism["speculative"],
        }
        missing = [name for name, value in required.items() if not present(value)]
        if hardware["count"] <= 0:
            missing.append("hardware.count")
        if hardware["memory_gib_per_gpu"] is None or hardware["memory_gib_per_gpu"] <= 0:
            missing.append("hardware.memory_gib_per_gpu")
        if not present(hardware["compute_capability"]):
            missing.append("hardware.compute_capability")
        if parallelism["tp"] <= 0:
            missing.append("parallelism.tp")
        if parallelism["dp"] <= 0:
            missing.append("parallelism.dp")
        if missing:
            raise EvidenceError(f"provenance: runtime PASS missing {', '.join(missing)}")


def collect_paths(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(path.rglob("*.evidence.json"))
        elif path.is_file():
            files.append(path)
        else:
            raise EvidenceError(f"input path does not exist: {path}")
    return sorted({item.resolve() for item in files})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Evidence JSON files or directories")
    parser.add_argument("--root", default=".", help="Project root for relative artifact paths")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA), help="Evidence JSON Schema")
    args = parser.parse_args()

    try:
        root = Path(args.root).resolve()
        if not root.is_dir():
            raise EvidenceError(f"--root is not a directory: {root}")
        schema_path = Path(args.schema)
        if not schema_path.is_file():
            raise EvidenceError(f"--schema does not exist: {schema_path}")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        files = collect_paths(args.paths)
    except (EvidenceError, OSError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 2

    if not files:
        print("no *.evidence.json files found", file=sys.stderr)
        return 1

    errors: list[str] = []
    record_ids: dict[str, Path] = {}
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise EvidenceError("record must be a JSON object")
            validate_schema(record, schema, "$")
            validate_semantics(record, root)
            record_id = record["record_id"]
            if record_id in record_ids:
                raise EvidenceError(
                    f"record_id: duplicate with {record_ids[record_id]}"
                )
            record_ids[record_id] = path
        except (EvidenceError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {len(files)} evidence record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
