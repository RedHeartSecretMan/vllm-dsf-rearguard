"""Offline audit and per-cell comparison of every profiling campaign attempt.

GPQA must be normalized by an actual evaluator-output validator, not by this
reporter: {status: PASS, questions: N, scored: N, validator: NAME, rawrefs:
[{path: PATH, sha256: SHA256}]}. The reporter verifies those raw files, but does
not claim to understand the external evaluator's scoring format.

Optional resolutions are a list (or {resolutions: [...]}) of {attempt_id,
reason, rawrefs: [{path, sha256}], replacement_profile_ids: [...]}. They never
erase the failed attempt; each needs original failure evidence and verified
PASS replacements in the final schedule.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TPS = ("input_tps", "output_tps", "total_tps")
METRICS = TPS + tuple(f"{name}_p{p}_s" for name in ("ttft", "tpot", "e2e") for p in (50, 95, 99))
ELIGIBLE = {"APPLICABLE", "PENDING_PREPARATION"}
MATRIX = {"matrix_single", "matrix_concurrent"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class Evidence:
    def __init__(self):
        self.files: dict[str, dict] = {}

    def record(self, path: Path):
        self.files[str(path.resolve())] = {"sha256": _sha(path), "bytes": path.stat().st_size}

    def read(self, path: Path):
        self.record(path)
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as stream:
                return json.load(stream)
        return json.loads(path.read_text())

    def rawrefs(self, refs: list, directory: Path) -> list[Path]:
        _require(isinstance(refs, list) and refs, "rawrefs must contain real evidence files")
        paths = []
        for ref in refs:
            path = Path(ref["path"])
            path = path if path.is_absolute() else directory / path
            self.record(path)
            _require(self.files[str(path.resolve())]["sha256"] == ref["sha256"], f"raw evidence hash mismatch: {path}")
            paths.append(path.resolve())
        return paths


def _quantile(values, percent):
    return sorted(values)[math.ceil(len(values) * percent / 100) - 1] if values else None


def _measurement(measurement, concurrency):
    _require(measurement.get("status") == "PASS", "measurement is not PASS")
    rows = measurement.get("requests")
    _require(isinstance(rows, list) and rows, "performance measurement lacks per-request raw metrics")
    _require(measurement.get("cache_verified") is True, "cold cache not verified")
    _require(len(rows) == measurement.get("request_count") == measurement.get("success"), "request count/success mismatch")
    _require(measurement.get("failure_rate") == 0, "nonzero request failure rate")
    _require(not measurement.get("errors"), "measurement contains errors")
    _require(all(r.get("status") == "PASS" and type(r.get("cached_tokens")) is int and r["cached_tokens"] == 0 for r in rows),
             "failed request or nonzero/missing cached_tokens")
    lengths = []
    for row in rows:
        _require(all(type(row.get(k)) is int and row[k] > 0 for k in ("input_tokens", "output_tokens")), "invalid actual token counts")
        latency_keys = ("ttft_s", "e2e_s") + (("tpot_s",) if row["output_tokens"] > 1 else ())
        _require(all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) and row[k] > 0 for k in latency_keys),
                 "required per-request latency is missing or invalid")
        lengths.append((row["input_tokens"], row["output_tokens"]))
    elapsed = measurement.get("elapsed_s")
    _require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed > 0, "invalid elapsed time")
    totals = (sum(x[0] for x in lengths), sum(x[1] for x in lengths))
    _require(totals == (measurement.get("observed_input_tokens"), measurement.get("observed_output_tokens")), "actual token totals mismatch")
    metrics = {}
    for key, total in zip(TPS, (*totals, sum(totals)), strict=True):
        value = total / elapsed
        _require(math.isclose(measurement.get(key, -1), value, rel_tol=1e-9), f"{key} is not tokens / whole-burst wall clock")
        metrics[key] = value
    for name in ("ttft", "tpot", "e2e"):
        values = [r[name + "_s"] for r in rows if r.get(name + "_s") is not None]
        _require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values), "invalid request latency")
        for percent in (50, 95, 99):
            key = f"{name}_p{percent}_s"
            metrics[key] = _quantile(values, percent)
    if measurement.get("concurrency") is not None:
        _require(measurement["concurrency"] == concurrency, "concurrency differs from saved plan")
    return metrics, lengths


def _performance(planned, measured, inherited_references=()):
    result = {}
    measurements = measured["measurements"] + list(inherited_references)
    streams = ("decode", "prefill") if planned.get("execution_variant") == "fixed_arrival_schedule" else (None,)
    for stream in streams:
        key = planned["cell_id"] + ("/" + stream if stream else "")
        spec = planned["flows"][stream] if stream else planned
        rounds, signatures = [], []
        for measurement in measurements:
            part = measurement["streams"][stream] if stream else measurement
            metrics, lengths = _measurement(part, spec["concurrency"])
            _require(measurement.get("payload_sha256"), "missing payload_sha256")
            _require(all(pair[1] == spec["osl"] for pair in lengths), "output length differs from plan")
            if not stream:
                _require(len(lengths) == spec["concurrency"], "burst request count differs from planned concurrency")
                expected = spec.get("expected_input_by_index")
                if expected is not None:
                    _require([p[0] for p in lengths] == expected, "per-request input differs from plan")
                elif spec.get("actual_expected_input") is not None:
                    _require(all(p[0] == spec["actual_expected_input"] for p in lengths), "input length differs from prepared input")
            else:
                _require(all(p[0] == spec["isl"] for p in lengths), "mixed input length differs")
                _require(measurement.get("execution_variant") == planned["execution_variant"], "mixed derivation differs")
                _require(measurement.get("request_count") == measurement.get("expected_request_count")
                         == sum(s["request_count"] for s in measurement["streams"].values()), "mixed request count differs")
            rounds.append(metrics)
            signatures.append({"payload_sha256": measurement["payload_sha256"],
                               "arrivals_sha256": measurement.get("arrivals_sha256"), "lengths": lengths})
        result[key] = {
            "case_id": planned["case_id"], "kind": planned["kind"], "stream": stream,
            "dataset": planned.get("dataset"), "concurrency": spec["concurrency"],
            "primary_repeat": 1, "primary_measurements": 1, "recorded_repeats": len(rounds),
            "metrics": rounds[0],
            "reference_measurements": [{"repeat": index + 1, "metrics": metrics,
                                         "signature": signatures[index]}
                                        for index, metrics in enumerate(rounds) if index > 0],
            "comparison_signature": {"kind": planned["kind"], "stream": stream, "dataset": planned.get("dataset"),
                                     "concurrency": spec["concurrency"], "primary_repeat": 1, "rounds": signatures[:1]},
        }
    return result


def _normalized_flags(command, image):
    args = list(command)
    _require(image in args, "launch command does not contain its immutable image ID")
    for flag in ("--name", "--max-model-len", "--max-num-batched-tokens", "--gpu-memory-utilization"):
        _require(args.count(flag) == 1, f"expected one {flag}")
        index = args.index(flag) + 1
        value = args[index]
        if flag == "--name":
            volume = f"type=volume,src={value}-cache,dst=/workspace"
            _require(args.count(volume) == 1, "unexpected task cache volume")
            args[args.index(volume)] = "type=volume,src=<attempt>-cache,dst=/workspace"
        args[index] = "<" + flag.removeprefix("--") + ">"
    return args


def _performance_cell(cell):
    return cell["group"] in ("regular", "long_tail") and cell["kind"] not in ("mixed", "soak")


def _legacy_repeats(cell):
    if cell["group"] not in ("regular", "long_tail") or cell["kind"] in ("mixed", "soak"):
        return 1
    if cell["case_id"] == "long_request_capacity" or cell["case_id"].startswith("capacity_fixed_profile_"):
        return 1
    return 3


def _policy_version(plan, summary):
    version = plan.get("policy_version", 1)
    _require(type(version) is int and version in (1, 2)
             and summary.get("policy_version", 1) == version, "unknown or inconsistent measurement policy version")
    if version == 2:
        policy = plan.get("measurement_policy", {})
        _require(policy == summary.get("measurement_policy") and policy.get("id") == "single_formal_v2"
                 and policy.get("primary_repeat") == 1 and policy.get("formal_repetitions") == 1,
                 "v2 requires the explicit single_formal_v2 policy")
    for cell in plan["cells"]:
        if cell["status"] in ELIGIBLE:
            _require(cell["repeats"] == (1 if version == 2 else _legacy_repeats(cell)),
                     f"{cell['cell_id']}: repetitions differ from policy v{version}")
    return version


def _warmup(warm, *, version, osl, global_scope=False):
    """Check the declared load and derive dual-DP activity from counters."""
    _require(isinstance(warm, dict) and warm.get("status") == "PASS" and not warm.get("errors"), "warmup is not PASS")
    rounds = warm.get("rounds")
    configured = 2 if global_scope or version == 1 else 1
    concurrency = 256 if global_scope else 2
    expected_osl = 128 if global_scope else min(osl, 128 if version == 1 else 2)
    _require(warm.get("concurrency") == concurrency and warm.get("osl") == expected_osl,
             "warmup concurrency/output budget differs from policy")
    _require(isinstance(rounds, list) and len(rounds) >= configured, "warmup rounds are incomplete")
    if version == 2:
        _require(warm.get("configured_rounds") == configured and len(rounds) <= configured + 2,
                 "v2 warmup configured/additional round count differs")
        additions = warm.get("additional_rounds", [])
        _require(len(additions) == len(rounds) - configured == warm.get("additional_round_count", 0),
                 "v2 additional warmup rounds lack their recorded reasons")
        for item, actual in zip(additions, rounds[configured:], strict=True):
            reason = item.get("reason", {})
            _require(item.get("round") == actual.get("round") and reason == actual.get("trigger")
                     and reason.get("code") in ("dp_not_covered", "service_not_ready", "replace_not_ready_round"),
                     "unexplained additional warmup round")
            if reason["code"] == "dp_not_covered":
                _require(reason.get("missing_engines") and reason.get("dp_deltas"), "missing-DP retry lacks counter evidence")
            else:
                _require(reason.get("errors") or reason.get("failed_rounds"), "readiness retry lacks failure evidence")
    passed = 0
    for index, row in enumerate(rounds, 1):
        _require(row.get("round") == index, "warmup round order differs")
        measurement = row.get("measurement", {})
        if measurement.get("status") == "PASS":
            passed += 1
            _require(measurement.get("cache_verified") is True, "warmup cache evidence is not cold")
        else:
            requests = measurement.get("requests", [])
            failures = [str(r.get("error", "")) for r in requests if r.get("status") != "PASS"]
            _require(version == 2 and failures and all(re.search(r"HTTP(?: Error)? 503\b|(?:server|service) (?:is )?not ready", e, re.I) for e in failures),
                     "warmup contains a real measurement failure")
            _require(all(r.get("cached_tokens") == 0 for r in requests if r.get("status") == "PASS"), "warmup contains warm cache")
    _require(passed >= configured, "warmup has too few successful rounds")
    before, after = warm.get("dp_before", {}), warm.get("dp_after", {})
    _require(len(before) == 2 and set(before) == set(after) and warm.get("both_dp_active") is True,
             "warmup lacks two-engine evidence")
    for engine in before:
        for key in ("prompt_tokens", "generation_tokens"):
            start, end = before[engine].get(key), after[engine].get(key)
            _require(isinstance(start, (int, float)) and isinstance(end, (int, float))
                     and math.isfinite(start) and math.isfinite(end) and 0 <= start < end,
                     "warmup DP counters are missing, reset, or inactive")


def _reference_json(ref, directory, evidence):
    path = evidence.rawrefs([ref], directory)[0]
    return evidence.read(path), path


def _payload_digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _bind_source_raw(folder, cell, rounds, evidence):
    """A summary cannot promote a partial/unwritten measurement to reused data."""
    if _performance_cell(cell):
        for index, measurement in enumerate(rounds, 1):
            directory = folder / f"repeat-{index}"
            _require(evidence.read(directory / "report.json") == measurement, "resume raw report differs from source measurement")
            payloads = evidence.read(directory / "request-payloads.json.gz")
            metrics = evidence.read(directory / "request-metrics.json.gz")
            _require(metrics == measurement.get("requests"), "resume raw request metrics differ")
            _require(_payload_digest(payloads) == measurement.get("payload_sha256"), "resume payload hash differs")
            _require(len(payloads) == cell["concurrency"] and all(p.get("max_tokens") == cell["osl"] for p in payloads),
                     "resume raw payload count/output budget differs")
    elif cell["kind"] == "mixed":
        _require(len(rounds) == 1 and evidence.read(folder / "result.json.gz") == rounds[0], "mixed resume report differs")
        raw = evidence.read(folder / "requests.json.gz")
        _require(_payload_digest(raw["bodies"]) == rounds[0]["payload_sha256"]
                 and _payload_digest(raw["arrivals"]) == rounds[0]["arrivals_sha256"], "mixed resume inputs differ")
    else:
        _require(len(rounds) == 1 and evidence.read(folder / "report.json") == rounds[0], "acceptance/PPL resume raw report differs")
        if rounds[0].get("raw_files"):
            evidence.rawrefs(rounds[0]["raw_files"], folder)
        if cell["kind"] == "ppl":
            evidence.rawrefs([{"path": r["raw_path"], "sha256": r["raw_sha256"]} for r in rounds[0]["rows"]], folder)


def _resume(root, directory, plan, summary, launch, evidence, seen=()):
    provenance = summary.get("resume_provenance")
    if provenance is None:
        _require(not summary.get("global_warmup_reuse"), "global warmup reuse lacks provenance")
        return {}, set()
    _require(summary.get("policy_version") == 2 and provenance.get("schema_version") == 1, "unsupported resume provenance")
    sources, mappings = provenance.get("sources"), provenance.get("cells")
    _require(isinstance(sources, dict) and sources and isinstance(mappings, dict) and mappings, "resume provenance is empty")
    _require(set(sources) == {v["source_attempt_id"] for v in mappings.values()}, "unused/missing resume source")
    current = {c["cell_id"]: c for c in summary["cells"]}
    expected = {c["cell_id"]: c for c in plan["cells"]}
    loaded, acknowledgements = {}, set()
    for ident, refs in sources.items():
        _require(ident not in seen and ident != directory.parent.name and Path(ident).name == ident, "cyclic or invalid resume source")
        old_plan, plan_path = _reference_json(refs["plan"], directory, evidence)
        source, _ = _reference_json(refs["summary"], directory, evidence)
        old_launch, _ = _reference_json(refs["launch"], directory, evidence)
        _require(plan_path.parent == (root / ident / "results").resolve(), "resume source plan is outside its attempt")
        version = _policy_version(old_plan, source)
        _require(source["profile"] == plan["profile"] == old_plan["profile"]
                 and source["frozen_inputs_sha256"] == summary["frozen_inputs_sha256"], "resume profile/frozen inputs differ")
        for key in ("image_id", "source_manifest_sha256", "container_id", "command"):
            _require(old_launch.get(key) == launch.get(key), f"resume launch {key} differs")
        _require(source["source"]["source_manifest_sha256"] == launch["source_manifest_sha256"]
                 and source["endpoint"] == summary["endpoint"], "resume source/endpoint differs")
        _require(not any(source.get(k) for k in ("error", "traceback", "endpoint_error", "observation_error"))
                 and source.get("observation", {}).get("status") == "PASS", "resume source has unresolved runtime failure")
        _require(not any(c["status"] == "FAIL" or c.get("error") or c.get("traceback") for c in source["cells"]),
                 "resume cannot erase a failed source cell")
        old_expected = {c["cell_id"]: c for c in old_plan["cells"]}
        _require(set(old_expected) == set(expected), "resume changes the planned case inventory")
        old_actual = {c["cell_id"]: c for c in source["cells"]}
        _require(len(old_actual) == len(source["cells"]), "resume source has duplicate cells")
        handoff = None
        if source["status"] != "PASS":
            handoff, handoff_path = _reference_json(provenance["handoff"], directory, evidence)
            _require(source["status"] in ("IN_PROGRESS", "POLICY_HANDOFF") and handoff.get("status") == "POLICY_HANDOFF"
                     and handoff.get("source_attempt_id") == ident and handoff.get("no_inflight") is True
                     and not handoff.get("pending_unsaved_payloads")
                     and handoff.get("source_snapshot_sha256") == refs["summary"]["sha256"], "invalid drained policy handoff")
            evidence.rawrefs(handoff.get("rawrefs"), handoff_path.parent)
            boundary = old_actual[handoff["current_cell"]]
            _require(handoff.get("pending_unsent") == list(range(len(boundary.get("measurements", [])) + 1,
                                                                 old_expected[handoff["current_cell"]]["repeats"] + 1)),
                     "handoff unsent repetitions differ from the saved boundary")
            acknowledgements.add(ident)
        nested, inherited_handoffs = _resume(root, plan_path.parent, old_plan, source, old_launch, evidence, (*seen, directory.parent.name, ident))
        acknowledgements.update(inherited_handoffs)
        if source.get("global_warmup_reuse"):
            _require(nested, "source global warmup reuse lacks verified provenance")
        else:
            _warmup(source.get("global_warmup"), version=version, osl=128, global_scope=True)
        loaded[ident] = (source, old_expected, old_actual, version, handoff, plan_path.parent, nested)
    output = {}
    identity_keys = ("case_id", "group", "kind", "dataset", "isl", "osl", "concurrency", "source_case", "source_launch",
                     "expected_input_by_index", "actual_expected_input", "execution_variant", "flows")
    for cell_id, origin in mappings.items():
        _require(cell_id in expected and origin.get("source_cell_id") == cell_id and origin.get("primary_repeat") == 1,
                 "resume cell identity/primary repeat differs")
        source, old_expected, old_actual, version, handoff, source_directory, nested = loaded[origin["source_attempt_id"]]
        before, after = old_expected[cell_id], expected[cell_id]
        _require(before["status"] in ELIGIBLE and after["status"] in ELIGIBLE
                 and all(before.get(k) == after.get(k) for k in identity_keys), "resume changes case input/type/concurrency")
        row = old_actual[cell_id]
        partial = row["status"] == "IN_PROGRESS" and handoff and handoff["current_cell"] == cell_id
        _require(row["status"] == "PASS" or partial, "resume source cell is neither PASS nor the explicit handoff boundary")
        rounds = row.get("measurements", [])
        _require(rounds and all(m.get("status") == "PASS" for m in rounds), "resume source contains failed/incomplete measurements")
        _require(len(rounds) <= before["repeats"] and (partial or len(rounds) == before["repeats"]), "resume source repetitions are incomplete")
        effective_version = nested.get(cell_id, {}).get("source_policy_version", version)
        if _performance_cell(before):
            _warmup(row.get("warmup"), version=effective_version, osl=before["osl"])
        _bind_source_raw(source_directory / "cells" / cell_id.replace("/", "__"), before, rounds, evidence)
        if cell_id in current:
            _require(current[cell_id].get("measurements") == rounds[:1]
                     and current[cell_id].get("warmup") == row.get("warmup"), "resumed primary measurement/warmup differs from source repeat-1")
        else:
            _require(summary.get("status") == "IN_PROGRESS", "final resumed summary omits an inherited cell")
        references = rounds[1:] or nested.get(cell_id, {}).get("reference_measurements", [])
        output[cell_id] = {"source_attempt_id": origin["source_attempt_id"], "source_policy_version": effective_version,
                           "reference_measurements": references, "primary_repeat": 1}
    if summary.get("global_warmup_reuse"):
        _require(any(summary.get("global_warmup") == item[0].get("global_warmup") for item in loaded.values()),
                 "inherited global warmup differs from its source")
    return output, acknowledgements


def _attempt(root, entry, evidence, policy=None):
    policy = policy or default_policy()
    ident = entry.get("attempt_id", entry["id"])
    _require(Path(ident).name == ident and ident not in (".", ".."), "attempt ID must be a directory name")
    directory = root / ident / "results"
    summary_path = directory / "summary.json"
    progress = not summary_path.exists()
    if progress:
        summary_path = directory / "progress.json"
    result = {"attempt_id": ident, "profile_id": entry["id"], "raw_status": entry["status"],
              "raw_attempt": copy.deepcopy(entry), "resolution_status": "NOT_NEEDED",
              "summary_path": str(summary_path), "validation_errors": [], "cells": {}, "complete": False}
    try:
        summary = evidence.read(summary_path)
        plan = evidence.read(directory / "plan.json")
        launch = evidence.read(directory / "launch.json")
        version = _policy_version(plan, summary)
        handoff = entry["status"] == "POLICY_HANDOFF" and summary.get("status") in ("IN_PROGRESS", "POLICY_HANDOFF")
        result.update(policy_version=version, policy_handoff=handoff)
        result.update(raw_summary_status=summary.get("status"), profile=summary["profile"],
                      source=summary["source"], image_id=summary["endpoint"]["image_id"],
                      source_fingerprint=summary["source"]["source_manifest_sha256"],
                      frozen_inputs_sha256=summary["frozen_inputs_sha256"], flags=launch["command"],
                      plan_counts=plan.get("counts"), selection=summary.get("selection", "all"),
                      raw_observation=summary.get("observation"),
                      failure={k: summary[k] for k in ("error", "traceback", "endpoint_error", "observation_error") if k in summary},
                      recorded_cells=[{k: c[k] for k in ("cell_id", "case_id", "status", "reason", "error", "traceback") if k in c}
                                      for c in summary["cells"]])
        result["normalized_flags"] = _normalized_flags(launch["command"], launch["image_id"])
        _require(summary["endpoint"]["image_id"] == launch["image_id"], "image differs from launch")
        _require(summary["endpoint"]["container_id"] == launch["container_id"], "container differs from launch")
        _require(result["source_fingerprint"] == launch["source_manifest_sha256"], "source fingerprint differs from launch")
        for axis in ("context", "memory", "chunk"):
            _require(summary["profile"][axis] == plan["profile"][axis] == entry[axis], f"{axis} differs from attempt or plan")
        for axis, flag in (("context", "--max-model-len"), ("memory", "--gpu-memory-utilization"), ("chunk", "--max-num-batched-tokens")):
            _require(float(launch["command"][launch["command"].index(flag) + 1]) == entry[axis], f"actual launch flag {flag} differs")
        _require(result["selection"] == entry.get("only", "all"), "selection differs from attempt")
        inherited, acknowledged = _resume(root, directory, plan, summary, launch, evidence)
        result["resume_provenance"] = summary.get("resume_provenance")
        result["acknowledged_handoffs"] = sorted(acknowledged)
        planned = [c for c in plan["cells"] if result["selection"] == "all" or c["case_id"] in MATRIX]
        expected = {c["cell_id"]: c for c in planned}
        actual = {c["cell_id"]: c for c in summary["cells"]}
        _require(len(expected) == len(planned) and len(actual) == len(summary["cells"]), "duplicate cell IDs")
        _require(not(set(actual) - set(expected)), "summary contains cells outside saved plan")
        missing = sorted(set(expected) - set(actual))
        result["missing_cells"] = missing
        applicable = [c for c in planned if c["status"] in ELIGIBLE]
        result["counts"] = {"planned": len(planned), "applicable": len(applicable), "recorded": len(actual),
                            "passed": sum(c["status"] == "PASS" for c in actual.values()),
                            "expected_repeats": sum(c["repeats"] for c in applicable),
                            "recorded_repeats": sum(len(c.get("measurements", [])) for c in actual.values())}
        for cell_id, measured in actual.items():
            cell = expected[cell_id]
            if cell["status"] not in ELIGIBLE:
                _require(measured["status"] == cell["status"], f"{cell_id}: inapplicable/once-only state changed")
                continue
            if measured["status"] != "PASS":
                if measured["status"] == "IN_PROGRESS" and (progress or handoff):
                    continue
                raise ValueError(f"{cell_id}: recorded status {measured['status']}")
            rounds = measured.get("measurements", [])
            _require(len(rounds) == cell["repeats"], f"{cell_id}: missing/extra repetitions")
            _require(all(m.get("status") == "PASS" for m in rounds), f"{cell_id}: non-PASS measurement")
            if version == 2 and _performance_cell(cell) and cell_id not in inherited:
                _warmup(measured.get("warmup"), version=2, osl=cell["osl"])
                folder = directory / "cells" / cell_id.replace("/", "__")
                _require({p.name for p in folder.glob("repeat-*") if p.is_dir()} == {"repeat-1"},
                         "fresh v2 cell contains undeclared formal repetitions")
                _bind_source_raw(folder, cell, rounds, evidence)
            if cell["group"] in ("regular", "long_tail"):
                values = _performance(cell, measured, inherited.get(cell_id, {}).get("reference_measurements", []))
                origin_version = inherited.get(cell_id, {}).get("source_policy_version", version)
                warm = measured.get("warmup") or summary.get("global_warmup", {})
                for value in values.values():
                    value["measurement_policy_version"] = version
                    value["warmup_policy"] = {
                        "policy_version": origin_version, "scope": "per_cell" if measured.get("warmup") else "global",
                        "configured_rounds": warm.get("configured_rounds", 2), "actual_rounds": len(warm.get("rounds", [])),
                        "concurrency": warm.get("concurrency"), "output_tokens": warm.get("osl"),
                    }
                result["cells"].update(values)
                if cell_id in inherited:
                    for key in result["cells"]:
                        if key == cell_id or key.startswith(cell_id + "/"):
                            result["cells"][key]["resume_origin"] = {k: v for k, v in inherited[cell_id].items() if k != "reference_measurements"}
            if cell["kind"] == "ppl":
                ppl = rounds[0]
                rows = ppl.get("rows", [])
                _require(rows and ppl.get("windows") == len(rows) == policy["ppl_windows"]
                         and all(r.get("status") == "PASS" for r in rows), "PPL windows incomplete")
                _require(all(type(r.get("cached_tokens")) is int and r["cached_tokens"] == 0 for r in rows), "PPL cache evidence is missing or not cold")
                _require(ppl.get("scored_tokens", 0) == sum(r["scored_tokens"] for r in rows) > 0, "PPL scored token total differs")
                _require(ppl.get("dataset_sha256"), "PPL dataset hash missing")
                result["ppl"] = {k: ppl.get(k) for k in ("status", "windows", "scored_tokens", "dataset_sha256", "perplexity")}
        if entry.get("only", "all") == "all":
            _require(any(c["kind"] == "ppl" and c["status"] in ELIGIBLE for c in planned), "saved plan omits mandatory PPL")
        if not progress and not handoff:
            _require(not missing, "final summary omits planned cells")
            _require(summary["status"] == "PASS", "final summary is not PASS")
            _require(all(c["status"] == "PASS" for c in actual.values() if expected[c["cell_id"]]["status"] in ELIGIBLE), "applicable cells unfinished")
            _require(summary.get("global_warmup", {}).get("status") == "PASS", "global warmup is not PASS")
            if version == 2 and not summary.get("global_warmup_reuse"):
                _warmup(summary.get("global_warmup"), version=2, osl=128, global_scope=True)
            _require(summary.get("observation", {}).get("status") == "PASS", "observer evidence is not PASS")
            if entry.get("only", "all") == "all":
                _require(result.get("ppl", {}).get("status") == "PASS", "mandatory PPL is missing")
            result["complete"] = entry["status"] == "PASS"
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        result["validation_errors"].append(f"{type(error).__name__}: {error}")
    result["in_progress"] = progress and entry["status"] in ("STARTING", "RUNNING", "IN_PROGRESS")
    return result


def _quality(gpqa_path, soak_path, evidence, policy, identity):
    result, errors = {}, []
    for label, path in (("gpqa", gpqa_path), ("soak", soak_path)):
        if path is None:
            errors.append(f"missing mandatory {label} receipt")
            continue
        try:
            path = Path(path)
            value = evidence.read(path)
            _require(value.get("status") == "PASS", f"{label} is not PASS")
            if label == "gpqa":
                _require(value.get("questions") == value.get("scored") == policy["gpqa_questions"], "GPQA questions/scored coverage incomplete")
                _require(value.get("validator"), "GPQA requires the real evaluator-output validator identity")
                evidence.rawrefs(value.get("rawrefs"), path.parent)
                result[label] = {"status": "PASS", "receipt": str(path), "normalized": value,
                                 "scope": "External validator attests raw evaluator semantics; this reporter verifies coverage and referenced file hashes."}
            else:
                duration = policy["soak_duration_s"]
                _require(value.get("measurement_window_s") == duration and value.get("duration_s") == duration
                         and value.get("elapsed_s", 0) >= duration, "soak did not cover its full offered-load window")
                _require(value.get("observation", {}).get("status") == "PASS" and value.get("warmup", {}).get("status") == "PASS", "soak runtime evidence or warmup failed")
                _require(value.get("frozen_inputs_sha256") == identity["frozen_inputs_sha256"], "soak used different frozen inputs")
                _require(value.get("endpoint", {}).get("image_id") == identity["image_id"], "soak image differs")
                rows = evidence.read(path.parent / "request-metrics.json.gz")
                _require(len(rows) >= policy["soak_concurrency"], "soak did not submit even its initial concurrency")
                measured = {**value, "requests": rows}
                _, lengths = _measurement(measured, policy["soak_concurrency"])
                _require(all(pair == (policy["soak_isl"], policy["soak_osl"]) for pair in lengths), "soak request lengths differ")
                references = {r["index"]: r["payload_sha256"] for r in value.get("request_identities", [])}
                _require(len(references) == len(rows) and {r["index"] for r in rows} == set(references), "soak submitted/completed request IDs differ")
                _require(all(r.get("payload_sha256") == references[r["index"]] for r in rows), "soak payload identity mismatch")
                result[label] = {"status": "PASS", "receipt": str(path), "request_count": len(rows), "window_s": duration,
                                 "elapsed_s": value["elapsed_s"], "cache_verified": True}
        except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
            errors.append(f"{label}: {type(error).__name__}: {error}")
    return result, errors


def default_policy():
    grid = json.loads((ROOT / "config/benchmarks/profile_matrix.json").read_text())
    quality = json.loads((ROOT / "config/acceptance/quality.json").read_text())
    tail = json.loads((ROOT / "config/benchmarks/long_tail.json").read_text())
    soak = next(c for c in tail["cases"] if c["kind"] == "soak")
    gpqa = next(c for c in quality["cases"] if c["kind"] == "gpqa")
    return {"contexts": grid["contexts"], "memory_levels": len(grid["memory_utilizations"]),
            "baseline_chunk": grid["baseline_chunk"], "chunk_candidates": grid["chunk_candidates"],
            "gpqa_questions": gpqa["questions"], "ppl_windows": grid["quality"].get("ppl_windows", 134), "soak_duration_s": soak["duration_s"],
            "soak_concurrency": soak["concurrency"], "soak_isl": soak["isl"], "soak_osl": soak["osl"]}


def build_report(campaign_dir: Path, schedules: list[Path], *, gpqa_result: Path | None = None,
                 soak_result: Path | None = None, resolutions: Path | None = None,
                 controlled_assessment: Path | None = None,
                 allow_incomplete: bool = False, reference: str | None = None,
                 policy: dict | None = None) -> dict:
    root, evidence = Path(campaign_dir).resolve(), Evidence()
    policy = copy.deepcopy(policy or default_policy())
    state = evidence.read(root / "campaign.json")
    errors, pending, schedule = [], [], {}
    for path in schedules:
        for profile in evidence.read(Path(path)):
            ident = profile["id"]
            if ident in schedule:
                _require(all(schedule[ident].get(k, "all" if k == "only" else None) == profile.get(k, "all" if k == "only" else None)
                             for k in ("context", "memory", "chunk", "only")), "schedules disagree on a profile")
            schedule[ident] = profile
    primary = [p for p in schedule.values() if p.get("only", "all") == "all"]
    controlled = [p for p in schedule.values() if p.get("only") == "matrix"]
    levels = {p["memory"] for p in primary}
    pairs = [(p["context"], p["memory"]) for p in primary]
    expected = {(context, memory) for context in policy["contexts"] for memory in levels}
    if (len(levels) != policy["memory_levels"] or len(pairs) != len(set(pairs)) or set(pairs) != expected
            or any(p["chunk"] != policy["baseline_chunk"] for p in primary)):
        pending.append("final schedule does not contain the complete distinct primary context × memory grid")
    if len(primary) + len(controlled) != len(schedule):
        errors.append("schedule contains an unknown selection scope")
    if any(p["chunk"] not in policy["chunk_candidates"] for p in schedule.values()):
        errors.append("schedule chunk is outside declared candidates")
    assessed = None
    if controlled_assessment is not None:
        try:
            from scripts.benchmarks.profile_controlled_report import validate_receipt
            assessed = validate_receipt(controlled_assessment, evidence)
            _require(not assessed["validation_errors"], "controlled assessment contains invalid raw evidence")
            matching = [row for row in controlled if row["id"] == assessed["profile_id"]]
            _require(len(matching) == 1 and all(matching[0][key] == assessed["profile"][key]
                                               for key in ("context", "memory", "chunk")),
                     "controlled assessment does not match its scheduled configuration")
            _require(assessed["candidate_outcome"] == "REJECTED", "unsupported controlled candidate disposition")
        except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
            errors.append(f"invalid controlled assessment: {type(error).__name__}: {error}")
            assessed = None
    attempts = [_attempt(root, entry, evidence, policy) for entry in state.get("profiles", [])]
    # A later primary-only schedule cannot erase a controlled experiment that
    # already ran. In particular, omitting --controlled-assessment must not
    # make an ordinary primary replacement resolution valid for its failure.
    historical_controlled = {entry["id"] for entry in state.get("profiles", []) if entry.get("only") == "matrix"}
    scheduled_controlled = {profile["id"] for profile in controlled}
    for ident in sorted(historical_controlled - scheduled_controlled):
        errors.append(f"historical controlled profile must remain in the controlled schedule: {ident}")
    controlled_ids = historical_controlled | scheduled_controlled
    ids = [a["attempt_id"] for a in attempts]
    if len(ids) != len(set(ids)):
        errors.append("duplicate attempt IDs in campaign state")
    known = set(ids)
    if assessed is not None:
        for ref in assessed["inputs"]["summaries"]:
            path = Path(ref["path"]).resolve()
            if path.is_relative_to(root):
                known.add(path.relative_to(root).parts[0])
    for path in (gpqa_result, soak_result):
        if path is not None and Path(path).resolve().is_relative_to(root):
            known.add(Path(path).resolve().relative_to(root).parts[0])
    for directory in root.iterdir():
        if directory.is_dir() and directory.name not in known and ((directory / "results").exists() or (directory / "launch.log").exists()):
            errors.append(f"untracked attempt directory must not be omitted: {directory.name}")
    selected = {}
    for ident, scheduled in schedule.items():
        if assessed is not None and ident == assessed["profile_id"]:
            if assessed["status"] != "COMPLETE_WITH_FAILURES" or assessed["coverage"]["not_run"]:
                pending.append(f"{ident}: controlled coverage lacks {assessed['coverage']['not_run']} of 42 cells")
            for missing in assessed.get("unresolved_failure_attempt_ids", []):
                pending.append(f"{ident}: original failure {missing} lacks its own terminal evidence")
            continue
        candidates = [a for a in attempts if a["profile_id"] == ident]
        passes = [a for a in candidates if a["complete"] and not a["validation_errors"]]
        if not passes:
            pending.append(f"{ident}: no complete validated PASS attempt")
        else:
            selected[ident] = passes[-1]
            for key in ("context", "memory", "chunk"):
                if selected[ident]["profile"][key] != scheduled[key]:
                    errors.append(f"{ident}: successful attempt differs from final schedule {key}")
            if selected[ident]["selection"] != scheduled.get("only", "all"):
                errors.append(f"{ident}: successful attempt has the wrong scope")
    resolution_rows = []
    if resolutions is not None:
        resolution_data = evidence.read(Path(resolutions))
        resolution_rows = resolution_data.get("resolutions", []) if isinstance(resolution_data, dict) else resolution_data
    by_id = {a["attempt_id"]: a for a in attempts}
    continued_handoffs = {ident for attempt in attempts if not attempt["validation_errors"]
                         for ident in attempt.get("acknowledged_handoffs", [])}
    resolved = set()
    for row in resolution_rows:
        try:
            ident = row["attempt_id"]
            _require(ident in by_id and ident not in resolved, "resolution names an unknown/duplicate attempt")
            _require(by_id[ident]["raw_status"] == "FAIL" or by_id[ident].get("raw_summary_status") == "FAIL", "resolution target is not a failed attempt")
            _require(by_id[ident]["profile_id"] not in controlled_ids,
                     "controlled failures cannot be replaced by PASS configurations; require an independent controlled assessment")
            _require(isinstance(row.get("reason"), str) and row["reason"].strip(), "resolution requires a reason")
            replacements = row["replacement_profile_ids"]
            _require(isinstance(replacements, list) and replacements and all(p in selected for p in replacements), "resolution replacements must be validated PASS profiles in the final schedule")
            paths = evidence.rawrefs(row["rawrefs"], Path(resolutions).parent)
            _require(any(p.is_relative_to(root / ident) or p == root / "campaign.json" for p in paths), "resolution lacks original failed-attempt evidence")
            by_id[ident].update(resolution_status="RESOLVED", resolution=copy.deepcopy(row))
            resolved.add(ident)
        except (ValueError, KeyError, OSError, TypeError) as error:
            errors.append(f"invalid resolution: {error}")
    for attempt in attempts:
        if attempt.get("policy_handoff"):
            if attempt["attempt_id"] in continued_handoffs:
                attempt["resolution_status"] = "POLICY_HANDOFF_REFERENCED"
            else:
                attempt["resolution_status"] = "POLICY_HANDOFF_PENDING"
                pending.append(f"{attempt['attempt_id']}: policy handoff awaits a validated continuation")
        failed = attempt["raw_status"] == "FAIL" or attempt.get("raw_summary_status") == "FAIL"
        if (failed and assessed is not None and attempt["profile_id"] == assessed["profile_id"]
                and attempt["attempt_id"] in assessed["covered_failure_attempt_ids"]):
            attempt["resolution_status"] = "ASSESSED_FAILURE"
            attempt["controlled_assessment_outcome"] = "REJECTED"
        elif failed and attempt["attempt_id"] not in resolved:
            attempt["resolution_status"] = "UNRESOLVED"
            errors.append(f"unresolved failed attempt: {attempt['attempt_id']}")
        elif attempt["validation_errors"] and not attempt["in_progress"] and not failed:
            errors.extend(f"{attempt['attempt_id']}: {message}" for message in attempt["validation_errors"])
    identity = None
    for ident, attempt in selected.items():
        current = {key: attempt[key] for key in ("image_id", "source_fingerprint", "frozen_inputs_sha256", "normalized_flags")}
        if identity is None:
            identity = current
        elif current != identity:
            errors.append(f"{ident}: image/source/frozen inputs/fixed launch flags differ")
    ppl_hashes = {a["ppl"]["dataset_sha256"] for a in selected.values() if a.get("ppl")}
    if identity and state.get("inputs_sha256") != identity["frozen_inputs_sha256"]:
        errors.append("campaign and selected attempts disagree on frozen inputs")
    if len(ppl_hashes) > 1:
        errors.append("primary profiles used different PPL datasets")
    if assessed is not None and identity:
        fields = {"image_id": "image_id", "source_fingerprint": "source_fingerprint",
                  "frozen_inputs_sha256": "frozen_inputs_sha256", "normalized_flags": "normalized_flags"}
        if any(assessed[source] != identity[target] for source, target in fields.items()):
            errors.append("controlled assessment image/source/frozen inputs/fixed flags differ from primary profiles")
    signatures = {}
    for ident, attempt in selected.items():
        for cell_id, cell in attempt["cells"].items():
            if cell_id in signatures and signatures[cell_id] != cell["comparison_signature"]:
                errors.append(f"{ident}/{cell_id}: payload sequence, kind, actual lengths or concurrency differ")
            signatures.setdefault(cell_id, cell["comparison_signature"])
    reference = reference or (primary[0]["id"] if primary else None)
    comparisons = {}
    if reference in selected:
        baseline = selected[reference]
        for ident, attempt in selected.items():
            if ident == reference:
                continue
            cells = {}
            for cell_id in sorted(set(attempt["cells"]) & set(baseline["cells"])):
                current, before = attempt["cells"][cell_id], baseline["cells"][cell_id]
                if current["comparison_signature"] != before["comparison_signature"]:
                    cells[cell_id] = {"status": "NOT_COMPARABLE", "reason": "payload/kind/length/concurrency mismatch"}
                else:
                    changes = {}
                    for key in METRICS:
                        a, b = current["metrics"][key], before["metrics"][key]
                        changes[key] = None if a is None or b is None or b == 0 else (a / b - 1) * 100
                    warmup_differs = current.get("warmup_policy") != before.get("warmup_policy")
                    cells[cell_id] = {"status": "COMPARABLE", "change_pct": changes,
                                      "warmup_policy_differs": warmup_differs,
                                      "warmup": {"reference": before.get("warmup_policy"), "current": current.get("warmup_policy")},
                                      "caveats": (["warmup_policy_differs: repeat-1 inputs match, but warmup settings differ; assess this when choosing a recommended configuration."] if warmup_differs else [])}
            comparisons[ident] = {"reference": reference, "cells": cells}
    elif reference is not None:
        pending.append(f"reference {reference} has no complete validated result")
    quality, quality_errors = _quality(gpqa_result, soak_result, evidence, policy, identity or {})
    pending.extend(message for message in quality_errors if message.startswith("missing mandatory"))
    errors.extend(message for message in quality_errors if not message.startswith("missing mandatory"))
    controlled_comparison = None
    if assessed is not None:
        baselines = [(ident, value) for ident, value in selected.items()
                     if value["profile"]["context"] == assessed["profile"]["context"]
                     and value["profile"]["memory"] == assessed["profile"]["memory"]
                     and value["profile"]["chunk"] == policy["baseline_chunk"]]
        controlled_comparison = {"profile_id": assessed["profile_id"], "reference": None, "cells": {}}
        if len(baselines) == 1:
            baseline_id, baseline = baselines[0]
            controlled_comparison["reference"] = baseline_id
            for cell_id, outcome in assessed["cells"].items():
                current = outcome.get("performance")
                if current is None:
                    controlled_comparison["cells"][cell_id] = {"status": "UNAVAILABLE", "outcome": outcome["outcome"]}
                    continue
                before = baseline["cells"].get(cell_id)
                if before is None or current["canonical_comparison_signature"] != before["comparison_signature"]:
                    controlled_comparison["cells"][cell_id] = {"status": "NOT_COMPARABLE", "reason": "canonical input/shape differs from the 16K reference"}
                    errors.append(f"controlled {cell_id}: canonical comparison inputs differ")
                    continue
                changes = {key: None if current["metrics"][key] is None or before["metrics"][key] in (None, 0)
                           else (current["metrics"][key] / before["metrics"][key] - 1) * 100 for key in METRICS}
                different = current.get("warmup_policy") != before.get("warmup_policy")
                controlled_comparison["cells"][cell_id] = {
                    "status": "COMPARABLE_WITH_VERIFIED_SALT_SCOPE", "change_pct": changes,
                    "actual_payload_sha256": current["actual_payload_sha256"], "canonical_payload_sha256": current["canonical_payload_sha256"],
                    "warmup_policy_differs": different, "candidate_outcome": "REJECTED",
                    "caveats": ["Metrics cover this completed cell; the candidate retains terminal failures elsewhere."]
                    + (["warmup_policy_differs"] if different else []),
                }
        else:
            pending.append("controlled comparison needs one complete 16K reference with the same context and memory")
    if assessed is None:
        status = "FAIL" if errors else "IN_PROGRESS" if pending and allow_incomplete else "FAIL" if pending else "FINAL_PASS"
    else:
        status = "FAIL" if errors else "INCOMPLETE_WITH_FAILURES" if pending else "COMPLETE_WITH_FAILURES"
    return {"status": status, "raw_campaign_status": state.get("status"), "policy": policy,
            "schedule": list(schedule.values()), "primary_profile_ids": [p["id"] for p in primary],
            "controlled_chunk_profile_ids": [p["id"] for p in controlled], "attempts": attempts,
            "selected_attempts": {key: a["attempt_id"] for key, a in selected.items()},
            "reference": reference, "comparisons": comparisons, "once_only": quality,
            "controlled_assessment": assessed, "controlled_comparison": controlled_comparison,
            "errors": errors, "pending": pending, "raw_evidence": evidence.files,
            "statistic": "Per-cell repeat-1 is the primary result for both v1 and v2. Additional legacy repetitions remain individually labelled references. Percent changes use only identical repeat-1 payload hashes, kind, actual input/output counts and concurrency, with cache=0 on every request."}


def markdown(report):
    lines = ["# Profiling campaign", "", f"状态：**{report['status']}**；原始 campaign 状态：`{report['raw_campaign_status']}`。", "",
             "每个 case 使用 repeat-1 作为主结果；旧版其余轮次单独列为波动参考。", "",
             "| Attempt | 范围 | 原始状态 | 回执状态 | 失败处理 | 已记录 / 计划单元 |", "|---|---|---|---|---|---|"]
    for attempt in report["attempts"]:
        counts = attempt.get("counts", {})
        lines.append(f"| {attempt['attempt_id']} | {attempt.get('selection', '/')} | {attempt['raw_status']} | {attempt.get('raw_summary_status', '/')} | {attempt['resolution_status']} | {counts.get('recorded', '/')} / {counts.get('planned', '/')} |")
    for title, values in (("错误", report["errors"]), ("未完成", report["pending"])):
        if values:
            lines += ["", f"## {title}", ""] + ["- " + v for v in values]
    if report.get("controlled_assessment") is not None:
        controlled = report["controlled_assessment"]
        coverage = controlled["coverage"]
        lines += ["", "## 32K controlled assessment", "",
                  f"候选：**{controlled['candidate_outcome']}**；覆盖 {coverage['covered']}/42，性能完成 {coverage['measured_pass']}，"
                  f"存在终止性失败的单元 {coverage['terminal_failed']}，未覆盖 {coverage['not_run']}。",
                  "原始失败及诊断记录保留；warmup 失败没有性能估计。"]
    def show(value):
        return "—" if value is None else f"{value:.4f}"
    for attempt in report["attempts"]:
        if not attempt["cells"]:
            continue
        lines += ["", f"## {attempt['attempt_id']}", "",
                  "| Case | 输入 token/s | 输出 token/s | 总 token/s | TTFT p50 s | TPOT p95 s | E2E p95 s |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for cell_id, cell in attempt["cells"].items():
            keys = TPS + ("ttft_p50_s", "tpot_p95_s", "e2e_p95_s")
            lines.append("| " + cell_id + " | " + " | ".join(show(cell["metrics"][k]) for k in keys) + " |")
            for extra in cell.get("reference_measurements", []):
                lines.append("| " + cell_id + f" · repeat-{extra['repeat']} 参考 | " + " | ".join(show(extra["metrics"][k]) for k in keys) + " |")
    for profile, comparison in report["comparisons"].items():
        lines += ["", f"## {profile} 相对 {comparison['reference']}", "",
                  "| Case | 输入吞吐变化 | 输出吞吐变化 | 总吞吐变化 | TTFT p95 变化 | TPOT p95 变化 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for cell_id, cell in comparison["cells"].items():
            if cell["status"] != "COMPARABLE":
                lines.append(f"| {cell_id} | 不可比较 | — | — | — | — |")
                continue
            changes = [cell["change_pct"][key] for key in TPS + ("ttft_p95_s", "tpot_p95_s")]
            label = cell_id + ("（预热策略不同）" if cell.get("warmup_policy_differs") else "")
            lines.append("| " + label + " | " + " | ".join("—" if v is None else f"{v:+.2f}%" for v in changes) + " |")
    lines += ["", "逐 case 百分比、全部 attempts、启动参数、PPL、一次性 GPQA/soak 验证和原始文件 SHA256 见同名 JSON。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, action="append", required=True)
    parser.add_argument("--gpqa-result", type=Path)
    parser.add_argument("--soak-result", type=Path)
    parser.add_argument("--resolutions", type=Path)
    parser.add_argument("--controlled-assessment", type=Path)
    parser.add_argument("--reference")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    report = build_report(args.directory, args.schedule, gpqa_result=args.gpqa_result,
                          soak_result=args.soak_result, resolutions=args.resolutions,
                          controlled_assessment=args.controlled_assessment,
                          reference=args.reference, allow_incomplete=args.allow_incomplete)
    prefix = args.output_prefix or args.directory / "comparison"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    prefix.with_suffix(".md").write_text(markdown(report))
    print(json.dumps({"status": report["status"], "attempts": len(report["attempts"]),
                      "selected_profiles": len(report["selected_attempts"]), "output": str(prefix)}))
    return 0 if report["status"] in ("FINAL_PASS", "IN_PROGRESS", "COMPLETE_WITH_FAILURES") else 1


if __name__ == "__main__":
    raise SystemExit(main())
