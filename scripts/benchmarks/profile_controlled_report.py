"""Offline assessment of the complete 32K controlled matrix, including failures.

A rejected candidate still needs a real outcome for every original matrix cell.
Diagnostic successes are retained separately; they never become performance PASS.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from scripts.benchmarks import profile_report as common
from scripts.benchmarks import profile_workload as workload

KIND = "controlled_chunk_assessment"
FOLLOWUP = "controlled_chunk_followup"
FATAL_PATTERN = r"OutOfMemoryError|EngineDeadError|\b(?:ERROR|FATAL)\b[^\n]*EngineCore[^\n]*(?:died|dead|fatal|exited unexpectedly)|Engine core initialization failed"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_ref(ref, base, evidence):
    path = evidence.rawrefs([ref], base)[0]
    return evidence.read(path), path


def file_ref(path):
    return {"path": str(path.resolve()), "sha256": common._sha(path)}


def verify_refs(value, base, evidence):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            evidence.rawrefs([value], base)
        else:
            for item in value.values():
                verify_refs(item, base, evidence)
    elif isinstance(value, list):
        for item in value:
            verify_refs(item, base, evidence)


def option(command, flag):
    require(command.count(flag) == 1, f"expected one {flag}")
    return command[command.index(flag) + 1]


def identity(summary, launch, profile, frozen_hash, expected_launch=None):
    require(summary["profile"] == profile, "controlled profile differs from the original")
    require(profile["chunk"] == 32768, "controlled assessment requires the 32K candidate")
    for axis, flag in (("context", "--max-model-len"), ("memory", "--gpu-memory-utilization"), ("chunk", "--max-num-batched-tokens")):
        require(float(option(launch["command"], flag)) == profile[axis], f"actual controlled {axis} differs")
    require(option(launch["command"], "-p") == "127.0.0.1:6688:8000", "controlled endpoint is not 6688")
    require(summary["endpoint"]["image_id"] == launch["image_id"]
            and summary["endpoint"]["container_id"] == launch["container_id"], "controlled endpoint/launch identity differs")
    require(summary["source"]["source_manifest_sha256"] == launch["source_manifest_sha256"], "controlled inference source differs")
    require(summary["frozen_inputs_sha256"] == frozen_hash, "controlled frozen inputs differ")
    normalized = common._normalized_flags(launch["command"], launch["image_id"])
    if expected_launch is not None:
        require(launch["image_id"] == expected_launch["image_id"]
                and launch["source_manifest_sha256"] == expected_launch["source_manifest_sha256"], "controlled image/source changed")
        require(normalized == common._normalized_flags(expected_launch["command"], expected_launch["image_id"]),
                "controlled fixed launch arguments changed")
    return normalized


def without_salt(bodies):
    return [{key: value for key, value in body.items() if key != "cache_salt"} for body in bodies]


def canonical_bodies(frozen, cell, label, count=None, osl=None):
    bodies = [workload.request_body(frozen, cell, label, index)
              for index in range(count if count is not None else cell["concurrency"])]
    if osl is not None:
        for body in bodies:
            body["max_tokens"] = osl
    return bodies


def measurement(folder, declared, canonical, evidence, *, replay):
    """Verify full raw records and, only locally, an explicit salt-only replay."""
    saved = evidence.read(folder / "report.json")
    require(saved == declared, "controlled summary/raw report mismatch")
    actual = evidence.read(folder / "request-payloads.json.gz")
    rows = evidence.read(folder / "request-metrics.json.gz")
    require(rows == saved.get("requests"), "controlled request metrics differ from the raw report")
    require(isinstance(actual, list) and len(actual) == len(canonical), "controlled request count changed")
    require(common._payload_digest(actual) == saved.get("payload_sha256"), "controlled actual payload digest mismatch")
    canonical_hash = common._payload_digest(canonical)
    if replay:
        stored = evidence.read(folder / "canonical-payloads.json.gz")
        proof = evidence.read(folder / "replay-provenance.json")
        require(proof.get("schema_version") == 1 and proof.get("allowed_changes") == ["cache_salt"], "unsupported replay change policy")
        require(stored == canonical, "replay canonical payload differs from frozen input")
        require(without_salt(actual) == without_salt(canonical), "replay changed more than cache_salt")
        require(all(body.get("cache_salt") != old.get("cache_salt")
                    for body, old in zip(actual, canonical, strict=True)), "replay did not change every cache salt")
        require(proof.get("canonical_payload_sha256") == canonical_hash
                and proof.get("actual_payload_sha256") == saved["payload_sha256"]
                and proof.get("content_without_salt_sha256") == common._payload_digest(without_salt(canonical)),
                "salt-only replay proof digest mismatch")
        for key, path in (("canonical_payloads_ref", folder / "canonical-payloads.json.gz"),
                          ("actual_payloads_ref", folder / "request-payloads.json.gz")):
            require(evidence.rawrefs([proof[key]], folder)[0] == path.resolve(), "replay proof references another payload")
    else:
        require(actual == canonical, "original controlled payload differs from frozen workload")
    salts = [body.get("cache_salt") for body in actual]
    require(all(isinstance(salt, str) and salt and len(salt) <= 128 and not any(char in salt for char in "@/\\\x00") for salt in salts)
            and len(set(salts)) == len(salts), "controlled cache salts are invalid or duplicated")
    return saved, {"canonical_payload_sha256": canonical_hash, "actual_payload_sha256": saved["payload_sha256"],
                   "salt_only_verified": replay, "raw_directory": str(folder.resolve())}


def instant(value):
    value = re.sub(r"(\.\d{6})\d+", r"\1", value.replace("Z", "+00:00"))
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "terminal evidence timestamp lacks a timezone")
    return result


def terminal(proof, cell, summary, launch, base, evidence, *, original=False):
    require(isinstance(proof, dict), "terminal failure lacks explicit evidence")
    start, end = instant(proof["window_started_at"]), instant(proof["window_finished_at"])
    require(start <= end, "terminal evidence window is reversed")
    outer_start, outer_end = instant(summary["started_at"]), instant(summary["finished_at"])
    require(outer_start <= start <= end <= outer_end, "terminal evidence lies outside its execution")
    if original and proof.get("window_scope") == "source_attempt":
        require(start == outer_start and end == outer_end, "source terminal window differs from the original attempt")
    else:
        require(start == instant(cell["started_at"]) and end == instant(cell["finished_at"]), "terminal window differs from its cell")
    log_path = evidence.rawrefs([proof["service_log"]], base)[0]
    lines = []
    with log_path.open(errors="replace") as stream:
        for line in stream:
            try:
                stamp = instant(line.split(" ", 1)[0])
            except (ValueError, IndexError):
                continue
            if start <= stamp <= end:
                lines.append(line.rstrip())
    text = "\n".join(lines)
    allocation = bool(re.search(r"\bOOM\b|out of memory", text, re.I))
    fatal = bool(re.search(FATAL_PATTERN, text, re.I))
    fatal = fatal or ("aten::new_empty" in text and bool(re.search(r"\bERROR\b|\bFATAL\b|Traceback", text)))
    exited = False
    if proof.get("container_inspect"):
        inspected, _ = read_ref(proof["container_inspect"], base, evidence)
        if proof.get("container_inspect_format") == "observer_environment":
            inspected = [inspected["docker_inspect"]]
        require(isinstance(inspected, list) and len(inspected) == 1, "terminal inspect is not one raw Docker inspect record")
        inspected = inspected[0]
        require(inspected["Id"] == launch["container_id"] and inspected["Image"] == launch["image_id"], "terminal container identity changed")
        command = launch["command"]
        require(inspected["Config"]["Cmd"] == command[command.index(launch["image_id"]) + 1:], "terminal container command changed")
        state = inspected["State"]
        require(state.get("StartedAt") == summary["endpoint"]["started_at"], "terminal container restarted")
        exited = (state.get("Running") is False and type(state.get("ExitCode")) is int and state["ExitCode"] != 0
                  and start <= instant(state["FinishedAt"]) <= end)
    else:
        require(summary.get("observation", {}).get("container_id") == launch["container_id"],
                "terminal log ownership is not bound to the launched container")
    require(allocation and (fatal or exited), "allocator warning alone is not a terminal OOM failure")
    signal_times = [instant(line.split(" ", 1)[0]) for line in lines
                    if re.search(FATAL_PATTERN + r"|aten::new_empty", line, re.I)]
    event_time = min(signal_times) if fatal and signal_times else instant(state["FinishedAt"])
    return {"status": "VERIFIED_TERMINAL_FAILURE", "window_started_at": proof["window_started_at"],
            "window_finished_at": proof["window_finished_at"], "allocation_context": allocation,
            "fatal_log_evidence": fatal, "container_nonzero_exit": exited,
            "terminal_event_time": event_time.isoformat(),
            "matching_log_lines": len(lines), "rawrefs": [ref for ref in (proof["service_log"], proof.get("container_inspect")) if ref]}


def warmup_failure(folder, declared, frozen, spec, evidence, *, replay):
    require(evidence.read(folder / "report.json") == declared and declared.get("status") == "FAIL", "warmup failure report differs")
    require(declared.get("concurrency") == 2, "matrix warmup failure must belong to the original C2 input shape")
    osl = declared["osl"]
    require(osl in (min(spec["osl"], 2), min(spec["osl"], 128)), "unexpected warmup output budget")
    rounds = declared.get("rounds", [])
    require(rounds, "no warmup requests were recorded")
    failed = []
    for index, row in enumerate(rounds, 1):
        require(row.get("round") == index and isinstance(row.get("measurement"), dict), "incomplete warmup request round")
        bodies = canonical_bodies(frozen, spec, f"{spec['cell_id']}-warmup-{index}", count=2, osl=osl)
        saved, _ = measurement(folder / f"round-{index}", row["measurement"], bodies, evidence, replay=replay)
        if saved["status"] == "FAIL":
            require(any(r.get("status") == "FAIL" for r in saved["requests"]), "warmup failure has no failed request")
            failed.append(index)
        else:
            common._measurement(saved, 2)
    require(failed, "warmup failure has no failed measurement round")
    return {"concurrency": 2, "output_tokens": osl, "failed_rounds": failed}


def warmup_success(folder, declared, frozen, spec, evidence, *, version, replay, global_scope=False):
    common._warmup(declared, version=version, osl=spec["osl"], global_scope=global_scope)
    require(evidence.read(folder / "report.json") == declared, "controlled raw warmup report differs")
    for number, row in enumerate(declared["rounds"], 1):
        canonical = canonical_bodies(frozen, spec, f"{spec['cell_id']}-warmup-{number}",
                                     count=declared["concurrency"], osl=declared["osl"])
        measurement(folder / f"round-{number}", row["measurement"], canonical, evidence, replay=replay)


def execution_errors(summary, terminal_facts):
    """Only service-read failures following a proven terminal event are explained."""
    observation = summary.get("observation", {})
    require(not observation.get("counter_resets"), "controlled observer counter reset is not attributable to OOM")
    errors = observation.get("observer_errors", [])
    if observation.get("status") == "PASS":
        require(not errors, "PASS observer contains acquisition errors")
    else:
        require(observation.get("status") == "FAIL" and terminal_facts and errors,
                "unattributed controlled observer failure")
    for error in errors:
        operation, detail = error.get("operation"), str(error.get("error", ""))
        require(operation in ("metrics", "final metrics", "final identity", "docker logs"),
                "unrelated controlled observer operation failed")
        pattern = (r"HTTP Error (?:500|503)|connection refused|connection reset|remote end closed|RemoteDisconnected|timed out|TimeoutError"
                   if "metrics" in operation else r"must be running|not running|running and publish"
                   if operation == "final identity" else r"log follower exited early")
        require(re.search(pattern, detail, re.I) is not None, "controlled observer error is not a terminal-service read failure")
        stamp = instant(error["time"])
        require(any(instant(fact["terminal_event_time"]) - timedelta(seconds=1) <= stamp <= instant(summary["finished_at"])
                    for fact in terminal_facts), "observer error predates the controlled terminal event")
    for key in ("error", "endpoint_error", "final_inspect_error"):
        if summary.get(key):
            require(terminal_facts and key != "final_inspect_error"
                    and re.search(r"service unhealthy|Container image/command/state differs from the launch receipt|not running", str(summary[key]), re.I),
                    "controlled execution contains an unrelated failure")
    require(not summary.get("traceback") or bool(summary.get("error")), "controlled execution has an unattributed traceback")


def add_performance(cell, spec, rounds, canonical_hash, actual_hash, attempt_id, version, warm):
    value = common._performance(spec, {"measurements": rounds})[spec["cell_id"]]
    signature = copy.deepcopy(value["comparison_signature"])
    signature["rounds"][0]["payload_sha256"] = canonical_hash
    value.update(canonical_comparison_signature=signature, actual_payload_sha256=actual_hash,
                 canonical_payload_sha256=canonical_hash, source_attempt_id=attempt_id, policy_version=version,
                 warmup_policy={"policy_version": version, "scope": "per_cell", "configured_rounds": warm.get("configured_rounds", 2),
                                "actual_rounds": len(warm.get("rounds", [])), "concurrency": warm.get("concurrency"), "output_tokens": warm.get("osl")})
    # The first eligible recorded repeat-1 remains primary; later executions are history.
    if cell["performance"] is None:
        cell["performance"] = value
    else:
        cell["additional_performance"].append(value)


def global_warmup(run, summary_path, launch, frozen, evidence, seen=()):
    require(summary_path not in seen, "cyclic startup warmup proof")
    if run.get("startup_proof_ref"):
        prior, path = read_ref(run["startup_proof_ref"], summary_path.parent, evidence)
        require(prior["status"] == "PASS" and prior["endpoint"] == run["endpoint"]
                and prior["profile"] == run["profile"] and prior["frozen_inputs_sha256"] == frozen["sha256"],
                "startup reuse proof differs from this process")
        require(prior["global_warmup"] == run["global_warmup"], "startup reuse report changed")
        prior_launch, _ = read_ref(prior["launch_ref"], path.parent, evidence)
        require(prior_launch == launch, "startup reuse launch changed")
        global_warmup(prior, path, prior_launch, frozen, evidence, (*seen, summary_path))
        return
    warm = run.get("global_warmup")
    spec = {"cell_id": "startup-graphs", "kind": "matrix", "isl": 1024, "osl": 128,
            "concurrency": 256, "dataset": "random", "source_case": {"seed": 1017}}
    warmup_success(summary_path.parent / "global-warmup", warm, frozen, spec, evidence, version=2, replay=True, global_scope=True)


def assess(manifest_path: Path, summaries: list[Path], *, evidence=None):
    evidence = evidence or common.Evidence()
    manifest_path = Path(manifest_path).resolve()
    report = {"schema_version": 1, "kind": KIND, "validation_errors": [], "status": "INVALID",
              "candidate_outcome": "UNDETERMINED", "coverage": {}, "cells": {}, "executions": [],
              "covered_failure_attempt_ids": [], "inputs": {"manifest": file_ref(manifest_path),
                                                            "summaries": [file_ref(Path(p)) for p in summaries]},
              "unresolved_failure_attempt_ids": []}
    try:
        manifest = evidence.read(manifest_path)
        require(manifest.get("schema_version") == 1 and manifest.get("kind") == FOLLOWUP, "unsupported controlled manifest")
        if "sha256" in manifest:
            require(common._payload_digest({k: v for k, v in manifest.items() if k != "sha256"}) == manifest["sha256"], "manifest internal digest mismatch")
        verify_refs(manifest, manifest_path.parent, evidence)
        refs = manifest["source_refs"]
        source, source_path = read_ref(refs["summary"], manifest_path.parent, evidence)
        plan, _ = read_ref(refs["plan"], manifest_path.parent, evidence)
        launch, _ = read_ref(refs["launch"], manifest_path.parent, evidence)
        frozen, _ = read_ref(refs["inputs"], manifest_path.parent, evidence)
        unsigned = {key: value for key, value in frozen.items() if key != "sha256"}
        require(frozen.get("sha256") == common._payload_digest(unsigned) == manifest["frozen_inputs_sha256"], "frozen input identity is invalid")
        profile = plan["profile"]
        require(manifest["profile"] == profile, "manifest changed the controlled profile")
        identity(source, launch, profile, frozen["sha256"])
        require(frozen["source"]["source_manifest_sha256"] == launch["source_manifest_sha256"], "frozen inference source differs")
        require(source["status"] == "FAIL" and manifest["source_attempt"]["status"] == "FAIL", "original controlled FAIL must be preserved")
        original_id = manifest["source_attempt"]["attempt_id"]
        require(source_path.parent.name == "results" and source_path.parent.parent.name == original_id,
                "original failure identity does not match its results directory")
        profile_id = manifest.get("profile_id", original_id)
        report.update(profile_id=profile_id, profile=profile, image_id=launch["image_id"],
                      source_fingerprint=launch["source_manifest_sha256"], frozen_inputs_sha256=frozen["sha256"],
                      normalized_flags=common._normalized_flags(launch["command"], launch["image_id"]), original_attempt_id=original_id)
        expected = [cell for cell in plan["cells"] if cell["case_id"] in common.MATRIX]
        ids = [cell["cell_id"] for cell in expected]
        require(len(ids) == len(set(ids)) == 42
                and all(c["status"] in common.ELIGIBLE and c["kind"] == "matrix" and c["group"] == "regular" for c in expected),
                "original controlled plan must contain all 42 applicable distinct matrix cells")
        require(manifest["expected_cell_ids"] == ids, "manifest cannot reduce or reorder the original matrix inventory")
        specs = {cell["cell_id"]: cell for cell in expected}
        if "cells" in manifest:
            require(manifest["cells"] == expected, "manifest cell specifications differ from the original plan")
        report["cells"] = {ident: {"outcome": "NOT_RUN", "performance": None, "additional_performance": [],
                                    "failures": [], "diagnostics": [], "history": []} for ident in ids}
        recorded = {cell["cell_id"]: cell for cell in source["cells"]}
        require(len(recorded) == len(source["cells"]) and set(recorded) <= set(specs), "original matrix contains duplicate or extra cells")
        passed = {key for key, value in recorded.items() if value["status"] == "PASS"}
        retained = manifest["retained"]
        require(len(retained) == len({cell["cell_id"] for cell in retained})
                and {cell["cell_id"] for cell in retained} == passed, "retained cells omit or fabricate original PASS measurements")
        require(set(manifest["pending_cell_ids"]) == set(ids) - passed
                and len(manifest["pending_cell_ids"]) == len(set(ids) - passed), "pending scope omits an unmeasured controlled cell")
        source_version = common._policy_version(plan, source)
        common._warmup(source.get("global_warmup"), version=source_version, osl=128, global_scope=True)
        for keep in retained:
            ident = keep["cell_id"]
            spec = specs[ident]
            row = recorded[ident]
            require(keep.get("primary_repeat") == 1, "original primary must remain repeat-1")
            rounds = row["measurements"]
            require(len(rounds) == spec["repeats"] and all(r["status"] == "PASS" for r in rounds), "original PASS repetitions are incomplete")
            verify_refs(keep["source_refs"], manifest_path.parent, evidence)
            folder = source_path.parent / "cells" / ident.replace("/", "__")
            warmup_success(folder / "warmup", row["warmup"], frozen, spec, evidence, version=source_version, replay=False)
            declared_rounds = [keep["source_refs"], *keep.get("supplemental_repeats", [])]
            require(len(declared_rounds) == len(rounds), "retained original references omit repetitions")
            for number, reference in enumerate(declared_rounds, 1):
                require(reference.get("repeat") == number, "retained reference round identity differs")
                for key, filename in (("report", "report.json"), ("payloads", "request-payloads.json.gz"), ("metrics", "request-metrics.json.gz")):
                    require(evidence.rawrefs([reference[key]], manifest_path.parent)[0] == (folder / f"repeat-{number}" / filename).resolve(),
                            "retained reference points to another measurement")
            hashes = []
            for number, value in enumerate(rounds, 1):
                _, proof = measurement(folder / f"repeat-{number}", value, canonical_bodies(frozen, spec, f"repeat-{number}"), evidence, replay=False)
                hashes.append(proof)
            add_performance(report["cells"][ident], spec, rounds, hashes[0]["canonical_payload_sha256"],
                            hashes[0]["actual_payload_sha256"], original_id, source_version, row["warmup"])
            report["cells"][ident]["history"].append({"attempt_id": original_id, "raw_status": "PASS", "origin": "retained_original"})
        failed_ids = [key for key, value in recorded.items() if value["status"] == "FAIL"]
        require(len(failed_ids) == 1 and manifest["failed_replay"]["cell_id"] == failed_ids[0], "original failure scope differs from the replay definition")
        failed_id = failed_ids[0]
        source_failure = manifest["failed_replay"]
        source_cell = recorded[failed_id]
        require(not source_cell.get("measurements"), "original warmup failure contains a formal performance result")
        failed_folder = source_path.parent / "cells" / failed_id.replace("/", "__")
        evidence.rawrefs(list(source_failure["source_refs"].values()), manifest_path.parent)
        warm = warmup_failure(failed_folder / "warmup", source_cell["warmup"], frozen, specs[failed_id], evidence, replay=False)
        report["cells"][failed_id]["history"].append({"attempt_id": original_id, "raw_status": "FAIL", "stage": "warmup", "warmup": warm})
        source_terminal = source_failure.get("terminal_evidence")
        if source_terminal:
            verified = terminal(source_terminal, source_cell, source, launch, manifest_path.parent, evidence, original=True)
            report["cells"][failed_id]["failures"].append({"attempt_id": original_id, "stage": "warmup", "terminal": verified})
            report["covered_failure_attempt_ids"].append(original_id)
            execution_errors(source, [verified])
        else:
            report["unresolved_failure_attempt_ids"].append(original_id)
        executions = set()
        for summary_path in summaries:
            summary_path = Path(summary_path).resolve()
            run = evidence.read(summary_path)
            require(run.get("schema_version") == 1 and run.get("kind") == FOLLOWUP and run.get("policy_version") == 2,
                    "unsupported controlled followup schema/policy")
            require(run["profile_id"] == profile_id, "followup belongs to another controlled profile")
            attempt = run["attempt_id"]
            require(attempt not in executions and attempt != original_id, "duplicate controlled execution")
            executions.add(attempt)
            require(evidence.rawrefs([run["manifest_ref"]], summary_path.parent)[0] == manifest_path, "followup uses another manifest")
            actual_launch, _ = read_ref(run["launch_ref"], summary_path.parent, evidence)
            identity(run, actual_launch, profile, frozen["sha256"], launch)
            require(run["status"] in ("PASS", "FAIL"), "followup is not terminal")
            if run.get("phase") == "fresh_replay":
                require(not run.get("startup_proof_ref"), "fresh replay cannot inherit startup warmup")
                require(run["endpoint"] != source["endpoint"], "fresh replay reused the original service process")
                before = run.get("global_warmup", {}).get("dp_before", {})
                require(len(before) == 2 and all(counters.get(key) == 0 for counters in before.values()
                                                for key in ("prompt_tokens", "generation_tokens")),
                        "fresh replay service already handled inference requests")
            global_warmup(run, summary_path, actual_launch, frozen, evidence)
            inspected, _ = read_ref(run["final_container_inspect_ref"], summary_path.parent, evidence)
            require(isinstance(inspected, list) and len(inspected) == 1 and inspected[0]["Id"] == actual_launch["container_id"]
                    and inspected[0]["Image"] == actual_launch["image_id"]
                    and inspected[0]["State"]["StartedAt"] == run["endpoint"]["started_at"], "followup final identity differs")
            scope = run["scope"]
            require(scope["kind"] in ("subset", "diagnostic") and scope["expected_cell_ids"] == ids
                    and set(scope["retained_cell_ids"]) == passed, "followup scope changed the original inventory")
            selected = scope["cell_ids"]
            require(selected and len(selected) == len(set(selected)) and set(selected) <= set(ids), "invalid controlled subset selection")
            require((scope["kind"] == "diagnostic") == (run["phase"] == "fresh_replay"), "diagnostic/measurement phase mismatch")
            actual = {cell["cell_id"]: cell for cell in run["cells"]}
            require(len(actual) == len(run["cells"]) and set(actual) <= set(selected), "followup contains duplicate or unselected cells")
            if run["status"] == "PASS":
                require(set(actual) == set(selected) and all(row["status"] == "PASS" for row in actual.values())
                        and inspected[0]["State"]["Running"] is True and run.get("observation", {}).get("status") == "PASS",
                        "PASS followup has missing cells or invalid service observation")
            execution_failures = []
            execution_terminals = []
            for ident, row in actual.items():
                spec = specs[ident]
                cell = report["cells"][ident]
                require(row["status"] in ("PASS", "FAIL", "NOT_RUN"), "unknown controlled cell state")
                if row["status"] == "NOT_RUN":
                    require(not row.get("measurements") and row["stage"] == "not_run", "NOT_RUN contains a fabricated measurement")
                    continue
                folder = Path(row["raw_directory"])
                folder = folder if folder.is_absolute() else summary_path.parent / folder
                require(folder.resolve().is_relative_to(summary_path.parent), "controlled raw directory escapes its execution")
                diagnostic = scope["kind"] == "diagnostic"
                if diagnostic:
                    require(ident == failed_id and len(row.get("measurements", [])) == 1, "fresh diagnostic does not replay the original failed cell")
                    canonical = evidence.read(failed_folder / "warmup/round-1/request-payloads.json.gz")
                    saved, proof = measurement(folder / "repeat-1", row["measurements"][0], canonical, evidence, replay=True)
                    if row["status"] == "PASS":
                        common._measurement(saved, 2)
                        require(all(r["input_tokens"] == spec["isl"] and r["output_tokens"] == warm["output_tokens"] for r in saved["requests"]), "diagnostic request lengths differ")
                        cell["diagnostics"].append({"attempt_id": attempt, "status": "PASS", "scope": "C2 warmup replay only", "proof": proof})
                        continue
                    require(saved["status"] == "FAIL" and any(r["status"] == "FAIL" for r in saved["requests"]), "diagnostic failure lacks failed requests")
                    stage = "warmup_replay"
                elif row["stage"] == "warmup" and row["status"] == "FAIL":
                    warmup_failure(folder / "warmup", row["warmup"], frozen, spec, evidence, replay=True)
                    stage = "warmup"
                else:
                    require(row["stage"] == "measurement" and len(row.get("measurements", [])) == 1, "formal controlled cell requires one measurement")
                    warmup_success(folder / "warmup", row["warmup"], frozen, spec, evidence, version=2, replay=True)
                    saved, proof = measurement(folder / "repeat-1", row["measurements"][0], canonical_bodies(frozen, spec, "repeat-1"), evidence, replay=True)
                    if row["status"] == "PASS":
                        common._measurement(saved, spec["concurrency"])
                        add_performance(cell, spec, [saved], proof["canonical_payload_sha256"], proof["actual_payload_sha256"], attempt, 2, row["warmup"])
                        cell["history"].append({"attempt_id": attempt, "raw_status": "PASS", "origin": "new_formal_measurement", "proof": proof})
                        continue
                    require(saved["status"] == "FAIL" and any(r["status"] == "FAIL" for r in saved["requests"]), "formal failure has no failed request")
                    stage = "measurement"
                verified = terminal(row.get("terminal_evidence"), row, run, actual_launch, summary_path.parent, evidence)
                cell["failures"].append({"attempt_id": attempt, "stage": stage, "terminal": verified})
                execution_terminals.append(verified)
                cell["history"].append({"attempt_id": attempt, "raw_status": "FAIL", "stage": stage})
                execution_failures.append(ident)
            if any(row["status"] == "FAIL" for row in run["cells"]):
                require(set(execution_failures) == {row["cell_id"] for row in run["cells"] if row["status"] == "FAIL"}, "unassessed followup failure remains")
                report["covered_failure_attempt_ids"].append(attempt)
            else:
                require(run["status"] == "PASS", "followup contains an unattributed execution/observation failure")
            execution_errors(run, execution_terminals)
            report["executions"].append({"attempt_id": attempt, "phase": run["phase"], "raw_status": run["status"], "scope": scope})
        measured = failed = overlap = 0
        for cell in report["cells"].values():
            has_measurement, has_failure = cell["performance"] is not None, bool(cell["failures"])
            measured += has_measurement
            failed += has_failure
            overlap += has_measurement and has_failure
            failure_kind = "FAIL_MEASUREMENT" if any(row["stage"] == "measurement" for row in cell["failures"]) else "FAIL_WARMUP"
            cell["outcome"] = ("PASS_WITH_FAILURE_HISTORY" if has_failure else "PASS") if has_measurement else (failure_kind if has_failure else "NOT_RUN")
        covered = measured + failed - overlap
        report["coverage"] = {"expected": 42, "covered": covered, "measured_pass": measured,
                              "terminal_failed": failed, "pass_with_failure_history": overlap, "not_run": 42 - covered,
                              "missing_cell_ids": [key for key, value in report["cells"].items() if value["outcome"] == "NOT_RUN"]}
        report["candidate_outcome"] = "REJECTED" if failed else "UNDETERMINED"
        report["status"] = "COMPLETE_WITH_FAILURES" if covered == 42 and failed and not report["unresolved_failure_attempt_ids"] else "INCOMPLETE_WITH_FAILURES"
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError) as error:
        report["validation_errors"].append(f"{type(error).__name__}: {error}")
    report["raw_evidence"] = evidence.files
    return report


def validate_receipt(path, evidence):
    path = Path(path).resolve()
    stored = evidence.read(path)
    require(stored.get("kind") == KIND and stored.get("schema_version") == 1, "unsupported controlled assessment receipt")
    manifest = evidence.rawrefs([stored["inputs"]["manifest"]], path.parent)[0]
    summaries = [evidence.rawrefs([ref], path.parent)[0] for ref in stored["inputs"]["summaries"]]
    actual = assess(manifest, summaries, evidence=evidence)
    for key in ("status", "candidate_outcome", "coverage", "profile_id", "validation_errors"):
        require(stored.get(key) == actual.get(key), f"saved controlled {key} differs from verified evidence")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "assessment output already exists")
    value = assess(args.manifest, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value[key] for key in ("status", "candidate_outcome", "coverage", "validation_errors")}))
    return 1 if value["validation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
