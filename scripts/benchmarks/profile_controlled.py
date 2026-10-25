"""Separate controlled follow-up: import valid cells without rewriting a FAIL.

Plan is offline. Execution never launches, stops or reconfigures a service.
A fresh replay preserves the failed warmup's output budget; a remaining matrix
run uses policy-v2 warmup and one formal repeat. Every sent body has a new salt
and an archived canonical body proving that no other request field changed.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts import bench_common
from scripts.benchmarks import profile_workload as inputs
from scripts.benchmarks.profile_matrix import normalized_command, validate_profile
from scripts.benchmarks.profile_measure import (
    ServiceObserver,
    engine_token_totals,
    measure,
    warmup,
)
from scripts.benchmarks.profile_plan import group_plan

MATRIX = {"matrix_single", "matrix_concurrent"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": inputs.sha256(path)}


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        return inputs.read_gzip(path)
    return json.loads(path.read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with path.open("xb") as stream:
            stream.write(gzip.compress(json.dumps(value, ensure_ascii=False).encode(), mtime=0))
    else:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")


def checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _without_salt(body):
    return {key: value for key, value in body.items() if key != "cache_salt"}


def salted_body(body, attempt_id, phase, cell_id, label, index):
    result = copy.deepcopy(body)
    result["cache_salt"] = "controlled:" + bench_common.workload_digest([attempt_id, phase, cell_id, label, index])
    require(result["cache_salt"] != body.get("cache_salt"), "new request must use a fresh salt")
    return result


def _canonical(frozen, cell, label, count, output_tokens=None):
    bodies = [inputs.request_body(frozen, cell, label, index) for index in range(count)]
    if output_tokens is not None:
        for body in bodies:
            body["max_tokens"] = output_tokens
    return bodies


def _catalog_cells(profile):
    return [c for c in group_plan(profile["context"], profile["chunk"], profile["memory"])["cells"]
            if c["case_id"] in MATRIX]


def _validate_pass(measurement, spec):
    rows = measurement.get("requests", [])
    require(measurement.get("status") == "PASS" and measurement.get("cache_verified") is True, "source measurement is not a cold PASS")
    require(len(rows) == spec["concurrency"] == measurement.get("request_count") == measurement.get("success"), "source request counts differ")
    require(not measurement.get("errors") and measurement.get("failure_rate") == 0, "source measurement contains failures")
    require(all(r.get("status") == "PASS" and type(r.get("cached_tokens")) is int and r["cached_tokens"] == 0
                and r.get("input_tokens") == spec["isl"] and r.get("output_tokens") == spec["osl"] for r in rows), "source request lengths/cache/status differ")
    elapsed = measurement["elapsed_s"]
    require(elapsed > 0, "source wall clock is invalid")
    totals = [sum(r[k] for r in rows) for k in ("input_tokens", "output_tokens")]
    for key, tokens in zip(("input_tps", "output_tps", "total_tps"), (*totals, sum(totals)), strict=True):
        require(math.isclose(measurement[key], tokens / elapsed, rel_tol=1e-9), "source throughput denominator differs")


def build_manifest(source_results: Path, inputs_path: Path):
    directory = Path(source_results).resolve()
    frozen = inputs.load(inputs_path)
    summary, plan, launch = (read(directory / name) for name in ("summary.json", "plan.json", "launch.json"))
    require(summary["status"] == "FAIL", "this follow-up must retain an actual failed source attempt")
    profile = summary["profile"]
    require(profile == plan["profile"] and profile["chunk"] == 32768, "source is not the controlled 32K profile")
    require(summary["frozen_inputs_sha256"] == frozen["sha256"], "source frozen inputs differ")
    require(summary["endpoint"]["image_id"] == launch["image_id"], "source image differs")
    require(summary["source"]["source_manifest_sha256"] == launch["source_manifest_sha256"]
            == frozen["source"]["source_manifest_sha256"], "source inference fingerprints differ")
    cells = [c for c in plan["cells"] if c["case_id"] in MATRIX]
    expected = {c["cell_id"]: c for c in _catalog_cells(profile)}
    specs = {c["cell_id"]: c for c in cells}
    require(len(specs) == len(cells) and set(specs) == set(expected), "source plan does not cover the complete matrix")
    identity = ("case_id", "kind", "dataset", "isl", "osl", "concurrency", "source_case")
    require(all(all(c.get(k) == expected[c["cell_id"]].get(k) for k in identity) for c in cells), "source matrix dimensions or inputs changed")
    require(all(c["status"] == "APPLICABLE" for c in cells), "controlled source contains inapplicable matrix cells")
    observed = {c["cell_id"]: c for c in summary["cells"]}
    require(len(observed) == len(summary["cells"]) and set(observed) <= set(specs), "duplicate or unknown source cells")
    retained = []
    failed = []
    for cell_id, row in observed.items():
        if row["status"] == "FAIL":
            failed.append(row)
            continue
        require(row["status"] == "PASS", "unexpected incomplete source cell")
        spec, folder = specs[cell_id], directory / "cells" / cell_id.replace("/", "__")
        measurements = row["measurements"]
        require(len(measurements) == spec["repeats"], "source PASS has incomplete original repetitions")
        require(row.get("warmup", {}).get("status") == "PASS" and row["warmup"].get("both_dp_active") is True, "source per-cell warmup is incomplete")
        rounds = []
        for repeat, measurement in enumerate(measurements, 1):
            where = folder / f"repeat-{repeat}"
            bodies = read(where / "request-payloads.json.gz")
            require(bodies == _canonical(frozen, spec, f"repeat-{repeat}", spec["concurrency"]), "source payload differs from canonical input")
            require(read(where / "report.json") == measurement, "source summary/raw report mismatch")
            require(read(where / "request-metrics.json.gz") == measurement["requests"], "source raw metrics mismatch")
            require(bench_common.workload_digest(bodies) == measurement["payload_sha256"], "source payload hash mismatch")
            _validate_pass(measurement, spec)
            rounds.append({"repeat": repeat, "report": ref(where / "report.json"),
                           "payloads": ref(where / "request-payloads.json.gz"), "metrics": ref(where / "request-metrics.json.gz")})
        retained.append({"cell_id": cell_id, "status": "REUSED_PASS", "primary_repeat": 1,
                         "source_refs": rounds[0], "supplemental_repeats": rounds[1:],
                         "warmup_ref": ref(folder / "warmup/report.json")})
    require(len(failed) == 1, "expected exactly one original warmup failure")
    failure = failed[0]
    require(not failure.get("measurements"), "failed warmup source must not contain a formal result")
    cell_id = failure["cell_id"]
    folder = directory / "cells" / cell_id.replace("/", "__") / "warmup"
    old_warmup = read(folder / "report.json")
    require(old_warmup == failure["warmup"] and old_warmup["status"] == "FAIL", "failed warmup report differs")
    bodies = read(folder / "round-1/request-payloads.json.gz")
    failed_report = read(folder / "round-1/report.json")
    require(failed_report == old_warmup["rounds"][0]["measurement"]
            and failed_report["status"] == "FAIL"
            and read(folder / "round-1/request-metrics.json.gz") == failed_report["requests"], "failed source raw reports disagree")
    require(bench_common.workload_digest(bodies) == failed_report["payload_sha256"], "failed source payload hash differs")
    require(old_warmup["concurrency"] == len(bodies) == 2 and old_warmup["osl"] == 128, "source failure is not the original C2/128 warmup")
    require(bodies == _canonical(frozen, specs[cell_id], f"{cell_id}-warmup-1", len(bodies), 128), "failed warmup input differs from original canonical bodies")
    pending = [c for c in cells if c["cell_id"] not in {r["cell_id"] for r in retained}]
    highest = max(c["isl"] for c in cells)
    pending.sort(key=lambda c: (c["cell_id"] == cell_id, c["isl"] == highest))
    terminal = {"service_log": ref(directory / "observation/docker.log"),
                "window_started_at": summary["started_at"], "window_finished_at": summary["finished_at"],
                "window_scope": "source_attempt"}
    environment = directory / "observation/environment-after.json"
    if environment.exists():
        terminal.update(container_inspect=ref(environment), container_inspect_format="observer_environment")
    manifest = {
        "schema_version": 1, "kind": "controlled_chunk_followup", "policy_version": 2,
        "profile_id": directory.parent.name, "profile": profile, "image_id": launch["image_id"],
        "source_fingerprint": launch["source_manifest_sha256"], "frozen_inputs_sha256": frozen["sha256"],
        "source_attempt": {"attempt_id": directory.parent.name, "status": summary["status"], "endpoint": summary["endpoint"]},
        "source_refs": {"summary": ref(directory / "summary.json"), "plan": ref(directory / "plan.json"),
                        "launch": ref(directory / "launch.json"), "inputs": ref(inputs_path)},
        "cells": copy.deepcopy(cells), "expected_cell_ids": [c["cell_id"] for c in cells],
        "retained": retained, "pending_cell_ids": [c["cell_id"] for c in pending],
        "failed_replay": {"cell_id": cell_id, "concurrency": len(bodies), "osl": 128,
                          "source_refs": {"warmup_report": ref(folder / "report.json"),
                                          "report": ref(folder / "round-1/report.json"),
                                          "payloads": ref(folder / "round-1/request-payloads.json.gz"),
                                          "metrics": ref(folder / "round-1/request-metrics.json.gz")},
                          "terminal_evidence": terminal},
        "counts": {"expected": len(cells), "retained_pass": len(retained), "pending": len(pending),
                   "source_failed_warmup": len(failed), "source_not_run": len(cells) - len(observed)},
        "ordering": "Unrun lower-input cells first, then highest-input nature, then the original failed random cell.",
    }
    manifest["sha256"] = bench_common.workload_digest(manifest)
    return manifest


def load_manifest(path):
    manifest = read(path)
    content = {k: v for k, v in manifest.items() if k != "sha256"}
    require(bench_common.workload_digest(content) == manifest["sha256"], "manifest digest mismatch")
    require(manifest["kind"] == "controlled_chunk_followup" and manifest["policy_version"] == 2, "unsupported controlled manifest")
    def verify(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                require(inputs.sha256(Path(value["path"])) == value["sha256"], "source evidence hash changed: " + value["path"])
            else:
                for item in value.values():
                    verify(item)
        elif isinstance(value, list):
            for item in value:
                verify(item)
    verify(content)
    retained = [r["cell_id"] for r in manifest["retained"]]
    pending = manifest["pending_cell_ids"]
    require(len(set(retained + pending)) == len(retained + pending)
            and set(retained + pending) == set(manifest["expected_cell_ids"]), "retained/pending coverage overlaps or is incomplete")
    return manifest


def archive_replay(directory, canonical, source_refs):
    """Bind the actual payloads emitted by measure() to their canonical bodies."""
    actual_path = directory / "request-payloads.json.gz"
    actual = read(actual_path)
    require(len(actual) == len(canonical), "replay request count changed")
    require(all(_without_salt(a) == _without_salt(c) and a.get("cache_salt") != c.get("cache_salt")
                for a, c in zip(actual, canonical, strict=True)), "replay changed something besides cache_salt")
    salts = [a.get("cache_salt") for a in actual]
    require(all(isinstance(s, str) and s for s in salts) and len(salts) == len(set(salts)), "replay salts are missing or duplicated")
    canonical_path = directory / "canonical-payloads.json.gz"
    write(canonical_path, canonical)
    provenance = {
        "schema_version": 1, "allowed_changes": ["cache_salt"],
        "canonical_payloads_ref": ref(canonical_path), "actual_payloads_ref": ref(actual_path),
        "canonical_payload_sha256": bench_common.workload_digest(canonical),
        "actual_payload_sha256": bench_common.workload_digest(actual),
        "content_without_salt_sha256": bench_common.workload_digest([_without_salt(c) for c in canonical]),
        "source_refs": source_refs,
        "per_request": [{"index": i, "canonical_sha256": bench_common.workload_digest(c),
                         "actual_sha256": bench_common.workload_digest(a),
                         "content_without_salt_sha256": bench_common.workload_digest(_without_salt(c))}
                        for i, (a, c) in enumerate(zip(actual, canonical, strict=True))],
    }
    write(directory / "replay-provenance.json", provenance)
    return provenance


def _warm(base, frozen, spec, observer, directory, attempt_id, phase, *, global_start=False, source_refs=None):
    selected = copy.deepcopy(spec)
    selected.update(warmup_rounds=2 if global_start else 1,
                    warmup_concurrency=256 if global_start else 2,
                    warmup_osl=128 if global_start else min(spec["osl"], 2),
                    warmup_max_additional_rounds=0 if global_start else 2)
    canonical = {}
    def factory(label, index):
        body = inputs.request_body(frozen, selected, label, index)
        body["max_tokens"] = selected["warmup_osl"]
        canonical.setdefault(label, []).append(body)
        return salted_body(body, attempt_id, phase, selected["cell_id"], label, index)
    result = warmup(base, factory, selected, observer, directory)
    for index, bodies in enumerate(canonical.values(), 1):
        where = directory / f"round-{index}"
        if (where / "request-payloads.json.gz").exists():
            archive_replay(where, bodies, source_refs or {})
    return result


def choose_cells(manifest, selected):
    if selected:
        require(len(selected) == len(set(selected)), "duplicate requested cell")
        require(set(selected) <= set(manifest["pending_cell_ids"]), "subset includes a retained or unknown cell")
    return [c for c in manifest["pending_cell_ids"] if not selected or c in selected]


def _health():
    with bench_common.request(inputs.BASE_URL, "/health", timeout=10) as response:
        require(response.status == 200, "controlled service is unhealthy")


def run(args):
    manifest = load_manifest(args.manifest)
    frozen = inputs.load(args.inputs)
    receipt = read(args.launch_receipt)
    profile = manifest["profile"]
    phase = args.phase.replace("-", "_")
    require(phase in ("fresh_replay", "remaining"), "unknown execution phase")
    require(args.attempt_id and args.attempt_id != manifest["source_attempt"]["attempt_id"], "use a new distinct attempt ID")
    require(frozen["sha256"] == manifest["frozen_inputs_sha256"], "controlled frozen inputs differ")
    validate_profile(receipt, profile["context"], profile["memory"], profile["chunk"])
    source_launch = read(manifest["source_refs"]["launch"]["path"])
    require(receipt["image_id"] == manifest["image_id"] and receipt["source_manifest_sha256"] == manifest["source_fingerprint"], "controlled runtime identity differs")
    require(normalized_command(receipt) == normalized_command(source_launch), "controlled launch configuration changed")
    endpoint = bench_common.verify_endpoint(inputs.BASE_URL, receipt)
    bench_common.require_model(inputs.BASE_URL)
    require(not (phase == "fresh_replay" and (args.cell or args.startup_proof)), "fresh replay cannot select cells or reuse startup")
    selected = [manifest["failed_replay"]["cell_id"]] if phase == "fresh_replay" else choose_cells(manifest, args.cell)
    require(selected, "selected subset is empty")
    specs = {c["cell_id"]: c for c in manifest["cells"]}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "launch.json", receipt)
    report = {"schema_version": 1, "kind": "controlled_chunk_followup", "phase": phase,
              "profile_id": manifest["profile_id"], "attempt_id": args.attempt_id, "policy_version": 2,
              "status": "IN_PROGRESS", "profile": profile, "endpoint": endpoint,
              "source": inputs.source_identity(), "frozen_inputs_sha256": frozen["sha256"],
              "manifest_ref": ref(args.manifest), "launch_ref": ref(output / "launch.json"),
              "source_refs": manifest["source_refs"], "started_at": now(),
              "scope": {"kind": "diagnostic" if phase == "fresh_replay" else "subset", "cell_ids": selected,
                        "expected_cell_ids": manifest["expected_cell_ids"],
                        "retained_cell_ids": [r["cell_id"] for r in manifest["retained"]]},
              "cells": [{"cell_id": cell_id, "case_id": specs[cell_id]["case_id"], "group": specs[cell_id]["group"],
                         "kind": specs[cell_id]["kind"], "status": "NOT_RUN", "stage": "not_run", "measurements": [],
                         "raw_directory": str(output / "cells" / cell_id.replace("/", "__")),
                         "started_at": None, "finished_at": None, "terminal_evidence": None}
                        for cell_id in selected]}
    observer = ServiceObserver(inputs.BASE_URL, receipt["container_id"], output / "observation")
    checkpoint(output / "progress.json", report)
    try:
        with observer:
            if args.startup_proof:
                proof = read(args.startup_proof)
                require(proof["status"] == "PASS" and proof["endpoint"] == endpoint
                        and proof["frozen_inputs_sha256"] == frozen["sha256"]
                        and proof["profile"] == profile
                        and proof["source"]["source_manifest_sha256"] == manifest["source_fingerprint"],
                        "startup proof is not for this running process")
                require(proof["global_warmup"]["status"] == "PASS" and proof["global_warmup"]["both_dp_active"] is True, "startup DP proof incomplete")
                report.update(global_warmup=proof["global_warmup"], startup_proof_ref=ref(args.startup_proof))
            else:
                require(endpoint != manifest["source_attempt"]["endpoint"], "new process required before startup warmup")
                totals = engine_token_totals(observer.snapshot())
                require(all(value == 0 for counters in totals.values() for value in counters.values()), "fresh process already handled inference requests")
                startup = {"cell_id": "startup-graphs", "case_id": "startup-graphs", "kind": "matrix",
                           "isl": 1024, "osl": 128, "concurrency": 256, "dataset": "random", "source_case": {"seed": 1017}}
                report["global_warmup"] = _warm(inputs.BASE_URL, frozen, startup, observer, output / "global-warmup",
                                                 args.attempt_id, phase, global_start=True, source_refs=manifest["source_refs"])
                require(report["global_warmup"]["status"] == "PASS", "global startup warmup failed")
            checkpoint(output / "progress.json", report)
            for row in report["cells"]:
                cell = copy.deepcopy(specs[row["cell_id"]])
                directory = Path(row["raw_directory"])
                row.update(status="IN_PROGRESS", stage="warmup", started_at=now(), source_refs=manifest["source_refs"])
                checkpoint(output / "progress.json", report)
                try:
                    if phase == "fresh_replay":
                        replay = manifest["failed_replay"]
                        canonical = read(replay["source_refs"]["payloads"]["path"])
                        cell.update(concurrency=replay["concurrency"], osl=replay["osl"],
                                    expected_input_by_index=[cell["isl"]] * replay["concurrency"])
                        row["source_refs"] = replay["source_refs"]
                    else:
                        row["warmup"] = _warm(inputs.BASE_URL, frozen, cell, observer, directory / "warmup", args.attempt_id, phase,
                                              source_refs=manifest["source_refs"])
                        require(row["warmup"]["status"] == "PASS", "v2 per-cell warmup failed")
                        canonical = _canonical(frozen, cell, "repeat-1", cell["concurrency"])
                    row["stage"] = "measurement"
                    actual = [salted_body(body, args.attempt_id, phase, cell["cell_id"], "repeat-1", i)
                              for i, body in enumerate(canonical)]
                    result = measure(inputs.BASE_URL, actual, cell, directory / "repeat-1", args.timeout)
                    row["measurements"].append(result)
                    row["replay_provenance"] = archive_replay(directory / "repeat-1", canonical, row["source_refs"])
                    require(result["status"] == "PASS", "controlled measurement failed")
                    _health()
                    require(bench_common.verify_endpoint(inputs.BASE_URL, receipt) == endpoint, "running process changed")
                    row["status"] = "PASS"
                except Exception as error:
                    row.update(status="FAIL", error=f"{type(error).__name__}: {error}")
                    row["finished_at"] = now()
                    write(directory / "failure.json", row)
                    report.update(status="FAIL", stopped_at=row["cell_id"])
                    break
                finally:
                    row["finished_at"] = row["finished_at"] or now()
                    checkpoint(output / "progress.json", report)
            if report["status"] != "FAIL":
                report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
    finally:
        report["finished_at"] = now()
        report["observation"] = observer.summary()
        if report["observation"]["status"] != "PASS":
            report["status"] = "FAIL"
        try:
            inspection = json.loads(subprocess.check_output(["docker", "inspect", receipt["container_id"]], text=True))
            write(output / "container-final.json", inspection)
            report["final_container_inspect_ref"] = ref(output / "container-final.json")
            report["final_container_inspect_format"] = "docker_inspect_list"
            require(inspection[0]["Id"] == endpoint["container_id"] and inspection[0]["Image"] == endpoint["image_id"], "final container identity differs")
            require(inspection[0]["State"]["StartedAt"] == endpoint["started_at"], "final container process restarted")
            if not inspection[0]["State"]["Running"]:
                report["status"] = "FAIL"
        except Exception as error:
            report.update(status="FAIL", final_inspect_error=f"{type(error).__name__}: {error}")
        log_path = output / "observation/docker.log"
        for row in report["cells"]:
            if row["status"] == "FAIL":
                row["terminal_evidence"] = {
                    "service_log": ref(log_path) if log_path.exists() else None,
                    "container_inspect": report.get("final_container_inspect_ref"),
                    "window_started_at": row["started_at"], "window_finished_at": row["finished_at"],
                    "window_scope": "cell", "container_inspect_format": "docker_inspect_list",
                }
        report["counts"] = {status: sum(row["status"] == status for row in report["cells"])
                            for status in ("PASS", "FAIL", "NOT_RUN")}
        report["formal_pass_count"] = report["counts"]["PASS"] if phase == "remaining" else 0
        write(output / "summary.json", report)
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "fresh-replay", "remaining"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-results", type=Path)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attempt-id")
    parser.add_argument("--startup-proof", type=Path)
    parser.add_argument("--cell", action="append")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    if args.phase == "plan":
        if args.source_results is None:
            parser.error("plan requires --source-results")
        manifest = build_manifest(args.source_results, args.inputs)
        write(args.manifest, manifest)
        print(json.dumps({"manifest": str(args.manifest), "counts": manifest["counts"]}))
        return 0
    if not args.launch_receipt or not args.output or not args.attempt_id:
        parser.error("execution requires --launch-receipt, --output and --attempt-id")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
