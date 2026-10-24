"""Run one complete GPQA Diamond evaluation through the installed bidevs CLI.

The caller owns the GPU service. This runner only targets port 6688 and preserves
CLI evidence plus the native predictions/reviews before normalizing the score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from scripts.bench_common import MODEL, ROOT, verify_endpoint
from scripts.benchmarks.profile_measure import ServiceObserver
from scripts.check_runtime import expected_sources, fingerprint

BASE_URL = "http://127.0.0.1:6688"
CASE_ID = "final_recommended_gpqa_diamond"
QUESTIONS = 198


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference(report: dict, path: Path, kind: str, **metadata) -> dict:
    row = {"path": str(path.resolve()), "sha256": sha256(path),
           "bytes": path.stat().st_size, "kind": kind, **metadata}
    report["rawrefs"].append(row)
    return row


def copy_evidence(report: dict, source: Path, destination: Path, kind: str) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = sha256(source)
    require(not destination.exists(), f"evidence destination already exists: {destination}")
    shutil.copyfile(source, destination)
    saved = reference(report, destination, kind, source_path=str(source), source_sha256=original)
    require(saved["sha256"] == original and sha256(source) == original,
            "GPQA source evidence changed during snapshot")
    return saved


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class Commands:
    """Capture every CLI invocation, including nonzero exits and launch errors."""

    def __init__(self, executable: str, workspace: Path, output: Path, report: dict):
        self.executable, self.workspace, self.output, self.report = executable, workspace, output, report
        (output / "commands").mkdir()

    def call(self, *parts: str, action: str | None = None, allow_failure: bool = False):
        number = len(self.report["commands"])
        prefix = self.output / "commands" / f"{number:02d}-{parts[0]}"
        argv = [self.executable]
        if parts != ("--version",):
            argv += ["--workspace", str(self.workspace), "--format", "json"]
        argv += list(parts)
        row = {"argv": argv, "returncode": None}
        self.report["commands"].append(row)
        started = time.perf_counter()
        stdout = stderr = ""
        try:
            result = subprocess.run(argv, capture_output=True, text=True, check=False)
            stdout, stderr = result.stdout, result.stderr
            row["returncode"] = result.returncode
        except OSError as error:
            stderr = f"{type(error).__name__}: {error}"
            raise
        finally:
            row["elapsed_s"] = time.perf_counter() - started
            for name, value in (("stdout", stdout), ("stderr", stderr)):
                path = prefix.with_suffix(f".{name}.txt")
                path.write_text(value, encoding="utf-8")
                row[name] = reference(self.report, path, f"bidevs-{name}")
            path = prefix.with_suffix(".command.json")
            write_json(path, row)
            reference(self.report, path, "bidevs-command")
        require(allow_failure or result.returncode == 0, f"bidevs {' '.join(parts)} exited {result.returncode}")
        if action is None:
            return stdout.strip()
        document = json.loads(stdout)
        require(isinstance(document, dict), f"bidevs {action} did not return a JSON object")
        require(document.get("schema_version") == "2.0" and document.get("action") == action,
                f"unsupported bidevs {action} response schema")
        return document


def jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"blank JSONL record at {path}:{number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"non-object JSONL record at {path}:{number}")
            rows.append(value)
    return rows


def validate_selection(document: dict, case_id: str, report: dict, output: Path) -> dict:
    require(document.get("status") == "found", "bidevs show did not resolve the requested case")
    cases = document.get("cases")
    require(isinstance(cases, list) and len(cases) == 1, "GPQA must resolve exactly one case")
    case = cases[0]
    require(case.get("name") == case_id and case.get("type") == "accuracy", "unexpected GPQA case identity/type")
    require(case.get("repetitions") == 1 and case.get("retries") == 0,
            "GPQA must use exactly one round without retries")
    require(not case.get("skip") and not case.get("prerequisites"), "GPQA cannot be skipped or depend on another case")
    phases, items = case.get("phases"), case.get("items")
    require(isinstance(phases, list) and len(phases) == 1 and phases[0].get("repetition") == 1,
            "GPQA must resolve exactly one phase/round")
    require(isinstance(items, list) and len(items) == 1 and items[0].get("label") == "gpqa_diamond",
            "GPQA must resolve exactly one gpqa_diamond item")
    require(case.get("case_acceptance") == [] and items[0].get("acceptance") == [],
            "GPQA selection must not add separate acceptance executions")
    for model in (case.get("model"), phases[0].get("model")):
        require(isinstance(model, dict) and model.get("name") == MODEL
                and model.get("api_base") == BASE_URL + "/v1",
                "GPQA must use the existing 1017 endpoint on port 6688")
    execution = items[0].get("execution") or {}
    require(execution.get("method") == "evalscope.accuracy", "GPQA requires native evalscope.accuracy")
    options = execution.get("options") or {}
    require(options.get("datasets") == "gpqa_diamond" and options.get("limit") == 0,
            "GPQA must evaluate the full gpqa_diamond dataset")
    require(options.get("backend", "native") == "native" and not options.get("judge"),
            "GPQA must use native scoring without an additional judge")
    require(not (options.get("client") or {}).get("args"), "GPQA cannot override native task configuration")
    expected_generation = {"temperature": 0.0, "top_p": 0.95, "max_tokens": 65536, "reasoning_effort": "max"}
    require(options.get("generation") == expected_generation and options.get("eval_batch_size") == 32,
            "GPQA generation settings must match the prepared batch-32 configuration")
    dataset_options = options.get("dataset_options") or {}
    require(set(dataset_options) == {"gpqa_diamond"}, "unexpected GPQA dataset options")
    local = dataset_options["gpqa_diamond"]
    require(isinstance(local, dict) and set(local) == {"local_path", "repeats"} and local["repeats"] == 1,
            "GPQA dataset must use one generation from the frozen local source")
    dataset = Path(local["local_path"]).resolve() / "train.jsonl"
    dataset_ref = copy_evidence(report, dataset, output / "inputs/gpqa-diamond-train.jsonl", "gpqa-input-dataset")
    rows = jsonl_rows(Path(dataset_ref["path"]))
    fields = {"Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"}
    require(len(rows) == QUESTIONS and all(fields <= row.keys() for row in rows),
            "GPQA source must contain all 198 complete question records")
    return {"case_id": case_id, "phase": phases[0]["label"],
            "phase_id": phases[0]["target_id"], "item": items[0]["label"],
            "dataset": dataset_ref, "generation": expected_generation, "eval_batch_size": 32}


def wait_completed(commands: Commands, run_id: str, *, poll_interval: float, max_wait: float) -> dict:
    deadline = time.monotonic() + max_wait
    while True:
        state = commands.call("status", run_id, action="status")
        require(state.get("run_id") == run_id, "bidevs status returned another run")
        if state.get("status") == "completed":
            return state
        require(state.get("status") == "running", f"GPQA run is not running/completed: {state.get('status')}")
        require(time.monotonic() < deadline, "GPQA status polling deadline expired; preserve the run ID for inspection")
        time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))


def complete_document(document: dict, run_id: str) -> None:
    require(document.get("run_id") == run_id and document.get("run_status") == "completed"
            and document.get("status") == "completed", "GPQA result/report is not a completed run")
    require(not document.get("incomplete_evidence") and not document.get("execution_failure"),
            "GPQA operational evidence is incomplete or failed")
    coverage = document.get("coverage") or {}
    require(coverage.get("completeness") == "complete" and coverage.get("expected_results") == 1
            and coverage.get("accounted_results") == 1 and coverage.get("missing_results") == [],
            "GPQA run coverage is incomplete or contains additional measurements")


def copy_native(report: dict, path: Path, output: Path, role: str) -> Path:
    destination = output / "native" / (role + ".jsonl")
    copy_evidence(report, path, destination, role)
    return destination


def collect_native_logs(document: dict, run_id: str, workspace: Path, output: Path, report: dict) -> None:
    """Snapshot evaluator logs even when the run has failed before scoring."""
    rows = (document.get("results") or {}).get("results")
    require(isinstance(rows, list), "bidevs result lacks rows for native log collection")
    run_root = (workspace / "logs" / run_id).resolve()
    for row in rows:
        require(row.get("case_id") == CASE_ID, "unexpected case while collecting GPQA logs")
        if not row.get("work_dir"):
            continue
        work = Path(row["work_dir"]).resolve()
        require(work.is_relative_to(run_root) and work.is_dir(), "native log work_dir is outside the current run")
        for path in sorted(work.rglob("*.log")):
            if path.is_file():
                copy_evidence(report, path, output / "native-logs" / path.relative_to(run_root), "native-log")


def final_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    require(isinstance(content, list), "prediction lacks native assistant content")
    parts = [part.get("text") for part in content if isinstance(part, dict) and part.get("type") == "text"]
    require(parts and all(isinstance(part, str) for part in parts), "prediction lacks final text parts")
    return "\n".join(parts)


def indexed(rows: list[dict], name: str) -> dict[int, dict]:
    require(len(rows) == QUESTIONS, f"{name} contains {len(rows)} records, expected 198")
    result = {}
    for row in rows:
        index = row.get("index")
        require(type(index) is int and index >= 0, f"{name} lacks a valid sample index")
        require(index not in result, f"duplicate {name} index {index}")
        result[index] = row
    return result


def normalize(result: dict, summary: dict, run_id: str, selection: dict,
              workspace: Path, output: Path, report: dict) -> dict:
    complete_document(result, run_id)
    complete_document(summary, run_id)
    rows = (result.get("results") or {}).get("results")
    require(isinstance(rows, list) and len(rows) == 1, "GPQA must return one result row")
    row = rows[0]
    require(row.get("case_id") == selection["case_id"] and row.get("case_item") == selection["item"]
            and row.get("phase") == selection["phase"] and row.get("phase_id") == selection["phase_id"]
            and row.get("round") == 1 and row.get("model") == MODEL,
            "GPQA result identity differs from the selected case")
    require(row.get("outcome_kind") == "test_result" and row.get("status") == "OK"
            and row.get("execution_failure") is None, "GPQA measurement did not complete successfully")
    require(summary.get("test_results") == rows, "bidevs result and report disagree")
    failures = summary.get("typed_failures")
    require(isinstance(failures, dict) and failures.get("execution_failures") == []
            and failures.get("invocation_failures") == [], "GPQA report contains typed failures")
    require(any(ref["kind"] == "native-log" for ref in report["rawrefs"]),
            "GPQA native evaluator logs are missing")
    metrics, metadata = row.get("metrics") or {}, row.get("metric_metadata") or {}
    candidates = [(key, info) for key, info in metadata.items()
                  if isinstance(info, dict) and info.get("dataset") == "gpqa_diamond"
                  and info.get("subset") is None and info.get("metric") in ("acc", "mean_acc", "accuracy", "score")
                  and key in metrics]
    require(bool(candidates), "GPQA dataset-level accuracy metadata is missing")
    priority = {"acc": 0, "mean_acc": 1, "accuracy": 2, "score": 3}
    key, info = min(candidates, key=lambda item: priority[item[1]["metric"]])
    counts = info.get("counts") or {}
    require(counts.get("planned") == QUESTIONS and counts.get("scored") == QUESTIONS,
            "GPQA must declare planned=198 and scored=198")
    require(all(counts.get(name) is None or (type(counts[name]) is int and counts[name] == 0)
                for name in ("failed", "skipped")),
            "GPQA declares failed or skipped samples")
    work = Path(row["work_dir"]).resolve()
    run_root = (workspace / "logs" / run_id).resolve()
    require(work.is_relative_to(run_root) and work.is_dir(), "GPQA work_dir is outside the current run")
    paths = {}
    for role in ("predictions", "reviews"):
        paths[role] = [file for directory in work.rglob(role) if directory.is_dir()
                       for file in directory.rglob("*.jsonl")]
        require(len(paths[role]) == 1 and paths[role][0].name == "gpqa_diamond_default.jsonl",
                f"expected one unambiguous GPQA default {role} file")
    predictions = indexed(jsonl_rows(copy_native(report, paths["predictions"][0], output, "predictions")), "predictions")
    reviews = indexed(jsonl_rows(copy_native(report, paths["reviews"][0], output, "reviews")), "reviews")
    require(predictions.keys() == reviews.keys(), "prediction/review sample identities differ")
    stops: dict[str, int] = {}
    for index, prediction in predictions.items():
        model_output = prediction.get("model_output")
        require(isinstance(model_output, dict) and not model_output.get("error"), f"prediction {index} has a model error")
        choices = model_output.get("choices")
        require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict),
                f"prediction {index} lacks one native completion")
        message = choices[0].get("message") or {}
        require(message.get("role") == "assistant" and bool(final_text(message.get("content")).strip()),
                f"prediction {index} lacks a final assistant answer")
        reason = choices[0].get("stop_reason")
        require(isinstance(reason, str) and bool(reason), f"prediction {index} lacks native stop_reason")
        stops[reason] = stops.get(reason, 0) + 1
    values = []
    for index, review in reviews.items():
        value = ((review.get("sample_score") or {}).get("score") or {}).get("value")
        require(isinstance(value, dict) and type(value.get("acc")) in (float, int)
                and value["acc"] in (0, 1), f"review {index} lacks a binary acc score")
        values.append(value["acc"])
    score = sum(values) / QUESTIONS
    backend_score = float(metrics[key])
    require(math.isfinite(backend_score), "GPQA backend accuracy is not finite")
    scale = info.get("scale")
    require(scale in ("fraction", "percentage", "unknown"), f"unsupported accuracy scale: {scale}")
    comparable = backend_score / 100 if scale == "percentage" else backend_score
    # Unknown backend units never determine our unit: normalized accuracy is
    # independently derived from the 198 binary acc reviews above.
    require(math.isclose(comparable, score, abs_tol=0.000051, rel_tol=0),
            "GPQA backend score disagrees with the complete binary review scores")
    return {"questions": QUESTIONS, "scored": len(reviews), "successful_requests": len(predictions),
            "failed_requests": 0, "failure_count_source": "198 unique error-free native predictions",
            "request_count_scope": "unique sample predictions; transport retry attempts are not counted",
            "successful_requests_scope": "logical samples with final error-free predictions, not HTTP attempts",
            "failed_requests_scope": "logical sample predictions, not HTTP attempts or retry failures",
            "backend_counts": counts, "correct": int(sum(values)), "score": score,
            "score_unit": "fraction", "score_source": "mean of 198 binary review acc values",
            "backend_metric": key, "backend_score": metrics[key], "backend_scale": scale,
            "stop_reasons": stops, "work_dir": str(work),
            "coverage": {"status": "complete", "planned": QUESTIONS, "predictions": len(predictions),
                         "reviews": len(reviews), "sample_indices_sha256": hashlib.sha256(json.dumps(sorted(predictions)).encode()).hexdigest()},
            "runtime_environment": (row.get("extra") or {}).get("runtime_environment")}


def run(workspace: Path, case_id: str, bidevs: str, launch_receipt: Path, output: Path,
        *, poll_interval: float = 5, max_wait: float = 10800) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "FAIL", "case_id": case_id, "questions": QUESTIONS,
              "scored": None, "successful_requests": None, "score": None,
              "validator": "scripts.acceptance.gpqa_bidevs:v1",
              "base_url": BASE_URL, "rawrefs": [], "commands": [], "error": None}
    started = time.perf_counter()
    endpoint = receipt = commands = observer = None
    observer_entered = False
    try:
        workspace = Path(workspace).resolve()
        require(workspace.is_dir() and case_id == CASE_ID, "use the prepared final_recommended_gpqa_diamond workspace/case")
        require(poll_interval >= 0 and max_wait > 0, "invalid status polling bounds")
        report["workspace"] = str(workspace)
        receipt = json.loads(Path(launch_receipt).read_text())
        write_json(output / "launch.json", receipt)
        reference(report, output / "launch.json", "launch-receipt")
        source_hash = fingerprint(expected_sources(ROOT))
        require(receipt.get("source_manifest_sha256") == source_hash, "launch receipt inference source differs from this checkout")
        report["source_manifest_sha256"] = source_hash
        copy_evidence(report, Path(__file__).resolve(), output / "sources/gpqa_bidevs.py", "wrapper-source")
        endpoint = verify_endpoint(BASE_URL, receipt)
        report["endpoint"] = endpoint
        commands = Commands(bidevs, workspace, output, report)
        version = commands.call("--version")
        require(version.startswith("bidevs ") and len(version.splitlines()) == 1, "unrecognized bidevs version output")
        report["bidevs_version"] = version
        lint = commands.call("config", "lint", action="config lint")
        require(lint.get("status") == "valid" and lint.get("issues") == [], "bidevs config lint failed")
        shown = commands.call("show", case_id, action="show")
        selection = validate_selection(shown, case_id, report, output)
        report["selection"] = selection
        configurations = sorted((workspace / "configs").rglob("*.yaml"))
        require(bool(configurations), "workspace contains no YAML configuration")
        for path in configurations:
            copy_evidence(report, path, output / "configs" / path.relative_to(workspace / "configs"), "bidevs-config")
        ready = commands.call("check", case_id, action="check")
        require(ready.get("status") == "ready" and ready.get("findings") == []
                and ready.get("dependency_profile") == "benchmark"
                and ready.get("selection", {}).get("cases") == [case_id], "bidevs GPQA preflight is not ready")
        require(all(sha256(Path(ref["path"])) == ref["sha256"] for ref in report["rawrefs"]),
                "GPQA inputs changed during preflight")
        require(all(sha256(Path(ref["source_path"])) == ref["source_sha256"]
                    for ref in report["rawrefs"] if "source_path" in ref), "GPQA source inputs changed during preflight")
        observer = ServiceObserver(BASE_URL, receipt["container_id"], output / "observation")
        observer.__enter__()
        observer_entered = True
        launched = commands.call("run", case_id, action="run", allow_failure=True)
        run_id = launched.get("run_id")
        require(isinstance(run_id, str) and bool(run_id) and Path(run_id).name == run_id
                and run_id not in (".", ".."), "bidevs run did not return a valid run ID")
        report["run_id"] = run_id
        require(report["commands"][-1]["returncode"] == 0 and launched.get("claimed") is True,
                "bidevs did not successfully claim and execute this GPQA run")
        require(launched.get("status") in ("running", "completed"), "unsupported or failed bidevs run state")
        report["terminal"] = wait_completed(commands, run_id, poll_interval=poll_interval, max_wait=max_wait)
        result = commands.call("result", run_id, action="result", allow_failure=True)
        result_exit = report["commands"][-1]["returncode"]
        collect_native_logs(result, run_id, workspace, output, report)
        summary = commands.call("report", run_id, action="report", allow_failure=True)
        require(result_exit == 0 and report["commands"][-1]["returncode"] == 0,
                "bidevs result/report query did not succeed")
        report.update(normalize(result, summary, run_id, selection, workspace, output, report))
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if commands is not None and report.get("run_id"):
            # Query partial/failed runs through public commands; never retry run.
            for action in ("result", "report"):
                if any(row["argv"][-2:] == [action, report["run_id"]] for row in report["commands"]):
                    continue
                try:
                    document = commands.call(action, report["run_id"], action=action, allow_failure=True)
                    if action == "result":
                        collect_native_logs(document, report["run_id"], workspace, output, report)
                except Exception as query_error:
                    report.setdefault("query_errors", []).append(f"{action}: {type(query_error).__name__}: {query_error}")
    finally:
        if observer is not None:
            try:
                if observer_entered:
                    observer.__exit__(None, None, None)
            except Exception as error:
                report.update(status="FAIL", observer_error=f"{type(error).__name__}: {error}")
            try:
                report["service_observation"] = observer.summary()
                require(observer_entered and not report.get("observer_error")
                        and report["service_observation"].get("status") == "PASS",
                        "GPQA service observer acquisition or identity validation failed")
            except Exception as error:
                report["status"] = "FAIL"
                report.setdefault("observer_error", f"{type(error).__name__}: {error}")
                report.setdefault("service_observation", {}).update(status="FAIL", wrapper_error=report["observer_error"])
                report["error"] = report["error"] or report["observer_error"]
            path = output / "service-observation.json"
            write_json(path, report["service_observation"])
            reference(report, path, "service-observation-summary")
            for path in sorted((output / "observation").rglob("*")):
                if path.is_file():
                    try:
                        reference(report, path, "service-observation")
                    except OSError as error:
                        report.update(status="FAIL", observer_artifact_error=f"{type(error).__name__}: {error}")
            observed = report.get("service_observation") or {}
            report["http_attempts"] = {
                "status": "NOT_COUNTED", "attempts": None, "retries": None, "failed_attempts": None,
                "window_started_at": observed.get("started_at"), "window_finished_at": observed.get("finished_at"),
                "endpoint": BASE_URL + "/v1/chat/completions",
                "evidence": [ref["path"] for ref in report["rawrefs"]
                             if ref["kind"] == "native-log" or Path(ref["path"]).name == "docker.log"],
                "scope": "Filter full service access logs to this observation window and endpoint; native evaluator logs preserve SDK retry evidence. Final predictions do not establish HTTP attempt counts.",
            }
        if endpoint is not None:
            try:
                require(verify_endpoint(BASE_URL, receipt) == endpoint, "GPU service identity changed during GPQA")
            except Exception as error:
                report.update(status="FAIL", endpoint_error=f"{type(error).__name__}: {error}")
        report["elapsed_s"] = time.perf_counter() - started
        write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--case", default=CASE_ID)
    parser.add_argument("--bidevs", default="/root/.local/bin/bidevs")
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.workspace, args.case, args.bidevs, args.launch_receipt, args.output)
    print(json.dumps({key: report.get(key) for key in ("status", "run_id", "questions", "scored", "successful_requests", "score", "error")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
