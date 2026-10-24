"""Synthetic saved bidevs/EvalScope schemas exercise one-shot GPQA evidence."""

import copy
import gzip
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.acceptance import gpqa_bidevs as runner


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class GpqaBidevsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workspace = self.root / "workspace"
        (self.workspace / "configs").mkdir(parents=True)
        (self.workspace / "configs/cases.yaml").write_text("cases: []\n")
        self.dataset = self.root / "gpqa-data/train.jsonl"
        write_jsonl(self.dataset, [{"Question": f"Question {index}", "Correct Answer": "A",
                                   "Incorrect Answer 1": "B", "Incorrect Answer 2": "C", "Incorrect Answer 3": "D"}
                                  for index in range(198)])
        self.receipt = self.root / "launch.json"
        self.receipt.write_text(json.dumps({"source_manifest_sha256": "source", "container_id": "container"}))
        self.run_id = "20261025T000000Z_run1"
        self.work = self.workspace / "logs" / self.run_id / "model/case/single_gpqa_diamond"
        self.prediction_path = self.work / "20261025_000000/predictions/model/gpqa_diamond_default.jsonl"
        self.review_path = self.work / "20261025_000000/reviews/model/gpqa_diamond_default.jsonl"
        self.predictions = [{"index": index, "model": runner.MODEL, "model_output": {
            "error": None, "choices": [{"stop_reason": "stop", "message": {
                "role": "assistant", "content": [{"type": "reasoning", "reasoning": "reason"},
                                                  {"type": "text", "text": "Answer: A"}],
            }}],
        }} for index in range(198)]
        self.reviews = [{"index": index, "sample_score": {"score": {"value": {"acc": float(index < 190)}}}}
                        for index in range(198)]
        write_jsonl(self.prediction_path, self.predictions)
        write_jsonl(self.review_path, self.reviews)
        self.native_log = self.work / "20261025_000000/logs/eval_log.log"
        self.native_log.parent.mkdir(parents=True)
        self.native_log.write_text("SDK retry diagnostic retained verbatim\n")
        model = {"name": runner.MODEL, "api_base": runner.BASE_URL + "/v1"}
        self.case = {
            "name": runner.CASE_ID, "type": "accuracy", "repetitions": 1, "retries": 0,
            "skip": {}, "prerequisites": [], "model": model, "case_acceptance": [],
            "phases": [{"label": "single", "target_id": "single", "repetition": 1, "model": model}],
            "items": [{"label": "gpqa_diamond", "acceptance": [], "execution": {"method": "evalscope.accuracy", "options": {
                "datasets": "gpqa_diamond", "limit": 0, "eval_batch_size": 32,
                "generation": {"temperature": 0.0, "top_p": 0.95, "max_tokens": 65536, "reasoning_effort": "max"},
                "dataset_options": {"gpqa_diamond": {"local_path": str(self.dataset.parent), "repeats": 1}},
                "qa_detail": True, "client": {"api_key": "EMPTY"},
            }}}],
        }
        self.row = {
            "case_id": runner.CASE_ID, "case_item": "gpqa_diamond", "phase": "single", "phase_id": "single",
            "round": 1, "model": runner.MODEL, "outcome_kind": "test_result", "status": "OK",
            "execution_failure": None, "work_dir": str(self.work),
            "metrics": {"gpqa_diamond:acc": "0.9596"},
            "metric_metadata": {"gpqa_diamond:acc": {"dataset": "gpqa_diamond", "subset": None,
                "metric": "acc", "scale": "unknown", "counts": {"planned": 198, "scored": 198, "failed": None, "skipped": None}}},
        }
        self.coverage = {"completeness": "complete", "expected_results": 1, "accounted_results": 1, "missing_results": []}
        self.states = ["completed"]
        self.launch_status = "completed"
        self.run_returncode = 0
        self.query_returncode = 0
        self.calls = []
        self.override = {}
        self.observers = []
        self.observer_status = "PASS"
        self.observer_close_error = False
        self.observer_enter_error = False
        for name, value in (("expected_sources", {}), ("fingerprint", "source"),
                            ("verify_endpoint", {"container_id": "same"})):
            mocked = patch.object(runner, name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)
        observer = patch.object(runner, "ServiceObserver", side_effect=self.make_observer)
        observer.start()
        self.addCleanup(observer.stop)

    def make_observer(self, base, container, output):
        self.assertEqual((base, container), (runner.BASE_URL, "container"))
        owner = self

        class Observer:
            entered = closed = False

            def __enter__(self):
                output.mkdir()
                (output / "docker.log").write_text("full service access log\n")
                for name in ("gpu", "metrics"):
                    (output / (name + ".jsonl.gz")).write_bytes(gzip.compress(b'{"sample":1}\n'))
                if owner.observer_enter_error:
                    raise OSError("collector startup failed")
                self.entered = True
                return self

            def __exit__(self, *args):
                self.closed = True
                if owner.observer_close_error:
                    raise OSError("collector shutdown failed")
                (output / "observer-summary.json").write_text(json.dumps(self.summary()))

            def summary(self):
                return {"status": owner.observer_status, "started_at": "2026-10-25T12:00:00+00:00",
                        "finished_at": "2026-10-25T12:30:00+00:00" if self.closed else None}

        instance = Observer()
        self.observers.append(instance)
        return instance

    def cli(self, argv, **kwargs):
        parts = argv[1:] if argv[1] == "--version" else argv[5:]
        action = " ".join(parts[:2]) if parts[0] == "config" else parts[0]
        self.calls.append(parts)
        if action == "--version":
            stdout, code = "bidevs 0.3.0\n", 0
        else:
            value = {"schema_version": "2.0", "action": action}
            if action == "config lint":
                value.update(status="valid", issues=[], changes=[])
            elif action == "show":
                value.update(status="found", cases=[self.case])
            elif action == "check":
                value.update(status="ready", findings=[], dependency_profile="benchmark", selection={"cases": [runner.CASE_ID]})
            elif action == "run":
                self.assertTrue(self.observers[-1].entered)
                self.assertFalse(self.observers[-1].closed)
                value.update(run_id=self.run_id, claimed=True, status=self.launch_status)
            elif action == "status":
                value.update(run_id=self.run_id, status=self.states.pop(0))
            elif action in ("result", "report"):
                value.update(run_id=self.run_id, status="completed", run_status="completed", coverage=self.coverage)
                if action == "result":
                    value["results"] = {"schema_version": "2.0", "run_id": self.run_id, "results": [self.row]}
                else:
                    value.update(test_results=[self.row], typed_failures={"execution_failures": [], "invocation_failures": []})
            else:
                raise AssertionError(parts)
            value.update(self.override.get(action, {}))
            stdout = json.dumps(value)
            code = self.run_returncode if action == "run" else self.query_returncode if action in ("result", "report") else 0
        return subprocess.CompletedProcess(argv, code, stdout, "diagnostic stderr\n")

    def execute(self, name="output"):
        with patch.object(runner.subprocess, "run", side_effect=self.cli), patch.object(runner.time, "sleep"):
            return runner.run(self.workspace, runner.CASE_ID, "/installed/bidevs", self.receipt,
                              self.root / name, poll_interval=0)

    def test_complete_single_round_derives_success_count_from_native_evidence(self):
        report = self.execute()
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual((report["questions"], report["scored"], report["successful_requests"]), (198, 198, 198))
        self.assertEqual(report["failed_requests"], 0)
        self.assertIsNone(report["backend_counts"]["failed"])
        self.assertEqual(report["correct"], 190)
        self.assertAlmostEqual(report["score"], 190 / 198)
        self.assertEqual(report["backend_scale"], "unknown")
        self.assertEqual(report["validator"], "scripts.acceptance.gpqa_bidevs:v1")
        self.assertEqual(report["coverage"]["status"], "complete")
        self.assertEqual(report["service_observation"]["status"], "PASS")
        self.assertTrue(self.observers[0].closed)
        self.assertIsNone(report["http_attempts"]["retries"])
        self.assertIsNone(report["http_attempts"]["failed_attempts"])
        self.assertEqual(report["http_attempts"]["status"], "NOT_COUNTED")
        self.assertIn("logical", report["successful_requests_scope"])
        self.assertTrue(any(ref["kind"] == "native-log" for ref in report["rawrefs"]))
        self.assertTrue(any(Path(path).name == "docker.log" for path in report["http_attempts"]["evidence"]))
        self.assertEqual(report["stop_reasons"], {"stop": 198})
        self.assertEqual(sum(parts[0] == "run" for parts in self.calls), 1)
        self.assertNotIn(["--stream"], self.calls)
        for ref in report["rawrefs"]:
            self.assertTrue(Path(ref["path"]).is_relative_to((self.root / "output").resolve()))
            self.assertEqual(hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest(), ref["sha256"])
        saved = json.loads((self.root / "output/report.json").read_text())
        self.assertEqual(saved["status"], "PASS")
        self.assertTrue((self.root / "output/native/predictions.jsonl").exists())
        self.assertTrue((self.root / "output/native/reviews.jsonl").exists())

    def test_async_status_is_observed_without_launching_a_second_run(self):
        self.launch_status = "running"
        self.states = ["running", "completed"]
        report = self.execute()
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(sum(parts[0] == "run" for parts in self.calls), 1)
        self.assertEqual(sum(parts[0] == "status" for parts in self.calls), 2)

    def test_unknown_async_state_fails_and_preserves_public_diagnostic_queries(self):
        self.launch_status = "running"
        self.states = ["queued"]
        report = self.execute()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("not running/completed", report["error"])
        self.assertEqual(sum(parts[0] == "run" for parts in self.calls), 1)
        self.assertIn(["result", self.run_id], self.calls)
        self.assertIn(["report", self.run_id], self.calls)
        self.assertIsNone(report["successful_requests"])

    def test_nonzero_run_exit_is_not_a_score_or_an_automatic_retry(self):
        self.run_returncode = 1
        report = self.execute()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["run_id"], self.run_id)
        self.assertEqual(sum(parts[0] == "run" for parts in self.calls), 1)
        self.assertIn(["report", self.run_id], self.calls)
        self.assertTrue(self.observers[0].closed)
        self.assertTrue(any(ref["kind"] == "native-log" for ref in report["rawrefs"]))

    def test_preflight_wrong_endpoint_rounds_or_dataset_never_runs(self):
        original = copy.deepcopy(self.case)
        for index, problem in enumerate(("endpoint", "rounds", "item", "source_count")):
            self.case = copy.deepcopy(original)
            if problem == "endpoint":
                self.case["model"]["api_base"] = "http://127.0.0.1:6789/v1"
            elif problem == "rounds":
                self.case["repetitions"] = 4
            elif problem == "item":
                self.case["items"] *= 2
            else:
                write_jsonl(self.dataset, [{"Question": "incomplete"}])
            self.calls = []
            report = self.execute(f"preflight-{index}")
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(any(parts[0] == "run" for parts in self.calls))

    def test_source_mismatch_is_rejected_before_cli_or_gpu_requests(self):
        self.receipt.write_text(json.dumps({"source_manifest_sha256": "wrong"}))
        report = self.execute()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("source differs", report["error"])
        self.assertEqual(self.calls, [])

    def test_missing_duplicate_or_error_predictions_do_not_become_198_successes(self):
        for index, problem in enumerate(("missing", "duplicate", "error", "empty")):
            predictions = copy.deepcopy(self.predictions)
            if problem == "missing":
                predictions.pop()
            elif problem == "duplicate":
                predictions[-1]["index"] = 0
            elif problem == "error":
                predictions[-1]["model_output"]["error"] = "API failure"
            else:
                predictions[-1]["model_output"]["choices"] = []
            write_jsonl(self.prediction_path, predictions)
            self.states = ["completed"]
            report = self.execute(f"bad-predictions-{index}")
            self.assertEqual(report["status"], "FAIL", report)
            self.assertIsNone(report["successful_requests"])
            self.assertTrue((self.root / f"bad-predictions-{index}/native/predictions.jsonl").exists())

    def test_missing_duplicate_or_unknown_reviews_fail_closed(self):
        for index, problem in enumerate(("missing", "duplicate", "unknown", "wrong_identity")):
            reviews = copy.deepcopy(self.reviews)
            if problem == "missing":
                reviews.pop()
            elif problem == "duplicate":
                reviews[-1]["index"] = 0
            elif problem == "wrong_identity":
                reviews[-1]["index"] = 1000
            else:
                reviews[-1]["sample_score"]["score"]["value"] = {"unrecognized_metric": 1}
            write_jsonl(self.review_path, reviews)
            self.states = ["completed"]
            report = self.execute(f"bad-reviews-{index}")
            self.assertEqual(report["status"], "FAIL", report)
            self.assertIsNone(report["score"])

    def test_incomplete_declared_counts_or_contradictory_score_cannot_pass(self):
        original = copy.deepcopy(self.row)
        for index, problem in enumerate(("scored", "failed", "score", "scale")):
            self.row = copy.deepcopy(original)
            info = self.row["metric_metadata"]["gpqa_diamond:acc"]
            if problem == "scored":
                info["counts"]["scored"] = 197
            elif problem == "failed":
                info["counts"]["failed"] = 1
            elif problem == "scale":
                info["scale"] = "guess_from_value"
            else:
                self.row["metrics"]["gpqa_diamond:acc"] = "0.1"
            self.states = ["completed"]
            report = self.execute(f"bad-metric-{index}")
            self.assertEqual(report["status"], "FAIL", report)

    def test_endpoint_failure_after_measurement_preserves_complete_evidence(self):
        with patch.object(runner, "verify_endpoint", side_effect=[{"container_id": "same"}, OSError("container exited")]):
            report = self.execute()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["scored"], 198)
        self.assertIn("container exited", report["endpoint_error"])
        self.assertTrue((self.root / "output/native/reviews.jsonl").exists())
        self.assertTrue(self.observers[0].closed)
        self.assertEqual(json.loads((self.root / "output/report.json").read_text())["status"], "FAIL")

    def test_unknown_cli_schema_and_missing_executable_preserve_stdout_stderr(self):
        self.override["show"] = {"schema_version": "future"}
        report = self.execute()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("schema", report["error"])
        self.assertTrue(any(ref["kind"] == "bidevs-stdout" for ref in report["rawrefs"]))
        with patch.object(runner.subprocess, "run", side_effect=FileNotFoundError("no bidevs")):
            missing = runner.run(self.workspace, runner.CASE_ID, "/missing", self.receipt, self.root / "missing")
        self.assertEqual(missing["status"], "FAIL")
        self.assertTrue(any("no bidevs" in Path(ref["path"]).read_text()
                            for ref in missing["rawrefs"] if ref["kind"] == "bidevs-stderr"))

    def test_existing_output_never_overwrites_evidence_or_invokes_bidevs(self):
        output = self.root / "existing"
        output.mkdir()
        (output / "report.json").write_text("preserve")
        with patch.object(runner.subprocess, "run") as command:
            with self.assertRaises(FileExistsError):
                runner.run(self.workspace, runner.CASE_ID, "/bidevs", self.receipt, output)
            command.assert_not_called()
        self.assertEqual((output / "report.json").read_text(), "preserve")

    def test_observer_failure_overrides_a_successful_gpqa_result_and_keeps_evidence(self):
        self.observer_status = "FAIL"
        report = self.execute()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["scored"], 198)
        self.assertEqual(report["service_observation"]["status"], "FAIL")
        self.assertTrue(self.observers[0].closed)
        self.assertTrue((self.root / "output/service-observation.json").exists())
        self.assertTrue(any(Path(ref["path"]).name == "metrics.jsonl.gz" for ref in report["rawrefs"]))

    def test_observer_shutdown_exception_keeps_partial_summary_and_completed_native_outputs(self):
        self.observer_close_error = True
        report = self.execute()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("collector shutdown failed", report["observer_error"])
        self.assertEqual(report["service_observation"]["status"], "FAIL")
        self.assertEqual(report["scored"], 198)
        self.assertTrue((self.root / "output/native/reviews.jsonl").exists())
        self.assertEqual(json.loads((self.root / "output/service-observation.json").read_text())["status"], "FAIL")

    def test_observer_startup_failure_prevents_the_only_gpqa_run(self):
        self.observer_enter_error = True
        report = self.execute()
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(any(parts[0] == "run" for parts in self.calls))
        self.assertEqual(report["service_observation"]["status"], "FAIL")
        self.assertTrue((self.root / "output/service-observation.json").exists())

    def test_missing_native_logs_cannot_pass_with_only_final_predictions(self):
        self.native_log.unlink()
        report = self.execute()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("native evaluator logs are missing", report["error"])
        self.assertTrue(self.observers[0].closed)
        self.assertTrue(any(Path(ref["path"]).name == "docker.log" for ref in report["rawrefs"]))

    def test_failed_result_query_still_preserves_native_logs_and_closes_observer(self):
        self.query_returncode = 1
        report = self.execute()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("result/report query", report["error"])
        self.assertTrue(self.observers[0].closed)
        self.assertTrue(any(ref["kind"] == "native-log" for ref in report["rawrefs"]))
        self.assertEqual(sum(parts[0] == "run" for parts in self.calls), 1)


if __name__ == "__main__":
    unittest.main()
