"""Controlled coverage requires raw, configuration-bound outcomes for all 42 cells."""

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmarks import profile_controlled_report as audit
from scripts.benchmarks import profile_report as common


class ControlledFixture:
    def __init__(self, root):
        self.root = root
        self.source_dir = root / "controlled-original/results"
        self.profile = {"context": 1048576, "memory": .90, "chunk": 32768}
        self.source_id = "controlled-original"
        self.launch = self.make_launch("source")
        self.frozen = {"run_id": "fixed", "token_pool": [100, 200, 300], "nature_tokens": [100] * 64,
                       "source": {"source_manifest_sha256": "source-fingerprint"}}
        self.frozen["sha256"] = common._payload_digest(self.frozen)
        self.cells = []
        for dataset in ("random", "nature"):
            for isl in (8, 16, 24, 32, 40):
                self.cells.append(self.spec("matrix_single", dataset, isl, 1))
        for dataset in ("random", "nature"):
            for isl in (4, 8, 12, 16):
                for concurrency in (1, 2, 3, 4):
                    self.cells.append(self.spec("matrix_concurrent", dataset, isl, concurrency))
        self.failed = self.cells[4]
        self.summary = self.base_summary("source", status="FAIL")
        startup = {"cell_id": "startup-graphs", "kind": "matrix", "isl": 1024, "osl": 128, "concurrency": 256,
                   "dataset": "random", "source_case": {"seed": 1017}}
        self.summary["global_warmup"] = self.warm(self.source_dir / "global-warmup", startup, version=1, global_scope=True, replay=False)
        self.summary["cells"] = []
        retained = []
        for cell in self.cells[:4]:
            folder = self.source_dir / "cells" / cell["cell_id"].replace("/", "__")
            warm = self.warm(folder / "warmup", cell, version=1, replay=False)
            rounds = []
            refs = []
            for repeat in (1, 2, 3):
                bodies = audit.canonical_bodies(self.frozen, cell, f"repeat-{repeat}")
                where = folder / f"repeat-{repeat}"
                value = self.archive(where, bodies, replay=False)
                rounds.append(value)
                refs.append({"repeat": repeat, "report": self.ref(where / "report.json"),
                             "payloads": self.ref(where / "request-payloads.json.gz"), "metrics": self.ref(where / "request-metrics.json.gz")})
            self.summary["cells"].append({"cell_id": cell["cell_id"], "status": "PASS", "warmup": warm, "measurements": rounds})
            retained.append({"cell_id": cell["cell_id"], "primary_repeat": 1, "source_refs": refs[0], "supplemental_repeats": refs[1:]})
        failure_dir = self.source_dir / "cells" / self.failed["cell_id"].replace("/", "__")
        warm = self.warm(failure_dir / "warmup", self.failed, version=1, replay=False, failed=True)
        self.summary["cells"].append({"cell_id": self.failed["cell_id"], "status": "FAIL", "warmup": warm, "measurements": []})
        log = self.source_dir / "observation/docker.log"
        self.text(log, "2026-10-25T01:09:00.000000000Z [rank0]:[W] allocation failed with OOM\n"
                       "2026-10-25T01:09:01.000000000Z ERROR aten::new_empty failed\n"
                       "2026-10-25T01:09:02.000000000Z ERROR EngineDeadError\n")
        inspect = self.source_dir / "container.json"
        self.save(inspect, [self.inspect(self.launch, running=False)])
        proof = {"service_log": self.ref(log), "container_inspect": self.ref(inspect), "window_scope": "source_attempt",
                 "window_started_at": self.summary["started_at"], "window_finished_at": self.summary["finished_at"]}
        self.plan = {"profile": self.profile, "cells": self.cells}
        self.save(self.source_dir / "summary.json", self.summary)
        self.save(self.source_dir / "plan.json", self.plan)
        self.save(self.source_dir / "launch.json", self.launch)
        self.save(root / "inputs.json.gz", self.frozen)
        self.manifest_path = root / "manifest.json"
        self.manifest = {
            "schema_version": 1, "kind": audit.FOLLOWUP, "policy_version": 2,
            "profile_id": self.source_id, "profile": self.profile, "frozen_inputs_sha256": self.frozen["sha256"],
            "source_attempt": {"attempt_id": self.source_id, "status": "FAIL"},
            "source_refs": {"summary": self.ref(self.source_dir / "summary.json"), "plan": self.ref(self.source_dir / "plan.json"),
                            "launch": self.ref(self.source_dir / "launch.json"), "inputs": self.ref(root / "inputs.json.gz")},
            "cells": self.cells, "expected_cell_ids": [c["cell_id"] for c in self.cells], "retained": retained,
            "pending_cell_ids": [c["cell_id"] for c in self.cells[4:]],
            "failed_replay": {"cell_id": self.failed["cell_id"], "source_refs": {
                "warmup_report": self.ref(failure_dir / "warmup/report.json"),
                "payloads": self.ref(failure_dir / "warmup/round-1/request-payloads.json.gz"),
                "metrics": self.ref(failure_dir / "warmup/round-1/request-metrics.json.gz")}, "terminal_evidence": proof},
        }
        self.save(self.manifest_path, self.manifest)

    def spec(self, name, dataset, isl, concurrency):
        return {"cell_id": f"{name}/{dataset}/i{isl}-o1000-c{concurrency}", "case_id": name, "kind": "matrix", "group": "regular",
                "dataset": dataset, "isl": isl, "osl": 1000, "concurrency": concurrency, "status": "APPLICABLE", "repeats": 3,
                "source_case": {"id": name, "seed": 1017}}

    def make_launch(self, container):
        return {"image_id": "image", "container_id": container, "source_manifest_sha256": "source-fingerprint", "command": [
            "docker", "run", "--name", container, "--mount", f"type=volume,src={container}-cache,dst=/workspace",
            "-p", "127.0.0.1:6688:8000", "image", "serve", "model", "--max-model-len", "1048576",
            "--max-num-batched-tokens", "32768", "--gpu-memory-utilization", "0.9"]}

    def base_summary(self, container, status="PASS"):
        return {"status": status, "profile": self.profile, "source": {"source_manifest_sha256": "source-fingerprint"},
                "endpoint": {"image_id": "image", "container_id": container, "started_at": "2026-10-25T00:00:00+00:00"},
                "frozen_inputs_sha256": self.frozen["sha256"], "started_at": "2026-10-25T01:00:00+00:00",
                "finished_at": "2026-10-25T01:10:00+00:00", "observation": {"status": "PASS", "container_id": container}}

    def inspect(self, launch, running=True):
        return {"Id": launch["container_id"], "Image": launch["image_id"],
                "Config": {"Cmd": launch["command"][launch["command"].index("image") + 1:]},
                "State": {"Running": running, "ExitCode": 0 if running else 1,
                          "StartedAt": "2026-10-25T00:00:00+00:00", "FinishedAt": "2026-10-25T01:09:03+00:00"}}

    def text(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gz":
            path.write_bytes(gzip.compress(json.dumps(value).encode()))
        else:
            path.write_text(json.dumps(value))

    def ref(self, path):
        return audit.file_ref(path)

    def archive(self, directory, canonical, *, replay, failed=False):
        bodies = copy.deepcopy(canonical)
        if replay:
            for index, body in enumerate(bodies):
                body["cache_salt"] = "controlled:" + common._payload_digest([str(directory), index])
        rows = [{"status": "FAIL", "error": "HTTP Error 500 EngineDeadError"} for _ in bodies] if failed else [
            {"status": "PASS", "input_tokens": len(body["prompt"]), "output_tokens": body["max_tokens"], "cached_tokens": 0,
             "e2e_s": 2.0, "ttft_s": .1, "tpot_s": .01} for body in bodies]
        inputs = sum(r.get("input_tokens", 0) for r in rows)
        outputs = sum(r.get("output_tokens", 0) for r in rows)
        value = {"status": "FAIL" if failed else "PASS", "cache_verified": not failed, "requests": rows,
                 "concurrency": len(rows), "request_count": len(rows), "success": 0 if failed else len(rows),
                 "failure_rate": 1 if failed else 0, "elapsed_s": 2.0, "observed_input_tokens": inputs,
                 "observed_output_tokens": outputs, "input_tps": inputs / 2, "output_tps": outputs / 2,
                 "total_tps": (inputs + outputs) / 2, "errors": ["engine died"] if failed else [],
                 "payload_sha256": common._payload_digest(bodies)}
        self.save(directory / "report.json", value)
        self.save(directory / "request-payloads.json.gz", bodies)
        self.save(directory / "request-metrics.json.gz", rows)
        if replay:
            self.save(directory / "canonical-payloads.json.gz", canonical)
            self.save(directory / "replay-provenance.json", {
                "schema_version": 1, "allowed_changes": ["cache_salt"],
                "canonical_payloads_ref": self.ref(directory / "canonical-payloads.json.gz"),
                "actual_payloads_ref": self.ref(directory / "request-payloads.json.gz"),
                "canonical_payload_sha256": common._payload_digest(canonical), "actual_payload_sha256": value["payload_sha256"],
                "content_without_salt_sha256": common._payload_digest(audit.without_salt(canonical)),
            })
        return value

    def warm(self, directory, spec, *, version=2, global_scope=False, replay=True, failed=False):
        count = 2 if global_scope or version == 1 else 1
        concurrency = 256 if global_scope else 2
        osl = 128 if global_scope or version == 1 else min(spec["osl"], 2)
        rows = []
        for number in range(1, (1 if failed else count) + 1):
            canonical = audit.canonical_bodies(self.frozen, spec, f"{spec['cell_id']}-warmup-{number}", count=concurrency, osl=osl)
            measured = self.archive(directory / f"round-{number}", canonical, replay=replay, failed=failed)
            rows.append({"round": number, "measurement": measured})
        value = {"status": "FAIL" if failed else "PASS", "concurrency": concurrency, "osl": osl, "rounds": rows,
                 "errors": ["warmup failed"] if failed else [], "both_dp_active": not failed,
                 "dp_before": {str(i): {"prompt_tokens": 0, "generation_tokens": 0} for i in range(2)},
                 "dp_after": {str(i): {"prompt_tokens": 100, "generation_tokens": 10} for i in range(2)}}
        if version == 2:
            value.update(configured_rounds=count, additional_round_count=0, additional_rounds=[])
        self.save(directory / "report.json", value)
        return value

    def followup(self, name="followup", *, diagnostic=False, failed=False, selected=None):
        directory = self.root / name
        launch = self.make_launch(name)
        summary = self.base_summary(name, status="FAIL" if failed else "PASS")
        selected = selected or ([self.failed] if diagnostic else self.cells[5:])
        summary.update(schema_version=1, kind=audit.FOLLOWUP, phase="fresh_replay" if diagnostic else "remaining",
                       profile_id=self.source_id, attempt_id=name, policy_version=2,
                       manifest_ref=self.ref(self.manifest_path), scope={"kind": "diagnostic" if diagnostic else "subset",
                       "cell_ids": [c["cell_id"] for c in selected], "expected_cell_ids": self.manifest["expected_cell_ids"],
                       "retained_cell_ids": [c["cell_id"] for c in self.cells[:4]]}, cells=[])
        self.save(directory / "launch.json", launch)
        summary["launch_ref"] = self.ref(directory / "launch.json")
        startup = {"cell_id": "startup-graphs", "kind": "matrix", "isl": 1024, "osl": 128, "concurrency": 256,
                   "dataset": "random", "source_case": {"seed": 1017}}
        summary["global_warmup"] = self.warm(directory / "global-warmup", startup, global_scope=True)
        for spec in selected:
            folder = directory / "cells" / spec["cell_id"].replace("/", "__")
            row = {"cell_id": spec["cell_id"], "case_id": spec["case_id"], "kind": "matrix", "group": "regular",
                   "status": "FAIL" if failed else "PASS", "stage": "measurement", "raw_directory": str(folder.resolve()),
                   "started_at": summary["started_at"], "finished_at": summary["finished_at"]}
            if diagnostic:
                canonical = audit.canonical_bodies(self.frozen, spec, f"{spec['cell_id']}-warmup-1", count=2, osl=128)
            else:
                row["warmup"] = self.warm(folder / "warmup", spec)
                canonical = audit.canonical_bodies(self.frozen, spec, "repeat-1")
            row["measurements"] = [self.archive(folder / "repeat-1", canonical, replay=True, failed=failed)]
            if failed:
                log = directory / "observation/docker.log"
                self.text(log, "2026-10-25T01:09:00.000000000Z [W] allocation failed with OOM\n"
                               "2026-10-25T01:09:01.000000000Z ERROR aten::new_empty failed\n"
                               "2026-10-25T01:09:02.000000000Z ERROR EngineDeadError\n")
                row["terminal_evidence"] = {"service_log": self.ref(log), "container_inspect": {},
                                            "window_started_at": row["started_at"], "window_finished_at": row["finished_at"]}
            summary["cells"].append(row)
        self.save(directory / "container-final.json", [self.inspect(launch, running=not failed)])
        summary["final_container_inspect_ref"] = self.ref(directory / "container-final.json")
        for row in summary["cells"]:
            if failed:
                row["terminal_evidence"]["container_inspect"] = summary["final_container_inspect_ref"]
        self.save(directory / "summary.json", summary)
        return directory / "summary.json"


class ControlledAssessmentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.fixture = ControlledFixture(Path(directory.name))

    def test_rejection_does_not_hide_the_37_unrun_cells(self):
        result = audit.assess(self.fixture.manifest_path, [])
        self.assertEqual(result["validation_errors"], [])
        self.assertEqual(result["status"], "INCOMPLETE_WITH_FAILURES")
        self.assertEqual(result["coverage"], {"expected": 42, "covered": 5, "measured_pass": 4, "terminal_failed": 1,
                         "pass_with_failure_history": 0, "not_run": 37, "missing_cell_ids": [c["cell_id"] for c in self.fixture.cells[5:]]})
        failed = result["cells"][self.fixture.failed["cell_id"]]
        self.assertEqual(failed["outcome"], "FAIL_WARMUP")
        self.assertIsNone(failed["performance"])

    def test_all_42_real_outcomes_close_assessment_without_fabricating_performance(self):
        completed = self.fixture.followup()
        replay = self.fixture.followup("fresh", diagnostic=True, failed=True)
        result = audit.assess(self.fixture.manifest_path, [completed, replay])
        self.assertEqual(result["validation_errors"], [], result)
        self.assertEqual(result["status"], "COMPLETE_WITH_FAILURES")
        self.assertEqual(result["candidate_outcome"], "REJECTED")
        self.assertEqual((result["coverage"]["covered"], result["coverage"]["measured_pass"], result["coverage"]["not_run"]), (42, 41, 0))
        failed = result["cells"][self.fixture.failed["cell_id"]]
        self.assertIsNone(failed["performance"])
        self.assertEqual(len(failed["failures"]), 2)
        first = result["cells"][self.fixture.cells[0]["cell_id"]]["performance"]
        self.assertEqual(first["source_attempt_id"], self.fixture.source_id)
        self.assertEqual(len(first["reference_measurements"]), 2)

    def test_diagnostic_pass_never_becomes_formal_matrix_pass(self):
        path = self.fixture.followup("diagnostic", diagnostic=True)
        result = audit.assess(self.fixture.manifest_path, [path])
        self.assertEqual(result["validation_errors"], [])
        self.assertEqual(result["coverage"]["measured_pass"], 4)
        self.assertIsNone(result["cells"][self.fixture.failed["cell_id"]]["performance"])

    def test_manifest_cannot_remove_cells_or_count_duplicates(self):
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["expected_cell_ids"] = manifest["expected_cell_ids"][:4]
        self.fixture.save(self.fixture.manifest_path, manifest)
        self.assertTrue(audit.assess(self.fixture.manifest_path, [])["validation_errors"])

    def test_allocator_warning_and_out_of_window_fatal_are_not_terminal(self):
        fixture = self.fixture
        proof = fixture.manifest["failed_replay"]["terminal_evidence"]
        log = Path(proof["service_log"]["path"])
        inspect = Path(proof["container_inspect"]["path"])
        fixture.save(inspect, [fixture.inspect(fixture.launch, running=True)])
        for text in ("2026-10-25T01:09:00Z [W] allocation failed with OOM\naten::new_empty\n",
                     "2026-10-25T01:09:00Z (EngineCore_DP0 pid=123) WARNING failed to allocate with OOM; recovered\n",
                     "2026-10-25T01:09:00Z [W] allocation failed with OOM\n2026-10-25T00:50:00Z ERROR EngineDeadError\n"):
            fixture.text(log, text)
            proof["service_log"] = fixture.ref(log)
            proof["container_inspect"] = fixture.ref(inspect)
            fixture.save(fixture.manifest_path, fixture.manifest)
            result = audit.assess(fixture.manifest_path, [])
            self.assertTrue(any("warning alone" in error for error in result["validation_errors"]))

    def test_salt_only_proof_rejects_changed_tokens_even_with_updated_hashes(self):
        path = self.fixture.followup(selected=[self.fixture.cells[5]])
        summary = json.loads(path.read_text())
        folder = Path(summary["cells"][0]["raw_directory"]) / "repeat-1"
        with gzip.open(folder / "canonical-payloads.json.gz", "rt") as stream:
            bodies = json.load(stream)
        bodies[0]["prompt"][0] += 1
        summary["cells"][0]["measurements"] = [self.fixture.archive(folder, bodies, replay=True)]
        self.fixture.save(path, summary)
        result = audit.assess(self.fixture.manifest_path, [path])
        self.assertTrue(any("canonical payload differs" in error for error in result["validation_errors"]))

    def test_replay_must_actually_change_each_salt(self):
        path = self.fixture.followup(selected=[self.fixture.cells[5]])
        summary = json.loads(path.read_text())
        folder = Path(summary["cells"][0]["raw_directory"]) / "repeat-1"
        with gzip.open(folder / "canonical-payloads.json.gz", "rt") as stream:
            canonical = json.load(stream)
        summary["cells"][0]["measurements"] = [self.fixture.archive(folder, canonical, replay=False)]
        proof = json.loads((folder / "replay-provenance.json").read_text())
        proof.update(actual_payload_sha256=common._payload_digest(canonical),
                     actual_payloads_ref=self.fixture.ref(folder / "request-payloads.json.gz"))
        self.fixture.save(folder / "replay-provenance.json", proof)
        self.fixture.save(path, summary)
        errors = audit.assess(self.fixture.manifest_path, [path])["validation_errors"]
        self.assertTrue(any("did not change every cache salt" in error for error in errors), errors)

    def test_fresh_diagnostic_requires_unused_new_process_and_own_startup(self):
        for case in ("nonzero_counters", "inherited_startup", "original_process"):
            with self.subTest(case=case):
                path = self.fixture.followup("diagnostic-" + case, diagnostic=True)
                run = json.loads(path.read_text())
                if case == "nonzero_counters":
                    run["global_warmup"]["dp_before"]["0"]["prompt_tokens"] = 1
                elif case == "inherited_startup":
                    run["startup_proof_ref"] = self.fixture.ref(self.fixture.source_dir / "summary.json")
                else:
                    run["endpoint"] = self.fixture.summary["endpoint"]
                    self.fixture.save(path.parent / "launch.json", self.fixture.launch)
                    run["launch_ref"] = self.fixture.ref(path.parent / "launch.json")
                    inspect_path = Path(run["final_container_inspect_ref"]["path"])
                    self.fixture.save(inspect_path, [self.fixture.inspect(self.fixture.launch)])
                    run["final_container_inspect_ref"] = self.fixture.ref(inspect_path)
                self.fixture.save(path, run)
                errors = audit.assess(self.fixture.manifest_path, [path])["validation_errors"]
                self.assertTrue(any("fresh replay" in error for error in errors), errors)

    def test_16k_configuration_cannot_replace_any_32k_cell(self):
        path = self.fixture.followup(selected=[self.fixture.cells[5]])
        value = json.loads(path.read_text())
        value["profile"] = {**value["profile"], "chunk": 16384}
        self.fixture.save(path, value)
        self.assertTrue(any("profile differs" in error for error in audit.assess(self.fixture.manifest_path, [path])["validation_errors"]))

    def test_raw_metrics_mismatch_cannot_be_hidden_by_a_pass_summary(self):
        path = self.fixture.source_dir / "cells" / self.fixture.cells[0]["cell_id"].replace("/", "__") / "repeat-1/request-metrics.json.gz"
        self.fixture.save(path, [])
        for value in self.fixture.manifest["retained"]:
            if value["cell_id"] == self.fixture.cells[0]["cell_id"]:
                value["source_refs"]["metrics"] = self.fixture.ref(path)
        self.fixture.save(self.fixture.manifest_path, self.fixture.manifest)
        result = audit.assess(self.fixture.manifest_path, [])
        self.assertTrue(any("request metrics differ" in error for error in result["validation_errors"]))

    def test_saved_complete_receipt_is_revalidated_not_trusted(self):
        value = audit.assess(self.fixture.manifest_path, [])
        value["status"] = "COMPLETE_WITH_FAILURES"
        path = self.fixture.root / "assessment.json"
        self.fixture.save(path, value)
        with self.assertRaisesRegex(ValueError, "differs"):
            audit.validate_receipt(path, common.Evidence())

    def test_later_terminal_replay_cannot_explain_the_original_unproven_failure(self):
        self.fixture.manifest["failed_replay"].pop("terminal_evidence")
        self.fixture.save(self.fixture.manifest_path, self.fixture.manifest)
        replay = self.fixture.followup("new-failure", diagnostic=True, failed=True)
        result = audit.assess(self.fixture.manifest_path, [replay])
        self.assertEqual(result["validation_errors"], [])
        self.assertEqual(result["unresolved_failure_attempt_ids"], [self.fixture.source_id])
        self.assertNotIn(self.fixture.source_id, result["covered_failure_attempt_ids"])
        self.assertEqual(result["status"], "INCOMPLETE_WITH_FAILURES")

    def test_formal_pass_requires_its_real_warmup_files(self):
        path = self.fixture.followup(selected=[self.fixture.cells[5]])
        row = json.loads(path.read_text())["cells"][0]
        (Path(row["raw_directory"]) / "warmup/round-1/request-metrics.json.gz").unlink()
        self.assertTrue(audit.assess(self.fixture.manifest_path, [path])["validation_errors"])

    def test_terminal_failure_does_not_absorb_unrelated_observer_or_storage_errors(self):
        path = self.fixture.followup("failed-diagnostic", diagnostic=True, failed=True)
        original = json.loads(path.read_text())
        for index, problem in enumerate(("gpu-observer", "early-network", "storage")):
            run = copy.deepcopy(original)
            if problem == "storage":
                run["error"] = "OSError: failed to persist evidence"
            else:
                run["observation"].update(status="FAIL", observer_errors=[{
                    "operation": "nvidia-smi" if problem == "gpu-observer" else "metrics",
                    "error": "device read failed" if problem == "gpu-observer" else "URLError: connection refused",
                    "time": "2026-10-25T01:09:04+00:00" if problem == "gpu-observer" else "2026-10-25T01:08:00+00:00",
                }])
            self.fixture.save(path, run)
            result = audit.assess(self.fixture.manifest_path, [path])
            self.assertTrue(result["validation_errors"], (index, result))

    def test_terminal_service_exit_can_explain_subsequent_metrics_read_failure(self):
        path = self.fixture.followup("failed-diagnostic", diagnostic=True, failed=True)
        run = json.loads(path.read_text())
        run["observation"].update(status="FAIL", observer_errors=[{
            "operation": "final metrics", "error": "URLError: connection refused",
            "time": "2026-10-25T01:09:04+00:00",
        }])
        self.fixture.save(path, run)
        result = audit.assess(self.fixture.manifest_path, [path])
        self.assertEqual(result["validation_errors"], [], result)
        self.assertIn("failed-diagnostic", result["covered_failure_attempt_ids"])

    def test_new_warmup_terminal_failure_has_no_formal_performance(self):
        spec = self.fixture.cells[5]
        path = self.fixture.followup("warmup-failure", failed=True, selected=[spec])
        run = json.loads(path.read_text())
        row = run["cells"][0]
        row.update(stage="warmup", measurements=[])
        row["warmup"] = self.fixture.warm(Path(row["raw_directory"]) / "warmup", spec, failed=True)
        self.fixture.save(path, run)
        result = audit.assess(self.fixture.manifest_path, [path])
        self.assertEqual(result["validation_errors"], [], result)
        outcome = result["cells"][spec["cell_id"]]
        self.assertEqual(outcome["outcome"], "FAIL_WARMUP")
        self.assertIsNone(outcome["performance"])


if __name__ == "__main__":
    unittest.main()
