"""Offline evidence must be complete and comparable before FINAL_PASS."""

import copy
import gzip
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmarks import profile_report as report


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class ProfileReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = report.default_policy()
        self.policy["ppl_windows"] = 1  # Small synthetic dataset; production policy requires 134.
        self.schedule = []
        self.state = {"status": "PHASE_PASS", "inputs_sha256": digest("inputs"), "profiles": []}
        for context in self.policy["contexts"]:
            for memory in (.90, .93, .96):
                profile = {"id": f"p{len(self.schedule)}", "context": context, "memory": memory,
                           "chunk": self.policy["baseline_chunk"], "only": "all"}
                self.schedule.append(profile)
                self.add_attempt(profile)
        self.save(self.root / "schedule.json", self.schedule)
        self.save_state()
        self.gpqa = self.root / "quality/gpqa.json"
        raw = self.root / "quality/evaluator.txt"
        raw.parent.mkdir()
        raw.write_text("synthetic complete evaluator fixture")
        self.save(self.gpqa, {"status": "PASS", "questions": self.policy["gpqa_questions"],
                              "scored": self.policy["gpqa_questions"], "validator": "synthetic-test-validator",
                              "rawrefs": [{"path": raw.name, "sha256": report._sha(raw)}]})
        self.soak = self.root / "quality/soak/summary.json"
        count, isl, osl = (self.policy[k] for k in ("soak_concurrency", "soak_isl", "soak_osl"))
        duration = self.policy["soak_duration_s"]
        rows = [self.row(isl, osl, index=i) for i in range(count)]
        for row in rows:
            row["payload_sha256"] = digest("soak" + str(row["index"]))
        value = self.measurement(rows, duration + 1, count)
        value.update(duration_s=duration, measurement_window_s=duration,
                     frozen_inputs_sha256=digest("inputs"), endpoint={"image_id": "sha256:" + digest("image")},
                     warmup={"status": "PASS"}, observation={"status": "PASS"},
                     request_identities=[{"index": r["index"], "payload_sha256": r["payload_sha256"]} for r in rows])
        value.pop("requests")
        self.save(self.soak, value)
        self.save(self.soak.parent / "request-metrics.json.gz", rows)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gz":
            with gzip.open(path, "wt") as stream:
                json.dump(value, stream)
        else:
            path.write_text(json.dumps(value))

    def save_state(self):
        self.save(self.root / "campaign.json", self.state)

    def row(self, isl=8, osl=2, index=0):
        return {"status": "PASS", "index": index, "input_tokens": isl, "output_tokens": osl,
                "cached_tokens": 0, "ttft_s": .2, "tpot_s": .1, "e2e_s": .8}

    def measurement(self, rows, elapsed=2.0, concurrency=1):
        total_in = sum(r["input_tokens"] for r in rows)
        total_out = sum(r["output_tokens"] for r in rows)
        return {"status": "PASS", "cache_verified": True, "request_count": len(rows), "success": len(rows),
                "failure_rate": 0, "requests": rows, "elapsed_s": elapsed,
                "observed_input_tokens": total_in, "observed_output_tokens": total_out,
                "input_tps": total_in / elapsed, "output_tps": total_out / elapsed,
                "total_tps": (total_in + total_out) / elapsed, "concurrency": concurrency}

    def warmup(self, version=1, global_scope=False, osl=2):
        count = 2 if version == 1 or global_scope else 1
        value = {"status": "PASS", "concurrency": 256 if global_scope else 2,
                 "osl": 128 if global_scope else min(osl, 128 if version == 1 else 2),
                 "rounds": [{"round": i + 1, "measurement": {"status": "PASS", "cache_verified": True}}
                            for i in range(count)], "errors": [], "both_dp_active": True,
                 "dp_before": {str(i): {"prompt_tokens": 0, "generation_tokens": 0} for i in range(2)},
                 "dp_after": {str(i): {"prompt_tokens": 100, "generation_tokens": 10} for i in range(2)}}
        if version == 2:
            value.update(configured_rounds=count, additional_round_count=0, additional_rounds=[])
        return value

    def add_attempt(self, profile, *, attempt_id=None, state="PASS"):
        ident = attempt_id or profile["id"]
        entry = {**profile, "attempt_id": ident, "status": state}
        self.state["profiles"].append(entry)
        directory = self.root / ident / "results"
        launch = {"image_id": "sha256:" + digest("image"), "container_id": ident,
                  "source_manifest_sha256": digest("source"), "command": [
                      "docker", "run", "--name", ident, "--mount", f"type=volume,src={ident}-cache,dst=/workspace",
                      "-p", "127.0.0.1:6688:8000", "sha256:" + digest("image"), "serve", "model",
                      "--max-model-len", str(profile["context"]), "--max-num-batched-tokens", str(profile["chunk"]),
                      "--gpu-memory-utilization", str(profile["memory"])]}
        cells = [
            {"cell_id": "matrix_single/random/i8-o2-c1", "case_id": "matrix_single", "kind": "matrix",
             "group": "regular", "dataset": "random", "isl": 8, "osl": 2, "concurrency": 1,
             "expected_input_by_index": [8], "status": "APPLICABLE", "repeats": 3},
            {"cell_id": "quality-ppl-wikitext", "case_id": "quality-ppl-wikitext", "kind": "ppl",
             "group": "quality", "status": "APPLICABLE", "repeats": 1},
            {"cell_id": "historical", "case_id": "historical", "kind": "ab", "group": "ab",
             "status": "NOT_APPLICABLE", "repeats": 0},
            {"cell_id": "quality-gpqa-diamond", "case_id": "quality-gpqa-diamond", "kind": "gpqa",
             "group": "quality", "status": "ONCE_ONLY", "repeats": 1},
        ]
        axis = {k: profile[k] for k in ("context", "memory", "chunk")}
        self.save(directory / "plan.json", {"profile": axis, "cells": cells, "counts": {"total": 4}})
        rounds = []
        for index, elapsed in enumerate((1, 2, 4), 1):
            bodies = [{"prompt": [index] * 8, "max_tokens": 2, "cache_salt": f"repeat-{index}"}]
            measured = {**self.measurement([self.row()], elapsed), "payload_sha256": report._payload_digest(bodies)}
            folder = directory / "cells" / cells[0]["cell_id"].replace("/", "__") / f"repeat-{index}"
            self.save(folder / "report.json", measured)
            self.save(folder / "request-payloads.json.gz", bodies)
            self.save(folder / "request-metrics.json.gz", measured["requests"])
            rounds.append(measured)
        actual = [{"cell_id": cells[0]["cell_id"], "case_id": "matrix_single", "status": "PASS", "measurements": rounds,
                   "warmup": self.warmup()}]
        if profile.get("only", "all") == "all":
            actual += [
                {"cell_id": "quality-ppl-wikitext", "status": "PASS", "measurements": [
                    {"status": "PASS", "windows": 1, "scored_tokens": 15, "dataset_sha256": digest("ppl"),
                     "rows": [{"status": "PASS", "scored_tokens": 15, "cached_tokens": 0}], "perplexity": 2.7}]},
                {"cell_id": "historical", "status": "NOT_APPLICABLE"},
                {"cell_id": "quality-gpqa-diamond", "status": "ONCE_ONLY"},
            ]
        summary = {"status": state, "profile": axis, "source": {"source_manifest_sha256": digest("source")},
                   "endpoint": {"image_id": launch["image_id"], "container_id": ident},
                   "frozen_inputs_sha256": digest("inputs"), "selection": profile.get("only", "all"),
                   "global_warmup": self.warmup(global_scope=True), "observation": {"status": "PASS"}, "cells": actual}
        self.save(directory / "launch.json", launch)
        self.save(directory / "summary.json", summary)
        return entry

    def build(self, **kwargs):
        kwargs.setdefault("policy", self.policy)
        return report.build_report(self.root, [self.root / "schedule.json"],
                                   gpqa_result=self.gpqa, soak_result=self.soak, **kwargs)

    def mutate_summary(self, profile, change):
        path = self.root / profile / "results/summary.json"
        value = json.loads(path.read_text())
        change(value)
        self.save(path, value)

    def version2(self, ident, *, inherited=False):
        directory = self.root / ident / "results"
        plan = json.loads((directory / "plan.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        policy = {"id": "single_formal_v2", "primary_repeat": 1, "formal_repetitions": 1}
        plan.update(policy_version=2, measurement_policy=policy)
        summary.update(policy_version=2, measurement_policy=policy)
        for cell in plan["cells"]:
            if cell["status"] in report.ELIGIBLE:
                cell.update(legacy_repeats=cell["repeats"], repeats=1)
        for cell in summary["cells"]:
            if cell["status"] == "PASS":
                cell["measurements"] = cell["measurements"][:1]
                if "warmup" in cell and not inherited:
                    cell["warmup"] = self.warmup(version=2)
                    for path in (directory / "cells" / cell["cell_id"].replace("/", "__")).glob("repeat-*"):
                        if path.name != "repeat-1":
                            shutil.rmtree(path)
        if not inherited:
            summary["global_warmup"] = self.warmup(version=2, global_scope=True)
        self.save(directory / "plan.json", plan)
        self.save(directory / "summary.json", summary)
        return directory, plan, summary

    def resumed(self, completed=2):
        source = self.root / "p0/results"
        target_id = "p0-v2"
        target = self.root / target_id / "results"
        shutil.copytree(source, target)
        source_summary = json.loads((source / "summary.json").read_text())
        cell_id = source_summary["cells"][0]["cell_id"]
        source_summary.update(status="IN_PROGRESS", cells=[source_summary["cells"][0]])
        source_summary["cells"][0].update(status="IN_PROGRESS", measurements=source_summary["cells"][0]["measurements"][:completed])
        for path in (source / "cells" / cell_id.replace("/", "__")).glob("repeat-*"):
            if int(path.name.split("-")[-1]) > completed:
                shutil.rmtree(path)
        self.save(source / "summary.json", source_summary)
        self.state["profiles"][0]["status"] = "POLICY_HANDOFF"
        entry = {**self.state["profiles"][0], "attempt_id": target_id, "status": "PASS"}
        self.state["profiles"].append(entry)
        self.save_state()
        _, _, summary = self.version2(target_id, inherited=True)

        def ref(path):
            return {"path": str(path.resolve()), "sha256": report._sha(path)}
        quiescent = self.root / "handoff/quiescence.json"
        self.save(quiescent, {"running": 0, "waiting": 0})
        handoff = self.root / "handoff/handoff.json"
        self.save(handoff, {"status": "POLICY_HANDOFF", "source_attempt_id": "p0", "current_cell": cell_id,
                            "no_inflight": True, "pending_unsent": list(range(completed + 1, 4)),
                            "source_snapshot_sha256": report._sha(source / "summary.json"), "rawrefs": [ref(quiescent)]})
        summary["resume_provenance"] = {
            "schema_version": 1, "handoff": ref(handoff),
            "sources": {"p0": {name: ref(source / (name + ".json")) for name in ("plan", "summary", "launch")}},
            "cells": {cell_id: {"source_attempt_id": "p0", "source_cell_id": cell_id, "primary_repeat": 1}},
        }
        summary["global_warmup_reuse"] = "same live container and source proof"
        self.save(target / "summary.json", summary)
        return source, target, handoff

    def test_complete_grid_and_once_evidence_can_pass(self):
        result = self.build()
        self.assertEqual(result["status"], "FINAL_PASS", (result["errors"], result["pending"]))
        self.assertEqual(len(result["primary_profile_ids"]), len(self.policy["contexts"]) * self.policy["memory_levels"])
        metrics = result["attempts"][0]["cells"]["matrix_single/random/i8-o2-c1"]["metrics"]
        self.assertEqual(metrics["output_tps"], 2)
        reference = result["attempts"][0]["cells"]["matrix_single/random/i8-o2-c1"]["reference_measurements"]
        self.assertEqual([(row["repeat"], row["metrics"]["output_tps"]) for row in reference], [(2, 1), (3, .5)])
        self.assertIn("FINAL_PASS", report.markdown(result))

    def test_missing_group_and_missing_repetition_fail(self):
        self.state["profiles"].pop()
        self.save_state()
        result = self.build()
        self.assertNotEqual(result["status"], "FINAL_PASS")
        self.assertTrue(any("untracked" in e for e in result["errors"]))
        self.mutate_summary("p0", lambda s: s["cells"][0]["measurements"].pop())
        result = self.build()
        self.assertTrue(any("repetitions" in e for e in result["errors"]))

    def test_wrong_payload_or_kind_never_gets_a_percentage(self):
        self.mutate_summary("p1", lambda s: s["cells"][0]["measurements"][0].update(payload_sha256=digest("different")))
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        comparison = result["comparisons"]["p1"]["cells"]["matrix_single/random/i8-o2-c1"]
        self.assertEqual(comparison["status"], "NOT_COMPARABLE")
        self.assertNotIn("change_pct", comparison)

    def test_identical_ids_with_different_kinds_cannot_be_compared(self):
        path = self.root / "p1/results/plan.json"
        plan = json.loads(path.read_text())
        plan["cells"][0]["kind"] = "chat_burst"
        self.save(path, plan)
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["comparisons"]["p1"]["cells"][plan["cells"][0]["cell_id"]]["status"], "NOT_COMPARABLE")

    def test_bad_actual_lengths_and_invalid_gpqa_raw_hash_fail(self):
        self.mutate_summary("p1", lambda s: s["cells"][0]["measurements"][0]["requests"][0].update(input_tokens=9))
        value = json.loads(self.gpqa.read_text())
        value["rawrefs"][0]["sha256"] = digest("changed")
        self.save(self.gpqa, value)
        result = self.build(allow_incomplete=True)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("token totals mismatch" in e for e in result["errors"]))
        self.assertTrue(any("raw evidence hash mismatch" in e for e in result["errors"]))

    def test_incomplete_schedule_cannot_certify_a_smaller_grid(self):
        self.save(self.root / "schedule.json", self.schedule[:-1])
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("complete distinct primary" in p for p in result["pending"]))

    def test_warm_cache_wrong_lengths_and_changed_fixed_flags_fail(self):
        self.mutate_summary("p1", lambda s: s["cells"][0]["measurements"][0]["requests"][0].update(cached_tokens=8))
        path = self.root / "p2/results/launch.json"
        launch = json.loads(path.read_text())
        launch["command"] += ["--enforce-eager"]
        self.save(path, launch)
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("cached_tokens" in e for e in result["errors"]))
        self.assertTrue(any("fixed launch flags" in e for e in result["errors"]))

    def test_failed_attempt_requires_explicit_hashed_resolution(self):
        old = copy.deepcopy(self.schedule[0])
        old["id"] = "retired-high-water"
        self.add_attempt(old, state="FAIL")
        self.save_state()
        self.assertEqual(self.build()["status"], "FAIL")
        raw = self.root / old["id"] / "results/summary.json"
        path = self.root / "resolutions.json"
        value = [{"attempt_id": old["id"], "reason": "Synthetic diagnosed failure superseded by the final grid.",
                  "replacement_profile_ids": ["p0"], "rawrefs": [{"path": str(raw), "sha256": report._sha(raw)}]}]
        self.save(path, value)
        result = self.build(resolutions=path)
        self.assertEqual(result["status"], "FINAL_PASS", result["errors"])
        failure = next(a for a in result["attempts"] if a["attempt_id"] == old["id"])
        self.assertEqual(failure["raw_status"], "FAIL")
        self.assertEqual(failure["resolution_status"], "RESOLVED")
        value[0]["rawrefs"][0]["sha256"] = digest("wrong")
        self.save(path, value)
        self.assertEqual(self.build(resolutions=path)["status"], "FAIL")

    def test_matrix_only_is_not_primary_and_does_not_require_ppl(self):
        extra = {**self.schedule[0], "id": "controlled", "chunk": self.policy["chunk_candidates"][-1], "only": "matrix"}
        self.add_attempt(extra)
        self.schedule.append(extra)
        self.save(self.root / "schedule.json", self.schedule)
        self.save_state()
        result = self.build(reference="controlled")
        self.assertEqual(result["status"], "FINAL_PASS", result["errors"])
        self.assertEqual(result["controlled_chunk_profile_ids"], ["controlled"])
        self.assertEqual(len(result["primary_profile_ids"]), 9)

    def test_in_progress_is_never_final_and_missing_once_receipts_block(self):
        path = self.root / "p8/results/summary.json"
        summary = json.loads(path.read_text())
        summary.update(status="IN_PROGRESS", cells=[])
        self.save(path.with_name("progress.json"), summary)
        path.unlink()
        self.state["profiles"][-1]["status"] = "RUNNING"
        self.save_state()
        result = self.build(allow_incomplete=True)
        self.assertEqual(result["status"], "IN_PROGRESS", result["errors"])
        result = report.build_report(self.root, [self.root / "schedule.json"], allow_incomplete=False, policy=self.policy)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("gpqa" in e for e in result["pending"]))
        self.assertTrue(any("soak" in e for e in result["pending"]))

    def test_omitting_ppl_or_shortening_soak_blocks_final_pass(self):
        self.mutate_summary("p0", lambda s: s["cells"].pop(1))
        soak = json.loads(self.soak.read_text())
        soak["measurement_window_s"] -= 1
        self.save(self.soak, soak)
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("omits planned cells" in e for e in result["errors"]))
        self.assertTrue(any("full offered-load window" in e for e in result["errors"]))

    def test_v2_single_measurement_uses_repeat_one_without_median_or_range(self):
        self.version2("p0")
        result = self.build()
        self.assertEqual(result["status"], "FINAL_PASS", result["errors"])
        cell = result["attempts"][0]["cells"]["matrix_single/random/i8-o2-c1"]
        self.assertEqual(cell["metrics"]["output_tps"], 2)
        self.assertEqual(cell["reference_measurements"], [])
        self.assertEqual(result["comparisons"]["p1"]["cells"]["matrix_single/random/i8-o2-c1"]["change_pct"]["output_tps"], 0)
        self.assertTrue(result["comparisons"]["p1"]["cells"]["matrix_single/random/i8-o2-c1"]["warmup_policy_differs"])
        self.assertNotIn("median", report.markdown(result))

    def test_v1_cannot_shrink_repetitions_without_an_explicit_new_policy(self):
        path = self.root / "p0/results/plan.json"
        plan = json.loads(path.read_text())
        plan["cells"][0]["repeats"] = 1
        self.save(path, plan)
        self.mutate_summary("p0", lambda s: s["cells"][0].update(measurements=s["cells"][0]["measurements"][:1]))
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("policy v1" in e for e in result["errors"]))

    def test_v2_rejects_wrong_warmup_and_unknown_policy(self):
        _, plan, summary = self.version2("p0")
        original = copy.deepcopy(summary)
        for problem in ("budget", "dp", "version"):
            summary = copy.deepcopy(original)
            if problem == "budget":
                summary["cells"][0]["warmup"]["osl"] = 128
            elif problem == "dp":
                summary["global_warmup"]["dp_after"]["1"]["generation_tokens"] = 0
            else:
                summary["policy_version"] = 3
            self.save(self.root / "p0/results/summary.json", summary)
            self.assertEqual(self.build()["status"], "FAIL", problem)

    def test_handoff_accepts_complete_primary_and_retains_unsent_reference_scope(self):
        self.resumed(completed=2)
        result = self.build()
        self.assertEqual(result["status"], "FINAL_PASS", (result["errors"], result["pending"]))
        self.assertEqual(result["selected_attempts"]["p0"], "p0-v2")
        old = result["attempts"][0]
        self.assertEqual((old["raw_status"], old["raw_summary_status"]), ("POLICY_HANDOFF", "IN_PROGRESS"))
        self.assertEqual(old["resolution_status"], "POLICY_HANDOFF_REFERENCED")
        cell = result["attempts"][-1]["cells"]["matrix_single/random/i8-o2-c1"]
        self.assertEqual([r["repeat"] for r in cell["reference_measurements"]], [2])
        self.assertEqual(cell["metrics"]["output_tps"], 2)

    def test_handoff_after_all_three_saved_rounds_matches_the_real_boundary(self):
        self.resumed(completed=3)
        result = self.build()
        self.assertEqual(result["status"], "FINAL_PASS", (result["errors"], result["pending"]))
        self.assertEqual(len(result["attempts"][-1]["cells"]["matrix_single/random/i8-o2-c1"]["reference_measurements"]), 2)

    def test_handoff_without_a_verified_continuation_stays_pending(self):
        self.resumed()
        self.state["profiles"].pop()
        self.save_state()
        shutil.rmtree(self.root / "p0-v2")
        result = self.build(allow_incomplete=True)
        self.assertNotEqual(result["status"], "FINAL_PASS")
        self.assertTrue(any("handoff awaits" in p for p in result["pending"]))

    def test_resume_rejects_tampered_primary_raw_metrics_and_false_handoff(self):
        source, target, handoff = self.resumed()
        original = json.loads((target / "summary.json").read_text())
        self.mutate_summary("p0-v2", lambda s: s["cells"][0]["measurements"][0].update(elapsed_s=99))
        self.assertEqual(self.build()["status"], "FAIL")
        self.save(target / "summary.json", original)
        raw = next((source / "cells").glob("*/repeat-1/request-metrics.json.gz"))
        self.save(raw, [])
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("raw request metrics" in e for e in result["errors"]))
        self.save(raw, original["cells"][0]["measurements"][0]["requests"])
        value = json.loads(handoff.read_text())
        value["no_inflight"] = False
        self.save(handoff, value)
        original["resume_provenance"]["handoff"]["sha256"] = report._sha(handoff)
        self.save(target / "summary.json", original)
        self.assertEqual(self.build()["status"], "FAIL")

    def test_actual_source_failure_cannot_be_reclassified_as_policy_handoff(self):
        source, target, handoff = self.resumed()
        value = json.loads((source / "summary.json").read_text())
        value["cells"][0]["measurements"][1]["status"] = "FAIL"
        self.save(source / "summary.json", value)
        sidecar = json.loads(handoff.read_text())
        sidecar["source_snapshot_sha256"] = report._sha(source / "summary.json")
        self.save(handoff, sidecar)
        resumed = json.loads((target / "summary.json").read_text())
        resumed["resume_provenance"]["sources"]["p0"]["summary"]["sha256"] = report._sha(source / "summary.json")
        resumed["resume_provenance"]["handoff"]["sha256"] = report._sha(handoff)
        self.save(target / "summary.json", resumed)
        self.assertEqual(self.build()["status"], "FAIL")

    def test_production_policy_rejects_a_single_window_receipt_marked_pass(self):
        self.policy["ppl_windows"] = 134
        for ident in (p["id"] for p in self.schedule):
            def complete(summary):
                ppl = summary["cells"][1]["measurements"][0]
                ppl.update(windows=134, rows=ppl["rows"] * 134, scored_tokens=15 * 134)
            self.mutate_summary(ident, complete)
        self.assertEqual(self.build()["status"], "FINAL_PASS")
        self.mutate_summary("p0", lambda s: s["cells"][1]["measurements"][0].update(
            windows=1, rows=[{"status": "PASS", "scored_tokens": 15, "cached_tokens": 0}], scored_tokens=15))
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("p0:" in error and "PPL windows incomplete" in error for error in result["errors"]))

    def controlled_receipt_fixture(self, *, complete=True):
        selected = copy.deepcopy(self.schedule[0])
        selected.update(id="controlled-32k", chunk=32768, only="matrix")
        entry = self.add_attempt(selected, state="FAIL")
        self.schedule.append(selected)
        self.save(self.root / "schedule.json", self.schedule)
        self.save_state()
        primary = report._attempt(self.root.resolve(), self.state["profiles"][0], report.Evidence(), self.policy)
        launch = json.loads((self.root / entry["id"] / "results/launch.json").read_text())
        key = "matrix_single/random/i8-o2-c1"
        performance = copy.deepcopy(primary["cells"][key])
        performance.update(canonical_comparison_signature=performance["comparison_signature"],
                           actual_payload_sha256=digest("new-salt"),
                           canonical_payload_sha256=performance["comparison_signature"]["rounds"][0]["payload_sha256"])
        assessed = {
            "schema_version": 1, "kind": "controlled_chunk_assessment", "validation_errors": [],
            "status": "COMPLETE_WITH_FAILURES" if complete else "INCOMPLETE_WITH_FAILURES", "candidate_outcome": "REJECTED",
            "profile_id": entry["id"], "profile": {k: entry[k] for k in ("context", "memory", "chunk")},
            "image_id": launch["image_id"], "source_fingerprint": launch["source_manifest_sha256"],
            "frozen_inputs_sha256": digest("inputs"), "normalized_flags": report._normalized_flags(launch["command"], launch["image_id"]),
            "covered_failure_attempt_ids": [entry["id"]], "inputs": {"summaries": []},
            "coverage": {"expected": 42, "covered": 42 if complete else 5, "measured_pass": 41 if complete else 4,
                         "terminal_failed": 1, "not_run": 0 if complete else 37},
            "cells": {key: {"outcome": "PASS", "performance": performance},
                      "matrix_single/random/failed": {"outcome": "FAIL_TERMINAL", "performance": None}},
        }
        return assessed

    def test_complete_controlled_rejection_is_never_final_pass(self):
        assessed = self.controlled_receipt_fixture()
        with patch("scripts.benchmarks.profile_controlled_report.validate_receipt", return_value=assessed):
            result = self.build(controlled_assessment=self.root / "control.json")
        self.assertEqual(result["status"], "COMPLETE_WITH_FAILURES", result["errors"])
        self.assertNotIn("controlled-32k", result["selected_attempts"])
        original = next(row for row in result["attempts"] if row["attempt_id"] == "controlled-32k")
        self.assertEqual(original["raw_status"], "FAIL")
        self.assertEqual(original["resolution_status"], "ASSESSED_FAILURE")
        self.assertEqual(result["controlled_comparison"]["cells"]["matrix_single/random/failed"]["status"], "UNAVAILABLE")
        self.assertNotIn("change_pct", result["controlled_comparison"]["cells"]["matrix_single/random/failed"])

    def test_partial_controlled_rejection_keeps_37_pending(self):
        assessed = self.controlled_receipt_fixture(complete=False)
        with patch("scripts.benchmarks.profile_controlled_report.validate_receipt", return_value=assessed):
            result = self.build(controlled_assessment=self.root / "control.json", allow_incomplete=True)
        self.assertEqual(result["status"], "INCOMPLETE_WITH_FAILURES")
        self.assertTrue(any("37 of 42" in item for item in result["pending"]))

    def test_controlled_assessment_cannot_replace_32k_failure_with_16k_pass(self):
        assessed = self.controlled_receipt_fixture()
        source = self.root / "controlled-32k/results/summary.json"
        path = self.root / "resolution.json"
        self.save(path, [{"attempt_id": "controlled-32k", "reason": "16K passes", "replacement_profile_ids": ["p0"],
                          "rawrefs": [{"path": str(source), "sha256": report._sha(source)}]}])
        with patch("scripts.benchmarks.profile_controlled_report.validate_receipt", return_value=assessed):
            result = self.build(controlled_assessment=self.root / "control.json", resolutions=path)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("cannot be replaced" in error for error in result["errors"]))

    def test_omitted_controlled_failure_cannot_be_resolved_by_a_primary_pass(self):
        self.controlled_receipt_fixture()
        self.schedule = [p for p in self.schedule if p["id"] != "controlled-32k"]
        self.save(self.root / "schedule.json", self.schedule)
        source = self.root / "controlled-32k/results/summary.json"
        path = self.root / "resolution.json"
        self.save(path, [{"attempt_id": "controlled-32k", "reason": "16K passes; omit the failed candidate",
                          "replacement_profile_ids": ["p0"],
                          "rawrefs": [{"path": str(source), "sha256": report._sha(source)}]}])
        result = self.build(resolutions=path)  # Deliberately no controlled-assessment parameter.
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("historical controlled profile" in e for e in result["errors"]))
        self.assertTrue(any("controlled failures cannot be replaced" in e for e in result["errors"]))
        original = next(a for a in result["attempts"] if a["attempt_id"] == "controlled-32k")
        self.assertEqual((original["raw_status"], original["resolution_status"]), ("FAIL", "UNRESOLVED"))

    def test_listed_controlled_failure_still_requires_assessment_without_parameter(self):
        self.controlled_receipt_fixture()
        source = self.root / "controlled-32k/results/summary.json"
        path = self.root / "resolution.json"
        self.save(path, [{"attempt_id": "controlled-32k", "reason": "Use a primary configuration instead",
                          "replacement_profile_ids": ["p0"],
                          "rawrefs": [{"path": str(source), "sha256": report._sha(source)}]}])
        result = self.build(resolutions=path)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("independent controlled assessment" in e for e in result["errors"]))

    def test_even_passed_historical_controlled_profile_must_remain_in_schedule(self):
        profile = {**self.schedule[0], "id": "controlled-completed", "chunk": 32768, "only": "matrix"}
        self.add_attempt(profile)
        self.save_state()
        result = self.build()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("historical controlled profile" in e for e in result["errors"]))

    def test_primary_semantic_failure_can_still_reference_a_validated_same_profile_retry(self):
        self.add_attempt(self.schedule[0], attempt_id="p0-highres-failed", state="FAIL")
        failed = self.state["profiles"].pop()
        self.state["profiles"].insert(0, failed)
        self.save_state()
        source = self.root / "p0-highres-failed/results/summary.json"
        path = self.root / "resolution.json"
        self.save(path, [{"attempt_id": "p0-highres-failed",
                          "reason": "Original semantic failure retained; validated same-profile retry passed.",
                          "replacement_profile_ids": ["p0"],
                          "rawrefs": [{"path": str(source), "sha256": report._sha(source)}]}])
        result = self.build(resolutions=path)
        self.assertEqual(result["status"], "FINAL_PASS", result["errors"])
        original = next(a for a in result["attempts"] if a["attempt_id"] == "p0-highres-failed")
        self.assertEqual((original["raw_status"], original["resolution_status"]), ("FAIL", "RESOLVED"))

    def test_complete_controlled_assessment_still_requires_primary_and_once_quality(self):
        assessed = self.controlled_receipt_fixture()
        with patch("scripts.benchmarks.profile_controlled_report.validate_receipt", return_value=assessed):
            result = report.build_report(self.root, [self.root / "schedule.json"], policy=self.policy,
                                         controlled_assessment=self.root / "control.json", soak_result=self.soak)
        self.assertEqual(result["status"], "INCOMPLETE_WITH_FAILURES")
        self.assertTrue(any("gpqa" in item for item in result["pending"]))
        self.mutate_summary("p0", lambda value: value["cells"].pop(1))
        with patch("scripts.benchmarks.profile_controlled_report.validate_receipt", return_value=assessed):
            self.assertEqual(self.build(controlled_assessment=self.root / "control.json")["status"], "FAIL")

    def test_unvalidated_decision_reject_does_not_change_original_failure(self):
        self.controlled_receipt_fixture()
        path = self.root / "control.json"
        self.save(path, {"schema_version": 1, "kind": "controlled_chunk_assessment", "decision": "reject",
                         "status": "COMPLETE_WITH_FAILURES", "coverage": {"expected": 42, "not_run": 0}})
        result = self.build(controlled_assessment=path)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("invalid controlled" in error for error in result["errors"]))
        self.assertEqual(self.build()["status"], "FAIL")  # No new parameter preserves the old contract.


if __name__ == "__main__":
    unittest.main()
