"""Offline evidence must be complete and comparable before FINAL_PASS."""

import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmarks import profile_report as report


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class ProfileReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = report.default_policy()
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
        rounds = [{**self.measurement([self.row()], elapsed), "payload_sha256": digest(f"repeat-{i}")}
                  for i, elapsed in enumerate((1, 2, 4))]
        actual = [{"cell_id": cells[0]["cell_id"], "case_id": "matrix_single", "status": "PASS", "measurements": rounds}]
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
                   "global_warmup": {"status": "PASS"}, "observation": {"status": "PASS"}, "cells": actual}
        self.save(directory / "launch.json", launch)
        self.save(directory / "summary.json", summary)
        return entry

    def build(self, **kwargs):
        return report.build_report(self.root, [self.root / "schedule.json"],
                                   gpqa_result=self.gpqa, soak_result=self.soak, **kwargs)

    def mutate_summary(self, profile, change):
        path = self.root / profile / "results/summary.json"
        value = json.loads(path.read_text())
        change(value)
        self.save(path, value)

    def test_complete_grid_and_once_evidence_can_pass(self):
        result = self.build()
        self.assertEqual(result["status"], "FINAL_PASS", (result["errors"], result["pending"]))
        self.assertEqual(len(result["primary_profile_ids"]), len(self.policy["contexts"]) * self.policy["memory_levels"])
        metrics = result["attempts"][0]["cells"]["matrix_single/random/i8-o2-c1"]["metrics"]
        self.assertEqual(metrics["output_tps"], {"median": 1, "min": .5, "max": 2})
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
        self.mutate_summary("p1", lambda s: s["cells"][0]["measurements"][1].update(payload_sha256=digest("different")))
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
        result = report.build_report(self.root, [self.root / "schedule.json"], allow_incomplete=False)
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


if __name__ == "__main__":
    unittest.main()
