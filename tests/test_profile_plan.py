"""Catalog-derived expansion, launch applicability, and non-truncating admission."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmarks import profile_plan


class ProfilePlanTests(unittest.TestCase):
    def test_all_source_cases_and_ab_variants_are_audited(self):
        inventory = profile_plan.catalog_inventory()
        for catalog, entry in inventory["catalogs"].items():
            config = json.loads((profile_plan.CONFIG_DIR / (catalog + ".json")).read_text())
            self.assertEqual(entry["case_count"], len(config["cases"]))
            self.assertTrue({c["id"] for c in config["cases"]} <= {r["source_case"]["id"] for r in entry["cells"]})
            if catalog == "benchmarks/ab":
                for case in config["cases"]:
                    actual = {r["variant"] for r in entry["cells"] if r["case_id"] == case["id"]}
                    self.assertEqual(actual, {v["id"] for v in case["variants"]})
        cells = [c for entry in inventory["catalogs"].values() for c in entry["cells"]]
        self.assertEqual(len(cells), len({c["cell_id"] for c in cells}))
        self.assertEqual(inventory["counts"]["audit_cells"], len(cells))
        self.assertEqual(inventory["counts"]["derived_cells"], 3)

    def test_matrix_expansion_comes_from_source_dimensions(self):
        original = Path.read_text
        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path == profile_plan.CONFIG_DIR / "benchmarks/regular.json":
                config = json.loads(text)
                case = next(c for c in config["cases"] if c["id"] == "matrix_single")
                case["isl"] = [2048, 8192]
                return json.dumps(config)
            return text
        with patch.object(Path, "read_text", changed):
            rows = profile_plan.catalog_inventory()["catalogs"]["benchmarks/regular"]["cells"]
        self.assertEqual(len([r for r in rows if r["case_id"] == "matrix_single"]), 4)
        self.assertEqual({r["isl"] for r in rows if r["case_id"] == "matrix_single"}, {2048, 8192})

    def test_context_overrides_history_but_never_truncates(self):
        for context, applicable in ((262144, 38), (524288, 40), (1048576, 42)):
            with self.subTest(context=context):
                plan = profile_plan.group_plan(context)
                matrix = [c for c in plan["cells"] if c["case_id"] in ("matrix_single", "matrix_concurrent")]
                self.assertEqual(len(matrix), 42)
                self.assertEqual(sum(c["status"] == "APPLICABLE" for c in matrix), applicable)
                for row in matrix:
                    self.assertEqual(row["status"] == "APPLICABLE", row["isl"] + row["osl"] <= context)
                    self.assertEqual(row["requested_repeats"], 3)

    def test_needle_admission_includes_output_budget(self):
        plan = profile_plan.group_plan(524288)
        rows = {r["case_id"]: r for r in plan["cells"] if r["kind"] == "needle"}
        self.assertEqual(rows["needle-target-523000"]["admission_tokens"], 523512)
        self.assertEqual(rows["needle-target-523000"]["status"], "PENDING_PREPARATION")
        self.assertEqual(rows["needle-target-1047000"]["status"], "NOT_APPLICABLE")
        self.assertEqual(rows["needle-target-523000"]["repeats"], 1)

    def test_historical_subprofiles_and_derived_capacity_are_distinct(self):
        plan = profile_plan.group_plan(1048576)
        rows = plan["cells"]
        old_short = [r for r in rows if r["case_id"] == "short_concurrency" and r["max_seqs"] == 128]
        self.assertEqual(len(old_short), 2)
        self.assertTrue(all(r["status"] == "NOT_APPLICABLE" for r in old_short))
        old_capacity = [r for r in rows if r["case_id"] == "long_request_capacity"]
        self.assertTrue(all(r["status"] == "NOT_APPLICABLE" for r in old_capacity))
        derived = [r for r in rows if r["case_id"].startswith("capacity_fixed_profile_")]
        self.assertEqual({r["cell_id"] for r in derived}, {"capacity_fixed_profile_c4", "capacity_fixed_profile_c16"})
        for row in derived:
            self.assertEqual(row["status"], "APPLICABLE")
            self.assertEqual(row["repeats"], 1)
            self.assertEqual((row["isl"], row["osl"], row["seed"]), (1047000, 1000, 20261023))
            self.assertIn("prefill-throttle", row["source_case"]["extra_launch"])
            self.assertEqual(row["derivation"]["changes"], {"prefill_throttle": False})
        throttled = profile_plan.group_plan(1048576, throttle=True)["cells"]
        self.assertTrue(all(r["status"] == "APPLICABLE" for r in throttled if r["case_id"] == "long_request_capacity"))
        self.assertTrue(all(r["status"] == "NOT_APPLICABLE" for r in throttled if r["case_id"].startswith("capacity_fixed_profile_")))

    def test_visual_modes_boundary_and_highres_semantics(self):
        for chunk in (16384, 32768):
            rows = [r for r in profile_plan.group_plan(1048576, chunk=chunk)["cells"] if r["group"] == "visual"]
            rejected = {r["case_id"] for r in rows if r["status"] == "NOT_APPLICABLE"}
            self.assertEqual(rejected, {"visual-video-chunked-disabled", "visual-video-chunked-frame-boundary"})
            self.assertEqual(sum(r["status"] == "PENDING_PREPARATION" for r in rows), 9)
            highres = next(r for r in rows if r["case_id"] == "visual-video-highres-semantic")
            self.assertEqual(highres["visual_tokens"], 27904)
            self.assertEqual(highres["frame_width"], 1200)
            self.assertEqual(highres["source_case"]["id"], "visual-video-long")

    def test_quality_and_soak_once_policy_does_not_skip_required_work(self):
        plan = profile_plan.group_plan(1048576)
        once = [r for r in plan["cells"] if r["status"] == "ONCE_ONLY"]
        self.assertEqual({r["case_id"] for r in once}, {"quality-gpqa-diamond", "long_soak_1024_c64"})
        self.assertTrue(all(r["repeats"] == 1 for r in once))
        ppl = next(r for r in plan["cells"] if r["case_id"] == "quality-ppl-wikitext")
        self.assertEqual((ppl["status"], ppl["repeats"], ppl["isl"], ppl["osl"]), ("APPLICABLE", 1, 2048, 1))
        self.assertIn("all_ppl_windows", ppl["requires"])

    def test_unknown_chat_lengths_remain_pending_and_counted(self):
        plan = profile_plan.group_plan(1048576)
        for row in plan["cells"]:
            if row["status"] == "PENDING_PREPARATION":
                self.assertIsNone(row["actual_expected_input"])
                self.assertIn("tokenize_before_run", row["requires"])
                self.assertEqual(row["length_check"], "PENDING")
        counts = plan["counts"]
        self.assertEqual(counts["total"], counts["applicable"] + counts["not_applicable"] + counts["once_only"])
        self.assertEqual(counts["repeats"], sum(r["repeats"] for r in plan["cells"] if r["status"] in ("APPLICABLE", "PENDING_PREPARATION")))
        self.assertEqual(counts["once_repeats"], 2)

    def test_repeats_and_mixed_admission_match_the_requested_categories(self):
        plan = profile_plan.group_plan(1048576)
        three = {"short_concurrency", "single_prefill", "single_decode",
                 "matrix_single", "matrix_concurrent", "tool_history_16384_c16", "vision_mix_4096_c4"}
        for row in plan["cells"]:
            if row["case_id"] in three:
                self.assertEqual(row["requested_repeats"], 3)
            if row["catalog"].startswith("acceptance/"):
                self.assertEqual(row["requested_repeats"], 1)
        mixed = next(r for r in plan["cells"] if r["kind"] == "mixed" and r["group"] == "long_tail")
        self.assertEqual(mixed["repeats"], 1)
        self.assertEqual(mixed["admission_tokens"], max(f["isl"] + f["osl"] for f in mixed["flows"].values()))
        small = profile_plan.group_plan(262144)
        mixed_small = next(r for r in small["cells"] if r["case_id"] == mixed["case_id"])
        self.assertEqual(mixed_small["status"], "NOT_APPLICABLE")

    def test_planning_does_not_mutate_catalogs_or_other_calls(self):
        first = profile_plan.catalog_inventory()
        snapshot = copy.deepcopy(first)
        profile_plan.group_plan(262144)
        self.assertEqual(first, snapshot)
        self.assertEqual(first, profile_plan.catalog_inventory())
        for parameters in ({"context": 0}, {"context": 1048576, "max_seqs": 3}, {"context": 1048576, "memory": 0}):
            with self.assertRaises(ValueError):
                profile_plan.group_plan(**parameters)


if __name__ == "__main__":
    unittest.main()
