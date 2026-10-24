"""Profiling inputs must remain identical across deployments and cache-isolated."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmarks import profile_workload as workload
from scripts.benchmarks.profile_matrix import resolve_cell
from scripts.benchmarks.profile_mixed import schedule
from scripts.benchmarks.profile_plan import group_plan


class FrozenWorkloadTests(unittest.TestCase):
    def frozen(self):
        return {"run_id": "same-campaign", "token_pool": [100, 200, 300],
                "nature_tokens": list(range(100, 500)), "bodies": {}, "observed_input_tokens": {}}

    def test_matrix_requests_are_reproducible_and_salts_are_protocol_safe(self):
        cell = next(c for c in group_plan(1048576)["cells"] if c["case_id"] == "matrix_concurrent")
        a = workload.request_body(self.frozen(), cell, "repeat-1", 0)
        b = workload.request_body(self.frozen(), cell, "repeat-1", 0)
        other = workload.request_body(self.frozen(), cell, "repeat-2", 0)
        self.assertEqual(a, b)
        self.assertEqual(len(a["prompt"]), cell["isl"])
        self.assertEqual(a["max_tokens"], 1000)
        self.assertNotEqual(a["cache_salt"], other["cache_salt"])
        self.assertNotIn("/", a["cache_salt"])
        self.assertLessEqual(len(a["cache_salt"]), 128)

    def test_real_plan_chat_prefill_resolves_actual_capacity_without_truncation(self):
        cell = next(c for c in group_plan(262144)["cells"] if c["kind"] == "chat_prefill")
        frozen = self.frozen()
        key = f"single_prefill-{cell['prompt_repetitions']}"
        frozen["bodies"][key] = {"model": workload.MODEL, "messages": [], "max_tokens": 1}
        frozen["observed_input_tokens"][key] = 262144
        result = resolve_cell(cell, frozen, 262144)
        self.assertEqual(result["status"], "NOT_APPLICABLE")
        self.assertIn("not truncated", result["reason"])
        body = workload.request_body(frozen, cell, "repeat-1", 0)
        self.assertEqual(body["max_tokens"], 1)

    def test_prepare_deduplicates_c128_and_uses_catalog_needles(self):
        regular = {"cases": [
            {"id": "matrix", "kind": "matrix", "isl": [4]},
            {"id": "short", "kind": "chat_burst", "prompt": "same C{n}", "osl": 128,
             "thinking": False, "runs": [{"concurrency": 128}, {"concurrency": 128}]},
        ]}
        def config(name):
            return regular if name == "regular" else {"cases": []}
        def needle(_base, target):
            return {"name": f"needle-target-{target}", "messages": [], "expected": {}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "inputs.json.gz"
            (Path(directory) / "long-1200.mp4").write_bytes(b"media")
            def probe_length(_base, _body, name, _out):
                return int(name.split("-")[-1]) if name.startswith("needle-target-") else 12
            with patch.object(workload, "load_config", side_effect=config), \
                    patch.object(workload, "load_fixtures", return_value={"images": {}, "videos": {}}), \
                    patch.object(workload, "token_pool", return_value=[10]), \
                    patch.object(workload, "build_prompts", return_value=[[1, 2, 3, 4]]), \
                    patch.object(workload, "json_request", return_value={"tokens": [1, 2]}), \
                    patch.object(workload, "_probe", side_effect=probe_length) as probe, \
                    patch.object(workload, "acceptance_catalog", return_value={"needle_tokens": [123, 456]}), \
                    patch.object(workload.core, "needle_case", side_effect=needle), \
                    patch.object(workload, "highres_video_case"), \
                    patch.object(workload, "source_identity", return_value={"commit": "fixed"}):
                result = workload.prepare(workload.BASE_URL, Path(directory), output, "run")
            self.assertEqual(probe.call_count, 3)  # One C128 probe and two catalog needles.
            self.assertIn("needle-target-123", result["bodies"])
            self.assertIn("needle-target-456", result["bodies"])
            self.assertEqual(workload.load(output), result)

    def test_mixed_schedule_has_identical_fixed_request_counts(self):
        plan = {"decode_batches": 5, "decode_requests_per_batch": 64,
                "decode_batch_interval_s": 60, "prefill_requests": 15, "prefill_interval_s": 20}
        rows = schedule(plan)
        self.assertEqual(len(rows), 335)
        self.assertEqual(sum(r["stream"] == "decode" for r in rows), 320)
        self.assertEqual(sum(r["stream"] == "prefill" for r in rows), 15)
        self.assertEqual(max(r["at_s"] for r in rows), 280)


if __name__ == "__main__":
    unittest.main()
