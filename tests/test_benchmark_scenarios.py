"""Configuration contracts for the four long-tail benchmark scenarios."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkScenarioConfigTests(unittest.TestCase):
    def test_fixed_isl_osl_and_case_kinds(self):
        config = json.loads((ROOT / "config/benchmarks/scenarios.json").read_text())
        cases = {case["id"]: case for case in config["cases"]}
        self.assertEqual(
            set(cases),
            {
                "tool_history_16384_c16",
                "vision_mix_4096_c4",
                "long_soak_1024_c64",
                "mixed_prefill_decode_same_service",
            },
        )
        for case_id in (
            "tool_history_16384_c16",
            "vision_mix_4096_c4",
            "long_soak_1024_c64",
        ):
            self.assertEqual(cases[case_id]["osl"], 1000)
        mixed = cases["mixed_prefill_decode_same_service"]
        self.assertEqual(mixed["decode_isl"], 1024)
        self.assertEqual(mixed["decode_osl"], 1000)
        self.assertEqual(mixed["prefill_isl"], 262144)
        self.assertEqual(mixed["prefill_osl"], 1000)
