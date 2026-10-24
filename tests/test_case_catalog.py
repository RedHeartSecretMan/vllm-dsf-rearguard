"""Repository contracts for the benchmark and acceptance case catalogs."""

import importlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


class CaseCatalogTests(unittest.TestCase):
    def test_duplicate_scenarios_catalog_was_removed(self):
        self.assertFalse((ROOT / "config/benchmarks/scenarios.json").exists())
        self.assertFalse((ROOT / "scripts/benchmark_scenarios.py").exists())

    def test_long_tail_keeps_the_four_named_cases(self):
        config = read("config/benchmarks/long_tail.json")
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

    def test_acceptance_catalogs_have_unique_ids_and_importable_runners(self):
        expected = {"core": 8, "visual": 10, "responses": 7, "quality": 2}
        ids = []
        for name, count in expected.items():
            with self.subTest(catalog=name):
                config = read(f"config/acceptance/{name}.json")
                cases = config["cases"]
                self.assertEqual(len(cases), count)
                case_ids = [case["id"] for case in cases]
                self.assertEqual(len(case_ids), len(set(case_ids)))
                self.assertEqual(config["model"], "deepseek-ai/DeepSeek-V4.1-Flash-1017")
                importlib.import_module(config["runner"])
                ids.extend(case_ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_core_catalog_matches_the_existing_core_runner_cases(self):
        from scripts.acceptance import core

        configured = {case["id"] for case in read("config/acceptance/core.json")["cases"]}
        generated = {case["name"] for case in core.cases(True)}
        self.assertEqual(configured, generated)

    def test_benchmark_and_acceptance_ids_do_not_collide(self):
        benchmark_ids = []
        for name in ("regular", "long_tail", "ab", "excluded"):
            benchmark_ids.extend(
                case["id"] for case in read(f"config/benchmarks/{name}.json")["cases"]
            )
        acceptance_ids = []
        for name in ("core", "visual", "responses", "quality"):
            acceptance_ids.extend(
                case["id"] for case in read(f"config/acceptance/{name}.json")["cases"]
            )
        self.assertFalse(set(benchmark_ids) & set(acceptance_ids))

    def test_benchmark_catalogs_name_their_runner(self):
        expected = {
            "regular": "scripts.benchmarks.bench_regular",
            "long_tail": "scripts.benchmarks.bench_long_tail",
            "ab": "scripts.benchmarks.bench_ab",
            "excluded": "scripts.benchmarks.bench_excluded",
        }
        for name, runner in expected.items():
            with self.subTest(catalog=name):
                config = read(f"config/benchmarks/{name}.json")
                self.assertEqual(config["runner"], runner)
                importlib.import_module(runner)

    def test_visual_limits_match_runtime_contract(self):
        limits = read("config/acceptance/visual.json")["limits"]
        self.assertEqual(
            limits,
            {
                "max_images_per_request": 64,
                "max_videos_per_request": 2,
                "max_frames_per_video": 32,
                "default_fps": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
