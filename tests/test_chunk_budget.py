"""Frozen A/B payloads, profile isolation and measured cold-cache contracts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.bench_common import workload_digest
from scripts.benchmarks import bench_ab, chunk_budget


class ChunkBudgetTests(unittest.TestCase):
    def case(self):
        _, case = chunk_budget.case_config()
        return case

    def receipt(self, profile="32k_mm_off"):
        chunk = "16384" if profile == "16k_mm_on" else "32768"
        flag = "--disable-chunked-mm-input" if profile == "32k_mm_off" else "--no-disable-chunked-mm-input"
        command = ["serve", "model", "--tensor-parallel-size", "4", "--data-parallel-size", "2",
                   "--data-parallel-size-local", "2", "--max-model-len", "1048576",
                   "--max-num-seqs", "128", "--max-num-batched-tokens", chunk,
                   "--gpu-memory-utilization", "0.9", "--api-server-count", "1",
                   "--kv-cache-dtype", "fp8_ds_mla", flag, "--enable-eplb",
                   "--enable-prefix-caching", "--enable-chunked-prefill", "--enable-prompt-tokens-details",
                   "--speculative-config", '{"method":"dspark","num_speculative_tokens":5}',
                   "--limit-mm-per-prompt", '{"image":64,"video":{"count":2,"num_frames":32}}',
                   "--media-io-kwargs", '{"video":{"num_frames":32,"fps":1}}']
        for key, value in {"OMP_NUM_THREADS": "8", "VLLM_ENABLE_RESPONSES_API_STORE": "1",
                           "VLLM_DSV41_STACKED_DRAFT_KV": "1", "VLLM_DSV41_BOUNDED_PREFILL_WORKSPACE": "1",
                           "VLLM_DSV41_PREFILL_THROTTLE": "0", "VLLM_SM80_EP8_CUSTOM_AG_RS": "1"}.items():
            command += ["--env", f"{key}={value}"]
        return {"command": command}

    def row(self, tokens=20000, cache=0):
        return {"status": "PASS", "input_tokens": tokens, "output_tokens": 128,
                "ttft_s": 2.0, "e2e_s": 4.0, "tpot_s": 0.01, "decode_span_s": 1.27,
                "itl_s": [0.1], "cached_tokens": cache}

    def test_catalog_has_three_profiles_and_eleven_cells(self):
        case = self.case()
        self.assertEqual([v["id"] for v in case["variants"]], ["32k_mm_off", "32k_mm_on", "16k_mm_on"])
        self.assertEqual(len(case["workload"]["cells"]), 11)
        self.assertEqual(case["workload"]["repeats"], 3)
        self.assertEqual(case["runner"], "scripts.benchmarks.chunk_budget")
        with self.assertRaisesRegex(SystemExit, "scripts.benchmarks.chunk_budget"):
            bench_ab.measure("unused", case, case["variants"][0], 1)

    def test_frozen_payloads_are_deterministic_and_repeats_are_cold(self):
        case = copy.deepcopy(self.case())
        case["workload"]["cells"] = [
            {"id": "text", "kind": "text", "isl": 16, "osl": 128, "concurrency": 2},
            {"id": "video", "kind": "video", "min_input_tokens": 16385, "osl": 128, "concurrency": 1},
        ]
        media = {"images": {str(n): {"path": f"{n}.png"} for n in (1, 2, 3)}, "video": {"path": "long.mp4", "video_id": "vid_deadbeef"}}
        with patch.object(chunk_budget, "media_part", side_effect=lambda p, k: {"type": f"{k}_url", f"{k}_url": {"url": str(p)}}):
            a = chunk_budget.build_workload(case, [100, 200, 300, 400], media, "run")
            b = chunk_budget.build_workload(case, [100, 200, 300, 400], media, "run")
        self.assertEqual(a, b)
        self.assertNotEqual(a["cells"][0]["repeats"][0]["bodies"], a["cells"][0]["repeats"][1]["bodies"])
        self.assertEqual(len(a["cells"][0]["repeats"][0]["bodies"][0]["prompt"]), 16)
        for cell in a["cells"]:
            for repeat in cell["repeats"]:
                for body in repeat["bodies"]:
                    self.assertTrue(body["ignore_eos"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workload.json"
            path.write_text(json.dumps(a))
            self.assertEqual(chunk_budget.load_workload(path), a)
            a["cells"][0]["repeats"][0]["bodies"][0]["prompt"][0] += 1
            path.write_text(json.dumps(a))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                chunk_budget.load_workload(path)

    def test_workload_rejects_reused_salt_even_with_updated_digest(self):
        artifact = {"schema_version": 1, "case": "chunk_budget", "model": chunk_budget.MODEL,
                    "cells": [{"concurrency": 1, "warmup": {"cache_salt": "same"},
                               "repeats": [{"bodies": [{"cache_salt": "same"}],
                                            "payload_sha256": workload_digest([{"cache_salt": "same"}])}]}]}
        artifact["sha256"] = workload_digest(artifact)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workload.json"
            path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "reuses a cache salt"):
                chunk_budget.load_workload(path)

    def test_profile_checks_settings_that_can_confound_comparison(self):
        for profile in self.case()["variants"]:
            receipt = self.receipt(profile["id"])
            chunk_budget.verify_profile(profile, receipt)
            for flag in ("--max-num-batched-tokens", "--tensor-parallel-size", "--max-num-seqs"):
                wrong = copy.deepcopy(receipt)
                wrong["command"][wrong["command"].index(flag) + 1] = "999"
                with self.assertRaisesRegex(ValueError, "profile requires"):
                    chunk_budget.verify_profile(profile, wrong)
        receipt = self.receipt()
        receipt["command"].remove("--enable-eplb")
        with self.assertRaisesRegex(ValueError, "enable-eplb"):
            chunk_budget.verify_profile(self.case()["variants"][0], receipt)

    def test_measure_requires_cold_cache_and_a_real_video_boundary(self):
        cell = {"id": "video", "kind": "video", "osl": 128, "concurrency": 1, "min_input_tokens": 16385}
        for tokens, cache, status in ((20000, 0, "PASS"), (8000, 0, "FAIL"), (20000, 4, "FAIL"), (20000, None, "FAIL")):
            with patch.object(chunk_budget, "run_burst", return_value=([self.row(tokens, cache)], 4.0)):
                result = chunk_budget.measure("local", cell, [{}], 10)
            self.assertEqual(result["status"], status)
            if status == "PASS":
                self.assertEqual(result["input_tps"], 5000)
                self.assertEqual(result["output_tps"], 32)
                self.assertEqual(result["total_tps"], 5032)

    def test_text_measurement_requires_exact_input_and_output_counts(self):
        cell = {"id": "text", "kind": "text", "isl": 1024, "osl": 128, "concurrency": 64}
        with patch.object(chunk_budget, "run_burst", return_value=([self.row(1024)], 4.0)) as burst:
            chunk_budget.measure("local", cell, [{}], 10)
        burst.assert_called_once_with("local", [{}], 1, 10, expected_output=128, expected_input=1024)


if __name__ == "__main__":
    unittest.main()
