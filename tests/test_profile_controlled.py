"""Salt-only replay, retained PASS provenance and bounded controlled subsets."""

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmarks import profile_controlled as controlled
from scripts.benchmarks import profile_workload as inputs


def measurement(bodies, spec, passed=True):
    rows = [{"index": i, "status": "PASS", "cached_tokens": 0,
             "input_tokens": len(b["prompt"]), "output_tokens": b["max_tokens"]}
            for i, b in enumerate(bodies)]
    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    return {"status": "PASS" if passed else "FAIL", "cache_verified": passed,
            "request_count": len(rows), "success": len(rows) if passed else 0,
            "failure_rate": 0 if passed else 1, "elapsed_s": 2,
            "input_tps": total_in / 2, "output_tps": total_out / 2,
            "total_tps": (total_in + total_out) / 2,
            "errors": [] if passed else ["synthetic terminal failure"], "requests": rows,
            "payload_sha256": controlled.bench_common.workload_digest(bodies),
            "concurrency": spec["concurrency"], "osl": spec["osl"]}


class FakeObserver:
    def __init__(self, _base, _container, output):
        self.output = output

    def __enter__(self):
        self.output.mkdir(parents=True)
        (self.output / "docker.log").write_text("synthetic diagnostic service log\n")
        return self

    def __exit__(self, *_args):
        return False

    def snapshot(self):
        return {f'vllm:{kind}_total{{engine="{engine}"}}': 0
                for engine in (0, 1) for kind in ("prompt_tokens", "generation_tokens")}

    def summary(self):
        return {"status": "PASS"}


class ControlledTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.old = self.root / "old-attempt/results"
        self.old.mkdir(parents=True)
        self.frozen = {"run_id": "campaign", "token_pool": [10, 20, 30, 40],
                       "nature_tokens": list(range(100, 500)), "source": {"source_manifest_sha256": "source"}}
        self.frozen["sha256"] = controlled.bench_common.workload_digest(self.frozen)
        self.input_path = self.root / "inputs.json.gz"
        controlled.write(self.input_path, self.frozen)
        self.profile = {"context": 1048576, "memory": .90, "chunk": 32768}
        self.launch = {"image_id": "image", "source_manifest_sha256": "source", "container_id": "old",
                       "command": ["docker", "run", "--name", "old", "--mount", "type=volume,src=old-cache,dst=/workspace",
                                   "image", "serve", "--max-model-len", "1048576", "--gpu-memory-utilization", "0.9",
                                   "--max-num-batched-tokens", "32768"]}
        self.cells = []
        for dataset, length in (("random", 8), ("random", 16), ("random", 32), ("random", 64),
                                ("random", 128), ("nature", 32), ("nature", 128)):
            self.cells.append({"cell_id": f"matrix_single/{dataset}/i{length}-o1000-c1",
                               "case_id": "matrix_single", "group": "regular", "kind": "matrix", "dataset": dataset,
                               "isl": length, "osl": 1000, "concurrency": 1, "repeats": 3,
                               "source_case": {"id": "matrix_single", "seed": 1017}, "status": "APPLICABLE"})
        self.summary = {"status": "FAIL", "profile": self.profile,
                        "endpoint": {"image_id": "image", "container_id": "old", "started_at": "old-start"},
                        "source": {"source_manifest_sha256": "source"}, "frozen_inputs_sha256": self.frozen["sha256"],
                        "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:01:00+00:00", "cells": []}
        for spec in self.cells[:4]:
            directory = self.old / "cells" / spec["cell_id"].replace("/", "__")
            warm = {"status": "PASS", "both_dp_active": True}
            controlled.write(directory / "warmup/report.json", warm)
            row = {"cell_id": spec["cell_id"], "status": "PASS", "warmup": warm, "measurements": []}
            for repeat in range(1, 4):
                bodies = controlled._canonical(self.frozen, spec, f"repeat-{repeat}", 1)
                result = measurement(bodies, spec)
                row["measurements"].append(result)
                self.raw(directory / f"repeat-{repeat}", bodies, result)
            self.summary["cells"].append(row)
        spec = self.cells[4]
        bodies = controlled._canonical(self.frozen, spec, f"{spec['cell_id']}-warmup-1", 2, 128)
        result = measurement(bodies, {**spec, "concurrency": 2, "osl": 128}, False)
        warm = {"status": "FAIL", "concurrency": 2, "osl": 128, "rounds": [{"measurement": result}]}
        directory = self.old / "cells" / spec["cell_id"].replace("/", "__") / "warmup"
        self.raw(directory / "round-1", bodies, result)
        controlled.write(directory / "report.json", warm)
        self.summary["cells"].append({"cell_id": spec["cell_id"], "status": "FAIL", "warmup": warm, "measurements": []})
        controlled.write(self.old / "summary.json", self.summary)
        controlled.write(self.old / "plan.json", {"profile": self.profile, "cells": self.cells})
        controlled.write(self.old / "launch.json", self.launch)
        (self.old / "observation").mkdir()
        (self.old / "observation/docker.log").write_text("original failed attempt log\n")
        with patch.object(controlled, "_catalog_cells", return_value=self.cells):
            self.manifest = controlled.build_manifest(self.old, self.input_path)
        self.manifest_path = self.root / "manifest.json"
        controlled.write(self.manifest_path, self.manifest)

    def raw(self, directory, bodies, result):
        controlled.write(directory / "request-payloads.json.gz", bodies)
        controlled.write(directory / "request-metrics.json.gz", result["requests"])
        controlled.write(directory / "report.json", result)

    def test_plan_retains_individual_passes_and_orders_long_failure_last(self):
        self.assertEqual(self.manifest["source_attempt"]["status"], "FAIL")
        self.assertEqual(self.manifest["counts"], {"expected": 7, "retained_pass": 4, "pending": 3,
                                                   "source_failed_warmup": 1, "source_not_run": 2})
        self.assertEqual(self.manifest["pending_cell_ids"], [self.cells[5]["cell_id"], self.cells[6]["cell_id"], self.cells[4]["cell_id"]])
        self.assertEqual(self.manifest["cells"], self.cells)
        self.assertTrue(all(len(row["supplemental_repeats"]) == 2 for row in self.manifest["retained"]))
        self.assertEqual(controlled.load_manifest(self.manifest_path), self.manifest)

    def test_source_mutation_and_retained_requests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "retained or unknown"):
            controlled.choose_cells(self.manifest, [self.cells[0]["cell_id"]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            controlled.choose_cells(self.manifest, [self.cells[5]["cell_id"]] * 2)
        (self.old / "observation/docker.log").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            controlled.load_manifest(self.manifest_path)

    def test_a_consistently_saved_warm_source_cannot_be_retained_as_cold(self):
        row = self.summary["cells"][0]
        row["measurements"][0]["requests"][0]["cached_tokens"] = 8
        folder = self.old / "cells" / row["cell_id"].replace("/", "__") / "repeat-1"
        (self.old / "summary.json").write_text(json.dumps(self.summary))
        (folder / "report.json").write_text(json.dumps(row["measurements"][0]))
        (folder / "request-metrics.json.gz").unlink()
        controlled.write(folder / "request-metrics.json.gz", row["measurements"][0]["requests"])
        with patch.object(controlled, "_catalog_cells", return_value=self.cells), self.assertRaisesRegex(ValueError, "lengths/cache/status"):
            controlled.build_manifest(self.old, self.input_path)

    def test_salt_rotation_does_not_regenerate_random_prompt(self):
        cell = self.cells[0]
        original = inputs.request_body(self.frozen, cell, "repeat-1", 0)
        sent = controlled.salted_body(original, "new", "remaining", cell["cell_id"], "repeat-1", 0)
        self.assertEqual(controlled._without_salt(original), controlled._without_salt(sent))
        self.assertNotEqual(original["cache_salt"], sent["cache_salt"])
        self.assertNotEqual(original["prompt"], inputs.request_body(self.frozen, cell, "new-label", 0)["prompt"])

    def test_archive_proves_actual_salt_only_changes(self):
        original = controlled._canonical(self.frozen, self.cells[0], "repeat-1", 1)
        sent = [controlled.salted_body(original[0], "next", "remaining", self.cells[0]["cell_id"], "repeat-1", 0)]
        directory = self.root / "replay"
        controlled.write(directory / "request-payloads.json.gz", sent)
        proof = controlled.archive_replay(directory, original, self.manifest["source_refs"])
        self.assertNotEqual(proof["canonical_payload_sha256"], proof["actual_payload_sha256"])
        self.assertEqual(controlled.read(proof["canonical_payloads_ref"]["path"]), original)
        bad = copy.deepcopy(sent)
        bad[0]["max_tokens"] += 1
        other = self.root / "bad"
        controlled.write(other / "request-payloads.json.gz", bad)
        with self.assertRaisesRegex(ValueError, "besides cache_salt"):
            controlled.archive_replay(other, original, {})

    def runtime(self, phase, *, fail=False, observer_type=FakeObserver, startup_proof=None, fail_warmup=False):
        live = copy.deepcopy(self.launch)
        live["container_id"] = "new"
        live["command"] = [x.replace("old-cache", "new-cache") if "old-cache" in x else "new" if x == "old" else x
                           for x in live["command"]]
        live_path = self.root / "new-launch.json"
        controlled.write(live_path, live)
        endpoint = {"container_id": "new", "image_id": "image", "started_at": "new-start"}
        args = argparse.Namespace(manifest=self.manifest_path, inputs=self.input_path,
                                  launch_receipt=live_path, phase=phase, attempt_id="new-attempt",
                                  output=self.root / "run", startup_proof=startup_proof, cell=None, timeout=12)
        calls = []
        def measured(_base, bodies, cell, directory, timeout):
            calls.append((copy.deepcopy(bodies), copy.deepcopy(cell), timeout))
            result = measurement(bodies, cell, not fail)
            self.raw(directory, bodies, result)
            return result
        inspection = [{"Id": "new", "Image": "image", "Config": {"Cmd": []},
                       "State": {"Running": not fail, "StartedAt": "new-start", "ExitCode": 1 if fail else 0,
                                 "FinishedAt": "2026-01-01T00:02:00Z"}}]
        with patch.object(controlled, "validate_profile"), patch.object(controlled.bench_common, "verify_endpoint", return_value=endpoint), \
                patch.object(controlled.bench_common, "require_model"), patch.object(controlled, "_health"), \
                patch.object(controlled.inputs, "source_identity", return_value={"source_manifest_sha256": "source"}), \
                patch.object(controlled, "ServiceObserver", observer_type), \
                patch.object(controlled, "_warm", side_effect=[{"status": "PASS", "both_dp_active": True}, {"status": "FAIL"}]
                             if fail_warmup else None, return_value={"status": "PASS", "both_dp_active": True}) as warmed, \
                patch.object(controlled, "measure", side_effect=measured), \
                patch.object(controlled.subprocess, "check_output", return_value=json.dumps(inspection)):
            code = controlled.run(args)
        return code, controlled.read(args.output / "summary.json"), calls, warmed.call_args_list

    def test_fresh_replay_preserves_original_128_output_and_is_not_formal(self):
        code, summary, calls, warmed = self.runtime("fresh-replay")
        self.assertEqual(code, 0)
        self.assertEqual(summary["scope"]["kind"], "diagnostic")
        self.assertEqual(summary["phase"], "fresh_replay")
        self.assertEqual(summary["formal_pass_count"], 0)
        self.assertEqual(len(warmed), 1)
        self.assertTrue(warmed[0].kwargs["global_start"])
        original = controlled.read(self.manifest["failed_replay"]["source_refs"]["payloads"]["path"])
        bodies, cell, _ = calls[0]
        self.assertEqual(cell["concurrency"], 2)
        self.assertEqual(cell["osl"], 128)
        self.assertEqual([controlled._without_salt(b) for b in bodies], [controlled._without_salt(b) for b in original])

    def test_terminal_failure_keeps_later_cells_not_run_and_exit_inspect(self):
        code, summary, calls, _ = self.runtime("remaining", fail=True)
        self.assertEqual(code, 1)
        self.assertEqual(summary["counts"], {"PASS": 0, "FAIL": 1, "NOT_RUN": 2})
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0][0]), 1)
        self.assertEqual(calls[0][1]["osl"], 1000)
        failed = summary["cells"][0]
        self.assertEqual(failed["stage"], "measurement")
        self.assertTrue(failed["started_at"] and failed["finished_at"])
        terminal = failed["terminal_evidence"]
        self.assertEqual(terminal["window_started_at"], failed["started_at"])
        self.assertFalse(controlled.read(terminal["container_inspect"]["path"])[0]["State"]["Running"])
        self.assertEqual(controlled.inputs.sha256(Path(terminal["service_log"]["path"])), terminal["service_log"]["sha256"])

    def test_remaining_measures_each_pending_cell_only_once(self):
        code, summary, calls, warmed = self.runtime("remaining")
        self.assertEqual(code, 0)
        self.assertEqual(summary["formal_pass_count"], 3)
        self.assertEqual([cell["cell_id"] for _, cell, _ in calls], self.manifest["pending_cell_ids"])
        self.assertEqual(len(warmed), 4)
        self.assertEqual(summary["scope"]["retained_cell_ids"], [c["cell_id"] for c in self.cells[:4]])

    def test_fresh_replay_rejects_a_process_that_already_handled_requests(self):
        class UsedObserver(FakeObserver):
            def snapshot(self):
                values = super().snapshot()
                values['vllm:prompt_tokens_total{engine="0"}'] = 8
                return values
        code, summary, calls, warmed = self.runtime("fresh-replay", observer_type=UsedObserver)
        self.assertEqual(code, 1)
        self.assertIn("already handled inference", summary["error"])
        self.assertEqual(calls, [])
        self.assertEqual(warmed, [])

    def test_failed_light_warmup_does_not_send_formal_or_later_cells(self):
        code, summary, calls, _ = self.runtime("remaining", fail_warmup=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(summary["cells"][0]["stage"], "warmup")
        self.assertEqual(summary["counts"], {"PASS": 0, "FAIL": 1, "NOT_RUN": 2})

    def test_same_process_startup_proof_does_not_repeat_global_work(self):
        proof = self.root / "startup-proof.json"
        controlled.write(proof, {"status": "PASS", "profile": self.profile,
                                 "endpoint": {"container_id": "new", "image_id": "image", "started_at": "new-start"},
                                 "source": {"source_manifest_sha256": "source"},
                                 "frozen_inputs_sha256": self.frozen["sha256"],
                                 "global_warmup": {"status": "PASS", "both_dp_active": True}})
        code, summary, calls, warmed = self.runtime("remaining", startup_proof=proof)
        self.assertEqual(code, 0)
        self.assertEqual(len(warmed), 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(summary["startup_proof_ref"], controlled.ref(proof))

    def test_light_warmup_archives_each_canonical_round(self):
        cell = self.cells[0]
        directory = self.root / "warm"
        def warmed(_base, factory, spec, _observer, out):
            self.assertEqual((spec["warmup_rounds"], spec["warmup_concurrency"], spec["warmup_osl"]), (1, 2, 2))
            label = cell["cell_id"] + "-warmup-1"
            bodies = [factory(label, i) for i in range(2)]
            self.assertTrue(all(b["max_tokens"] == 2 for b in bodies))
            controlled.write(out / "round-1/request-payloads.json.gz", bodies)
            return {"status": "PASS", "both_dp_active": True}
        with patch.object(controlled, "warmup", side_effect=warmed):
            controlled._warm(inputs.BASE_URL, self.frozen, cell, None, directory, "next", "remaining")
        proof = controlled.read(directory / "round-1/replay-provenance.json")
        self.assertEqual(len(controlled.read(proof["canonical_payloads_ref"]["path"])), 2)


if __name__ == "__main__":
    unittest.main()
