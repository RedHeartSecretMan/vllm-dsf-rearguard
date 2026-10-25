"""Fixed service invariants and catalog routing for campaign integration."""

import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import launch
from scripts.benchmarks import profile_campaign, profile_workload
from scripts.benchmarks.profile_matrix import (
    execution_kind,
    normalized_command,
    resume_evidence,
    validate_profile,
)
from scripts.benchmarks.profile_plan import group_plan


class ProfileMatrixTests(unittest.TestCase):
    def receipt(self, model, context=1048576, memory=.9, chunk=16384, name="profile-a"):
        expected = next(r["sha256"] for r in json.loads(launch.MANIFEST.read_text())["files"] if r["path"] == "config.json")
        with patch.object(launch, "sha256", return_value=expected):
            plan = launch.make_plan(launch.parser().parse_args([
                str(model), "--name", name, "--context", str(context), "--memory", str(memory),
                "--chunk", str(chunk), "--port", "6688", "--responses-store",
                "--experimental", "stacked-kv", "--experimental", "bounded-prefill", "--no-disable-chunked-mm-input",
            ]))
        plan["source_manifest_sha256"] = "same-source"
        return plan

    def test_actual_launcher_plan_validates_and_only_permitted_axes_normalize(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "config.json").write_text(json.dumps({"text_config": {"dspark_block_size": 5}}))
            a = self.receipt(model)
            b = self.receipt(model, 262144, .96, 32768, "profile-b")
            with patch("scripts.benchmarks.profile_workload.source_identity", return_value={"source_manifest_sha256": "same-source"}):
                validate_profile(a, 1048576, .9, 16384)
                validate_profile(b, 262144, .96, 32768)
                self.assertEqual(normalized_command(a), normalized_command(b))
                for flag, wrong in (("--kv-cache-dtype", "bfloat16"),
                                    ("--limit-mm-per-prompt", '{"image":4,"video":1}'),
                                    ("--media-io-kwargs", '{"video":{"num_frames":8,"fps":2}}')):
                    changed = copy.deepcopy(a)
                    changed["command"][changed["command"].index(flag) + 1] = wrong
                    with self.subTest(flag=flag), self.assertRaises(ValueError):
                        validate_profile(changed, 1048576, .9, 16384)

    def test_visual_mixed_is_routed_to_visual_acceptance(self):
        cells = group_plan(1048576)["cells"]
        visual = next(c for c in cells if c["case_id"] == "visual-image-video-image-order")
        mixed = next(c for c in cells if c["case_id"] == "mixed_prefill_decode_same_service")
        self.assertEqual(visual["kind"], "mixed")
        self.assertEqual(execution_kind(visual), "visual")
        self.assertEqual(execution_kind(mixed), "mixed")

    def test_resume_schedule_cannot_restart_service_before_identity_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schedule = root / "schedule.json"
            schedule.write_text(json.dumps([{"id": "same", "resume_from": "old-results"}]))
            args = SimpleNamespace(directory=root / "campaign", schedule=schedule)
            with patch.object(profile_campaign, "launch") as launch_service, self.assertRaisesRegex(ValueError, "adopting"):
                profile_campaign.run(args)
            launch_service.assert_not_called()


class PolicyResumeTests(unittest.TestCase):
    endpoint = {"container_id": "running", "image_id": "image", "started_at": "original-start"}

    def fixture(self, root):
        directory = root / "legacy-attempt/results"
        directory.mkdir(parents=True)
        plan = group_plan(1048576)
        cell = next(c for c in plan["cells"] if c["case_id"] == "matrix_single" and c["isl"] == 4096 and c["dataset"] == "random")
        plan["cells"] = [cell]
        frozen = {"sha256": "frozen", "run_id": "fixed", "token_pool": [1, 2, 3]}
        receipt = {"image_id": "image", "source_manifest_sha256": "source", "container_id": "running", "command": ["unchanged"]}
        bodies = [profile_workload.request_body(frozen, cell, "repeat-1", 0)]
        folder = directory / "cells" / cell["cell_id"].replace("/", "__") / "repeat-1"
        folder.mkdir(parents=True)
        with gzip.open(folder / "request-payloads.json.gz", "wt") as stream:
            json.dump(bodies, stream)
        measurement = {"status": "PASS", "cache_verified": True, "payload_sha256": "primary"}
        (folder / "report.json").write_text(json.dumps(measurement))
        row = {"cell_id": cell["cell_id"], "status": "IN_PROGRESS", "warmup": {"status": "PASS"}, "measurements": [measurement]}
        source = {"status": "IN_PROGRESS", "profile": plan["profile"], "frozen_inputs_sha256": "frozen", "endpoint": self.endpoint,
                  "global_warmup": {"status": "PASS", "both_dp_active": True}, "observation": {"status": "PASS"}, "cells": [row]}
        old_plan = copy.deepcopy(plan)
        old_plan["cells"][0]["repeats"] = 3
        for name, value in (("plan.json", old_plan), ("summary.json", source), ("launch.json", receipt)):
            (directory / name).write_text(json.dumps(value))
        boundary = root / "boundary.json"
        boundary.write_text('{"drained":true}')
        handoff = {"status": "POLICY_HANDOFF", "no_inflight": True, "source_attempt_id": "legacy-attempt",
                   "current_cell": cell["cell_id"], "pending_unsaved_payloads": [],
                   "source_snapshot_sha256": profile_workload.sha256(directory / "summary.json"),
                   "rawrefs": [{"path": str(boundary), "sha256": profile_workload.sha256(boundary)}],
                   "metrics": [f'vllm:num_requests_{kind}{{engine="{i}"}} 0.0' for kind in ("running", "waiting") for i in range(2)]}
        path = root / "handoff.json"
        path.write_text(json.dumps(handoff))
        return directory, plan, frozen, receipt, path, cell

    def test_partial_primary_is_reused_only_at_bound_drained_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, plan, frozen, receipt, handoff, cell = self.fixture(Path(tmp))
            rows, provenance, global_warmup = resume_evidence(directory, plan, frozen, receipt, handoff, endpoint=self.endpoint)
            self.assertEqual(rows[cell["cell_id"]]["status"], "PASS")
            self.assertEqual(len(rows[cell["cell_id"]]["measurements"]), 1)
            self.assertEqual(provenance["cells"][cell["cell_id"]]["primary_repeat"], 1)
            self.assertTrue(global_warmup["both_dp_active"])
            with self.assertRaisesRegex(ValueError, "explicit policy handoff"):
                resume_evidence(directory, plan, frozen, receipt, None, endpoint=self.endpoint)
            with self.assertRaisesRegex(ValueError, "start time differs"):
                resume_evidence(directory, plan, frozen, receipt, handoff, endpoint={**self.endpoint, "started_at": "restarted"})

    def test_failure_or_changed_primary_is_not_hidden_by_single_repeat(self):
        for corruption in ("failure", "payload", "cache", "identity", "inflight"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as tmp:
                directory, plan, frozen, receipt, handoff, cell = self.fixture(Path(tmp))
                if corruption == "failure":
                    source = json.loads((directory / "summary.json").read_text())
                    source["cells"][0]["status"] = "FAIL"
                    (directory / "summary.json").write_text(json.dumps(source))
                elif corruption == "payload":
                    p = directory / "cells" / cell["cell_id"].replace("/", "__") / "repeat-1/request-payloads.json.gz"
                    with gzip.open(p, "wt") as stream:
                        json.dump([{"prompt": [999]}], stream)
                elif corruption == "cache":
                    p = directory / "cells" / cell["cell_id"].replace("/", "__") / "repeat-1/report.json"
                    value = json.loads(p.read_text())
                    value["cache_verified"] = False
                    p.write_text(json.dumps(value))
                elif corruption == "identity":
                    receipt["image_id"] = "other"
                else:
                    value = json.loads(handoff.read_text())
                    value["metrics"][0] = value["metrics"][0].replace("0.0", "1.0")
                    handoff.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    resume_evidence(directory, plan, frozen, receipt, handoff, endpoint=self.endpoint)


if __name__ == "__main__":
    unittest.main()
