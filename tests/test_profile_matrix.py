"""Fixed service invariants and catalog routing for campaign integration."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import launch
from scripts.benchmarks.profile_matrix import (
    execution_kind,
    normalized_command,
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


if __name__ == "__main__":
    unittest.main()
