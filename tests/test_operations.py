"""Regression tests for fail-closed model launch and measurement validity."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import launch
from scripts import model_check
from scripts.benchmark import consume_sse, percentile, verify_endpoint
from scripts.check_runtime import expected_sources, fingerprint, require_matching_report


class SourcePayloadTests(unittest.TestCase):
    def test_old_image_receipt_is_rejected_even_when_it_reports_pass(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "rebuild"):
            require_matching_report(
                {"status": "PASS", "installed_source_files": 30}, root
            )
        expected = fingerprint(expected_sources(root))
        self.assertEqual(
            require_matching_report(
                {"status": "PASS", "source_manifest_sha256": expected}, root
            ),
            expected,
        )
        with self.assertRaisesRegex(ValueError, "rebuild"):
            require_matching_report(
                {"status": "PASS", "source_manifest_sha256": "0" * 64}, root
            )

    def test_all_pinned_payloads_are_present(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / "adaptation"
        for row in json.loads((folder / "manifest.json").read_text())["files"]:
            with self.subTest(path=row["path"]):
                self.assertEqual(
                    hashlib.sha256(
                        (folder / "files" / row["path"]).read_bytes()
                    ).hexdigest(),
                    row["patched_sha256"],
                )

    def test_adaptation_manifest_matches_its_single_source_tree(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "adaptation/manifest.json").read_text())
        files = manifest["files"]
        stages = {stage: sum(row["stage"] == stage for row in files)
                  for stage in {row["stage"] for row in files}}
        self.assertEqual(sum(component["files"] for component in manifest["components"]),
                         len(files))
        self.assertEqual(
            stages,
            {
                "base_integration": 26,
                "model_support": 9,
                "sm80_acceleration": 4,
            },
        )
        self.assertTrue(all((root / "adaptation/files" / row["path"]).is_file()
                            for row in files))


class ModelReceiptTests(unittest.TestCase):
    def test_partial_hash_never_authorizes_launch_and_changed_weight_invalidates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "model"
            model.mkdir()
            weight = model / "model.safetensors"
            weight.write_bytes(b"abcd")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "modelscope_revision": "1" * 40,
                        "files": [
                            {
                                "path": weight.name,
                                "size": 4,
                                "sha256": model_check.sha256(weight),
                            }
                        ],
                    }
                )
            )
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(model_check.audit(model, False, manifest)))
            with self.assertRaises(ValueError):
                model_check.require_receipt(model, receipt, manifest)
            receipt.write_text(json.dumps(model_check.audit(model, True, manifest)))
            model_check.require_receipt(model, receipt, manifest)
            weight.write_bytes(b"efgh")
            with self.assertRaisesRegex(ValueError, "changed"):
                model_check.require_receipt(model, receipt, manifest)

    def test_incomplete_file_reports_not_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "modelscope_revision": "1" * 40,
                        "files": [
                            {
                                "path": "missing.safetensors",
                                "size": 4,
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                )
            )
            result = model_check.audit(root, True, manifest)
            self.assertEqual(result["status"], "NOT_VERIFIED")
            self.assertEqual(result["files"][0]["status"], "MISSING")


class LaunchTests(unittest.TestCase):
    def plan(self, *options):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text(
                json.dumps({"text_config": {"dspark_block_size": 5}})
            )
            expected = next(
                r["sha256"]
                for r in json.loads(launch.MANIFEST.read_text())["files"]
                if r["path"] == "config.json"
            )
            with patch.object(launch, "sha256", return_value=expected):
                return launch.make_plan(
                    launch.parser().parse_args([str(path), *options])
                )

    def test_tp_changes_preserve_global_budget_and_model_identity(self):
        a, b = self.plan("--parallel", "tp8"), self.plan("--parallel", "tp4dp2")
        self.assertEqual(
            a["parallel"]["max_global_seqs"], b["parallel"]["max_global_seqs"]
        )
        for plan, per_rank in ((a, "64"), (b, "32")):
            command = plan["command"]
            self.assertEqual(command[command.index("--max-num-seqs") + 1], per_rank)
            self.assertIn(launch.MODEL_NAME, command)
            self.assertNotIn("--language-model-only", command)
            self.assertNotIn("--privileged", command)
            self.assertNotIn("stop", command)
            self.assertIn("127.0.0.1:8005:8000", command)

    def test_k7_rejected_and_no_speculation_means_no_spec_config(self):
        with self.assertRaisesRegex(ValueError, "trained block"):
            self.plan("--dspark", "7")
        self.assertNotIn("--speculative-config", self.plan("--dspark", "0")["command"])

    def test_new_hotpaths_are_opt_in_and_independently_recorded(self):
        baseline = self.plan()
        self.assertEqual(baseline["experimental"], [])
        for variable in launch.EXPERIMENTS.values():
            self.assertIn(variable + "=0", baseline["command"])
        changed = self.plan("--experimental", "ragged-reuse")
        self.assertEqual(changed["experimental"], ["ragged-reuse"])
        self.assertIn("VLLM_DSV41_REUSE_TOPK_RAGGED=1", changed["command"])
        self.assertIn("VLLM_DSV41_STACKED_DRAFT_KV=0", changed["command"])
        bounded = self.plan(
            "--context", "1048576", "--experimental", "bounded-prefill"
        )
        self.assertIn("VLLM_DSV41_BOUNDED_PREFILL_WORKSPACE=1", bounded["command"])
        with self.assertRaisesRegex(ValueError, "requires DSpark"):
            self.plan("--dspark", "0", "--experimental", "stacked-kv")

    def test_graph_has_verification_and_draft_shapes(self):
        command = self.plan()["command"]
        sizes = json.loads(command[command.index("--compilation-config") + 1])[
            "cudagraph_capture_sizes"
        ]
        self.assertIn(32 * 5, sizes)
        self.assertIn(32 * 6, sizes)

    def test_active_gpu_prevents_start(self):
        output = "\n".join(["NVIDIA A100-SXM4-80GB, 80000"] * 8)
        with (
            patch("subprocess.check_output", return_value=output),
            self.assertRaisesRegex(ValueError, "idle"),
        ):
            launch.require_idle_gpus()

    def test_stale_container_receipt_cannot_authorize_measurement(self):
        receipt = {
            "command": [
                "docker",
                "run",
                "-p",
                "127.0.0.1:8005:8000",
                "sha256:expected",
                "serve",
                "model",
            ],
            "image_id": "sha256:expected",
            "container_id": "container",
        }
        container = {
            "State": {"Running": True},
            "Image": "sha256:wrong",
            "Config": {"Cmd": ["serve", "model"]},
        }
        with (
            patch("subprocess.check_output", return_value=json.dumps([container])),
            self.assertRaisesRegex(ValueError, "differs"),
        ):
            verify_endpoint("http://127.0.0.1:8005", receipt)
        with self.assertRaisesRegex(ValueError, "local port"):
            verify_endpoint("http://127.0.0.1:6789", receipt)


class StreamingMetricTests(unittest.TestCase):
    def test_usage_tokens_not_chunk_count_drive_metrics(self):
        stream = io.BytesIO(
            b'data: {"choices":[{"text":"abc","finish_reason":null}]}\n\n'
            b'data: {"choices":[{"text":"def","finish_reason":"length"}]}\n\n'
            b'data: {"choices":[],"usage":{"completion_tokens":10,"prompt_tokens":32,'
            b'"prompt_tokens_details":{"cached_tokens":0}}}\n\n'
            b"data: [DONE]\n\n"
        )
        result = consume_sse(stream, 0, 1000, clock=iter(range(1, 30)).__next__)
        self.assertEqual(result["output_tokens"], 10)
        self.assertEqual(result["content_chunks"], 2)
        self.assertEqual(result["cached_tokens"], 0)
        self.assertAlmostEqual(
            result["stream_tpot_s"], result["stream_decode_span_s"] / 9
        )

    def test_missing_done_and_usage_cannot_pass(self):
        stream = io.BytesIO(
            b'data: {"choices":[{"text":"abc","finish_reason":"length"}]}\n'
        )
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            consume_sse(stream, 0, 1000, clock=lambda: 1)

    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile(list(range(1, 101)), 0.95), 95)
        self.assertIsNone(percentile([], 0.95))
