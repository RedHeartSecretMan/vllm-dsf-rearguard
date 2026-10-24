"""Acceptance must assert media semantics and retain failed endpoint evidence."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.acceptance import media, visual
from scripts.acceptance.catalog import load


def receipt(flag="--no-disable-chunked-mm-input"):
    return {"command": [
        "vllm", "serve", "model", flag,
        "--limit-mm-per-prompt", '{"image":64,"video":{"count":2,"num_frames":32}}',
        "--media-io-kwargs", '{"video":{"num_frames":32,"fps":1}}',
        "--max-num-batched-tokens", "64",
    ]}


def response(value):
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(value),
    }}], "usage": {"prompt_tokens": 100, "completion_tokens": 12}}


class VisualAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.cases = {case["id"]: case for case in load("visual")["cases"]}
        self.fixtures = {"images": {}, "videos": {}}
        for number, (color, _) in media.COLORS.items():
            path = self.directory / f"{number}.png"
            media._number_png(number, path)
            self.fixtures["images"][str(number)] = {
                "path": str(path), "sha256": media.sha256(path), "number": number, "color": color,
            }
        for name, numbers in (("short", [1, 2, 3]), ("long", [1, 2, 3]), ("second", [3, 1, 2])):
            path = self.directory / f"{name}.mp4"
            path.write_bytes(name.encode())
            self.fixtures["videos"][name] = {
                "path": str(path), "sha256": media.sha256(path), "numbers": numbers,
                "colors": [media.COLORS[number][0] for number in numbers],
            }

    def execute(self, case_id, result, **kwargs):
        case = self.cases[case_id]
        with patch.object(visual, "request", return_value=response(result)):
            return visual.run_case(
                "http://localhost", visual.MODEL, case, self.fixtures,
                kwargs.pop("launch", receipt()), 5, self.directory / "evidence", **kwargs,
            )

    def test_video_wrong_order_fails_and_keeps_raw_output(self):
        report = self.execute("visual-video-short", {"numbers": [3, 2, 1], "colors": ["red", "green", "blue"]})
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("semantic mismatch", report["error"])
        raw = self.directory / "evidence/visual-video-short/turn-1/response.json"
        self.assertEqual(json.loads(raw.read_text())["choices"][0]["message"]["content"], json.dumps({"numbers": [3, 2, 1], "colors": ["red", "green", "blue"]}))

    def test_nonempty_unstructured_video_answer_does_not_pass(self):
        with patch.object(visual, "request", return_value=response("a video")):
            report = visual.run_case("http://localhost", visual.MODEL, self.cases["visual-video-short"], self.fixtures, receipt(), 5, self.directory)
        self.assertEqual(report["status"], "FAIL")

    def test_all_requests_disable_thinking_and_use_nontrivial_order(self):
        for case in self.cases.values():
            with self.subTest(case=case["id"]):
                body, _, _ = visual.build_case(case, visual.MODEL, self.fixtures)
                self.assertEqual(body["chat_template_kwargs"], {"thinking": False})
                self.assertGreaterEqual(body["max_tokens"], 1024)
        _, expected, _ = visual.build_case(self.cases["visual-image-limit-64"], visual.MODEL, self.fixtures)
        self.assertEqual(len(expected["numbers"]), 64)
        self.assertNotEqual(expected["numbers"][:6], [1, 2, 3, 1, 2, 3])

    def test_two_videos_and_mixed_order_have_separate_expectations(self):
        body, expected, _ = visual.build_case(self.cases["visual-video-two-items"], visual.MODEL, self.fixtures)
        self.assertEqual(expected["videos"][0]["numbers"], [1, 2, 3])
        self.assertEqual(expected["videos"][1]["numbers"], [3, 1, 2])
        videos = body["messages"][0]["content"][1:]
        self.assertNotEqual(videos[0], videos[1])
        body, expected, _ = visual.build_case(self.cases["visual-image-video-image-order"], visual.MODEL, self.fixtures)
        self.assertEqual([part["type"] for part in body["messages"][0]["content"]], ["text", "image_url", "video_url", "image_url"])
        self.assertEqual((expected["before_image"], expected["after_image"]), (3, 1))

    def test_history_uses_two_real_calls_and_first_assistant_output(self):
        sent = []

        def fake_request(base_url, body, timeout):
            sent.append(copy.deepcopy(body))
            if len(sent) == 1:
                identity = visual.expected_video_id(body["messages"][0]["content"][1])
                return response({"video_id": identity})
            return response({"numbers": [1, 2, 3], "colors": ["red", "green", "blue"]})

        with patch.object(visual, "request", side_effect=fake_request):
            report = visual.run_case("http://localhost", visual.MODEL, self.cases["visual-video-history-uuid"], self.fixtures, receipt(), 5, self.directory)
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0]["messages"][0], sent[1]["messages"][0])
        self.assertEqual(json.loads(sent[1]["messages"][1]["content"]), {"video_id": report["history_video_id"]})
        self.assertIsInstance(sent[1]["messages"][2]["content"], str)
        self.assertIn(report["history_video_id"], sent[1]["messages"][2]["content"])

    def test_chunk_comparison_requires_passing_identical_request(self):
        expected = {"numbers": [1, 2, 3], "colors": ["red", "green", "blue"]}
        baseline = self.execute("visual-video-chunked-disabled", expected, launch=receipt("--disable-chunked-mm-input"))
        self.assertEqual(baseline["status"], "PASS")
        path = self.directory / "evidence/visual-video-chunked-disabled/case.json"
        matched = self.execute("visual-video-chunked-frame-boundary", expected, chunk_baseline=path)
        self.assertEqual(matched["status"], "PASS", matched)
        self.assertEqual(matched["chunk_comparison"]["status"], "PASS")
        baseline["steps"][0]["request_sha256"] = "wrong"
        visual.write_json(path, baseline)
        rejected = self.execute("visual-video-chunked-frame-boundary", expected, chunk_baseline=path)
        self.assertEqual(rejected["status"], "FAIL")

    def test_truncated_output_never_passes(self):
        partial = response({"numbers": [1]})
        partial["choices"][0]["finish_reason"] = "length"
        with patch.object(visual, "request", return_value=partial):
            report = visual.run_case("http://localhost", visual.MODEL, self.cases["visual-image-single"], self.fixtures, receipt(), 5, self.directory)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("finish normally", report["error"])

    def test_launch_receipt_rejects_wrong_media_policy_and_mode(self):
        for changed in (
            {"command": []}, receipt("--disable-chunked-mm-input"),
            {"command": [part.replace('"fps":1', '"fps":2') for part in receipt()["command"]]},
        ):
            with self.subTest(receipt=changed), self.assertRaises(ValueError):
                visual.verify_receipt(self.cases["visual-video-chunked-frame-boundary"], changed)

    def test_manifest_rejects_changed_fixture_bytes(self):
        visual.write_json(self.directory / "manifest.json", self.fixtures)
        self.assertEqual(media.load_fixtures(self.directory), self.fixtures)
        Path(self.fixtures["videos"]["long"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "videos.long"):
            media.load_fixtures(self.directory)

    def test_relative_manifest_resolves_from_moved_mount(self):
        relative = copy.deepcopy(self.fixtures)
        for group in relative.values():
            for item in group.values():
                item["path"] = Path(item["path"]).name
        visual.write_json(self.directory / "manifest.json", relative)
        self.assertEqual(media.load_fixtures(self.directory), self.fixtures)

    def test_chunk_case_cannot_pass_without_crossing_budget(self):
        launch = receipt()
        launch["command"][-1] = "1024"
        report = self.execute("visual-video-chunked-frame-boundary", {"numbers": [1, 2, 3], "colors": ["red", "green", "blue"]}, launch=launch)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("does not cross chunk size", report["error"])


if __name__ == "__main__":
    unittest.main()
