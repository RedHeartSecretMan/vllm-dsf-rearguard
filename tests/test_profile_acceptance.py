"""Fixed-profile acceptance retains semantics, capacity and failure evidence."""

import copy
import gzip
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.acceptance import responses, visual
from scripts.acceptance.catalog import load
from scripts.benchmarks import profile_acceptance as runner

BASE = "http://127.0.0.1:6688"


def receipt(context=32768, chunk=16384):
    return {"context": context, "text_only": False, "command": [
        "--no-disable-chunked-mm-input", "--max-num-batched-tokens", str(chunk),
        "--limit-mm-per-prompt", '{"image":64,"video":{"count":2,"num_frames":32}}',
        "--media-io-kwargs", '{"video":{"num_frames":32,"fps":1}}',
        "--env", "VLLM_ENABLE_RESPONSES_API_STORE=1", "--api-server-count", "1",
    ]}


def chat(value, tokens=100, cached=0, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {
        "role": "assistant", "content": json.dumps(value),
    }}], "usage": {"prompt_tokens": tokens, "completion_tokens": 10,
                   "prompt_tokens_details": {"cached_tokens": cached}}}


def response(text, cached=5):
    return {"object": "response", "id": "resp_test", "status": "completed", "output": [{
        "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}],
    }], "usage": {"input_tokens": 20, "output_tokens": 5, "input_tokens_details": {"cached_tokens": cached}}}


class HttpResponse(io.BytesIO):
    status = 200


class ProfileAcceptanceTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        self.frozen = {"run_id": "fixed-campaign", "bodies": {}, "fixtures": {"images": {}, "videos": {}},
                       "fixture_directory": str(self.directory)}
        for number in (1, 2, 3):
            path = self.directory / f"{number}.png"
            path.write_bytes(f"image-{number}".encode())
            self.frozen["fixtures"]["images"][str(number)] = {"path": str(path), "number": number}
        for name in ("short", "long", "second"):
            path = self.directory / f"{name}.mp4"
            path.write_bytes(name.encode())
            self.frozen["fixtures"]["videos"][name] = {
                "path": str(path), "numbers": [1, 2, 3], "colors": ["red", "green", "blue"],
            }
        endpoint = patch.object(runner.bench_common, "verify_endpoint", return_value={"verified": True})
        endpoint.start()
        self.addCleanup(endpoint.stop)

    def cell(self, name, family="core"):
        source = next(item for item in load(family)["cases"] if item["id"] == name)
        return {"cell_id": name, "case_id": name, "kind": source["kind"] if "kind" in source else "responses",
                "source_case": source}

    def read_raw(self, report, index=0):
        path = Path(report["rawpaths"][index])
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), report["raw_files"][index]["sha256"])
        return json.loads(gzip.decompress(raw))

    def test_non_test_ports_and_existing_directories_never_send_requests(self):
        for entry in (runner.run_core, runner.run_visual, runner.run_responses):
            with self.subTest(entry=entry.__name__), patch.object(runner.bench_common, "json_request") as request:
                for base in ("http://127.0.0.1:6789", "http://127.0.0.1:8005"):
                    with self.assertRaisesRegex(ValueError, "6688"):
                        entry(base, self.cell("tool-call"), self.frozen, receipt(), self.directory / "out")
                with self.assertRaises(FileExistsError):
                    entry(BASE, self.cell("tool-call"), self.frozen, receipt(), self.directory)
                request.assert_not_called()
        self.assertEqual((self.directory / "1.png").read_bytes(), b"image-1")

    def test_core_runs_only_the_selected_case_with_frozen_request_controls(self):
        cell = self.cell("arithmetic-thinking-False")
        with patch.object(runner.bench_common, "json_request", return_value=chat({"answer": 410})) as request:
            report = runner.run_core(BASE, cell, self.frozen, receipt(), self.directory / "core")
        self.assertEqual(report["status"], "PASS", report)
        request.assert_called_once()
        body = self.read_raw(report)["request"]
        self.assertEqual((body["seed"], body["temperature"], body["max_tokens"]), (1017, 0, 2048))
        self.assertEqual(body["chat_template_kwargs"], {"thinking": False, "reasoning_effort": "high"})
        self.assertTrue(body["cache_salt"])
        self.assertEqual(report["actual_input_tokens"], 100)

    def test_core_wrong_answer_truncation_or_warm_cache_keeps_raw_failure(self):
        candidates = [chat({"answer": 411}), chat({"answer": 410}, finish="length"), chat({"answer": 410}, cached=1)]
        for index, value in enumerate(candidates):
            with self.subTest(index=index), patch.object(runner.bench_common, "json_request", return_value=value):
                report = runner.run_core(BASE, self.cell("arithmetic-thinking-False"), self.frozen, receipt(), self.directory / f"fail-{index}")
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(self.read_raw(report)["response"], value)
            self.assertTrue(report["error"])

    def test_core_rejects_wrong_tool_arguments(self):
        value = chat(None, finish="tool_calls")
        value["choices"][0]["message"]["tool_calls"] = [{"function": {"name": "lookup", "arguments": '{"key":"beta"}'}}]
        with patch.object(runner.bench_common, "json_request", return_value=value):
            report = runner.run_core(BASE, self.cell("tool-call"), self.frozen, receipt(), self.directory / "tool")
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("Wrong tool arguments", report["error"])
        self.assertEqual(self.read_raw(report)["request"]["tool_choice"]["function"]["name"], "lookup")

    def test_needle_uses_frozen_content_and_checks_exact_context_boundary(self):
        name = "needle-target-1024"
        case = {"name": name, "target_tokens": 1024, "thinking": False,
                "messages": [{"role": "user", "content": "frozen complete archive"}], "expected": {"secret": "A"}}
        self.frozen["bodies"][name] = copy.deepcopy(case)
        cell = {"cell_id": name, "case_id": name, "kind": "needle", "isl": 1024, "source_case": {}}
        with patch.object(runner.bench_common, "json_request") as request:
            report = runner.run_core(BASE, cell, self.frozen, receipt(context=1535), self.directory / "oversized")
            request.assert_not_called()
        self.assertEqual(report["status"], "FAIL")
        for actual, expected in ((1024, "PASS"), (1025, "FAIL")):
            with patch.object(runner.bench_common, "json_request", return_value=chat({"secret": "A"}, tokens=actual)):
                report = runner.run_core(BASE, cell, self.frozen, receipt(context=1536), self.directory / f"needle-{actual}")
            self.assertEqual(report["status"], expected, report)
            body = self.read_raw(report)["request"]
            self.assertEqual(body["messages"], case["messages"])
            self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(self.frozen["bodies"][name], case)

    def test_visual_preserves_case_semantics_and_rejects_dedicated_chunk_cases(self):
        with patch.object(runner.bench_common, "json_request", return_value=chat({"numbers": [1]})) as request:
            report = runner.run_visual(BASE, self.cell("visual-image-single", "visual"), self.frozen, receipt(), self.directory / "image")
            self.assertEqual(report["status"], "PASS", report)
            self.assertEqual(request.call_count, 1)
            rejected = runner.run_visual(BASE, self.cell("visual-video-chunked-frame-boundary", "visual"), self.frozen, receipt(), self.directory / "chunk")
            self.assertEqual(rejected["status"], "FAIL")
            self.assertEqual(request.call_count, 1)

    def test_video_history_replays_the_same_media_and_allows_second_turn_kv_reuse(self):
        cell = self.cell("visual-video-history-uuid", "visual")
        body, expected, _ = visual.build_case(cell["source_case"], runner.bench_common.MODEL, self.frozen["fixtures"])
        identity = visual.expected_video_id(body["messages"][0]["content"][1])
        replies = [chat({"video_id": identity}), chat(expected, tokens=150, cached=50)]
        with patch.object(runner.bench_common, "json_request", side_effect=replies):
            report = runner.run_visual(BASE, cell, self.frozen, receipt(), self.directory / "history")
        self.assertEqual(report["status"], "PASS", report)
        first, second = [self.read_raw(report, index)["request"] for index in (0, 1)]
        self.assertEqual(first["messages"][0], second["messages"][0])
        self.assertEqual(first["cache_salt"], second["cache_salt"])
        self.assertIn(identity, second["messages"][-1]["content"])
        self.assertEqual(report["steps"][1]["cached_tokens"], 50)
        self.assertEqual(report["steps"][1]["kv_cache_policy"], "history_reuse")

    def test_highres_must_cross_16k_and_records_actual_32k_behavior(self):
        cell = self.cell("visual-video-long", "visual")
        cell.update(cell_id="visual-video-highres-semantic", case_id="visual-video-highres-semantic")
        body, expected, _ = visual.build_case(cell["source_case"], runner.bench_common.MODEL, self.frozen["fixtures"])
        for chunk, tokens, status in ((16384, 16384, "FAIL"), (16384, 28000, "PASS"), (32768, 28000, "PASS")):
            with patch.object(runner.profile_workload, "highres_video_case", return_value=(body, expected)), patch.object(
                runner.bench_common, "json_request", return_value=chat(expected, tokens=tokens),
            ):
                report = runner.run_visual(BASE, cell, self.frozen, receipt(chunk=chunk), self.directory / f"highres-{chunk}-{tokens}")
            self.assertEqual(report["status"], status, report)
            if status == "PASS":
                self.assertEqual(report["chunk_prefill"]["actual_input_tokens"], 28000)
                self.assertEqual(report["chunk_prefill"]["crosses_chunk_budget"], chunk == 16384)

    def test_responses_retains_protocol_payload_and_observed_cache_information(self):
        raw = json.dumps(response("ACCEPTANCE_OK"))
        with patch.object(runner.bench_common, "request", return_value=HttpResponse(raw.encode())):
            report = runner.run_responses(BASE, self.cell("responses-basic", "responses"), self.frozen, receipt(), self.directory / "responses")
        self.assertEqual(report["status"], "PASS", report)
        exchange = self.read_raw(report)
        self.assertEqual(exchange["raw_response"], raw)
        self.assertNotIn("cache_salt", exchange["request"])
        self.assertEqual(report["steps"][0]["cached_tokens"], 5)
        self.assertEqual(report["steps"][0]["kv_cache_policy"], "protocol")

    def test_responses_store_and_error_validation_preserve_failures(self):
        cell = self.cell("responses-previous-response-id-tools", "responses")
        with patch.object(runner.bench_common, "request") as request:
            report = runner.run_responses(BASE, cell, self.frozen, {"command": []}, self.directory / "store")
            request.assert_not_called()
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("responses-store", report["error"])
        for index, value in enumerate((response("WRONG"), ["malformed response"])):
            raw = json.dumps(value)
            with patch.object(runner.bench_common, "request", return_value=HttpResponse(raw.encode())):
                report = runner.run_responses(BASE, self.cell("responses-basic", "responses"), self.frozen, receipt(), self.directory / f"responses-bad-{index}")
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(self.read_raw(report)["raw_response"], raw)

    def test_responses_stream_preserves_original_sse_without_rewriting_it(self):
        call = {"type": "custom_tool_call", "id": "tool1", "call_id": "call1", "name": "record_text", "input": responses.TOOL_INPUT}
        final = response("")
        final["output"] = [call]
        events = [
            {"type": "response.output_item.added", "output_index": 0, "item": {**call, "input": ""}},
            {"type": "response.custom_tool_call_input.delta", "item_id": "tool1", "output_index": 0, "delta": responses.TOOL_INPUT},
            {"type": "response.custom_tool_call_input.done", "item_id": "tool1", "output_index": 0, "input": responses.TOOL_INPUT},
            {"type": "response.output_item.done", "output_index": 0, "item": call},
            {"type": "response.completed", "response": final},
        ]
        raw = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        with patch.object(runner.bench_common, "request", return_value=HttpResponse(raw.encode())):
            report = runner.run_responses(BASE, self.cell("responses-custom-tool-stream", "responses"), self.frozen, receipt(), self.directory / "stream")
        self.assertEqual(report["status"], "PASS", report)
        saved = self.read_raw(report)
        self.assertEqual(saved["raw_response"], raw)
        self.assertEqual(saved["events"], events)


if __name__ == "__main__":
    unittest.main()
