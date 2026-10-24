"""Exercise the acceptance runner's HTTP coverage and rejection of false passes."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.acceptance import responses as runner
from scripts.acceptance.catalog import load


def response(text="", output=None, previous=None):
    value = {
        "object": "response", "id": "resp_123", "status": "completed",
        "output": output if output is not None else [{
            "type": "message", "id": "msg_123", "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }],
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }
    if previous:
        value["previous_response_id"] = previous
    return value


def call_item(kind="custom"):
    value = {"type": "custom_tool_call" if kind == "custom" else "function_call",
             "id": "ctc_123", "call_id": "call_123", "name": "record_text"}
    if kind == "custom":
        value["input"] = runner.TOOL_INPUT
    else:
        value["arguments"] = json.dumps({"input": runner.TOOL_INPUT})
    return value


def stream_events():
    call = call_item()
    result = response(output=[call])
    return [
        {"type": "response.created", "response": {"id": result["id"]}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {**call, "input": ""}},
        {"type": "response.custom_tool_call_input.delta", "item_id": call["id"],
         "output_index": 0, "delta": runner.TOOL_INPUT[:6]},
        {"type": "response.custom_tool_call_input.delta", "item_id": call["id"],
         "output_index": 0, "delta": runner.TOOL_INPUT[6:]},
        {"type": "response.custom_tool_call_input.done", "item_id": call["id"],
         "output_index": 0, "input": runner.TOOL_INPUT},
        {"type": "response.output_item.done", "output_index": 0, "item": call},
        {"type": "response.completed", "response": result},
    ]


class HttpResponse(io.BytesIO):
    status = 200


class ResponsesAcceptanceTest(unittest.TestCase):
    def test_all_catalog_entries_execute_http_with_assertions(self):
        """No UNIT/placeholder case can silently count as acceptance."""
        config = load("responses")
        with tempfile.TemporaryDirectory() as tmp:
            image, video = Path(tmp) / "number.png", Path(tmp) / "sequence.mp4"
            image.write_bytes(b"image-fixture")
            video.write_bytes(b"video-fixture")
            for case in config["cases"]:
                with self.subTest(case=case["id"]):
                    self.assertEqual(case["mode"], "live")
                    case_id = case["id"]
                    if case_id == "responses-custom-tool-stream":
                        raw = "".join("data: " + json.dumps(event) + "\n\n" for event in stream_events())
                        replies = [HttpResponse(raw.encode())]
                    elif case_id == "responses-previous-response-id-tools":
                        values = []
                        for kind in ("custom", "function"):
                            marker = f"RESULT_{kind.upper()}_71953"
                            values.extend([response(output=[call_item(kind)]),
                                           response(marker, previous="resp_123"),
                                           response(marker, previous="resp_123")])
                        replies = [HttpResponse(json.dumps(value).encode()) for value in values]
                    else:
                        text = {
                            "responses-basic": "ACCEPTANCE_OK",
                            "responses-custom-tool-history": runner.HISTORY_MARKER,
                            "responses-multipart-output": "PART_A_5371 PART_B_6842 PART_C_9205",
                            "responses-refusal-output": "REFUSAL_84563 TAIL_92017",
                            "responses-image-video-input": '{"image_number":1,"video_numbers":[1,2,3]}',
                        }[case_id]
                        replies = [HttpResponse(json.dumps(response(text)).encode())]
                    with patch.object(runner.bench_common, "request", side_effect=replies) as http:
                        client = runner.Client("http://localhost:8005", 10)
                        runner.run_case(client, runner.MODEL, case, image=image, video=video)
                        self.assertGreater(http.call_count, 0)
                        self.assertEqual(len(client.exchanges), http.call_count)
                        self.assertTrue(all("raw_response" in exchange for exchange in client.exchanges))
                        self.assertTrue(all(call.args[1] == "/v1/responses" for call in http.call_args_list))

    def test_basic_rejects_empty_or_wrong_text(self):
        for text in ("", "not the expected content"):
            with patch.object(runner.bench_common, "request", return_value=HttpResponse(json.dumps(response(text)).encode())):
                with self.assertRaises(AssertionError):
                    runner.run_case(runner.Client("http://localhost", 1), runner.MODEL, {"id": "responses-basic"})

    def test_rejects_incomplete_or_error_response(self):
        for mutation in ({"status": "incomplete"}, {"error": {"message": "failed"}}, {"output": None}, {"usage": {}}):
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                runner.validate_response({**response("OK"), **mutation})

    def test_stream_requires_actual_custom_tool_events(self):
        with self.assertRaisesRegex(AssertionError, "exactly one custom_tool_call"):
            runner.assert_custom_stream([{"type": "response.completed", "response": response("OK")}])
        for index in (1, 4, 5):
            events = stream_events()
            del events[index]
            with self.subTest(index=index), self.assertRaises(AssertionError):
                runner.assert_custom_stream(events)

    def test_stream_detects_corrupted_input_and_identity(self):
        for field, value in (("delta", "wrong"), ("item_id", "wrong"), ("output_index", 7)):
            events = stream_events()
            events[2][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                runner.assert_custom_stream(events)
        events = stream_events()
        events[4], events[5] = events[5], events[4]
        with self.assertRaisesRegex(AssertionError, "lifecycle order"):
            runner.assert_custom_stream(events)

    def test_sse_parser_handles_crlf_and_reports_server_error(self):
        events = stream_events()
        raw = "".join("event: " + event["type"] + "\r\ndata: " + json.dumps(event) + "\r\n\r\n" for event in events)
        self.assertEqual(runner.parse_sse(raw), events)
        with self.assertRaisesRegex(AssertionError, "failed SSE"):
            runner.parse_sse('data: {"type":"response.failed"}\n\n')

    def test_previous_response_requires_a_real_generated_call(self):
        with patch.object(runner.bench_common, "request", return_value=HttpResponse(json.dumps(response("OK")).encode())):
            with self.assertRaisesRegex(AssertionError, "exactly one custom_tool_call"):
                runner.run_previous(runner.Client("http://localhost", 1), runner.MODEL)

    def test_previous_response_reuses_generated_call_id(self):
        client = runner.Client("http://localhost", 1)
        output = []
        for kind in ("custom", "function"):
            first = response(output=[{**call_item(kind), "call_id": "server_generated_" + kind}])
            marker = f"RESULT_{kind.upper()}_71953"
            output.extend([first, response(marker, previous="resp_123"), response(marker, previous="resp_123")])
        with patch.object(client, "send", side_effect=output) as send:
            runner.run_previous(client, runner.MODEL)
        bodies = [item.args[0] for item in send.call_args_list]
        self.assertEqual(bodies[0]["tool_choice"], {"type": "custom", "name": "record_text"})
        self.assertEqual(bodies[1]["input"][0]["call_id"], "server_generated_custom")
        self.assertEqual(bodies[4]["input"][0]["call_id"], "server_generated_function")
        self.assertEqual(bodies[1]["previous_response_id"], "resp_123")

    def test_multipart_and_refusal_history_do_not_reveal_expected_answer_in_question(self):
        for refusal in (False, True):
            body, expected = runner.history_output(runner.MODEL, refusal)
            self.assertEqual(len(body["input"][1]["content"]), 2 if refusal else 3)
            self.assertEqual(body["input"][2]["content"], [])
            for marker in expected.split():
                self.assertNotIn(marker, body["input"][-1]["content"])

    def test_multimodal_requires_fixtures_and_checks_order(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            runner.run_case(runner.Client("http://localhost", 1), runner.MODEL, {"id": "responses-image-video-input"})
        with tempfile.TemporaryDirectory() as tmp:
            image, video = Path(tmp) / "n.png", Path(tmp) / "v.mp4"
            image.write_bytes(b"a")
            video.write_bytes(b"b")
            body = runner.multimodal_payload(runner.MODEL, image, video)
            self.assertEqual([part["type"] for part in body["input"][0]["content"]], ["input_text", "input_image", "video_url"])
            bad = response('{"image_number":1,"video_numbers":[3,2,1]}')
            with patch.object(runner.bench_common, "request", return_value=HttpResponse(json.dumps(bad).encode())):
                with self.assertRaisesRegex(AssertionError, "ordering mismatch"):
                    runner.run_case(runner.Client("http://localhost", 1), runner.MODEL,
                                    {"id": "responses-image-video-input"}, image=image, video=video)

    def test_output_records_failed_checks_and_verifies_launch_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, output = Path(tmp) / "launch.json", Path(tmp) / "result.json"
            receipt.write_text(json.dumps({"container_id": "container"}))
            with patch.object(runner.bench_common, "verify_endpoint", return_value={"container_id": "container"}) as verify:
                with patch.object(runner.bench_common, "request", return_value=HttpResponse(json.dumps(response("WRONG")).encode())):
                    code = runner.main(["--case", "responses-basic", "--launch-receipt", str(receipt), "--output", str(output)])
            self.assertEqual(code, 1)
            verify.assert_called_once()
            evidence = json.loads(output.read_text())
            self.assertEqual(evidence["status"], "FAIL")
            self.assertEqual(evidence["cases"][0]["status"], "FAIL")
            self.assertEqual(evidence["cases"][0]["exchanges"][0]["response"], response("WRONG"))

    def test_launch_mismatch_never_sends_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, output = Path(tmp) / "launch.json", Path(tmp) / "result.json"
            receipt.write_text("{}")
            with patch.object(runner.bench_common, "verify_endpoint", side_effect=ValueError("image mismatch")):
                with patch.object(runner.bench_common, "request") as http:
                    code = runner.main(["--launch-receipt", str(receipt), "--output", str(output)])
            self.assertEqual(code, 1)
            http.assert_not_called()
            self.assertEqual(json.loads(output.read_text())["cases"], [])

    def test_previous_case_requires_store_and_single_api_worker_receipt(self):
        valid = {"command": ["docker", "run", "--env", "VLLM_ENABLE_RESPONSES_API_STORE=1",
                             "image", "serve", "--api-server-count", "1"]}
        runner.require_stored_history(valid)
        for command in (
            ["--api-server-count", "1"],
            ["--env", "VLLM_ENABLE_RESPONSES_API_STORE=0", "--api-server-count", "1"],
            ["--env", "VLLM_ENABLE_RESPONSES_API_STORE=1"],
            ["--env", "VLLM_ENABLE_RESPONSES_API_STORE=1", "--api-server-count", "2"],
        ):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "--responses-store"):
                runner.require_stored_history({"command": command})

        with tempfile.TemporaryDirectory() as tmp:
            receipt, output = Path(tmp) / "launch.json", Path(tmp) / "result.json"
            receipt.write_text("{}")
            with patch.object(runner.bench_common, "verify_endpoint", return_value={}):
                with patch.object(runner.bench_common, "request") as http:
                    code = runner.main(["--case", "responses-previous-response-id-tools",
                                        "--launch-receipt", str(receipt), "--output", str(output)])
            self.assertEqual(code, 1)
            http.assert_not_called()
            row = json.loads(output.read_text())["cases"][0]
            self.assertEqual(row["status"], "BLOCKED")
            self.assertIn("--responses-store", row["error"])


if __name__ == "__main__":
    unittest.main()
