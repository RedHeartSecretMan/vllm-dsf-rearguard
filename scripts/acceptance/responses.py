"""Run the seven Responses acceptance cases against a launched GPU service.

    python -m scripts.acceptance.responses --case all \
        --launch-receipt launch.json --fixtures-dir /tmp/visual-fixtures \
        --output /tmp/responses-acceptance.json

Every selected case sends HTTP requests and asserts the returned protocol and
content. The evidence includes requests, responses and the original SSE stream.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import time
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts import bench_common
from scripts.acceptance.catalog import load, print_cases, select

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
TOOL_INPUT = 'line one\nline "two" \\ end'
HISTORY_MARKER = "HISTORY_RESULT_58319"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def response_text(response: dict[str, Any]) -> str:
    return "\n".join(
        part["text"]
        for item in response.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text" and isinstance(part.get("text"), str)
    )


def validate_response(response: dict[str, Any]) -> None:
    require(response.get("object") == "response", "missing response object discriminator")
    require(bool(response.get("id")), "missing response ID")
    require(response.get("status") == "completed", f"response not completed: {response.get('status')}")
    require(not response.get("error"), f"response error: {response.get('error')}")
    require(isinstance(response.get("output"), list), "response output must be a list")
    usage = response.get("usage") or {}
    require(usage.get("input_tokens", 0) > 0, "missing positive input token usage")
    require(usage.get("output_tokens", 0) > 0, "missing positive output token usage")


def parse_sse(raw: str) -> list[dict[str, Any]]:
    events = []
    for frame in raw.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:"))
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        require(isinstance(event, dict), "SSE data is not an object")
        require(event.get("type") not in ("error", "response.failed", "response.incomplete"), f"failed SSE event: {event}")
        events.append(event)
    require(bool(events), "empty SSE stream")
    return events


class Client:
    """Record every exchange before checking it so failures retain evidence."""

    def __init__(self, base_url: str, timeout: float):
        self.base_url, self.timeout = base_url, timeout
        self.exchanges: list[dict[str, Any]] = []

    def send(self, body: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        exchange: dict[str, Any] = {"request": body}
        self.exchanges.append(exchange)
        started = time.perf_counter()
        try:
            with bench_common.request(self.base_url, "/v1/responses", body, self.timeout) as result:
                exchange["http_status"] = result.status
                raw = result.read().decode()
                exchange["raw_response"] = raw
                if body.get("stream"):
                    events = parse_sse(raw)
                    exchange["events"] = events
                    return events
                response = json.loads(raw)
                exchange["response"] = response
                validate_response(response)
                return response
        except urllib.error.HTTPError as error:
            exchange["http_status"] = error.code
            exchange["raw_response"] = error.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {exchange['raw_response']}") from error
        finally:
            exchange["seconds"] = time.perf_counter() - started


def payload(model: str, items: str | list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return {
        "model": model,
        "input": items,
        "temperature": 0,
        "max_output_tokens": 256,
        "chat_template_kwargs": {"thinking": False},
        "stream": False,
        **kwargs,
    }


def assert_text(response: dict[str, Any], expected: str) -> None:
    text = response_text(response)
    require(text.strip() == expected, f"expected text {expected!r}, got {text!r}")


def custom_history(model: str) -> dict[str, Any]:
    return payload(model, [
        {"role": "user", "content": "Read the acceptance record."},
        {"id": "ctc_acceptance", "type": "custom_tool_call", "status": "completed",
         "call_id": "call_acceptance", "name": "read_record", "input": "record.txt"},
        {"id": "cto_acceptance", "type": "custom_tool_call_output", "status": "completed",
         "call_id": "call_acceptance", "output": HISTORY_MARKER},
        {"role": "user", "content": "Repeat the exact tool result, with no other text."},
    ])


def tool_payload(model: str, kind: str, stream: bool = False) -> dict[str, Any]:
    tool: dict[str, Any] = {"type": kind, "name": "record_text", "description": "Record the supplied text exactly."}
    if kind == "function":
        tool["parameters"] = {"type": "object", "properties": {"input": {"type": "string"}},
                              "required": ["input"], "additionalProperties": False}
    return payload(
        model,
        "Call record_text once with the following exact text (JSON string notation): " + json.dumps(TOOL_INPUT),
        tools=[tool], tool_choice={"type": kind, "name": "record_text"},
        parallel_tool_calls=False, stream=stream, store=True,
    )


def assert_tool(response: dict[str, Any], kind: str) -> dict[str, Any]:
    item_type = "custom_tool_call" if kind == "custom" else "function_call"
    calls = [item for item in response["output"] if item.get("type") in ("function_call", "custom_tool_call")]
    require(len(calls) == 1 and calls[0].get("type") == item_type, f"expected exactly one {item_type}, got {calls}")
    call = calls[0]
    require(call.get("name") == "record_text", "tool name mismatch")
    require(bool(call.get("id")) and bool(call.get("call_id")), "tool call lacks item ID/call ID")
    value = call.get("input") if kind == "custom" else json.loads(call["arguments"]).get("input")
    require(value == TOOL_INPUT, f"tool input changed: {value!r}")
    return call


def assert_custom_stream(events: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [event for event in events if event.get("type") == "response.completed"]
    require(len(completed) == 1, "stream must contain exactly one response.completed")
    final = completed[0]["response"]
    validate_response(final)
    call = assert_tool(final, "custom")
    added = [event for event in events if event.get("type") == "response.output_item.added" and event.get("item", {}).get("id") == call["id"]]
    done = [event for event in events if event.get("type") == "response.output_item.done" and event.get("item", {}).get("id") == call["id"]]
    deltas = [event for event in events if event.get("type") == "response.custom_tool_call_input.delta"]
    inputs_done = [event for event in events if event.get("type") == "response.custom_tool_call_input.done"]
    require(len(added) == len(done) == len(inputs_done) == 1 and bool(deltas), "missing custom-tool add/delta/input.done/item.done events")
    require(added[0]["item"].get("type") == "custom_tool_call", "stream opened a function call instead of a custom call")
    require(added[0]["item"].get("call_id") == call["call_id"], "stream changed the call ID")
    require(all(event.get("item_id") == call["id"] for event in deltas + inputs_done), "custom events reference the wrong item")
    require(all(event.get("output_index") == added[0].get("output_index") for event in deltas + inputs_done + done), "custom events reference the wrong output index")
    require("".join(event["delta"] for event in deltas) == TOOL_INPUT, "custom deltas do not reconstruct free-form input")
    require(inputs_done[0].get("input") == TOOL_INPUT and done[0]["item"].get("input") == TOOL_INPUT, "custom done events differ from final input")
    require(done[0]["item"].get("call_id") == call["call_id"] and done[0]["item"].get("name") == call["name"], "custom done event changed tool identity")
    require(not any(event.get("type", "").startswith("response.function_call_arguments") for event in events), "custom stream emitted function argument events")
    positions = [events.index(added[0]), events.index(deltas[0]), events.index(inputs_done[0]), events.index(done[0]), events.index(completed[0])]
    require(positions == sorted(positions) and len(set(positions)) == len(positions), "custom stream lifecycle order is invalid")
    return {"response_id": final["id"], "call_id": call["call_id"], "event_count": len(events)}


def run_previous(client: Client, model: str) -> dict[str, Any]:
    rows = []
    for kind in ("custom", "function"):
        first = client.send(tool_payload(model, kind))
        call = assert_tool(first, kind)
        marker = f"RESULT_{kind.upper()}_71953"
        output_type = "custom_tool_call_output" if kind == "custom" else "function_call_output"
        second = client.send(payload(model, [
            {"type": output_type, "call_id": call["call_id"], "output": marker},
            {"role": "user", "content": "Repeat only the exact result of the tool just called."},
        ], previous_response_id=first["id"], store=True))
        assert_text(second, marker)
        require(second.get("previous_response_id") == first["id"], "response lost previous_response_id")
        third = client.send(payload(model, "Repeat that exact tool result again, with no other text.",
                                    previous_response_id=second["id"], store=True))
        assert_text(third, marker)
        rows.append({"kind": kind, "call_id": call["call_id"], "response_ids": [first["id"], second["id"], third["id"]]})
    return {"tool_history_chains": rows}


def require_stored_history(receipt: dict[str, Any]) -> None:
    command = receipt.get("command", [])
    environment = [command[index + 1] for index, value in enumerate(command[:-1])
                   if value == "--env"]
    try:
        workers = command[command.index("--api-server-count") + 1]
    except (ValueError, IndexError):
        workers = None
    if "VLLM_ENABLE_RESPONSES_API_STORE=1" not in environment or workers != "1":
        raise ValueError(
            "previous_response_id acceptance requires launch.py --responses-store "
            "(VLLM_ENABLE_RESPONSES_API_STORE=1 and --api-server-count 1)"
        )


def history_output(model: str, refusal: bool = False) -> tuple[dict[str, Any], str]:
    if refusal:
        content = [{"type": "refusal", "refusal": "Recorded refusal marker: REFUSAL_84563."},
                   {"type": "output_text", "text": "Recorded trailing marker: TAIL_92017.", "annotations": []}]
        expected = "REFUSAL_84563 TAIL_92017"
    else:
        content = [{"type": "output_text", "text": f"Marker {index}: {word}.", "annotations": []}
                   for index, word in enumerate(("PART_A_5371", "PART_B_6842", "PART_C_9205"), 1)]
        expected = "PART_A_5371 PART_B_6842 PART_C_9205"
    items = [
        {"role": "user", "content": "Record the following acceptance markers."},
        {"id": "msg_history", "type": "message", "role": "assistant", "status": "completed", "content": content},
        {"id": "msg_empty", "type": "message", "role": "assistant", "status": "completed", "content": []},
        {"role": "user", "content": "Repeat every recorded marker in original order, separated by single spaces, with no other text."},
    ]
    return payload(model, items), expected


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0]
    if mime is None:
        raise ValueError(f"cannot determine media type: {path}")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def multimodal_payload(model: str, image: Path, video: Path) -> dict[str, Any]:
    return payload(model, [{"role": "user", "content": [
        {"type": "input_text", "text": 'Return only JSON {"image_number": N, "video_numbers": [N, ...]}. Read the large digit in the still image and the video. For the video, list digits in chronological order, collapsing adjacent repeats.'},
        {"type": "input_image", "image_url": data_uri(image), "detail": "auto"},
        {"type": "video_url", "video_url": {"url": data_uri(video)}},
    ]}])


def parse_object(text: str) -> dict[str, Any]:
    if text.strip().startswith("```"):
        text = text.strip().split("\n", 1)[1].rsplit("```", 1)[0]
    result = json.loads(text)
    require(isinstance(result, dict), "expected JSON object")
    return result


def run_case(client: Client, model: str, case: dict[str, Any], *, image: Path | None = None,
             video: Path | None = None, expected_image: int = 1,
             expected_video: list[int] | None = None) -> dict[str, Any]:
    case_id = case["id"]
    if case_id == "responses-basic":
        result = client.send(payload(model, "Reply with exactly ACCEPTANCE_OK, no other text."))
        assert_text(result, "ACCEPTANCE_OK")
    elif case_id == "responses-custom-tool-history":
        result = client.send(custom_history(model))
        assert_text(result, HISTORY_MARKER)
    elif case_id == "responses-custom-tool-stream":
        return assert_custom_stream(client.send(tool_payload(model, "custom", stream=True)))
    elif case_id == "responses-previous-response-id-tools":
        return run_previous(client, model)
    elif case_id in ("responses-multipart-output", "responses-refusal-output"):
        body, expected = history_output(model, refusal=case_id == "responses-refusal-output")
        result = client.send(body)
        assert_text(result, expected)
    elif case_id == "responses-image-video-input":
        if image is None or video is None:
            raise ValueError("multimodal case requires --image and --video, or --fixtures-dir")
        result = client.send(multimodal_payload(model, image, video))
        actual = parse_object(response_text(result))
        expected = {"image_number": expected_image, "video_numbers": expected_video or [1, 2, 3]}
        require(actual == expected, f"multimodal ordering mismatch: expected {expected}, got {actual}")
    else:
        raise ValueError(f"case has no live implementation: {case_id}")
    return {"response_id": result["id"], "text": response_text(result)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", default="all")
    parser.add_argument("--base-url", default=bench_common.DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixtures-dir", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--expected-image-number", type=int, default=1)
    parser.add_argument("--expected-video-numbers", default="1,2,3")
    args = parser.parse_args(argv)
    config = load("responses")
    if args.list:
        print_cases(config)
        return 0
    cases = select(config, None if args.case == "all" else args.case)
    report: dict[str, Any] = {
        "status": "FAIL", "runner": config["runner"], "model": args.model,
        "base_url": args.base_url, "started_at": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": [],
    }
    try:
        if args.launch_receipt is None:
            raise ValueError("GPU acceptance requires --launch-receipt")
        receipt = json.loads(args.launch_receipt.read_text())
        report["launch"] = receipt
        report["endpoint"] = bench_common.verify_endpoint(args.base_url, receipt)
        if args.fixtures_dir:
            from scripts.acceptance.media import load_fixtures
            fixtures = load_fixtures(args.fixtures_dir)
            args.image = Path(fixtures["images"]["1"]["path"])
            args.video = Path(fixtures["videos"]["short"]["path"])
        for name in ("image", "video"):
            path = getattr(args, name)
            if path:
                report[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for case in cases:
            client = Client(args.base_url, args.timeout)
            row: dict[str, Any] = {"case": case["id"], "status": "FAIL", "exchanges": client.exchanges}
            report["cases"].append(row)
            try:
                if case["id"] == "responses-previous-response-id-tools":
                    try:
                        require_stored_history(receipt)
                    except ValueError as error:
                        row["status"] = "BLOCKED"
                        raise error
                row.update(run_case(client, args.model, case, image=args.image, video=args.video,
                                    expected_image=args.expected_image_number,
                                    expected_video=[int(value) for value in args.expected_video_numbers.split(",")]))
                row["status"] = "PASS"
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            print(json.dumps({key: row[key] for key in ("case", "status", "error") if key in row}), flush=True)
        report["status"] = "PASS" if all(row["status"] == "PASS" for row in report["cases"]) else "FAIL"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
