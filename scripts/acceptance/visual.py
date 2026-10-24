"""GPU endpoint runner for the visual acceptance catalog.

Image cases use the checked-in numbered fixtures. Video cases require a local
video supplied with ``--video``; the repository deliberately does not ship a
copyrighted or machine-specific video asset.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scripts.acceptance.catalog import load, print_cases, select
from scripts.acceptance.core import image, parse_object

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEFAULT_BASE_URL = "http://127.0.0.1:8005"
ROOT = Path(__file__).resolve().parents[2]


def video_part(path: Path, video_id: str | None = None) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    encoded = base64.b64encode(path.read_bytes()).decode()
    value: dict[str, Any] = {"url": f"data:{mime};base64,{encoded}"}
    if video_id:
        value["video_id"] = video_id
    return {"type": "video_url", "video_url": value}


def request(base_url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    encoded = json.dumps(body, separators=(",", ":")).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"visual request returned HTTP {error.code}: {detail}") from error
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not choices[0].get("message"):
        raise RuntimeError("visual response has no assistant message")
    return result


def image_body(case: dict[str, Any], model: str) -> tuple[dict[str, Any], list[int]]:
    count = int(case["images"])
    numbers = [(index % 3) + 1 for index in range(count)]
    content = [{"type": "text", "text": "Return only JSON with a numbers array in image order."}]
    content.extend(image(number) for number in numbers)
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max(256, count * 4),
    }, numbers


def video_body(
    case: dict[str, Any], model: str, path: Path, video_id_prefix: str = "vid_acceptance"
) -> dict[str, Any]:
    videos = int(case.get("videos", 1))
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "Describe the supplied video frames in chronological order. Return concise JSON.",
        }
    ]
    for index in range(videos):
        content.append(video_part(path, f"{video_id_prefix}_{index + 1}"))
    if case["kind"] == "mixed":
        content.insert(1, image(1))
        content.append(image(2))
    return {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 256}


def run_case(
    base_url: str,
    model: str,
    case: dict[str, Any],
    video: Path | None,
    launch_receipt: Path | None,
    timeout: float,
) -> dict[str, Any]:
    launch_arg = case.get("launch_arg")
    if launch_arg:
        if launch_receipt is None:
            raise ValueError(f"{case['id']} requires --launch-receipt to verify {launch_arg}")
        receipt = json.loads(launch_receipt.read_text())
        command = receipt.get("command", [])
        if launch_arg not in command:
            raise ValueError(f"launch receipt does not contain required argument {launch_arg}")
    if case["kind"] != "image" and launch_receipt is None:
        raise ValueError(f"{case['id']} requires --launch-receipt for media limits")
    if launch_receipt is not None:
        receipt = json.loads(launch_receipt.read_text())
        command = receipt.get("command", [])
        try:
            limit = json.loads(command[command.index("--limit-mm-per-prompt") + 1])
        except (ValueError, IndexError, json.JSONDecodeError) as error:
            raise ValueError("launch receipt lacks valid multimodal limits") from error
        if int(case.get("images", 0)) > int(limit.get("image", 0)):
            raise ValueError("case image count exceeds the launch receipt limit")
        if case["kind"] != "image":
            try:
                media = json.loads(command[command.index("--media-io-kwargs") + 1])
            except (ValueError, IndexError, json.JSONDecodeError) as error:
                raise ValueError("launch receipt lacks valid media IO policy") from error
            if int(case.get("videos", 1)) > int(limit.get("video", {}).get("count", 0)):
                raise ValueError("case video count exceeds the launch receipt limit")
            if int(case.get("max_frames", 0)) > int(limit.get("video", {}).get("num_frames", 0)):
                raise ValueError("case frame count exceeds the launch receipt limit")
            if "fps" in case and media.get("video", {}).get("fps") != case["fps"]:
                raise ValueError("case fps differs from the launch receipt media policy")
    expected_numbers = None
    if case["kind"] == "image":
        body, expected_numbers = image_body(case, model)
    else:
        if video is None:
            raise ValueError("video cases require --video PATH")
        prefix = "vid_acceptance_history" if case["kind"] == "video_history" else "vid_acceptance"
        body = video_body(case, model, video, prefix)
        if case["kind"] == "video_history":
            replay = video_body(case, model, video, prefix)
            if body["messages"][0]["content"] != replay["messages"][0]["content"]:
                raise RuntimeError("video history payload is not stable across replay")
            video_id = f"{prefix}_1"
            body["messages"].extend(
                [
                    {"role": "assistant", "content": f"I inspected video {video_id}."},
                    {"role": "user", "content": f"Continue analysis of {video_id}."},
                ]
            )
    result = request(base_url, body, timeout)
    message = result["choices"][0]["message"]
    text = message.get("content") or ""
    if not text and not message.get("tool_calls"):
        raise RuntimeError("visual response contains no assistant output")
    report: dict[str, Any] = {"status": "PASS", "case": case["id"], "response_chars": len(text)}
    if case["kind"] == "video_history":
        report["history_video_id"] = "vid_acceptance_history_1"
        report["assertions"] = list(case.get("requires", []))
    if case["kind"] == "image":
        try:
            report["parsed"] = parse_object(text)
        except (ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("image case did not return JSON") from error
        if report["parsed"].get("numbers") != expected_numbers:
            raise RuntimeError(
                f"image order mismatch: expected {expected_numbers}, "
                f"got {report['parsed'].get('numbers')}"
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list catalog cases")
    parser.add_argument("--case")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--video", type=Path, help="local video fixture for video cases")
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    config = load("visual")
    if args.list:
        print_cases(config)
        return 0
    cases = select(config, args.case)
    if len(cases) != 1:
        parser.error("--case is required when executing a visual case")
    report = run_case(
        args.base_url,
        args.model,
        cases[0],
        args.video,
        args.launch_receipt,
        args.timeout,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
