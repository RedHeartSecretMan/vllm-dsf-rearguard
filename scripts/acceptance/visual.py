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


def video_body(case: dict[str, Any], model: str, path: Path) -> dict[str, Any]:
    videos = int(case.get("videos", 1))
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "Describe the supplied video frames in chronological order. Return concise JSON.",
        }
    ]
    for index in range(videos):
        content.append(video_part(path, f"vid_acceptance_{index + 1}"))
    if case["kind"] == "mixed":
        content.insert(1, image(1))
        content.append(image(2))
    return {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 256}


def run_case(base_url: str, model: str, case: dict[str, Any], video: Path | None, timeout: float) -> dict[str, Any]:
    expected_numbers = None
    if case["kind"] == "image":
        body, expected_numbers = image_body(case, model)
    else:
        if video is None:
            raise ValueError("video cases require --video PATH")
        body = video_body(case, model, video)
    result = request(base_url, body, timeout)
    message = result["choices"][0]["message"]
    text = message.get("content") or ""
    if not text and not message.get("tool_calls"):
        raise RuntimeError("visual response contains no assistant output")
    report: dict[str, Any] = {"status": "PASS", "case": case["id"], "response_chars": len(text)}
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
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    config = load("visual")
    if args.list:
        print_cases(config)
        return 0
    cases = select(config, args.case)
    if len(cases) != 1:
        parser.error("--case is required when executing a visual case")
    report = run_case(args.base_url, args.model, cases[0], args.video, args.timeout)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
