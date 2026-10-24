"""GPU endpoint acceptance with generated media, exact assertions and raw evidence.

Prepare fixtures with --prepare-fixtures --fixtures-dir DIR. Run the disabled
chunk case first, then use its case.json as --chunk-baseline on the enabled
service. --case all runs every case compatible with the supplied launch receipt;
the case requiring the other launch mode is explicitly recorded as NOT_RUN.
Endpoint answers establish observable behavior. Scheduler boundary guarantees
require separate runtime processor/scheduler evidence.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scripts.acceptance.catalog import load, print_cases, select
from scripts.acceptance.core import parse_object
from scripts.acceptance.media import load_fixtures, prepare_fixtures, sha256
from scripts.bench_common import verify_endpoint

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEFAULT_BASE_URL = "http://127.0.0.1:8005"


def encoded_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def json_hash(value: Any) -> str:
    return hashlib.sha256(encoded_json(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def media_part(path: Path, kind: str) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or ("video/mp4" if kind == "video" else "image/png")
    encoded = base64.b64encode(path.read_bytes()).decode()
    return {"type": f"{kind}_url", f"{kind}_url": {"url": f"data:{mime};base64,{encoded}"}}


def video_part(path: Path, video_id: str | None = None) -> dict[str, Any]:
    part = media_part(path, "video")
    if video_id:
        part["video_url"]["video_id"] = video_id
    return part


def expected_video_id(part: dict[str, Any]) -> str:
    # The public stable identity contract excludes the protocol type alias.
    canonical = {"media": part["video_url"], "options": {}}
    return "vid_" + json_hash(canonical)[:16]


def request(base_url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=encoded_json(body),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"visual request returned HTTP {error.code}: {detail}") from error


def base_body(model: str, content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": model, "messages": [{"role": "user", "content": content}],
        "temperature": 0, "max_tokens": 2048,
        "chat_template_kwargs": {"thinking": False},
    }


def command_option(command: list[str], name: str) -> str:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"launch receipt lacks {name}") from error


def verify_receipt(case: dict[str, Any], receipt: dict[str, Any]) -> None:
    command = receipt.get("command")
    if not isinstance(command, list) or not all(isinstance(arg, str) for arg in command):
        raise ValueError("launch receipt must contain a command argument array")
    flag = case.get("launch_arg")
    if flag and flag not in command:
        raise ValueError(f"launch receipt does not contain required argument {flag}")
    limits = json.loads(command_option(command, "--limit-mm-per-prompt"))
    if int(case.get("images", 0)) > int(limits.get("image", 0)):
        raise ValueError("case image count exceeds the launch receipt limit")
    if case["kind"] == "image":
        return
    video_limits = limits.get("video", {})
    if not isinstance(video_limits, dict):
        raise ValueError("launch receipt must declare video count and frame limits")
    if int(case.get("videos", 1)) > int(video_limits.get("count", 0)):
        raise ValueError("case video count exceeds the launch receipt limit")
    media = json.loads(command_option(command, "--media-io-kwargs")).get("video", {})
    if media.get("num_frames") != 32 or video_limits.get("num_frames") != 32:
        raise ValueError("visual acceptance requires the 32-frame media policy")
    if media.get("fps") != 1:
        raise ValueError("visual acceptance requires the 1 FPS media policy")


def build_case(
    case: dict[str, Any], model: str, fixtures: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    used: list[dict[str, Any]] = []

    def image(number: int) -> dict[str, Any]:
        fixture = fixtures["images"][str(number)]
        used.append(fixture)
        return media_part(Path(fixture["path"]), "image")

    if case["kind"] == "image":
        count = int(case["images"])
        numbers = [((index * 7 + index // 3) % 3) + 1 for index in range(count)]
        content = [{"type": "text", "text": (
            f"This request contains exactly {count} images. "
            "Read the black digit in each image once, in supplied order. Preserve repetitions. "
            'Return only JSON {"numbers": [integer, ...]} '
            f"with exactly {count} integers, one per image. "
            "End your response after the closing JSON brace."
        )}]
        content.extend(image(number) for number in numbers)
        return base_body(model, content), {"numbers": numbers}, used

    names = case.get("fixtures", ["short"])
    video_items = [fixtures["videos"][name] for name in names]
    used.extend(video_items)
    videos = [video_part(Path(item["path"])) for item in video_items]
    # Stage sequences must be read from the pixels, not disclosed in the prompt.
    prompt = (
        "For each video, read the black digits and colored outer backgrounds around the white digit panels of its successive stages. "
        "Collapse adjacent repeated frames into one stage; preserve chronological stage order. "
        "Use lowercase English color names. "
    )
    if case["id"].startswith("visual-video-chunked-"):
        # Give both chunk modes the same request, distinct from earlier cases,
        # so a prior long-video request cannot supply a fully cached prefix.
        prompt = "Chunk mode comparison. " + prompt
    if case["kind"] == "mixed":
        prompt += (
            'Return only JSON {"before_image": integer, "video_numbers": [integer, ...], '
            '"video_colors": [string, ...], "after_image": integer}. '
            "before_image and after_image refer to the still images before and after the video."
        )
        content = [{"type": "text", "text": prompt}, image(3), videos[0], image(1)]
        expected = {
            "before_image": 3, "video_numbers": video_items[0]["numbers"],
            "video_colors": video_items[0]["colors"], "after_image": 1,
        }
    elif len(videos) == 2:
        prompt += 'Return only JSON {"videos": [{"numbers": [integer, ...], "colors": [string, ...]}, ...]} in input video order.'
        content = [{"type": "text", "text": prompt}, *videos]
        expected = {"videos": [{"numbers": item["numbers"], "colors": item["colors"]} for item in video_items]}
    else:
        prompt += 'Return only JSON {"numbers": [integer, ...], "colors": [string, ...]}.'
        content = [{"type": "text", "text": prompt}, videos[0]]
        expected = {"numbers": video_items[0]["numbers"], "colors": video_items[0]["colors"]}
    return base_body(model, content), expected, used


def exchange(
    base_url: str, body: dict[str, Any], timeout: float, directory: Path,
    expected: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "request.json", body)
    write_json(directory / "expected.json", expected)
    result = request(base_url, body, timeout)
    write_json(directory / "response.json", result)
    choices = result.get("choices", [])
    if not choices or choices[0].get("finish_reason") != "stop":
        raise ValueError(f"response did not finish normally: {choices}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("response has no text content")
    parsed = parse_object(content)
    if parsed != expected:
        raise ValueError(f"semantic mismatch: expected {expected}, got {parsed}")
    return message, {
        "request_sha256": json_hash(body), "response_sha256": json_hash(result),
        "parsed": parsed, "usage": result.get("usage"),
    }


def run_case(
    base_url: str, model: str, case: dict[str, Any], fixtures: dict[str, Any],
    receipt: dict[str, Any], timeout: float, output: Path,
    chunk_baseline: Path | None = None,
) -> dict[str, Any]:
    directory = output / case["id"]
    directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "case": case["id"], "status": "FAIL", "launch_receipt_sha256": json_hash(receipt),
        "endpoint": base_url, "evidence_scope": "GPU endpoint behavior", "steps": [],
    }
    try:
        verify_receipt(case, receipt)
        body, expected, used = build_case(case, model, fixtures)
        report["fixtures"] = used
        report["expected"] = expected
        if case["kind"] == "video_history":
            original_video = copy.deepcopy(body["messages"][0]["content"][1])
            identity = expected_video_id(original_video)
            first = base_body(model, [
                {"type": "text", "text": 'Read the Video ID from the video wrapper. Return only JSON {"video_id": "..."}.'},
                original_video,
            ])
            assistant, step = exchange(base_url, first, timeout, directory / "turn-1", {"video_id": identity})
            report["steps"].append(step)
            second = copy.deepcopy(first)
            second["messages"].extend([
                {"role": "assistant", "content": assistant["content"]},
                {"role": "user", "content": (
                    f"Continue analyzing {identity} from the earlier turn. "
                    + body["messages"][0]["content"][0]["text"]
                )},
            ])
            if second["messages"][0]["content"][1] != original_video:
                raise AssertionError("video content changed during history replay")
            _, step = exchange(base_url, second, timeout, directory / "turn-2", expected)
            report["history_video_id"] = identity
            report["history_media_sha256"] = json_hash(original_video)
            report["steps"].append(step)
        else:
            _, step = exchange(base_url, body, timeout, directory / "turn-1", expected)
            report["steps"].append(step)
        if case["id"] == "visual-video-chunked-frame-boundary":
            chunk_size = int(command_option(receipt["command"], "--max-num-batched-tokens"))
            usage = report["steps"][0].get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if prompt_tokens <= chunk_size:
                raise ValueError(f"request has {prompt_tokens} tokens, does not cross chunk size {chunk_size}")
            if cached_tokens is not None and prompt_tokens - int(cached_tokens) <= chunk_size:
                raise ValueError("uncached prompt tokens do not cross the chunk budget")
            report["chunk_prefill"] = {
                "max_num_batched_tokens": chunk_size, "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
            }
            if chunk_baseline is None:
                raise ValueError("chunk comparison requires --chunk-baseline PATH to the disabled case.json")
            baseline = json.loads(chunk_baseline.read_text())
            if baseline.get("case") != "visual-video-chunked-disabled" or baseline.get("status") != "PASS":
                raise ValueError("chunk baseline must be a passing disabled-mode case")
            before, after = baseline["steps"][0], report["steps"][0]
            if before["request_sha256"] != after["request_sha256"] or before["parsed"] != after["parsed"]:
                raise ValueError("chunk modes must receive the identical request and return identical semantic JSON")
            report["chunk_comparison"] = {"baseline_sha256": sha256(chunk_baseline), "status": "PASS"}
        report["status"] = "PASS"
    except (ValueError, RuntimeError, OSError, KeyError, IndexError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    write_json(directory / "case.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", help="case ID or all (all requires two launch modes for full coverage)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--fixtures-dir", type=Path)
    parser.add_argument("--prepare-fixtures", action="store_true")
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--chunk-baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    config = load("visual")
    if args.list:
        print_cases(config)
        return 0
    if args.fixtures_dir is None:
        parser.error("--fixtures-dir is required")
    if args.prepare_fixtures:
        prepare_fixtures(args.fixtures_dir)
        print(args.fixtures_dir / "manifest.json")
        if not args.case:
            return 0
    if not args.case or args.output is None or args.launch_receipt is None:
        parser.error("execution requires --case ID/all, --output and --launch-receipt")
    fixtures = load_fixtures(args.fixtures_dir)
    receipt = json.loads(args.launch_receipt.read_text())
    cases = select(config, None if args.case == "all" else args.case)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "launch-receipt.json", receipt)
    write_json(args.output / "fixture-manifest.json", fixtures)
    write_json(args.output / "endpoint.json", verify_endpoint(args.base_url, receipt))
    write_json(args.output / "runner.json", {
        "sha256": sha256(Path(__file__)), "source": str(Path(__file__).resolve()),
        "media_helper_sha256": sha256(Path(__file__).with_name("media.py")),
    })
    reports = []
    for case in cases:
        flag = case.get("launch_arg")
        if args.case == "all" and flag and flag not in receipt.get("command", []):
            report = {"case": case["id"], "status": "NOT_RUN", "reason": f"requires separate launch with {flag}"}
        else:
            report = run_case(args.base_url, args.model, case, fixtures, receipt, args.timeout, args.output, args.chunk_baseline)
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        write_json(args.output / "summary.json", {"cases": reports})
    return int(any(report["status"] == "FAIL" for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
