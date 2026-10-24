"""Run one frozen acceptance cell on the dedicated profiling endpoint."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import time
import urllib.error
from pathlib import Path
from typing import Any

from scripts import bench_common
from scripts.acceptance import core, responses, visual
from scripts.benchmarks import profile_workload


def _begin(base: str, cell: dict, receipt: dict, out: Path) -> tuple[dict, Path, float]:
    if base != profile_workload.BASE_URL:
        raise ValueError("profiling acceptance must use http://127.0.0.1:6688")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "cell_id": cell["cell_id"], "case_id": cell["case_id"], "status": "FAIL",
        "error": None, "base_url": base, "rawpaths": [], "raw_files": [], "steps": [],
        "launch_receipt_sha256": visual.json_hash(receipt),
    }
    return report, out, time.perf_counter()


def _finish(report: dict, out: Path, started: float) -> dict:
    report["elapsed_s"] = time.perf_counter() - started
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _raw(report: dict, out: Path, name: str, value: Any) -> dict:
    raw = gzip.compress(json.dumps(value, ensure_ascii=False).encode(), mtime=0)
    path = out / f"{name}.json.gz"
    with path.open("xb") as handle:
        handle.write(raw)
    reference = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    report["rawpaths"].append(str(path))
    report["raw_files"].append(reference)
    return reference


def _salt(frozen: dict, cell: dict) -> str:
    return "profile-acceptance:" + hashlib.sha256(
        json.dumps([frozen["run_id"], cell["cell_id"]]).encode()
    ).hexdigest()


def _chat(base: str, body: dict, receipt: dict, report: dict, out: Path,
          timeout: float, *, cold: bool = True) -> tuple[dict, dict]:
    step = {"kv_cache_policy": "cold" if cold else "history_reuse", "usage": None}
    report["steps"].append(step)
    evidence = {"request": body, "response": None, "error": None}
    try:
        result = bench_common.json_request(base, "/v1/chat/completions", body, timeout)
        evidence["response"] = result
        usage = result.get("usage") or {}
        step["usage"] = usage
        actual = usage.get("prompt_tokens")
        output = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        step.update(actual_input_tokens=actual, output_tokens=output,
                    output_budget=body["max_tokens"], cached_tokens=cached)
        if type(actual) is not int or actual <= 0:
            raise ValueError("missing positive prompt_tokens usage")
        if type(output) is not int or not 0 < output <= body["max_tokens"]:
            raise ValueError("output token usage is missing or exceeds the requested budget")
        if actual + body["max_tokens"] > receipt["context"]:
            raise ValueError("actual input plus output budget exceeds launched context")
        if cold and (type(cached) is not int or cached != 0):
            raise ValueError(f"cold cache requires cached_tokens=0, observed {cached}")
        return result, step
    except Exception as error:
        if isinstance(error, urllib.error.HTTPError):
            evidence["response"] = {"http_status": error.code, "body": error.read().decode(errors="replace")}
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        step["raw"] = _raw(report, out, f"turn-{len(report['steps']):02d}", evidence)


def _semantic_chat(base: str, body: dict, expected: dict, receipt: dict,
                   report: dict, out: Path, timeout: float, *, cold: bool = True) -> tuple[dict, dict]:
    result, step = _chat(base, body, receipt, report, out, timeout, cold=cold)
    choices = result.get("choices", [])
    if not choices or choices[0].get("finish_reason") != "stop":
        raise ValueError("visual response did not finish normally")
    message = choices[0].get("message", {})
    parsed = core.parse_object(message.get("content") or "")
    if parsed != expected:
        raise ValueError(f"semantic mismatch: expected {expected}, got {parsed}")
    step["parsed"] = parsed
    return message, step


def run_core(base, cell, frozen, receipt, out, timeout=3600) -> dict:
    report, out, started = _begin(base, cell, receipt, out)
    report["kv_cache_policy"] = "cold; encoder cache may be reused"
    try:
        report["endpoint"] = bench_common.verify_endpoint(base, receipt)
        case_id = cell["case_id"]
        if case_id.startswith("needle-target-"):
            case = copy.deepcopy(frozen["bodies"][case_id])
            if case["target_tokens"] != cell["isl"]:
                raise ValueError("frozen needle target differs from the selected cell")
        else:
            case = next(item for item in core.cases(True) if item["name"] == case_id)
        budget = 512 if case.get("target_tokens") else 2048
        if case.get("target_tokens", 0) + budget > receipt["context"]:
            raise ValueError("needle target plus output budget exceeds launched context")
        if receipt.get("text_only") and (case_id.startswith("vision-") or cell.get("kind") in ("image", "image_tool_history")):
            raise ValueError("image case requires a multimodal launch")
        body = {
            "model": bench_common.MODEL, "messages": copy.deepcopy(case["messages"]),
            "temperature": 0, "seed": 1017, "max_tokens": budget,
            "chat_template_kwargs": {"thinking": case["thinking"], "reasoning_effort": "high"},
            **copy.deepcopy(case.get("extra", {})), "cache_salt": _salt(frozen, cell),
        }
        result, step = _chat(base, body, receipt, report, out, timeout)
        choice = result["choices"][0]
        message = choice["message"]
        if choice.get("finish_reason") not in (("stop", "tool_calls") if "expected_tool" in case else ("stop",)):
            raise ValueError("generation did not finish normally")
        if core.repetitive((message.get("reasoning_content") or "") + " " + (message.get("content") or "")):
            raise ValueError("Repeated output tail")
        if "expected_tool" in case:
            calls = message.get("tool_calls", [])
            expected = case["expected_tool"]
            if len(calls) != 1 or calls[0]["function"]["name"] != expected["name"]:
                raise ValueError("Missing or wrong tool call")
            if json.loads(calls[0]["function"]["arguments"]) != expected["arguments"]:
                raise ValueError("Wrong tool arguments")
        elif core.parse_object(message.get("content") or "") != case["expected"]:
            raise ValueError("Answer differs from fixture")
        if case.get("target_tokens") and not 0.90 * case["target_tokens"] <= step["actual_input_tokens"] <= 1.05 * case["target_tokens"]:
            raise ValueError("Needle input length outside declared 10% tolerance")
        report.update(status="PASS", output=core.response_digest(message),
                      actual_input_tokens=step["actual_input_tokens"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return _finish(report, out, started)


def run_visual(base, cell, frozen, receipt, out, timeout=600) -> dict:
    report, out, started = _begin(base, cell, receipt, out)
    report["kv_cache_policy"] = "cold; video-history second turn may reuse KV; encoder cache may be reused"
    try:
        report["endpoint"] = bench_common.verify_endpoint(base, receipt)
        case = cell["source_case"]
        if cell["case_id"].startswith("visual-video-chunked-"):
            raise ValueError("dedicated chunk comparison cases are not part of fixed-profile acceptance")
        visual.verify_receipt(case, receipt)
        highres = cell["case_id"] == "visual-video-highres-semantic"
        if highres:
            body, expected = profile_workload.highres_video_case(frozen)
        else:
            body, expected, used = visual.build_case(case, bench_common.MODEL, frozen["fixtures"])
            report["fixtures"] = used
        body = copy.deepcopy(body)
        body.update(seed=1017, cache_salt=_salt(frozen, cell))
        report["expected"] = expected
        if case["kind"] == "video_history":
            original = copy.deepcopy(body["messages"][0]["content"][1])
            identity = visual.expected_video_id(original)
            first = visual.base_body(bench_common.MODEL, [
                {"type": "text", "text": 'Read the Video ID from the video wrapper. Return only JSON {"video_id": "..."}.'},
                original,
            ])
            first.update(seed=1017, cache_salt=body["cache_salt"])
            assistant, _ = _semantic_chat(base, first, {"video_id": identity}, receipt, report, out, timeout)
            second = copy.deepcopy(first)
            second["messages"].extend([
                {"role": "assistant", "content": assistant["content"]},
                {"role": "user", "content": f"Continue analyzing {identity} from the earlier turn. " + body["messages"][0]["content"][0]["text"]},
            ])
            if second["messages"][0]["content"][1] != original:
                raise ValueError("video content changed during history replay")
            _semantic_chat(base, second, expected, receipt, report, out, timeout, cold=False)
            report.update(history_video_id=identity, history_media_sha256=visual.json_hash(original))
        else:
            _, step = _semantic_chat(base, body, expected, receipt, report, out, timeout)
            if highres:
                chunk = int(visual.command_option(receipt["command"], "--max-num-batched-tokens"))
                actual = step["actual_input_tokens"]
                if actual <= 16384:
                    raise ValueError("high-resolution video must have more than 16384 input tokens")
                if chunk == 16384 and "--no-disable-chunked-mm-input" not in receipt["command"]:
                    raise ValueError("16K high-resolution video requires chunked multimodal input")
                report["chunk_prefill"] = {
                    "max_num_batched_tokens": chunk, "actual_input_tokens": actual,
                    "cached_tokens": step["cached_tokens"], "crosses_chunk_budget": actual > chunk,
                }
        report["actual_input_tokens"] = [step["actual_input_tokens"] for step in report["steps"]]
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return _finish(report, out, started)


def run_responses(base, cell, frozen, receipt, out, timeout=600) -> dict:
    report, out, started = _begin(base, cell, receipt, out)
    report["kv_cache_policy"] = "protocol; original Responses requests and history reuse are preserved"
    client = responses.Client(base, timeout)
    try:
        report["endpoint"] = bench_common.verify_endpoint(base, receipt)
        case = cell["source_case"]
        if case["id"] == "responses-previous-response-id-tools":
            responses.require_stored_history(receipt)
        kwargs = {}
        if case["id"] == "responses-image-video-input":
            image = frozen["fixtures"]["images"]["1"]
            video = frozen["fixtures"]["videos"]["short"]
            kwargs = {"image": Path(image["path"]), "video": Path(video["path"]),
                      "expected_image": image.get("number", 1), "expected_video": video["numbers"]}
        report["result"] = responses.run_case(client, bench_common.MODEL, case, **kwargs)
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        for index, exchange in enumerate(client.exchanges):
            raw = _raw(report, out, f"exchange-{index + 1:02d}", exchange)
            response = exchange.get("response")
            if response is None:
                response = next((event["response"] for event in exchange.get("events", [])
                                 if event.get("type") == "response.completed"), {})
            usage = response.get("usage") if isinstance(response, dict) else None
            usage = usage if isinstance(usage, dict) else {}
            details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
            report["steps"].append({
                "usage": usage, "cached_tokens": details.get("cached_tokens"), "kv_cache_policy": "protocol",
                "raw": raw,
            })
    return _finish(report, out, started)
