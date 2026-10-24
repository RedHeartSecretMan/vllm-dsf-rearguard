"""Live regression runner for the Responses protocol acceptance catalog.

The request deliberately uses the two input item types that previously failed
at the Responses request schema: ``custom_tool_call`` and
``custom_tool_call_output``. Run it against a service started from the image
being tested::

    python3 -m scripts.acceptance.responses --list
    python3 -m scripts.acceptance.responses --case responses-custom-tool-history \
        --base-url http://127.0.0.1:8005
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any

from scripts.acceptance.catalog import load, print_cases, select

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def smoke_payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "id": "ctc_responses_smoke",
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": "call_responses_smoke",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** End Patch",
            },
            {
                "id": "cto_responses_smoke",
                "type": "custom_tool_call_output",
                "status": "completed",
                "call_id": "call_responses_smoke",
                "output": "patch applied",
            },
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": 32,
        "temperature": 0,
        "stream": False,
    }


def basic_payload(model: str, prompt: str, stream: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": 32,
        "temperature": 0,
        "stream": stream,
    }


def run(base_url: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    body = json.dumps(smoke_payload(model, prompt), separators=(",", ":")).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/responses",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Responses smoke returned HTTP {error.code}: {detail}") from error
    if result.get("object") != "response" or not isinstance(result.get("output"), list):
        raise RuntimeError("Responses smoke returned an invalid response object")
    return {
        "status": "PASS",
        "response_id": result.get("id"),
        "output_items": len(result["output"]),
    }


def run_basic(
    base_url: str, model: str, prompt: str, timeout: float, stream: bool = False
) -> dict[str, Any]:
    body = json.dumps(basic_payload(model, prompt, stream), separators=(",", ":")).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/responses",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if stream:
                chunks = [line for line in response if line.startswith(b"data:")]
                if not any(b"response.completed" in line for line in chunks):
                    raise RuntimeError("Responses stream did not complete")
                return {"status": "PASS", "events": len(chunks), "stream": True}
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Responses request returned HTTP {error.code}: {detail}") from error
    if result.get("object") != "response" or not isinstance(result.get("output"), list):
        raise RuntimeError("Responses request returned an invalid response object")
    return {"status": "PASS", "response_id": result.get("id"), "stream": False}


def run_stream(base_url: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    payload = smoke_payload(model, prompt)
    payload["stream"] = True
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/responses",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            events = [line for line in response if line.startswith(b"data:")]
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Responses stream returned HTTP {error.code}: {detail}") from error
    if not any(b"response.completed" in line for line in events):
        raise RuntimeError("Responses custom-tool stream did not complete")
    return {"status": "PASS", "events": len(events), "stream": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list catalog cases")
    parser.add_argument("--case", default="responses-custom-tool-history")
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt", default="Reply with OK.")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    config = load("responses")
    if args.list:
        print_cases(config)
        return 0
    case = select(config, args.case)[0]
    if case["mode"] == "unit":
        report = {
            "status": "UNIT",
            "case": case["id"],
            "detail": "covered by tests/test_responses_custom_tools.py",
        }
    elif case["id"] == "responses-basic":
        report = run_basic(args.base_url, args.model, args.prompt, args.timeout)
    elif case["id"] == "responses-custom-tool-stream":
        report = run_stream(args.base_url, args.model, args.prompt, args.timeout)
    else:
        report = run(args.base_url, args.model, args.prompt, args.timeout)
    report["case"] = case["id"]
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"PASS", "UNIT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
