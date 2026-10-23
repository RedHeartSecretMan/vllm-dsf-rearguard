"""Live regression smoke for Responses custom-tool history replay.

The request deliberately uses the two input item types that previously failed
at the Responses request schema: ``custom_tool_call`` and
``custom_tool_call_output``. Run it against a service started from the image
being tested::

    python3 -m scripts.responses_smoke --base-url http://127.0.0.1:8005
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt", default="Reply with OK.")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    report = run(args.base_url, args.model, args.prompt, args.timeout)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
