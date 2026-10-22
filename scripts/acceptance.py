"""Real-model smoke gates; small task coverage, not a lossless-quality certificate."""

import argparse
import base64
import copy
import json
import re
import time
from pathlib import Path

from scripts.benchmark import MODEL, json_request, verify_endpoint
from scripts.quality_metrics import (
    aggregate_speculative,
    speculative_metrics,
    text_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def image(number):
    content = (ROOT / f"tests/fixtures/video_frames/{number:02}.png").read_bytes()
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(content).decode()
        },
    }


def parse_object(content):
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
    return json.loads(content)


def repetitive(text):
    # Flag obvious runaway tails without treating repeated punctuation as a loop.
    words = text.split()
    return len(words) >= 32 and any(
        len(set(words[-n:])) > 1 and words[-n:] * 4 == words[-4 * n :]
        for n in range(2, min(33, len(words) // 4 + 1))
    )


def response_digest(message):
    """Hash all assistant-visible output, including structured tool calls."""

    parts = []
    if message.get("reasoning_content"):
        parts.append("<reasoning>" + message["reasoning_content"])
    if message.get("content"):
        parts.append(message["content"])
    if message.get("tool_calls"):
        parts.append(
            "<tool_calls>"
            + json.dumps(message["tool_calls"], sort_keys=True, separators=(",", ":"))
        )
    text = "".join(parts)
    return {"sha256": text_sha256(text), "chars": len(text)}


def cases(vision):
    cases = []
    for thinking in (False, True):
        cases.append(
            {
                "name": f"arithmetic-thinking-{thinking}",
                "thinking": thinking,
                "messages": [
                    {
                        "role": "user",
                        "content": "Calculate 17 * 23 + 19. Return only JSON with integer field answer.",
                    }
                ],
                "expected": {"answer": 410},
            }
        )
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a key",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
    cases.append(
        {
            "name": "tool-call",
            "thinking": False,
            "messages": [
                {"role": "user", "content": "Use lookup to get the value of key alpha."}
            ],
            "extra": {
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            },
            "expected_tool": {"name": "lookup", "arguments": {"key": "alpha"}},
        }
    )
    history = [
        {
            "role": "user",
            "content": 'Look up alpha and beta, return JSON {"sum": their sum}.',
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I need both values.",
            "tool_calls": [
                {
                    "id": key,
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": json.dumps({"key": key}),
                    },
                }
                for key in ("alpha", "beta")
            ],
        },
        {"role": "tool", "tool_call_id": "beta", "content": "29"},
        {"role": "tool", "tool_call_id": "alpha", "content": "17"},
    ]
    cases.append(
        {
            "name": "tool-continuation-reversed",
            "thinking": True,
            "messages": history,
            "expected": {"sum": 46},
        }
    )
    if vision:
        for numbers in ([1], [3, 1], [2, 1, 3, 2]):
            cases.append(
                {
                    "name": f"vision-{len(numbers)}-images",
                    "thinking": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Read the number at the top of each image in supplied order. "
                                    'Return only JSON {"numbers": [integer, ...]}. Preserve repetitions.',
                                }
                            ]
                            + [image(n) for n in numbers],
                        }
                    ],
                    "expected": {"numbers": numbers},
                }
            )
        calls = [
            {
                "id": ident,
                "type": "function",
                "function": {
                    "name": "view_image",
                    "arguments": json.dumps({"label": ident}),
                },
            }
            for ident in ("call_a", "call_b")
        ]
        messages = [
            {"role": "user", "content": "Inspect both sets of images."},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "call_b", "content": [image(3), image(2)]},
            {"role": "tool", "tool_call_id": "call_a", "content": [image(2), image(1)]},
            {
                "role": "user",
                "content": "Read the top image numbers. Return only JSON with fields "
                "call_a and call_b containing each tool call ID's numbers in image order.",
            },
        ]
        cases.append(
            {
                "name": "vision-reversed-tool-id",
                "thinking": False,
                "messages": messages,
                "expected": {"call_a": [2, 1], "call_b": [3, 2]},
            }
        )
    return cases


def needle_case(base, target):
    filler = (
        "This is an unrelated archive entry about ordinary engineering maintenance. "
    )
    unit = len(
        json_request(base, "/tokenize", {"model": MODEL, "prompt": filler})["tokens"]
    )
    count = max(1, (target - 512) // unit)
    parts = [filler] * count
    expected = {}
    for i, fraction in enumerate((0.02, 0.25, 0.50, 0.75, 0.98)):
        key, value = f"secret_{i}", f"AMBER-{1017 + i * 3989}"
        parts[int(count * fraction)] += f"\nThe exact value for {key} is {value}.\n"
        expected[key] = value
    text = (
        "".join(parts)
        + "\nReturn only JSON mapping secret_0 through secret_4 to their exact values."
    )
    return {
        "name": f"needle-target-{target}",
        "thinking": False,
        "target_tokens": target,
        "messages": [{"role": "user", "content": text}],
        "expected": expected,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8005")
    p.add_argument("--text-only", action="store_true")
    p.add_argument(
        "--needle-tokens", type=int, choices=(1024, 32768, 131072, 523000, 1047000)
    )
    p.add_argument("--launch-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; use a fresh evidence path")
    launch = json.loads(args.launch_receipt.read_text())
    if "image_id" not in launch:
        p.error("A real launch receipt is required")
    if launch["text_only"] and not args.text_only:
        p.error("Vision tests require a multimodal launch")
    if args.needle_tokens and args.needle_tokens + 512 > launch["context"]:
        p.error("Needle + output budget exceeds launched context")
    runtime = verify_endpoint(args.base_url, launch)
    ids = [r["id"] for r in json_request(args.base_url, "/v1/models")["data"]]
    if MODEL not in ids:
        raise ValueError("The served model is not 1017")
    suite = cases(not args.text_only)
    if args.needle_tokens:
        suite.append(needle_case(args.base_url, args.needle_tokens))
    rows = []
    speculative_rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in suite:
        row = {"name": case["name"], "status": "FAIL"}
        started = time.perf_counter()
        try:
            body = {
                "model": MODEL,
                "messages": copy.deepcopy(case["messages"]),
                "temperature": 0,
                "seed": 1017,
                "max_tokens": 512 if case.get("target_tokens") else 2048,
                "chat_template_kwargs": {
                    "thinking": case["thinking"],
                    "reasoning_effort": "high",
                },
            }
            body.update(case.get("extra", {}))
            result = json_request(args.base_url, "/v1/chat/completions", body)
            row["response"] = result
            choice = result["choices"][0]
            message = choice["message"]
            row["output"] = response_digest(message)
            row["speculative"] = speculative_metrics(
                result.get("usage"), result.get("metrics"), result.get("speculative")
            )
            speculative_rows.append(row)
            if choice["finish_reason"] == "length":
                raise ValueError("Truncated generation")
            if repetitive(
                (message.get("reasoning_content") or "")
                + " "
                + (message.get("content") or "")
            ):
                raise ValueError("Repeated output tail")
            if "expected_tool" in case:
                calls = message.get("tool_calls", [])
                if (
                    len(calls) != 1
                    or calls[0]["function"]["name"] != case["expected_tool"]["name"]
                ):
                    raise ValueError("Missing or wrong tool call")
                actual = json.loads(calls[0]["function"]["arguments"])
                if actual != case["expected_tool"]["arguments"]:
                    raise ValueError("Wrong tool arguments")
            elif parse_object(message.get("content") or "") != case["expected"]:
                raise ValueError("Answer differs from fixture")
            if case.get("target_tokens"):
                actual_tokens = result["usage"]["prompt_tokens"]
                row["actual_input_tokens"] = actual_tokens
                row["target_tokens"] = case["target_tokens"]
                if (
                    not 0.90 * case["target_tokens"]
                    <= actual_tokens
                    <= 1.05 * case["target_tokens"]
                ):
                    raise ValueError(
                        "Needle input length outside declared 10% tolerance"
                    )
            row["status"] = "PASS"
        except Exception as error:
            row["error"] = str(error)
        row["elapsed_s"] = time.perf_counter() - started
        rows.append(row)
        args.output.write_text(
            json.dumps(
                {"status": "IN_PROGRESS", "launch": launch, "cases": rows}, indent=2
            )
            + "\n"
        )
        print(case["name"], row["status"])
    passed = all(r["status"] == "PASS" for r in rows)
    args.output.write_text(
        json.dumps(
            {
                "status": "PASS" if passed else "FAIL",
                "scope": "Small deterministic smoke suite only; no whole-model quality equivalence claim",
                "launch": launch,
                "runtime": runtime,
                "speculative": aggregate_speculative(speculative_rows),
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
