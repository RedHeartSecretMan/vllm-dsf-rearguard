"""Compute token-level perplexity through the OpenAI-compatible prompt API.

The input is JSONL with one record per line.  A record may contain ``tokens``
(exact model token IDs) or ``text`` (tokenized by the served model).  The
runner fails closed when the API omits a target token's logprob; a top-k list
that does not contain the evaluated token is not a valid PPL measurement.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
import urllib.error
from pathlib import Path
from typing import Any

from scripts.bench_common import DEFAULT_BASE_URL, MODEL, json_request

REQUEST_FORMS = {
    "prompt_logprobs": lambda k: {"prompt_logprobs": k},
    "echo": lambda k: {"logprobs": k, "echo": True},
}


def target_logprobs(response: dict[str, Any], tokens: list[int] | None = None) -> list[float]:
    """Extract the selected prompt token logprob at every scored position."""
    choice = response["choices"][0]
    rows = choice.get("prompt_logprobs")
    response_tokens = choice.get("prompt_token_ids")
    if tokens is None:
        tokens = response_tokens
    if not isinstance(rows, list) or not isinstance(tokens, list) or len(rows) != len(tokens):
        raise ValueError("response lacks aligned prompt_logprobs and prompt tokens")
    values: list[float] = []
    # Position zero has no preceding token and is intentionally unscored.
    for index in range(1, len(tokens)):
        row = rows[index]
        if not isinstance(row, dict):
            raise ValueError(f"missing prompt logprob at position {index}")
        entry = row.get(tokens[index]) or row.get(str(tokens[index]))
        if not isinstance(entry, dict) or not isinstance(entry.get("logprob"), (int, float)):
            raise ValueError(f"target token {tokens[index]} absent from top-k logprobs at {index}")
        values.append(float(entry["logprob"]))
    return values


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or ("tokens" not in value and "text" not in value):
            raise ValueError(f"line {line_number} must contain tokens or text")
        records.append(value)
    if not records:
        raise ValueError("dataset contains no records")
    return records


def token_windows(
    base: str,
    records: list[dict[str, Any]],
    seq_len: int,
    stride: int,
    timeout: float,
) -> list[list[int]]:
    if seq_len < 2 or stride <= 0 or stride > seq_len:
        raise ValueError("seq_len must be >= 2 and stride must be in (0, seq_len]")
    windows: list[list[int]] = []
    for record in records:
        if "tokens" in record:
            tokens = record["tokens"]
            if not isinstance(tokens, list) or not all(isinstance(x, int) for x in tokens):
                raise ValueError("tokens must be a list of integers")
        else:
            result = json_request(base, "/tokenize", {"model": MODEL, "prompt": record["text"]}, timeout)
            tokens = result["tokens"]
        for start in range(0, max(1, len(tokens) - 1), stride):
            window = tokens[start : start + seq_len]
            if len(window) >= 2:
                windows.append(window)
            if start + seq_len >= len(tokens):
                break
    if not windows:
        raise ValueError("dataset has no window containing two tokens")
    return windows


def evaluate(
    base: str,
    windows: list[list[int]],
    top_k: int,
    timeout: float,
    request_form: str = "prompt_logprobs",
    *,
    evidence_dir: Path | None = None,
    cache_salt: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Score every window; ``request_form`` is recorded because the two forms
    ask the server for different fields.

    ``prompt_logprobs`` asks for the top-k list at each prompt position (the
    served model's default API). ``echo`` asks for ``logprobs=top_k`` with
    ``echo=true`` and still reads ``prompt_logprobs`` from the response.

    With ``evidence_dir``, retain every request/response (including failures),
    require observed cold cache and matching prompt lengths, and return FAIL
    instead of dropping earlier evidence when a window cannot be scored.
    """
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20 for the pinned API")
    if request_form not in REQUEST_FORMS:
        raise ValueError(f"request_form must be one of {sorted(REQUEST_FORMS)}")
    if evidence_dir is not None:
        evidence_dir = Path(evidence_dir).resolve()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        if any(evidence_dir.iterdir()):
            raise ValueError("evidence directory must be empty; use a fresh evidence path")
    started = time.perf_counter()
    total_nll = 0.0
    scored = 0
    rows = []
    for index, tokens in enumerate(windows):
        body = {
            "model": MODEL,
            "prompt": tokens,
            "max_tokens": 1,
            "temperature": 0,
            **REQUEST_FORMS[request_form](top_k),
        }
        if cache_salt is not None:
            body["cache_salt"] = cache_salt
        if seed is not None:
            body["seed"] = seed
        result = None
        row = {"window": index, "tokens": len(tokens), "scored_tokens": 0}
        error = None
        try:
            result = json_request(base, "/v1/completions", body, timeout)
            if evidence_dir is not None:
                usage = result.get("usage")
                details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
                cached = details.get("cached_tokens") if isinstance(details, dict) else None
                observed_input = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                row.update({
                    "input_tokens": len(tokens), "observed_input_tokens": observed_input,
                    "expected_scored_tokens": len(tokens) - 1,
                    "usage": usage, "cached_tokens": cached,
                })
                if observed_input != len(tokens):
                    raise ValueError(f"expected {len(tokens)} prompt tokens, observed {observed_input}")
                if type(cached) is not int or cached != 0:
                    raise ValueError(f"cold cache requires cached_tokens=0, observed {cached}")
                returned_tokens = result["choices"][0].get("prompt_token_ids")
                if returned_tokens is not None and returned_tokens != tokens:
                    raise ValueError("response prompt_token_ids differ from the requested window")
            values = target_logprobs(result, tokens)
            if evidence_dir is not None and not all(math.isfinite(value) for value in values):
                raise ValueError("target logprobs must be finite")
            window_nll = -sum(values)
            total_nll += window_nll
            scored += len(values)
            row["scored_tokens"] = len(values)
            if evidence_dir is not None:
                row.update({"status": "PASS", "total_nll": window_nll})
        except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError) as exc:
            if evidence_dir is None:
                raise
            if isinstance(exc, urllib.error.HTTPError):
                result = {"http_status": exc.code, "body": exc.read().decode(errors="replace")}
            error = f"{type(exc).__name__}: {exc}"
            row.update({"status": "FAIL", "error": error})
        if evidence_dir is not None:
            row.setdefault("input_tokens", len(tokens))
            row.setdefault("expected_scored_tokens", len(tokens) - 1)
            row.setdefault("observed_input_tokens", None)
            row.setdefault("usage", None)
            row.setdefault("cached_tokens", None)
            raw = gzip.compress(json.dumps({
                "request": body, "response": result, "error": error,
            }, ensure_ascii=False).encode(), mtime=0)
            path = evidence_dir / f"window-{index:04d}.json.gz"
            with path.open("xb") as handle:
                handle.write(raw)
            row.update({
                "raw_path": str(path), "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_bytes": len(raw),
            })
        rows.append(row)
    expected_scored = sum(len(tokens) - 1 for tokens in windows)
    complete = scored == expected_scored
    mean_nll = total_nll / scored if scored and complete else None
    failed = any(row.get("status") == "FAIL" for row in rows)
    report = {
        "status": "PASS" if scored and not failed else "FAIL",
        "model": MODEL,
        "windows": len(windows),
        "scored_tokens": scored,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll is not None and mean_nll < 700 else None,
        "elapsed_s": time.perf_counter() - started,
        "top_k": top_k,
        "request_form": request_form,
        "rows": rows,
        "total_nll": total_nll,
    }
    if evidence_dir is not None:
        report.update({
            "base_url": base, "evidence_dir": str(evidence_dir),
            "cache_salt": cache_salt, "seed": seed, "required_cache": "cold",
            "expected_scored_tokens": expected_scored,
            "successful_windows": sum(row["status"] == "PASS" for row in rows),
            "failed_windows": sum(row["status"] == "FAIL" for row in rows),
        })
    return report


def add_unit_perplexities(report: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Word and byte perplexity from the raw counts a dataset record carries.

    The counts describe the whole corpus, so they are only used when the run
    scored it end to end with a single record.
    """
    if len(records) != 1 or report["total_nll"] is None or report.get("status") == "FAIL":
        return
    for name in ("word_count", "byte_count"):
        count = records[0].get(name)
        if isinstance(count, int) and count > 0:
            report[name] = count
            report[name.replace("count", "perplexity")] = math.exp(report["total_nll"] / count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--request-form", choices=sorted(REQUEST_FORMS), default="prompt_logprobs")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--evidence-dir", type=Path, help="save gzipped window requests/responses and require cold cache")
    parser.add_argument("--cache-salt", help="prefix-cache salt; leaves evaluated token IDs unchanged")
    parser.add_argument("--seed", type=int, help="sampling seed for the one generated token")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a fresh evidence path")
    records = read_records(args.dataset)
    windows = token_windows(args.base_url, records, args.seq_len, args.stride, args.timeout)
    report = evaluate(
        args.base_url, windows, args.top_k, args.timeout, args.request_form,
        evidence_dir=args.evidence_dir, cache_salt=args.cache_salt, seed=args.seed,
    )
    add_unit_perplexities(report, records)
    report["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.evidence_dir is not None:
        report.update({"seq_len": args.seq_len, "stride": args.stride})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
