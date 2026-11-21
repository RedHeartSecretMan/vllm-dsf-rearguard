r"""Probe LMCache MP sharing on an idle TP4/DP2 test service.

    python3 -m scripts.benchmarks.lmcache_mp_probe \
        --base-url http://127.0.0.1:16688 --lmcache-url http://127.0.0.1:19180 \
        --output probe.json

Use --prompt-tokens 20992 for the prompt length in Issue #1. The probe sends
one cold DP0 completion, waits for its L1 writes, then sends one DP1 completion.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from scripts import bench_common as bench
from scripts.benchmarks.profile_measure import _split_key, parse_metrics

# vllm-backport: generate/base/serving.py parses the header. The completions
# handler passes it to the engine client, which indexes engines in rank order.
VLLM_SOURCE_COMMIT = "a350766628514d597670d2a5217777fdee2ba7f4"
RANK_HEADER = "X-data-parallel-rank"
RANKS = ("0", "1")
GPU_COUNT = 8
COMPLETION_PATH = "/v1/completions"
COUNTERS = {
    "vllm:request_success_total": "finished_requests",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:prefix_cache_hits_total": "local_hits",
    "vllm:external_prefix_cache_queries_total": "external_queries",
    "vllm:external_prefix_cache_hits_total": "external_hits",
    "vllm:prompt_tokens_by_source_total": "external_prefill_tokens",
}
CACHE_COUNTERS = {
    "lmcache_mp_l1_write_chunks_total": "l1_written_keys",
    "lmcache_mp_num_chunks_loaded_total": "loaded_keys",
}
GET_TIMEOUT = 10.0
OUTPUT_TOKENS = 8
PROBE_ERRORS = (
    OSError,
    http.client.HTTPException,
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
)


def loopback_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
        or not parsed.port
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or "?" in value
        or "#" in value
    ):
        raise ValueError("expected a loopback HTTP origin with an explicit port")
    return value.rstrip("/")


@contextmanager
def open_response(base, path, timeout, *, body=None, headers=None):
    """Open one direct HTTP connection without proxy, redirect, or retry."""
    parsed = urlsplit(loopback_url(base))
    host = "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname
    conn = http.client.HTTPConnection(host, parsed.port, timeout=timeout)
    try:
        headers = dict(headers or {})
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, separators=(",", ":")).encode()
        conn.request("GET" if data is None else "POST", path, data, headers)
        with conn.getresponse() as response:
            if response.status != 200:
                detail = response.read(2048).decode("utf-8", errors="replace")
                raise RuntimeError(f"{path} HTTP {response.status}: {detail}")
            yield response
    finally:
        conn.close()


def get_metrics(base, headers=None):
    with open_response(base, "/metrics", GET_TIMEOUT, headers=headers) as response:
        return parse_metrics(response.read().decode("utf-8"))


def counter_value(sample, value):
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise ValueError(f"invalid counter {sample}: {value}")
    return int(value)


def vllm_counters(base, headers):
    totals = {rank: {} for rank in RANKS}
    for key, value in get_metrics(base, headers).items():
        name, labels = _split_key(key)
        counter = COUNTERS.get(name)
        if counter is None:
            continue
        if labels.get("model_name") != bench.MODEL or labels.get("engine") not in RANKS:
            raise ValueError(f"expected {bench.MODEL} with DP engines 0 and 1: {key}")
        if (
            counter == "external_prefill_tokens"
            and labels.get("source") != "external_kv_transfer"
        ):
            continue
        engine = totals[labels["engine"]]
        if counter in engine and counter != "finished_requests":
            raise ValueError(f"ambiguous engine counter: {key}")
        # request_success_total is partitioned by finished_reason.
        engine[counter] = engine.get(counter, 0) + counter_value(key, value)
    for rank, engine in totals.items():
        if missing := set(COUNTERS.values()) - engine.keys():
            raise ValueError(f"engine {rank} missing vLLM counters: {sorted(missing)}")
    return totals


def lmcache_counters(base, salt):
    # OTel creates a salted counter series on its first observation.
    totals = dict.fromkeys(CACHE_COUNTERS.values(), 0)
    for key, value in get_metrics(base).items():
        name, labels = _split_key(key)
        if name in CACHE_COUNTERS and labels.get("cache_salt") == salt:
            totals[CACHE_COUNTERS[name]] += counter_value(key, value)
    return totals


def lmcache_status(base):
    with open_response(base, "/status", GET_TIMEOUT) as response:
        status = json.load(response)
    if not isinstance(status, dict):
        raise ValueError("LMCache /status must return a JSON object")
    return status


def registered_gpus(status):
    ids = status.get("registered_gpu_ids")
    if not isinstance(ids, list) or any(type(gpu) is not int or gpu < 0 for gpu in ids):
        raise ValueError(
            "LMCache /status lacks registered_gpu_ids as a list of GPU instance IDs"
        )
    if len(ids) != GPU_COUNT or len(set(ids)) != GPU_COUNT:
        raise ValueError(f"expected {GPU_COUNT} unique registered GPU IDs, got {ids}")
    return sorted(ids)


def l1_object_groups_per_chunk(status, gpu_ids):
    group_counts = set()
    for gpu_id in gpu_ids:
        context = status["cache_context_meta"][str(gpu_id)]
        if context["world_size"] != 1:
            raise ValueError("expected replicated KV object keys for the DSF probe")
        groups = {
            group["object_group_idx"]
            for group in context["kv_cache_layout"]["kernel_groups"]
        }
        group_counts.add(len(groups))
    if len(group_counts) != 1 or 0 in group_counts:
        raise ValueError("registered GPUs disagree on the number of L1 object groups")
    return group_counts.pop()


def build_prompt(base, headers, count):
    # Repeat one tokenized natural-text prefix so the requested token count is
    # exact and both completions send byte-for-byte identical prompt token IDs.
    with open_response(
        base,
        "/tokenize",
        GET_TIMEOUT,
        body={
            "model": bench.MODEL,
            "prompt": bench.NATURE_CORPUS.strip(),
            "add_special_tokens": False,
        },
        headers=headers,
    ) as response:
        tokens = json.load(response)["tokens"]
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("/tokenize did not return a nonempty token ID list")
    return (tokens * ((count + len(tokens) - 1) // len(tokens)))[:count]


def complete_once(args, report, headers, body, phase, rank):
    request_id = f"lmcache-mp-{report['cache_salt']}-{phase}"
    row = {"phase": phase, "requested_dp_rank": int(rank), "request_id": request_id}
    report["requests"].append(row)
    report["completion_post_attempts"] += 1
    started = time.perf_counter()
    try:
        with open_response(
            args.base_url,
            COMPLETION_PATH,
            args.timeout,
            body={**body, "request_id": request_id},
            headers={**headers, RANK_HEADER: rank},
        ) as response:
            row.update(
                bench.stream_metrics(
                    response,
                    started,
                    chat=False,
                    expected_input=args.prompt_tokens,
                    expected_output=OUTPUT_TOKENS,
                )
            )
        if row["finish_reason"] != "length":
            raise ValueError(
                f"unexpected completion finish reason: {row['finish_reason']}"
            )
    except PROBE_ERRORS as error:
        row.update(status="FAIL", error=str(error))
        raise
    row["status"] = "PASS"
    return row


def poll(sample, ready, timeout):
    """Collect GET observations while asynchronous metrics catch up."""
    deadline = time.monotonic() + timeout
    while True:
        observed = sample()
        if ready(observed):
            return observed
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return observed
        time.sleep(min(0.2, remaining))


def phase_deltas(before, after):
    result = {}
    for rank, counters in before.items():
        result[rank] = {
            key: after[rank][key] - value for key, value in counters.items()
        }
        if min(result[rank].values()) < 0:
            raise ValueError(f"vLLM counters reset on engine {rank}: {result[rank]}")
    return result


def require_phase_evidence(deltas, rank, request, min_external):
    other = next(candidate for candidate in RANKS if candidate != rank)
    active = deltas[rank]
    if any(deltas[other].values()):
        raise ValueError(
            f"unexpected metric activity on engine {other}: {deltas[other]}"
        )
    prompt_tokens = request["input_tokens"]
    expected = {
        "finished_requests": 1,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": request["output_tokens"],
        "local_hits": 0,
        "external_queries": prompt_tokens,
    }
    if any(active[key] != value for key, value in expected.items()):
        raise ValueError(
            f"engine {rank} counters disagree with this completion: {active}"
        )
    hits, used = active["external_hits"], active["external_prefill_tokens"]
    if min_external:
        if not min_external <= used <= hits < prompt_tokens:
            raise ValueError(
                f"engine {rank} lacks compatible external hit and prefill counts: {active}"
            )
    elif hits or used:
        raise ValueError(f"cold engine {rank} had an external cache hit: {active}")


def run_probe(args):
    # The namespace is fresh per invocation, shared by exactly these two POSTs,
    # and safe to include in the evidence so the LMCache metrics can be joined.
    salt = f"probe-{uuid.uuid4().hex}"
    report = {
        "status": "FAIL",
        "stage": "lmcache_registration",
        "vllm_reference_commit": VLLM_SOURCE_COMMIT,
        "model": bench.MODEL,
        "base_url": args.base_url,
        "lmcache_url": args.lmcache_url,
        "rank_header": RANK_HEADER,
        "cache_salt": salt,
        "prompt_tokens": args.prompt_tokens,
        "completion_post_attempts": 0,
        "completion_retry_limit": 0,
        "requests": [],
        "vllm_metrics": {"names": COUNTERS},
        "lmcache_metrics": {"names": CACHE_COUNTERS},
    }
    vm, lm = report["vllm_metrics"], report["lmcache_metrics"]
    api_key = os.environ.get("BENCH_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def vllm_sample():
        return vllm_counters(args.base_url, headers)

    def lmcache_sample():
        return lmcache_counters(args.lmcache_url, salt)

    def measure_completion(phase, rank, before, min_external):
        report["stage"] = f"{phase}_completion"
        row = complete_once(args, report, headers, body, phase, rank)
        report["stage"] = f"{phase}_vllm_metrics"
        after = vm[f"after_{phase}"] = poll(
            vllm_sample,
            lambda current: any(
                current[r]["finished_requests"] != before[r]["finished_requests"]
                for r in RANKS
            ),
            GET_TIMEOUT,
        )
        deltas = vm[f"{phase}_delta"] = phase_deltas(before, after)
        require_phase_evidence(deltas, rank, row, min_external)
        row["observed_engine"] = int(rank)
        return row

    try:
        status = lmcache_status(args.lmcache_url)
        report["registration"] = {
            "source": "GET /status",
            "before": status.get("registered_gpu_ids"),
        }
        initial_gpus = registered_gpus(status)
        chunk = status.get("chunk_size")
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("LMCache /status lacks a positive chunk_size")
        # A final partial chunk cannot be reused. A fully chunk-aligned prompt
        # can also need its last token recomputed, depending on the KV layout.
        reusable_chunks = (args.prompt_tokens - 1) // chunk
        groups_per_chunk = l1_object_groups_per_chunk(status, initial_gpus)
        expected_l1_writes = args.prompt_tokens // chunk * groups_per_chunk
        report.update(
            lmcache_chunk_tokens=chunk,
            full_chunks_before_last_token=reusable_chunks,
            l1_object_groups_per_chunk=groups_per_chunk,
            expected_cold_l1_writes=expected_l1_writes,
            minimum_warm_external_tokens=chunk,
        )
        if not reusable_chunks:
            raise ValueError(
                f"--prompt-tokens must exceed the LMCache chunk size ({chunk})"
            )

        report["stage"] = "preflight"
        prompt = build_prompt(args.base_url, headers, args.prompt_tokens)
        report["prompt_sha256"] = bench.workload_digest(prompt)
        body = {
            **bench.completion_body(prompt, OUTPUT_TOKENS, seed=0),
            "add_special_tokens": False,
            "cache_salt": salt,
        }
        lm["before_cold"] = lmcache_sample()
        if any(lm["before_cold"].values()):
            raise ValueError("LMCache already has counters for this probe's salt")
        vm["before_cold"] = vllm_sample()
        cold = measure_completion("cold", RANKS[0], vm["before_cold"], 0)

        report["stage"] = "cold_l1_store"
        started = time.monotonic()
        # The metric counts objects. Wait for every group of every complete
        # prompt chunk so an early prefill batch cannot satisfy the gate.
        lm["before_warm"] = poll(
            lmcache_sample,
            lambda counters: counters["l1_written_keys"] >= expected_l1_writes,
            args.store_timeout,
        )
        report["store_wait_s"] = time.monotonic() - started
        completed = lm["before_warm"]["l1_written_keys"]
        if completed < expected_l1_writes:
            raise ValueError(
                f"observed {completed}/{expected_l1_writes} completed L1 writes for the cold salt"
            )
        if lm["before_warm"]["loaded_keys"]:
            raise ValueError(
                "LMCache already served loads for this salt before the warm request"
            )

        report["stage"] = "warm_baseline"
        vm["before_warm"] = vllm_sample()
        if vm["before_warm"] != vm["after_cold"]:
            raise ValueError(
                "vLLM counters changed between phases. Use an idle test endpoint."
            )
        warm = measure_completion("warm", RANKS[1], vm["before_warm"], chunk)

        report["stage"] = "warm_lmcache_load"
        lm["after_warm"] = poll(
            lmcache_sample, lambda counters: counters["loaded_keys"] > 0, GET_TIMEOUT
        )
        if not lm["after_warm"]["loaded_keys"]:
            raise ValueError(
                "did not observe a completed LMCache load for the warm salt"
            )
        if lm["after_warm"]["l1_written_keys"] < lm["before_warm"]["l1_written_keys"]:
            raise ValueError("LMCache write counter reset during the warm phase")

        report["stage"] = "final_checks"
        status = lmcache_status(args.lmcache_url)
        report["registration"]["after"] = status.get("registered_gpu_ids")
        if registered_gpus(status) != initial_gpus:
            raise ValueError("LMCache registered GPU IDs changed during the probe")
        report["output_matches"] = cold["output_sha256"] == warm["output_sha256"]
        if not report["output_matches"]:
            raise ValueError("cold and warm greedy completion text differs")
        report.update(status="PASS", stage="complete")
    except PROBE_ERRORS as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-url", required=True, help="loopback vLLM origin with explicit port"
    )
    parser.add_argument(
        "--lmcache-url",
        required=True,
        help="loopback LMCache MP HTTP origin with /status and /metrics",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=2049,
        help="exact input length (default 2049, Issue #1 20992)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="HTTP timeout in seconds per completion (default 300)",
    )
    parser.add_argument(
        "--store-timeout",
        type=float,
        default=30,
        help="L1 store observation window in seconds (default 30)",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="new JSON evidence path"
    )
    args = parser.parse_args(argv)
    try:
        args.base_url = loopback_url(args.base_url)
        bench.require_performance_endpoint(args.base_url)
        args.lmcache_url = loopback_url(args.lmcache_url)
    except ValueError as error:
        parser.error(str(error))
    if args.prompt_tokens <= 0 or not all(
        math.isfinite(timeout) and timeout > 0
        for timeout in (args.timeout, args.store_timeout)
    ):
        parser.error("prompt length and timeouts must be finite and positive")
    if args.output.exists():
        parser.error(f"output exists, choose a fresh evidence path: {args.output}")
    return bench.write_report(args.output, run_probe(args))


if __name__ == "__main__":
    raise SystemExit(main())
