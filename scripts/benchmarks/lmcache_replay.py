"""Replay frozen synthetic multi-turn token IDs on an isolated vLLM service.

    python3 -m scripts.benchmarks.lmcache_replay prepare --output workload.json
    python3 -m scripts.benchmarks.lmcache_replay run --mode closed \
        --workload workload.json --base-url http://127.0.0.1:16688 \
        --launch-receipt launch.json --output replay.json

The default lengths approximate marginal statistics from Issue #1. Neither the
prompt tokens nor the inter-turn delays are from a business trace. The frozen
inputs grow by exact prefixes. Returned completions are measured and hashed,
and are independent of the frozen tokens used for the next turn.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import random
import socket
import statistics
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, Timer
from urllib.parse import urlsplit

from scripts import bench_common as bench
from scripts.benchmarks import lmcache_mp_probe as probe
from scripts.benchmarks.profile_measure import _split_key, parse_metrics

WORKLOAD_KIND = "lmcache_replay_workload"
SCHEMA_VERSION = 1
ISSUE_INPUT = (
    (0, 1024), (0.1, 8192), (0.25, 20480), (0.5, 33662), (0.75, 51000),
    (0.9, 82000), (0.95, 126757), (1, 185744),
)
ISSUE_OUTPUT = (
    (0, 64), (0.25, 160), (0.5, 320), (0.75, 1070), (0.9, 1800),
    (0.95, 2600), (1, 4096),
)
REQUIRED_COUNTERS = {"finished_requests", "prompt_tokens", "generation_tokens"}
RUN_ERRORS = (*probe.PROBE_ERRORS, subprocess.SubprocessError)
clock = time.perf_counter


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_int(value):
    return type(value) is int and value > 0


def nonnegative_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def stratified_lengths(anchors, count):
    result = []
    for index in range(count):
        quantile = index / (count - 1) if count > 1 else 0.5
        for (left, low), (right, high) in zip(anchors, anchors[1:], strict=False):
            if quantile <= right:
                result.append(round(low + (high - low) * (quantile - left) / (right - left)))
                break
    return result


def lengths_summary(lengths):
    return {
        "median": statistics.median(lengths),
        "mean": statistics.mean(lengths),
        "p95": bench.percentile(lengths, 0.95),
        "max": max(lengths),
    }


def prepare(args):
    require(positive_int(args.users) and positive_int(args.turns), "users and turns must be positive")
    require(0 <= args.seed < 2**63, "seed must be in [0, 2**63)")
    require(all(nonnegative_time(t) for t in (args.user_spacing, args.turn_spacing)), "arrival spacing must be finite and nonnegative")
    require(args.input_tokens is None or positive_int(args.input_tokens), "input length must be positive")
    require(args.output_tokens is None or positive_int(args.output_tokens), "output length must be positive")
    require(positive_int(args.input_step), "input step must be positive")
    rng = random.Random(args.seed)
    count = args.users * args.turns
    inputs = stratified_lengths(ISSUE_INPUT, count)
    outputs = ([args.output_tokens] * count if args.output_tokens is not None
               else stratified_lengths(ISSUE_OUTPUT, count))
    rng.shuffle(inputs)
    rng.shuffle(outputs)
    token_pool = rng.sample(range(129, 4096), 256)
    users = []
    for user in range(args.users):
        start = user * args.turns
        user_inputs = sorted(inputs[start:start + args.turns])
        turns = []
        for turn in range(args.turns):
            target = (
                args.input_tokens + turn * args.input_step if args.input_tokens is not None
                else max(user_inputs[turn], turns[-1]["input_tokens"] + 1) if turns
                else user_inputs[turn]
            )
            turns.append({
                "turn": turn + 1,
                "input_tokens": target,
                "output_tokens": outputs[start + turn],
                "planned_at_s": round(user * args.user_spacing + turn * args.turn_spacing, 9),
            })
        users.append({"user_id": f"user{user:06}", "pool_offset": user % len(token_pool), "turns": turns})
    all_turns = [turn for user in users for turn in user["turns"]]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": WORKLOAD_KIND,
        "workload_kind": "synthetic",
        "model": bench.MODEL,
        "seed": args.seed,
        "users": args.users,
        "turns": args.turns,
        "token_pool": token_pool,
        "chains": users,
        "observed_lengths": {
            key: lengths_summary([turn[key] for turn in all_turns])
            for key in ("input_tokens", "output_tokens")
        },
        "notes": {
            "provenance": "Synthetic token IDs and arrivals. Marginal length targets from Issue #1, not a business trace.",
            "prompt": "Rotate the frozen token pool by pool_offset and repeat to input_tokens. Completions are not fed back.",
            "fixed": "planned_at_s is measured from run start. Each chain waits for its previous request to finish.",
            "closed": "Each user starts at its first planned_at_s. Later turns have zero think time. A new session repeats the frozen turns under a new salt.",
        },
    }
    write_json(args.output, manifest)
    print(json.dumps({key: manifest[key] for key in ("workload_kind", "users", "turns", "seed", "observed_lengths")}, indent=2))
    return 0


def load_workload(path):
    payload = path.read_bytes()
    manifest = json.loads(payload)
    require(
        manifest["schema_version"] == SCHEMA_VERSION and manifest["kind"] == WORKLOAD_KIND
        and manifest["workload_kind"] == "synthetic" and manifest["model"] == bench.MODEL,
        "expected a v1 synthetic replay workload for the served model",
    )
    require(type(manifest["seed"]) is int and 0 <= manifest["seed"] < 2**63, "invalid workload seed")
    pool = manifest["token_pool"]
    require(isinstance(pool, list) and pool and all(type(token) is int and 129 <= token < 4096 for token in pool), "invalid frozen token pool")
    require(positive_int(manifest["users"]) and positive_int(manifest["turns"]), "invalid workload size")
    require(len(manifest["chains"]) == manifest["users"], "workload user count mismatch")
    for index, chain in enumerate(manifest["chains"]):
        require(chain["user_id"] == f"user{index:06}", "invalid or duplicate user identifier")
        require(type(chain["pool_offset"]) is int and 0 <= chain["pool_offset"] < len(pool), "invalid token pool offset")
        require(len(chain["turns"]) == manifest["turns"], "workload turn count mismatch")
        last_length, last_arrival = 0, 0.0
        for turn, entry in enumerate(chain["turns"], 1):
            length, arrival = entry["input_tokens"], entry["planned_at_s"]
            require(
                entry["turn"] == turn and positive_int(length) and length > last_length
                and positive_int(entry["output_tokens"])
                and nonnegative_time(arrival) and arrival >= last_arrival,
                "each chain must have consecutive turns, increasing input lengths and ordered arrivals",
            )
            last_length, last_arrival = length, arrival
    return manifest, hashlib.sha256(payload).hexdigest()


def option(command, flag):
    require(command.count(flag) == 1 and command[-1] != flag, f"launch receipt lacks a unique {flag}")
    return command[command.index(flag) + 1]


def verify_endpoint(base_url, receipt):
    """Bind loopback to a running launch.py host network/IPC instance."""
    url = urlsplit(base_url)
    launch = receipt["command"]
    image, container = receipt["image_id"], receipt["container_id"]
    require(
        launch[:2] == ["docker", "run"] and isinstance(image, str) and image.startswith("sha256:")
        and isinstance(container, str) and len(container) == 64
        and all(c in "0123456789abcdef" for c in container),
        "expected a launch.py receipt with the actual image and full container ID",
    )
    require(launch.count(image) == 1, "launch receipt lacks a unique image ID in argv")
    image_index = launch.index(image)
    docker, command = launch[:image_index], launch[image_index + 1:]
    require(
        receipt.get("host_network_ipc") is True and "--network=host" in docker and "--ipc=host" in docker
        and command[:1] == ["serve"], "expected the host network/IPC vLLM launch receipt",
    )
    require(
        option(command, "--host") == "127.0.0.1" and url.hostname in ("127.0.0.1", "localhost")
        and int(option(command, "--port")) == url.port,
        "loopback URL does not match the launch receipt host and port",
    )
    require(option(command, "--served-model-name") == bench.MODEL, "unexpected served model in launch receipt")
    observed, = json.loads(subprocess.check_output(["docker", "inspect", container], timeout=probe.GET_TIMEOUT))
    state, host = observed["State"], observed["HostConfig"]
    require(
        observed["Id"] == container and observed["Image"] == image and observed["Config"]["Cmd"] == command
        and state["Running"] is True and state["StartedAt"]
        and host["NetworkMode"] == "host" and host["IpcMode"] == "host",
        "running container ID, image, argv, state, network or IPC differs from the launch receipt",
    )
    return {
        "container_id": observed["Id"],
        "image_id": observed["Image"],
        "started_at": state["StartedAt"],
        "command": observed["Config"]["Cmd"],
        "network_mode": host["NetworkMode"],
        "ipc_mode": host["IpcMode"],
        "port": url.port,
    }


def prepare_requests(workload, record_progress=False):
    """Build immutable HTTP request templates and canonical hashes outside measurement."""
    pool = workload["token_pool"]
    plans = []
    for chain in workload["chains"]:
        offset = chain["pool_offset"]
        rotated = pool[offset:] + pool[:offset]
        turns = []
        for turn in chain["turns"]:
            size = turn["input_tokens"]
            repeats, remainder = divmod(size, len(rotated))
            prompt = rotated * repeats + rotated[:remainder]
            body = {
                **bench.completion_body(prompt, turn["output_tokens"], workload["seed"]),
                "add_special_tokens": False,
            }
            if record_progress:
                body["return_token_ids"] = True
            encoded = json.dumps(body, separators=(",", ":")).encode()
            turns.append({**turn, "body": encoded, "request_sha256": hashlib.sha256(encoded).hexdigest()})
        plans.append({"user_id": chain["user_id"], "turns": turns})
    return plans


def partial_stream_state():
    return {
        "first_output_s": None,
        "last_output_s": None,
        "output_events": 0,
        "output_token_ids_seen": None,
    }


def planned_row(namespace, user, turn, cycle, planned_at, record_progress=False):
    salt = f"replay-{namespace}-{user}-{cycle}"
    row = {
        "user_id": user,
        "chain_id": f"{user}/session-{cycle}",
        "cycle": cycle,
        "turn": turn["turn"],
        "request_sha256": turn["request_sha256"],
        "request_id": f"{salt}-{turn['turn']}",
        "cache_salt": salt,
        "planned_at_s": planned_at,
        "actual_started_s": None,
        "completed_at_s": None,
        "client_queue_s": None,
        "posted": False,
        "status": "NOT_SENT",
        "error": "preflight did not complete",
        "expected_input_tokens": turn["input_tokens"],
        "expected_output_tokens": turn["output_tokens"],
        "input_tokens": None,
        "output_tokens": None,
        "cached_tokens": None,
        "ttft_s": None,
        "e2e_s": None,
        "output_sha256": None,
    }
    if record_progress:
        row["partial_stream"] = partial_stream_state()
    return row


def abort_socket(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # The completion could have closed the socket already.


@contextmanager
def completion_response(base_url, body, headers, deadline):
    """Send one direct POST with a whole-request deadline, including SSE reads."""
    url = urlsplit(base_url)
    remaining = deadline - clock()
    if remaining <= 0:
        raise TimeoutError("completion exceeded its request deadline")
    # verify_endpoint enforces a 127.0.0.1 binding, including localhost aliases.
    connection = http.client.HTTPConnection("127.0.0.1", url.port, timeout=remaining)
    watchdog = None
    try:
        connection.connect()
        # A socket idle timeout alone allows a drip-fed HTTP header or SSE
        # line to hold a worker indefinitely. Shutdown also wakes buffered reads.
        watchdog = Timer(max(0.0, deadline - clock()), abort_socket, (connection.sock,))
        watchdog.daemon = True
        watchdog.start()
        connection.request(
            "POST", probe.COMPLETION_PATH, body,
            {**headers, "Content-Type": "application/json", "Connection": "close"},
        )
        with connection.getresponse() as response:
            if response.status != 200:
                detail = response.read(2048).decode("utf-8", errors="replace")
                raise RuntimeError(f"{probe.COMPLETION_PATH} HTTP {response.status}: {detail}")
            yield response
    finally:
        if watchdog is not None:
            watchdog.cancel()
        connection.close()
        if clock() >= deadline:
            raise TimeoutError("completion exceeded its request deadline")


def execute_requests(args, plans, rows, report, headers):
    started = clock()
    cutoff = started + args.duration
    attempts_lock = Lock()

    def complete(row, turn):
        suffix = json.dumps({"cache_salt": row["cache_salt"], "request_id": row["request_id"]}, separators=(",", ":")).encode()
        body = turn["body"][:-1] + b"," + suffix[1:]
        with attempts_lock:
            sent = clock()
            if args.mode == "closed" and sent >= cutoff:
                return False
            report["completion_post_attempts"] += 1
        offset = sent - started
        row.update(posted=True, actual_started_s=offset, client_queue_s=max(0.0, offset - row["planned_at_s"]))
        try:
            with completion_response(args.base_url, body, headers, sent + args.timeout) as response:
                row.update(
                    bench.stream_metrics(
                        response,
                        sent,
                        chat=False,
                        progress=row.get("partial_stream"),
                    )
                )
            require(row["input_tokens"] == row["expected_input_tokens"], f"expected {row['expected_input_tokens']} input tokens, got {row['input_tokens']}")
            require(row["output_tokens"] == row["expected_output_tokens"], f"expected {row['expected_output_tokens']} output tokens, got {row['output_tokens']}")
            require(row["finish_reason"] == "length", f"unexpected completion finish reason {row['finish_reason']!r}")
            cached = row["cached_tokens"]
            require(cached is None or type(cached) is int and 0 <= cached <= row["input_tokens"], "invalid cached token usage")
            require(row["turn"] != 1 or cached in (None, 0), "first turn of a new salted session reported cached tokens")
            row.update(status="PASS", error=None)
        except probe.PROBE_ERRORS as error:
            row.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        finally:
            completed = clock()
            row.update(completed_at_s=completed - started, e2e_s=completed - sent)
        return True

    def wait_until(offset):
        remaining = started + offset - clock()
        if remaining > 0:
            time.sleep(remaining)

    def fixed(plan, offered):
        for turn, row in zip(plan["turns"], offered, strict=True):
            wait_until(row["planned_at_s"])
            complete(row, turn)
        return offered

    def closed(plan):
        user = plan["user_id"]
        turns = plan["turns"]
        planned_at = float(turns[0]["planned_at_s"])
        wait_until(planned_at)
        completed = []
        cycle = 0
        while clock() < cutoff:
            for turn in turns:
                row = planned_row(
                    report["namespace"],
                    user,
                    turn,
                    cycle,
                    planned_at,
                    args.record_progress,
                )
                if not complete(row, turn):
                    return completed
                completed.append(row)
                if row["status"] != "PASS":
                    return completed
                planned_at = row["completed_at_s"]
            cycle += 1
        return completed

    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        if args.mode == "fixed":
            futures = [pool.submit(fixed, plan, offered) for plan, offered in zip(plans, rows, strict=True)]
        else:
            futures = [pool.submit(closed, plan) for plan in plans]
        report["rows"] = [row for future in futures for row in future.result()]
    report["elapsed_s"] = clock() - started


def finite_metrics(base_url, headers):
    with probe.open_response(base_url, "/metrics", probe.GET_TIMEOUT, headers=headers) as response:
        samples = parse_metrics(response.read().decode("utf-8"))
    return {key: value if math.isfinite(value) else None for key, value in samples.items()}


def engine_counters(metrics):
    by_engine = {rank: {} for rank in probe.RANKS}
    for key, value in metrics.items():
        name, labels = _split_key(key)
        counter = probe.COUNTERS.get(name)
        if counter is None:
            continue
        if counter == "external_prefill_tokens" and labels.get("source") != "external_kv_transfer":
            continue
        rank = labels.get("engine")
        require(rank in by_engine and labels.get("model_name") == bench.MODEL, f"unexpected vLLM engine/model labels: {key}")
        require(value is not None, f"non-finite vLLM counter: {key}")
        engine = by_engine[rank]
        require(counter == "finished_requests" or counter not in engine, f"ambiguous vLLM counter: {key}")
        engine[counter] = engine.get(counter, 0) + probe.counter_value(key, value)
    for rank, counters in by_engine.items():
        require(REQUIRED_COUNTERS <= counters.keys(), f"missing DP{rank} counters: {sorted(REQUIRED_COUNTERS - counters.keys())}")
    return by_engine


def counter_deltas(before, after):
    result = {}
    for rank in probe.RANKS:
        entry = result[rank] = {}
        for key in probe.COUNTERS.values():
            first, last = before[rank].get(key), after[rank].get(key)
            entry[key] = None if first is None or last is None else last - first
            require(entry[key] is None or entry[key] >= 0, f"vLLM counter reset on DP{rank}: {key}")
    return result


def observe_after(args, headers, before, report, good):
    expected = {
        "finished_requests": len(good),
        "prompt_tokens": sum(row["input_tokens"] for row in good),
        "generation_tokens": sum(row["output_tokens"] for row in good),
    }

    def sample():
        report["metrics_after"] = finite_metrics(args.base_url, headers)
        delta = counter_deltas(before, engine_counters(report["metrics_after"]))
        report["metrics_delta_by_engine"] = delta
        return {key: sum(delta[rank][key] for rank in probe.RANKS) for key in expected}

    total = probe.poll(sample, lambda values: all(values[k] >= n for k, n in expected.items()), probe.GET_TIMEOUT)
    report["both_dp_active"] = all(report["metrics_delta_by_engine"][rank]["finished_requests"] > 0 for rank in probe.RANKS)
    if good and len(good) == len(report["rows"]):
        require(total == expected, f"DP counters do not match replay usage: {total} vs {expected}")
        require(args.mode != "closed" or report["users"] == 1 or report["both_dp_active"], "only one DP engine completed closed-loop requests")


def run(args):
    workload, workload_digest = load_workload(args.workload)
    receipt = json.loads(args.launch_receipt.read_bytes())
    namespace = uuid.uuid4().hex
    num_users, num_turns = workload["users"], workload["turns"]
    last_arrival = max(
        float(chain["turns"][-1 if args.mode == "fixed" else 0]["planned_at_s"])
        for chain in workload["chains"]
    )
    require(last_arrival < args.duration, f"duration must exceed the last offered arrival ({last_arrival:g}s)")
    max_context = max(turn["input_tokens"] + turn["output_tokens"] for chain in workload["chains"] for turn in chain["turns"])
    require(positive_int(receipt["context"]) and max_context <= receipt["context"], f"input plus output requires at least {max_context} context tokens in the receipt")
    plans = prepare_requests(workload, args.record_progress)
    fixed = args.mode == "fixed"
    rows = [
        [
            planned_row(
                namespace,
                plan["user_id"],
                turn,
                0,
                float(turn["planned_at_s"]),
                args.record_progress,
            )
            for turn in plan["turns"]
        ]
        for plan in plans
    ] if fixed else []
    if fixed:
        schedule = [[r["chain_id"], r["turn"], r["planned_at_s"], r["request_sha256"]] for chain in rows for r in chain]
    else:
        schedule = [
            [plan["user_id"], float(plan["turns"][0]["planned_at_s"]), [turn["request_sha256"] for turn in plan["turns"]]]
            for plan in plans
        ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "lmcache_replay",
        "status": "FAIL",
        "mode": args.mode,
        "workload_kind": workload["workload_kind"],
        "workload_sha256": workload_digest,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "offered_schedule_sha256": bench.workload_digest([args.mode, args.duration, schedule]),
        "users": num_users,
        "turns": num_turns,
        "model": bench.MODEL,
        "run_key": args.run_key,
        "namespace": namespace,
        "base_url": args.base_url,
        "launch_receipt": receipt,
        "endpoint": None,
        "endpoint_after": None,
        "duration_s": args.duration,
        "elapsed_s": None,
        "request_timeout_s": args.timeout,
        "completion_retry_limit": 0,
        "completion_offered": num_users * num_turns if fixed else None,
        "completion_post_attempts": 0,
        "rows": [row for chain in rows for row in chain],
        "metrics_before": None,
        "metrics_after": None,
        "metrics_delta_by_engine": None,
        "both_dp_active": None,
        "errors": [],
    }
    if args.record_progress:
        report["record_progress"] = True
    api_key = os.environ.get("BENCH_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    step = "endpoint"
    try:
        report["endpoint"] = verify_endpoint(args.base_url, receipt)
        require(int(option(report["endpoint"]["command"], "--max-model-len")) == receipt["context"], "launch receipt context differs from the running argv")
        step = "metrics_before"
        report["metrics_before"] = finite_metrics(args.base_url, headers)
        before = engine_counters(report["metrics_before"])
        step = "replay"
        execute_requests(args, plans, rows, report, headers)
        good = [row for row in report["rows"] if row["status"] == "PASS"]
        step = "metrics_after"
        observe_after(args, headers, before, report, good)
    except RUN_ERRORS as error:
        report["errors"].append(f"{step}: {type(error).__name__}: {error}")
    if report["endpoint"] is not None:
        try:
            report["endpoint_after"] = verify_endpoint(args.base_url, receipt)
            require(report["endpoint_after"] == report["endpoint"], "container identity changed during replay")
        except RUN_ERRORS as error:
            report["errors"].append(f"endpoint_after: {type(error).__name__}: {error}")
    passed = [row for row in report["rows"] if row["status"] == "PASS"]
    report["success"] = success = len(passed)
    report["success_rate"] = success / len(report["rows"]) if report["rows"] else 0.0
    report["success_in_window"] = sum(row["completed_at_s"] <= args.duration for row in passed)
    report["success_in_drain"] = success - report["success_in_window"]
    report["drain_s"] = None if report["elapsed_s"] is None else max(0.0, report["elapsed_s"] - args.duration)
    if success and success == len(report["rows"]) and not report["errors"]:
        report["status"] = "PASS"
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "mode", "workload_sha256", "duration_s", "elapsed_s", "completion_post_attempts", "success", "errors")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare", help="freeze a synthetic token and arrival workload offline")
    build.add_argument("--users", type=int, default=70)
    build.add_argument("--turns", type=int, default=3)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--user-spacing", type=float, default=0.1, help="seconds between user starts (default 0.1)")
    build.add_argument("--turn-spacing", type=float, default=20, help="fixed arrival seconds between turns (default 20)")
    build.add_argument("--input-tokens", type=int, help="override the synthetic distribution with this first-turn length")
    build.add_argument("--input-step", type=int, default=16, help="input growth per turn with --input-tokens (default 16)")
    build.add_argument("--output-tokens", type=int, help="override the synthetic distribution with a fixed output length")
    build.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("run", help="replay a frozen workload against its verified Docker launch")
    replay.add_argument("--mode", choices=("fixed", "closed"), required=True)
    replay.add_argument("--workload", type=Path, required=True)
    replay.add_argument("--base-url", required=True)
    replay.add_argument("--launch-receipt", type=Path, required=True)
    replay.add_argument("--duration", type=float, default=600.0, help="offering window in seconds (default 600)")
    replay.add_argument("--timeout", type=float, default=600.0, help="seconds per completion (default 600)")
    replay.add_argument(
        "--record-progress",
        action="store_true",
        help="request completion token IDs and retain partial SSE progress",
    )
    replay.add_argument("--run-key", help="optional repeat identifier shared outside the unique cache namespace")
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), f"output exists; choose a fresh evidence path: {args.output}")
        if args.command == "prepare":
            return prepare(args)
        args.base_url = probe.loopback_url(args.base_url)
        bench.require_performance_endpoint(args.base_url)
        require(all(nonnegative_time(t) and t > 0 for t in (args.duration, args.timeout)), "duration and timeout must be finite and positive")
        return run(args)
    except RUN_ERRORS as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
