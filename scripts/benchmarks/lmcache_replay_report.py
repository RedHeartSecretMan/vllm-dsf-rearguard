"""Compare two local LMCache replay result files from the same replay mode.

    python3 -m scripts.benchmarks.lmcache_replay_report OFF.json ON.json --output report.json

Exit 0 for a healthy comparison, 1 for a failed acceptance with observations
retained, and 2 for incompatible or invalid input. Omitting --output writes JSON
on stdout. The report reads files only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from scripts.bench_common import MODEL, SERVICE_PORT, percentile, workload_digest
from scripts.benchmarks.lmcache_mp_probe import loopback_url
from scripts.benchmarks.lmcache_replay import counter_deltas, engine_counters
from scripts.lmcache_mp import LMCacheMP

TIMING_EPSILON_S = 1e-6
EXTERNAL_COUNTERS = {
    "external_lookup_query_tokens": "external_queries",
    "external_lookup_hit_tokens": "external_hits",
    "external_prompt_tokens": "external_prefill_tokens",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def argv(value, label):
    require(
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item for item in value),
        f"{label} must be a nonempty argv array",
    )
    return value


def option(args, flag, *, required=True):
    """Find the separated argv options emitted by launch.py."""
    require(
        not any(item.startswith(flag + "=") for item in args),
        f"use launch receipt syntax for {flag}",
    )
    if not required and flag not in args:
        return None, None
    require(
        args.count(flag) == 1, f"expected exactly one {flag} in actual launch command"
    )
    index = args.index(flag)
    require(
        index + 1 < len(args) and not args[index + 1].startswith("--"),
        f"missing value for {flag}",
    )
    return index, args[index + 1]


def docker_port(port):
    return type(port) is int and 1 <= port <= 65535 and port != SERVICE_PORT


def sidecar_contract(launch, connector, api_port):
    if connector is None:
        require(
            all(
                launch.get(key) is None
                for key in (
                    "lmcache_mp",
                    "lmcache_server_command",
                    "lmcache_container_id",
                )
            ),
            "LMCache OFF launch must omit the sidecar",
        )
        return None
    profile = LMCacheMP(**launch["lmcache_mp"])
    require(
        all(type(value) is int and value > 0 for value in launch["lmcache_mp"].values())
        and profile.l1_init_size_gb <= profile.l1_size_gb,
        "invalid LMCache sidecar profile",
    )
    ports = (api_port, profile.port, profile.http_port)
    require(
        all(docker_port(port) for port in ports) and len(set(ports)) == 3,
        "vLLM and LMCache sidecar require distinct loopback test ports",
    )
    require(
        profile.transfer_config() == connector,
        "LMCache sidecar profile differs from the actual connector config",
    )
    command = argv(launch["lmcache_server_command"], "LMCache sidecar command")
    name = option(command, "--name")[1]
    require(
        command == profile.server_command(name, launch["image_id"]),
        "LMCache sidecar command differs from its loopback host network/IPC profile or image",
    )
    require(
        sha256(launch["lmcache_container_id"])
        and launch["lmcache_container_id"] != launch["container_id"],
        "LMCache sidecar must record its own actual container ID",
    )
    return launch["lmcache_mp"]


def launch_contract(arm, enabled):
    launch, endpoint = arm["launch_receipt"], arm["endpoint"]
    image = launch["image_id"]
    require(
        isinstance(image, str) and image.startswith("sha256:") and sha256(image[7:]),
        "actual launch requires an immutable Docker image ID",
    )
    require(
        sha256(launch["source_manifest_sha256"]),
        "actual launch requires the runtime source manifest SHA256",
    )
    command = argv(launch["command"], "launch command")
    require(
        command[:2] == ["docker", "run"] and command.count(image) == 1,
        "actual launch command must run its recorded immutable image",
    )
    split = command.index(image)
    docker, serve = command[:split], command[split + 1 :]
    require(
        len(serve) >= 2 and serve[0] == "serve" and serve[1].startswith("/"),
        "actual launch command must serve a mounted model",
    )
    require(
        docker.count("--network=host") == docker.count("--ipc=host") == 1,
        "both actual launch commands require host network and host IPC",
    )
    for flag, value in (
        ("--host", "127.0.0.1"),
        ("--max-model-len", "1048576"),
        ("--gpu-memory-utilization", "0.925"),
    ):
        require(
            option(serve, flag)[1] == value, f"actual launch requires {flag} {value}"
        )
    port = int(option(serve, "--port")[1])
    require(
        docker_port(port),
        f"actual launch must use a test port other than {SERVICE_PORT}",
    )
    base = urlsplit(loopback_url(arm["base_url"]))
    require(
        base.hostname in ("127.0.0.1", "localhost") and base.port == port,
        "replay loopback base URL differs from the actual launch port",
    )
    model = option(serve, "--served-model-name")[1]

    container_id = launch["container_id"]
    require(sha256(container_id), "actual launch requires a full container ID")
    if endpoint is None:
        require(
            arm.get("status") == "FAIL" and arm.get("completion_post_attempts") == 0,
            "a replay with POSTs requires an inspected endpoint",
        )
    else:
        require(
            isinstance(endpoint.get("started_at"), str) and endpoint["started_at"],
            "inspected endpoint lacks an actual start time",
        )
        require(
            all(
                endpoint.get(key) == value
                for key, value in {
                    "image_id": image,
                    "container_id": container_id,
                    "command": serve,
                    "network_mode": "host",
                    "ipc_mode": "host",
                    "port": port,
                }.items()
            ),
            "inspected endpoint differs from actual launch command or identities",
        )

    transfer_index, raw_transfer = option(
        serve, "--kv-transfer-config", required=enabled
    )
    require(
        enabled == (transfer_index is not None),
        "OFF actual launch must have no KV transfer connector",
    )
    connector = json.loads(raw_transfer) if enabled else None
    if enabled:
        require(
            isinstance(connector, dict),
            "ON actual launch requires a KV transfer JSON object",
        )
        del serve[transfer_index : transfer_index + 2]
    lmcache = sidecar_contract(launch, connector, port)
    require(
        launch.get("kv_transfer_config", connector) == connector,
        "KV transfer receipt differs from the actual launch command",
    )
    require(
        launch.get("kv_transfer_config_sha256", digest(connector) if enabled else None)
        == (digest(connector) if enabled else None),
        "KV transfer receipt SHA256 differs from the actual command",
    )

    name_index, name = option(docker, "--name")
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", name) is not None,
        "invalid benchmark container name",
    )
    mounts = [
        (i + 1, docker[i + 1]) for i, arg in enumerate(docker[:-1]) if arg == "--mount"
    ]
    bind_suffix = f",dst={serve[1]},readonly"
    weight_mounts = [
        mount
        for _, mount in mounts
        if mount.startswith("type=bind,src=/") and mount.endswith(bind_suffix)
    ]
    require(
        len(weight_mounts) == 1,
        "actual launch must bind one verified model source read-only",
    )
    compile_mounts = [
        index
        for index, mount in mounts
        if re.fullmatch(
            r"type=volume,src=[A-Za-z0-9][A-Za-z0-9_.-]*,dst=/workspace", mount
        )
    ]
    require(
        len(compile_mounts) == 1,
        "actual launch requires one compilation volume at /workspace",
    )
    docker[name_index + 1] = "<arm-container>"
    docker[compile_mounts[0]] = "type=volume,src=<compile-cache>,dst=/workspace"
    return {
        "image_id": image,
        "source_manifest_sha256": launch["source_manifest_sha256"],
        "weight_mount": weight_mounts[0],
        "normalized_command": docker + [image] + serve,
        "port": port,
        "served_model_name": model,
        "lmcache_mp": lmcache,
    }


def compare_launches(off, on):
    baseline, candidate = launch_contract(off, False), launch_contract(on, True)
    for key, label in (
        ("image_id", "immutable image"),
        ("source_manifest_sha256", "runtime source"),
        ("weight_mount", "weight/model source"),
    ):
        require(baseline[key] == candidate[key], f"{label} differs across arms")
    receipts = [arm["launch_receipt"] for arm in (off, on)]
    for field in ("source_lock", "model_manifest_sha256", "model_revision"):
        require(
            receipts[0].get(field) == receipts[1].get(field),
            f"{field} differs across arms",
        )
    require(
        sha256(receipts[0].get("model_manifest_sha256"))
        and isinstance(receipts[0].get("model_revision"), str)
        and receipts[0]["model_revision"],
        "launch must record the SHA256 and revision of the verified weight manifest",
    )
    baseline.update(
        model_manifest_sha256=receipts[0]["model_manifest_sha256"],
        model_revision=receipts[0]["model_revision"],
    )
    left, right = (
        baseline.pop("normalized_command"),
        candidate.pop("normalized_command"),
    )
    if left != right:
        index = next(
            (i for i, (x, y) in enumerate(zip(left, right, strict=False)) if x != y),
            min(len(left), len(right)),
        )
        raise ValueError(
            f"actual launch command differs across arms at normalized argv index {index}"
        )
    require(
        receipts[0]["container_id"] != receipts[1]["container_id"],
        "arms must record distinct actual containers",
    )
    baseline.update(
        normalized_command_sha256=digest(left), lmcache_mp=candidate["lmcache_mp"]
    )
    return baseline


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def count(value):
    return type(value) is int and value >= 0


def workload_contract(off, on, launch):
    for arm in (off, on):
        require(
            arm["schema_version"] == 1 and arm["kind"] == "lmcache_replay",
            "expected replay result schema v1",
        )
        require(
            arm["workload_kind"] == "synthetic",
            "this report only accepts the synthetic replay workload",
        )
        require(
            arm["model"] == launch["served_model_name"] == MODEL,
            "replay model differs from the launched model",
        )
        require(arm["status"] in ("PASS", "FAIL"), "invalid replay status")
        require(
            isinstance(arm["namespace"], str)
            and re.fullmatch(r"[A-Za-z0-9_-]+", arm["namespace"]),
            "replay requires a recorded salt namespace",
        )
    mode = off["mode"]
    require(
        mode in ("fixed", "closed") and on["mode"] == mode,
        "fixed and closed replay modes cannot be compared together",
    )
    for field in ("workload_sha256", "runner_sha256", "offered_schedule_sha256"):
        require(
            sha256(off[field]) and off[field] == on[field],
            f"{field} differs across arms or is invalid",
        )
    for field in (
        "users",
        "turns",
        "duration_s",
        "request_timeout_s",
        "completion_retry_limit",
    ):
        require(off[field] == on[field], f"{field} differs across arms")
    require(off.get("run_key") == on.get("run_key"), "run_key differs across arms")
    require(
        off["namespace"] != on["namespace"],
        "arms require independent replay salt namespaces",
    )
    require(
        all(
            count(arm[field]) and arm[field] > 0
            for arm in (off, on)
            for field in ("users", "turns")
        ),
        "replay requires positive frozen user and turn counts",
    )
    require(
        all(
            number(arm[field]) and arm[field] > 0
            for arm in (off, on)
            for field in ("duration_s", "request_timeout_s")
        ),
        "replay duration and request timeout must be positive",
    )
    require(
        all(
            count(arm["completion_retry_limit"]) and arm["completion_retry_limit"] == 0
            for arm in (off, on)
        ),
        "replay requires a zero-retry POST policy",
    )
    if mode == "closed":
        require(
            off["users"] == 70 and off["duration_s"] == 600,
            "closed replay requires 70 users and a 600 s admission window",
        )
    return mode


def validate_row(row, namespace, duration, elapsed, turns, mode):
    user, turn, cycle = row["user_id"], row["turn"], row["cycle"]
    require(
        isinstance(user, str)
        and user
        and count(turn)
        and 1 <= turn <= turns
        and count(cycle),
        "invalid replay user, turn or cycle",
    )
    chain = f"{user}/session-{cycle}"
    require(
        row["chain_id"] == chain and (mode == "closed" or cycle == 0),
        f"invalid chain ID {row['chain_id']}",
    )
    require(
        row["cache_salt"] == f"replay-{namespace}-{user}-{cycle}",
        f"{chain}: cache salt differs from its frozen session",
    )
    require(sha256(row["request_sha256"]), f"{chain}: missing canonical request SHA256")
    require(
        all(
            count(row[field]) and row[field] > 0
            for field in ("expected_input_tokens", "expected_output_tokens")
        ),
        f"{chain}: missing frozen input length or output contract",
    )
    planned = row["planned_at_s"]
    require(
        number(planned) and planned < duration,
        f"{chain}: planned arrival is outside the admission window",
    )
    status, posted = row["status"], row["posted"]
    require(
        type(posted) is bool
        and status in ("PASS", "FAIL", "NOT_SENT")
        and posted == (status != "NOT_SENT"),
        f"{chain}: invalid POST evidence and outcome",
    )
    started, completed = row.get("actual_started_s"), row.get("completed_at_s")
    if posted:
        require(
            number(started)
            and number(completed)
            and planned - TIMING_EPSILON_S
            <= started
            <= completed
            <= elapsed + TIMING_EPSILON_S,
            f"{chain}: invalid POST arrival/start/completion times",
        )
        require(
            number(row["client_queue_s"])
            and abs(row["client_queue_s"] - (started - planned)) <= TIMING_EPSILON_S,
            f"{chain}: client queue differs from planned arrival and actual HTTP start",
        )
    else:
        require(started is None, f"{chain}: NOT_SENT has an HTTP start time")
    if status == "PASS":
        ttft, e2e = row["ttft_s"], row["e2e_s"]
        require(
            number(ttft)
            and number(e2e)
            and ttft <= e2e <= completed - started + TIMING_EPSILON_S,
            f"{chain}: invalid HTTP TTFT or E2E sample",
        )
        require(
            count(row["input_tokens"])
            and row["input_tokens"] == row["expected_input_tokens"]
            and count(row["output_tokens"])
            and row["output_tokens"] == row["expected_output_tokens"]
            and row["finish_reason"] == "length"
            and sha256(row["output_sha256"]),
            f"{chain}: PASS usage violates the frozen input/output contract",
        )
        cached = row.get("cached_tokens")
        require(
            cached is None or (count(cached) and cached <= row["input_tokens"]),
            f"{chain}: invalid API cached token count",
        )
    else:
        require(
            isinstance(row.get("error"), str) and row["error"].strip(),
            f"{chain}: failed outcome needs its original error",
        )
    return user, cycle, turn


def request_contract(arm):
    mode, duration, elapsed = arm["mode"], arm["duration_s"], arm["elapsed_s"]
    require(
        number(elapsed)
        or (
            elapsed is None
            and arm["status"] == "FAIL"
            and arm["completion_post_attempts"] == 0
        ),
        "replay elapsed time must be finite and nonnegative when POSTs were attempted",
    )
    scheduled, frozen, first_arrivals, users = {}, {}, {}, defaultdict(list)
    rows = arm["rows"]
    require(isinstance(rows, list), "replay rows must be an array")
    for row in rows:
        user, cycle, turn = validate_row(
            row, arm["namespace"], duration, elapsed, arm["turns"], mode
        )
        key = (row["chain_id"], turn)
        contract = (
            row["request_sha256"],
            row["expected_input_tokens"],
            row["expected_output_tokens"],
        )
        require(key not in scheduled, f"duplicate POST or planned row: {key}")
        require(
            frozen.setdefault((user, turn), contract) == contract,
            f"{user} turn {turn}: frozen request changed across cycles",
        )
        scheduled[key] = (row["planned_at_s"], *contract)
        if cycle == 0 and turn == 1:
            first_arrivals[user] = row["planned_at_s"]
        users[user].append((cycle, turn, row))
    require(len(users) <= arm["users"], "more replay users than the frozen plan")
    for user, entries in users.items():
        previous_end = None
        previous_length = 0
        for index, (cycle, turn, row) in enumerate(sorted(entries)):
            require(
                (cycle, turn - 1) == divmod(index, arm["turns"]),
                f"{user}: replay sequence skipped or repeated a turn",
            )
            if turn > 1:
                require(
                    row["expected_input_tokens"] > previous_length,
                    f"{user}: input lengths must increase within a session",
                )
            previous_length = row["expected_input_tokens"]
            started = row.get("actual_started_s")
            if previous_end is not None and started is not None:
                require(
                    started >= previous_end - TIMING_EPSILON_S,
                    f"{user}: overlapping POSTs in one user chain",
                )
                if mode == "closed":
                    require(
                        abs(row["planned_at_s"] - previous_end) <= TIMING_EPSILON_S,
                        f"{user}: closed user deviates from zero think time",
                    )
            previous_end = row.get("completed_at_s")
    attempts = arm["completion_post_attempts"]
    require(
        count(attempts) and attempts == sum(row["posted"] for row in rows),
        "actual POST attempt count disagrees with replay rows",
    )
    if mode == "fixed":
        offered = arm["completion_offered"]
        require(
            count(offered) and offered == len(rows) == arm["users"] * arm["turns"],
            "fixed result must retain a row for every offered turn",
        )
        plan = [
            [chain, turn, float(values[0]), values[1]]
            for (chain, turn), values in sorted(scheduled.items())
        ]
        require(
            workload_digest(["fixed", duration, plan])
            == arm["offered_schedule_sha256"],
            "fixed offered schedule hash does not match its recorded chain/turn plans",
        )
        return scheduled
    require(
        arm["completion_offered"] is None,
        "closed replay has no fixed offered POST count",
    )
    if len(frozen) == arm["users"] * arm["turns"]:
        plan = [
            [
                user,
                float(first_arrivals[user]),
                [frozen[user, t][0] for t in range(1, arm["turns"] + 1)],
            ]
            for user in sorted(first_arrivals)
        ]
        require(
            workload_digest(["closed", duration, plan])
            == arm["offered_schedule_sha256"],
            "closed offered schedule hash does not match the recorded users, first arrivals and frozen turns",
        )
    return frozen, first_arrivals


def percentiles(samples):
    return {
        "n": len(samples),
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
    }


def cache_evidence(arm, good, enabled, reasons):
    cached = [
        row["cached_tokens"] for row in good if row.get("cached_tokens") is not None
    ]
    cache = {
        "api_cached_tokens": sum(cached) if cached else None,
        "api_cached_token_samples": len(cached),
        "dp_counters": None,
        "dp_counters_match_completed_usage": None,
        **dict.fromkeys(EXTERNAL_COUNTERS),
    }
    try:
        before, after = arm.get("metrics_before"), arm.get("metrics_after")
        require(
            isinstance(before, dict) and isinstance(after, dict),
            "missing before/after snapshots",
        )
        delta = counter_deltas(engine_counters(before), engine_counters(after))
        require(
            delta == arm.get("metrics_delta_by_engine"),
            "counter deltas differ from raw metrics snapshots",
        )
    except (ValueError, TypeError, AttributeError) as error:
        reasons.append(f"DP metrics unavailable or invalid: {error}")
        return cache

    cache["dp_counters"] = delta
    for label, key in EXTERNAL_COUNTERS.items():
        values = [engine[key] for engine in delta.values()]
        if all(value is not None for value in values):
            cache[label] = sum(values)
    expected = {
        "finished_requests": len(good),
        "prompt_tokens": sum(row["input_tokens"] for row in good),
        "generation_tokens": sum(row["output_tokens"] for row in good),
    }
    cache["dp_counters_match_completed_usage"] = all(
        sum(engine[key] for engine in delta.values()) == value
        for key, value in expected.items()
    )
    if len(good) == len(arm["rows"]) and not cache["dp_counters_match_completed_usage"]:
        reasons.append("DP counters do not match completed replay usage")
    if arm["mode"] == "closed" and not all(
        engine["finished_requests"] > 0 for engine in delta.values()
    ):
        reasons.append("closed replay lacks completed traffic on both DP engines")
    if enabled and cache["external_prompt_tokens"] is None:
        reasons.append("LMCache ON lacks per-DP external prompt token source counters")
    if not enabled and any(
        engine[key]
        for engine in delta.values()
        for key in ("external_hits", "external_prefill_tokens")
    ):
        reasons.append("LMCache OFF recorded external cache hits")
    for engine in delta.values():
        used, hits, queries = (
            engine[key]
            for key in ("external_prefill_tokens", "external_hits", "external_queries")
        )
        if (
            (used is not None and used > engine["prompt_tokens"])
            or (used is not None and hits is not None and used > hits)
            or (hits is not None and queries is not None and hits > queries)
        ):
            reasons.append("inconsistent external cache counter deltas")
            break
    return cache


def summarize(arm, *, enabled):
    mode, limit, elapsed = arm["mode"], arm["duration_s"], arm["elapsed_s"]
    rows = arm["rows"]
    good = [row for row in rows if row["status"] == "PASS"]
    failed = [row for row in rows if row["status"] != "PASS"]
    in_window = [row for row in good if row["completed_at_s"] <= limit]
    late_posts = [
        row for row in rows if row["posted"] and row["actual_started_s"] >= limit
    ]
    output = sum(row["output_tokens"] for row in good)
    window_output = sum(row["output_tokens"] for row in in_window)
    last_completion = max((row["completed_at_s"] for row in good), default=None)
    errors = arm["errors"]
    require(
        isinstance(errors, list)
        and all(isinstance(error, str) and error for error in errors),
        "replay errors must be an array of messages",
    )
    reasons = []
    if failed:
        reasons.append("one or more requests failed or were not sent")
    if late_posts:
        reasons.append("POST started outside the admission window")
    if not good:
        reasons.append("no request completed the output contract")
    if mode == "closed" and (elapsed is None or elapsed + TIMING_EPSILON_S < limit):
        reasons.append("closed replay did not run for the full admission window")
    if arm["status"] != "PASS" or errors:
        reasons.append("runner recorded errors")
    if arm.get("endpoint_after") != arm["endpoint"]:
        reasons.append("end-of-run endpoint identity is missing or changed")
    if (
        mode == "closed"
        and len({row["user_id"] for row in rows if row["posted"]}) != arm["users"]
    ):
        reasons.append("one or more closed users missed their first POST")
    cache = cache_evidence(arm, good, enabled, reasons)
    return {
        "status": "FAIL" if reasons else "PASS",
        "acceptance_issues": reasons,
        "elapsed_s": elapsed,
        "completion_offered": arm["completion_offered"],
        "completion_post_attempts": arm["completion_post_attempts"],
        "completed_ok": len(good),
        "success_rate": len(good) / len(rows) if rows else None,
        "completed_within_window": len(in_window),
        "completed_during_drain": len(good) - len(in_window),
        "failed_posts": sum(row["posted"] for row in failed),
        "not_sent": sum(not row["posted"] for row in failed),
        "late_posts": len(late_posts),
        "failed_rows": failed,
        "errors": errors,
        "observed_output_tokens": output,
        "output_tokens_completed_within_window": window_output,
        "output_tokens_per_s": window_output / limit
        if mode == "closed"
        else (output / elapsed if elapsed else None),
        "drained_output_tokens_per_s": output / elapsed if elapsed else None,
        "full_round_complete_s": (
            last_completion if not failed and not late_posts else None
        )
        if mode == "fixed"
        else None,
        "closed_last_completion_s": last_completion if mode == "closed" else None,
        "latency_s": {
            "http_ttft": percentiles([row["ttft_s"] for row in good]),
            "http_e2e": percentiles([row["e2e_s"] for row in good]),
            "arrival_ttft": percentiles(
                [row["client_queue_s"] + row["ttft_s"] for row in good]
            ),
            "arrival_e2e": percentiles(
                [row["client_queue_s"] + row["e2e_s"] for row in good]
            ),
            "client_queue": percentiles(
                [row["client_queue_s"] for row in rows if row["posted"]]
            ),
        },
        "cache": cache,
        "endpoint": arm["endpoint"],
        "weights_rechecked_at_utc": arm["launch_receipt"].get(
            "weights_rechecked_at_utc"
        ),
    }


def compare(off, on):
    launch = compare_launches(off, on)
    mode = workload_contract(off, on, launch)
    off_contract, on_contract = request_contract(off), request_contract(on)
    if mode == "fixed":
        require(
            off_contract == on_contract,
            "fixed chain ID/turn, input length, output contract or planned arrival differs",
        )
    else:
        for off_values, on_values in zip(off_contract, on_contract, strict=True):
            common = off_values.keys() & on_values.keys()
            require(
                all(off_values[key] == on_values[key] for key in common),
                "closed frozen request contract or first arrival differs",
            )
    arms = {"off": summarize(off, enabled=False), "on": summarize(on, enabled=True)}
    healthy = all(arm["status"] == "PASS" for arm in arms.values())
    baseline, candidate = [arm["output_tokens_per_s"] for arm in arms.values()]
    return {
        "kind": "lmcache_replay_comparison",
        "schema_version": 1,
        "status": "PASS" if healthy else "FAIL",
        "mode": mode,
        "workload_kind": "synthetic",
        "scope": "Generated token chains. The original request trace and its traffic distribution were unavailable.",
        "latency_scope": "HTTP and arrival latency distributions cover successful responses through drain; client queue covers all POST attempts.",
        "cache_scope": "Prompt tokens served by external KV transfer are measured by per-engine Prometheus counters through drain. Per-engine counts aggregate all chains.",
        "launch": launch,
        "workload": {
            key: off[key]
            for key in (
                "workload_sha256",
                "runner_sha256",
                "offered_schedule_sha256",
                "users",
                "turns",
                "duration_s",
                "request_timeout_s",
            )
        },
        "arms": arms,
        "performance_comparison": {
            "status": "VALID" if healthy else "WITHHELD",
            "throughput_basis": (
                "successful output tokens / elapsed wall clock including drain"
                if mode == "fixed"
                else "output tokens from successful responses completed within the 600 s window / 600 s"
            ),
            "on_over_off_output_tokens_per_s": candidate / baseline
            if healthy and baseline
            else None,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("off", type=Path, help="LMCache-off replay result JSON")
    parser.add_argument("on", type=Path, help="LMCache-on replay result JSON")
    parser.add_argument(
        "--output", type=Path, help="write the comparison JSON to this path"
    )
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            require(
                args.output.resolve() not in {args.off.resolve(), args.on.resolve()},
                "--output must differ from both replay input files",
            )
        report = compare(
            json.loads(args.off.read_text()), json.loads(args.on.read_text())
        )
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(text)
        else:
            args.output.write_text(text)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"lmcache replay comparison: {error}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
