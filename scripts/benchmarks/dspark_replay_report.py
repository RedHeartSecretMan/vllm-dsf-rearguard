"""Compare fixed DSpark greedy and probabilistic replay evidence.

    python3 -m scripts.benchmarks.dspark_replay_report GREEDY.json PROBABILISTIC.json

The command reads two completed replay reports and writes a JSON comparison to
stdout.  ``--output`` follows the other benchmark report commands.  Exit 0 is a
complete, accepted comparison, exit 1 retains observations while withholding
an invalid comparison, and exit 2 identifies incompatible or malformed input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from scripts.bench_common import MODEL, percentile

SCHEMA_VERSION = 1
EXPECTED_PARALLEL = {"tp": 4, "dp": 2, "ep": 8}
EXPECTED_CONTEXT = 1048576
EXPECTED_BATCH_TOKENS = 16384
EXPECTED_MEMORY = "0.925"
EXPECTED_MODEL_MOUNT = "/models/DeepSeek-V4.1-Flash-1017"
EXPECTED_SPECULATIVE = {
    "method": "dspark",
    "num_speculative_tokens": 5,
    "rejection_sample_method": "standard",
    "enable_adaptive_verification": False,
}
EXPECTED_COUNTERS = {
    "vllm:spec_decode_num_drafts_total": "drafts",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
}
COUNTER_FIELDS = tuple(EXPECTED_COUNTERS.values())
DP_RANKS = ("0", "1")
OSL_THRESHOLD = 2405
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LABEL_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\"(?P<value>[^\"]*)\"")


class ContractError(ValueError):
    """An input violates the public report contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha256(value):
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def finite_number(value, *, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def positive_int(value):
    return type(value) is int and value > 0


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def option(args, flag, *, required=True):
    require(isinstance(args, list), "launch command must be an argv array")
    require(not any(item.startswith(flag + "=") for item in args), f"use separated argv syntax for {flag}")
    count = args.count(flag)
    if not required and count == 0:
        return None
    require(count == 1, f"expected exactly one {flag} in actual launch command")
    index = args.index(flag)
    require(index + 1 < len(args) and not args[index + 1].startswith("--"), f"missing value for {flag}")
    return args[index + 1]


def json_option(args, flag):
    try:
        return json.loads(option(args, flag))
    except json.JSONDecodeError as error:
        raise ContractError(f"{flag} must contain JSON") from error


def _mounts(docker):
    mounts = []
    for index, value in enumerate(docker[:-1]):
        if value == "--mount":
            require(index + 1 < len(docker), "--mount is missing its value")
            mounts.append(docker[index + 1])
    return mounts


def _model_mount(command):
    mounts = _mounts(command)
    weights = [
        mount
        for mount in mounts
        if mount.startswith("type=bind,src=/")
        and mount.endswith(f",dst={EXPECTED_MODEL_MOUNT},readonly")
    ]
    require(len(weights) == 1, "actual launch must bind one verified model source read-only")
    compile_mounts = [
        mount
        for mount in mounts
        if re.fullmatch(r"type=volume,src=[A-Za-z0-9][A-Za-z0-9_.-]*,dst=/workspace", mount)
    ]
    require(len(compile_mounts) == 1, "actual launch requires one compilation volume at /workspace")
    return weights[0], "type=volume,src=<compile-cache>,dst=/workspace"


def _serve_contract(serve, method):
    require(serve[:2] == ["serve", EXPECTED_MODEL_MOUNT], "actual launch must serve the pinned model mount")
    require(option(serve, "--served-model-name") == MODEL, "actual launch served model differs")
    require(option(serve, "--host") in ("0.0.0.0", "127.0.0.1"), "actual launch must bind a loopback or container host")
    port = option(serve, "--port")
    require(port.isdigit() and 1 <= int(port) <= 65535, "actual launch port is invalid")
    for flag, expected in (
        ("--tensor-parallel-size", "4"),
        ("--data-parallel-size", "2"),
        ("--data-parallel-size-local", "2"),
        ("--max-model-len", str(EXPECTED_CONTEXT)),
        ("--max-num-batched-tokens", str(EXPECTED_BATCH_TOKENS)),
        ("--gpu-memory-utilization", EXPECTED_MEMORY),
        ("--all2all-backend", "allgather_reducescatter"),
    ):
        require(option(serve, flag) == expected, f"actual launch requires {flag} {expected}")
    require(serve.count("--enable-expert-parallel") == 1, "actual launch requires expert parallelism")
    spec = json_option(serve, "--speculative-config")
    require(isinstance(spec, dict), "actual launch speculative config must be an object")
    expected = {**EXPECTED_SPECULATIVE, "draft_sample_method": method, "use_local_argmax_reduction": method == "greedy"}
    require(spec == expected, "actual launch DSpark config differs from the selected draft method")
    return {
        "port": int(port),
        "speculative_config": spec,
        "served_model_name": MODEL,
    }


def _launch_contract(arm, expected_method):
    receipt = arm.get("launch_receipt")
    require(isinstance(receipt, dict), "replay requires a launch receipt")
    method = receipt.get("dspark_draft_sample_method")
    require(method == expected_method, "launch receipt draft method differs from its arm")
    require(receipt.get("dspark") == 5, "launch receipt requires DSpark k5")
    require(receipt.get("context") == EXPECTED_CONTEXT, "launch receipt requires a 1M context")
    parallel = receipt.get("parallel")
    require(isinstance(parallel, dict), "launch receipt requires TP4/DP2/EP8 parallelism")
    require(all(parallel.get(key) == value for key, value in EXPECTED_PARALLEL.items()), "launch receipt requires TP4/DP2/EP8 parallelism")
    image_id = receipt.get("image_id")
    require(isinstance(image_id, str) and image_id.startswith("sha256:") and sha256(image_id[7:]), "launch receipt requires an immutable runtime image ID")
    require(sha256(receipt.get("source_manifest_sha256")), "launch receipt requires the runtime source manifest SHA256")
    require(sha256(receipt.get("model_manifest_sha256")), "launch receipt requires the verified weight manifest SHA256")
    require(isinstance(receipt.get("model_revision"), str) and receipt["model_revision"], "launch receipt requires the verified weight revision")
    command = receipt.get("command")
    require(isinstance(command, list) and all(isinstance(value, str) and value for value in command), "launch command must be a nonempty argv array")
    require(command[:2] == ["docker", "run"], "actual launch command must run Docker")
    image = receipt["image_id"]
    require(command.count(image) == 1, "actual launch command must run its recorded immutable image")
    split = command.index(image)
    docker, serve = command[:split], command[split + 1 :]
    require("--gpus" in docker and option(docker, "--gpus") == "all", "actual launch must expose all GPUs")
    weight_mount, compile_mount = _model_mount(docker)
    serve_values = _serve_contract(serve, expected_method)
    container_id = receipt.get("container_id")
    require(sha256(container_id), "launch receipt requires a full container ID")
    endpoint = arm.get("endpoint")
    require(isinstance(endpoint, dict), "replay requires an inspected endpoint receipt")
    require(endpoint.get("container_id") == container_id and endpoint.get("image_id") == image, "endpoint identity differs from the actual launch receipt")
    require(isinstance(endpoint.get("started_at"), str) and endpoint["started_at"], "endpoint receipt lacks an actual start time")
    if endpoint.get("command") is not None:
        require(endpoint["command"] == serve, "endpoint command differs from the actual launch command")
    if endpoint.get("port") is not None:
        require(endpoint["port"] == serve_values["port"], "endpoint port differs from the actual launch command")
    endpoint_after = arm.get("endpoint_after")
    if endpoint_after is not None:
        require(isinstance(endpoint_after, dict), "end-of-run endpoint receipt must be an object")
        require(endpoint_after == endpoint, "end-of-run endpoint identity is missing or changed")
    lmcache_fields = ("lmcache_mp", "lmcache_server_command", "kv_transfer_config", "kv_transfer_config_sha256")
    lmcache = tuple(receipt.get(field) for field in lmcache_fields)
    if any(value is not None for value in lmcache):
        require(all(value is not None for value in lmcache), "LMCache configuration must be complete when present")
    normalized_docker = list(docker)
    name_index = normalized_docker.index("--name") if "--name" in normalized_docker else None
    if name_index is not None:
        require(name_index + 1 < len(normalized_docker), "Docker --name is missing its value")
        normalized_docker[name_index + 1] = "<arm-container>"
    for index, value in enumerate(normalized_docker):
        if re.fullmatch(r"type=volume,src=[A-Za-z0-9][A-Za-z0-9_.-]*,dst=/workspace", value):
            normalized_docker[index] = "type=volume,src=<compile-cache>,dst=/workspace"
    normalized_serve = list(serve)
    spec_index = normalized_serve.index("--speculative-config")
    normalized_serve[spec_index + 1] = json.dumps(serve_values["speculative_config"], sort_keys=True, separators=(",", ":"))
    return {
        "image_id": image,
        "source_manifest_sha256": receipt["source_manifest_sha256"],
        "model_manifest_sha256": receipt["model_manifest_sha256"],
        "model_revision": receipt["model_revision"],
        "weight_mount": weight_mount,
        "compile_mount": compile_mount,
        "normalized_command": normalized_docker + [image] + normalized_serve,
        "speculative_config": serve_values["speculative_config"],
        "lmcache": lmcache,
        "container_id": container_id,
        "port": serve_values["port"],
        "parallel": parallel,
        "source_lock": receipt.get("source_lock"),
    }


def compare_launches(greedy, probabilistic):
    left = _launch_contract(greedy, "greedy")
    right = _launch_contract(probabilistic, "probabilistic")
    left_receipt = greedy["launch_receipt"]
    right_receipt = probabilistic["launch_receipt"]
    variable_receipt_fields = {
        "created_at", "container_name", "container_id", "command",
        "dspark_draft_sample_method",
    }
    for field in set(left_receipt) | set(right_receipt):
        if field not in variable_receipt_fields:
            require(left_receipt.get(field) == right_receipt.get(field), f"launch receipt {field} differs across arms")
    for key, label in (
        ("image_id", "runtime image"),
        ("source_manifest_sha256", "runtime source"),
        ("model_manifest_sha256", "weight manifest"),
        ("model_revision", "weight revision"),
        ("weight_mount", "weight/model source"),
        ("compile_mount", "compilation volume"),
        ("parallel", "parallelism"),
        ("source_lock", "runtime source lock"),
        ("lmcache", "LMCache configuration"),
    ):
        require(left[key] == right[key], f"{label} differs across arms")
    require(left["normalized_command"] != right["normalized_command"], "arms must differ in their DSpark sampling method")
    left_command = list(left["normalized_command"])
    right_command = list(right["normalized_command"])
    spec_position = left_command.index("--speculative-config")
    left_command[spec_position : spec_position + 2] = ["--speculative-config", "<sampling-config>"]
    right_command[spec_position : spec_position + 2] = ["--speculative-config", "<sampling-config>"]
    require(left_command == right_command, "actual launch command differs across arms outside DSpark sampling")
    require(left["container_id"] != right["container_id"], "arms must record distinct actual containers")
    return {
        "image_id": left["image_id"],
        "source_manifest_sha256": left["source_manifest_sha256"],
        "model_manifest_sha256": left["model_manifest_sha256"],
        "model_revision": left["model_revision"],
        "weight_mount": left["weight_mount"],
        "source_lock": left["source_lock"],
        "parallel": left["parallel"],
        "normalized_command_sha256": digest(left_command),
        "lmcache": left["lmcache"],
        "draft_sample_methods": {"greedy": "greedy", "probabilistic": "probabilistic"},
    }


def _parse_labels(value):
    return {match.group("key"): match.group("value") for match in LABEL_RE.finditer(value)}


def _counter_snapshot(snapshot):
    """Extract speculative counters from a raw /metrics response or nested map."""
    require(isinstance(snapshot, (dict, str)), "raw /metrics snapshot must be an object or text")
    by_dp = {rank: {} for rank in DP_RANKS}
    if isinstance(snapshot, dict) and all(isinstance(value, dict) for value in snapshot.values()):
        for rank in DP_RANKS:
            values = snapshot.get(rank)
            if values is None:
                continue
            for raw_key, value in values.items():
                field = EXPECTED_COUNTERS.get(str(raw_key).split("{", 1)[0])
                if field is None:
                    field = raw_key if raw_key in COUNTER_FIELDS else {"draft": "drafts", "accepted": "accepted_tokens"}.get(raw_key)
                if field is not None and field in COUNTER_FIELDS:
                    require(finite_number(value), f"invalid DP{rank} speculative counter {field}")
                    require(field not in by_dp[rank], f"duplicate speculative counter for DP{rank}: {field}")
                    by_dp[rank][field] = value
        return by_dp
    entries = snapshot.items() if isinstance(snapshot, dict) else (
        line.split(None, 1) for line in snapshot.splitlines() if line.strip() and not line.startswith("#")
    )
    for entry in entries:
        try:
            key, value = entry
        except (TypeError, ValueError) as error:
            raise ContractError("raw /metrics line must contain a sample and value") from error
        match = re.match(r"(?P<name>[^\{]+)(?:\{(?P<labels>.*)\})?$", key)
        if match is None:
            continue
        field = EXPECTED_COUNTERS.get(match.group("name"))
        if field is None:
            continue
        labels = _parse_labels(match.group("labels") or "")
        rank = labels.get("engine") or labels.get("dp_rank") or labels.get("data_parallel_rank")
        require(rank in DP_RANKS, f"speculative metric lacks a DP engine label: {key}")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ContractError(f"invalid speculative metric value: {key}") from error
        require(finite_number(number), f"invalid speculative metric value: {key}")
        require(field not in by_dp[rank], f"duplicate speculative counter for DP{rank}: {field}")
        by_dp[rank][field] = int(number) if number.is_integer() else number
    return by_dp


def _counter_delta(before, after):
    result = {rank: {} for rank in DP_RANKS}
    for rank in DP_RANKS:
        for field in COUNTER_FIELDS:
            first, last = before[rank].get(field), after[rank].get(field)
            result[rank][field] = None if first is None or last is None else last - first
            if result[rank][field] is not None:
                require(result[rank][field] >= 0, f"speculative counter reset on DP{rank}: {field}")
        if all(result[rank][field] is not None for field in COUNTER_FIELDS):
            result[rank]["accepted_per_draft"] = (
                result[rank]["accepted_tokens"] / result[rank]["drafts"]
                if result[rank]["drafts"]
                else None
            )
        else:
            result[rank]["accepted_per_draft"] = None
        result[rank]["draft"] = result[rank].get("drafts")
        result[rank]["accepted"] = result[rank].get("accepted_tokens")
    return result


def _aggregate_counters(by_dp):
    total = {}
    for field in COUNTER_FIELDS:
        values = [by_dp[rank].get(field) for rank in DP_RANKS]
        total[field] = sum(values) if all(value is not None for value in values) else None
    total["accepted_per_draft"] = (
        total["accepted_tokens"] / total["drafts"] if total.get("drafts") else None
    )
    total["draft"] = total.get("drafts")
    total["accepted"] = total.get("accepted_tokens")
    return total


def _raw_metric_pairs(arm, reports):
    def from_value(value):
        pairs = []
        for before_key, after_key in (("metrics_before", "metrics_after"), ("vllm_metrics_before", "vllm_metrics_after")):
            if before_key in value or after_key in value:
                require(before_key in value and after_key in value, "raw /metrics before and after must both be present")
                pairs.append((value[before_key], value[after_key]))
        metrics = value.get("vllm_metrics")
        if isinstance(metrics, dict):
            for before_key, after_key in (("before", "after"), ("metrics_before", "metrics_after")):
                if before_key in metrics or after_key in metrics:
                    require(before_key in metrics and after_key in metrics, "raw /metrics before and after must both be present")
                    pairs.append((metrics[before_key], metrics[after_key]))
        return pairs

    arm_pairs = from_value(arm)
    report_pairs = [pair for report in reports for pair in from_value(report)]
    return report_pairs or arm_pairs


def _reported_counters(reports):
    by_dp = {rank: {field: 0 for field in COUNTER_FIELDS} for rank in DP_RANKS}
    found = False
    for report in reports:
        counters = report.get("draft_counters")
        if not isinstance(counters, dict) or not isinstance(counters.get("by_engine"), dict):
            continue
        found = True
        for rank in DP_RANKS:
            values = counters["by_engine"].get(rank)
            if values is None:
                continue
            for field in COUNTER_FIELDS:
                value = values.get(field)
                if value is not None:
                    require(finite_number(value), f"invalid reported DP{rank} speculative counter {field}")
                    by_dp[rank][field] += value
    if not found:
        return None
    for rank in DP_RANKS:
        by_dp[rank]["accepted_per_draft"] = (
            by_dp[rank]["accepted_tokens"] / by_dp[rank]["drafts"]
            if by_dp[rank]["drafts"]
            else None
        )
        by_dp[rank]["draft"] = by_dp[rank].get("drafts")
        by_dp[rank]["accepted"] = by_dp[rank].get("accepted_tokens")
    return by_dp


def _draft_metrics(arm, reports, issues):
    pairs = _raw_metric_pairs(arm, reports)
    if pairs:
        deltas = []
        for before, after in pairs:
            deltas.append(_counter_delta(_counter_snapshot(before), _counter_snapshot(after)))
        by_dp = {rank: {} for rank in DP_RANKS}
        for rank in DP_RANKS:
            for field in COUNTER_FIELDS:
                values = [delta[rank].get(field) for delta in deltas]
                by_dp[rank][field] = sum(values) if all(value is not None for value in values) else None
            by_dp[rank]["accepted_per_draft"] = (
                by_dp[rank]["accepted_tokens"] / by_dp[rank]["drafts"]
                if by_dp[rank].get("drafts")
                else None
            ) if all(by_dp[rank].get(field) is not None for field in COUNTER_FIELDS) else None
            by_dp[rank]["draft"] = by_dp[rank].get("drafts")
            by_dp[rank]["accepted"] = by_dp[rank].get("accepted_tokens")
        if any(
            by_dp[rank].get(field) is None
            for rank in DP_RANKS
            for field in COUNTER_FIELDS
        ):
            issues.append("raw /metrics speculative counters incomplete")
        metrics = {"source": "raw_metrics_before_after", "by_dp": by_dp, "total": _aggregate_counters(by_dp)}
        reported = _reported_counters(reports)
        if reported is not None:
            for rank in DP_RANKS:
                for field in COUNTER_FIELDS:
                    if reported[rank][field] != by_dp[rank].get(field):
                        issues.append("raw /metrics speculative counters disagree with report deltas")
                        return metrics
        return metrics
    issues.append("raw /metrics before/after unavailable")
    reported = _reported_counters(reports)
    if reported is None:
        return {"source": None, "by_dp": None, "total": None}
    return {"source": "reported_delta_without_raw_metrics", "by_dp": reported, "total": _aggregate_counters(reported)}


def _validate_request(row, report_key, issues):
    require(isinstance(row, dict), f"{report_key}: request record must be an object")
    status = row.get("status")
    require(status in ("PASS", "FAIL"), f"{report_key}: request status must be PASS or FAIL")
    if "partial_stream" in row:
        require(type(row["partial_stream"]) is bool, f"{report_key}: partial_stream must be boolean")
    if status == "PASS":
        for field in ("input_tokens", "output_tokens"):
            require(positive_int(row.get(field)), f"{report_key}: successful request lacks {field}")
        for field in ("ttft_s", "tpot_s", "e2e_s"):
            if not finite_number(row.get(field)):
                issues.append(f"{report_key}: request {field} unavailable")
    else:
        error = row.get("error")
        if not (isinstance(error, str) and error.strip()):
            errors = row.get("errors")
            require(isinstance(errors, list) and errors and all(isinstance(value, str) and value for value in errors), f"{report_key}: failed request needs its original error")
        if "partial_stream" in row:
            require(type(row["partial_stream"]) is bool, f"{report_key}: partial_stream must be boolean")
            if row["partial_stream"]:
                issues.append(f"{report_key}: failed request retained partial_stream=true")


def _validate_report(report, index, issues, *, default_deadline=None, default_drain=None):
    require(isinstance(report, dict), f"report {index} must be an object")
    required = ("case", "round", "concurrency", "temperature", "input_tokens_each", "output_tokens_each", "workload_sha256", "status", "request_count", "success", "elapsed_s", "requests")
    for field in required:
        require(field in report, f"report {index} lacks {field}")
    key = (report["case"], report["round"])
    require(isinstance(report["case"], str) and report["case"], f"report {index} has an invalid case")
    require(positive_int(report["round"]), f"{key}: round must be positive")
    require(positive_int(report["concurrency"]), f"{key}: concurrency must be positive")
    require(finite_number(report["temperature"]), f"{key}: temperature must be finite")
    require(positive_int(report["input_tokens_each"]) and positive_int(report["output_tokens_each"]), f"{key}: fixed token lengths must be positive")
    require(sha256(report["workload_sha256"]), f"{key}: workload SHA256 is invalid")
    require(report["status"] in ("PASS", "FAIL"), f"{key}: invalid report status")
    if "errors" in report:
        require(isinstance(report["errors"], list) and all(isinstance(value, str) and value for value in report["errors"]), f"{key}: errors must be an array of messages")
    require(positive_int(report["request_count"]) and report["request_count"] == report["concurrency"] and 0 <= report["success"] <= report["request_count"], f"{key}: invalid request or success count")
    require(type(report["success"]) is int, f"{key}: success count must be an integer")
    require(finite_number(report["elapsed_s"], minimum=1e-12), f"{key}: elapsed_s must be positive")
    requests = report["requests"]
    require(isinstance(requests, list) and len(requests) == report["request_count"], f"{key}: POST records do not match request_count")
    statuses = []
    for row in requests:
        _validate_request(row, key, issues)
        statuses.append(row["status"])
    require(sum(status == "PASS" for status in statuses) == report["success"], f"{key}: success count differs from POST records")
    successful_rows = [row for row in requests if row["status"] == "PASS"]
    if all(positive_int(row.get("input_tokens")) for row in successful_rows) and report.get("observed_input_tokens") is not None:
        if report["observed_input_tokens"] != sum(row["input_tokens"] for row in successful_rows):
            issues.append(f"{key}: observed_input_tokens differs from POST records")
    if all(positive_int(row.get("output_tokens")) for row in successful_rows) and report.get("observed_output_tokens") is not None:
        if report["observed_output_tokens"] != sum(row["output_tokens"] for row in successful_rows):
            issues.append(f"{key}: observed_output_tokens differs from POST records")
    if "failure_rate" in report:
        require(finite_number(report["failure_rate"]) and report["failure_rate"] <= 1, f"{key}: invalid failure_rate")
        require(abs(report["failure_rate"] - (report["request_count"] - report["success"]) / report["request_count"]) <= 1e-9, f"{key}: failure_rate differs from POST records")
    if report["status"] != "PASS":
        issues.append(f"{key}: runner recorded a failed workload")
    if report["success"] != report["request_count"]:
        issues.append(f"{key}: one or more POSTs failed")
    for field in ("output_tps", "ttft_p95_s", "tpot_p95_s", "e2e_p95_s"):
        if not finite_number(report.get(field)):
            issues.append(f"{key}: {field} unavailable")
    if not finite_number(report.get("observed_output_tokens")):
        issues.append(f"{key}: observed_output_tokens unavailable")
    if (
        report.get("deadline_s") is None
        and report.get("request_timeout_s") is None
        and report.get("timeout_s") is None
        and default_deadline is None
    ):
        issues.append(f"{key}: deadline unavailable")
    if (
        report.get("drain_s") is None
        and report.get("drain_complete") is None
        and report.get("drained") is None
        and default_drain is None
    ):
        issues.append(f"{key}: drain completion evidence unavailable")
    return key


def _workload_contract(greedy, probabilistic):
    reports = {}
    for arm_name, arm in (("greedy", greedy), ("probabilistic", probabilistic)):
        require(isinstance(arm.get("reports"), list) and arm["reports"], f"{arm_name}: reports must be a nonempty array")
        issues = []
        if arm.get("mode") is None:
            issues.append("arm mode unavailable")
        else:
            require(arm["mode"] == arm_name, f"{arm_name}: top-level mode differs from the arm")
        if arm.get("status") is None:
            issues.append("arm runner status unavailable")
        else:
            require(arm["status"] in ("PASS", "FAIL"), f"{arm_name}: invalid top-level status")
        if "errors" in arm:
            require(isinstance(arm["errors"], list) and all(isinstance(value, str) and value for value in arm["errors"]), f"{arm_name}: errors must be an array of messages")
        seen = {}
        default_deadline = arm.get("deadline_s", arm.get("request_timeout_s"))
        default_drain = arm.get("drain_s", arm.get("drain_complete", arm.get("drained")))
        for index, report in enumerate(arm["reports"]):
            key = _validate_report(
                report,
                index,
                issues,
                default_deadline=default_deadline,
                default_drain=default_drain,
            )
            require(key not in seen, f"{arm_name}: duplicate workload report {key}")
            seen[key] = report
        reports[arm_name] = (seen, issues)
    left, right = reports["greedy"][0], reports["probabilistic"][0]
    require(set(left) == set(right), "fixed workload cases or rounds differ across arms")
    compare_fields = ("case", "round", "concurrency", "temperature", "input_tokens_each", "output_tokens_each", "workload_sha256")
    for key in sorted(left):
        for field in compare_fields:
            require(left[key][field] == right[key][field], f"workload contract differs across arms at {key}: {field}")
    return reports


def _latency_summary(reports):
    all_values = {field: [] for field in ("ttft_p95_s", "tpot_p95_s", "e2e_p95_s")}
    high_values = {field: [] for field in all_values}
    for report in reports:
        for field in all_values:
            value = report.get(field)
            if finite_number(value):
                all_values[field].append(value)
                if report["output_tokens_each"] >= OSL_THRESHOLD:
                    high_values[field].append(value)
    def summarize(values):
        return {"n": len(values), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95)} if values else None
    return {
        "all": {field: summarize(values) for field, values in all_values.items()},
        "osl_ge_2405": {field: summarize(values) for field, values in high_values.items()},
    }


def _arm_summary(arm, mode, report_data):
    seen, issues = report_data
    report_list = [seen[key] for key in sorted(seen)]
    draft_metrics = _draft_metrics(arm, report_list, issues)
    reported_output = [report["output_tps"] for report in report_list if finite_number(report.get("output_tps"))]
    arm_status = "FAIL" if issues or arm.get("status") == "FAIL" else "PASS"
    total_output = arm.get("total_output_tps")
    total_output_source = "explicit_total_output_tps"
    if total_output is None and reported_output:
        total_output = percentile(reported_output, 0.5)
        total_output_source = "reported_output_tps_median"
    if total_output is not None and not finite_number(total_output):
        issues.append("total_output_tps is invalid")
        total_output = None
        total_output_source = None
    if total_output is None:
        issues.append("total output TPS unavailable")
    failed_rows = []
    for report in report_list:
        for row in report["requests"]:
            if row["status"] != "PASS":
                failed_rows.append({"case": report["case"], "round": report["round"], **row})
    latency = _latency_summary(report_list)
    errors = list(arm.get("errors", [])) if isinstance(arm.get("errors", []), list) else []
    for report in report_list:
        if isinstance(report.get("errors"), list):
            errors.extend(report["errors"])
    return {
        "status": arm_status,
        "acceptance_issues": issues,
        "errors": errors,
        "mode": mode,
        "reports": report_list,
        "request_count": sum(report["request_count"] for report in report_list),
        "successful_requests": sum(report["success"] for report in report_list),
        "success": sum(report["success"] for report in report_list),
        "failed_requests": sum(report["request_count"] - report["success"] for report in report_list),
        "failed_rows": failed_rows,
        "output_tps": {
            "reported_by_workload": [
                {"case": report["case"], "round": report["round"], "value": report.get("output_tps")}
                for report in report_list
            ],
            "median": percentile(reported_output, 0.5) if reported_output else None,
            "total": total_output,
            "total_source": total_output_source,
        },
        "total_output_tps": total_output,
        "output_tps_total": total_output,
        "latency_p95_s": latency,
        "deadline_s": [
            report.get(
                "deadline_s",
                report.get(
                    "request_timeout_s",
                    report.get("timeout_s", arm.get("deadline_s", arm.get("request_timeout_s"))),
                ),
            )
            for report in report_list
        ],
        "deadline": [
            report.get(
                "deadline_s",
                report.get(
                    "request_timeout_s",
                    report.get("timeout_s", arm.get("deadline_s", arm.get("request_timeout_s"))),
                ),
            )
            for report in report_list
        ],
        "drain": [
            report.get(
                "drain_s",
                report.get(
                    "drain_complete",
                    report.get("drained", arm.get("drain_s", arm.get("drain_complete", arm.get("drained")))),
                ),
            )
            for report in report_list
        ],
        "draft_metrics": draft_metrics,
        "draft_counters": draft_metrics,
    }


def _comparison(greedy, probabilistic, launch, workload):
    arms = {
        "greedy": _arm_summary(greedy, "greedy", workload["greedy"]),
        "probabilistic": _arm_summary(probabilistic, "probabilistic", workload["probabilistic"]),
    }
    for mode, arm in arms.items():
        receipt = (greedy if mode == "greedy" else probabilistic).get("launch_receipt", {})
        arm["launch"] = {
            "image_id": receipt.get("image_id"),
            "container_id": receipt.get("container_id"),
            "draft_sample_method": receipt.get("dspark_draft_sample_method"),
        }
    keys = sorted(workload["greedy"][0])
    by_workload = []
    for key in keys:
        left = workload["greedy"][0][key]
        right = workload["probabilistic"][0][key]
        left_tps, right_tps = left.get("output_tps"), right.get("output_tps")
        by_workload.append({
            "case": key[0],
            "round": key[1],
            "greedy_output_tps": left_tps,
            "probabilistic_output_tps": right_tps,
            "probabilistic_over_greedy_output_tps": right_tps / left_tps
            if finite_number(left_tps, minimum=1e-12) and finite_number(right_tps)
            else None,
        })
    valid = all(arm["status"] == "PASS" for arm in arms.values())
    total_left, total_right = arms["greedy"]["total_output_tps"], arms["probabilistic"]["total_output_tps"]
    comparison_status = "VALID" if valid and finite_number(total_left, minimum=1e-12) and finite_number(total_right) else "WITHHELD"
    return {
        "kind": "dspark_replay_comparison",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if comparison_status == "VALID" else "FAIL",
        "mode": "fixed",
        "workload_kind": "fixed",
        "launch": launch,
        "workload": {
            "cases": [
                {
                    "case": key[0],
                    "round": key[1],
                    **{field: workload["greedy"][0][key][field] for field in ("concurrency", "temperature", "input_tokens_each", "output_tokens_each", "workload_sha256")},
                }
                for key in keys
            ]
        },
        "arms": arms,
        "performance_comparison": {
            "status": comparison_status,
            "throughput_basis": "reported total_output_tps from the fixed replay evidence",
            "probabilistic_over_greedy_output_tps": total_right / total_left
            if comparison_status == "VALID"
            else None,
            "by_workload": by_workload,
        },
    }


def compare(greedy, probabilistic):
    require(isinstance(greedy, dict) and isinstance(probabilistic, dict), "replay results must be JSON objects")
    launch = compare_launches(greedy, probabilistic)
    workload = _workload_contract(greedy, probabilistic)
    return _comparison(greedy, probabilistic, launch, workload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("greedy", type=Path, help="greedy DSpark replay result JSON")
    parser.add_argument("probabilistic", type=Path, help="probabilistic DSpark replay result JSON")
    parser.add_argument("--output", type=Path, help="write the comparison JSON to this path")
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            require(args.output.resolve() not in {args.greedy.resolve(), args.probabilistic.resolve()}, "--output must differ from both replay input files")
        report = compare(json.loads(args.greedy.read_text()), json.loads(args.probabilistic.read_text()))
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            sys.stdout.write(text)
        else:
            args.output.write_text(text)
    except (OSError, json.JSONDecodeError, ContractError, KeyError, TypeError, AttributeError) as error:
        print(f"dspark replay comparison: {error}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
