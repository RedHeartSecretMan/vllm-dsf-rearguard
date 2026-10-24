"""Measurement and service evidence for the isolated port-6688 profile sweep.

ServiceObserver preserves complete Prometheus text, GPU snapshots and Docker
logs. Its summaries describe observed counters and matching log lines; absent
metrics are unavailable, never inferred zeros. All timings use bench_common.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
import platform
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from scripts import bench_common

TEST_PORT = 6688
GPU_FIELDS = "index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,driver_version"
_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(?:[^"{}]|"(?:\\.|[^"\\])*")*\})?\s+([^\s]+)(?:\s+[^\s]+)?\s*$')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"(?:,|$)')
_TOKEN_COUNTERS = {
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm_prompt_tokens_total": "prompt_tokens",
    "vllm_generation_tokens_total": "generation_tokens",
}


class MetricsUnavailableError(ValueError):
    """The exposed metric labels cannot prove that both DP engines ran."""


def require_test_endpoint(base_url: str) -> None:
    url = urlsplit(base_url)
    if (url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1")
            or url.port != TEST_PORT or url.path not in ("", "/")
            or url.query or url.fragment or url.username or url.password):
        raise ValueError(f"profile measurements require localhost test port {TEST_PORT}")


def _cell_id(cell: dict[str, Any]) -> str:
    identity = cell.get("cell_id") or cell.get("id")
    if not isinstance(identity, str) or not identity:
        raise ValueError("measurement cell requires cell_id")
    return identity


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _write_gzip(path: Path, value: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output:
        json.dump(_json_safe(value), output, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def _labels(text: str) -> dict[str, str]:
    if not text:
        return {}
    body = text[1:-1]
    result = {}
    position = 0
    while position < len(body):
        match = _LABEL.match(body, position)
        if match is None:
            raise ValueError(f"invalid Prometheus labels: {text}")
        key, value = match.groups()
        if key in result:
            raise ValueError(f"duplicate Prometheus label: {key}")
        result[key] = json.loads('"' + value + '"')
        position = match.end()
    return result


def _sample_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    fields = ",".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in sorted(labels.items()))
    return f"{name}{{{fields}}}"


def parse_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus samples, preserving every label in the canonical key."""
    samples = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # OpenMetrics exemplars follow a whitespace-separated '#'.
        line = re.split(r"\s+#\s+", line, maxsplit=1)[0]
        match = _SAMPLE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid Prometheus sample: {line[:160]}")
        name, raw_labels, raw_value = match.groups()
        key = _sample_key(name, _labels(raw_labels or ""))
        if key in samples:
            raise ValueError(f"duplicate Prometheus sample: {key}")
        samples[key] = float(raw_value)
    if not samples:
        raise MetricsUnavailableError("/metrics contains no numeric samples")
    return samples


def _split_key(key: str) -> tuple[str, dict[str, str]]:
    name, separator, rest = key.partition("{")
    return name, _labels("{" + rest) if separator else {}


def engine_token_totals(samples: dict[str, float]) -> dict[str, dict[str, float]]:
    """Require prompt and generation counters explicitly labelled by two engines."""
    totals: dict[str, dict[str, float]] = {}
    for key, value in samples.items():
        name, labels = _split_key(key)
        kind = _TOKEN_COUNTERS.get(name)
        if kind is None:
            continue
        engine = labels.get("engine") or labels.get("engine_id")
        if engine is None:
            raise MetricsUnavailableError(f"token counter lacks engine/engine_id label: {key}")
        if not math.isfinite(value) or value < 0:
            raise MetricsUnavailableError(f"invalid token counter: {key}={value}")
        entry = totals.setdefault(engine, {})
        if kind in entry:
            raise MetricsUnavailableError(f"ambiguous {kind} counters for engine {engine}; inspect raw /metrics")
        entry[kind] = value
    if len(totals) != 2 or any(set(entry) != {"prompt_tokens", "generation_tokens"} for entry in totals.values()):
        raise MetricsUnavailableError(
            f"expected two engine-labelled prompt/generation counter pairs, found {totals}; inspect metrics.jsonl.gz"
        )
    return totals


def _run(command: list[str], timeout: float = 20) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=timeout)


class ServiceObserver:
    """Capture one running container without starting, stopping or modifying it."""

    def __init__(self, base_url: str, container_id: str, output_dir: Path, interval: float = 2):
        require_test_endpoint(base_url)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", container_id):
            raise ValueError("invalid container identity")
        if interval <= 0:
            raise ValueError("observer interval must be positive")
        self.base_url, self.container_id = base_url, container_id
        self.output_dir, self.interval = Path(output_dir), interval
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._metrics_file = self._gpu_file = self._log_file = None
        self._log_process: subprocess.Popen | None = None
        self._first: dict[str, float] = {}
        self._latest: dict[str, float] = {}
        self._peaks: dict[str, float] = {}
        self._resets: set[str] = set()
        self._errors: list[dict[str, str]] = []
        self._sample_count = 0
        self._gpu_count = 0
        self._before: dict[str, Any] | None = None
        self._after: dict[str, Any] | None = None
        self._started_at: str | None = None
        self._finished_at: str | None = None

    def _inspect(self) -> dict[str, Any]:
        inspected = json.loads(_run(["docker", "inspect", self.container_id]))[0]
        ports = inspected.get("NetworkSettings", {}).get("Ports", {}).get("8000/tcp") or []
        if not inspected.get("State", {}).get("Running") or not any(
            binding.get("HostIp") == "127.0.0.1" and binding.get("HostPort") == str(TEST_PORT) for binding in ports
        ):
            raise ValueError("observed container must be running and publish 127.0.0.1:6688:8000")
        return {
            "observed_at": _now(), "docker_inspect": inspected,
            "identity": {"container_id": inspected["Id"], "image_id": inspected["Image"],
                         "started_at": inspected["State"]["StartedAt"], "command": inspected["Config"]["Cmd"]},
            "platform": dict(platform.uname()._asdict()), "cpu_count": os.cpu_count(),
            "nvidia_smi": _run(["nvidia-smi", "-q"]),
        }

    def _record_error(self, operation: str, error: Exception) -> None:
        with self._lock:
            self._errors.append({"time": _now(), "operation": operation, "error": f"{type(error).__name__}: {error}"})

    def snapshot(self) -> dict[str, float]:
        """Fetch and archive a fresh metrics sample; return canonical key -> value."""
        with self._lock:
            if self._metrics_file is None:
                raise RuntimeError("ServiceObserver must be entered before sampling")
            with bench_common.request(self.base_url, "/metrics", timeout=15) as response:
                raw = response.read().decode("utf-8")
            # Preserve the original even if parsing fails.
            self._metrics_file.write(json.dumps({"time": _now(), "monotonic": time.monotonic(), "text": raw}) + "\n")
            self._metrics_file.flush()
            values = parse_metrics(raw)
            for key, value in values.items():
                if math.isfinite(value):
                    if key.endswith("_total") or "_total{" in key:
                        previous = self._latest.get(key)
                        if previous is not None and value < previous:
                            self._resets.add(key)
                    self._first.setdefault(key, value)
                    self._peaks[key] = max(self._peaks.get(key, value), value)
            self._latest = values
            self._sample_count += 1
            return dict(values)

    def dp_totals(self) -> dict[str, dict[str, float]]:
        return engine_token_totals(self.snapshot())

    def _gpu_snapshot(self) -> None:
        csv = _run([
            "nvidia-smi", f"--query-gpu={GPU_FIELDS}",
            "--format=csv,noheader,nounits",
        ])
        with self._lock:
            self._gpu_file.write(json.dumps({"time": _now(), "monotonic": time.monotonic(), "fields": GPU_FIELDS.split(","), "csv": csv}) + "\n")
            self._gpu_file.flush()
            self._gpu_count += 1

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval):
            for name, operation in (("metrics", self.snapshot), ("nvidia-smi", self._gpu_snapshot)):
                try:
                    operation()
                except Exception as error:
                    self._record_error(name, error)

    def __enter__(self) -> ServiceObserver:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any((self.output_dir / name).exists() for name in (
            "environment-before.json", "metrics.jsonl.gz", "gpu.jsonl.gz", "docker.log"
        )):
            raise FileExistsError("observer evidence already exists; use a fresh directory")
        self._started_at = _now()
        self._before = self._inspect()
        _write_json(self.output_dir / "environment-before.json", self._before)
        try:
            self._metrics_file = gzip.open(self.output_dir / "metrics.jsonl.gz", "wt", encoding="utf-8")
            self._gpu_file = gzip.open(self.output_dir / "gpu.jsonl.gz", "wt", encoding="utf-8")
            self._log_file = (self.output_dir / "docker.log").open("wb")
            self._log_process = subprocess.Popen(
                ["docker", "logs", "--follow", "--timestamps", self.container_id],
                stdout=self._log_file, stderr=subprocess.STDOUT,
            )
            self.snapshot()
            self._gpu_snapshot()
            self._thread = threading.Thread(target=self._sample_loop, name="profile-service-observer", daemon=True)
            self._thread.start()
        except Exception:
            self._close_handles()
            raise
        return self

    def _close_handles(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=45)
            if self._thread.is_alive():
                raise RuntimeError("observer sampler did not stop before artifact close")
        if self._log_process is not None:
            if self._log_process.poll() is None:
                self._log_process.terminate()
                try:
                    self._log_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._log_process.kill()
                    self._log_process.wait(timeout=5)
            elif self._finished_at is None:
                self._record_error("docker logs", RuntimeError(f"log follower exited early: {self._log_process.returncode}"))
        for name in ("_metrics_file", "_gpu_file", "_log_file"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=45)
        for name, operation in (("final metrics", self.snapshot), ("final nvidia-smi", self._gpu_snapshot)):
            try:
                operation()
            except Exception as error:
                self._record_error(name, error)
        try:
            self._after = self._inspect()
            _write_json(self.output_dir / "environment-after.json", self._after)
            if self._after["identity"] != self._before["identity"]:
                self._record_error("identity", ValueError("container image/command/start time changed during measurement"))
        except Exception as error:
            self._record_error("final identity", error)
        self._close_handles()
        self._finished_at = _now()
        _write_json(self.output_dir / "observer-summary.json", self.summary())
        return False

    def summary(self) -> dict[str, Any]:
        with self._lock:
            latest, first, peaks = dict(self._latest), dict(self._first), dict(self._peaks)
            errors, resets = list(self._errors), sorted(self._resets)
        counters = {}
        for key, value in latest.items():
            if re.search(r"preempt|recomput|oom|error", key, re.I) and (key.endswith("_total") or "_total{" in key):
                initial = first.get(key)
                counters[key] = {
                    "first": initial, "last": value if math.isfinite(value) else None,
                    "delta": value - initial if initial is not None and math.isfinite(value) and key not in resets else None,
                }
        peak_samples = {key: value for key, value in peaks.items()
                        if re.search(r"cache_usage|requests_(?:running|waiting|swapped)|queue", key)}
        log_path = self.output_dir / "docker.log"
        lines = log_path.read_text(errors="replace").splitlines() if log_path.exists() else []
        kv_lines = [line for line in lines if re.search(r"KV cache (?:memory|size)|Available KV cache|Maximum concurrency", line, re.I)]
        error_lines = [line for line in lines if re.search(r"\bERROR\b|\bTraceback\b|OutOfMemoryError|out of memory|\bOOM\b", line, re.I)]
        scheduling_lines = [line for line in lines if re.search(r"preempt|recomput", line, re.I)]
        return {
            "status": "FAIL" if errors or resets else "PASS", "status_scope": "observer acquisition and container identity",
            "base_url": self.base_url,
            "container_id": self.container_id, "started_at": self._started_at, "finished_at": self._finished_at,
            "metrics_samples": self._sample_count, "gpu_samples": self._gpu_count,
            "identity_before": self._before["identity"] if self._before else None,
            "identity_after": self._after["identity"] if self._after else None,
            "observed_counter_changes": counters, "counter_resets": resets,
            "observed_gauge_peaks": peak_samples, "kv_log_lines": kv_lines,
            "error_log_lines": error_lines, "preemption_recompute_log_lines": scheduling_lines,
            "oom_log_lines": [line for line in error_lines if re.search(r"OutOfMemoryError|out of memory|\bOOM\b", line, re.I)],
            "observer_errors": errors,
            "availability": {name: any(re.search(pattern, key, re.I) for key in counters)
                             for name, pattern in (("preemption_counter", "preempt"), ("recompute_counter", "recomput"),
                                                   ("oom_counter", "oom"), ("error_counter", "error"))},
            "note": "Log entries are matching lines, not counts of distinct events. Absent counters are unavailable. Gauge peaks are sampled, not continuous maxima.",
            "artifacts": {"metrics": "metrics.jsonl.gz", "gpu": "gpu.jsonl.gz", "logs": "docker.log",
                          "before": "environment-before.json", "after": "environment-after.json"},
        }


def measure(base: str, bodies: list[dict[str, Any]], cell: dict[str, Any], out: Path, timeout: float = 3600) -> dict[str, Any]:
    """Measure one burst and preserve payloads plus all per-request parsed metrics."""
    require_test_endpoint(base)
    cell_id = _cell_id(cell)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    _write_gzip(output / "request-payloads.json.gz", bodies)
    expected = {"expected_output": cell["osl"]}
    expected_by_index = cell.get("expected_input_by_index")
    if expected_by_index is not None and len(expected_by_index) != len(bodies):
        raise ValueError("expected_input_by_index must contain one token count per request")
    expected_input = cell.get("actual_expected_input") or cell.get("isl")
    if expected_by_index is None and expected_input is not None:
        expected.update(expected_input=expected_input, input_tolerance=cell.get("input_tolerance", 0.0))
    if not bodies:
        raise ValueError("measurement needs at least one body")
    concurrency = int(cell["concurrency"])
    if not 1 <= concurrency <= len(bodies):
        raise ValueError("concurrency must be positive and no larger than the request count")
    rows, elapsed = bench_common.run_burst(base, bodies, concurrency, timeout, **expected)
    if expected_by_index is not None:
        for index, row in enumerate(rows):
            if row["status"] == "PASS" and row["input_tokens"] != expected_by_index[index]:
                row.update(status="FAIL", error=f"request {index} input tokens {row['input_tokens']} != expected {expected_by_index[index]}")
    report = bench_common.summarize(rows, elapsed, cell_id=cell_id, case_id=cell.get("case_id", cell_id), concurrency=concurrency, osl=cell["osl"])
    report["payload_sha256"] = bench_common.workload_digest(bodies)
    report["cache_verified"] = bench_common.verify_cache(rows, "cold")
    if not report["cache_verified"]:
        report["status"] = "FAIL"
        report["errors"].append("cold KV state requires cached_tokens == 0 on every successful request")
    good = [row for row in rows if row["status"] == "PASS"]
    for field in ("e2e", "ttft", "tpot"):
        values = [row[field + "_s"] for row in good if row.get(field + "_s") is not None]
        for quantile in (50, 95, 99):
            report[f"{field}_p{quantile}_s"] = bench_common.percentile(values, quantile / 100)
    for row in good:
        if row.get("e2e_s", 0) > 0:
            row["input_tps"] = row["input_tokens"] / row["e2e_s"]
            row["output_tps"] = row["output_tokens"] / row["e2e_s"]
            row["total_tps"] = (row["input_tokens"] + row["output_tokens"]) / row["e2e_s"]
    _write_gzip(output / "request-metrics.json.gz", rows)
    report["artifacts"] = {"payloads": "request-payloads.json.gz", "request_metrics": "request-metrics.json.gz"}
    report["response_evidence"] = "bench_common parsed SSE timing/usage/output digests; original SSE bytes are not retained"
    _write_json(output / "report.json", report)
    return report


def warmup(
    base: str, bodies_factory: Callable[[str, int], dict[str, Any]], cell: dict[str, Any],
    observer: ServiceObserver, out: Path, timeout: float = 3600,
) -> dict[str, Any]:
    """Warm the original input shape twice and prove activity on both DP engines.

    The factory receives (round_label, request_index); it must supply a distinct
    cache_salt and max_tokens matching warmup_osl. Defaults are C2 and OSL<=128.
    A separate global C256 warmup is the controller's responsibility.
    """
    require_test_endpoint(base)
    cell_id = _cell_id(cell)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    rounds = int(cell.get("warmup_rounds", 2))
    concurrency = int(cell.get("warmup_concurrency", 2))
    osl = int(cell.get("warmup_osl", min(cell["osl"], 128)))
    if rounds < 2 or not 2 <= concurrency <= 256 or not 1 <= osl <= cell["osl"]:
        raise ValueError("warmup requires >=2 rounds, concurrency 2..256 and OSL in 1..cell OSL")
    warm_cell = {**cell, "osl": osl, "concurrency": concurrency}
    if cell.get("expected_input_by_index"):
        expected_inputs = cell["expected_input_by_index"]
        warm_cell["expected_input_by_index"] = [expected_inputs[index % len(expected_inputs)] for index in range(concurrency)]
    report: dict[str, Any] = {"status": "FAIL", "cell_id": cell_id, "case_id": cell.get("case_id", cell_id), "concurrency": concurrency,
                              "osl": osl, "rounds": [], "errors": []}
    seen_salts: set[str] = set()
    first_totals = None
    try:
        for index in range(rounds):
            label = f"{cell_id}-warmup-{index + 1}"
            before_samples = observer.snapshot()
            before = engine_token_totals(before_samples)
            if first_totals is None:
                first_totals = before
            bodies = [copy.deepcopy(bodies_factory(label, item)) for item in range(concurrency)]
            for body in bodies:
                salt = body.get("cache_salt")
                if not isinstance(salt, str) or not salt or salt in seen_salts:
                    raise ValueError("warmup factory must provide a unique nonempty cache_salt for every request")
                if body.get("max_tokens") != osl:
                    raise ValueError(f"warmup factory max_tokens must equal warmup_osl={osl}")
                seen_salts.add(salt)
            result = measure(base, bodies, warm_cell, output / f"round-{index + 1}", timeout)
            after_samples = observer.snapshot()
            after = engine_token_totals(after_samples)
            deltas = {engine: {key: after.get(engine, {}).get(key, 0) - value for key, value in values.items()}
                      for engine, values in before.items()}
            _write_gzip(output / f"round-{index + 1}-metrics.json.gz", {"before": before_samples, "after": after_samples})
            dp_verified = set(before) == set(after) and all(value > 0 for values in deltas.values() for value in values.values())
            report["rounds"].append({"round": index + 1, "measurement": result, "dp_before": before,
                                     "dp_after": after, "dp_deltas": deltas, "both_dp_active": dp_verified})
            if result["status"] != "PASS":
                raise ValueError(f"warmup round {index + 1} failed")
        deadline = time.monotonic() + float(cell.get("warmup_metric_timeout", 15))
        while True:
            cumulative = {engine: {key: after.get(engine, {}).get(key, 0) - value for key, value in values.items()}
                          for engine, values in first_totals.items()}
            active = set(first_totals) == set(after) and all(value > 0 for values in cumulative.values() for value in values.values())
            if active or time.monotonic() >= deadline:
                break
            # Engines publish asynchronously; allow their counters to catch up.
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            after_samples = observer.snapshot()
            after = engine_token_totals(after_samples)
        _write_gzip(output / "metrics-final.json.gz", after_samples)
        report.update(dp_before=first_totals, dp_after=after, dp_deltas=cumulative, both_dp_active=active)
        if not active:
            raise ValueError("warmup did not advance both engines' prompt/generation counters")
        report["status"] = "PASS"
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    _write_json(output / "report.json", report)
    return report
