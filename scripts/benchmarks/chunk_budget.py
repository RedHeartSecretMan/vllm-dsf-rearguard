"""Compare 32K/16K budgets using one frozen, cold-KV workload across services.

Prepare once on a serving endpoint (one tokenizer call), then reuse the same
JSON for all profiles. Each request has a distinct prefix and cache salt;
repeats have different requests but each profile receives identical bytes.
Multimodal encoder cache state is not controlled by the KV cache salt.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.acceptance.media import load_fixtures, prepare_fixtures, sha256
from scripts.acceptance.visual import command_option, media_part
from scripts.bench_common import (
    DEFAULT_BASE_URL,
    MODEL,
    chat_body,
    completion_body,
    load_config,
    require_model,
    run_burst,
    select_cases,
    summarize,
    token_pool,
    verify_cache,
    verify_endpoint,
    workload_digest,
    write_report,
)
from scripts.benchmarks.bench_ab import variant_plan


def case_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config("ab")
    (case,) = select_cases(config, "chunk_budget")
    return config, case


def prepare_media(directory: Path) -> dict[str, Any]:
    """Reuse acceptance fixtures and upscale the long video before frame sampling."""
    fixtures = (
        load_fixtures(directory)
        if (directory / "manifest.json").exists()
        else prepare_fixtures(directory)
    )
    source = fixtures["videos"]["long"]
    target = directory.resolve() / "long-1200.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", source["path"],
        "-vf", "scale=1200:1200", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ]
    subprocess.run(command, check=True)
    video_digest = sha256(target)
    video = {
        **source, "path": str(target), "sha256": video_digest, "video_id": "vid_" + video_digest[:16],
        "width": 1200, "height": 1200, "sampled_frames": 32,
        "requested_fps": 1, "generation_command": command,
        "source_sha256": source["sha256"],
    }
    return {"images": fixtures["images"], "video": video}


def build_workload(
    case: dict[str, Any], pool: list[int], media: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Build all payloads offline after the single shared token-pool query."""
    if not pool:
        raise ValueError("token pool must not be empty")
    config = case["workload"]
    seed = config["seed"]
    image_parts = [media_part(Path(media["images"][str(n)]["path"]), "image") for n in (1, 2, 3, 1)]
    video = media_part(Path(media["video"]["path"]), "video")
    video["video_url"]["video_id"] = media["video"]["video_id"]

    def payload(cell: dict[str, Any], label: str, index: int) -> dict[str, Any]:
        salt = f"chunk-budget:{run_id}:{cell['id']}:{label}:{index}"
        # Seed from the stable request identity, not Python's randomized hash().
        rng = random.Random(workload_digest([seed, salt]))
        if cell["kind"] == "text":
            prompt = [rng.choice(pool) for _ in range(cell["isl"])]
            body = completion_body(prompt, cell["osl"], seed)
        else:
            content = [{"type": "text", "text": (
                f"Request {salt}. Describe the visible digits and background colors in order. "
                "Explain each stage in detail."
            )}]
            content += copy.deepcopy(image_parts if cell["kind"] == "image" else [video])
            body = chat_body(content, cell["osl"], seed)
            body["ignore_eos"] = True
        body["cache_salt"] = salt
        return body

    cells = []
    for cell in config["cells"]:
        repeats = []
        for repeat in range(config["repeats"]):
            bodies = [payload(cell, f"repeat-{repeat + 1}", i) for i in range(cell["concurrency"])]
            repeats.append({"repeat": repeat + 1, "bodies": bodies, "payload_sha256": workload_digest(bodies)})
        # Warm each shape once. The warm-up salt is separate from all measurements.
        warmup = payload(cell, "warmup", 0)
        cells.append({**cell, "warmup": warmup, "repeats": repeats})
    artifact = {
        "schema_version": 1, "case": case["id"], "model": MODEL,
        "run_id": run_id, "seed": seed, "media": media, "cells": cells,
        "cache_policy": "cold KV; distinct request salts; cached_tokens must be zero",
        "multimodal_encoder_cache": "not controlled; warmup precedes measurement",
    }
    artifact["sha256"] = workload_digest(artifact)
    return artifact


def load_workload(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text())
    recorded = artifact.pop("sha256")
    if workload_digest(artifact) != recorded:
        raise ValueError("workload JSON digest mismatch")
    artifact["sha256"] = recorded
    if artifact.get("schema_version") != 1 or artifact.get("case") != "chunk_budget" or artifact.get("model") != MODEL:
        raise ValueError("unsupported workload artifact")
    salts = []
    for cell in artifact["cells"]:
        salts.append(cell["warmup"]["cache_salt"])
        for repeat in cell["repeats"]:
            bodies = repeat["bodies"]
            if len(bodies) != cell["concurrency"] or workload_digest(bodies) != repeat["payload_sha256"]:
                raise ValueError("workload request count or payload digest mismatch")
            salts.extend(body["cache_salt"] for body in bodies)
    if len(set(salts)) != len(salts):
        raise ValueError("workload reuses a cache salt")
    return artifact


def verify_profile(profile: dict[str, Any], receipt: dict[str, Any]) -> None:
    command = receipt["command"]
    extra = profile["extra_launch"]
    chunk = extra[extra.index("--chunk") + 1]
    expected = {
        "--tensor-parallel-size": "4", "--data-parallel-size": "2",
        "--data-parallel-size-local": "2", "--max-model-len": "1048576",
        "--max-num-seqs": "128", "--max-num-batched-tokens": chunk,
        "--gpu-memory-utilization": "0.9", "--api-server-count": "1",
        "--kv-cache-dtype": "fp8_ds_mla",
    }
    for flag, value in expected.items():
        actual = command_option(command, flag)
        # The launcher renders 0.90 as 0.9; accept either numeric spelling.
        matches = float(actual) == float(value) if flag == "--gpu-memory-utilization" else actual == value
        if not matches:
            raise ValueError(f"profile requires {flag} {value}; got {actual}")
    mm_flag = "--disable-chunked-mm-input" if profile["id"] == "32k_mm_off" else "--no-disable-chunked-mm-input"
    opposite = "--no-disable-chunked-mm-input" if mm_flag == "--disable-chunked-mm-input" else "--disable-chunked-mm-input"
    if mm_flag not in command or opposite in command:
        raise ValueError(f"profile requires {mm_flag}")
    for flag in ("--enable-eplb", "--enable-prefix-caching", "--enable-chunked-prefill", "--enable-prompt-tokens-details"):
        if flag not in command:
            raise ValueError(f"profile requires {flag}")
    speculative = json.loads(command_option(command, "--speculative-config"))
    if speculative.get("method") != "dspark" or speculative.get("num_speculative_tokens") != 5:
        raise ValueError("profile requires DSpark 5")
    limits = json.loads(command_option(command, "--limit-mm-per-prompt"))
    policy = json.loads(command_option(command, "--media-io-kwargs"))
    if limits != {"image": 64, "video": {"count": 2, "num_frames": 32}} or policy != {"video": {"num_frames": 32, "fps": 1}}:
        raise ValueError("profile requires 64 images / 2 videos, 32 frames and 1 FPS")
    env = dict(command[index + 1].split("=", 1) for index, item in enumerate(command[:-1]) if item == "--env")
    required_env = {
        "OMP_NUM_THREADS": "8", "VLLM_ENABLE_RESPONSES_API_STORE": "1",
        "VLLM_DSV41_STACKED_DRAFT_KV": "1", "VLLM_DSV41_BOUNDED_PREFILL_WORKSPACE": "1",
        "VLLM_DSV41_PREFILL_THROTTLE": "0", "VLLM_SM80_EP8_CUSTOM_AG_RS": "1",
    }
    for key, value in required_env.items():
        if env.get(key) != value:
            raise ValueError(f"profile requires {key}={value}")


def measure(base: str, cell: dict[str, Any], bodies: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
    expectations = {"expected_output": cell["osl"]}
    if cell["kind"] == "text":
        expectations["expected_input"] = cell["isl"]
    rows, elapsed = run_burst(base, bodies, min(cell["concurrency"], len(bodies)), timeout, **expectations)
    for row in rows:
        if row["status"] == "PASS" and row["e2e_s"] > 0:
            row["input_tps"] = row["input_tokens"] / row["e2e_s"]
            row["output_tps"] = row["output_tokens"] / row["e2e_s"]
            row["total_tps"] = (row["input_tokens"] + row["output_tokens"]) / row["e2e_s"]
        if row["status"] == "PASS" and row["input_tokens"] < cell.get("min_input_tokens", 0):
            row.update(status="FAIL", error=f"video boundary probe needs >= {cell['min_input_tokens']} prompt tokens; got {row['input_tokens']}")
    report = summarize(rows, elapsed, cell=cell["id"], concurrency=min(cell["concurrency"], len(bodies)), osl=cell["osl"])
    report["payload_sha256"] = workload_digest(bodies)
    report["cache_verified"] = verify_cache(rows, "cold")
    if not report["cache_verified"]:
        report["status"] = "FAIL"
        report["errors"].append("cold KV state not verified: every successful request must report cached_tokens == 0")
    return report


def run_profile(
    base: str, profile: dict[str, Any], receipt: dict[str, Any], artifact: dict[str, Any],
    output: Path, timeout: float, cell_ids: list[str] | None = None,
) -> int:
    verify_profile(profile, receipt)
    endpoint = verify_endpoint(base, receipt)
    require_model(base)
    cells = artifact["cells"]
    if cell_ids:
        unknown = set(cell_ids) - {cell["id"] for cell in cells}
        if unknown:
            raise ValueError(f"unknown cells: {sorted(unknown)}")
        cells = [cell for cell in cells if cell["id"] in cell_ids]
    output.mkdir(parents=True, exist_ok=False)
    (output / "launch-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    report: dict[str, Any] = {
        "status": "PASS", "profile": profile["id"], "endpoint": endpoint,
        "workload_sha256": artifact["sha256"], "run_id": artifact["run_id"],
        "started_at": datetime.now(timezone.utc).isoformat(), "reports": [], "warmups": [],
    }
    for cell in cells:
        warmup = measure(base, cell, [cell["warmup"]], timeout)
        report["warmups"].append(warmup)
        (output / f"{cell['id']}-warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
        if warmup["status"] != "PASS":
            report["status"] = "FAIL"
            report["error"] = f"warmup failed for {cell['id']}"
            break
        for repeat in cell["repeats"]:
            result = measure(base, cell, repeat["bodies"], timeout)
            result["repeat"] = repeat["repeat"]
            report["reports"].append(result)
            name = f"{cell['id']}-repeat-{repeat['repeat']}.json"
            (output / name).write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: value for key, value in result.items() if key != "requests"}), flush=True)
            if result["status"] != "PASS":
                report["status"] = "FAIL"
        (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    # A service restart during measurement invalidates the deployment identity.
    final_endpoint = verify_endpoint(base, receipt)
    if final_endpoint != endpoint:
        report.update(status="FAIL", error="container identity or start time changed during benchmark")
    return write_report(output / "summary.json", report, hide=("reports", "warmups"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", "--list", action="store_true", dest="plan")
    parser.add_argument("--prepare-workload", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--fixtures-dir", type=Path)
    parser.add_argument("--run-id", default="chunk-budget-20261024")
    parser.add_argument("--profile")
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cell", action="append")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    config, case = case_config()
    if args.plan:
        print(json.dumps({"profiles": {v["id"]: variant_plan(config, v) for v in case["variants"]}, "workload": case["workload"]}, indent=2))
        return 0
    if args.workload is None:
        parser.error("--workload is required")
    if args.prepare_workload:
        if args.fixtures_dir is None:
            parser.error("--prepare-workload requires --fixtures-dir")
        if args.workload.exists():
            parser.error("workload exists; use a fresh artifact path")
        require_model(args.base_url)
        pool = token_pool(args.base_url)
        media = prepare_media(args.fixtures_dir)
        artifact = build_workload(case, pool, media, args.run_id)
        args.workload.parent.mkdir(parents=True, exist_ok=True)
        args.workload.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
        print(json.dumps({"workload": str(args.workload), "sha256": artifact["sha256"], "cells": len(artifact["cells"])}))
        return 0
    if args.profile is None or args.launch_receipt is None or args.output is None:
        parser.error("execution requires --profile, --launch-receipt and --output")
    profile = next((v for v in case["variants"] if v["id"] == args.profile), None)
    if profile is None:
        parser.error(f"unknown profile; choose from {[v['id'] for v in case['variants']]}")
    return run_profile(args.base_url, profile, json.loads(args.launch_receipt.read_text()), load_workload(args.workload), args.output, args.timeout, args.cell)


if __name__ == "__main__":
    raise SystemExit(main())
