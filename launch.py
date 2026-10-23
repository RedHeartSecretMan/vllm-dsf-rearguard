"""Plan by default. --start requires verified weights and eight idle A100 GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.check_runtime import require_matching_report
from scripts.model_check import MANIFEST, ROOT, require_receipt, sha256

MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
CONTAINER_MODEL = "/models/DeepSeek-V4.1-Flash-1017"
EXPERIMENTS = {
    "decode-metadata": "VLLM_DSV41_DECODE_METADATA_FASTPATH",
    "ragged-reuse": "VLLM_DSV41_REUSE_TOPK_RAGGED",
    "stacked-kv": "VLLM_DSV41_STACKED_DRAFT_KV",
    "bounded-prefill": "VLLM_DSV41_BOUNDED_PREFILL_WORKSPACE",
    "prefill-throttle": "VLLM_DSV41_PREFILL_THROTTLE",
}
KV_TRANSFER_CONNECTORS = {
    "SimpleCPUOffloadConnector",
    "OffloadingConnector",
}
KV_TRANSFER_ROLES = {"kv_both", "kv_producer", "kv_consumer"}


def load_kv_transfer_config(value: str | None) -> dict | None:
    """Load and validate the small public surface of vLLM KV transfer config.

    The pinned image may reject a connector at startup; this function only
    validates the launch contract and deliberately does not claim runtime
    support.  ``@path`` is accepted so a JSON object does not need shell
    escaping in repeatable launch scripts.
    """
    if value is None:
        return None
    raw = Path(value[1:]).read_text() if value.startswith("@") else value
    try:
        config = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("--kv-transfer-config must be a JSON object or @file") from error
    if not isinstance(config, dict):
        raise ValueError("--kv-transfer-config must be a JSON object")
    connector = config.get("kv_connector")
    role = config.get("kv_role")
    if connector not in KV_TRANSFER_CONNECTORS:
        raise ValueError(
            "Unsupported kv connector; choose "
            + ", ".join(sorted(KV_TRANSFER_CONNECTORS))
        )
    if role not in KV_TRANSFER_ROLES:
        raise ValueError("kv_role must be kv_both, kv_producer, or kv_consumer")
    extra = config.get("kv_connector_extra_config", {})
    if not isinstance(extra, dict):
        raise ValueError("kv_connector_extra_config must be an object")
    for key in (
        "num_cpu_blocks",
        "num_cpu_blocks_override",
        "cpu_bytes_to_use",
        "cpu_bytes_to_use_per_rank",
    ):
        if key in extra and (
            not isinstance(extra[key], int)
            or isinstance(extra[key], bool)
            or extra[key] <= 0
        ):
            raise ValueError(f"{key} must be a positive integer")
    if "lazy_offload" in extra and not isinstance(extra["lazy_offload"], bool):
        raise ValueError("lazy_offload must be a boolean")
    return config


def make_plan(args):
    model = args.model.resolve(strict=True)
    if any(c in str(model) for c in ",\n\r"):
        raise ValueError("Model path contains Docker mount delimiters")
    expected = next(
        r
        for r in json.loads(MANIFEST.read_text())["files"]
        if r["path"] == "config.json"
    )
    if sha256(model / "config.json") != expected["sha256"]:
        raise ValueError("config.json is not the pinned 1017 release")
    config = json.loads((model / "config.json").read_text())
    trained_k = config["text_config"]["dspark_block_size"]
    if args.dspark > trained_k and args.dspark % trained_k:
        raise ValueError(
            f"Runtime rejects DSpark k{args.dspark}: trained block size is {trained_k}"
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", args.name):
        raise ValueError("Invalid container name")
    if (
        not 0 < args.memory < 1
        or not 1 <= args.port <= 65535
        or args.max_seqs <= 0
        or args.chunk <= 0
    ):
        raise ValueError("Invalid memory, port, sequence count, or chunk budget")
    tp, dp = (8, 1) if args.parallel == "tp8" else (4, 2)
    if args.max_seqs % dp:
        raise ValueError("Global max-seqs must be divisible by DP")
    per_rank = args.max_seqs // dp
    if args.chunk < per_rank * (args.dspark + 1):
        raise ValueError("Chunk budget is too small for speculative decode")
    # Baseline variables that apply to every launch. The options below add the
    # variables that depend on a launch argument.
    env = json.loads((ROOT / "config/runtime/environment.json").read_text())
    env["VLLM_DSV4_LOGITS_ROW_CHUNK"] = str(args.logits_row_chunk)
    selected = set(args.experimental)
    kv_transfer = load_kv_transfer_config(args.kv_transfer_config)
    if "stacked-kv" in selected and not args.dspark:
        raise ValueError("stacked-kv requires DSpark")
    for experiment, variable in EXPERIMENTS.items():
        env[variable] = "1" if experiment in selected else "0"
    # NVSwitch hosts let NCCL choose its algorithms; "ring" pins them.
    if args.nccl == "ring":
        env.update(NCCL_ALGO="Ring", NCCL_PROTO="Simple")
    if args.kernels == "control":
        env.update(
            VLLM_AMPERE_DENSE_BF16_MIN_TOKENS="0",
            VLLM_SM80_EP8_CUSTOM_AG_RS="0",
            VLLM_SM80_CANDIDATE_MQA="0",
        )
    buckets = sorted({1, 2, 4, 8, 16, 32, 64, per_rank})
    buckets = [n for n in buckets if n <= per_rank]
    # Include target verification (k+1) and draft (k) shapes for each request bucket.
    captures = sorted(
        {
            n * factor
            for n in buckets
            for factor in ({args.dspark, args.dspark + 1} if args.dspark else {1})
        }
    )
    serve = [
        "serve",
        CONTAINER_MODEL,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        str(tp),
        "--data-parallel-size",
        str(dp),
        "--data-parallel-size-local",
        str(dp),
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--max-model-len",
        str(args.context),
        "--max-num-seqs",
        str(per_rank),
        "--max-num-batched-tokens",
        str(args.chunk),
        "--gpu-memory-utilization",
        str(args.memory),
        "--kv-cache-dtype",
        "fp8_ds_mla",
        "--engram-config",
        '{"cpu_offload":true}',
        "--tokenizer-mode",
        "deepseek_v41",
        "--tool-call-parser",
        "deepseek_v41",
        "--reasoning-parser",
        "deepseek_v41",
        "--enable-auto-tool-choice",
        "--enable-prompt-tokens-details",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--generation-config",
        "vllm",
        "--hf-overrides",
        '{"head_dtype":"float32"}',
        "--compilation-config",
        json.dumps(
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": captures,
                "max_cudagraph_capture_size": max(captures),
            }
        ),
        "--enable-eplb",
        "--eplb-config",
        '{"window_size":1000,"step_interval":3000,"num_redundant_experts":0,"use_async":true,"log_balancedness":true,"log_balancedness_interval":500}',
    ]
    if args.dspark:
        serve += [
            "--speculative-config",
            json.dumps(
                {
                    "method": "dspark",
                    "num_speculative_tokens": args.dspark,
                    "draft_sample_method": "greedy",
                    "rejection_sample_method": "standard",
                    "enable_adaptive_verification": False,
                    "use_local_argmax_reduction": True,
                }
            ),
        ]
    if kv_transfer is not None:
        serve += ["--kv-transfer-config", json.dumps(kv_transfer, separators=(",", ":"))]
    if args.text_only:
        serve += ["--language-model-only"]
    else:
        serve += [
            "--limit-mm-per-prompt",
            '{"image":64,"video":{"count":2,"num_frames":32}}',
            "--media-io-kwargs",
            '{"video":{"num_frames":32,"fps":1}}',
            "--disable-chunked-mm-input",
        ]
    if args.eager:
        serve += ["--enforce-eager"]
    command = [
        "docker",
        "run",
        "-d",
        "--pull",
        "never",
        "--init",
        "--name",
        args.name,
        "--label",
        "dsv41-a100.owner=deepseek-v4.1-flash-a100",
        "--gpus",
        "all",
        "--ipc=private",
        "--shm-size",
        "64g",
        "--ulimit",
        "memlock=-1",
        "-p",
        f"127.0.0.1:{args.port}:8000",
        "--mount",
        f"type=bind,src={model},dst={CONTAINER_MODEL},readonly",
        "--mount",
        f"type=volume,src={args.name}-cache,dst=/workspace",
        "--log-opt",
        "max-size=100m",
        "--log-opt",
        "max-file=5",
    ]
    for key, value in sorted(env.items()):
        command += ["--env", f"{key}={value}"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_lock": json.loads((ROOT / "manifests/sources.json").read_text()),
        "experimental": sorted(selected),
        "parallel": {"tp": tp, "dp": dp, "ep": 8, "max_global_seqs": args.max_seqs},
        "dspark": args.dspark,
        "context": args.context,
        "kernels": args.kernels,
        "text_only": args.text_only,
        "gpu_validation": "NOT_RUN",
        "kv_transfer_config": kv_transfer,
        "kv_transfer_config_sha256": (
            hashlib.sha256(
                json.dumps(kv_transfer, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if kv_transfer is not None
            else None
        ),
        "command": command + [args.image] + serve,
    }


def require_idle_gpus():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = [line.rsplit(",", 1) for line in output.strip().splitlines()]
    if len(rows) != 8 or any(
        "A100-SXM4-80GB" not in name or int(used) > 1024 for name, used in rows
    ):
        raise ValueError(
            "Eight idle A100-SXM4-80GB GPUs are required; no service was stopped"
        )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--parallel", choices=("tp8", "tp4dp2"), default="tp4dp2")
    p.add_argument("--dspark", type=int, choices=(0, 5, 7), default=5)
    p.add_argument(
        "--context", type=int, choices=(262144, 524288, 1048576), default=262144
    )
    p.add_argument(
        "--max-seqs", type=int, default=256, help="Global budget across both DP ranks"
    )
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument(
        "--logits-row-chunk",
        type=int,
        choices=(0, 128, 256, 512, 1024, 2048),
        default=0,
        help="Opt-in SM80 prefill logits row chunk; 0 keeps the monolithic path",
    )
    p.add_argument("--memory", type=float, default=0.90)
    p.add_argument(
        "--kv-transfer-config",
        help="JSON object or @file for SimpleCPUOffloadConnector/OffloadingConnector",
    )
    p.add_argument("--kernels", choices=("control", "optimized"), default="optimized")
    p.add_argument("--nccl", choices=("auto", "ring"), default="auto")
    p.add_argument("--text-only", action="store_true")
    p.add_argument(
        "--experimental",
        action="append",
        choices=tuple(EXPERIMENTS),
        default=[],
        help="Opt-in hot path; repeat to combine. Not GPU-accepted defaults.",
    )
    p.add_argument(
        "--eager", action="store_true", help="Correctness control without CUDA graphs"
    )
    p.add_argument("--port", type=int, default=8005)
    p.add_argument("--name", default="deepseek-v41-a100")
    p.add_argument("--image", default="deepseek-v4.1-flash-a100:dev")
    p.add_argument("--receipt", type=Path, default=ROOT / "runtime/model-verified.json")
    p.add_argument("--start", action="store_true")
    return p


def main():
    args = parser().parse_args()
    plan = make_plan(args)
    if not args.start:
        print(json.dumps(plan, indent=2))
        return
    require_idle_gpus()
    # Pin the actual launch to its local immutable image ID.
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", args.image])
    )[0]
    runtime_report = json.loads(
        subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--runtime=runc",
                "--network",
                "none",
                "--env",
                "CUDA_VISIBLE_DEVICES=",
                "--env",
                "NVIDIA_VISIBLE_DEVICES=void",
                "--entrypoint",
                "python3",
                image["Id"],
                "/opt/dsv41-a100/check_runtime.py",
            ],
        )
    )
    plan["source_manifest_sha256"] = require_matching_report(runtime_report, ROOT)
    require_receipt(args.model, args.receipt)
    require_idle_gpus()
    plan["command"][plan["command"].index(args.image)] = image["Id"]
    plan["image_id"] = image["Id"]
    destination = (
        ROOT / "runtime" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    )
    destination.mkdir(parents=True)
    (destination / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    plan["container_id"] = subprocess.check_output(plan["command"], text=True).strip()
    (destination / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        "Launch requested; health and GPU acceptance are still required. Evidence:",
        destination,
    )


if __name__ == "__main__":
    main()
