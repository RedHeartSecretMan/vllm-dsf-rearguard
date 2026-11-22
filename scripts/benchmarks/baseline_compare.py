"""Run a fixed basic-performance matrix for one vLLM image.

The process assumes an already-open maintenance window. It only starts and
stops its test container on port 6688. Production port 6789 is untouched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from scripts import bench_common as bench
from scripts.benchmarks import lmcache_mp_probe as probe

BASE_URL = "http://127.0.0.1:6688"
RECEIPT = Path("/tmp/vllm-dsf-dspark-1m-next/results/greedy-launch.json")
CONTROL_IMAGE = "sha256:c96d88d027c229ad0b773b358993a4257ee943d734ad7e438842b31778a68f23"


def wait_health(timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if subprocess.run(
            ["curl", "-fsS", "--max-time", "3", f"{BASE_URL}/health"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError(f"service did not become healthy at {BASE_URL}")


def option(argv: list[str], flag: str, value: str) -> None:
    index = argv.index(flag)
    argv[index + 1] = value


def make_command(image: str, name: str, volume: str, official: bool, draft_method: str) -> list[str]:
    receipt = json.loads(RECEIPT.read_text())
    argv = receipt["command"].copy()
    argv[argv.index(receipt["image_id"])] = image
    option(argv, "--name", name)
    option(argv, "--max-num-seqs", "256")
    speculative = json.loads(argv[argv.index("--speculative-config") + 1])
    speculative["draft_sample_method"] = draft_method
    if draft_method == "probabilistic":
        speculative["use_local_argmax_reduction"] = False
    option(argv, "--speculative-config", json.dumps(speculative, sort_keys=True))
    old_port = next(item for item in argv if item.startswith("127.0.0.1:6688:"))
    argv[argv.index(old_port)] = "127.0.0.1:6688:8000"
    old_volume = next(
        item
        for item in argv
        if item.startswith("type=volume,src=") and item.endswith(",dst=/workspace")
    )
    argv[argv.index(old_volume)] = f"type=volume,src={volume},dst=/workspace"
    if official:
        image_index = argv.index(image)
        if image_index + 1 >= len(argv) or argv[image_index + 1] != "serve":
            argv.insert(image_index + 1, "serve")
        argv[image_index:image_index] = ["--entrypoint", "vllm"]
        argv[argv.index("--ipc=private")] = "--ipc=host"
        shm_index = argv.index("--shm-size")
        del argv[shm_index : shm_index + 2]
        filtered: list[str] = []
        index = 0
        while index < len(argv):
            if argv[index] == "--env" and index + 1 < len(argv):
                env = argv[index + 1]
                if env.split("=", 1)[0].startswith(("VLLM_DSV", "VLLM_SM80", "VLLM_FUSED_STABLE", "VLLM_AMPERE")):
                    index += 2
                    continue
            filtered.append(argv[index])
            index += 1
        argv = filtered
        option(argv, "--kv-cache-dtype", "fp8")
        option(argv, "--engram-config", json.dumps({"cpu_offload": True, "dp_shared_memory": True}))
    return argv


def request(body: dict[str, Any], *, rank: str | None = None, timeout: float = 900) -> dict[str, Any]:
    started = time.perf_counter()
    headers = {probe.RANK_HEADER: rank} if rank is not None else None
    try:
        with probe.open_response(
            BASE_URL,
            "/v1/completions",
            timeout,
            body=body,
            headers=headers,
        ) as response:
            return {"status": "PASS", **bench.stream_metrics(response, started, False, expected_input=None, expected_output=body["max_tokens"])}
    except Exception as error:
        return {"status": "FAIL", "error": str(error), "e2e_s": time.perf_counter() - started}


def decode_matrix(label: str, temperature: float) -> dict[str, Any]:
    reports = []
    for concurrency in (1, 16, 64, 256):
        prompts = bench.build_prompts(BASE_URL, "nature", 1024, concurrency, 1017 + concurrency)
        bodies = []
        for index, prompt in enumerate(prompts):
            body = bench.completion_body(prompt, 128, 1017)
            body.update(
                temperature=temperature,
                cache_salt=f"basic:{label}:decode:c{concurrency}:{index}",
            )
            bodies.append(body)
        rows, elapsed = bench.run_burst(
            BASE_URL,
            bodies,
            concurrency,
            900,
            expected_input=1024,
            expected_output=128,
        )
        reports.append(
            bench.summarize(
                rows,
                elapsed,
                kind="decode",
                concurrency=concurrency,
                input_tokens=1024,
                output_tokens=128,
                temperature=temperature,
            )
        )
    return {"decode": reports}


def prefill_matrix(label: str) -> dict[str, Any]:
    rows = []
    for input_tokens, ranks in ((260015, probe.RANKS), (1047000, (probe.RANKS[0],))):
        pool = bench.nature_pool(BASE_URL, input_tokens)
        for rank in ranks:
            body = bench.completion_body(pool, 8, 1017)
            body.update(
                cache_salt=f"basic:{label}:prefill:{input_tokens}:{rank}",
                add_special_tokens=False,
            )
            row = request(body, rank=rank, timeout=1800)
            row.update(input_tokens=input_tokens, rank=rank)
            rows.append(row)
    return {"prefill": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--draft-method", choices=("greedy", "probabilistic"), default="probabilistic")
    parser.add_argument("--official", action="store_true")
    args = parser.parse_args()

    command = make_command(args.image, args.name, args.volume, args.official, args.draft_method)
    container = subprocess.check_output(command, text=True).strip()
    output: dict[str, Any] = {
        "status": "FAIL",
        "label": args.label,
        "image": args.image,
        "official": args.official,
        "config": {
            "context": 1048576,
            "gpu_memory_utilization": 0.925,
            "chunk": 16384,
            "parallel": "tp4dp2/ep8",
            "max_num_seqs": 256,
            "dspark": 5,
            "draft_sample_method": args.draft_method,
            "decode_temperature": 0.7,
        },
    }
    try:
        wait_health()
        output.update(decode_matrix(args.label, 0.7))
        output.update(prefill_matrix(args.label))
        output["status"] = "PASS"
    finally:
        subprocess.run(["docker", "stop", "--time", "120", container], check=False)
        logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True, check=False)
        log_path = args.result.with_suffix(".server.log")
        log_path.write_text(logs.stdout + logs.stderr)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
