"""Run one frozen fixed-profile suite on the isolated GPU test endpoint.

Service startup is separate: this runner validates its actual launch receipt.
The same frozen inputs are reused for context/memory and controlled chunk axes.
"""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.bench_common import (
    ROOT,
    json_request,
    require_model,
    verify_endpoint,
)
from scripts.benchmarks import profile_workload as inputs
from scripts.benchmarks.profile_plan import catalog_inventory, group_plan


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def option(command: list[str], name: str) -> str:
    if command.count(name) != 1:
        raise ValueError(f"expected exactly one {name}")
    return command[command.index(name) + 1]


def validate_profile(receipt: dict[str, Any], context: int, memory: float, chunk: int) -> None:
    command = receipt["command"]
    expected = {"-p": "127.0.0.1:6688:8000", "--max-model-len": str(context),
                "--max-num-batched-tokens": str(chunk), "--tensor-parallel-size": "4",
                "--data-parallel-size": "2", "--data-parallel-size-local": "2",
                "--max-num-seqs": "128", "--api-server-count": "1", "--kv-cache-dtype": "fp8_ds_mla"}
    for key, value in expected.items():
        if option(command, key) != value:
            raise ValueError(f"profile receipt mismatch: {key}")
    if float(option(command, "--gpu-memory-utilization")) != memory:
        raise ValueError("profile memory utilization mismatch")
    if "--no-disable-chunked-mm-input" not in command or "--disable-chunked-mm-input" in command:
        raise ValueError("fixed profiles require enabled multimodal chunking")
    for flag in ("--enable-prefix-caching", "--enable-chunked-prefill", "--enable-prompt-tokens-details"):
        if flag not in command:
            raise ValueError(f"fixed profiles require {flag}")
    if json.loads(option(command, "--limit-mm-per-prompt")) != {"image": 64, "video": {"count": 2, "num_frames": 32}}:
        raise ValueError("fixed multimodal limits differ")
    if json.loads(option(command, "--media-io-kwargs")) != {"video": {"num_frames": 32, "fps": 1}}:
        raise ValueError("fixed video sampling differs")
    env = dict(command[i + 1].split("=", 1) for i, part in enumerate(command[:-1]) if part == "--env")
    required = {"VLLM_DSV41_STACKED_DRAFT_KV": "1", "VLLM_DSV41_BOUNDED_PREFILL_WORKSPACE": "1",
                "VLLM_DSV41_PREFILL_THROTTLE": "0", "VLLM_DSV41_DECODE_METADATA_FASTPATH": "0",
                "VLLM_DSV41_REUSE_TOPK_RAGGED": "0", "OMP_NUM_THREADS": "8",
                "VLLM_SM80_EP8_CUSTOM_AG_RS": "1", "VLLM_DSV4_LOGITS_ROW_CHUNK": "0",
                "VLLM_ENABLE_RESPONSES_API_STORE": "1"}
    if any(env.get(k) != v for k, v in required.items()) or "--enable-eplb" not in command:
        raise ValueError("fixed-profile environment mismatch")
    speculative = json.loads(option(command, "--speculative-config"))
    if speculative["method"] != "dspark" or speculative["num_speculative_tokens"] != 5:
        raise ValueError("DSpark configuration differs")
    if receipt["source_manifest_sha256"] != inputs.source_identity()["source_manifest_sha256"]:
        raise ValueError("inference source fingerprint differs")


def normalized_command(receipt: dict[str, Any]) -> list[str]:
    """Only the two matrix axes, controlled chunk, name and cache volume may vary."""
    command = list(receipt["command"])
    name = option(command, "--name")
    for flag in ("--name", "--max-model-len", "--gpu-memory-utilization", "--max-num-batched-tokens"):
        command[command.index(flag) + 1] = f"<{flag}>"
    volume = f"type=volume,src={name}-cache,dst=/workspace"
    if command.count(volume) != 1:
        raise ValueError("unexpected compilation cache mount")
    command[command.index(volume)] = "type=volume,src=<task-cache>,dst=/workspace"
    return command


def execution_kind(cell: dict[str, Any]) -> str:
    if cell["group"] in ("regular", "long_tail"):
        return "mixed" if cell["group"] == "long_tail" and cell["kind"] == "mixed" else "performance"
    return cell["group"]


def resolve_cell(cell: dict[str, Any], frozen: dict[str, Any], context: int) -> dict[str, Any]:
    result = copy.deepcopy(cell)
    if cell["group"] in ("regular", "long_tail") and cell["kind"] not in ("mixed", "soak"):
        counts = [inputs.observed_input(frozen, cell, i) for i in range(cell["concurrency"])]
        if all(isinstance(n, int) for n in counts):
            result.update(expected_input_by_index=counts, actual_expected_input=counts[0],
                          actual_input_max=max(counts), length_check="KNOWN", input_tolerance=0)
            if max(counts) + cell["osl"] > context:
                result.update(status="NOT_APPLICABLE", repeats=0,
                              reason=f"Prepared input {max(counts)} + output {cell['osl']} exceeds {context}; not truncated")
            elif cell["status"] == "PENDING_PREPARATION":
                result.update(status="APPLICABLE", reason=None)
    return result


def run_profile(args) -> int:
    from scripts.benchmarks.profile_acceptance import (
        run_core,
        run_responses,
        run_visual,
    )
    from scripts.benchmarks.profile_measure import ServiceObserver, measure, warmup
    from scripts.benchmarks.profile_mixed import run as run_mixed

    receipt = json.loads(args.launch_receipt.read_text())
    validate_profile(receipt, args.context, args.memory, args.chunk)
    endpoint = verify_endpoint(inputs.BASE_URL, receipt)
    require_model(inputs.BASE_URL)
    frozen = inputs.load(args.inputs)
    if frozen["source"]["source_manifest_sha256"] != receipt["source_manifest_sha256"]:
        raise ValueError("input preparation and measured inference sources differ")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch.json", receipt)
    config = json.loads((ROOT / "config/benchmarks/profile_matrix.json").read_text())
    plan = group_plan(args.context, args.chunk, args.memory)
    plan["cells"] = [resolve_cell(c, frozen, args.context) if c["status"] in ("APPLICABLE", "PENDING_PREPARATION") else c for c in plan["cells"]]
    plan["counts_before_preparation"] = dict(plan["counts"])
    eligible = [c for c in plan["cells"] if c["status"] in ("APPLICABLE", "PENDING_PREPARATION")]
    plan["counts"].update(applicable=len(eligible),
                          pending_preparation=sum(c["status"] == "PENDING_PREPARATION" for c in plan["cells"]),
                          not_applicable=sum(c["status"] == "NOT_APPLICABLE" for c in plan["cells"]),
                          repeats=sum(c["repeats"] for c in eligible))
    for cell in plan["cells"]:
        if execution_kind(cell) == "mixed":
            cell["execution_variant"] = "fixed_arrival_schedule"
            cell["derivation"] = {"from_case": cell["case_id"],
                                  "changes": {"arrival_policy": "5 fixed decode bursts of 64 plus 15 scheduled prefill requests"},
                                  "preserved": ["input/output lengths", "concurrency limits", "300s arrival window", "seed", "same service"]}
    write_json(output / "plan.json", plan)
    report = {"status": "IN_PROGRESS", "profile": plan["profile"], "endpoint": endpoint,
              "source": inputs.source_identity(), "frozen_inputs_sha256": frozen["sha256"],
              "started_at": datetime.now(timezone.utc).isoformat(), "cells": [],
              "selection": args.only, "warmup_policy": config["warmup"]}
    observer = ServiceObserver(inputs.BASE_URL, receipt["container_id"], output / "observation")

    def checkpoint():
        write_json(output / "progress.json", report)

    try:
        with observer:
            startup = {"cell_id": "startup-graphs", "kind": "matrix", "isl": 1024, "osl": 128,
                       "concurrency": 256, "dataset": "random", "source_case": {"seed": 1017},
                       "warmup_concurrency": 256, "warmup_osl": 128}
            def startup_factory(label, index):
                return inputs.request_body(frozen, startup, label, index)
            global_warm = warmup(inputs.BASE_URL, startup_factory, startup, observer, output / "global-warmup")
            report["global_warmup"] = global_warm
            checkpoint()
            if global_warm["status"] != "PASS":
                raise RuntimeError("DP startup warmup failed")
            for cell in plan["cells"]:
                if args.only == "matrix" and cell["case_id"] not in ("matrix_single", "matrix_concurrent"):
                    continue
                if cell["status"] not in ("APPLICABLE", "PENDING_PREPARATION"):
                    report["cells"].append({"cell_id": cell["cell_id"], "status": cell["status"], "reason": cell["reason"]})
                    continue
                directory = output / "cells" / cell["cell_id"].replace("/", "__")
                row = {"cell_id": cell["cell_id"], "case_id": cell["case_id"], "group": cell["group"],
                       "status": "IN_PROGRESS", "measurements": []}
                report["cells"].append(row)
                checkpoint()
                try:
                    if execution_kind(cell) == "performance":
                        warm_cell = copy.deepcopy(cell)
                        if cell["kind"] == "vision_mix":
                            maximum = max(cell["source_case"]["image_counts"])
                            warm_cell["source_case"]["image_counts"] = [maximum]
                            key = f"{cell['source_case']['id']}-{maximum}"
                            warm_cell["expected_input_by_index"] = [frozen["observed_input_tokens"][key]] * 2
                            warm_cell["encoder_warmup"] = "maximum-image-count inputs on both DP instances"
                        def factory(label, index, selected=warm_cell):
                            body = inputs.request_body(frozen, selected, label, index)
                            body["max_tokens"] = min(selected["osl"], 128)
                            return body
                        row["warmup"] = warmup(inputs.BASE_URL, factory, warm_cell, observer, directory / "warmup")
                        if row["warmup"]["status"] != "PASS":
                            raise RuntimeError("per-cell DP warmup failed")
                        for repeat in range(1, cell["repeats"] + 1):
                            bodies = [inputs.request_body(frozen, cell, f"repeat-{repeat}", i) for i in range(cell["concurrency"])]
                            result = measure(inputs.BASE_URL, bodies, cell, directory / f"repeat-{repeat}", timeout=args.timeout)
                            row["measurements"].append(result)
                            checkpoint()
                    elif execution_kind(cell) == "mixed":
                        row["measurements"].append(run_mixed(inputs.BASE_URL, cell, frozen, config["mixed_comparison"], directory, args.timeout))
                    elif cell["group"] == "core":
                        row["measurements"].append(run_core(inputs.BASE_URL, cell, frozen, receipt, directory, args.timeout))
                    elif cell["group"] == "visual":
                        row["measurements"].append(run_visual(inputs.BASE_URL, cell, frozen, receipt, directory))
                    elif cell["group"] == "responses":
                        row["measurements"].append(run_responses(inputs.BASE_URL, cell, frozen, receipt, directory))
                    elif cell["kind"] == "ppl":
                        from scripts.acceptance import ppl_score
                        dataset = args.ppl_dataset
                        records = ppl_score.read_records(dataset)
                        windows = ppl_score.token_windows(inputs.BASE_URL, records, 2048, 2048, args.timeout)
                        result = ppl_score.evaluate(inputs.BASE_URL, windows, 1, args.timeout, "echo",
                                                    evidence_dir=directory / "raw", cache_salt=f"profile:{frozen['run_id']}:ppl", seed=1017)
                        ppl_score.add_unit_perplexities(result, records)
                        result["dataset_sha256"] = inputs.sha256(dataset)
                        write_json(directory / "report.json", result)
                        row["measurements"].append(result)
                    else:
                        raise ValueError(f"missing profiling runner: {cell['kind']}")
                    row["status"] = "PASS" if row["measurements"] and all(r["status"] == "PASS" for r in row["measurements"]) else "FAIL"
                except Exception as error:
                    row.update(status="FAIL", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                    write_json(directory / "failure.json", row)
                checkpoint()
                print(json.dumps({k: row[k] for k in ("cell_id", "status", "error") if k in row}), flush=True)
                try:
                    if json_request(inputs.BASE_URL, "/health", timeout=10) is None:
                        pass
                except json.JSONDecodeError:
                    pass  # /health returns an empty 200 body.
                except Exception as error:
                    raise RuntimeError("service unhealthy; remaining cells must not run") from error
            report["status"] = "FAIL" if any(c["status"] == "FAIL" for c in report["cells"]) else "PASS"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["observation"] = observer.summary()
        if report["observation"]["status"] != "PASS":
            report.update(status="FAIL", observation_error="service evidence or identity collection was incomplete")
        try:
            if verify_endpoint(inputs.BASE_URL, receipt) != endpoint:
                raise ValueError("container identity changed")
        except Exception as error:
            report.update(status="FAIL", endpoint_error=str(error))
        write_json(output / "summary.json", report)
    return 0 if report["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--context", type=int, default=1048576)
    parser.add_argument("--memory", type=float, default=.90)
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--base-url", default=inputs.BASE_URL)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--fixtures-dir", type=Path)
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--ppl-dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", default="profile-20261025")
    parser.add_argument("--only", choices=("all", "matrix"), default="all")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    if args.base_url != inputs.BASE_URL:
        parser.error("all profiling requests must use http://127.0.0.1:6688")
    if args.inventory or args.plan:
        print(json.dumps(catalog_inventory() if args.inventory else group_plan(args.context, args.chunk, args.memory), indent=2))
        return 0
    if args.inputs is None:
        parser.error("--inputs is required")
    if args.prepare:
        if args.fixtures_dir is None:
            parser.error("--prepare requires --fixtures-dir")
        artifact = inputs.prepare(args.base_url, args.fixtures_dir, args.inputs, args.run_id)
        print(json.dumps({"status": "PASS", "inputs_sha256": artifact["sha256"]}))
        return 0
    if args.output is None or args.launch_receipt is None or args.ppl_dataset is None:
        parser.error("execution requires --output, --launch-receipt and --ppl-dataset")
    return run_profile(args)


if __name__ == "__main__":
    raise SystemExit(main())
