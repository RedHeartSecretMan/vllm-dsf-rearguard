"""Sequential, resumable service deployment for the controlled profile suite.

Only explicitly named project-owned containers are stopped. Failed deployments
and measurements are retained and halt the campaign for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from scripts.bench_common import ROOT
from scripts.benchmarks.profile_matrix import (
    normalized_command,
    validate_profile,
    write_json,
)
from scripts.benchmarks.profile_workload import BASE_URL, load, source_identity


def command(args: list[str], **kwargs):
    return subprocess.check_output(args, text=True, **kwargs)


def inspect(name: str) -> dict[str, Any]:
    return json.loads(command(["docker", "inspect", name]))[0]


def stop(name: str) -> None:
    state = inspect(name)
    if state["Config"]["Labels"].get("dsv41-a100.owner") != "deepseek-v4.1-flash-a100":
        raise ValueError("refusing to stop a container outside this project")
    if state["State"]["Running"]:
        result = subprocess.run(["docker", "stop", "-t", "60", name], capture_output=True, text=True)
        if result.returncode and inspect(name)["State"]["Running"]:
            raise RuntimeError(result.stderr)


def cache_copy(previous: str, target: str, image: str) -> None:
    command(["docker", "volume", "create", target + "-cache"])
    subprocess.run(["docker", "run", "--rm", "--runtime=runc", "--network", "none",
                    "--mount", f"type=volume,src={previous}-cache,dst=/from,readonly",
                    "--mount", f"type=volume,src={target}-cache,dst=/to",
                    "--entrypoint", "sh", image, "-c", "cp -a /from/. /to/"], check=True)


def launch(profile: dict[str, Any], args, previous: str) -> tuple[str, dict[str, Any]]:
    destination = args.directory / profile["attempt_id"]
    destination.mkdir(parents=True, exist_ok=False)
    name = "dsf-prof-" + profile["attempt_id"].replace("_", "-")
    cache_copy(previous, name, args.image)
    write_json(destination / "previous-container.json", inspect(previous))
    stop(previous)
    launch_command = [sys.executable, str(ROOT / "launch.py"), str(args.model), "--image", args.image,
                      "--name", name, "--context", str(profile["context"]), "--memory", str(profile["memory"]),
                      "--chunk", str(profile["chunk"]), "--receipt", str(args.model_receipt)]
    launch_command += args.fixed_launch + ["--start"]
    write_json(destination / "launch-command.json", {"command": launch_command, "source": source_identity()})
    with (destination / "launch.log").open("w") as stream:
        result = subprocess.run(launch_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"launch failed for {profile['id']}; see {destination / 'launch.log'}")
    actual = inspect(name)
    matches = []
    for path in (ROOT / "runtime").glob("*/launch.json"):
        receipt = json.loads(path.read_text())
        if receipt.get("container_id") == actual["Id"]:
            matches.append(receipt)
    if len(matches) != 1:
        raise RuntimeError("could not bind launch to one real receipt")
    receipt = matches[0]
    write_json(destination / "launch.json", receipt)
    for _ in range(360):
        if not inspect(name)["State"]["Running"]:
            raise RuntimeError("container exited during startup")
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(5)
    else:
        raise TimeoutError("test endpoint startup timed out")
    validate_profile(receipt, profile["context"], profile["memory"], profile["chunk"])
    return name, receipt


def run(args) -> int:
    config = json.loads((ROOT / "config/benchmarks/profile_matrix.json").read_text())
    args.fixed_launch = config["fixed_launch"]
    args.directory.mkdir(parents=True, exist_ok=True)
    schedule = json.loads(args.schedule.read_text())
    if not isinstance(schedule, list) or not schedule or len({p["id"] for p in schedule}) != len(schedule):
        raise ValueError("schedule must contain distinct, nonempty profile IDs")
    frozen = load(args.inputs)
    state_path = args.directory / "campaign.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "status": "IN_PROGRESS", "source": source_identity(), "inputs_sha256": frozen["sha256"], "profiles": [],
        "status_scope": "Only the explicitly supplied phase; final acceptance separately requires all nine grid points and once-only tests"}
    if state["inputs_sha256"] != frozen["sha256"]:
        raise ValueError("campaign input identity changed")
    previous = args.previous_container
    done = {row["id"]: row for row in state["profiles"] if row["status"] == "PASS"}
    for profile in schedule:
        if profile["id"] in done:
            for key in ("context", "memory", "chunk", "only"):
                if done[profile["id"]].get(key, "all" if key == "only" else None) != profile.get(key, "all" if key == "only" else None):
                    raise ValueError("completed profile configuration changed; refusing to reuse evidence")
            previous = done[profile["id"]]["container"]
            continue
        attempt = 1 + sum(p["id"] == profile["id"] for p in state["profiles"])
        profile = {**profile, "attempt_id": profile["id"] if attempt == 1 else f"{profile['id']}-attempt-{attempt}"}
        entry = {**profile, "status": "STARTING", "attempt": attempt}
        state["profiles"].append(entry)
        write_json(state_path, state)
        try:
            if profile.get("adopt_receipt"):
                receipt = json.loads(Path(profile["adopt_receipt"]).read_text())
                validate_profile(receipt, profile["context"], profile["memory"], profile["chunk"])
                container = inspect(receipt["container_id"])["Name"].lstrip("/")
                target = args.directory / profile["attempt_id"]
                target.mkdir(parents=True, exist_ok=False)
                write_json(target / "launch.json", receipt)
            else:
                container, receipt = launch(profile, args, previous)
                target = args.directory / profile["attempt_id"]
            normalized = normalized_command(receipt)
            if "normalized_command" not in state:
                state["normalized_command"] = normalized
            elif normalized != state["normalized_command"]:
                raise ValueError("service parameters differ outside permitted matrix/chunk axes")
            entry["container"] = container
            invocation = [sys.executable, "-m", "scripts.benchmarks.profile_matrix", "--inputs", str(args.inputs),
                          "--context", str(profile["context"]), "--memory", str(profile["memory"]),
                          "--chunk", str(profile["chunk"]), "--launch-receipt", str(target / "launch.json"),
                          "--ppl-dataset", str(args.ppl_dataset), "--output", str(target / "results"),
                          "--only", profile.get("only", "all")]
            entry.update(status="RUNNING", runner_command=invocation)
            write_json(state_path, state)
            with (target / "runner.log").open("w") as stream:
                result = subprocess.run(invocation, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            report = json.loads((target / "results/summary.json").read_text())
            if result.returncode or report["status"] != "PASS":
                raise RuntimeError("profile measurements failed; retained for diagnosis")
            entry.update(status="PASS", summary=str(target / "results/summary.json"))
            previous = container
        except Exception as error:
            entry.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            state.update(status="NEEDS_REVIEW", stopped_at=profile["id"])
            write_json(state_path, state)
            print(json.dumps(entry), flush=True)
            return 1
        write_json(state_path, state)
        print(json.dumps({k: entry[k] for k in ("id", "status", "container")}), flush=True)
    state["status"] = "PHASE_PASS"
    write_json(state_path, state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--ppl-dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--previous-container", required=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
