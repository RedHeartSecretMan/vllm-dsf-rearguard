"""Manage one shared 8-GPU benchmark maintenance window.

The window owns the production container lifecycle. Individual A/B runners only
start and stop their test containers. This keeps port 6789 down across a group
of experiments and makes a cached control result reusable when its identity is
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

WINDOW_VERSION = 1


class MaintenanceWindowError(RuntimeError):
    """Raised when a window state or lifecycle transition is invalid."""


def digest(value: Any) -> str:
    """Return a stable digest for a launch or workload identity."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def control_identity(
    *, experiment: str, baseline_image: str, launch: dict[str, Any]
) -> dict[str, Any]:
    """Build the exact identity required before reusing a control result."""
    return {
        "experiment": experiment,
        "baseline_image": baseline_image,
        "launch_sha256": digest(launch),
    }


def cache_file(cache_dir: Path, experiment: str, identity: dict[str, Any]) -> Path:
    """Return a deterministic control cache path."""
    return cache_dir / experiment / f"{digest(identity)}.json"


def load_control_cache(
    cache_dir: Path, experiment: str, identity: dict[str, Any]
) -> dict[str, Any] | None:
    """Read a matching control result, returning ``None`` for a cache miss."""
    path = cache_file(cache_dir, experiment, identity)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MaintenanceWindowError(f"Invalid control cache {path}") from error
    if record.get("identity") != identity or "result" not in record:
        raise MaintenanceWindowError(f"Control cache identity mismatch: {path}")
    return record["result"]


def save_control_cache(
    cache_dir: Path,
    experiment: str,
    identity: dict[str, Any],
    result: dict[str, Any],
) -> Path:
    """Persist a control result under its exact launch/workload identity."""
    path = cache_file(cache_dir, experiment, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cache_version": 1,
                "identity": identity,
                "result": result,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def _inspect_container(name: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.Id}} {{.Image}} {{.State.Status}}",
            name,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise MaintenanceWindowError(f"Production container not found: {name}")
    container_id, image, status = result.stdout.strip().split(maxsplit=2)
    return {"name": name, "id": container_id, "image": image, "status": status}


def _wait_http_health(url: str, timeout: float) -> None:
    host, port_text = url.removeprefix("http://").split(":", 1)
    port = int(port_text.rstrip("/"))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        connection = HTTPConnection(host, port, timeout=3)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            if response.status == 200:
                response.read()
                return
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(2)
    raise TimeoutError(f"Health check did not pass at {url}")


def open_window(
    *, production_container: str, state_path: Path, health_url: str = "http://127.0.0.1:6789"
) -> dict[str, Any]:
    """Stop production once and record the state needed for final restoration."""
    current = _inspect_container(production_container)
    if current["status"] == "running":
        subprocess.run(["docker", "stop", "--time", "90", production_container], check=True)
        current = _inspect_container(production_container)
    if current["status"] != "exited":
        raise MaintenanceWindowError(
            f"Production container must be stopped before the window opens: {current['status']}"
        )
    state = {
        "window_version": WINDOW_VERSION,
        "status": "open",
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "health_url": health_url,
        "production": current,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state


def close_window(*, state_path: Path, timeout: float = 900) -> dict[str, Any]:
    """Restore the recorded production container and close the window."""
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MaintenanceWindowError(f"Invalid maintenance window state: {state_path}") from error
    if state.get("window_version") != WINDOW_VERSION or state.get("status") != "open":
        raise MaintenanceWindowError("Maintenance window is missing or already closed")
    production = state.get("production", {})
    name = production.get("name")
    if not isinstance(name, str) or not name:
        raise MaintenanceWindowError("Window state has no production container name")
    current = _inspect_container(name)
    if current["id"] != production.get("id") or current["image"] != production.get("image"):
        raise MaintenanceWindowError("Production container identity changed during the window")
    subprocess.run(["docker", "start", production["id"]], check=True)
    _wait_http_health(state["health_url"], timeout)
    state["status"] = "closed"
    state["closed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--production-container", required=True)
    start.add_argument("--state", type=Path, required=True)
    start.add_argument("--health-url", default="http://127.0.0.1:6789")
    finish = subparsers.add_parser("finish")
    finish.add_argument("--state", type=Path, required=True)
    finish.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.action == "start":
        result = open_window(
            production_container=args.production_container,
            state_path=args.state,
            health_url=args.health_url,
        )
    else:
        result = close_window(state_path=args.state, timeout=args.timeout)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
