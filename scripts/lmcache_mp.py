"""Build and start the same-host LMCache MP sidecar for a launch plan."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from dataclasses import dataclass, fields
from http.client import HTTPConnection
from pathlib import Path

HOST = "127.0.0.1"
DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "config/runtime/lmcache-mp.json"


@dataclass(frozen=True)
class LMCacheMP:
    l1_size_gb: int
    l1_init_size_gb: int
    chunk_size: int
    port: int
    http_port: int

    @classmethod
    def load(cls, path: Path) -> LMCacheMP:
        values = json.loads(path.read_text())
        if not isinstance(values, dict) or set(values) != {field.name for field in fields(cls)}:
            raise ValueError("LMCache MP profile must specify L1 size, initial size, chunk size and both ports")
        if any(type(value) is not int or value <= 0 for value in values.values()):
            raise ValueError("LMCache MP profile values must be positive integers")
        profile = cls(**values)
        if profile.l1_init_size_gb > profile.l1_size_gb:
            raise ValueError("LMCache MP initial allocation cannot exceed the L1 limit")
        if max(profile.port, profile.http_port) > 65535 or profile.port == profile.http_port:
            raise ValueError("LMCache MP needs two distinct valid TCP ports")
        return profile

    def transfer_config(self) -> dict:
        return {
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "lmcache.mp.host": f"tcp://{HOST}",
                "lmcache.mp.port": self.port,
                "lmcache.mp.mp_transfer_mode": "lmcache_driven",
                "lmcache.mp.mq_timeout": 300,
                "lmcache.mp.heartbeat_interval": 5,
                "lmcache.mp.lazy_offload": False,
            },
        }

    def server_command(self, name: str, image: str) -> list[str]:
        return [
            "docker", "run", "-d", "--pull", "never", "--init", "--name", name,
            "--label", "vllm-dsf-rearguard.owner=vllm-dsf-rearguard",
            "--gpus", "all", "--network=host", "--ipc=host", "--ulimit", "memlock=-1",
            "--log-opt", "max-size=100m", "--log-opt", "max-file=5",
            "--entrypoint", "lmcache", image, "server",
            "--host", HOST, "--port", str(self.port),
            "--http-host", HOST, "--http-port", str(self.http_port),
            "--l1-size-gb", str(self.l1_size_gb), "--l1-use-lazy",
            "--l1-init-size-gb", str(self.l1_init_size_gb), "--eviction-policy", "LRU",
            "--chunk-size", str(self.chunk_size), "--separate-object-groups",
        ]


def require_free_ports(ports: tuple[int, ...]) -> None:
    for port in ports:
        with socket.socket() as listener:
            try:
                listener.bind((HOST, port))
            except OSError as exc:
                raise ValueError(f"Cannot bind {HOST}:{port}: {exc}") from exc


def require_l1_capacity(profile: LMCacheMP) -> None:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            if available_kib >= profile.l1_size_gb * 1024 ** 2:
                return
            raise ValueError(f"Host MemAvailable is below the LMCache L1 limit ({profile.l1_size_gb} GiB)")
    raise ValueError("Host MemAvailable is missing from /proc/meminfo")


def wait_for_health(port: int, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        conn = HTTPConnection(HOST, port, timeout=2)
        try:
            conn.request("GET", "/healthcheck")
            response = conn.getresponse()
            if response.status == 200 and json.loads(response.read(8192)) == {"status": "healthy"}:
                return
        except (OSError, ValueError):
            pass
        finally:
            conn.close()
        time.sleep(1)
    raise TimeoutError(f"LMCache MP health check did not pass on {HOST}:{port}")


def stop_sidecar(container_id: str) -> None:
    subprocess.run(["docker", "stop", "--time", "20", container_id], check=False)
