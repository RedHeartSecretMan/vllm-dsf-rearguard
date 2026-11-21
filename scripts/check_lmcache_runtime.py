"""Verify the opt-in LMCache MP runtime without starting a server or using a GPU."""

import json
import platform
import sys
from importlib import import_module, metadata
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "lmcache-mp.json"


def require_equal(name, actual, expected):
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected!r}, found {actual!r}")


def check_runtime(manifest):
    from packaging.requirements import Requirement

    runtime = manifest["runtime"]
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    require_equal("Python", python_version, runtime["python"])
    require_equal("architecture", platform.machine(), runtime["architecture"])
    packages = runtime["packages"]
    for name, expected in packages.items():
        require_equal(name, metadata.version(name), expected)

    # The Docker layer installs four locked wheels without allowing pip to
    # resolve or replace the base image's Torch and vLLM dependencies.
    for name in ("lmcache", "grpcio", "grpcio-observability", "grpcio-status"):
        for entry in metadata.requires(name) or ():
            req = Requirement(entry)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            installed = metadata.version(req.name)
            if not req.specifier.contains(installed, prereleases=True):
                raise RuntimeError(
                    f"{name} requires {req}; found {req.name}=={installed}"
                )

    import torch

    require_equal("imported Torch", torch.__version__, packages["torch"])
    require_equal("Torch CUDA ABI", torch.version.cuda, runtime["cuda"])
    require_equal("Torch C++11 ABI", torch.compiled_with_cxx11_abi(), True)
    require_equal(
        "imported LMCache", import_module("lmcache").__version__, packages["lmcache"]
    )
    for name in (
        "lmcache.cuda_ops",
        "lmcache.lmcache_native",
        "lmcache.v1.multiprocess.server",
    ):
        import_module(name)

    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    # The backport registers an older implementation with the same class name.
    # Use the real factory to confirm the module override and HMA detection.
    config = KVTransferConfig(
        **manifest["connector"], kv_role="kv_both", kv_buffer_device="cuda"
    )
    connector = KVConnectorFactory.get_connector_class(config)
    require_equal(
        "connector module", connector.__module__, config.kv_connector_module_path
    )
    require_equal(
        "hybrid KV cache manager support",
        KVConnectorFactory.supports_hma_config(config),
        True,
    )
    require_equal(
        "unimplemented connector methods", sorted(connector.__abstractmethods__), []
    )
    adapter_module = import_module(connector.__module__)
    for name in ("LMCacheMPSchedulerAdapter", "LMCacheMPWorkerAdapter"):
        require_equal(
            f"{name} module",
            getattr(adapter_module, name).__module__,
            "lmcache.integration.vllm.vllm_multi_process_adapter",
        )
    return {
        "status": "PASS",
        "lmcache_version": packages["lmcache"],
        "torch_version": packages["torch"],
        "connector_module": connector.__module__,
        "hybrid_kv_cache_manager": True,
        "gpu_validation": "NOT_RUN",
    }


def main():
    try:
        print(json.dumps(check_runtime(json.loads(MANIFEST.read_text()))))
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"LMCache MP runtime check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
