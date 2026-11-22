"""Build and install the optional DSpark KV-only CUDA operator."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "kv_only"
BUILD = Path("/tmp/vllm-dsf-kv-only-build")


def main() -> None:
    if os.environ.get("VLLM_BUILD_KV_ONLY_NATIVE") != "1":
        raise SystemExit("VLLM_BUILD_KV_ONLY_NATIVE must be 1")

    BUILD.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    os.environ.setdefault("MAX_JOBS", "4")
    target_version = "0x020D000000000000"
    library = load(
        name="vllm_kv_only_native",
        sources=[
            str(NATIVE / "fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu"),
            str(NATIVE / "kv_only_bindings.cpp"),
        ],
        extra_include_paths=[str(NATIVE)],
        extra_cflags=["-DUSE_CUDA", f"-DTORCH_TARGET_VERSION={target_version}"],
        extra_cuda_cflags=["-DUSE_CUDA", f"-DTORCH_TARGET_VERSION={target_version}"],
        with_cuda=True,
        verbose=True,
        is_python_module=False,
        build_directory=str(BUILD),
    )
    import vllm

    destination = Path(vllm.__file__).resolve().parent / "_C_kv_only_native.so"
    shutil.copy2(library, destination)
    print(destination)


if __name__ == "__main__":
    main()
