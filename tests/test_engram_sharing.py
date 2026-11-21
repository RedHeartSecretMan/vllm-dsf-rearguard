"""CPU checks for the pinned Engram sharing overlay without importing vLLM."""

import ast
import gc
import importlib.util
import mmap
import multiprocessing
import os
import tempfile
import unittest
import weakref
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.source_extract import load_class, load_functions, run_function_tree

OVERLAY = Path(__file__).resolve().parents[1] / "adaptation/files/vllm"
ENGRAM = OVERLAY / "models/deepseek_v4_1/common/engram.py"
GIB = 1 << 30


def _storage_worker(rank, directory, scenario):
    import numpy as np
    import torch

    dist = torch.distributed
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    active = set()

    def cuda_register(pointer, _size, _flags):
        # Registration must happen after the name has been removed by rank 0.
        if list(root.glob("vllm_engram_*")):
            raise RuntimeError("registration started before unlink")
        if rank == 1 and scenario == "register_failure":
            return SimpleNamespace(value=1)
        active.add(pointer)
        return SimpleNamespace(value=0)

    def cuda_unregister(pointer):
        active.remove(pointer)
        return SimpleNamespace(value=0)

    def broadcast(value):
        obj = [value]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def uva_view(value):
        assert value.is_pinned(), "private UVA fallback was reached"
        return value

    group = SimpleNamespace(
        rank_in_group=rank,
        ranks=[0, 1],
        world_size=2,
        cpu_group=dist.group.WORLD,
        broadcast_object=broadcast,
    )
    names = {
        "DPSharedEngramStorage",
        "load_shared_engram_weights",
        "_raise_engram_group_failures",
        "_engram_head_shard_weight_loader",
    }
    module = ast.parse(ENGRAM.read_text())
    module.body = [n for n in module.body if getattr(n, "name", None) in names]
    symbols = {
        "ExitStack": ExitStack,
        "logger": SimpleNamespace(warning=lambda *_: None),
        "mmap": mmap,
        "np": np,
        "os": os,
        "tempfile": tempfile,
        "torch": torch,
        "weakref": weakref,
        "SHM_PATH": str(root),
        "_check_engram_shm_capacity": lambda *_: None,
        "_allow_any_engram_numa_node": lambda *_: True,
        "get_accelerator_view_from_cpu_tensor": uva_view,
    }
    storage_cls = run_function_tree(module, ENGRAM, symbols, "DPSharedEngramStorage")
    cudart = SimpleNamespace(
        cudaHostRegister=cuda_register, cudaHostUnregister=cuda_unregister
    )
    try:
        with (
            patch.object(torch.cuda, "cudart", lambda: cudart),
            patch.object(
                torch.Tensor,
                "is_pinned",
                lambda value: (
                    value.data_ptr() in active
                    and not (rank == 1 and scenario == "not_pinned")
                ),
            ),
        ):
            if scenario in ("register_failure", "not_pinned"):
                match = (
                    "rank 1.*did not pin"
                    if scenario == "not_pinned"
                    else "rank 1.*cudaHostRegister"
                )
                with unittest.TestCase().assertRaisesRegex(RuntimeError, match):
                    storage_cls(2, 32, 32, group, numa_bound=False)
                assert not active
                return
            storage = storage_cls(2, 32, 32, group, numa_bound=False)
            params = [
                torch.nn.Parameter(tensor, requires_grad=False)
                for tensor in (storage.weight, storage.weight_scale_inv)
            ]
            for param in params:
                param.engram_vocab_start = 1
            payloads = [
                torch.arange(128, dtype=torch.uint8)
                .reshape(4, 32)
                .view(torch.float8_e4m3fn),
                torch.arange(120, 124, dtype=torch.uint8)
                .reshape(4, 1)
                .view(torch.float8_e8m0fnu),
            ]

            def load(storage=storage, params=params):
                order = (0, 1) if rank == 0 else (1, 0)
                for idx in order:
                    if scenario != "missing_scale" or rank == 1 or idx == 0:
                        storage.load_weight(
                            params[idx], payloads[idx] if rank == 0 else None
                        )
                return {"sentinel"}

            complete = symbols["load_shared_engram_weights"]
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "not ready"):
                storage.get_views(*params)
            if scenario == "missing_scale":
                with unittest.TestCase().assertRaisesRegex(
                    RuntimeError, "rank 0.*scale"
                ):
                    complete([("test", storage)], load)
            else:
                assert complete([("test", storage)], load) == {"sentinel"}
                weight, scale = storage.get_views(*params)
                assert weight.view(torch.uint8).flatten().tolist() == list(
                    range(32, 96)
                )
                assert scale.flatten().tolist() == [121, 122]
                del weight, scale
            del load, params, param, storage
            gc.collect()
            assert not active
    finally:
        dist.destroy_process_group()


class EngramSharingTests(unittest.TestCase):
    def test_config_requires_cpu_offload_and_nonelastic_dp(self):
        config = load_class(
            OVERLAY / "config/engram.py",
            {
                "config": dataclass,
                "Field": field,
                "_default_cpu_offload": lambda: False,
            },
            annotations=True,
        )
        parallel = SimpleNamespace(data_parallel_size=2, enable_elastic_ep=False)
        self.assertFalse(config().dp_shared_memory)
        config(cpu_offload=True, dp_shared_memory=True).verify_parallel_config(parallel)
        with self.assertRaisesRegex(ValueError, "requires cpu_offload"):
            config(dp_shared_memory=True).verify_parallel_config(parallel)
        parallel.enable_elastic_ep = True
        with self.assertRaisesRegex(ValueError, "elastic EP"):
            config(cpu_offload=True, dp_shared_memory=True).verify_parallel_config(
                parallel
            )
        parallel.enable_elastic_ep = False
        parallel.data_parallel_size = 1
        with self.assertRaisesRegex(ValueError, "data_parallel_size"):
            config(cpu_offload=True, dp_shared_memory=True).verify_parallel_config(
                parallel
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
        "requires CPU PyTorch, NumPy and Gloo",
    )
    def test_two_ranks_share_ready_weights_and_cleanup_failures(self):
        import torch

        if not torch.distributed.is_gloo_available():
            self.skipTest("requires Gloo")
        context = multiprocessing.get_context("spawn")
        for scenario in ("success", "missing_scale", "register_failure", "not_pinned"):
            with (
                self.subTest(scenario=scenario),
                tempfile.TemporaryDirectory() as directory,
            ):
                workers = [
                    context.Process(
                        target=_storage_worker, args=(rank, directory, scenario)
                    )
                    for rank in range(2)
                ]
                try:
                    for worker in workers:
                        worker.start()
                    for worker in workers:
                        worker.join(timeout=25)
                    self.assertEqual([worker.exitcode for worker in workers], [0, 0])
                    self.assertFalse(list(Path(directory).glob("vllm_engram_*")))
                finally:
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                        worker.join()

    def test_preflight_uses_the_same_tp_rank_across_dp_replicas(self):
        group = SimpleNamespace(ranks=[2, 6])
        world = SimpleNamespace(world_size=8)
        checked = []

        def report_failure(_group, stage, error):
            if error:
                raise RuntimeError(f"{stage}: {error}")

        prepare = load_functions(
            ENGRAM,
            ("prepare_engram_sharing",),
            {
                "get_world_group": lambda: world,
                "get_dp_group": lambda: group,
                "get_tensor_model_parallel_rank": lambda: 2,
                "get_node_count": lambda: 1,
                "torch": SimpleNamespace(
                    get_default_device=lambda: SimpleNamespace(type="cuda")
                ),
                "mmap": mmap,
                "_check_engram_shm_capacity": lambda *args: checked.append(args),
                "_raise_engram_group_failures": report_failure,
            },
        )["prepare_engram_sharing"]
        config = SimpleNamespace(
            engram_config=SimpleNamespace(cpu_offload=True, dp_shared_memory=True),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=4,
                data_parallel_size=2,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                enable_elastic_ep=False,
            ),
            load_config=SimpleNamespace(
                load_format="auto", safetensors_load_strategy=None
            ),
        )
        layout = SimpleNamespace(
            layer_ids=(1, 14), num_embeddings=(384006168, 384016682), head_dim=256
        )
        self.assertIs(prepare(config, layout), group)
        self.assertEqual(checked, [(202_758_032_400 + 8 * (mmap.PAGESIZE - 1), 8)])
        group.ranks = [2, 3]
        with self.assertRaisesRegex(RuntimeError, "matching TP shards"):
            prepare(config, layout)
        config.engram_config.dp_shared_memory = False
        self.assertIsNone(prepare(config, layout))

    def test_capacity_requires_physical_ram_even_when_tmpfs_fits(self):
        with self.subTest("host RAM"):
            capacity = load_functions(
                ENGRAM,
                ("_check_engram_shm_capacity",),
                {
                    "os": SimpleNamespace(
                        path=SimpleNamespace(ismount=lambda _: True),
                        statvfs=lambda _: SimpleNamespace(
                            f_bavail=256 * GIB // 4096,
                            f_frsize=4096,
                            f_favail=64,
                        ),
                    ),
                    "_available_engram_host_memory": lambda: 128 * GIB,
                    "SHM_PATH": "/dev/shm",
                    "sys": SimpleNamespace(platform="linux"),
                },
            )["_check_engram_shm_capacity"]
            with self.assertRaisesRegex(RuntimeError, r"188\.8.*128\.0.*RAM"):
                capacity(202_758_049_792)
