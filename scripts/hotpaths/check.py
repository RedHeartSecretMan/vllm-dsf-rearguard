"""Exercise the actual SM80 hot-path methods with CPU tensors.

Run with an empty CUDA_VISIBLE_DEVICES::

    python3 -m scripts.hotpaths.check --output runtime/hotpaths-cpu.json
    python3 -m scripts.hotpaths.check --native-loader   # needs the vLLM image

CUDA and Marlin kernels are not exercised.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from scripts.hotpaths.fixtures import (
    assert_loaded,
    assert_ragged,
    attention_fixture,
    context_method,
    draft_fixture,
    ragged_reference,
)
from scripts.hotpaths.prefix import metadata_prefix
from scripts.source_extract import load_function, run_function_tree

ROOT = Path(__file__).resolve().parents[2]
FILES = ROOT / "adaptation/files"


class Operations(TorchDispatchMode):
    def __init__(self):
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[func._schema.name] += 1
        return func(*args, **(kwargs or {}))


def metadata_function(path, cls, kind, split, enabled):
    namespace = dict(
        torch=torch,
        _FAST_DECODE_METADATA=enabled,
        split_decodes_and_prefills=lambda *a, **k: split,
        get_compressed_slot_mapping=lambda *a, out: out[: a[0]],
    )
    return run_function_tree(
        metadata_prefix(path, cls, kind), path, namespace, "build"
    )


def check_metadata():
    cases = []
    path = FILES / "vllm/v1/attention/backends/mla/indexer.py"
    for ratio in (1, 2, 4):
        for prefill in (False, True):
            split = (1, 1, 1, 4) if prefill else (2, 0, 2, 0)
            starts = torch.tensor([0, 1, 5] if prefill else [0, 1, 2])
            common = NS(
                num_reqs=2,
                num_actual_tokens=sum(split[2:]),
                query_start_loc=starts,
                query_start_loc_cpu=starts,
                seq_lens=torch.tensor([1025, 2051], dtype=torch.int32),
                slot_mapping=torch.arange(sum(split[2:])),
                block_table_tensor=torch.arange(32).view(2, 16),
                dcp_local_seq_lens=None,
            )
            owner = NS(
                decode_threshold=6,
                use_flattening=True,
                supports_varlen=True,
                use_pcp=False,
                compress_ratio=ratio,
                kernel_block_size=None,
                pcp_world_size=1,
                kv_cache_spec=NS(num_states=16),
                compressed_slot_mapping_buffer=torch.zeros(16, dtype=torch.int64),
            )
            outputs, counts = [], []
            for enabled in (False, True):
                build = metadata_function(
                    path, "DeepseekV32IndexerMetadataBuilder", "indexer", split, enabled
                )
                with Operations() as trace:
                    outputs.append(build(owner, 0, common))
                counts.append(trace.counts)
            assert counts[0]["aten::floor_divide"] == int(ratio > 1)
            assert counts[1]["aten::floor_divide"] == int(ratio > 1 and prefill)
            for key in ("compressed_slot_mapping", "indexer_block_table"):
                torch.testing.assert_close(
                    outputs[0][key], outputs[1][key], rtol=0, atol=0
                )
            if prefill:
                torch.testing.assert_close(
                    outputs[0]["compressed_seq_lens"],
                    outputs[1]["compressed_seq_lens"],
                    rtol=0,
                    atol=0,
                )
            cases.append(
                dict(
                    kind="indexer-prefix",
                    ratio=ratio,
                    prefill=prefill,
                    old_divisions=counts[0]["aten::floor_divide"],
                    new_divisions=counts[1]["aten::floor_divide"],
                )
            )
    path = FILES / "vllm/v1/attention/backends/mla/sparse_swa.py"
    for size in (0, 1, 6, 33, 192):
        slots = torch.arange(size) - size // 2
        common = NS(
            seq_lens=torch.tensor([1025]),
            seq_lens_cpu_upper_bound=torch.tensor([1025]),
            query_start_loc=torch.tensor([0, size]),
            query_start_loc_cpu=torch.tensor([0, size]),
            block_table_tensor=torch.zeros((1, 32), dtype=torch.int32),
            slot_mapping=slots,
            token_to_req_indices=lambda buffer, size=size: buffer[:size],
        )
        counts = []
        for enabled in (False, True):
            owner = NS(
                decode_threshold=6,
                token_to_req_indices=torch.zeros(size + 4, dtype=torch.int32),
                is_valid_token=torch.ones(size + 4, dtype=torch.bool),
            )
            build = metadata_function(
                path,
                "DeepseekSparseSWAMetadataBuilder",
                "swa",
                (1, 0, size, 0),
                enabled,
            )
            with Operations() as trace:
                output = build(owner, 0, common)["is_valid_token"]
            assert output.data_ptr() == owner.is_valid_token[:size].data_ptr()
            assert owner.is_valid_token[size:].all()
            torch.testing.assert_close(output, slots >= 0, rtol=0, atol=0)
            counts.append(trace.counts)
        assert counts[0]["aten::copy_"] == 1
        assert counts[1]["aten::copy_"] == 0
        cases.append(dict(kind="swa-prefix", size=size, old_copies=1, new_copies=0))
    return cases


def check_reuse():
    counts = Counter()

    def pack(*args):
        counts["pack"] += 1
        return ragged_reference(*args)

    cases = []
    for ratio in (1, 2):
        source, consumers, swa, meta = attention_fixture("cpu", pack, ratio=ratio)
        for step in range(3):
            before = counts["pack"]
            actual = source.build(swa, meta)
            for consumer in consumers:
                assert consumer.build(swa, meta) is actual
            assert counts["pack"] - before == 1
            expected = ragged_reference(
                source.topk_indices_buffer[: swa.num_decode_tokens],
                swa.token_to_req_indices[: swa.num_decode_tokens],
                meta.block_table,
                meta.block_size // ratio,
                swa.is_valid_token[: swa.num_decode_tokens],
            )
            assert_ragged(actual, expected)
            source.topk_indices_buffer[:, 0].add_(1)
            meta.block_table.add_(1)
            swa.is_valid_token.logical_not_()
            cases.append(dict(kind="reuse", ratio=ratio, step=step, layers=5, packs=1))
        # Same address/stride with a narrower logical top-k view must not reuse.
        source.build(swa, meta)
        consumers[0].topk_indices_buffer = source.topk_indices_buffer[:, :128]
        before = counts["pack"]
        consumers[0].build(swa, meta)
        assert counts["pack"] == before + 1
        # Different block-table storage and changed active batch are separate entries.
        for changed in (
            NS(block_size=meta.block_size, block_table=meta.block_table.clone()),
            NS(block_size=meta.block_size * 2, block_table=meta.block_table),
        ):
            before = counts["pack"]
            consumers[1].build(swa, changed)
            assert counts["pack"] == before + 1
        swa.num_decode_tokens = 2
        before = counts["pack"]
        consumers[1].build(swa, meta)
        assert counts["pack"] == before + 1
        for consumer in consumers:
            consumer._reuse_topk_ragged = False
        before = counts["pack"]
        for consumer in consumers:
            consumer.build(swa, meta)
        assert counts["pack"] == before + len(consumers)
    return cases


def check_empty_pack():
    pack = load_function(
        FILES / "vllm/models/deepseek_v4_1/amd/rocm.py",
        "compute_global_topk_ragged_indices_and_indptr",
        {"torch": torch},
    )
    actual = pack(
        torch.empty((0, 1, 512), dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.zeros((1, 16), dtype=torch.int32),
        32,
        torch.empty(0, dtype=torch.bool),
    )
    assert actual[0].numel() == actual[2].numel() == 0
    assert actual[1].tolist() == [0]
    return [{"kind": "empty-ragged-pack", "status": "PASS"}]


def check_projection():
    torch.manual_seed(1017)
    hidden, head, qrank = 64, 16, 32
    weights = [torch.randn(qrank + head, hidden) * 0.05 for _ in range(3)]
    stacked = torch.cat([weight[qrank:] for weight in weights])
    method = context_method()
    cases = []
    for rows in (0, 1, 5, 6, 31, 32, 64, 129):
        for mappings in (
            None,
            [None, torch.arange(rows) + 11, torch.arange(rows) + 29],
        ):
            recorded, ops, traces = [], [], []
            for enabled in (False, True):
                layers = []
                for weight in weights:
                    attn = NS(
                        q_lora_rank=qrank,
                        fused_wqa_wkv=lambda x, w=weight: (
                            torch.nn.functional.linear(x, w),
                            None,
                        ),
                        kv_norm=lambda x: torch.nn.functional.rms_norm(
                            x, (head,), eps=1e-6
                        ),
                    )
                    layers.append(NS(attn=attn))
                owner = NS(
                    layers=layers,
                    num_dspark_layers=3,
                    config=NS(head_dim=head),
                    context_wkv_proj=(lambda x: torch.nn.functional.linear(x, stacked))
                    if enabled
                    else None,
                )
                # Same input for both paths; change between cases to expose stale results.
                x = (
                    torch.arange(rows * hidden, dtype=torch.float32).view(rows, hidden)
                    % 17
                    / 17
                )
                with Operations() as trace:
                    method(owner, x, torch.arange(rows), mappings)
                recorded.append(
                    [getattr(layer.attn, "recorded", None) for layer in layers]
                )
                ops.append(trace.counts["aten::linear"])
                traces.append(dict(trace.counts))
            assert ops == [3, 1], (ops, traces)
            for old, new in zip(*recorded, strict=True):
                if old is None:
                    assert new is None
                else:
                    torch.testing.assert_close(new[0], old[0], rtol=1e-5, atol=1e-6)
                    for index in (1, 2):
                        torch.testing.assert_close(
                            new[index], old[index], rtol=0, atol=0
                        )
            cases.append(
                dict(
                    kind="context-projection",
                    rows=rows,
                    profiling=mappings is None,
                    old_linear_calls=3,
                    new_linear_calls=1,
                )
            )
    return cases


def check_native_loader():
    # Uses real installed vLLM parameter classes and shard loading; backend selection
    # is stubbed only because no CUDA device is exposed. Does not run native GEMM.
    for enabled in (False, True):
        owner, weights, references = draft_fixture("cpu", enabled=enabled)
        owner.load_weights(iter(weights))
        assert_loaded(owner, references)
        if enabled:
            owner, weights, _ = draft_fixture("cpu", enabled=True)
            missing = [
                (name, value)
                for name, value in weights
                if name != "mtp.1.attn.wkv.scale"
            ]
            try:
                owner.load_weights(iter(missing))
            except ValueError as error:
                assert "Incomplete DSpark context KV weights" in str(error)
            else:
                raise AssertionError("Incomplete native scale shard was accepted")
    return {
        "status": "PASS",
        "native_gemm": "NOT_RUN",
        "checks": [
            "TP rank 3/4 still loads full Wkv",
            "FP8 bytes and UE8M0 scale shards",
            "random load order",
            "missing scale rejected",
            "disabled fallback",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-loader", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("This CPU test must run with CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(1)
    cases = check_metadata() + check_reuse() + check_empty_pack() + check_projection()
    native = check_native_loader() if args.native_loader else {"status": "NOT_RUN"}
    report = {
        "status": "PASS",
        "torch": torch.__version__,
        "cuda_available": False,
        "gpu_validation": "NOT_RUN",
        "native_loader": native,
        "scope": "Actual metadata prefixes and projection/control methods with CPU tensors; pack uses an independent CPU reference.",
        "manifest_sha256": hashlib.sha256(
            (ROOT / "adaptation/manifest.json").read_bytes()
        ).hexdigest(),
        "cases": cases,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
