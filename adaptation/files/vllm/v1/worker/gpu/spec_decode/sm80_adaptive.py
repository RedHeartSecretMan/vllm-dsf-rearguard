# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eligibility for the isolated SM80 V4.1 1017 adaptive experiment."""

import hashlib
import os
from pathlib import Path

EXPERIMENT_ENV = "VLLM_DSV41_SM80_ADAPTIVE_EXPERIMENT"
_CHECKPOINT_SHA256 = {
    "config.json": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "model.safetensors.index.json": "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8",
}


def _verify_checkpoint(model_path):
    for filename, expected in _CHECKPOINT_SHA256.items():
        try:
            with (model_path / filename).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as exc:
            raise ValueError(f"SM80 adaptive requires local 1017 {filename}") from exc
        if digest != expected:
            raise ValueError(f"SM80 adaptive requires the pinned 1017 {filename}")


def require_sm80_adaptive(config, capability):
    """Validate the candidate before attention initialization or graph profiling."""
    spec = config.speculative_config
    if os.environ.get(EXPERIMENT_ENV) != "1" or not (
        spec is not None and spec.enable_adaptive_verification
    ):
        return False

    if capability != (8, 0) or not config.use_v2_model_runner:
        raise ValueError("SM80 adaptive requires a CUDA SM80 GPU and the V2 runner")
    draft = spec.draft_model_config
    if (
        spec.method != "dspark"
        or spec.num_speculative_tokens != 5
        or spec.num_speculative_tokens_per_batch_size is not None
        or spec.draft_sample_method != "greedy"
        or spec.rejection_sample_method != "standard"
        or not spec.use_local_argmax_reduction
        or draft is None
        or "DSparkV41DraftModel" not in draft.architectures
        or getattr(draft.hf_config, "dspark_block_size", None) != 5
        or not getattr(draft.hf_config, "sample_from_anchor", True)
    ):
        raise ValueError(
            "SM80 adaptive requires the 1017 greedy DSpark k5 draft with standard rejection"
        )
    if os.environ.get("VLLM_DSPARK_FUSED_MARKOV") != "0":
        raise ValueError("SM80 adaptive confidence requires VLLM_DSPARK_FUSED_MARKOV=0")
    if (
        config.compilation_config.cudagraph_mode.name != "FULL_DECODE_ONLY"
        or config.compilation_config.mode != 0
    ):
        raise ValueError(
            "SM80 adaptive requires FULL_DECODE_ONLY with compilation mode 0"
        )

    p = config.parallel_config
    if (
        not p.enable_expert_parallel
        or p.all2all_backend != "allgather_reducescatter"
        or p.tensor_parallel_size != 4
        or p.data_parallel_size != 2
        or spec.draft_tensor_parallel_size != 4
        or p.pipeline_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or p.use_ubatching
        or p.enable_batch_sharded_sampling
    ):
        raise ValueError(
            "SM80 adaptive requires TP4/DP2/EP8 without PP, CP, ubatching or sampling shards"
        )
    if (
        config.lora_config is not None
        or config.kv_transfer_config is not None
        or config.cache_config.cache_dtype != "fp8_ds_mla"
        or spec.kv_cache_dtype not in (None, "fp8_ds_mla")
        or config.attention_config.use_non_causal
        or config.attention_config.resolve_indexer_kv_dtype("fp8") != "fp8"
    ):
        raise ValueError(
            "SM80 adaptive requires causal FP8 indexer and fp8_ds_mla without LoRA or KV transfer"
        )
    for key in (
        "VLLM_INDEXER_QUERY_SHARD",
        "VLLM_SM80_CANDIDATE_MQA",
        "VLLM_SM80_CANDIDATE_MQA_COMPACT",
        "VLLM_DSV41_DECODE_METADATA_FASTPATH",
        "VLLM_DSV41_REUSE_TOPK_RAGGED",
        "VLLM_DSV41_PREFILL_THROTTLE",
        "VLLM_DSV4_LOGITS_ROW_CHUNK",
    ):
        if os.environ.get(key, "0") != "0":
            raise ValueError(f"SM80 adaptive requires {key}=0")
    model = config.model_config
    if (
        "DeepseekV41ForCausalLM" not in model.architectures
        or draft.model != model.model
    ):
        raise ValueError(
            "SM80 adaptive requires the same local DeepSeek V4.1 1017 target and draft snapshot"
        )
    _verify_checkpoint(Path(model.model))
    return True


def sm80_adaptive_batch(active, *, has_prefill, has_drafts):
    """Return (trim_drafts, vote_eager) before any DP mode collective."""
    if active and has_prefill:
        return False, True
    return has_drafts, False
