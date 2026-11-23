#!/usr/bin/env bash
set -Eeuo pipefail

# Official vLLM DeepSeek-V4.1-Flash-1017 production service for:
#   8x NVIDIA A100-SXM4-80GB / NVSwitch
#   TP4 / DP2 / EP8
#
# The external service contract is preserved:
#   host port 6789, model alias deepseek-flash, API key file and Docker log rotation.
#
# Validated on 48 with official vLLM commit f8b29fab4. The CPU KV budget is
# per DP engine. 256 GiB per DP gives approximately 512 GiB in total.

IMAGE="${IMAGE:-sha256:5110d24384dc2977be947a398d4a31c21695c49acb78310ae379bdea408af6b0}"
MODEL_PATH="${MODEL_PATH:-/data1/wanghao/Projects/Datas/models/deepseek/DeepSeek-V4.1-Flash-1017}"
CONTAINER_NAME="${CONTAINER_NAME:-dsf-official-a100-offloading}"
DSF_ROOT="${DSF_ROOT:-/data1/wanghao/Projects/Scripts/dsf}"
ACTIVE_DEPLOYMENT_FILE="${ACTIVE_DEPLOYMENT_FILE:-$DSF_ROOT/active-deployment}"
DEPLOYMENT_DIR="${DEPLOYMENT_DIR:-$(cat "$ACTIVE_DEPLOYMENT_FILE")}"
API_KEY_FILE="${API_KEY_FILE:-$DEPLOYMENT_DIR/api-key}"
PORT="${PORT:-6789}"
MODEL_NAME="${MODEL_NAME:-deepseek-flash}"
CPU_KV_BYTES="${CPU_KV_BYTES:-274877906944}"

if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'MODEL_PATH does not exist or is not a directory: %s\n' "$MODEL_PATH" >&2
  exit 1
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  printf 'PORT must be an integer from 1 to 65535: %s\n' "$PORT" >&2
  exit 1
fi

if [[ ! "$CPU_KV_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  printf 'CPU_KV_BYTES must be a positive integer: %s\n' "$CPU_KV_BYTES" >&2
  exit 1
fi

if [[ ! -f "$API_KEY_FILE" ]]; then
  printf 'API_KEY_FILE does not exist: %s\n' "$API_KEY_FILE" >&2
  exit 1
fi

API_KEY="$(tr -d '\r\n' < "$API_KEY_FILE")"
if [[ -z "$API_KEY" ]]; then
  printf 'API_KEY_FILE is empty: %s\n' "$API_KEY_FILE" >&2
  exit 1
fi

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  printf 'Container already exists: %s\n' "$CONTAINER_NAME" >&2
  printf 'Stop and remove it explicitly before rerunning this script.\n' >&2
  exit 1
fi

ENGRAM_CONFIG='{"cpu_offload":true,"dp_shared_memory":true}'
KV_TRANSFER_CONFIG=$(cat <<JSON
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": ${CPU_KV_BYTES},
    "blocks_per_chunk": 4
  }
}
JSON
)
SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"standard","enable_adaptive_verification":false,"use_local_argmax_reduction":false}'
COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,80,96,160,192,320,384,640,768],"max_cudagraph_capture_size":768}'
EPLB_CONFIG='{"window_size":1000,"step_interval":3000,"num_redundant_experts":0,"use_async":true,"log_balancedness":false}'

# Keep the current DSF service's Responses API lifecycle and one API server.
export VLLM_ENABLE_RESPONSES_API_STORE=1

# OffloadingConnector is incompatible with expandable_segments unless the
# CuMem allocator is explicitly enabled. Do not set expandable_segments here.
# Host IPC is required because each DP engine creates its CPU KV mmap in the
# host /dev/shm. The two DP regions are intentionally separate.
docker run \
  --name "$CONTAINER_NAME" \
  --label vllm-dsf-rearguard.owner=vllm-dsf-rearguard \
  --restart unless-stopped \
  --pull never \
  --init \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  -p "${PORT}:8000" \
  --mount "type=bind,src=${MODEL_PATH},dst=/model,readonly" \
  --mount "type=volume,src=${CONTAINER_NAME}-cache,dst=/workspace" \
  --log-opt max-size=100m \
  --log-opt max-file=5 \
  --env DO_NOT_TRACK=1 \
  --env HF_DATASETS_OFFLINE=1 \
  --env HF_HUB_OFFLINE=1 \
  --env NCCL_DEBUG=INFO \
  --env OMP_NUM_THREADS=8 \
  --env PYTHONUNBUFFERED=1 \
  --env TILELANG_CACHE_DIR=/workspace/compiled-cache/tilelang \
  --env TORCHINDUCTOR_CACHE_DIR=/workspace/compiled-cache/torchinductor \
  --env TRANSFORMERS_OFFLINE=1 \
  --env TRITON_CACHE_DIR=/workspace/compiled-cache/triton \
  --env VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --env VLLM_HTTP_TIMEOUT_KEEP_ALIVE=210 \
  --env VLLM_NO_USAGE_STATS=1 \
  --env VLLM_USE_RUST_FRONTEND=1 \
  --env VLLM_ENABLE_RESPONSES_API_STORE=1 \
  --entrypoint vllm \
  "$IMAGE" \
  serve \
  /model \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-server-count 1 \
  --api-key "$API_KEY" \
  --tokenizer-mode deepseek_v41 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --max-model-len 1048576 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --engram-config "$ENGRAM_CONFIG" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --speculative-config "$SPECULATIVE_CONFIG" \
  --compilation-config "$COMPILATION_CONFIG" \
  --enable-eplb \
  --eplb-config "$EPLB_CONFIG" \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --generation-config vllm \
  --hf-overrides '{"head_dtype":"float32"}' \
  --enable-prompt-tokens-details \
  --no-disable-chunked-mm-input \
  --mm-encoder-tp-mode data
