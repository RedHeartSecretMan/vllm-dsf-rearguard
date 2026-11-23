# vllm-dsf 生产服务配置前后对照

本文对照 48 上两套真实容器配置。

- 左侧是旧生产容器 `dsf-main-492221be-m0925` 的 `docker inspect` 和 `docker logs` 回执。
- 右侧是当前 vllm-dsf 完整指标套件使用的 `dsf-full-current-20261122` 配置。
- 当前右侧配置是已实测的 GPU 性能配置。它使用 `127.0.0.1:6688` 测试端口，生产部署时把端口改回 `6789`，保留生产模型别名 `deepseek-flash`。
- 旧生产容器已经包含周末前一部分优化，因此它不是仓库 `main` 的裸基线。

## 关键参数

| 项目 | 旧生产服务 | 当前最佳已实测配置 | 变化与说明 |
|---|---|---|---|
| 镜像 | `sha256:a7e510d5a4db99a1803f4969b5de090b07858d54a17929fac9ffb3c2a1d32991` | `sha256:d82fc790c03e3891feb3205e5327fd90ef8c23f2e37da73017971d650a326add` | 当前镜像包含本轮适配和性能候选代码 |
| 模型 | `DeepSeek-V4.1-Flash-1017` | `DeepSeek-V4.1-Flash-1017` | 不变 |
| 模型 revision | `f04a978641fa9bfe6f448ec5da3b2e1c2520a6ad` | `f04a978641fa9bfe6f448ec5da3b2e1c2520a6ad` | 不变 |
| 外部模型别名 | `deepseek-flash` | 测试回执为完整模型名，生产建议继续使用 `deepseek-flash` | 保持现有客户端契约 |
| 对外端口 | `6789` | 测试为 `6688`，生产使用 `6789` | 测试端口与生产端口隔离 |
| Docker 网络 | bridge，端口映射 | bridge，端口映射 | 不变 |
| Docker IPC | private | private | 不变 |
| `/dev/shm` | `64g` | `64g` | 不变 |
| TP / DP / EP | TP4 / DP2 / EP8 | TP4 / DP2 / EP8 | 不变 |
| `max-model-len` | `1048576` | `1048576` | 不变 |
| `max-num-seqs` | `128` | `256` | 当前最明确的容量和高并发调度变化 |
| `max-num-batched-tokens` | `16384` | `16384` | 不变 |
| GPU memory utilization | `0.925` | `0.925` | 不变 |
| KV cache dtype | `fp8_ds_mla` | `fp8_ds_mla` | 当前实现仍使用自有 SM80 KV layout |
| Engram | `{"cpu_offload":true}` | `{"cpu_offload":true}` | 当前完整性能回执没有把 DP shared memory 混入 |
| Prefix caching | 开启 | 开启 | 不变 |
| Chunked prefill | 开启 | 开启 | 不变 |
| 多模态 chunking | 开启，使用 `--no-disable-chunked-mm-input` | 开启，使用 `--no-disable-chunked-mm-input` | 不变 |
| DSpark | k5，greedy，standard rejection | 默认 k5 greedy；temperature > 0 高并发时使用 k5 probabilistic | 右侧完整高并发回执使用 probabilistic，通用生产默认使用 greedy |
| DSpark adaptive verification | 关闭 | 关闭 | adaptive 仍是候选，不进入默认服务 |
| local argmax reduction | 开启 | greedy 时开启，probabilistic 时关闭 | 按 draft sample method 切换 |
| CUDA Graph | `FULL_DECODE_ONLY` | `FULL_DECODE_ONLY` | 不变 |
| CUDA graph capture sizes | `[5,6,10,12,20,24,40,48,80,96,160,192,320,384,640,768]` | 相同 | 不变 |
| EPLB | 开启，async，window 1000，step 3000 | 相同 | 不变 |
| OMP | `8` | `8` | 不变 |
| V2 Model Runner | 日志出现 `Using V2 Model Runner` | 日志出现 `Using V2 Model Runner` | 两边实际都使用 V2，未靠显式环境变量切换 |
| Rust frontend | `VLLM_USE_RUST_FRONTEND=0` | `VLLM_USE_RUST_FRONTEND=0` | 不变 |
| `expandable_segments` | 未设置 | 未设置 | 不变 |
| LMCache MP | 未启用 | 未启用 | 不变 |

## 旧生产启动命令核心

旧生产容器的实际 vLLM 参数为：

```bash
vllm serve /model \
  --served-model-name deepseek-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --max-model-len 1048576 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.925 \
  --kv-cache-dtype fp8_ds_mla \
  --engram-config '{"cpu_offload":true}' \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --generation-config vllm \
  --hf-overrides '{"head_dtype":"float32"}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,80,96,160,192,320,384,640,768],"max_cudagraph_capture_size":768}' \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":0,"use_async":true,"log_balancedness":true,"log_balancedness_interval":500}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false,"use_local_argmax_reduction":true}' \
  --api-server-count 1 \
  --limit-mm-per-prompt '{"image":64,"video":{"count":2,"num_frames":32}}' \
  --media-io-kwargs '{"video":{"num_frames":32,"fps":1}}' \
  --no-disable-chunked-mm-input
```

旧生产容器已经设置了以下优化环境变量：

```text
VLLM_DSV41_BOUNDED_PREFILL_WORKSPACE=1
VLLM_DSV41_DECODE_METADATA_FASTPATH=0
VLLM_DSV41_FUSED_PREFILL_INDICES=1
VLLM_DSV41_PREFILL_THROTTLE=1
VLLM_DSV41_PREFILL_THROTTLE_CADENCE=4
VLLM_DSV41_PREFILL_THROTTLE_MIN_TOKENS=4096
VLLM_DSV41_REUSE_TOPK_RAGGED=0
VLLM_DSV41_STACKED_DRAFT_KV=1
VLLM_DSV4_LOGITS_ROW_CHUNK=0
VLLM_INDEXER_QUERY_SHARD=1
VLLM_SM80_CANDIDATE_MQA=1
VLLM_SM80_EP8_CUSTOM_AG_RS=1
VLLM_FUSED_STABLE_MOE_ALIGN=1
VLLM_AMPERE_DENSE_BF16_MIN_TOKENS=32
```

## 当前最佳生产配置

当前右侧配置对应完整指标套件使用的镜像和服务参数。生产端口和模型别名按现有服务契约替换：

```bash
vllm serve /model \
  --served-model-name deepseek-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --max-model-len 1048576 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.925 \
  --kv-cache-dtype fp8_ds_mla \
  --engram-config '{"cpu_offload":true}' \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --enable-prompt-tokens-details \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --generation-config vllm \
  --hf-overrides '{"head_dtype":"float32"}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[5,6,10,12,20,24,40,48,80,96,160,192,320,384,640,768],"max_cudagraph_capture_size":768}' \
  --enable-eplb \
  --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":0,"use_async":true,"log_balancedness":true,"log_balancedness_interval":500}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false,"use_local_argmax_reduction":true}' \
  --api-server-count 1 \
  --limit-mm-per-prompt '{"image":64,"video":{"count":2,"num_frames":32}}' \
  --media-io-kwargs '{"video":{"num_frames":32,"fps":1}}' \
  --no-disable-chunked-mm-input
```

当前 vllm-dsf 环境变量继续使用旧生产中已验证的优化组合，上面的服务命令只把 `max-num-seqs` 提高到 `256`。temperature 0 的服务请求使用上面的 greedy 配置。temperature 大于 0 且以 C64/C256 为主的流量，可以把 speculative config 中的两项改为：

```json
{
  "draft_sample_method": "probabilistic",
  "use_local_argmax_reduction": false
}
```

这对应已经测过的高并发非零温度 profile。不能把 probabilistic 当作所有 temperature 0 请求的默认值。

## 资源和容量变化

旧生产容器日志显示：

```text
Available KV cache memory: 8.85 GiB
GPU KV cache size: 3,512,120 tokens
Maximum concurrency for 1,048,576 tokens per request: 3.35x
```

当前完整套件容器日志显示：

```text
Available KV cache memory: 7.9 GiB
GPU KV cache size: 3,136,442 tokens
Maximum concurrency for 1,048,576 tokens per request: 2.99x
```

`max-num-seqs=256` 增加了高并发调度和 CUDA Graph 预留，导致每个 DP engine 的 KV capacity 下降。它是吞吐与容量之间的取舍。当前 7 类完整指标、54 个对齐单元全部通过，1M 请求和 C256 场景均成功。

## 可选的主机内存优化

`--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'` 已完成 TP4/DP2 A/B，可以减少约 `188.83 GiB` 的同机 Engram host table 重复占用。它没有形成稳定 GPU 吞吐收益，因此没有放进右侧完整性能回执。主机内存紧张时可以单独开启，并在生产启动后检查 `/dev/shm` 和 `MemAvailable`。

adaptive verification、LMCache MP、native KV-only、Rust frontend 和 `expandable_segments` 没有放进当前最佳生产命令。

## 证据

- 旧生产容器：48 上 `docker inspect dsf-main-492221be-m0925` 和对应 `docker logs`。
- 当前完整套件配置：[full-suite-config.json](../runtime/48-next-perf-20261122/full-suite-config.json)。
- 当前完整套件回执：[full-current/summary.json](../runtime/48-next-perf-20261122/full-current/summary.json)。
- 当前启动日志：[full-current/server.log](../runtime/48-next-perf-20261122/full-current/server.log)。
- 性能决策：[performance.md](performance.md)。
