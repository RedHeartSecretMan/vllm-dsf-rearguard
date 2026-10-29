# vLLM 性能优化计划

本文以最新 `main` 提交 `492221be` 为代码基线，围绕 DeepSeek-V4.1-Flash-1017 在 `8×A100-SXM4-80GB / NVSwitch` 上的服务性能推进。每项工作遵循：

> 功能修复或启用 → 正确性准入 → 性能 A/B → 正式矩阵 → 推荐配置决策

性能结论必须绑定代码 SHA、镜像 digest、模型 revision、启动参数、输入、seed、并发、缓存策略和原始证据。性能测量至少进行 3 轮独立正式测量；功能、质量、稳定性和性能证据分开保存。

## 固定基线

| 项目 | 基线 |
|---|---|
| 并行 | `TP4 / DP2 / EP8` |
| 上下文 | `1M` 主矩阵，`262K` 补充矩阵 |
| chunk | `16K` |
| 显存水位 | `0.90` |
| 全局序列数 | `256` |
| DSpark | `k0` 基线、`k5` 正式路径 |
| 默认能力 | `stacked-kv`、`bounded-prefill`、OMP 8、EPLB、custom AG/RS |
| KV 策略 | 冷 KV；历史复用单独标记 |
| 请求指标 | 输入/输出/总 TPS、TTFT、TPOT、ITL P95、E2E P95、成功率、KV 和显存峰值 |

已有证据确认 `bounded-prefill` 应保留：1M C128 输出吞吐 `+52.6%`，C256 `+23.4%`，单请求 prefill 约 `-0.1%`。基线数据见 [性能数据](performance.md)。

## 阶段一：DSpark confidence adaptive

当前 `launch.py` 固定 `draft_sample_method="greedy"`、`enable_adaptive_verification=false`。模型代码已有 `confidence_head` 和 `compute_confidence()`，但 adaptive verification 因 DeepSeek indexer 不支持 device 侧 verification 裁剪而未启用。

| ID | 工作 | 代码边界 | 准入结果 |
|---|---|---|---|
| DSP-F1 | 暴露 adaptive verification opt-in | `launch.py` speculative config | `k5 + adaptive` 可生成启动命令，默认仍为关闭 |
| DSP-F2 | 校验 confidence head 权重 | `models/deepseek_v4_1/nvidia/dspark.py` | 缺失权重硬失败，加载后 confidence shape/range 正确 |
| DSP-F3 | 支持 device/CPU query-lens mismatch | `v1/attention/backends/mla/indexer.py` | 动态 `query_start_loc`、verification 长度和 sparse metadata 一致 |
| DSP-F4 | 修复 TP/DP collective 边界 | indexer metadata、scheduler、collective | 不同 request confidence 下无 rank divergence 或 hang |
| DSP-F5 | 适配 CUDA Graph replay | indexer、sparse SWA、graph buffer | capture/replay、padding、变长 query 均稳定 |
| DSP-F6 | 输出和质量准入 | acceptance、quality gate | 与 `k5 + greedy` 无输出 digest、PPL、工具和多模态回归 |

### DSpark 性能矩阵

| 模式 | 并发 | 输入 |
|---|---:|---|
| `k0` | C1/C64/C256 | ISL 17 / 1024 / 16384 |
| `k5 + greedy` | C1/C64/C256 | ISL 17 / 1024 / 16384 |
| `k5 + adaptive confidence` | C1/C64/C256 | ISL 17 / 1024 / 16384 |
| `k5 + adaptive confidence` | C64 | 262K prefill + decode |
| `k5 + adaptive confidence` | C64 | 1M mixed load |

额外记录 accepted draft tokens、实际 verification tokens、confidence 分布、draft/verify latency、显存和 Graph 稳定性。`k7` 不进入矩阵；`k3` 只在兼容性需要时作为探针，不作为优化目标。

## 阶段二：调度与 metadata

### `decode-metadata` / `ragged-reuse`

当前两者同时开启时 C256 输出吞吐下降 `8.5%`，缺少单项拆分。先完成四臂功能与性能对照：

| decode-metadata | ragged-reuse | 结果 |
|---|---|---|
| off | off | baseline |
| on | off | 单项收益/回归 |
| off | on | 单项收益/回归 |
| on | on | 组合收益/回归 |

覆盖 1M C128/C256、长 prefill、decode/prefill mixed，记录 metadata build、cache hit、GPU allocation 和吞吐。

### `prefill-throttle`

现有固定节拍在混合负载中让 decode 输出吞吐 `+16.9%`，但 prefill `-17.7%`、整机总吞吐 `-11%`。先将固定策略扩展为可观测的调度策略，再进行：

| 策略 | 参数 |
|---|---|
| off | 无节流 |
| fixed | cadence 1/2/4/8 |
| fixed | `min_tokens` 2K/4K/8K |
| adaptive | decode queue、TTFT/ITL SLO 驱动 |

负载固定为 C64 decode + 262K prefill，并补充 1M mixed load。报告两条流吞吐、TTFT、TPOT、ITL P95、排队、抢占和重计算。

### Indexer query sharding

在 A100/NVSwitch 上测量：

```text
TP4 / TP8
×
MIN_SHARD_TOKENS = 1024 / 2048 / 4096 / 8192
×
prefill = 1K / 2K / 4K / 8K / 16K
```

同时覆盖 decode query group、空 shard、collective 参与者一致性和 CUDA Graph 边界。

## 阶段三：多模态 chunk 动态选择

已有对照显示 16K 对 video/image 有收益，但 text 和长 prefill 变慢。因此增加动态选择策略：

| 策略 | text | image | video | 长 prefill |
|---|---|---|---|---|
| 固定 16K | 测量 | 测量 | 测量 | 测量 |
| 固定 32K | 测量 | 测量 | 测量 | 测量 |
| media-aware | 按媒体类型 | 按媒体类型 | 按媒体类型 | 单独规则 |
| concurrency-aware | 按 C | 按 C | 按 C | 按 ISL |

覆盖 text C128/C256、image C4、video C1/C4、260K cold prefill 和混合负载。功能准入必须检查 frame boundary、视频时间映射、视觉输出和多轮历史。

## 阶段四：1M workspace、KV 和 CUDA Graph

| 维度 | 对照 |
|---|---|
| bounded workspace | on/off |
| workspace 上界 | 单请求上界/多请求上界 |
| context | 262K/524K/1M |
| concurrency | C128/C256 |
| capture size | 当前配置/缩小/扩大 |
| graph mode | full decode / eager fallback |
| metadata | build/copy/reuse |
| KV admission | waiting/running/swapped |

当前 262K → 1M 的 260K prefill 下降约 `8.9%`，C256 输出吞吐下降约 `41.7%`，单请求 decode 基本不变。该阶段先做 workspace、KV admission、metadata、scheduler 和 graph pool 分解，再决定代码修复点。

## 统一验收矩阵

每个候选按照以下顺序进入正式矩阵：

1. CPU/static contract：配置、manifest、启动计划和 metadata contract；
2. GPU functional gate：启动、短请求、工具、图文、长输入和 DSpark 输出；
3. correctness gate：逐请求 output digest、PPL/质量、成功率和异常日志；
4. targeted performance：针对候选的最小矩阵；
5. full performance：纳入 262K/1M、C1/C64/C128/C256 和 mixed load；
6. recommendation：记录收益、代价、适用负载、默认配置和未覆盖范围。

正式结果不得把不同 workload 的 TPS 平均成单一分数；decode 优化同时报告 prefill 和整机代价，显存优化同时报告 KV 容量与 Graph/运行时峰值。
