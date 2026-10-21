# 实验协议

## 顺序

1. 完成下载，按固定 revision 全量 SHA256 核对。记录驱动、镜像 ID、源码 SHA、
   GPU 拓扑、功率/时钟、CPU/NUMA 和主存。
2. `tp8 / ep8`、DSpark 关闭、eager、control 开关先过文本、工具与图像 smoke。
   这只是本候选的保守对照，不能称为官方无损基线。
4. 在相同请求下逐项打开 Graph、k5、性能 kernel；再比较 `tp4dp2 / ep8`。
   每个候选均保留同样的权重、模态、采样、缓存和全局调度预算。
5. 功能通过后跑性能矩阵。复用现有精度集完成配对质量回归，另做并发混合
   prefill/decode、工具、图像和长文 soak；发现误差/挂死先定位再调优。

## 第一轮矩阵

| 维度 | 值 |
|---|---|
| TP/DP/EP | 8/1/8，4/2/8 |
| DSpark | 关闭、5；7 标为 `UNSUPPORTED` |
| 输入 token | 1024、32768、131072 |
| 客户端并发 | 1、8、32、64 |
| 输出 token | 固定 1024，`ignore_eos=true`，temperature 0 |
| 缓存 | 冷、热分别计时，核对服务端 cached token usage |
| 重复 | 每档至少 3 次，交错顺序，报告中位数和范围 |

先用 1K/C1、1K/C32、32K/C8 筛掉失败或明显较差候选，再扩展。
若 C64 仍有吞吐提升且 P95 可接受，再单独测 C128。
512K/1M 另开容量与长文检索实验：最大上下文是输入输出合计；
不能用“服务能启动”或排队完成证明所有请求同时驻留 KV。

`--max-seqs` 是全局预算，8192 是各引擎的 chunk budget；
`tp4dp2 / ep8` 的总 prefill 工作能力可能不同，必须连同该差异报告。
`--memory 0.90` 起步，显存水位调整属于单独实验。

## 当前目标机 GPU 首轮结果

固定 digest 镜像通过 Daocloud 前缀拉取后，完成了 TP8、TP4DP2/EP8、CUDA Graph、文本、工具、图像和长上下文 smoke。当前推荐配置为 `tp4dp2 / ep8`、DSpark k5、`stacked-kv`、全局 `max-seqs=256`。在固定 prompt、temperature 0、输出上限 128 token 的短请求矩阵中：

| 全局并发 | 成功率 | 整机输出吞吐 | TTFT P50 / P95 |
|---:|---:|---:|---:|
| 64 | 64/64 | 约 2358 token/s | 约 1.99 / 2.01 s |
| 128 | 128/128 | 约 4496 token/s | 约 1.37 / 1.78 s |
| 256 | 256/256 | 约 7270 token/s | 约 0.71 / 1.24 s |

这是短合成请求的首轮 GPU 结果，不等同于 1024 token 生产吞吐，也不代替自然任务质量回归。
另有专用 1M context 配置完成 1,000,016 prompt 单请求，首 token 延迟约 223.7 s；该配置用于容量验收，不作为高并发吞吐配置。
## 性能度量

`scripts.benchmark` 使用固定 seed 的普通 token ID 合成请求，可严格控制输入长度；
不是自然语言质量测试，也不是官方短 coding 吞吐的直接复现。

- 整机输出吞吐 = 所有成功请求的 **服务端 completion_tokens** / 整个测量墙钟。
- TTFT = 客户端发起到第一个含文本的 SSE chunk。记录 P95。
- TPOT / decode = 根据首末文本 chunk 时间和服务端 token 数估算。
  DSpark 可能每个 chunk 输出多个 token，客户端值不能等同于内核 decode latency。
- 缺少 `[DONE]`、finish reason、usage、输出不足、输入计数不符均为失败。
- 失败率单独报告；失败请求不进入成功请求延迟分位数，失败率不为零则整组不通过。
- 冷缓存要求每个请求 cached_tokens=0；热缓存先用同 prompt 预填，再要求 cached_tokens>0。
  若 usage 缺失，返回 FAIL，不能把想要的缓存状态当观测值。
- 固定 seed 的冷请求重跑前要重启候选服务清空缓存，或用另一组事先声明的 seed；
  同一次 A/B 的对应冷组要使用同一 seed。

自然任务另行记录思考开关/effort、自然 EOS、工具 token 和内容长度；
不能将 thinking-off 单流 decode 与 thinking-on 整机吞吐直接排序。

## 质量边界

固定官方权重不等于运行路径无差异。FP8 KV、FP4 Marlin、BF16 dense、
索引筛选、通信归约和 DSpark verification 均需检查。
本项目没有添加量化权重转换、视觉降采样覆盖或 adaptive verification。
最小 smoke 不能代替 GSM8K/GPQA/代码或真实 agent 任务的配对回归；
测试前约定容许波动和失败阈值，保留题级结果，不能只用几条输出声称精度不下降。

## 运行边界

每个候选使用独立的容器名、端口和回执目录。测试结束后保留镜像、启动计划、日志和指标，
不把运行时文件、权重或凭据写入 Git。

## DeepInfer 对照边界

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) 当前公开 benchmark 将 prefill 测在 4 张 SM100、decode 测在 64 张 SM100，并使用独立的 EP/DP scale-up 配置。它适合作为 DeepSeek V4.1 Flash 的 infra 设计和指标口径参考，不能把其中的 token/s 直接换算成 8×A100 的成绩。

基于当前矩阵的差距分析和下一轮验收顺序见 [DeepInfer 对标与下一轮优化](deepinfer-gap-analysis.md)。

本项目的结果固定在 8×A100-SXM4-80GB / NVSwitch、TP4DP2/EP8、真实 1017 权重和本地 vLLM 适配层上；短请求矩阵使用服务端 completion tokens，prefill 使用首 token 延迟近似值，decode 同时报告单请求 TPOT 和整机并发吞吐。

## 1M 上限严格对照

在完全相同的 TP4DP2/EP8、DSpark k5、stacked-kv、CUDA Graph、global `max-seqs=256` 配置下，只切换 `context=262144/1048576`：260K prefill 近似吞吐为约 8498 / 7740 token/s，单请求 decode 约 279.5 / 279.3 token/s，C256 整机吞吐约 7270 / 4240 token/s。1M 适合容量请求，262K 更适合作为当前高并发默认值。

## 1M bounded prefill workspace A/B

为减少 1M 上限对高并发的静态工作区占用，新增 `--experimental bounded-prefill`。它把 indexer prefill workspace 的预留下限设为 `max_model_len / compress_ratio`，正好覆盖一个完整 1M 请求；更大的多请求 prefill 仍由 metadata builder 分块处理。该开关不改变模型上限、KV 格式或 decode kernel。

同一固定 digest 镜像、TP4DP2/EP8、DSpark k5、stacked-kv、CUDA Graph、全局 C256、1K 输入和 128 输出的可复现脚本结果如下。每组请求均为 0 失败并通过冷缓存校验；C256 使用相同 seed 8080，unbounded 与 bounded 分别使用独立新容器。

| profile | C128 token/s | C256 token/s | 260K prefill token/s | 1M prefill token/s |
|---|---:|---:|---:|---:|
| 1M unbounded | 约 843 | 约 1109 | — | — |
| 1M bounded | 约 1286 | 约 1369 | 约 9067 | 约 4772 |

bounded 在当前统一脚本口径下将同 seed 的 C256 提升约 23%，并保持 1M 单请求通过；它适合作为 1M 高并发 profile。旧的短矩阵使用另一套客户端计时口径，不能与本表直接合并排序。

## C256 深度调优 A/B

为复现历史短矩阵，使用固定 10-token prompt，先跑 C128 热身，再测 C256；每个请求输出 128 token，Graph、EPLB、DSpark k5、stacked-kv 和 bounded workspace 保持不变。

| 变体 | C128 token/s | C256 token/s | 结论 |
|---|---:|---:|---|
| OMP 8 + EPLB + custom AG/RS | 约 3016 | 约 4338 | 当前推荐组合 |
| 关闭 EPLB | 约 2952 | 约 4147 | 下降约 4.4%，保留 EPLB |
| 关闭 custom AG/RS | 约 2335 | 约 3815 | 下降约 12%，保留 custom AG/RS |
| OMP 1 | 约 2493 | 约 2753 | 下降约 36.5%，保留 OMP 8 |

另测 `gpu-memory-utilization=0.95`：KV cache 从约 10.65 GiB 增至约 14.61 GiB，但随机冷 C256 约 1057 token/s，低于 0.90 profile 的约 1369 token/s；显存增加没有转化为吞吐，继续使用 0.90。

当前结论是：1M 高并发应使用 bounded workspace、OMP 8、EPLB 和 SM80 custom AG/RS。历史 4240 token/s 与本轮固定 prompt 热 C256 约 4338 token/s 同属服务层固定 prompt 口径；随机冷 prompt 结果单独用于评估 KV 规划，不混入该矩阵。

最后将 `decode-metadata` 与 `ragged-reuse` 同时开启做组合 A/B。功能 smoke 和 256 请求均通过，但热 C256 约 3969 token/s，低于基线约 4338 token/s；两者继续保持显式开关关闭，不进入推荐 profile。

## ISL / OSL 矩阵（2026-10-21）

当前最佳 1M bounded profile 使用项目内 `scripts/benchmark_matrix.py` 完成了完整矩阵。所有请求 `OSL=1000`、temperature 0、`ignore_eos=true`，服务端返回 usage 且成功请求计数完整。`nature` 数据由固定自然语言语料经目标 tokenizer 生成 token 序列，`random` 使用普通 token ID 随机序列；两种数据都保持目标 ISL 精确。

- 单流：`ISL=4096/16384/65536/262144/1047000`，并发 1。
- 并发：`ISL=256/1024/4096/16384`，并发 `4/16/64/256`。
- 1,048,576 token 输入加上 1,000 token 输出超过 1,048,576 上限，因此用 `ISL=1047000 + OSL=1000` 表示可执行的 1M 输入变体。

runner 为每个请求记录 TTFT、TPOT、ITL、输入/输出 token 数；summary 中的 `prefill_tps` 是总输入 token / 测量墙钟，`decode_tps` 是总 completion token / 测量墙钟，逐请求纯 decode TPS 保存在 `requests[].decode_tps`。ITL 是 SSE inter-event 间隔，TPOT 使用服务端 completion token 数，二者保留各自口径。

当前矩阵的代表性 C256 结果（summary aggregate TPS；TTFT 为 P50/P95）如下：

| 数据 | ISL | prefill TPS | decode TPS | TTFT P50 / P95 | TPOT P50 | ITL P50 |
|---|---:|---:|---:|---:|---:|---:|
| random | 256 | 1288 | 5033 | 2.11 / 3.42 s | 35.6 ms | 199.6 ms |
| nature | 256 | 1612 | 6296 | 2.22 / 3.58 s | 34.7 ms | 202.1 ms |
| random | 1024 | 4669 | 4560 | 6.24 / 11.63 s | 42.0 ms | 212.4 ms |
| nature | 1024 | 5177 | 5056 | 6.44 / 11.47 s | 40.4 ms | 200.6 ms |
| random | 4096 | 13256 | 3236 | 20.71 / 44.56 s | 56.7 ms | 201.6 ms |
| nature | 4096 | 14540 | 3550 | 15.91 / 36.23 s | 50.5 ms | 198.2 ms |
| random | 16384 | 19415 | 1185 | 77.89 / 180.63 s | 132.3 ms | 209.3 ms |
| nature | 16384 | 23241 | 1418 | 63.44 / 145.28 s | 110.7 ms | 205.8 ms |

单流 1,047,000 token 补测通过：random prefill 约 4501 TPS、TTFT 约 228.96 s，逐请求 decode 约 274.7 TPS；nature prefill 约 4757 TPS、TTFT 约 216.49 s，逐请求 decode 约 277.9 TPS。完整逐请求 JSON 和 bidevs/runner 日志保留在目标机 `runtime/benchmark-matrix-20261021` 与临时 bidevs workspace。

## 四个长尾场景（2026-10-21）

四个场景固化在 [`config/benchmarks/scenarios.json`](../config/benchmarks/scenarios.json)，使用当前 1M bounded profile 执行；除 mixed 场景的两条流外，OSL 统一为 1,000。

| Case | 输入与并发 | 请求数 / 时长 | 结果摘要 |
|---|---|---:|---|
| `tool_history_16384_c16` | 工具历史目标 ISL 16,384，观测 ISL 15,601，C16，OSL 1,000 | 16 | 0 失败；TTFT P50/P95 1.84 / 1.89 s；TPOT P50 13.31 ms；decode aggregate 约 816 TPS |
| `vision_mix_4096_c4` | 1/2/4 图像混合，观测 ISL 4,062–4,125，C4，OSL 1,000 | 4 | 0 失败；TTFT P50/P95 1.12 / 1.16 s；TPOT P50 5.59 ms；decode aggregate 约 158 TPS |
| `long_soak_1024_c64` | ISL 1,024，C64，OSL 1,000 | 1,800 s | 7,940 / 7,940 成功；TTFT P50/P95 0.350 / 0.765 s；TPOT P50/P95 13.17 / 16.85 ms；prefill/decode aggregate 约 4517 / 4411 TPS |
| `mixed_prefill_decode_same_service` | decode：ISL 1,024/C64/OSL 1,000；prefill：ISL 262,144/C1/OSL 1,000，每 20 s 注入 | 300 s | decode 741 成功；TTFT P50/P95 0.636 / 2.060 s；TPOT P50/P95 19.99 / 48.45 ms；prefill 12 成功，TTFT P50/P95 0.634 / 33.24 s，prefill aggregate 约 10183 TPS |

`mixed_prefill_decode_same_service` 使用同一个 vLLM endpoint、同一个 scheduler、同一个 KV pool，不是 PD 分离服务。完整逐请求证据保留在目标机 `runtime/benchmark-scenarios-20261021`。

表中数值是已完成的原始回执；随后 runner 增加了输入长度校验、独立冷 prompt、future drain 和 mixed lane 计数。下一轮复跑应使用新 runner，不能把两种口径的结果直接合并排序。
