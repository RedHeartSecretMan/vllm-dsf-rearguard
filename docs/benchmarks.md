# 基准结果

本文记录 `v0.1.0` 在 `8×A100-SXM4-80GB / NVSwitch` 上完成的性能测试。默认配置为 `tp4dp2 / ep8`、DSpark k5、stacked-kv、CUDA Graph、OMP 8、EPLB、SM80 custom AG/RS 和全局 `max-seqs=256`。

## 指标口径

- 整机 Prefill TPS：成功请求的服务端输入 token 总数除以测量墙钟。
- 整机 Decode TPS：成功请求的服务端 `completion_tokens` 总数除以同一测量墙钟，包含 prefill 阶段的耗时。
- 单请求 Prefill TPS：输入 token 除以 TTFT。
- 单请求 Decode TPS：`输出 token - 1` 除以首末文本 chunk 的间隔；TPOT 为同一间隔除以 `输出 token - 1`。
- TTFT：请求发出到第一个含文本的 SSE chunk 的时间。
- ITL：相邻有效 SSE chunk 的时间间隔。DSpark 可能在一个 chunk 中返回多个 token，因此 ITL 不等于逐 token 延迟。
- 所有请求使用 temperature 0；矩阵测试使用 `OSL=1000`，短吞吐测试使用 `OSL=128`。
- 任一请求缺少 `[DONE]`、finish reason 或服务端 usage，或输出 token 数不符合预期时记为失败。

## 短请求并发

262K 上限、`tp4dp2 / ep8`、DSpark k5、stacked-kv、固定 prompt、每个请求输出 128 token，所有请求均成功。两组数据来自 `max-seqs` 不同的两轮测试。

| 全局 `max-seqs` | 全局并发 | 成功率 | 整机输出吞吐 | TTFT P50 / P95 |
|---:|---:|---:|---:|---:|
| 128 | 64 | 64/64 | 2358 token/s | 1.99 / 2.01 s |
| 128 | 128 | 128/128 | 4496 token/s | 1.37 / 1.78 s |
| 256 | 128 | 128/128 | 3928 token/s | — |
| 256 | 256 | 256/256 | 7270 token/s | 0.71 / 1.24 s |

推荐配置采用 `max-seqs=256`。

## 单请求 prefill / decode

| 输入 | Prefill TPS | TTFT | Decode TPS | TPOT |
|---:|---:|---:|---:|---:|
| 1016 token | 3222 | 0.315 s | — | — |
| 32016 token | 5924 | 5.405 s | — | — |
| 128016 token | 9770 | 13.103 s | — | — |
| 250016 token | 8630 | 28.972 s | — | — |
| 260016 token | 8498 | 30.597 s | — | — |
| 17 prompt / 128 output | — | 0.566 s | 279.5 | 3.58 ms |

## 262K 与 1M 上限

两组使用相同的 `tp4dp2 / ep8`、DSpark k5、stacked-kv、CUDA Graph、全局 `max-seqs=256`、chunk 和显存比例，只切换最大上下文。

| 最大上下文 | 260K prefill | 单请求 decode | C256 整机吞吐 |
|---:|---:|---:|---:|
| 262144 | 8498 token/s | 279.5 token/s | 7270 token/s |
| 1048576 | 7740 token/s | 279.3 token/s | 4240 token/s |

## 1M bounded workspace

`bounded-prefill` 把 indexer prefill workspace 的预留下限收敛到单个 1M 请求可容纳的压缩上下文，保留 1M 单请求能力，减少高并发时的静态预留。两组使用相同的镜像、`tp4dp2 / ep8`、DSpark k5、stacked-kv、CUDA Graph、全局 C256、1K 输入和 128 输出，C256 使用相同的 seed 8080，各自使用独立的新容器，请求均为冷缓存且 0 失败。

| Profile | C128 | C256 | 260K prefill | 1M prefill |
|---|---:|---:|---:|---:|
| unbounded | 843 token/s | 1109 token/s | — | — |
| bounded | 1286 token/s | 1369 token/s | 9067 token/s | 4772 token/s |
| 变化 | +52.6% | +23.4% | — | — |

## 配置 A/B

每次只改动一个配置项，其余保持不变，用来决定默认配置。基线是 OMP 8、EPLB、SM80 custom AG/RS 同时开启。

### 固定 prompt 热请求

1M bounded profile、固定 10-token prompt、先跑 C128 热身再测 C256、每个请求输出 128 token。

| 配置 | C128 | C256 | C256 变化 |
|---|---:|---:|---:|
| OMP 8 + EPLB + custom AG/RS（基线） | 3016 token/s | 4338 token/s | — |
| 关闭 EPLB | 2952 token/s | 4147 token/s | -4.4% |
| 关闭 custom AG/RS | 2335 token/s | 3815 token/s | -12.1% |
| OMP 1 | 2493 token/s | 2753 token/s | -36.5% |
| 同时开启 `decode-metadata` 与 `ragged-reuse` | — | 3969 token/s | -8.5% |

EPLB、custom AG/RS 和 OMP 8 都带来正收益，保留为默认配置。两个实验开关组合后没有收益，继续保持关闭。

### 显存水位

1M bounded profile、1K 随机冷请求、C256、每个请求输出 128 token。

| `gpu-memory-utilization` | KV cache | C256 | 变化 |
|---:|---:|---:|---:|
| 0.90（基线） | 约 10.65 GiB | 1369 token/s | — |
| 0.95 | 约 14.61 GiB | 1057 token/s | -22.8% |

KV cache 增加没有转化为吞吐，继续使用 0.90。

两组表使用不同的请求口径，数值不能互相比较。

## ISL / OSL 矩阵

1M bounded profile。所有请求 `OSL=1000`、temperature 0、`ignore_eos=true`。`random` 使用随机 token ID，`nature` 使用固定自然语言语料经目标 tokenizer 生成的 token 序列，两者都保持 ISL 精确。ISL 256 至 16384 为 C256 的整机汇总值。

| 数据 | ISL | Prefill TPS | Decode TPS | TTFT P50 / P95 | TPOT P50 | ITL P50 |
|---|---:|---:|---:|---:|---:|---:|
| random | 256 | 1288 | 5033 | 2.11 / 3.42 s | 35.6 ms | 199.6 ms |
| nature | 256 | 1612 | 6296 | 2.22 / 3.58 s | 34.7 ms | 202.1 ms |
| random | 1024 | 4669 | 4560 | 6.24 / 11.63 s | 42.0 ms | 212.4 ms |
| nature | 1024 | 5177 | 5056 | 6.44 / 11.47 s | 40.4 ms | 200.6 ms |
| random | 4096 | 13256 | 3236 | 20.71 / 44.56 s | 56.7 ms | 201.6 ms |
| nature | 4096 | 14540 | 3550 | 15.91 / 36.23 s | 50.5 ms | 198.2 ms |
| random | 16384 | 19415 | 1185 | 77.89 / 180.63 s | 132.3 ms | 209.3 ms |
| nature | 16384 | 23241 | 1418 | 63.44 / 145.28 s | 110.7 ms | 205.8 ms |
| random | 1047000 | 4501 | 274.7 | 228.96 s | — | — |
| nature | 1047000 | 4757 | 277.9 | 216.49 s | — | — |

ISL 1047000 为单流（并发 1），该行的 Decode TPS 取逐请求值。输入加 1000 输出不能超过 1,048,576 上限，因此使用 1047000 作为可执行的 1M 输入。

## 长尾场景

| Case | 成功率 | TTFT P50 / P95 | TPOT P50 / P95 | 吞吐 |
|---|---:|---:|---:|---:|
| 工具历史，目标 ISL 16384（观测 15601），C16 | 16/16 | 1.84 / 1.89 s | 13.31 ms | 816 decode token/s |
| 图文混合（1/2/4 图），ISL 4096，C4 | 4/4 | 1.12 / 1.16 s | 5.59 ms | 158 decode token/s |
| 长时间 soak，ISL 1024，C64，1800 s | 7940/7940 | 0.350 / 0.765 s | 13.17 / 16.85 ms | 4517 prefill / 4411 decode token/s |
| Mixed：decode ISL 1024 / C64，每 20 s 注入 ISL 262144 prefill，300 s | decode 741；prefill 12 | decode 0.636 / 2.060 s | 19.99 / 48.45 ms | 10183 prefill token/s |

四个场景定义在 [`config/benchmarks/scenarios.json`](../config/benchmarks/scenarios.json)，工具历史、图文和 soak 的 OSL 均为 1000。Mixed 场景的两条流运行在同一个 vLLM endpoint、同一个 scheduler 和同一个 KV pool 上，不是 PD 分离服务；其 prefill 流共 12 个请求。

## Prefill throttle A/B

同一镜像、`tp4dp2 / ep8`、1M 上下文、DSpark k5、decode C64（ISL 1024）加 262K prefill 混合负载，只切换 `VLLM_DSV41_PREFILL_THROTTLE`。300 秒测量窗口，两组失败率均为 0。

| 配置 | Decode TPS | TTFT P50 / P95 | TPOT P50 | ITL P95 | Prefill TPS |
|---|---:|---:|---:|---:|---:|
| throttle=1，cadence=4 | 915.6 | 2.88 / 12.54 s | 61.0 ms | 1.88 s | 5384.8 |
| throttle=0 | 783.5 | 1.73 / 12.73 s | 73.7 ms | 2.09 s | 6546.6 |
| 变化 | +16.9% | TTFT P50 变差 | -17.2% | -10.0% | -17.7% |

`prefill-throttle` 提升 decode-heavy 混合负载的 decode 吞吐并降低 TPOT，代价是 prefill TPS 下降、TTFT P50 增加，因此保持显式 opt-in。

## 排除实验

以下候选经过同口径测试，没有收益或不满足精度准入，不进入默认配置。前三项为 1M profile 下的 260K 冷 prefill，基线为 16K chunk 的 28.85 s。

| 候选 | 结果 | 相对基线 |
|---|---|---:|
| `max-num-batched-tokens=32768` | 260K prefill 31.17 s | +8.0% 耗时 |
| SM80 `logits-row-chunk=512` | 260K prefill 30.98 s | +7.4% 耗时 |
| DSpark `dummy_q` 由 `zeros` 改为 `empty` | 260K prefill 30.95 s | +7.3% 耗时 |
| TP8 代替 TP4DP2 | 260K prefill 28.24 s | -2.1% 耗时（单流收益小，不采用） |
| compact candidate-MQA | C256 / ISL 1024 / OSL 128 decode 1096.6 token/s，低于 query-scratch 候选的 1158.9 token/s；C256 / ISL 16384 为 112/256 成功，失败率 56.25%；C256 下输出摘要与 dense 路径不一致 | 不满足精度准入 |

上述候选都不返回 draft/accepted 计数，没有接受率数据。自适应 DSpark 验证在当前 indexer backend 上被明确拒绝，已移除入口。
