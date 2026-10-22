# 基准结果

本文记录 `v0.1.0` 在 `8×A100-SXM4-80GB / NVSwitch` 上完成的性能测试。默认配置为 `tp4dp2 / ep8`、DSpark k5、stacked-kv、CUDA Graph、OMP 8、EPLB、SM80 custom AG/RS 和全局 `max-seqs=256`。

## 指标口径

- 整机输出吞吐：成功请求的服务端 `completion_tokens` 除以完整测量墙钟。
- Prefill TPS：服务端输入 token 数除以测量墙钟；单请求 prefill 使用首 token 延迟估算。
- Decode TPS：服务端输出 token 数除以 decode 测量墙钟；单请求同时记录 TPOT。
- TTFT：请求发出到第一个含文本的 SSE chunk 的时间。
- ITL：相邻有效 SSE chunk 的时间间隔。
- 所有请求使用 temperature 0；矩阵测试使用 `OSL=1000`，短吞吐测试使用 `OSL=128`。

## 短请求并发

| 全局并发 | 成功率 | 整机输出吞吐 | TTFT P50 / P95 |
|---:|---:|---:|---:|
| 64 | 64/64 | 2358 token/s | 1.99 / 2.01 s |
| 128 | 128/128 | 4496 token/s | 1.37 / 1.78 s |
| 256 | 256/256 | 7270 token/s | 0.71 / 1.24 s |

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

| Profile | C128 | C256 | 260K prefill | 1M prefill |
|---|---:|---:|---:|---:|
| unbounded | 843 token/s | 1109 token/s | — | — |
| bounded | 1286 token/s | 1369 token/s | 9067 token/s | 4772 token/s |

bounded profile 的 C256 相对 unbounded 提升约 23%，适合 1M 高并发配置；262K profile 仍是高并发默认选择。

## C256 配置 A/B

| 配置 | C128 | C256 |
|---|---:|---:|
| OMP 8 + EPLB + custom AG/RS | 3016 token/s | 4338 token/s |
| 关闭 EPLB | 2952 token/s | 4147 token/s |
| 关闭 custom AG/RS | 2335 token/s | 3815 token/s |
| OMP 1 | 2493 token/s | 2753 token/s |
| `decode-metadata` + `ragged-reuse` | — | 3969 token/s |
| `gpu-memory-utilization=0.95`，随机冷请求 | — | 1057 token/s |

## ISL / OSL 矩阵

所有请求 `OSL=1000`、temperature 0、`ignore_eos=true`。`random` 使用随机 token ID，`nature` 使用固定自然语言语料经目标 tokenizer 生成的 token 序列。

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

## 长尾场景

| Case | 成功率 | TTFT P50 / P95 | TPOT P50 / P95 | 吞吐 |
|---|---:|---:|---:|---:|
| 工具历史，ISL 16384，C16 | 16/16 | 1.84 / 1.89 s | 13.31 ms | 816 decode token/s |
| 图文混合，ISL 4096，C4 | 4/4 | 1.12 / 1.16 s | 5.59 ms | 158 decode token/s |
| 长时间 soak，ISL 1024，C64 | 7940/7940 | 0.350 / 0.765 s | 13.17 / 16.85 ms | 4517 prefill / 4411 decode token/s |
| Mixed prefill/decode，同一服务 | decode 741；prefill 12 | decode 0.636 / 2.060 s | 19.99 / 48.45 ms | 10183 prefill token/s |

## Prefill throttle A/B

| 配置 | Decode TPS | TTFT P50 / P95 | TPOT P50 | ITL P95 | Prefill TPS |
|---|---:|---:|---:|---:|---:|
| throttle=1，cadence=4 | 915.6 | 2.88 / 12.54 s | 61.0 ms | 1.88 s | 5384.8 |
| throttle=0 | 783.5 | 1.73 / 12.73 s | 73.7 ms | 2.09 s | 6546.6 |

`prefill-throttle` 保持显式 opt-in，因为它改善 decode-heavy mixed workload，同时降低 prefill TPS 并增加 TTFT P50。

## 已完成的排除实验

| 变体 | 结果 |
|---|---|
| `max-num-batched-tokens=32768` | 260K prefill 31.17 s，慢于 16K 的 28.85 s |
| SM80 `logits-row-chunk=512` | 260K prefill 30.98 s，慢于 16K 的 28.85 s |
| TP8 对 TP4DP2 | 260K prefill 28.24 s 对 28.85 s，提升约 2.1% |
| compact candidate-MQA | C256 / ISL1024 / OSL128 为 1096.6 token/s，低于旧路径 1158.9 token/s；ISL16384 失败率 56.25% |
