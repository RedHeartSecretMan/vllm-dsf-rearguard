# 基准结果

本文记录 `v0.1.0` 在 `8×A100-SXM4-80GB / NVSwitch` 上完成的性能测试。所有实验使用同一套基础配置，分为两类基准和两类对照：

| 类别 | 目的 | 章节 |
|---|---|---|
| 常规场景 | 短请求并发、单请求 prefill / decode，以及 1M 配置下不同输入长度和并发的 ISL / OSL 矩阵 | [常规场景](#常规场景) |
| 长尾场景 | 1M 配置下的工具历史、图文混合、长时间 soak 和混合负载 | [长尾场景](#长尾场景) |
| 配置 A/B | 每次只改一个参数，决定默认值 | [配置 A/B](#配置-ab) |
| 排除实验 | 没有收益、不满足精度或无法运行的候选 | [排除实验](#排除实验) |

## 基础配置

`tp4dp2 / ep8`、DSpark k5、stacked-kv、CUDA Graph、OMP 8、EPLB、SM80 custom AG/RS、全局 `max-seqs=256`、`max-num-batched-tokens=16384`、`gpu-memory-utilization=0.90`。在此基础上按上下文上限分为两套配置：

| 配置 | 上下文上限 | 启动参数 | 适用场景 |
|---|---:|---|---|
| 262K 配置 | 262144 | `--context 262144 --experimental stacked-kv` | 高并发默认配置 |
| 1M 配置 | 1048576 | `--context 1048576 --experimental stacked-kv --experimental bounded-prefill` | 需要 1M 上下文的容量型请求 |

`bounded-prefill` 把稀疏 indexer 的 prefill 工作区预留收敛到单个 1M 请求所需的压缩上下文，单请求 1M 能力不变，对照见 [bounded workspace](#bounded-workspace)。

## 复现

四类测试各有一份配置和一个脚本，配置记录每个 case 的启动参数、ISL / OSL、并发和 seed：

| 类别 | 配置 | 脚本 |
|---|---|---|
| 常规场景 | [`regular.json`](../config/benchmarks/regular.json) | [`bench_regular.py`](../scripts/bench_regular.py) |
| 长尾场景 | [`long_tail.json`](../config/benchmarks/long_tail.json) | [`bench_long_tail.py`](../scripts/bench_long_tail.py) |
| 配置 A/B | [`ab.json`](../config/benchmarks/ab.json) | [`bench_ab.py`](../scripts/bench_ab.py) |
| 排除实验 | [`excluded.json`](../config/benchmarks/excluded.json) | [`bench_excluded.py`](../scripts/bench_excluded.py) |

脚本不带 `--case` 时列出该类的全部 case 和对应的启动参数。先按 case 的启动参数用 `launch.py` 启动服务，再运行脚本，结果写入新的 JSON 文件：

```bash
python3 -m scripts.bench_regular --case matrix_concurrent --output runtime/matrix-concurrent.json
python3 -m scripts.bench_ab --plan
python3 -m scripts.bench_ab --case bounded_workspace --variant on --output runtime/bounded-on.json
python3 -m scripts.bench_excluded --case chunk32768 --output runtime/excluded-chunk32768.json
```

全模型 PPL 在 1M 配置（未开启 `bounded-prefill`）的服务上运行，先生成 token 数据集，再逐窗打分：

```bash
python3 -m scripts.ppl_wikitext MODEL_DIR --output runtime/wikitext-2-test.jsonl
python3 -m scripts.ppl_score runtime/wikitext-2-test.jsonl --request-form echo --top-k 1 --output runtime/ppl-wikitext-2-test.json
```

## 指标口径

并发指同一时刻在服务中的请求数，如 C256 表示 256 个请求同时进入。ISL / OSL 分别是输入和输出 token 数。

**整机吞吐**统计一组并发请求的合计，三个指标共用同一个分母，即从发出第一批请求到最后一个请求完成的整段测量墙钟：

| 指标 | 定义 |
|---|---|
| 输入 token 吞吐 | 所有成功请求的服务端输入 token 总数 ÷ 测量墙钟 |
| 输出 token 吞吐 | 所有成功请求的服务端 `completion_tokens` 总数 ÷ 测量墙钟 |
| 总 token 吞吐 | 输入与输出 token 总数 ÷ 测量墙钟 |

**单请求指标**只看一个请求自己的时间线，与并发数无关：

| 指标 | 定义 |
|---|---|
| TTFT | 请求发出到第一个含文本的 SSE chunk 的时间 |
| 单请求 Prefill TPS | 输入 token ÷ TTFT |
| 单请求 Decode TPS | `输出 token - 1` ÷ 首末文本 chunk 的间隔 |
| TPOT | 同一间隔 ÷ `输出 token - 1`，即平均每个 token 的时间 |
| ITL | 相邻有效 SSE chunk 的时间间隔。DSpark 可能在一个 chunk 中返回多个 token，因此 ITL 不等于逐 token 延迟 |

整机吞吐和单请求指标不能互相换算。以 C256、ISL 16384 为例，输入 token 吞吐为 19415 token/s，而单个请求的 TTFT P50 是 77.89 s，对应单请求 Prefill TPS 约 210 token/s：256 个请求同时排队，每个请求的 TTFT 大部分在等其他请求的 prefill。

通用设定：所有请求 temperature 0；矩阵测试使用 `OSL=1000` 和 `ignore_eos=true`，短请求测试使用 `OSL=128`。任一请求缺少 `[DONE]`、finish reason 或服务端 usage，或输出 token 数不符合预期时记为失败，失败请求不计入吞吐和延迟。表中「—」表示该指标不适用，「/」表示没有测试。

## 常规场景

### 短请求并发

262K 配置。固定 prompt、每个请求输出 128 token，所有请求均成功。两组数据来自 `max-seqs` 不同的两轮测试。

| 全局 `max-seqs` | 全局并发 | 成功率 | 输出 token 吞吐 | TTFT P50 / P95 |
|---:|---:|---:|---:|---:|
| 128 | 64 | 64/64 | 2358 token/s | 1.99 / 2.01 s |
| 128 | 128 | 128/128 | 4496 token/s | 1.37 / 1.78 s |
| 256 | 128 | 128/128 | 3928 token/s | 2.17 / 2.20 s |
| 256 | 256 | 256/256 | 7270 token/s | 0.71 / 1.24 s |

推荐配置采用 `max-seqs=256`。

### 单请求 prefill / decode

262K 配置。一次只发一个请求；prefill 行用 `max_tokens=1` 测 TTFT，decode 行只有 17 个输入 token。

| 输入 | 单请求 Prefill TPS | TTFT | 单请求 Decode TPS | TPOT |
|---:|---:|---:|---:|---:|
| 1016 token | 3222 | 0.315 s | — | — |
| 32016 token | 5924 | 5.405 s | — | — |
| 128016 token | 9770 | 13.103 s | — | — |
| 250016 token | 8630 | 28.972 s | — | — |
| 260016 token | 8498 | 30.597 s | — | — |
| 17 prompt / 128 output | — | 0.566 s | 279.5 | 3.58 ms |

### 262K 与 1M 上限对照

对照参数是最大上下文（262144 对 1048576），其余基础配置相同，都不开启 `bounded-prefill`。

| 最大上下文 | 260K 单请求 Prefill TPS | 单请求 Decode TPS | C256 输出 token 吞吐 |
|---:|---:|---:|---:|
| 262144 | 8498 token/s | 279.5 token/s | 7270 token/s |
| 1048576 | 7740 token/s | 279.3 token/s | 4240 token/s |
| 变化 | -8.9% | -0.1% | -41.7% |

1M 上限下单请求 decode 基本不变，单请求 prefill 和高并发吞吐明显下降，因此 262K 作为高并发默认配置，1M 仅用于容量型请求。

### bounded workspace

1M 配置。对照参数是 `bounded-prefill`：关闭时按多请求上界预留 indexer prefill 工作区，开启时把预留下限收敛到 `max_model_len / compress_ratio`，较大的多请求 prefill 由 metadata builder 拆成独立 chunk。两组使用相同的镜像、基础配置、1K 输入和 128 输出，C256 使用相同的 seed 8080，各自使用独立的新容器，请求均为冷缓存且 0 失败。单请求 prefill 使用随机 token，ISL 260016 和 1000016，`max_tokens=1`。

| `bounded-prefill` | C128 输出 token 吞吐 | C256 输出 token 吞吐 | C256 总 token 吞吐 | 260K 单请求 Prefill TPS | 1M 单请求 Prefill TPS |
|---|---:|---:|---:|---:|---:|
| 关闭 | 843 token/s | 1109 token/s | 9979 token/s | 9077 token/s | 4776 token/s |
| 开启 | 1286 token/s | 1369 token/s | 12317 token/s | 9067 token/s | 4772 token/s |
| 变化 | +52.6% | +23.4% | +23.4% | -0.1% | -0.1% |

单请求 prefill 两组相同。bounded workspace 的收益来自高并发下更少的静态预留，单请求能力保持不变。

### ISL / OSL 矩阵

1M 配置。ISL 256 / 1024 / 4096 / 16384 各测 C4、C16、C64、C256 四个并发，ISL 4096 / 16384 / 65536 / 262144 / 1047000 各测单流（并发 1），每个 ISL 分别用 `random` 和 `nature` 两种数据。所有请求 `OSL=1000`、temperature 0、`ignore_eos=true`。`random` 使用随机 token ID，`nature` 使用固定自然语言语料经目标 tokenizer 生成的 token 序列，两者都保持 ISL 精确。每组正式测量前先用同长度的另一条 prompt 预热一次。

输入加 1000 输出不能超过 1,048,576 上限，因此 1M 单流使用 ISL 1047000。

### 单流

| 数据 | ISL | TTFT | 单请求 Prefill TPS | 单请求 Decode TPS | TPOT | ITL P50 | 输入 token 吞吐 | 输出 token 吞吐 | 总 token 吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 4096 | 0.39 s | 10543 | 290.3 | 3.44 ms | 20.0 ms | 1069 | 261.1 | 1330 |
| random | 16384 | 1.06 s | 15490 | 269.9 | 3.70 ms | 21.1 ms | 3443 | 210.1 | 3653 |
| random | 65536 | 4.63 s | 14149 | 273.7 | 3.65 ms | 21.5 ms | 7913 | 120.7 | 8034 |
| random | 262144 | 23.30 s | 11253 | 261.7 | 3.82 ms | 22.1 ms | 9668 | 36.9 | 9705 |
| random | 1047000 | 228.96 s | 4573 | 274.7 | 3.64 ms | 23.6 ms | 4501 | 4.3 | 4506 |
| nature | 4096 | 0.37 s | 10945 | 286.1 | 3.50 ms | 20.9 ms | 1059 | 258.6 | 1318 |
| nature | 16384 | 1.44 s | 11380 | 272.5 | 3.67 ms | 22.0 ms | 3209 | 195.9 | 3405 |
| nature | 65536 | 4.33 s | 15128 | 271.7 | 3.68 ms | 22.2 ms | 8183 | 124.9 | 8307 |
| nature | 262144 | 22.17 s | 11824 | 210.3 | 4.76 ms | 22.7 ms | 9737 | 37.1 | 9774 |
| nature | 1047000 | 216.49 s | 4836 | 277.9 | 3.60 ms | 24.1 ms | 4757 | 4.5 | 4762 |

### C4

| 数据 | ISL | 输入 token 吞吐 | 输出 token 吞吐 | 总 token 吞吐 | TTFT P50 / P95 | TPOT P50 | ITL P50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 256 | 151 | 591 | 742 | 0.20 / 0.20 s | 5.1 ms | 21.0 ms |
| nature | 256 | 260 | 1018 | 1278 | 0.15 / 0.32 s | 3.7 ms | 21.5 ms |
| random | 1024 | 818 | 799 | 1618 | 0.27 / 0.28 s | 4.7 ms | 27.0 ms |
| nature | 1024 | 1072 | 1047 | 2118 | 0.25 / 0.25 s | 3.6 ms | 21.3 ms |
| random | 4096 | 2408 | 588 | 2995 | 1.38 / 1.39 s | 5.2 ms | 21.0 ms |
| nature | 4096 | 3036 | 741 | 3777 | 0.77 / 0.77 s | 4.6 ms | 27.7 ms |
| random | 16384 | 9413 | 575 | 9987 | 1.61 / 3.26 s | 4.7 ms | 21.3 ms |
| nature | 16384 | 10654 | 650 | 11304 | 1.59 / 2.43 s | 3.7 ms | 22.3 ms |

### C16

| 数据 | ISL | 输入 token 吞吐 | 输出 token 吞吐 | 总 token 吞吐 | TTFT P50 / P95 | TPOT P50 | ITL P50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 256 | 334 | 1305 | 1639 | 0.74 / 0.93 s | 10.7 ms | 46.7 ms |
| nature | 256 | 689 | 2693 | 3382 | 0.35 / 0.52 s | 5.5 ms | 32.2 ms |
| random | 1024 | 1089 | 1064 | 2153 | 0.65 / 1.24 s | 8.4 ms | 46.7 ms |
| nature | 1024 | 1818 | 1775 | 3594 | 0.71 / 1.23 s | 8.2 ms | 46.4 ms |
| random | 4096 | 4603 | 1124 | 5727 | 2.36 / 3.33 s | 11.6 ms | 47.3 ms |
| nature | 4096 | 8180 | 1997 | 10177 | 2.50 / 2.77 s | 5.5 ms | 31.3 ms |
| random | 16384 | 14979 | 914 | 15893 | 5.05 / 9.15 s | 12.1 ms | 48.2 ms |
| nature | 16384 | 16709 | 1020 | 17728 | 5.67 / 10.13 s | 8.4 ms | 33.1 ms |

### C64

| 数据 | ISL | 输入 token 吞吐 | 输出 token 吞吐 | 总 token 吞吐 | TTFT P50 / P95 | TPOT P50 | ITL P50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 256 | 688 | 2689 | 3377 | 0.90 / 0.91 s | 16.1 ms | 75.3 ms |
| nature | 256 | 1220 | 4765 | 5985 | 0.89 / 0.91 s | 12.4 ms | 74.6 ms |
| random | 1024 | 3210 | 3135 | 6345 | 5.11 / 5.92 s | 14.1 ms | 76.6 ms |
| nature | 1024 | 3591 | 3507 | 7099 | 2.80 / 3.45 s | 15.3 ms | 75.7 ms |
| random | 4096 | 9488 | 2316 | 11805 | 6.87 / 12.02 s | 20.4 ms | 73.5 ms |
| nature | 4096 | 12583 | 3072 | 15655 | 5.34 / 9.41 s | 15.3 ms | 68.3 ms |
| random | 16384 | 19674 | 1201 | 20875 | 17.87 / 39.27 s | 34.8 ms | 73.8 ms |
| nature | 16384 | 22532 | 1375 | 23908 | 15.62 / 34.07 s | 30.3 ms | 71.4 ms |

### C256

| 数据 | ISL | 输入 token 吞吐 | 输出 token 吞吐 | 总 token 吞吐 | TTFT P50 / P95 | TPOT P50 | ITL P50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 256 | 1288 | 5033 | 6321 | 2.11 / 3.42 s | 35.6 ms | 199.6 ms |
| nature | 256 | 1612 | 6296 | 7908 | 2.22 / 3.58 s | 34.7 ms | 202.1 ms |
| random | 1024 | 4669 | 4560 | 9229 | 6.24 / 11.63 s | 42.0 ms | 212.4 ms |
| nature | 1024 | 5177 | 5056 | 10233 | 6.44 / 11.47 s | 40.4 ms | 200.6 ms |
| random | 4096 | 13256 | 3236 | 16493 | 20.71 / 44.56 s | 56.7 ms | 201.6 ms |
| nature | 4096 | 14540 | 3550 | 18089 | 15.91 / 36.23 s | 50.5 ms | 198.2 ms |
| random | 16384 | 19415 | 1185 | 20600 | 77.89 / 180.63 s | 132.3 ms | 209.3 ms |
| nature | 16384 | 23241 | 1418 | 24659 | 63.44 / 145.28 s | 110.7 ms | 205.8 ms |

同一个 seed 连续运行了 C4、C16、C64、C256 四组，后一组的前 N 个请求与前一组的 prompt 相同，而服务开启了前缀缓存，因此 C16 及以上的 TTFT 和吞吐略偏乐观。以 random、ISL 16384、C256 为例，复用 prompt 的 64 个请求 TTFT P50 为 12.70 s，其余 192 个为 106.08 s。单流组的数据不涉及复用。

### 1M 长请求容量

1M 配置加 `prefill-throttle`。多个 ISL 1047000 的请求同时进入，`OSL=1000`，prompt 为随机 token。

| 并发 | 成功率 | 墙钟 | 输入 token 吞吐 | 输出 token 吞吐 | 总 token 吞吐 | TTFT P50 / P95 | TPOT P50 / P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4/4 | 629.7 s | 6651 | 6.35 | 6657 | 264.5 / 625.8 s | 9 / 239 ms |
| 16 | 16/16 | 2878.2 s | 5820 | 5.56 | 5826 | 1308.9 / 2873.3 s | 296 / 382 ms |

每个引擎的 KV cache 可容纳 4,226,760 token，按 1M 请求计算最大并发约 4.03 个（启动日志中 worker 0 的可用 KV cache 内存为 10.65 GiB）。C16 超过这个并发，服务按容量排队，没有错误或中止，结束后 KV cache 使用率回到 0。输入 token 吞吐高于单流 1M 的 4501 和 4757 token/s，因为同一组内的请求共享整段墙钟，前一个请求的 decode 和后一个请求的 prefill 重叠。

C4 和 C16 使用同一个 seed，因此 C16 的前 4 个 prompt 与 C4 的 4 个相同。C16 的第 0 个请求 TTFT 只有 2.8 s，其余请求为 235.1 s 到 2873.3 s，该请求命中了 C4 留下的前缀缓存，C16 的总耗时略偏乐观。

### 全模型 PPL

1M 配置，未开启 `bounded-prefill`。wikitext-2 test，文档先按段去分词化，用空行连接后用模型自带的 tokenizer 分词一次，按 2048 token 不重叠的窗口逐窗打分，每个窗口的第一个 token 不计分。目标 token 的 logprob 缺失时该运行失败。

| 指标 | 结果 |
|---|---:|
| 文档 | 2891（非空） |
| token | 272936，打分 272802 |
| 窗口 | 134 |
| 平均 token NLL | 1.0196 |
| token PPL | 2.772 |
| word PPL | 3.168 |
| byte PPL | 1.2405 |
| 用时 | 50.1 s |

平均每个 token 的 NLL 为 1.0196，对应 token PPL 2.772；按词和字节折算后分别为 3.168 和 1.2405，约 0.31 bit/byte。

## 长尾场景

1M 配置。工具历史、图文和 soak 的 OSL 均为 1000。

| 场景 | 成功率 | TTFT P50 / P95 | TPOT P50 / P95 | 输入 token 吞吐 | 输出 token 吞吐 | 总 token 吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| 工具历史，目标 ISL 16384（观测 15601），C16 | 16/16 | 1.84 / 1.89 s | 13.31 / 17.71 ms | 12737 | 816 | 13553 |
| 图文混合（1/2/4 图），ISL 4096，C4 | 4/4 | 1.12 / 1.16 s | 5.59 / 8.11 ms | 647 | 158 | 804 |
| 长时间 soak，ISL 1024，C64，1800 s | 7940/7940 | 0.350 / 0.765 s | 13.17 / 16.85 ms | 4517 | 4411 | 8928 |
| Mixed decode 流，ISL 1024，C64，300 s | 741/741 | 0.636 / 2.060 s | 19.99 / 48.45 ms | 2456 | 2399 | 4855 |
| Mixed prefill 流，ISL 262144，每 20 s 注入一个 | 12/12 | 0.634 / 33.24 s | 19.83 / 20.62 ms | 10183 | 39 | 10222 |

Mixed 场景的两条流运行在同一个 vLLM endpoint、同一个 scheduler 和同一个 KV pool 上。两条流分别统计，分母是场景的完成墙钟。

## 配置 A/B

每次只改动一个配置项，其余保持不变，用来决定默认配置。

### 通信与调度参数

1M 配置。固定 10-token prompt，先跑 C128 热身再测 C256，每个请求输出 128 token。基线是 OMP 8、EPLB、SM80 custom AG/RS 同时开启。

| 配置 | C128 输出 token 吞吐 | C256 输出 token 吞吐 | C256 总 token 吞吐 | C256 变化 |
|---|---:|---:|---:|---:|
| OMP 8 + EPLB + custom AG/RS（基线） | 3016 token/s | 4338 token/s | 4677 token/s | — |
| 关闭 EPLB | 2952 token/s | 4147 token/s | 4471 token/s | -4.4% |
| 关闭 custom AG/RS | 2335 token/s | 3815 token/s | 4113 token/s | -12.1% |
| OMP 1 | 2493 token/s | 2753 token/s | 2968 token/s | -36.5% |
| 同时开启 `decode-metadata` 与 `ragged-reuse` | 1908 token/s | 3969 token/s | 4279 token/s | -8.5% |

EPLB、custom AG/RS 和 OMP 8 都带来正收益，保留为默认配置。两个实验开关组合后 C256 下降，继续保持关闭。

### 显存水位

1M 配置。ISL 1024 随机冷请求、C256、每个请求输出 128 token。

| `gpu-memory-utilization` | KV cache（worker 0） | 输入 token 吞吐 | 输出 token 吞吐 | 总 token 吞吐 | 变化 |
|---:|---:|---:|---:|---:|---:|
| 0.90（基线） | 约 10.65 GiB | 10949 token/s | 1369 token/s | 12317 token/s | — |
| 0.95 | 约 14.61 GiB | 8455 token/s | 1057 token/s | 9512 token/s | -22.8% |

KV cache 增加后吞吐下降，继续使用 0.90。

### Prefill throttle

1M 配置。decode C64（ISL 1024）加 262K prefill 的混合负载，只切换 `VLLM_DSV41_PREFILL_THROTTLE`。300 秒测量窗口，两组失败率均为 0。两条流分别统计，分母是场景的完成墙钟。

| 配置 | Decode 流输出 token 吞吐 | Decode 流 TTFT P50 / P95 | Decode 流 TPOT P50 | Decode 流 ITL P95 | Prefill 流输入 token 吞吐 | 整机总 token 吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| throttle=0（基线） | 783.5 | 1.73 / 12.73 s | 73.7 ms | 2.09 s | 6546.6 | 8157 |
| throttle=1，cadence=4 | 915.6 | 2.88 / 12.54 s | 61.0 ms | 1.88 s | 5384.8 | 7258 |
| 变化 | +16.9% | TTFT P50 变差 | -17.2% | -10.0% | -17.7% | -11.0% |

`prefill-throttle` 提高 decode 流的输出 token 吞吐并降低 TPOT，代价是 prefill 流的输入 token 吞吐下降、TTFT P50 增加，因此保持显式 opt-in。

## 排除实验

以下候选的结果是性能没有提升、精度准入未通过或在当前镜像上无法运行，默认配置不采用。260K 冷 prefill 的对照是 16K chunk 的 28.85 s。

| 候选 | 测试 | 结果 | 结论 |
|---|---|---|---|
| `max-num-batched-tokens=32768` | 1M 配置，260K 冷 prefill | 31.17 s，+8.0% 耗时 | 性能没有提升 |
| SM80 `logits-row-chunk=512` | 1M 配置，260K 冷 prefill | 30.98 s，+7.4% 耗时 | 性能没有提升 |
| DSpark `dummy_q` 由 `zeros` 改为 `empty` | 1M 配置，260K 冷 prefill | 30.95 s，+7.3% 耗时 | 性能没有提升 |
| TP8 代替 TP4DP2 | 1M 配置，260K 冷 prefill | 28.24 s，-2.1% 耗时 | 性能没有提升 |
| compact candidate-MQA | 1M 配置，C256，ISL 1024，OSL 128 | decode 1096.6 token/s，query-scratch 候选为 1158.9 token/s；输出摘要与 dense 路径不一致 | 不满足精度准入 |
| compact candidate-MQA | 1M 配置，C256，ISL 16384 | 112/256 成功，失败率 56.25% | 不满足精度准入 |
| `SimpleCPUOffloadConnector`，`cpu_bytes_to_use=8 GiB` | 1M 上限，未开启 `bounded-prefill`，启动后连续发送两个 ISL 1024 的随机请求 | 启动通过，每 rank 9320 个 CPU block（约 2 GiB）；请求返回 HTTP 500，dummy batch 阶段 worker 崩溃 | 请求失败 |
| `OffloadingConnector`，`blocks_per_chunk=1` | 1M 上限，未开启 `bounded-prefill`，启动 | `tokens_per_block=8` 不能被 `tokens_per_hash=32` 整除 | 启动失败 |
| 自适应 DSpark（`enable_adaptive_verification`） | 1M 配置，启动 | indexer backend 不支持在 device 上裁剪 verification 请求 | 启动失败 |
