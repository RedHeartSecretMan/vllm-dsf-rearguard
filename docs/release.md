# 发布说明

## 0.1.0

`0.1.0` 是面向固定 1017 权重和固定 digest 运行时的首个可复现版本，目标平台为 `8×A100-SXM4-80GB / NVSwitch`。

### 包含内容

- 统一的 vLLM 适配层和原子安装清单。
- 1017 模型协议、工具续轮、原生图文输入和视觉边界处理。
- `tp8 / ep8` 与 `tp4dp2 / ep8` 两种启动配置。
- DSpark k5、CUDA Graph、FP8 KV cache、稀疏 MLA、EPLB 和 EP8 通信路径。
- 长上下文 indexer 工作区约束、query-sharded decode metadata 和 DSpark KV 投影。
- 四类 benchmark 配置与 runner（常规场景、长尾场景、配置 A/B、排除实验），以及按 core、visual、responses、quality 分类的 GPU 验收配置与 runner；全模型 PPL 和 GPQA 质量评测单独归入 quality。

### 推荐配置

默认关闭多模态分块的 262K 配置和 1M 配置都使用 `tp4dp2 / ep8`、DSpark k5、`stacked-kv`、CUDA Graph、OMP 8、EPLB、SM80 custom AG/RS、全局 `max-seqs=256` 和 `max-num-batched-tokens=32768`，以容纳视觉配置的最大视频 item。262K 配置适合高并发，1M 配置在此基础上增加 `--context 1048576 --experimental bounded-prefill`，适合容量型请求。完整命令见 [README](../README.md#推荐配置)。

当前 1M 视觉服务显式使用 `--chunk 16384 --no-disable-chunked-mm-input`，以增加 KV 容量。三组分块预算对照完成 4,293 个计量请求，详见 [16K / 32K 测试](chunk-budget.md)。正式服务默认端口为 `6789`，该专项测试使用 `6688`。

### 结果

| 场景 | 结果 |
|---|---:|
| 262K 配置，C256 短请求输出 token 吞吐 | 7270 token/s |
| 1M 配置，C256、ISL 1024 冷请求输出 token 吞吐 | 1369 token/s |
| 260K 单请求 Prefill TPS | 8498 token/s |
| 单请求 Decode TPS | 279.5 token/s |
| 1M 配置，16 个 ISL 1047000 请求同时进入 | 16/16 成功，输入 token 吞吐 5820 token/s |
| wikitext-2 test token PPL | 2.772 |
| GPQA Diamond，4 轮平均 | 91.54% |

完整数据见 [基准结果](benchmarks.md)，验证范围见 [验证记录](validation.md)。

### 实验开关

以下开关默认关闭，需要通过 `launch.py --experimental`、`--logits-row-chunk`、`--kv-transfer-config` 或容器环境变量显式启用。`stacked-kv` 与 `bounded-prefill` 已包含在推荐配置中，启动时仍需要显式传入：

| 开关 | 作用 | 当前结论 |
|---|---|---|
| `stacked-kv` | 合并 DSpark context 投影 | 推荐配置使用 |
| `bounded-prefill` | 收敛 indexer prefill 工作区 | 1M 高并发推荐使用，C256 提升 23.4% |
| `prefill-throttle` | 按 cadence 释放长 prefill | decode 流输出 token 吞吐 +16.9%，prefill 流输入 token 吞吐 -17.7%，按负载选用 |
| `decode-metadata` + `ragged-reuse` | decode 元数据与 top-k 复用 | 组合后 C256 -8.5%，不采用 |
|  `VLLM_SM80_CANDIDATE_MQA_COMPACT=1` | 紧凑 candidate-MQA top-k | C256 下输出摘要与 dense 路径不一致，不采用 |
| `--logits-row-chunk` | 分块释放 SM80 MQA logits | 260K prefill +7.4% 耗时，不采用 |
| `--kv-transfer-config` | 传入 vLLM KV transfer 配置 | `SimpleCPUOffloadConnector` 请求返回 500，`OffloadingConnector` 启动失败，不采用 |

各开关的 A/B 和排除实验数据见 [基准结果](benchmarks.md)。

### 验证范围

结果覆盖固定 1017 权重的文本、工具、图文、长上下文、并发和 mixed prefill/decode 场景，包括 4 个和 16 个 ISL 1047000 长请求同时进入的容量，以及 wikitext-2 上的全模型 PPL。
