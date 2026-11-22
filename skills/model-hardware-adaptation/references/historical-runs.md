# 本仓库历史运行与事故记录

本文件只用于追溯过去观察到的指标、事故链和负向结果。所有内容都是 `HISTORICAL`，不能直接支撑当前 release 或替代新的 evidence record。

## 性能、容量和 KV

会话 `01a164c0-fb30-7cc3-af92-585e99d37288` 在 NVIDIA `8×A100-SXM4-80GB / NVSwitch` 上记录了 1M 配置：`TP4 + DP2 + EP8`、每 DP `max-num-seqs=128`、`max-num-batched-tokens=16384`、显存水位 `0.90`、FP8 DS MLA KV、DSpark `k=5`、`stacked-kv`、`bounded-prefill`、prefill throttle、SM80 MQA、custom AG/RS 和 `FULL_DECODE_ONLY` CUDA Graph。

- 1M C4：4/4，输入 TPS 约 `6650.9`，TTFT P50/P95 约 `264.5/625.8 s`。
- 1M C16：16/16，输入 TPS 约 `5820.3`，TTFT P50/P95 约 `1308.9/2873.3 s`。
- C16 中的 capacity queue 属于调度观测，不是请求失败。

该会话引用的以下 artifact 在当前 checkout 中不存在（2026-11-06 核对）。需要时从原会话或备份恢复，或按当前身份重跑：

```text
MISSING runtime/benchmark-1m-random-c4-20261023.json
MISSING runtime/benchmark-1m-random-c16-20261023.json
MISSING runtime/bench-1m-unbounded-prefill-260016-20261023.json
MISSING runtime/bench-1m-unbounded-prefill-1000016-20261023.json
MISSING runtime/ppl-wikitext-2-test-20261023.json
MISSING runtime/kv-transfer-simple-result-20261023.json
MISSING runtime/kv-transfer-offloading-result-20261023.json
```

同一会话记录了全模型 PPL：272,802 个计分 token，token PPL 约 `2.7722`，word PPL `3.1682`，byte PPL `1.2405`。

历史性能文档还记录了 262K / 524K / 1M × `0.90 / 0.925 / 0.95` 的 9 组矩阵：723 个适用测试项通过，token PPL 范围为 `2.770433–2.774319`；配置摘要称 1M C16 长请求 16/16 完成、输入 TPS 为 `7865–8018`，GPQA Diamond 4 轮平均准确率 `91.54%`，1M 30 分钟闭环 `7940/7940`。这些是历史文档事实，不代表当前 HEAD 已重新执行。

## A/B 和负向结果

会话 `01a162c3-11d2-7840-8d5b-fba386363070` 和 `01a1829d-dd3b-7e50-ae12-dcdea5eea8d9` 记录了以下实验规则和结果：

- `bounded-prefill` 在固定 1M 负载的某次对照中，C128 output TPS 约提升 52.6%，C256 约提升 23.4%；单请求长 prefill 近似不变。该组合值得复测，不是通用默认优化。
- prefill throttle 在固定 cadence=4、最小长 prefill=4096 的对照中提高 decode 输出和延迟指标，但降低 prefill 和总吞吐；这是 SLO 权衡。
- `decode-metadata` 与 `ragged-reuse` 曾同时开启，必须用 off/off、on/off、off/on、on/on 四臂拆分。
- 16K/32K 多模态 chunk 对不同文本和媒体负载方向不同，不能全局切换。
- compact candidate-MQA 出现输出不一致和长上下文失败；`logits-row-chunk`、TP8、dummy query 没有足够收益或未达采用门禁。负向结果同样保留。

历史边界检查曾记录：1M / 32K / `0.90` 和 `0.925` 因容量不足被启动检查拒绝；32K 长输入在 `0.95` 水位下 5 项、8 个请求全部通过；1M / 16K / `0.96` 的高水位功能、视觉、Responses 和 PPL 检查通过。

## KV connector 事故

会话 `01a16651-711c-74f1-8e38-225748e5d987` 记录了以下故障链：

- `native` KV 可以稳定完成 1M C4/C16。
- `SimpleCPUOffloadConnector` 可以启动，但首次短请求返回 HTTP 500 并触发 worker crash；之后的 DP/Gloo `connection closed by peer` 是二次症状，最早 worker 原始异常当时未完全取得。
- `OffloadingConnector` 在启动阶段因 hybrid KV layout 对齐失败；观察到 `tokens_per_block=8`、`tokens_per_hash=32`，8 不能整除 32。prefix cache 或 blocks per chunk 需要与 layout 约束一起检查。
- launcher 的 JSON 检查不能证明 connector 对当前模型和并行组合可用。
- 建议最小矩阵是关闭 EPLB、DSpark、CUDA Graph，使用 `DP=1`、eager、单卡或最小并行，再逐项恢复 DP、EP、EPLB、DSpark、graph、FP8 和 connector。

## Responses 与协议事故

会话 `01a164c0-fb30-7cc3-af92-585e99d37288` 记录了 Responses custom tool 历史因 `Unsupported input item type` 返回 HTTP 400；原因属于输入 schema/协议层，`tool_call_parser` 只解析模型输出。修复后真实 HTTP 请求通过，但首轮自主 custom tool parser 的能力仍需单独验收。

历史验收分别覆盖：

- Chat Completions：`tool_calls`、`role=tool`、reasoning 内容和续轮。
- Responses：`function_call`、`function_call_output`、`custom_tool_call`、`custom_tool_call_output`、流式事件和 `previous_response_id`。

## Responses store 生命周期

会话 `01a1829d-dd3b-7e50-ae12-dcdea5eea8d9` 与 `01a18419-52bf-7fc3-b16e-2da548241cee` 记录了一次统一生命周期修复和验收：

- `response_store`、`msg_store`、`event_store` 使用 OrderedDict LRU 上限。
- 淘汰 response 同步删除关联 message/event；读取 `previous_response_id` 更新 LRU 顺序。
- SSE 后台任务在 `finally` 清理 event queue；删除或取消先在锁内摘状态，锁外取消任务以避免死锁。
- 单 API worker 持有进程内历史；服务重启后历史清空。
- 7 项 live Responses 验收和重启后 health/models/最小 Responses smoke 分别记录。
- 有一次运行镜像与修复源码不匹配，基础 HTTP 200 不能替代 source/image/receipt 校验。
