# 发布说明

## 0.1.0

`0.1.0` 是当前固定 1017 权重和 fixed-digest runtime 的首个可复现实验版本，目标平台为 `8×A100-SXM4-80GB / NVSwitch`。

### 已包含

- 统一 vLLM 适配层和原子安装清单。
- 1017 模型协议、工具续轮、原生图文输入和视觉边界处理。
- TP8 / EP8 与 TP4DP2 / EP8 启动配置。
- DSpark k5、CUDA Graph、FP8 KV cache、稀疏 MLA、EPLB 和 EP8 通信路径。
- 长上下文 workspace 约束、query-sharded decode metadata、DSpark KV 投影和 benchmark 工具。
- CPU 合约、协议 parity、图文 smoke、长上下文和 GPQA Diamond 验证。

### 结果

- 262K 上限、C256 固定短请求：约 `7270 token/s`。
- 1M bounded profile、C256：约 `1369 token/s`。
- 262K 单请求 prefill：约 `8498 token/s`。
- 单请求 decode：约 `279 token/s`。
- GPQA Diamond 4 轮平均：`91.54%`。

详细数据见 [基准结果](benchmarks.md)，验证范围见 [验证记录](validation.md)。

### 边界

compact candidate-MQA、较大的 prefill chunk、SM80 logits row chunk 和 adaptive DSpark 等实验路径没有进入默认配置。它们保留在 `perf` 分支作为后续研究入口，并不构成 `0.1.0` 的性能承诺。
