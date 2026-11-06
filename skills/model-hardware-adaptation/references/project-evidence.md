# 本仓库项目事实与迁移边界

本文件只保存当前仓库中较稳定、会影响迁移决策的事实。具体历史指标、会话 ID 和事故链在 [historical-runs.md](historical-runs.md)；历史数字不代表当前 checkout 已重新执行。

## 历史快照

- 分支：`main`
- commit：`e13c72af367ff13091dda057b20b92c8bfcaea28`
- 快照状态：工作区干净
- 说明：skill 文件不属于该基线；当前 HEAD 或镜像变化后，以下事实需要重新核验

## 项目事实

| 事实 | 当前来源 |
|---|---|
| 目标模型是 DeepSeek-V4.1-Flash-1017，目标机器为 `8×A100-SXM4-80GB / NVSwitch`；默认 `TP4 / DP2 / EP8`，另支持 `TP8 / EP8` | `README.md` 的 “Supported scope” / “适用范围”；`docs/architecture.md` |
| 支持 262K、524K、1M 上下文，文本、图片、工具调用、Chat Completions 和 Responses；视频由服务端采样为图片帧 | `README.md` 的 “Why vllm-dsf-rearguard” / “Supported scope”；`docs/architecture.md` 的 “协议与多模态” |
| ModelScope revision、weight release revision、snapshot 字节数和每个模型文件大小/SHA256 已写入清单 | `manifests/model-files.json` |
| 基础镜像按 digest 和来源 commit 固定 | `manifests/sources.json` |
| 适配层按文件记录基础/最终哈希、用途和阶段，并用统一安装器原子写入 | `adaptation/manifest.json`；`adaptation/install.py`；`adaptation/README.md` |
| 构建要求工作树干净，从 `git archive HEAD` 构建，网络关闭执行 CPU/runtime 检查，并明确写出 `GPU validation: NOT_RUN` | `scripts/build.sh` |
| 镜像内 runtime 检查逐文件核对适配哈希并生成 `source_manifest_sha256` | `scripts/check_runtime.py` |
| launcher 默认先生成 plan；`--start` 前检查镜像源码、模型 receipt 和 8 张空闲 A100；plan 中的 connector JSON 形状校验不证明底层 connector 运行可用 | `launch.py` |
| 启动参数涵盖 FP8 DS MLA KV、Engram CPU offload、tokenizer/tool/reasoning parser、prefix cache、chunked prefill、CUDA Graph、EPLB、DSpark、视觉上限和 Responses 单 API worker | `launch.py`；`config/runtime/environment.json` |
| Responses 历史保存在进程内，LRU 淘汰 response 时联动删除 message/event，SSE 结束清理 event queue，重启后历史清空 | `docs/architecture.md`；`adaptation/files/vllm/entrypoints/openai/responses/serving.py` |
| 长工具历史编码只扫描一次边界，owned-message 路径避免重复深拷贝；CPU 基准与参考编码器逐字一致 | `docs/architecture.md` 的 “编码热路径” |
| `bounded-prefill`、`prefill-throttle`、`stacked-kv`、`decode-metadata`、`ragged-reuse` 都是显式实验开关；显存预算计入模型前向、初始化、完整图捕获、KV 和 Attention workspace | `docs/architecture.md` 的 “长上下文与 decode 优化” / “显存预算与 Attention 工作区” |

## Benchmark 方法

| 规则 | 当前来源 |
|---|---|
| 配置矩阵是 262K / 524K / 1M × `0.90 / 0.925 / 0.95`，默认 16K chunk、`TP4 / DP2 / EP8`、DSpark k5、CUDA Graph、FP8 DS MLA KV | `docs/performance.md`；`config/benchmarks/profile_matrix.json` |
| 统一记录输入/输出/总 TPS、TTFT、TPOT、ITL、E2E；失败、长度异常和缓存异常不计正式结果 | `docs/performance.md` 的 “测试方法” / “指标口径” |
| 启动后先做通用预热；逐点预热按配置执行；正式请求要求 `cached_tokens=0`，媒体历史和 Responses 续轮保留协议缓存语义 | `docs/performance.md` 的 “预热与缓存” |
| PPL 使用固定 134 个窗口、272,802 个计分 token、`echo`、`top_k=1` 和 `seed=1017`；GPQA 为 198 题；持续负载为 1,800 秒 | `docs/performance.md` 的 “精度与稳定性测试”；`docs/validation.md` |
| 每个正式结果和失败 attempt 都需要 schedule/grid、payload、metrics/hash、warmup、source/image/container、PPL/GPQA/soak evidence；缺失或未哈希的失败 resolution 不能 `FINAL_PASS` | `tests/test_profile_report.py` |
| 多模态验收验证 endpoint、launch receipt、fixture、媒体 identity/hash、视频历史和 chunk A/B 的同语义条件 | `scripts/acceptance/visual.py` |
| Responses 验收要求真实 HTTP、custom/function tool、SSE 生命周期、`previous_response_id` 和多模态顺序；缺 store 或多 worker 时记 `BLOCKED` | `tests/test_acceptance_responses.py` |

## 历史优化项迁移边界

| 经验 | 类别 | 可迁移部分 | 必须重测 |
|---|---|---|---|
| SM80 稀疏 MLA、FP8 DS MLA KV | 模型支持/硬件加速 | 分层、开关、数值和容量门禁 | kernel、dtype、layout、精度、显存 |
| DSpark `k5` | speculative/runtime | draft/target 对照和失败归档 | shape、拒绝采样、输出一致性 |
| EP8 custom AG/RS | 通信/并行 | 拓扑和 rank 隔离方法 | 通信后端、rank、故障恢复 |
| TP4/DP2/EP8、TP8/EP8 | 部署并行 | 每组合独立 baseline | 新硬件拓扑、容量和负载均衡 |
| CUDA Graph | runtime 优化 | eager 对照、capture 记录 | shape、地址、并发、驱动 |
| `bounded-prefill`、`prefill-throttle` | 调度优化 | 单变量 A/B 和 SLO 取舍 | 新模型 scheduler、chunk、负载 |
| `stacked-kv`、indexer sharding | 模型/runtime 优化 | 依赖图和实验开关 | 权重、MLA/indexer、并行布局 |
| 多模态 chunk、视频采样 | processor/scheduler | 上限、ledger、边界 chunk | processor 契约、训练语义、时间映射 |
| 1M budget、KV admission | 资源管理 | 联合预算和拒绝门禁 | workspace、graph、KV capacity、allocator |
| Responses store LRU | 协议/资源管理 | 联动淘汰、SSE 清理、重启语义 | 多 worker、外置 store、一致性 |

## 不可迁移为默认值的项目特例

- DeepSeek-V4.1-Flash 的稀疏 MLA、FP8 DS MLA KV、DSpark `k5`、SM80 MQA、EP8 custom AG/RS、TP4/DP2/EP8 和 1M 容量数字。
- A100 的 compute capability、显存水位、NVSwitch 拓扑、CUDA Graph shape、kernel、allocator 告警阈值和 connector 对齐约束。
- 视频最多 32 帧、`1 FPS`、图片/视频数量、视觉 token 预算以及服务端视频 wrapper。
- 历史 benchmark 的 TPS、TTFT、PPL、GPQA、queue、A/B 增益和 soak 成功数。

可复用方法是 provenance、模型/硬件分层、最小 bring-up、协议回放、容量门禁、connector 最小矩阵、首错归因、单变量 benchmark、失败归档、session 生命周期、媒体 ledger、canary 和回滚。

## 使用规则

1. 先按当前 checkout、镜像、模型和硬件重新核验本文件中的稳定事实。
2. 需要历史 TPS、TTFT、PPL、A/B 或具体事故链时，再读取 [historical-runs.md](historical-runs.md)。
3. 历史结果只能作为实验设计线索；当前结论必须使用当前 evidence record。
4. 当前身份与快照不同时，旧结果保持 `HISTORICAL`，并重新执行至少阶段 1、5、6、7、8，再按风险补齐 9–13。
