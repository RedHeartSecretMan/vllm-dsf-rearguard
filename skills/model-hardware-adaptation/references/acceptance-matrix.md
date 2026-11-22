# Acceptance matrix

本文件是能力标识、触发条件、正确性维度和性能维度的唯一规范。先生成 capability matrix，再生成阶段选择和验收矩阵；状态词按 [evidence-contract.md](evidence-contract.md) 使用。

## 迁移组合决策

| 组合 | 先验证 | 结论边界 |
|---|---|---|
| 新模型 / 旧硬件 | 模型加载、tokenizer、renderer、协议、kernel、权重和精度 | 旧硬件能力可复用；模型和协议必须重测 |
| 旧模型 / 新硬件 | compute capability、显存、拓扑、驱动、kernel、通信、图捕获 | 模型契约可复用；硬件路径和容量必须重测 |
| 新模型 / 新硬件 | 完整阶段 1–14，按能力条件裁剪 | 只有全适用门禁闭合后才能发布 |
| 同一模型 / 不同 vLLM | 上游差异、API schema、适配 manifest、baseline | 新版本建立新 provenance；历史结果不自动继承 |
| 同一服务 / 不同 API 协议 | schema、parser、事件、历史、错误和回放 | Chat 与 Responses 分开发布 |
| 多模态模型 / 纯文本模型 | processor、媒体限制、视觉 token、缓存和 chunk | 纯文本通过不能推出多模态可用 |
| 单卡、TP、DP、EP、混合并行 | rank 映射、collective、容量、调度和故障 | 每种组合独立记录和比较 |
| native KV、CPU offload、KV transfer | schema、启动、首请求、长请求、历史恢复 | 启动通过不等于 connector 可用 |

可行性标签只描述下一步工作：`SUPPORTED` 可按当前上游路径使用，`ADAPT` 需要本地改动，`REVALIDATE` 有支持证据但目标环境必须重测，`UNSUPPORTED` 无可行路径，`UNKNOWN` 信息不足。它们不是执行状态；无法继续执行的阶段状态另记 `BLOCKED`。

## 能力条件矩阵

| 能力 | 触发条件 | 最小正确性验收 | 未声明时 |
|---|---|---|---|
| 文本生成 | 所有模型 | tokenizer、短文本、停止条件、错误输入 | 不适用时任务本身不可开始 |
| 图片 | 模型声明 image input | 单图、尺寸/数量上限、视觉 token、混排 | `NA`：契约无图片 |
| 视频 | 模型或适配层声明 video input | 帧采样、时间映射、上限、媒体历史 | `NA`：契约无视频 |
| 工具调用 | 声明 tools | 首轮、工具结果续轮、错误 schema | `NA`：契约无 tools |
| Chat Completions | 暴露 Chat API | messages、streaming、停止和错误 schema；声明 tools 时再验 `tool_calls` / `role=tool` | `NA`：契约无 Chat |
| Responses | 暴露 Responses API | output、SSE、续轮；声明 tools 时再验 function/custom call 和工具结果 | `NA`：只支持 Chat |
| reasoning | 模型或协议声明 reasoning | reasoning 字段、续轮和输出顺序 | `NA`：无该字段 |
| 长上下文 | 目标声明上下文上限 | 长输入、实际 token、KV admission | `NA`：目标是短上下文 |
| native KV | 服务使用 native KV | 最大目标请求、容量、错误 | 适用时不可省略 |
| CPU offload/KV transfer | 目标部署选择该 connector | schema、启动、首请求、长请求、恢复 | `NA`：目标使用 native KV |
| speculative decoding | 目标启用 draft/target | 输出一致性、capture shape、拒绝采样 | `NA`：目标关闭 speculative |
| 压力/soak | 目标有稳定性 SLO | 固定时长、成功率、错误和资源趋势 | `NA`：目标契约无压力门禁 |

## 正确性矩阵

每行绑定 source/image/model/hardware/parallelism 和测试配置 hash：

| 维度 | 最小项 | 通过条件 |
|---|---|---|
| 来源 | 文件、manifest、镜像、模型 hash | 全部一致 |
| 文本 | 短请求、长输出、错误输入 | 响应 schema 和停止行为正确 |
| 图片 | 单图、边界尺寸、超限 | 视觉 token、顺序和错误正确 |
| 视频 | 单视频、帧数边界、超限 | 采样和时间映射稳定；无静默丢帧 |
| 工具 | 首轮、tool output、续轮 | call/output 成对、历史完整；无 tools 时为 `NA` |
| Responses | output、streaming、previous id；有 tools 时加 function/custom | 事件顺序和续轮可重放 |
| 精度 | golden、PPL 或任务集 | 在预先声明的门禁内；差异有 artifact |
| 重启 | health、models、最小请求 | 进程内历史清空或外置 store 恢复符合契约 |

## 性能矩阵

固定 baseline 后一次只改变一个主要变量。不同项目选择适用行，不要求所有项目执行全部行：

| 维度 | 推荐对照 | 必记指标 |
|---|---|---|
| 上下文 | 目标声明的短/中/长请求；可用约 25% / 50% / 100% 上下文上限作为设计起点 | 输入/输出/总 TPS、TTFT、TPOT、ITL、E2E、KV |
| 显存 | 目标水位或预算策略 | capacity、显存、allocator、错误 |
| chunk | 当前值与相邻预算的等距对照 | prefill、decode、视觉边界、峰值显存 |
| 并发 | 单请求、C4、C16、目标 C | TPS、排队、TTFT 分位数、成功数 |
| cache | cold KV、warm、prefix、媒体 cache | `cached_tokens`、命中率、独立吞吐 |
| 调度 | bounded/unbounded、throttle off/on | prefill/decode 吞吐和延迟权衡 |
| 并行 | TP/DP/EP/EPLB 逐项恢复 | rank、通信、容量和性能 |
| 稳定性 | 固定到达、闭环、soak | 成功率、错误、资源趋势、退出码 |
| 精度 | PPL、任务集、输出 golden | PPL/准确率/差异和 seed |

正式数据不混入失败请求、长度异常或缓存异常。失败和未采用方案保留原始记录，状态词按 evidence contract 标记。

## 项目特例的使用方式

项目中出现过的优化名、历史数值和故障链不是默认门禁。只在当前仓库确有对应证据文件时读取 [project-evidence.md](project-evidence.md)，并先按当前模型、硬件、runtime 和协议身份重新分类为 `SUPPORTED`、`ADAPT`、`REVALIDATE`、`UNSUPPORTED` 或 `UNKNOWN`。复用方法，不复用结论。

## 协议、媒体和 session 压缩的责任边界

- **Agent client**：维护媒体 ledger、摘要、滑动窗口、显式 compaction、工具状态和 `previous_response_id`。
- **Gateway**：负责认证、路由、请求大小/超时和协议字段保真，不静默改写历史项。
- **Provider adapter**：把外部协议映射为模型需要的消息、工具、reasoning 和媒体顺序，并提供可回放的转换记录。
- **vLLM engine**：解析模型输出、调度和资源保护，执行硬上限并返回明确错误；`tool_call_parser` 只解析模型输出，不能修复不支持的输入 schema。

图片和视频数量、视觉 token、帧数、请求大小和上下文上限是请求级保护。视频按帧采样不等于原生视频 token；稳定 `media_id` 或 UUID 可以复用 encoder/prefix cache，同时仍执行同样上限。多轮 Responses 可能重新携带历史媒体。服务端必须保留 agent 仍依赖的媒体，或返回明确错误并记录影响；迁移到其他视觉或长视频模型时，重新验证训练语义、采样、processor、缓存身份和时间映射。

服务端保存 session、响应或流式事件时，存储上限、关联淘汰、SSE 收尾、后台任务取消和重启语义必须一起定义。单进程内存历史在重启后清空；长期持久化需要外置 store，并重新定义一致性、TTL、隐私和回滚。
