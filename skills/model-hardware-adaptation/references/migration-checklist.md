# Migration checklist

本文件是阶段编号、名称、动作和证据目录的唯一规范。每个阶段都要写输入、动作、输出、完成条件、失败停止条件和证据位置；目录编号与阶段号一致。阶段由 [acceptance-matrix.md](acceptance-matrix.md) 的能力条件裁剪，状态词按 [evidence-contract.md](evidence-contract.md) 使用。

## 阶段 1：盘点与基线冻结

- **输入**：项目目录、目标、Git 状态、文档、测试、历史会话。
- **动作**：读取 README、架构/性能/验证文档、适配/来源清单、启动器、配置、测试和运行目录；记录分支、HEAD、未提交改动和结论类别。
- **输出**：`inventory`、范围矩阵、风险清单、baseline lock。
- **完成**：每项目标能力有代码/文档/测试/运行证据或 `UNKNOWN`；改动有归属；source/image/model/hardware/参数身份齐全。
- **停止**：无法确定 checkout、改动归属、模型来源或目标协议。
- **证据**：`artifacts/01-inventory/`。

## 阶段 2：模型与硬件可行性

- **输入**：inventory、上游支持矩阵、模型配置、GPU 规格。
- **动作**：分开核对架构/权重/tokenizer/processor/renderer/协议和 compute capability/显存/拓扑/驱动/CUDA/kernel/通信。
- **输出**：模型表、硬件表、显存预算、组合风险。
- **完成**：每项标 `SUPPORTED`、`ADAPT`、`REVALIDATE`、`UNSUPPORTED` 或 `UNKNOWN`，并有来源；无法继续执行的阶段状态另记 `BLOCKED`。
- **停止**：上游不支持且没有可核验边界，或预算不足。
- **证据**：`artifacts/02-feasibility/`。

## 阶段 3：上游差异与缺口

- **输入**：可行性表、上游 commit、适配层和失败记录。
- **动作**：按模型、硬件、协议、多模态、调度、KV、部署定位调用方、配置入口、测试和回滚点。
- **输出**：gap list、变更边界、依赖图、验证计划。
- **完成**：每个改动点有公共接口、兼容范围、测试和证据字段。
- **停止**：找不到首个可信调用链，或会改变未声明公共契约。
- **证据**：`artifacts/03-gap-analysis/`。

## 阶段 4：适配文件与接口设计

- **输入**：gap list、上游接口、目标协议、并行和回滚要求。
- **动作**：设计目录、manifest、原子安装、环境变量、CLI、协议转换、processor/renderer、scheduler/KV 和实验开关。
- **输出**：接口契约、schema、回滚方案、测试矩阵。
- **完成**：可指纹化、可生成 plan、可单项关闭、边界清楚。
- **停止**：依赖隐式全局状态、不能回滚或无法绑定来源。
- **证据**：`artifacts/04-design/`。

## 阶段 5：CPU、静态、hermetic

- **输入**：实现、schema、manifest、CPU 环境。
- **动作**：运行 JSON/schema、tokenizer/renderer golden、unit、热路径、lint、安装、无 GPU runtime 和 CLI help；不适用项写 `NA`，缺环境写 `BLOCKED`。
- **输出**：静态报告、golden diff、coverage 和未覆盖清单。
- **完成**：所有预先声明的 schema、golden、unit、lint、安装和无 GPU runtime 检查均为 `PASS`；每项 `NA`、`SKIP`、`BLOCKED` 或 `NOT_RUN` 有原因和范围；CPU 结果未写成 GPU `PASS`。
- **停止**：必要 schema、golden、导入、安装或静态检查失败。
- **证据**：`artifacts/05-static/`。

## 阶段 6：镜像与 provenance

- **输入**：干净源码、基础镜像 digest、模型/适配清单。
- **动作**：固定 digest 构建，尽量关闭网络，从指定 commit 导出；运行逐文件 runtime 检查、导入和 CLI help。
- **输出**：镜像 digest、source/model receipt、构建日志、`GPU validation: NOT_RUN`。
- **完成**：source manifest、最终 image digest、model file list 和 adaptation manifest hash 全部绑定；构建日志与来源快照已保存；阶段 7 之前 GPU 状态保持 `NOT_RUN`。
- **停止**：digest、hash 或工作区不一致。
- **证据**：`artifacts/06-image/`。

## 阶段 7：GPU 最小 bring-up

- **输入**：阶段 6 receipt、模型、硬件、plan。
- **动作**：只执行适用链路：最小文本 → 单图 → 工具 → Responses → 视频/多图/长上下文 → 性能压力；每一步检查 health、models、worker、graph、首请求和资源。
- **输出**：launch receipt、health、最小请求、GPU/日志快照。
- **完成**：每个声明的 minimum path 都能启动，health、models 和最小文本请求通过；无未解释的 worker、OOM、collective 或 source 错误；GPU 状态、启动配置、日志和首请求 artifact 已归档。
- **停止**：首个 worker 或请求错误；保存首错并转故障流程。
- **证据**：`artifacts/07-bringup/`。

## 阶段 8：核心协议验收（条件执行）

- **输入**：目标协议、工具定义、历史样例。
- **动作**：Chat 分支先验证 messages、streaming、停止和错误 schema，声明 tools 时再验 `tool_calls` / `role=tool`；Responses 分支验证 output、SSE、续轮，声明 tools 时再验 function/custom call 和工具结果；无该协议则 `NA`。
- **输出**：HTTP/SSE 报告、payload hash、事件序列。
- **完成**：每个声明协议和声明功能的首轮、续轮、流式及错误路径均通过；失败可定位到责任层；未声明分支为 `NA`。
- **停止**：schema、事件顺序、工具结果或历史重放失败。
- **证据**：`artifacts/08-protocol/`。

## 阶段 9：多模态与长上下文（条件执行）

- **输入**：视觉/长上下文契约、fixtures、媒体上限、chunk 策略。
- **动作**：视觉能力按单图、多图、视频、混排、历史媒体和超限顺序执行；纯文本或无长上下文契约写 `NA`。
- **输出**：能力矩阵、media ledger、长请求报告。
- **完成**：每个声明的媒体和长上下文能力都验证接受输入、边界输入、超限拒绝、token/帧数、顺序、缓存身份和 chunk 边界；实际值与预期值及对应 artifact 已记录。
- **停止**：媒体顺序、帧映射、缓存身份或长输入不一致。
- **证据**：`artifacts/09-multimodal-long/`。

## 阶段 10：显存、KV 和容量（条件执行）

- **输入**：容量目标、KV 配置、并行和 connector 选择。
- **动作**：联合预算 forward/init/graph/workspace/KV/allocator；native、offload、transfer 只测目标契约声明的分支。
- **输出**：容量表、admission、connector 报告。
- **完成**：每个声明的最大 workload 要么有通过结论及 capacity/queue/allocator 数据，要么明确 `FAIL` 或 `BLOCKED` 并保留首错；排队只有在请求最终失败时计为失败。
- **停止**：connector 启动/首请求失败、OOM 或 worker crash。
- **证据**：`artifacts/10-kv-capacity/`。

## 阶段 11：性能 A/B 与正式矩阵（条件执行）

- **输入**：功能/协议/容量门禁、baseline、工作负载、指标定义。
- **动作**：固定 source/image/model/hardware/parallelism/seed/request/cache/warmup；一次只改一个主要变量；按目标选择上下文、显存、chunk、并发、cache、调度、精度和 soak。
- **输出**：测量记录、矩阵、A/B、失败归档、采用决策。
- **完成**：每个必需矩阵行都具有预先声明的重复数、固定 baseline、warmup/cache 状态和完整指标；失败、长度异常和缓存异常已单独归档，未进入正式数据集。
- **停止**：baseline 变化、cache 污染、长度异常、变量不唯一或 receipt 缺失。
- **证据**：`artifacts/11-performance/<campaign>/`。

## 阶段 12：故障归因与回归（按需）

- **输入**：首错、阶段状态、分类、最小复现和修复候选。
- **动作**：从最早可信错误向上游回溯；一次只恢复一个变量；修复前后用相同请求回归。
- **输出**：根因报告、diff、回归矩阵、未决假设。
- **完成**：根因由首错、最小复现和代码/配置 diff 共同支持；修复前后使用相同请求和身份回归；证据不足时 `root_cause_status: unresolved`，阶段不得记 `PASS`。
- **停止**：缺首错、复现不稳定或多变量修复不可归因。
- **证据**：`artifacts/12-faults/<incident>/`。

## 阶段 13：安全切换、回滚与发布（按需）

- **输入**：正式矩阵、旧/新 receipt、回滚包、发布门禁。
- **动作**：canary、健康、协议、关键 workload、监控和回滚演练；记录服务端 session/store、缓存、重启和外部状态。
- **输出**：切换、canary、rollback、release receipt。
- **完成**：新旧身份可比，canary、关键协议、关键 workload、监控和回滚演练全部通过；服务端状态、重启语义和外部状态影响已归档。
- **停止**：来源不一致、canary 或回滚失败。
- **证据**：`artifacts/13-release/`。

## 阶段 14：迁移结论与归档

- **输入**：全部 evidence、故障、A/B、发布状态。
- **动作**：分为可迁移规则、模型特例、硬件特例、待验证假设；记录采用、拒绝和 backlog。
- **输出**：迁移报告、证据索引、限制和下一步。
- **完成**：每项声明能力和每个适用阶段都有状态与已校验 evidence；每个数字可追溯；`HISTORICAL`、当前结果和假设分开；artifact 目录通过 `verify_evidence.py`。
- **停止**：结论无来源或 artifact 不存在，阶段状态记 `BLOCKED`；根因未闭合时另标 `UNRESOLVED`。
- **证据**：`artifacts/14-conclusion/`。
