---
name: model-hardware-adaptation
description: Port or migrate models in vLLM, adapt vLLM to new GPU/runtime stacks, or diagnose hardware-specific startup, OOM, collective, KV-capacity, protocol, and performance regressions. Use for adaptation and migration work; handle routine API usage and isolated metric lookups directly.
metadata:
  short-description: 证据驱动的 vLLM 模型与硬件迁移工作流
---

# vLLM 模型与硬件适配工作流

## 使用边界

在以下任务中使用本 skill：

- 把新模型迁入 vLLM，或跟进新的 vLLM 版本、runtime 和依赖基线。
- 适配新的 GPU 架构、机器、驱动、CUDA、容器或通信栈。
- 建设协议、多模态、长上下文、KV、容量、性能、canary 或回滚门禁。
- 诊断启动失败、OOM、worker crash、Gloo/NCCL、CUDA Graph、吞吐或延迟回归。

单次 API 用法、普通参数解释或孤立指标查询直接处理；当同一请求还涉及迁移、运行时差异、协议验收或部署风险时，使用本 skill。

## 输入契约

先建立 `inventory`。未知值写 `UNKNOWN`；缺项阻塞适用阶段时，阶段状态写 `BLOCKED`。

| 输入 | 最低要求 |
|---|---|
| 项目 | 根目录、分支、HEAD、工作区状态、未提交改动及归属 |
| 模型 | 名称、revision、权重来源、配置、tokenizer、processor、renderer、权重清单 |
| vLLM | 发布版本、源码 commit、适配补丁、上游支持矩阵 |
| 镜像 | 基础镜像 digest、构建方法、最终镜像 digest、CUDA/Python 运行时 |
| 硬件 | GPU 型号、数量、显存、计算能力、拓扑、驱动、CUDA、空闲状态 |
| 启动 | 启动脚本或规范化参数、环境变量、端口、进程拓扑 |
| 协议 | Chat Completions、Responses、工具类型、流式事件、reasoning、多模态 schema |
| 目标 | 上下文、最大输入输出、并发、吞吐/延迟 SLO、精度和容量门禁 |
| 并行 | TP、DP、EP、EPLB、通信后端、CUDA Graph、speculative decoding、KV 策略 |
| 既有证据 | 测试、日志、receipt、benchmark 配置、artifact、未完成实验 |

## 执行循环

1. **冻结基线。** 读取项目文档、代码、配置、测试、启动器和既有证据，记录 source、image、model、hardware 和未提交改动。完成标准：输入契约中的每一项都有来源或 `UNKNOWN`，每项未提交改动有归属；否则停止盘点并报告缺口。
2. **建立能力矩阵。** 按 [acceptance-matrix.md](references/acceptance-matrix.md) 列出目标契约声明的能力，逐项写触发依据、可行性、下一门禁和 `NA` 依据。完成标准：每项声明能力都有可行性标签和来源，每项不适用能力都有明确依据。
3. **逐阶段执行。** 读取 [migration-checklist.md](references/migration-checklist.md)，一次执行一个适用阶段；每个阶段完成后写 JSON evidence record，并用 [verify_evidence.py](scripts/verify_evidence.py) 校验。完成标准：当前阶段满足 checklist 的输入、动作、完成、停止和证据条件，且 evidence record 校验通过；满足后再进入下一阶段。
4. **关闭适用门禁。** 按能力矩阵执行协议、多模态、长上下文、KV、容量、性能、故障、发布和回滚门禁。完成标准：每项声明能力都有当前状态；`PASS` 只由当前 `evidence_class: current` 且校验通过的 artifact 支撑，`NA` 有契约依据。
5. **生成迁移报告。** 复制 [adaptation-report.md](templates/adaptation-report.md)，填入 inventory、能力矩阵、阶段状态、故障和 artifact 索引。完成标准：通过下方“交付门禁”。

遇到首个可信 worker、CUDA、schema、OOM 或来源错误时，保存首错并进入 [failure-taxonomy.md](references/failure-taxonomy.md)；恢复变量时一次只改一个主要变量。

## 不可跳过的约束

1. **Provenance。** source、image、model、适配层、hardware、driver、CUDA、parallelism 和测试配置必须绑定；启动计划不等于服务可用。
2. **能力范围。** 只验收目标契约声明的能力；不适用能力写 `NA` 和依据。
3. **分层归因。** 模型、协议、kernel、通信、scheduler/KV、网关和部署分别定位；一个层的通过不能替代另一个层。
4. **首错。** 失败时先保存最早可信错误，再处理超时、HTTP 500、Gloo/NCCL 等下游症状。
5. **单变量。** 性能与故障恢复固定 baseline，一次只改变一个主要变量；cache、queue 和失败请求分开记录。
6. **证据闭环。** 每个阶段都生成 evidence record；状态词、字段和校验规则以 [evidence-contract.md](references/evidence-contract.md) 与 [evidence-schema.json](references/evidence-schema.json) 为准。

## 参考路由

只读取完成当前决策所需的参考：

- 组合决策、能力标识、正确性矩阵或性能矩阵：读取 [acceptance-matrix.md](references/acceptance-matrix.md)。
- evidence 状态、记录格式、artifact 哈希或校验：读取 [evidence-contract.md](references/evidence-contract.md) 和 [evidence-schema.json](references/evidence-schema.json)。
- 启动、KV、协议、OOM、通信或部署故障：读取 [failure-taxonomy.md](references/failure-taxonomy.md)。
- 具体阶段的动作、完成标准、停止条件和证据目录：读取 [migration-checklist.md](references/migration-checklist.md)。
- 当前仓库的稳定事实和迁移边界：读取 [project-evidence.md](references/project-evidence.md)。
- 当前仓库的历史指标、会话标识或事故链：读取 [historical-runs.md](references/historical-runs.md)。
- 交付报告骨架：复制 [adaptation-report.md](templates/adaptation-report.md)。

## 交付门禁

迁移报告只有在全部条件成立时才视为完成：

- 每项声明能力和每个适用阶段都有状态、依据和 artifact 路径。
- 每个用于当前结论的 `PASS` 都通过 `verify_evidence.py`，且 `evidence_class` 为 `current`。
- 每个 `FAIL` 都记录首错、责任层、最小复现和下一步；每个 `BLOCKED`、`SKIP`、`NA` 或 `NOT_RUN` 都写明原因。
- 所有 artifact 路径存在，SHA256 与文件内容一致；历史结果、当前结果和假设没有混写。
- 最终结论没有把未执行的必需门禁写成通过，并明确剩余风险、限制和停止条件。

校验命令：

```bash
python3 <skill-dir>/scripts/verify_evidence.py --root <project-root> <evidence-path-or-directory>
```
