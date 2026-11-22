# Evidence contract

每次构建、启动、测试、故障复现、回滚和发布都生成一条 JSON evidence record。字段、枚举和嵌套结构以 [evidence-schema.json](evidence-schema.json) 为唯一规范；本文件只定义含义、生命周期和校验要求。

record 放在对应阶段的 artifact 目录下，命名为 `<record-id>.evidence.json`，例如 `artifacts/07-bringup/bringup-chat-01.evidence.json`。

## 两个正交维度

| 维度 | 取值 | 含义 |
|---|---|---|
| `evidence_class` | `current` | 当前 checkout、当前镜像和当前环境中实际执行并可复核 |
|  | `historical` | 过去执行过，可复核但不能自动支撑当前结论 |
|  | `assumption` | 只有假设或计划，尚未执行；只能配合 `NOT_RUN` |
| `scope` | `general` | 可作为通用迁移规则 |
|  | `model_hardware_specific` | 只在声明的模型、GPU、驱动、CUDA、镜像或并行方式下成立 |

`HEAD` 变化、镜像变化、模型变化或测试配置变化后，旧 record 保持 `historical`。历史 `PASS` 可以作为参考，不能进入当前 release evidence。

## 状态与标签

- `PASS`：在 record 绑定的环境和配置中实际执行并通过门禁。
- `FAIL`：已执行且不满足门禁，保留首错和原始结果。
- `BLOCKED`：能力本来适用，但硬件、权限、来源或前置条件缺失。
- `SKIP`：按预先声明的范围主动跳过，并记录原因。
- `NA`：目标契约不适用，并记录“不适用”依据。
- `NOT_RUN`：已有计划但尚未执行，不能用于通过结论。
- `HISTORICAL`：报告标签，对应 `evidence_class: historical`，不替代执行状态。
- `UNRESOLVED`：报告标签，对应 `failure.root_cause_status: unresolved`，不替代执行状态。

## 创建与校验

1. 保存原始日志、请求/响应、benchmark 数据、截图或其他 artifact。
2. 按 JSON Schema 写出 evidence record；不适用字段使用 `null`、空数组或声明的枚举值。
3. 为每个 artifact 计算实际文件的 SHA256，并填写相对项目根目录的路径。
4. 运行校验器：

```bash
python3 <skill-dir>/scripts/verify_evidence.py \
  --root . artifacts
```

5. 只有校验通过后，record 才能进入报告或 gate 判断。

校验器至少检查：

- JSON Schema 的必填字段、类型、枚举、格式和未知字段。
- `record_id` 唯一，时间戳带时区且结束时间不早于开始时间。
- artifact 使用相对路径、文件真实存在、SHA256 与实际内容一致。
- `PASS` / `FAIL` 至少绑定一个 case 和 artifact；`FAIL` 包含首错、责任层、最小复现和下一步。
- `BLOCKED`、`SKIP`、`NA` 和 `NOT_RUN` 写明原因及后续动作。
- `PASS` 不使用 `dirty-unclassified` 工作区，也不带未闭合根因。
- 阶段 6 及之后的 `PASS` 使用固定 source revision，并包含 image、model、adaptation、launch 和 test provenance；GPU 阶段还包含 hardware、driver、CUDA、framework 和 parallelism。

## 结论边界

CPU 静态检查、schema 校验、镜像检查、GPU bring-up、正式 benchmark 和 release gate 分开记录。计划、smoke test、正式测量和发布结果分开记录。请求排队、请求失败、worker 首错和下游通信错误分开记录。

构建和启动至少锁定：源码 commit、基础/最终镜像 digest、模型 revision、权重/tokenizer/processor 文件 hash、适配 manifest hash、GPU/驱动/CUDA、TP/DP/EP/EPLB、启动参数 hash 和测试配置 hash。启动计划只说明将执行什么，receipt 只说明已经执行什么。
