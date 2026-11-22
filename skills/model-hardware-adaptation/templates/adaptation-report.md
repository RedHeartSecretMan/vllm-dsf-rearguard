# vLLM 模型与硬件适配报告

> 能力和阶段名称分别以 [acceptance-matrix.md](../references/acceptance-matrix.md) 与 [migration-checklist.md](../references/migration-checklist.md) 为准。状态和 evidence 字段以 [evidence-contract.md](../references/evidence-contract.md) 为准。

## 1. 目标与范围

目标模型：
目标硬件：
当前 vLLM 基线：
目标 vLLM 基线：
目标协议：
声明能力：
不适用能力及依据：

## 2. Provenance 与 inventory

- 项目根目录：
- 分支 / source revision：
- 工作区状态：
- 当前未提交改动及归属：
- image digest：
- model revision：
- model file artifacts：
- adaptation manifest SHA256：
- hardware / driver / CUDA：
- parallelism：
- launch args hash：
- test config hash：

## 3. 能力矩阵

从 `acceptance-matrix.md` 复制每项声明能力并补齐以下列；未声明能力保留 `NA` 及依据。

| 能力 | 触发依据 | 可行性 | 下一门禁 | artifact |
|---|---|---|---|---|
|  |  |  |  |  |

模型特有缺口：
硬件特有缺口：
协议和多模态缺口：
并行策略：
上下文目标：
显存和 KV 预算：
最小 bring-up：
回滚方案：

## 4. 阶段状态

从 `migration-checklist.md` 复制适用阶段及 `NA` 阶段；状态必须绑定 evidence record。

| 阶段 | 状态 | 完成 / 停止依据 | evidence record |
|---|---|---|---|
|  |  |  |  |

## 5. 验收门禁

按下表汇总每项声明能力的正确性、协议、媒体、容量和性能结果。

| 能力 / 门禁 | 结果 | 通过或失败依据 | evidence record |
|---|---|---|---|
|  |  |  |  |

## 6. 风险与故障

| 风险 / 故障 | 首错 | 责任层 | 状态 | 最小复现 | 下一步停止条件 | evidence record |
|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |

## 7. Evidence record 索引

运行校验器后再填写本节。artifact 路径相对项目根目录，SHA256 对应真实文件。

| evidence record | artifact path | SHA256 | evidence class | scope |
|---|---|---|---|---|
|  |  |  |  |  |

校验命令：
校验结果：

## 8. 最终结论

已验证事实：
历史证据：
模型/硬件特例：
未验证假设：
未覆盖范围：
最终状态：
剩余风险和停止条件：
下一步：
