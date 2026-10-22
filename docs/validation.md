# 验证记录

本文记录适配层已完成的功能、协议与精度验证，性能数据见 [基准结果](benchmarks.md)。

## 构建与来源校验

| 检查项 | 入口 | 结果 |
|---|---|---|
| 模型文件与权重校验 | `scripts/model_check.py` | 固定 ModelScope revision，83 个文件（含 48 个权重分片）大小与 SHA256 全部通过 |
| 适配层原子安装 | `adaptation/install.py` | 33 个适配文件与 `adaptation/manifest.json` SHA256 校验一致后原子写入 |
| 运行时源码复核 | `scripts/check_runtime.py` | 容器内已安装文件与源码清单指纹一致 |
| CPU 合约与回归测试 | `unittest` | 59 项通过（2 项 PyTorch 运行时测试在纯 CPU 环境跳过） |
| 固定 digest 镜像构建 | `scripts/build.sh` | 镜像构建、runtime 校验与原生 MXFP8 loader 检查通过 |

## 协议与模型功能

- **工具调用与推理续轮**：多轮工具续轮中的 `reasoning_content`、反序工具结果归并和图片顺序与官方编码器一致。
- **消息渲染一致性**：同步与异步 renderer 共用同一消息归一化与排序逻辑，跨 `user` / `system` / `assistant` 边界自动隔离工具状态。
- **官方编码器对齐**：官方 5 组输入夹具覆盖 40 种思考模式与 `reasoning_effort` 组合，编码输出逐字匹配。
- **原生多模态**：单图、多图、缓存命中图片、重复图片及完整 1024-token 视觉预算边界全部通过。
- **分布式与投机解码**：`tp8 / ep8` 与 `tp4dp2 / ep8`、CUDA Graph、FP8 MLA KV cache、DSpark k5、EPLB 和 EP8 通信路径均完成目标机启动验收，并通过文本、工具和图文 smoke。
- **长上下文容量**：260K 与 1,000,016 token 的单请求 prompt 均完成预填充验收。

## 精度准入

每个性能候选通过 `scripts/acceptance.py` 与 `scripts/quality_gate.py` 在固定请求集（`temperature=0`、`seed=1017`）上进行对比校验：

- 对文本、`reasoning_content`、结构化 `tool_calls`、工具续轮、图文和长上下文输出计算 SHA256 摘要。
- 候选与基线的全部 case 均需通过且输出摘要一致，方可纳入默认推荐配置。
- 当服务端返回 `draft_tokens` / `accepted_tokens` 计数时，`scripts/quality_metrics.py` 自动汇总投机解码接受率。

## GPQA Diamond

基于完整 198 题（`eval_batch_size=32`、`top_p=0.95`、`max_tokens=65536`、`reasoning_effort=max`、`temperature=0`）完成 4 轮独立评测：

| 轮次 | 准确率 |
|---:|---:|
| Round 1 | 92.42% |
| Round 2 | 90.40% |
| Round 3 | 92.93% |
| Round 4 | 90.40% |
| **平均** | **91.54%** |

4 轮均为 `198/198` 完整评分，失败率为 `0`。
