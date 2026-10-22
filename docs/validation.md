# 验证记录

本文记录当前适配层已经完成的功能、协议和精度验证。性能数字统一见 [基准结果](benchmarks.md)。

## 构建与来源

| 项目 | 结果 |
|---|---|
| 官方 1017 编码器来源 | 固定 revision，SHA256 与参考一致 |
| 官方文件清单 | 83 个文件的大小和 SHA256 全部一致 |
| 1017 权重 | 48/48 分片完整 SHA256 通过 |
| 适配层安装 | 全量 manifest 校验后原子写入 |
| CPU 合约测试 | 59 项通过，2 项在无 PyTorch 环境跳过 |
| Python 编译与静态检查 | 通过 |
| 固定 digest 镜像 | 构建、runtime 和 native loader 检查通过 |

## 协议与模型功能

- 工具续轮中的 reasoning 内容、工具结果顺序和图片关联与官方编码器一致。
- 同步和异步 renderer 共用稳定的消息排序逻辑。
- 官方五组输入文件覆盖 40 种模式和 effort 组合，输出逐字匹配。
- 原生图文输入、单图、多图、缓存图片和重复图片边界通过。
- `tp8 / ep8` 与 `tp4dp2 / ep8` 均完成启动和文本 smoke。
- CUDA Graph、FP8 KV cache、DSpark k5、EPLB 和 EP8 通信路径完成目标机启动验收。
- 260K 单请求和约 1M 单请求均完成容量验收。

## 精度准入

每个性能候选使用固定请求和 `seed=1017` 运行 acceptance suite，并比较 assistant-visible 输出摘要。文本、reasoning、结构化 `tool_calls`、工具续轮、图文和长上下文结果都纳入摘要；摘要缺失或变化时，候选不进入推荐配置。

服务没有返回 draft/accepted 计数时，speculative acceptance 标记为 unavailable，不从 TPS 反推接受率或精度。

## GPQA Diamond

使用完整 198 题、`eval_batch_size=32`、`top_p=0.95`、`max_tokens=65536`、`reasoning_effort=max` 和 temperature 0，完成 4 轮重复：

| Round | Score |
|---:|---:|
| 1 | 92.42% |
| 2 | 90.40% |
| 3 | 92.93% |
| 4 | 90.40% |
| 平均 | **91.54%** |

4 轮均为 198/198 评分，失败率为 0。当前没有旧实现的同配置 GPQA 结果，因此该分数作为当前实现的精度基准，不能单独计算相对旧版本的精度变化。

## 验证边界

当前结果覆盖固定 1017 权重、文本、工具、图文、长上下文、并发和 mixed prefill/decode case。全模型 PPL、2×/4× 1M 多请求容量以及未经 GPU 验收的实验开关不属于 `v0.1.0` 的默认性能结论。
