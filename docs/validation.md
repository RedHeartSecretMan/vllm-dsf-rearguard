# 验证记录

本文记录适配层已完成的功能、协议与精度验证，性能数据见 [基准结果](benchmarks.md)。

## 构建与来源校验

| 检查项 | 入口 | 结果 |
|---|---|---|
| 模型文件与权重校验 | `scripts/model_check.py` | 固定 ModelScope revision，83 个文件（含 48 个权重分片）大小与 SHA256 全部通过 |
| 适配层原子安装 | `adaptation/install.py` | 42 个适配文件与 `adaptation/manifest.json` SHA256 校验一致后原子写入 |
| 运行时源码复核 | `scripts/check_runtime.py` | 容器内已安装文件与源码清单指纹一致 |
| CPU 合约与回归测试 | `unittest` | uv Python 3.10 本地运行 114 项并有 3 项环境跳过；pinned vLLM runtime 另外执行 Responses custom-tool 套件，包含视频抽样、逐帧时间映射、稳定视频身份、媒体 UUID 和部署限制契约 |
| CPU 热路径检查 | `scripts/hotpaths/check.py` | 34 个用例全部通过，覆盖 decode 元数据、ragged 复用、空 ragged pack 和 context 投影 |
| 固定 digest 镜像构建 | `scripts/build.sh` | 镜像构建、runtime 校验与原生 MXFP8 loader 检查通过 |

## 协议与模型功能

- **工具调用与推理续轮**：多轮工具续轮中的 `reasoning_content`、反序工具结果归并和图片顺序与官方编码器一致。
- **消息渲染一致性**：同步与异步 renderer 共用同一消息归一化与排序逻辑，跨 `user` / `system` / `assistant` 边界自动隔离工具状态。
- **官方编码器对齐**：官方 5 组输入夹具覆盖 40 种思考模式与 `reasoning_effort` 组合，编码输出逐字匹配。
- **原生多模态**：单图、多图、缓存命中图片、重复图片及完整 1024-token 视觉预算边界全部通过。
- **视频扩展**：`video_url` 由 pinned vLLM media IO 在解码前按默认 1 FPS、`num_frames=32` 选取最多 32 个均匀时间点；每个视频保留稳定的 `vid_...` 身份和全局文本 wrapper，并将同一身份写入 vLLM content-part `uuid`，再展开为按时间顺序排列的图片帧 span。启动配置限制为每请求最多 64 张图片和 2 个视频。仓库内覆盖媒体抽样、帧边界、wrapper 和 UUID 合约；GPU endpoint 回执需单独归档。多轮重放相同媒体引用及参数时身份保持稳定，但仍需要同时保留媒体输入或可用的视觉/KV cache。
- **Responses 协议**：custom-tool history 和 streaming custom-tool events 的适配层回归已纳入测试；`scripts.acceptance.responses` 提供 pinned runtime 的 endpoint smoke 入口，实际 endpoint 回执需单独归档。
- **分布式与投机解码**：`tp8 / ep8` 与 `tp4dp2 / ep8`、CUDA Graph、FP8 MLA KV cache、DSpark k5、EPLB 和 EP8 通信路径均完成目标机启动验收，并通过文本、工具和图文 smoke。
- **实验开关**：`decode-metadata`、`ragged-reuse`、`stacked-kv` 分别完成目标机启动并通过文本和工具 smoke；`bounded-prefill` 与 `prefill-throttle` 在 1M 配置下完成启动和混合负载测试，结果见 [基准结果](benchmarks.md)。
- **长上下文容量**：260K 与 1,000,016 token 的单请求 prompt 均完成预填充验收；4 个和 16 个 ISL 1047000 的请求同时进入均全部成功，16 个请求超过 KV cache 最大并发 4.03 后按容量排队，没有错误或中止。

## 验收 case

功能验收和质量评测按职责拆成四份 catalog；配置中的 `runner` 是可直接用 `python -m` 调用的模块。视觉与 Responses case 执行真实 GPU 接口请求，保存请求、响应、期望值和失败详情；代码级回归由 `tests/` 单独验证。

| 类别 | 配置 | 入口 | 内容 |
|---|---|---|---|
| core | [`core.json`](../config/acceptance/core.json) | `scripts.acceptance.core` | 文本、工具续轮和基础图像 |
| visual | [`visual.json`](../config/acceptance/visual.json) | `scripts.acceptance.visual` | 图像上限、视频采样、混排和 chunk 边界 |
| responses | [`responses.json`](../config/acceptance/responses.json) | `scripts.acceptance.responses` | Responses 基础请求、custom-tool history 和 streaming |
| quality | [`quality.json`](../config/acceptance/quality.json) | `scripts.acceptance.quality` | PPL 和 GPQA Diamond 质量评测计划 |

每个入口都支持 `--list`。视觉与 Responses 共用合成的数字图片和短、长视频；先用 FFmpeg 生成素材，再传入真实启动回执和结果目录：

```bash
python3 -m scripts.acceptance.visual --prepare-fixtures --fixtures-dir runtime/visual-fixtures
python3 -m scripts.acceptance.visual --case all --fixtures-dir runtime/visual-fixtures \
  --launch-receipt runtime/<启动时间戳>/launch.json --output runtime/visual-acceptance
python3 -m scripts.acceptance.responses --case all --fixtures-dir runtime/visual-fixtures \
  --launch-receipt runtime/<启动时间戳>/launch.json --output runtime/responses-acceptance.json
```

视频 chunk 对照需分别启动关闭、开启两种模式：先运行 `visual-video-chunked-disabled`，再向开启模式的 runner 传入 `--chunk-baseline <关闭模式结果目录>/visual-video-chunked-disabled/case.json`。两侧使用相同请求并校验相同语义输出；开启模式还要求未缓存的输入超过 chunk 预算。`--case all` 会将当前启动模式不适用的 case 记为 `NOT_RUN`，完整验收需合并两种模式的结果。处理器时间映射和调度边界另做运行时结构检查。

`previous_response_id` 验收需要启动时加上 `--responses-store`，启用进程内历史存储并使用单个 API 进程；GPU 的 TP / DP 配置不变。历史仅在该服务进程存活期间有效，重启会清空，默认关闭。

catalog 的统计范围是 24 个 benchmark case、27 个 acceptance case；core 的 5 个 `needle_tokens` 是按目标长度动态追加的长上下文 case。编码器基准 `scripts.benchmarks.bench_encoding` 和 `scripts.hotpaths.check` 属于 CPU / 构建支持检查，结果分别写入 `docs/evidence/encoding-benchmark.json` 和 `docs/evidence/hotpaths-cpu.json`，不代表 GPU endpoint acceptance，也不重复放入上述 catalog。

## 精度准入

每个性能候选通过 `scripts.acceptance.core` 与 `scripts.acceptance.quality_gate` 在固定请求集（`temperature=0`、`seed=1017`）上进行对比校验：

- 对文本、`reasoning_content`、结构化 `tool_calls`、工具续轮、图文和长上下文输出计算 SHA256 摘要。
- 候选与基线的全部 case 均需通过且输出摘要一致，方可纳入默认推荐配置。
- 当服务端返回 `draft_tokens` / `accepted_tokens` 计数时，`scripts/quality_metrics.py` 自动汇总投机解码接受率。

对基线服务和候选服务各运行一次 `scripts.acceptance.core`，再用 `scripts.acceptance.quality_gate` 比较两份结果。`--launch-receipt` 是 `launch.py --start` 写入 `runtime/<时间戳>/launch.json` 的启动回执，脚本据此确认被测服务就是回执里的容器：

```bash
python3 -m scripts.acceptance.core --launch-receipt runtime/<基线时间戳>/launch.json \
  --output runtime/acceptance-baseline.json
python3 -m scripts.acceptance.core --launch-receipt runtime/<候选时间戳>/launch.json \
  --output runtime/acceptance-candidate.json
python3 -m scripts.acceptance.quality_gate --reference runtime/acceptance-baseline.json \
  --candidate runtime/acceptance-candidate.json --output runtime/quality-gate.json
```

`--needle-tokens` 追加一个长上下文大海捞针用例，可选 1024、32768、131072、523000 或 1047000，且不能超过启动时的上下文上限。

## 全模型 PPL

wikitext-2 test 的 token PPL 为 2.772，word PPL 3.168，byte PPL 1.2405，共 134 个 2048 token 窗口，打分 272802 token。数据准备和打分方式见 [基准结果](benchmarks.md#全模型-ppl)。

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
