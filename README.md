# DeepSeek-V4.1-Flash-1017 · 8×A100

面向 8×A100-SXM4-80GB / NVSwitch，提供 DeepSeek-V4.1-Flash-1017 的适配源码、构建配方和验证工具。

当前实现聚焦文本服务、多用户并发、长上下文和原生图文输入，支持 `tp8 / ep8` 与 `tp4dp2 / ep8` 两种部署方式，并围绕 DSpark、CUDA Graph、稀疏 MLA 和 FP8 KV cache 提供可组合配置。视频是基于官方图像路径的实验性帧序列扩展。

## 主要改进

- 适配 SM80 稀疏 MLA、candidate MQA、FP4 Marlin、dense BF16 dispatch、DSpark/EPLB 和 `ep8` 通信路径。
- 按 1017 官方编码协议处理工具续轮思考历史、工具结果顺序、图片关联和同步/异步 renderer。
- 恢复官方图像 token 布局和 1024-token 图像预算，完善缓存图片、重复图片、图片起点和长宽比边界。
- 实验性视频扩展沿用图像视觉塔：每个请求最多 2 个视频，最多 64 张图片；视频默认按 1 FPS 抽样，最多 32 帧，每帧沿用 1024-token 图像预算；模型本身没有官方 temporal encoder。
- 优化长上下文 indexer 工作区、decode metadata、top-k 共享和 DSpark context KV 投影。
- 将长工具历史的用户轮边界扫描改为单次计算；wrapper 复用已拥有的消息对象，减少重复遍历和深拷贝。
- 提供 `decode-metadata`、`ragged-reuse`、`stacked-kv` 和 `bounded-prefill` 四个独立实验开关，默认关闭，可单独进行 A/B 测试。

长工具历史的 CPU 编码已经在目标机完成对照：1024 轮工具历史约从 1.34 秒降至 28–30 ms；owned-message 路径额外约提升 1.83–1.92×。这些数字描述 prompt 编码阶段，完整服务性能需要在目标 GPU 上实测。

详细实现和实验记录见 [实现设计](docs/design.md)、[优化记录](docs/optimization-progress.md) 和 [验证记录](docs/validation.md)。

## 环境准备

目标环境需要 Linux、Python ≥3.10、Docker、NVIDIA Container Toolkit 和支持 CUDA 的 NVIDIA 驱动。准备好官方 1017 模型目录后，将路径放入 `MODEL` 环境变量。

```bash
export MODEL=/data/models/DeepSeek-V4.1-Flash-1017

python3 -m unittest discover -s tests -v
bash scripts/build.sh
python3 scripts/model_check.py "$MODEL" --full \
  --output runtime/model-verified.json
```

## 启动服务

先生成配置计划，确认参数后再启动：

```bash
# tp4dp2 / ep8、DSpark k5、原生图文输入（视频扩展可选）
python3 launch.py "$MODEL" \
  --parallel tp4dp2 \
  --dspark 5 \
  --max-seqs 256

# 文本基线：tp8 / ep8、关闭 DSpark、eager 模式
python3 launch.py "$MODEL" \
  --parallel tp8 \
  --dspark 0 \
  --kernels control \
  --eager
```

确认模型目录、镜像和 GPU 状态后，使用 `--start` 启动服务。API 默认绑定 `127.0.0.1:8005`，模型名为 `deepseek-ai/DeepSeek-V4.1-Flash-1017`。

常用配置如下：

| 维度 | 可选值 |
|---|---|
| 并行 | `tp8 / ep8`、`tp4dp2 / ep8` |
| 投机 | `--dspark 0`、`--dspark 5` |
| 输入 | 原生图文；视频为实验性帧序列扩展，每请求最多 64 张图片或 2 个视频（默认 1 FPS、最多 32 帧）；`--text-only` 运行文本对照 |
| 上下文 | `262144`、`524288`、`1048576` |
| 内核 | `control`、`optimized` |
| 通信 | NCCL 自动选择、`--nccl ring` |

1017 的 DSpark 训练块大小为 5，因此当前服务配置使用 k5。

视频扩展沿用 OpenAI 多模态消息格式，在 `content` 中传入一个
`{"type":"video_url","video_url":{"url":"..."}}` 块。服务会先由 vLLM
解码，再按时长均匀抽取帧，默认 1 FPS、最多 32 帧，送入现有图像视觉塔；超过限制的帧不会继续增加视觉
token 预算。这是服务侧扩展，不是 DeepSeek-V4.1 官方视频输入契约。

1M 上下文的高并发 profile 可额外启用 `--experimental bounded-prefill`。它将稀疏 indexer 的预留工作区收敛到单请求可容纳的压缩上下文，保留 1M 单请求能力，减少高并发时的静态预留。

## 性能实验

先运行功能 smoke，再运行性能测试：

```bash
python3 -m scripts.acceptance \
  --launch-receipt "$LAUNCH" \
  --needle-tokens 32768 \
  --output runtime/acceptance-32k.json

python3 -m scripts.benchmark \
  --launch-receipt "$LAUNCH" \
  --input-tokens 1024 \
  --concurrency 32 \
  --cache cold \
  --output runtime/c32-i1k-cold.json
```

实验会记录服务端 token usage、输出吞吐、P95 TTFT、TPOT、失败率和冷热缓存状态。测试矩阵与指标定义见 [实验协议](docs/experiments.md)。

目标机首轮 GPU 验收已确认 `tp4dp2 / ep8 + DSpark k5 + stacked-kv` 在全局 `max-seqs=256` 下稳定运行；短输出合成请求的 C128 约 3928 token/s，C256 约 7270 token/s，详情见 [验证记录](docs/validation.md)。

三个实验热路径可以组合启用：

```bash
python3 launch.py "$MODEL" \
  --parallel tp4dp2 \
  --dspark 5 \
  --context 1048576 \
  --max-seqs 256 \
  --experimental bounded-prefill \
  --experimental decode-metadata \
  --experimental ragged-reuse \
  --experimental stacked-kv
```

## 许可证

本项目遵循 vLLM 的 [Apache-2.0](LICENSE) 许可证，模型和官方编码器采用 [MIT](licenses/DeepSeek-LICENSE)，第三方许可见 [NOTICE](NOTICE)。
