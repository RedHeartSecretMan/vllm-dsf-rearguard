# DeepSeek V4.1 Flash A100

面向 `8×A100-SXM4-80GB / NVSwitch` 的 DeepSeek-V4.1-Flash-1017 推理适配，提供统一的 vLLM 适配层、启动配置、验证工具和性能基准。

项目实现了 1017 模型协议、Responses custom-tool 续轮、原生图文输入、实验性视频帧序列扩展、稀疏 MLA、FP8 KV cache、DSpark 投机解码、CUDA Graph 和 EP8 通信路径等功能。适配源码位于 `adaptation/files/vllm/`，由 `adaptation/install.py` 在构建阶段校验后原子安装。

## 快速开始

环境需要 Linux、Python 3.10 及以上、Docker、NVIDIA Container Toolkit 和支持 CUDA 的驱动。先准备好官方 1017 模型目录：

```bash
export MODEL=/data/models/DeepSeek-V4.1-Flash-1017

# 校验模型文件，生成启动所需的校验回执
python3 scripts/model_check.py "$MODEL" --full --output runtime/model-verified.json

# 构建固定 digest 的镜像，并运行 CPU 检查
bash scripts/build.sh

# 按下方“推荐配置”的命令生成启动计划，确认后追加 --start 启动
```

`launch.py` 不带 `--start` 时只打印启动计划。追加 `--start` 后会检查 GPU、已构建的镜像和完整的模型校验回执，全部通过后启动容器，服务绑定 `127.0.0.1:8005`，模型名为 `deepseek-ai/DeepSeek-V4.1-Flash-1017`，接口为 OpenAI 兼容的 `/v1`。

多模态启动默认允许每个请求最多 64 张图片和 2 个视频。视频由 vLLM 按默认 1 FPS、最多 32 帧解码，再复用图像视觉塔；这属于服务侧扩展，不是模型官方的视频编码器。

默认使用 vLLM 原生的 `--disable-chunked-mm-input`，保持视觉 span 原子性。chunked 实验使用 `--no-disable-chunked-mm-input`，可以尝试更小的 `--chunk`。
chunked 实验中，图片 span 仍由项目 scheduler 强制保持原子；只有视频项允许跨 chunk，由 runner 按当前窗口切取对应的视觉 embedding。

## 推荐配置

两套配置都使用 `tp4dp2 / ep8`、DSpark k5、CUDA Graph、OMP 8、EPLB、SM80 custom AG/RS、全局 `max-seqs=256` 和 `max-num-batched-tokens=32768`，不同点是上下文上限：262K 适合高并发，1M 适合容量型请求和长文任务。

262K 配置：

```bash
python3 launch.py "$MODEL" --parallel tp4dp2 --dspark 5 --max-seqs 256 --chunk 32768 \
  --context 262144 --experimental stacked-kv
```

1M 配置：

```bash
python3 launch.py "$MODEL" --parallel tp4dp2 --dspark 5 --max-seqs 256 --chunk 32768 \
  --context 1048576 --experimental stacked-kv --experimental bounded-prefill
```

实验开关的作用与状态见 [项目架构](docs/architecture.md)。

## 性能与精度

`v0.1.0` 在 8 张 A100 上完成了文本、工具调用、图文、长上下文和混合负载验收。

| 场景 | 结果 |
| --- | ---: |
| 262K 配置，C256 短请求输出 token 吞吐 | `7270 token/s` |
| 1M 配置，C256、ISL 1024 冷请求输出 token 吞吐 | `1369 token/s` |
| 262K 配置，260K 单请求 Prefill TPS | `8498 token/s` |
| 1M 配置，约 1M 单请求 Prefill TPS | `4772 token/s` |
| 单请求 Decode TPS | `279.5 token/s` |
| GPQA Diamond，4 轮平均 | `91.54%` |

测试配置、指标口径和完整数据见 [基准结果](docs/benchmarks.md)，验证范围见 [验证记录](docs/validation.md)，版本内容见 [发布说明](docs/release.md)。Responses 协议 smoke 可运行 `python3 scripts/responses_smoke.py --help` 查看入口。

## 本地检查

需要 CPU 版 PyTorch，和 CI 的检查一致：

```bash
ruff check .
python3 -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES= python3 -m scripts.hotpaths.check
```

`scripts/build.sh` 在无 GPU 的 `runc` 容器中执行安装、源码清单和 CPU 合约检查。

## 项目结构

```text
adaptation/   vLLM 源码覆盖、编码器与图像处理器参考实现、安装清单
config/       运行时环境变量与四类 benchmark 配置
docs/         架构、基准、验证和发布说明，以及原始回执
manifests/    模型文件清单与基础镜像锁定
scripts/      构建、校验、验收、基准、PPL 工具和 CPU 热路径检查（hotpaths/）
tests/        CPU 合约、协议、视觉和 benchmark 测试
launch.py     启动计划生成与服务启动入口
ruff.toml     代码检查配置
```

## 许可证

项目源码遵循 [Apache-2.0](LICENSE)。许可与第三方来源见 [NOTICE](NOTICE)。
