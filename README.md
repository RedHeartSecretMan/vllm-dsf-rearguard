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

`launch.py` 不带 `--start` 时只打印启动计划。追加 `--start` 后会检查 GPU、已构建的镜像和完整的模型校验回执，全部通过后启动容器，服务默认绑定 `127.0.0.1:6789`，可通过 `--port` 修改端口。模型名为 `deepseek-ai/DeepSeek-V4.1-Flash-1017`，接口为 OpenAI 兼容的 `/v1`。

多模态启动默认允许每个请求最多 64 张图片和 2 个视频。视频由 pinned vLLM media IO 按默认 1 FPS 和最多 32 帧计算均匀采样点，再交给 codec/backend 读取；processor 只接收这些选中帧并复用图像视觉塔。这属于服务侧扩展，不是模型官方的视频编码器。

每个视频会生成稳定的 `vid_...` 身份，并在普通文本 wrapper 中记录 media IO 返回的逐帧采样时间和一个视频占位符；占位符随后展开为同顺序的图片帧 span。多轮历史重放相同媒体引用及参数时会得到相同身份，单独发送身份文本不会恢复缺失的媒体内容。

默认使用 vLLM 原生的 `--disable-chunked-mm-input`，使单个多模态 item 在一个 prefill 窗口中处理。通过 `--no-disable-chunked-mm-input` 开启分块后，图片 span 仍保持完整，视频可以在完整帧之间跨窗口，由 runner 切取对应的视觉 embedding。

## 推荐配置

默认关闭多模态分块的两套视觉配置都使用 `tp4dp2 / ep8`、DSpark k5、CUDA Graph、OMP 8、EPLB、SM80 custom AG/RS、全局 `max-seqs=256` 和 `max-num-batched-tokens=32768`，以容纳 32 帧视频的最大视觉 item；不同点是上下文上限：262K 适合高并发，1M 适合容量型请求和长文任务。

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

当前 1M 视觉服务采用 16K 分块，以增加 KV 容量；启动命令如下。相对开启分块的 32K，视频 C4 吞吐提高约 4.5%，260K 冷 prefill 吞吐降低约 3.4%，完整对照见 [16K / 32K 测试](docs/chunk-budget.md)。

```bash
python3 launch.py "$MODEL" --parallel tp4dp2 --dspark 5 --max-seqs 256 --chunk 16384 \
  --context 1048576 --experimental stacked-kv --experimental bounded-prefill \
  --no-disable-chunked-mm-input --responses-store --port 6789
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

测试配置、指标口径和完整数据见 [基准结果](docs/benchmarks.md)，验证范围见 [验证记录](docs/validation.md)，版本内容见 [发布说明](docs/release.md)。四类 GPU 验收入口位于 `config/acceptance/` 和 `scripts/acceptance/`；Responses smoke 可运行 `python3 -m scripts.acceptance.responses --list` 查看入口。

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
config/       运行时环境变量、四类 benchmark 配置和四类 GPU 验收配置
docs/         架构、基准、验证和发布说明，以及原始回执
manifests/    模型文件清单与基础镜像锁定
scripts/      构建、校验、验收、基准、PPL 工具和 CPU 热路径检查（按 acceptance/、benchmarks/、hotpaths/ 分类）
tests/        CPU 合约、协议、视觉和 benchmark 测试
launch.py     启动计划生成与服务启动入口
ruff.toml     代码检查配置
```

## 许可证

项目源码遵循 [Apache-2.0](LICENSE)。许可与第三方来源见 [NOTICE](NOTICE)。
