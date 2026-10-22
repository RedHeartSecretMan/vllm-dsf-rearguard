# DeepSeek V4.1 Flash A100

面向 `8×A100-SXM4-80GB / NVSwitch` 的 DeepSeek-V4.1-Flash-1017 推理适配，提供统一的 vLLM 适配层、启动配置、验证工具和性能基准。

项目把 1017 模型协议、原生图文输入、稀疏 MLA、FP8 KV cache、DSpark 投机解码、CUDA Graph 和 EP8 通信路径收敛到同一套安装清单。适配源码位于 `adaptation/files/vllm/`，由 `adaptation/install.py` 在构建阶段校验后原子安装。

## 快速开始

环境需要 Linux、Python 3.10 及以上、Docker、NVIDIA Container Toolkit 和支持 CUDA 的驱动。先准备好官方 1017 模型目录：

```bash
export MODEL=/data/models/DeepSeek-V4.1-Flash-1017

# 校验模型文件，生成启动所需的校验回执
python3 scripts/model_check.py "$MODEL" --full --output runtime/model-verified.json

# 构建固定 digest 的镜像，并运行 CPU 检查
bash scripts/build.sh

# 生成启动计划；确认后追加 --start 启动
python3 launch.py "$MODEL" --parallel tp4dp2 --dspark 5 --max-seqs 256 \
  --experimental stacked-kv
```

`--start` 要求 8 张空闲 A100、已构建的镜像和完整的模型校验回执。服务绑定 `127.0.0.1:8005`，模型名为 `deepseek-ai/DeepSeek-V4.1-Flash-1017`。

## 推荐配置

262K 上下文的默认起点是 `tp4dp2 / ep8`、DSpark k5、`stacked-kv`、CUDA Graph、OMP 8、EPLB、SM80 custom AG/RS、全局 `max-seqs=256` 和 `max-num-batched-tokens=16384`。

1M 上下文服务在此基础上增加 `--context 1048576 --experimental bounded-prefill`。262K 适合高并发，1M 适合容量型请求和长文任务。

```bash
python3 launch.py "$MODEL" --parallel tp4dp2 --dspark 5 --max-seqs 256 \
  --context 1048576 --experimental stacked-kv --experimental bounded-prefill
```

实验开关的作用与状态见 [项目架构](docs/architecture.md)。

## 性能与精度

`v0.1.0` 在 8 张 A100 上完成了文本、工具调用、图文、长上下文和混合负载验收。

| 场景 | 结果 |
| --- | ---: |
| 262K 上限，C256 短请求 | `7270 token/s` |
| 1M 上限，C256 bounded profile | `1369 token/s` |
| 262K 上限，260K 单请求 prefill | `8498 token/s` |
| 1M 上限，约 1M 单请求 prefill | `4772 token/s` |
| 单请求 decode | `279.5 token/s` |
| GPQA Diamond，4 轮平均 | `91.54%` |

测试配置、指标口径和完整数据见 [基准结果](docs/benchmarks.md)，验证范围见 [验证记录](docs/validation.md)，版本内容见 [发布说明](docs/release.md)。

## 本地检查

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m compileall -q launch.py scripts adaptation config tests
```

`scripts/build.sh` 在无 GPU 的 `runc` 容器中执行安装、源码清单和 CPU 合约检查。

## 项目结构

```text
adaptation/   vLLM 源码覆盖、官方参考实现和安装清单
config/       运行时环境变量与 benchmark 场景配置
docs/         架构、基准、验证和发布说明
manifests/    模型文件清单与基础镜像锁定
scripts/      构建、校验、验收和性能工具
tests/        CPU 合约、协议、视觉和 benchmark 测试
launch.py     启动计划生成与服务启动入口
```

## 许可证

项目源码遵循 [Apache-2.0](LICENSE)。DeepSeek 官方参考文件遵循 MIT 许可，许可文本与第三方来源见 [NOTICE](NOTICE)。
