# DeepSeek V4.1 Flash A100

面向 `8×A100-SXM4-80GB / NVSwitch` 的 DeepSeek V4.1 Flash 推理适配，提供统一的 vLLM 适配层、启动配置、验证工具和性能基准。

项目把模型协议、原生图文输入、稀疏 MLA、FP8 KV cache、DSpark 投机解码、CUDA Graph 和 EP8 通信路径收敛到同一套安装清单中。适配源码全部写入 `adaptation/files/vllm/`，由 `adaptation/install.py` 在构建阶段完成校验和原子安装。

## 推荐配置

文本和图文服务的默认起点是 `tp4dp2 / ep8`、DSpark k5、CUDA Graph、OMP 8、EPLB、SM80 custom AG/RS、全局 `max-seqs=256` 和 `max-num-batched-tokens=16384`。

```bash
export MODEL=/data/models/DeepSeek-V4.1-Flash-1017

python3 launch.py "$MODEL" \
  --parallel tp4dp2 \
  --dspark 5 \
  --max-seqs 256 \
  --start
```

1M 上下文服务在同一配置上增加 `--context 1048576` 和 `--experimental bounded-prefill`。262K 上下文更适合高并发，1M 上下文用于容量型请求和长文任务。

## 性能结果

`v0.1.0` 在 8 张 A100 上完成了文本、工具调用、图文、长上下文和混合负载验收。

| 场景 | 结果 |
| --- | ---: |
| 262K 上限、C256 固定短请求 | 约 `7270 token/s` |
| 1M 上限、C256 bounded profile | 约 `1369 token/s` |
| 262K 单请求 prefill | 约 `8498 token/s` |
| 1M 单请求 prefill | 约 `4772 token/s` |
| 单请求 decode | 约 `279 token/s` |
| GPQA Diamond，4 轮平均 | `91.54%` |

完整 case、指标口径和已验证边界见 [基准结果](docs/benchmarks.md) 与 [验证记录](docs/validation.md)。

## 本地检查

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m compileall -q launch.py scripts adaptation config tests
bash scripts/build.sh
```

`scripts/build.sh` 使用固定 digest 基础镜像构建，并在无 GPU 的 `runc` 容器中执行安装、源码清单和 CPU 合约检查。

## 项目结构

```text
adaptation/   vLLM 源码覆盖、官方参考实现和安装清单
config/       运行时与 benchmark 配置
docs/         架构、基准、验证和发布说明
scripts/      启动、检查、验收和性能工具
tests/        CPU 合约、协议、视觉和 benchmark 测试
```

## 许可证

项目源码遵循 [Apache-2.0](LICENSE)，DeepSeek 官方参考文件遵循 [MIT](licenses/DeepSeek-LICENSE)，第三方来源见 [NOTICE](NOTICE)。
