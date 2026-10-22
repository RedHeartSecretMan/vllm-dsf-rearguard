# 项目架构

本文件描述适配层、运行配置和验证边界。

## 运行目标

目标硬件是 8×A100-SXM4-80GB / NVSwitch，目标模型固定为 DeepSeek-V4.1-Flash-1017。服务配置同时支持 `tp8 / ep8` 与 `tp4dp2 / ep8`，保留原生图文输入、FP8 MLA KV cache、Engram CPU 查表、前缀缓存和 DSpark k5。配置默认不启用未经目标 GPU 验证的实验开关。

模型目录、官方 revision、每文件 SHA256、运行时 digest 和适配层文件指纹都写入清单。启动前重新读取完整权重哈希；大小、mtime 或旧镜像自报 PASS 都不能替代字节核验。

## 统一适配层

`adaptation/files` 是写入基础 vLLM 镜像的唯一源码树。它把基础运行时集成、1017 模型支持和 SM80 加速实现收敛到同一套最终文件；`adaptation/manifest.json` 记录每个文件的基础哈希、最终哈希和职责归类，`adaptation/install.py` 负责全量校验后原子写入。清单中的职责字段只用于审计和定位，不会再触发多套安装流程。

`config/runtime` 只保存部署环境与服务默认值，`manifests/` 只保存模型文件清单和基础镜像的 digest。它们不包含源码覆盖，因此不会形成第二个适配入口。

## 1017 协议与多模态

- 规范化 developer、input_text、output_text、图片和工具消息，使服务输入进入统一的 1017 内部格式。
- 保留工具续轮中的 reasoning 内容，支持反序工具结果，并将工具调用与图片收集、tokenizer 使用相同的稳定顺序。
- 跨 user/system/assistant 边界清空旧工具排序键，防止无关轮次的图片或工具结果继承旧顺序。
- 移除非官方图像对齐 token，恢复完整 1024-token 图像预算。
- 对缓存命中、重复图片、精确图片起点、prefix-cache 尾部和极端长宽比增加调度与几何保护。
- 同步和异步 renderer 共享同一排序逻辑；官方五组输入文件的 40 种模式/effort 组合逐字通过。

## 性能实现

### 编码热路径

长工具历史只计算一次最后用户轮边界，避免每条消息重复倒序扫描。wrapper 已经拥有独立的规范化消息对象，因此编码阶段提供 owned-message 路径，避免二次深拷贝；公共接口仍默认复制输入，保留调用者不可变语义。

目标机 CPU 测量（Python 3.10，每档三次交错取中位数，输出均与官方编码器逐字一致）：

| 工具轮数 | chat：原 / 新 | thinking：原 / 新 | 加速比 |
|---:|---:|---:|---:|
| 64 | 7.115 / 1.876 ms | 7.102 / 1.922 ms | 3.7× |
| 256 | 87.919 / 7.333 ms | 87.604 / 7.312 ms | 12.0× |
| 1024 | 1341.594 / 28.453 ms | 1347.609 / 29.625 ms | 45.5–47.2× |

这是 Python prompt 编码阶段的结果，不包含 tokenizer、模型和端到端服务吞吐，原始采样见 [目标机回执](evidence/encoding-benchmark-target.json)。

### 长上下文与 decode 优化

下表列出各优化的实现和启用方式。实验开关默认关闭，通过 `launch.py --experimental <名称>` 显式启用，可重复传入以组合。各开关的 A/B 结果和采用结论见 [基准结果](benchmarks.md)。

| 开关 | 实现 |
|---|---|
| `bounded-prefill` | indexer 工作区按“单个完整上下文请求的压缩行数”预留，更大的多请求 prefill 由 metadata builder 拆成多个 chunk，不降低 1M 单请求容量。 |
| `prefill-throttle` | 长 prefill（默认 4096 token 及以上）在有 decode 请求运行时每 4 个 scheduler step 释放一次，短请求不受影响。只改变调度时序，不改变 KV 格式和 attention kernel。 |
| `stacked-kv` | DSpark 将三次包含无用 Q 输出的 context 投影合并为一次 KV-only 投影，保留逐层 norm、RoPE、KV group 和 slot mapping。开启后会增加权重副本。 |
| `decode-metadata` | 纯 decode 跳过未使用的压缩序列长度计算，并直接写入已有的 validity buffer。 |
| `ragged-reuse` | top-k ragged 在 index-source 层构建一次并由同一步的消费者共享；缓存键包含输入地址、stride、形状、block table、请求映射、validity 和 batch 维度。 |

另有两个没有采用的实验入口：`--logits-row-chunk` 分块释放 SM80 MQA logits（默认 `0`），环境变量 `VLLM_SM80_CANDIDATE_MQA_COMPACT=1` 启用紧凑 candidate-MQA top-k。

indexer 工作区容量按真实 chunk builder 的查询/键行上界计算，避免压缩比除法造成合法多请求 chunk 的空间不足。

```bash
python3 launch.py "$MODEL" --parallel tp4dp2 --dspark 5 \
  --experimental stacked-kv \
  --experimental decode-metadata
```

CPU 测试验证输出、地址、边界和变更输入，回执见 [CPU 热路径回执](evidence/hotpaths-cpu.json)。各开关的 GPU 启动、性能与精度结果分别见 [基准结果](benchmarks.md) 和 [验证记录](validation.md)。

## 构建与启动安全

构建脚本只接受干净 Git 提交，使用提交归档构建，并把源码 revision 写入镜像。启动前在无 GPU 的 runc 容器中检查统一适配层指纹，旧镜像不能冒充当前候选。实验变量写入启动回执，模型、端口、镜像 ID 和源码清单可独立回读。

GPU 验收按“数值 / Graph / `ep8` → 功能 smoke → 同负载 A/B → 长上下文和图文压力”的顺序执行。
