# 架构说明

本文介绍项目的运行目标、适配层结构、协议与多模态处理、关键实现，以及构建与启动机制。

## 运行目标

目标硬件是 `8×A100-SXM4-80GB / NVSwitch`，目标模型固定为 DeepSeek-V4.1-Flash-1017。服务支持 `TP4 / DP2 / EP8` 和 `TP8 / EP8` 两种并行方式，保留原生图文输入、FP8 DS MLA KV cache、Engram CPU 查表、前缀缓存和 DSpark k5。实验开关默认关闭，需要按负载显式启用。

模型 revision、每个模型文件的 SHA256、基础镜像 digest 和适配层文件指纹都写入清单。`launch.py --start` 启动前重新计算每个模型文件的 SHA256，并与清单和校验结果逐项比对。

## 适配层结构

`adaptation/files` 是写入基础 vLLM 镜像的唯一源码树，包含基础运行时集成、1017 模型支持和 SM80 加速实现。`adaptation/manifest.json` 记录每个文件的基础哈希、最终哈希和用途，便于审计和定位；`adaptation/install.py` 在全量校验后原子写入。

`config/runtime/environment.json` 是 `launch.py` 注入容器的环境变量基线。`manifests/model-files.json` 记录模型文件的大小和 SHA256，`manifests/sources.json` 锁定基础镜像的 digest 和源码 commit。

## 协议与多模态

- 规范化 developer、input_text、output_text、图片和工具消息，使服务输入进入统一的 1017 内部格式。
- 保留工具续轮中的 reasoning 内容，支持反序工具结果，并将工具调用与图片收集、tokenizer 使用相同的稳定顺序。
- 跨 user/system/assistant 边界清空旧工具排序键，防止无关轮次的图片或工具结果继承旧顺序。
- 移除非官方图像对齐 token，恢复完整的 1,024 token 图像预算。
- 视频占位符保留一个稳定的 `vid_<hash>` 身份和全局文本 wrapper；processor 再将其展开为按时间顺序排列的图片帧 span。每帧继续使用图片局部 span，scheduler 只在完整帧之间允许 chunk。
- 视频采样在固定版本的 vLLM media IO 中完成：按请求的 `1 FPS` 和 `num_frames=32` 计算目标帧索引，长视频在全视频范围内均匀选择最多 32 个时间点；processor 拒绝超过 32 帧的 media backend 输出。帧索引限制了送入 processor 的帧数，具体 codec 是否仍需顺序扫描由 backend 决定。
- renderer 将 media IO 返回的实际采样时间按帧序号写入视频 wrapper；时间清单与后续连续图片 span 一一对应，chunk 只切视觉帧边界，不改变映射顺序。
- 多轮请求重放相同媒体时 wrapper 身份保持稳定，并写入 vLLM 视频 content-part 的 `uuid` 以复用多模态 encoder/prefix cache；缺少原始媒体或服务端视觉/KV 缓存时，单独发送身份文本仍不能恢复视频。
- 对缓存命中、重复图片、精确图片起点、prefix-cache 尾部和极端长宽比增加调度与几何保护。
- 同步和异步 renderer 共享同一排序逻辑；官方 5 组输入文件的 40 种思考模式与 `reasoning_effort` 组合均与官方编码器逐字一致。

## 关键实现

### 编码热路径

长工具历史只计算一次最后用户轮边界，避免每条消息重复倒序扫描。wrapper 已经拥有独立的规范化消息对象，因此编码阶段提供 owned-message 路径，避免二次深拷贝；公共接口仍默认复制输入，保留调用者不可变语义。

目标机 CPU 测量使用 Python 3.10，每档交错运行三次取中位数，输出均与官方编码器逐字一致：

| 工具轮数 | chat：原 / 新 | thinking：原 / 新 | 加速比 |
|---:|---:|---:|---:|
| 64 | 7.115 / 1.876 ms | 7.102 / 1.922 ms | 3.7× |
| 256 | 87.919 / 7.333 ms | 87.604 / 7.312 ms | 12.0× |
| 1024 | 1341.594 / 28.453 ms | 1347.609 / 29.625 ms | 45.5–47.2× |

结果仅覆盖 Python prompt 编码阶段，原始数据见 [encoding-benchmark.json](evidence/encoding-benchmark.json)。复现时用 `scripts/benchmarks/bench_encoding.py` 对比两份编码器：`--baseline` 指向优化前的编码器文件，`--candidate` 默认是适配层中的 `deepseek_v41_encoding.py`，每组结果都会与 `adaptation/reference/encoder.py` 逐字比对。

```bash
python3 scripts/benchmarks/bench_encoding.py --baseline /path/to/baseline_encoder.py \
  --output runtime/encoding-benchmark.json
```

### 长上下文与 decode 优化

下表列出各优化的实现和启用方式。实验开关默认关闭，通过 `launch.py --experimental <名称>` 显式启用，可重复传入以组合。各开关的对照结果和使用建议见[性能与调优](performance.md)。

| 开关 | 实现 |
|---|---|
| `bounded-prefill` | indexer 工作区按单个完整上下文请求的压缩行数预留，更大的多请求 prefill 由 metadata builder 拆成多个 chunk；1M 单请求容量保持不变。 |
| `prefill-throttle` | 长 prefill（默认 4096 token 及以上）在有 decode 请求运行时每 4 个 scheduler step 释放一次，短请求照常调度。它只调整调度时序，KV 格式和 attention kernel 保持原样。 |
| `stacked-kv` | DSpark 将三次包含无用 Q 输出的 context 投影合并为一次只计算 KV 的投影，保留逐层 norm、RoPE、KV group 和 slot mapping。开启后会增加一份合并后的投影权重。 |
| `decode-metadata` | 纯 decode 跳过未使用的压缩序列长度计算，并直接写入已有的 validity buffer。 |
| `ragged-reuse` | top-k ragged 在 index-source 层构建一次并由同一步的消费者共享；缓存键包含输入地址、stride、形状、block table、请求映射、validity 和 batch 维度。 |

另有两个未进入默认配置的实验入口：`--logits-row-chunk` 分块释放 SM80 MQA logits（默认 `0`，即不分块），环境变量 `VLLM_SM80_CANDIDATE_MQA_COMPACT=1` 启用紧凑 candidate-MQA top-k。两者的测试结果见[未采用方案](performance.md#未采用方案)。

indexer 工作区容量按真实 chunk builder 的查询/键行上界计算，避免压缩比除法造成合法多请求 chunk 的空间不足。

```bash
python3 launch.py "$MODEL" --parallel tp4dp2 --dspark 5 \
  --experimental stacked-kv \
  --experimental decode-metadata
```

`launch.py --kv-transfer-config` 接收 JSON 字符串或 `@文件路径`，校验 connector 与 `kv_role` 后传入 vLLM。两个 CPU offload connector 的测试结果见[未采用方案](performance.md#未采用方案)。

`scripts/hotpaths/check.py` 用 CPU 张量执行适配层中的 decode 元数据、ragged 复用、空 ragged pack 和 context 投影方法，需要 PyTorch，并要求 `CUDA_VISIBLE_DEVICES` 为空。加 `--native-loader` 会用镜像中的 vLLM 参数类检查 MXFP8 权重加载，只能在镜像内运行。CI 和镜像构建都会执行该检查。

```bash
CUDA_VISIBLE_DEVICES= python3 -m scripts.hotpaths.check --output runtime/hotpaths-cpu.json
```

各开关的 GPU 启动、性能与精度结果分别见[性能与调优](performance.md)和[验证结果](validation.md)。

### 显存预算与 Attention 工作区

镜像和启动配置默认开启自动显存预算。预算会计入模型前向、初始化和图捕获占用，并单独核算 KV 本体与 Attention 工作区。完整图捕获和大块 prefill 都纳入启动准入；容量不足以容纳最大上下文时，启动检查会明确拒绝。

大块 prefill 会延后 Attention 输出分配，减少临时缓冲区的重叠存活期；较小的 decode 和捕获形状保持原分配顺序。稀疏 prefill 在布局兼容时直接写入已有输出缓冲，避免额外临时结果；不兼容的布局继续使用兼容路径。

## 构建与启动

构建脚本要求工作区没有未提交的改动，使用 `git archive` 导出当前提交作为构建上下文，并把源码 revision 写入镜像标签。启动前会在不分配 GPU 的容器中检查适配层指纹，镜像与当前源码不一致时拒绝启动。启动后，模型、端口、镜像、实验开关和源码清单写入 `runtime/` 下的启动记录，可独立核对。

GPU 验收依次检查数值与图捕获、基础功能、同负载性能对照、长上下文和图文压力，结果见[验证结果](validation.md)。
