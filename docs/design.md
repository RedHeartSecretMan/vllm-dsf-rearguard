# 实现设计

本文件描述当前仓库已经做出的改进、验证边界和运行取舍。

## 运行目标

目标硬件是 8×A100-SXM4-80GB / NVSwitch，目标模型固定为 DeepSeek-V4.1-Flash-1017。服务配置同时支持 `tp8 / ep8` 与 `tp4dp2 / ep8`，保留原生图文/视频输入、FP8 MLA KV cache、Engram CPU 查表、前缀缓存和 DSpark k5。配置默认不启用未经目标 GPU 验证的实验开关。

模型目录、官方 revision、每文件 SHA256、运行时 digest 和适配层文件指纹都写入清单。启动前重新读取完整权重哈希；大小、mtime 或旧镜像自报 PASS 都不能替代字节核验。

## 统一适配层

`adaptation/files` 是写入基础 vLLM 镜像的唯一源码树。它把基础运行时集成、1017 模型支持和 SM80 加速实现收敛到同一套最终文件；`adaptation/manifest.json` 记录每个文件的基础哈希、最终哈希和职责归类，`adaptation/install.py` 负责全量校验后原子写入。清单中的职责字段只用于审计和定位，不会再触发多套安装流程。

`config/runtime` 只保存部署环境与服务默认值，`manifests/` 只保存基础镜像、模型 revision 和来源锁定信息。它们不包含源码覆盖，因此不会形成第二个适配入口。

## 1017 协议与多模态

- 规范化 developer、input_text、output_text、图片和工具消息，使服务输入进入统一的 1017 内部格式。
- 保留工具续轮中的 reasoning 内容，支持反序工具结果，并将工具调用与图片收集、tokenizer 使用相同的稳定顺序。
- 跨 user/system/assistant 边界清空旧工具排序键，防止无关轮次的图片或工具结果继承旧顺序。
- 移除非官方图像对齐 token，恢复完整 1024-token 图像预算。
- 视频输入复用图像视觉塔：每个视频均匀抽样最多 8 帧，帧序列按独立图像 span 编码；默认每请求最多 1 个视频，避免把视频帧数和视觉 token 预算交给客户端任意放大。
- 对缓存命中、重复图片、精确图片起点、prefix-cache 尾部和极端长宽比增加调度与几何保护。
- 同步和异步 renderer 共享同一排序逻辑；官方五组输入文件的 40 种模式/effort 组合逐字通过。

## 性能实现

### 编码热路径

长工具历史只计算一次最后用户轮边界，避免每条消息重复倒序扫描。wrapper 已经拥有独立的规范化消息对象，因此编码阶段提供 owned-message 路径，避免二次深拷贝；公共接口仍默认复制输入，保留调用者不可变语义。

目标机 CPU 测量显示：64、256、1024 轮工具历史的 owned-message 路径约提升 1.83–1.92×；结合单次边界扫描后，1024 轮整体 prompt 编码约提升 45–47×。这些数据属于 Python 编码阶段，不代表端到端生成吞吐。

### 长上下文和 decode 候选

- indexer 工作区按真实 chunk builder 的查询/键行上界分配，避免压缩比除法造成合法多请求 chunk 的空间不足。
- decode metadata 候选跳过纯 decode 不使用的压缩序列长度计算，并直接写入已有 validity buffer。
- top-k ragged 候选在 index-source 层构建一次，由同一步消费者共享；缓存键包含输入地址、stride、形状、block table、请求映射、validity 和 batch 维度，避免跨步误复用。
- DSpark stacked KV 候选将三次包含无用 Q 输出的 context 投影合并为一次 KV-only 投影，保留逐层 norm、RoPE、KV group 和 slot mapping。

三个候选都通过 CPU 控制路径和独立安装器检查，默认关闭：

```bash
python3 launch.py "$MODEL" --parallel tp4dp2 --dspark 5 \
  --experimental decode-metadata \
  --experimental ragged-reuse \
  --experimental stacked-kv
```

CPU 测试验证输出、地址、边界和变更输入；原生 MXFP8 loader、Triton kernel、CUDA Graph replay、显存占用和真实 A100 速度仍需 GPU 验收。

## 构建与启动安全

构建脚本只接受干净 Git 提交，使用提交归档构建，并把源码 revision 写入镜像。启动前在无 GPU 的 runc 容器中检查统一适配层指纹，旧镜像不能冒充当前候选。实验变量写入启动回执，模型、端口、镜像 ID 和源码清单可独立回读。

GPU 验收按“数值 / Graph / `ep8` → 功能 smoke → 同负载 A/B → 长上下文和图文压力”的顺序执行。
