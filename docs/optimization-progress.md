# 下载期间优化记录

2026-10-20。继续保持原 dsv4 服务运行，以下工作没有执行 GPU 推理。

## 已验证的 CPU 编码优化

长工具历史中，纯工具结果不建立新的用户轮。原编码器为每条消息重新倒序扫描
最后一个用户轮，造成 O(n²) 的边界查找。现在在规范化/裁剪完成后计算一次，
渲染共享该结果，并一次性拼接字符串。独立调用 `render_message` 仍保留原行为。
没有跨请求缓存或共享可变状态。

实际 wrapper 在规范化阶段已经拥有独立消息对象；编码器新增 owned-message 路径，避免再次深拷贝工具历史。公共 `encode_messages` 默认仍复制输入。CPU 对照显示该路径约 1.85–2.09x，输出保持一致。

目标服务器 `/usr/bin/python3` 3.10 上，每档三次交错 A/B 的中位数：

| 合成工具轮数 | chat：原 / 新，ms | thinking：原 / 新，ms |
|---|---:|---:|
| 64 | 7.12 / 1.88 | 7.10 / 1.92 |
| 256 | 87.92 / 7.33 | 87.60 / 7.31 |
| 1024 | 1341.59 / 28.45 | 1347.61 / 29.62 |

六档每次输出均逐字匹配固定官方编码器。这是 **Python prompt 编码阶段** 的改善，
没有测 tokenizer、模型 decode、整机吞吐或在线 TTFT。原始采样、环境及前后源码 SHA
在 [目标机回执](evidence/encoding-benchmark-target.json)。

回归另外覆盖官方随模型提供的五个输入文件、两种思考模式、四种 effort，
共 40 种组合。多模态样例经过实际 tokenizer wrapper 的规范化函数后再编码；
直接把原始 image_url 喂给内部文本编码器不是服务的实际调用方式。
样例文件逐个与官方模型清单 SHA256 核对，遵循 DeepSeek MIT 许可。

## 三个待 GPU 验收的独立候选

它们安装到镜像，但默认关闭。通过重复 `--experimental` 参数组合，回执记录所选项。

| 参数 | 实现 | 当前证据 |
|---|---|---|
| `decode-metadata` | decode 不计算未使用的压缩序列长度；比较结果直接写入已有 SWA validity buffer | 实际 builder 前缀的 CPU ATen 记录：ratio>1 的纯 decode 少一次 floor_divide，validity 更新少一次 copy_，输出及存储地址一致 |
| `ragged-reuse` | 同一 index-source 层的消费者复用本步打包结果；生产者每步清空缓存 | 真实 helper 的 CPU 控制测试：五层 fixture 的 pack 从五次降到一次；变化的 batch、block table、压缩比和 top-k view 不误复用 |
| `stacked-kv` | 三次含无用 Q 输出的 context 投影改为一次 KV-only 投影；使用原生 merged-linear shard loader | CPU 方法测试：三次 linear 调用变为一次，逐层 norm、位置和不同 KV group 的 slot 保持一致；原生 MXFP8 loader 仍待固定镜像验证 |

```bash
python3 launch.py "$MODEL" --parallel tp4dp2 --dspark 5 \
  --experimental decode-metadata \
  --experimental ragged-reuse \
  --experimental stacked-kv
```

固定镜像准备前，目标机 Python 3.10 使用 PyTorch 2.6.0+cu124，运行时显式隐藏 GPU；
该回执只代表宿主 CPU 控制路径，正式固定镜像构建阶段使用 `runc` 和无网络配置，**34 个张量/控制流场景通过**；
[回执](evidence/hotpaths-cpu.json) 中 `native_loader=NOT_RUN`、`gpu_validation=NOT_RUN`。
测试执行实际候选方法；ragged pack 数学使用独立 CPU 参考，不代表 Triton kernel
或 CUDA Graph replay 通过。小型 FP32 投影对照不能代替原生 MXFP8/Marlin 数值验证。
固定镜像准备前的宿主环境缺少 `ModelOptLinearMethod`，因此显式 `--native-loader`
检查被旧依赖版本拒绝；该历史失败不计入最终结果，固定 digest 构建阶段的 native loader 已通过。

固定镜像的构建新增原生 MXFP8 权重/scale shard 检查，并拒绝缺失 shard。
现有旧镜像没有该版 `ModelOptLinearMethod`，因此没有用旧接口冒充固定运行时验收。
stacked KV 会增加权重副本；dense BF16 物化还可能追加 BF16 缓冲，实际显存必须实测。

## 实现边界

当前优化围绕四条路径展开：1017 协议和图像输入、长工具历史编码、长上下文 indexer 和 decode metadata、DSpark context KV 与 top-k 生命周期。新路径通过独立环境变量控制，默认关闭，避免把未经 A100 实测的候选写入默认性能结论。

保留已有的融合 prefill、负位置和 tail guards，不覆盖当前运行时的 decoder ABI。所有候选都记录源码清单、输入变化和 CPU 对照；没有 GPU 数值、CUDA Graph、显存或整机吞吐证据时，不宣称生产提速。

## 后续优先级

1. 完成自然任务质量回归：文本、工具调用、代码、视觉和 reasoning effort 使用固定请求集做配对对照，记录题级结果和允许波动。
2. 完成 256K / 1M 多请求长文 soak，分别覆盖冷缓存、热缓存、混合 prefill/decode、工具续轮和图文输入；当前 260K 与 1M 结果是单请求容量证明。
3. 在已完成的 C32/C64/C128/C256 短请求矩阵之上，补充 1024 输出、冷/热缓存、混合 prefill/decode、工具图文混合和生产 soak，统一报告 P95 TTFT、TPOT、失败率和显存水位。
4. 仅根据上述回归中出现的瓶颈继续调优；已验证候选保持默认开启，未验证或负收益候选继续保持关闭。

构建脚本现在只构建干净 Git 提交的归档，并写入 image revision 标签，
避免后台构建途中源码变化却把旧镜像当作新提交。启动还将实际安装源码指纹
与本地统一适配层清单绑定，旧镜像即便自己的检查返回 PASS 也不能冒充当前候选。

目标服务器 `/usr/bin/python3` 3.10 复测 owned-message 路径：64 / 256 / 1024 轮的 chat 中位数分别为 1.71→0.93 ms、6.61→3.44 ms、26.62→14.03 ms；thinking 分别为 1.78→0.97 ms、6.94→3.79 ms、28.82→15.76 ms，约 1.83–1.92x。该数据仍是 prompt 编码阶段，不代表端到端服务提升。

另外修复工具图片排序的会话边界：新的 user/system/assistant 轮会清空上一轮的 tool-call 排序键，避免无关连续 tool 结果继承旧顺序。新增跨边界回归；当前 41 项标准库测试全部通过。

## GPU 首轮验收（2026-10-21）

固定 digest 基础镜像通过 `docker.m.daocloud.io` 前缀拉取，内容 digest 仍为 `sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3`。镜像构建阶段的 41 项测试、源码指纹和原生 MXFP8/UE8M0 scale loader 检查均通过。

真实 A100 smoke 已覆盖：

- TP8 / eager / control：文本和工具调用通过；
- TP8 / optimized / CUDA Graph：capture 完成，文本和工具调用通过；
- `decode-metadata`、`ragged-reuse`、`stacked-kv`：分别启动并通过文本/工具 smoke；
- 原生图文：固定图像夹具返回正确描述，图像 token 计数为 578；
- TP4DP2 / EP8：文本、工具、图像和约 18K prompt 长上下文通过。

首轮 C128 测量使用 `tp4dp2 / ep8`、DSpark k5、`stacked-kv`、全局 `max-seqs=128`；短合成请求 C64 约 2358 token/s，C128 约 4496 token/s，成功率均为 100%。后续 C256 测量已将推荐配置提升到全局 `max-seqs=256`。一次 `VLLM_MARLIN_USE_ATOMIC_ADD=1` A/B 仅约 1768 token/s，已排除，不进入默认配置。

这些结果证明当前实现能在目标 A100 上稳定运行；自然任务精度、长时间稳定性和更完整的生产负载矩阵仍需后续测试。260K 与 1M 单请求容量已通过，尚未完成多请求长文 soak。

## 单请求 Prefill / Decode 拆分

此前 C128 服务上的单请求测量为 TP4DP2 / EP8、DSpark k5、stacked-kv、上下文上限 262144；当前 C256 单请求 decode 结果已更新为下方补充记录。

- decode：17 prompt tokens、128 output tokens，TTFT 1.129 s，TPOT 3.60 ms，约 278 token/s；
- prefill：18010 prompt tokens、`max_tokens=1`，首 token 延迟 5.386 s，按 prompt tokens / 首 token 延迟估算约 3344 token/s。

Prefill 数字包含调度、队列和首 token 开销，是服务层近似值；decode 数字受 DSpark speculative decoding 和流式 chunk 口径影响。C128 短请求整机吞吐约 4496 token/s，不能与单请求 decode 速度直接比较。

当前 C256 单请求 decode：17 prompt tokens、128 output tokens，TTFT 0.566 s，TPOT 3.58 ms，约 279.5 token/s；C128/C256 的整机吞吐不能与该单请求速度直接比较。

## C256 与长上下文补充验收（2026-10-21）

当前 `tp4dp2 / ep8 + DSpark k5 + stacked-kv` 在全局 `max-seqs=256` 下成功启动并通过文本 smoke。短合成请求、输出上限 128 token 的矩阵结果：C128 为约 3928 token/s，C256 为约 7270 token/s，均为 0 失败；C256 的 TTFT P50/P95 约为 0.71/1.24 s。当前最佳吞吐配置切换为全局 `max-seqs=256`。

当前服务上完成了约 1K、32K、128K、250K 和 260K prompt 的首 token 预填充测试，全部通过：

| prompt tokens | 首 token延迟 | prefill 近似吞吐 |
|---:|---:|---:|
| 1016 | 0.315 s | 约 3222 token/s |
| 32016 | 5.405 s | 约 5924 token/s |
| 128016 | 13.103 s | 约 9770 token/s |
| 250016 | 28.972 s | 约 8630 token/s |
| 260016 | 30.597 s | 约 8498 token/s |

260K prompt 接近当前 262144 context 上限；另一个专用 1M 配置已通过 1,000,016 prompt 单请求，首 token 延迟约 223.7 s、prefill 近似约 4470 token/s。多请求长文 soak、长时间稳定性和自然任务质量回归仍需单独完成。

## 1M context 严格 A/B（同配置）

保持 TP4DP2/EP8、DSpark k5、stacked-kv、CUDA Graph、全局 `max-seqs=256`、chunk 和显存比例不变，只切换 `context=262144/1048576`。同一约 260K prompt 的结果：

| context | 260K prefill 近似吞吐 | 单请求 decode | C256 整机吞吐 |
|---:|---:|---:|---:|
| 262144 | 约 8498 token/s | 约 279.5 token/s | 约 7270 token/s |
| 1048576 | 约 7740 token/s | 约 279.3 token/s | 约 4240 token/s |

结论：1M 上限不会明显改变单请求 decode，但会让同一 260K 输入的 prefill 约下降 9%，并使高并发 C256 吞吐下降约 42%。原因来自更大的 KV/cache 规划、调度和 Graph 工作区，而不是单纯的 decode kernel 速度。

## 1M bounded prefill workspace（2026-10-21）

新增 `VLLM_DSV41_BOUNDED_PREFILL_WORKSPACE=1`，仅通过 `--experimental bounded-prefill` 开启。1M 配置下，indexer workspace 从多请求上界收敛到 `max_model_len / compress_ratio`，即单请求完整压缩上下文；metadata builder 对更大的多请求 prefill 继续分块，因此不会降低 1M 容量上限。

同一固定镜像、TP4DP2/EP8、DSpark k5、stacked-kv、CUDA Graph、全局 C256 的 benchmark 结果（1K 输入、128 输出、冷缓存）为：unbounded C128 / C256 约 843 / 1109 token/s；bounded C128 / C256 约 1286 / 1369 token/s，C256 使用相同 seed 8080，均 0 失败。bounded 单请求 260K prefill 约 9067 token/s，约 1M prefill 约 4772 token/s，均通过；1K 单请求 128 输出的流式 decode 约 158 token/s。该候选将同 seed 的 C256 提升约 23%，但仍保持显式开关，不覆盖 262K 默认配置。

固定 prompt 热矩阵随后完成了 C256 深度调优：当前 OMP 8 + EPLB + custom AG/RS 为约 4338 token/s；关闭 EPLB、关闭 custom AG/RS、OMP 1 分别降至约 4147、3815、2753 token/s。将 GPU memory utilization 提到 0.95 虽使 KV cache 增至约 14.61 GiB，随机冷 C256 反而降至约 1057 token/s，因此未采用。当前 1M 高并发服务保持 OMP 8、EPLB、custom AG/RS、bounded workspace 和 0.90 显存预算。
