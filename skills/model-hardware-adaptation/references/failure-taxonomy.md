# Failure taxonomy and diagnosis

从首个可信错误反向定位根因。先按阶段和时间线找最早的 worker、CUDA、assert、OOM、schema 或来源错误，再处理下游 Gloo/NCCL、HTTP 500、超时和客户端异常。

## 通用故障流程

1. 保存启动参数、connector JSON、并行拓扑、模型 KV 配置、block/hash/chunk layout、hybrid KV 约束、首请求时间线和退出码；具体字段名按当前实现记录。
2. 判断阶段：来源/构建、启动解析、KV 初始化、首次请求、prefill、decode、后台 transfer、容量 admission 或进程退出。
3. 先建立控制组：native KV、单卡或 `DP=1`、`EP` 关闭、speculative decoding 关闭、CUDA Graph 关闭、eager、最小文本请求。
4. 对每个适用 connector 分开验证 schema、启动、首个短请求、长请求和历史恢复；全部检查通过后才写“可用”，不适用 connector 标 `NA`。
5. 逐项恢复 DP、EP、EPLB、speculative decoding、CUDA Graph、低精度路径、chunk 和 connector，每次只恢复一个变量。
6. Gloo/NCCL `connection closed` 出现时，按时间戳回到最早 worker 错误；将通信断链标为二次症状，除非有独立通信首错。
7. capacity queue 只有在请求最终失败时才归为失败；排队但完成属于容量/调度观测。
8. launcher 通过 JSON 或 CLI 校验只证明配置形状，不证明模型、并行和 connector 的 runtime 组合可用。

## 分类表

| 分类 | 首个可信症状 | 常见假象 | 必查日志 | 最小复现 | 是否继续测试 | 必存证据 |
|---|---|---|---|---|---|---|
| 来源/版本不一致 | manifest、镜像或模型 hash 不符 | 同名标签看似相同 | build、runtime、launch receipt | 只做 hash 校验 | 否 | digest、差异文件 |
| 权重/tokenizer 不匹配 | 配置加载、编码或输出异常 | 误判为 kernel | 模型校验、tokenizer、renderer | 单条文本 golden | 否 | 文件 hash、golden diff |
| schema/协议失败 | `Unsupported input item type`、400 | parser 被认为能修输入 | gateway、provider、engine 校验 | 单轮和续轮 payload | 否 | 脱敏请求、响应、责任层 |
| renderer/processor | token、媒体顺序或帧映射错误 | 误判为 cache miss | renderer、processor、media ledger | 单图/单视频/混排 | 否 | fixture hash、采样清单 |
| attention/kernel | assert、非法访问、数值差异 | 下游超时或 Gloo | worker、CUDA、kernel | eager、单卡、最小 shape | 否 | 首错、shape、dtype |
| parallel/collective | rank、NCCL、EP/TP 断链 | worker 首错被隐藏 | 最早 rank 日志和拓扑 | `DP=1`、`EP` off 逐项恢复 | 否 | 拓扑、rank、collective |
| CUDA Graph | capture/replay、地址或 shape 失败 | 只在并发随机失败 | capture、allocator、eager 对照 | eager + 单 shape | 否 | capture shapes、对照 |
| 显存/KV capacity | admission 拒绝、容量不足或 queue | 把 queue 当失败 | 启动预算、KV、allocator | 缩小上下文/并发、native KV | 仅低负载诊断 | 预算、capacity、曲线 |
| OOM | CUDA/allocator 明确 out of memory | 后续超时被当根因 | 首个 worker、CUDA、allocator | 降低上下文、chunk、并发 | 否，先回容量门禁 | 首错、显存曲线 |
| worker crash | 首请求 HTTP 500、进程退出 | Gloo/NCCL 二次断链 | worker stderr、退出码、core | 最小文本请求 | 否 | 首错和时间线 |
| Gloo/NCCL 二次错误 | peer closed | 误归因通信 | worker 首错、rank 时间线 | 修首错后复现 | 否 | 二次关系说明 |
| 多模态超限 | 数量/视觉 token/帧超限 | 静默丢媒体 | API、processor、scheduler | 逐步增加媒体 | 否 | 限制、错误、ledger |
| 结果错误 | golden、PPL、工具或事件不一致 | 吞吐提升掩盖错误 | 输出、logprobs、协议回放 | 固定 seed/输入 | 否 | 差异样本、精度 |
| 性能回归 | 指标超 baseline | 预热/cache 污染 | 原始样本、warmup、cache | 重跑固定 baseline | 否 | 样本、配置 hash |
| 部署/回滚 | canary、健康或回滚失败 | 仅看容器启动 | 发布、监控、旧 receipt | 单实例 canary/rollback | 否 | 回滚记录、外部状态 |

## KV connector 特例

- CPU offload 或 KV transfer connector 能启动不代表可用；若首个短请求触发 worker crash，先保存 worker 首错，再把 Gloo/NCCL 断链标为二次症状。
- transfer/offload connector 可能在启动阶段因 hybrid KV layout 对齐失败；把 block、hash、chunk、prefix cache 和 cache 约束作为一组联合检查。
- `native` KV 成功只证明 native 路径；不能证明 offload、transfer、DP、EP、EPLB、speculative decoding、CUDA Graph 或低精度路径的组合可用。
- connector 失败后只恢复一个变量，直到最小矩阵能解释差异。

## Responses、媒体和 store 特例

- `tool_call_parser` 只解析模型输出；`Unsupported input item type` 属于输入 schema/协议层，先查 agent、gateway、provider adapter 和 engine 的责任边界。
- 声明 tools 时，Chat 的 `tool_calls` / `role=tool` 与 Responses 的 `function_call` / `function_call_output`、`custom_tool_call` / `custom_tool_call_output` 分开回放。
- Responses 必须覆盖首轮、SSE delta/done/completed、`previous_response_id` 和错误输入；声明 tools 时加工具结果续轮，声明媒体输入时加多模态历史。
- 有界 session store 需要关联对象联动淘汰、SSE 收尾清理、锁外取消任务、读取更新淘汰顺序和重启语义；单进程历史不是长期持久化。
- 多模态数量、视觉 token、帧数和上下文超限应明确拒绝；稳定 media identity 只能帮助缓存复用，不能绕过上限。
