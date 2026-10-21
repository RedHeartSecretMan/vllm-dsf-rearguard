# 验证记录

2026-10-20。本文件只记录本项目当前实际完成的验证。

| 项目 | 状态 | 范围 |
|---|---|---|
| 官方 1017 编码器来源 | PASS | 独立下载固定 revision，SHA256 与参考完全一致 |
| 官方文件清单 | PASS | ModelScope API 返回的 83 个文件，大小与 SHA256 全部吻合；Git 元数据 `.gitattributes` 不在 API 返回中 |
| 目标机已有 config / encoding | PASS | 两文件 SHA256 与固定官方快照一致 |
| 工具续轮问题复现 | PASS | 旧编码器丢失同轮思考历史，官方保留；修复后参考差分通过 |
| CPU 编码/renderer 测试 | PASS | 包含全部 5 个官方输入文件的 40 种模式/effort 组合，及长工具历史和上下文裁剪；不运行模型 |
| 全部标准库回归 | PASS | 共 40 项在本地通过；包括安装、来源、配置和计量契约、消息边界和 wrapper 归属语义 |
| CPU 张量/控制流回归 | PASS | 34 个场景在目标机 GPU 不可见环境中通过；固定镜像构建阶段的 runc/无网络原生 loader 检查也通过 |
| 长工具历史编码 A/B | PASS | 6 档目标机 CPU 对照，全部逐字匹配官方参考；详见 [优化记录](optimization-progress.md) |
| Python 源码编译与静态检查 | PASS | 源码编译、项目自有代码 Ruff 检查、构建脚本 Bash 语法检查 |
| 目标机镜像构建/导入 | PASS | 通过 Daocloud 前缀拉取同一固定 digest；镜像 ID `sha256:9445e84b98408f4c341da945b456a7ca3d3bf806d64d748ea982bf8ab09881ab`，33 个源码文件安装成功 |
| 1017 完整权重校验 | PASS | 48/48 分片完整 SHA256 通过，revision `f04a978641fa9bfe6f448ec5da3b2e1c2520a6ad` |
| SM80 数值 / Graph / `ep8` | PASS | TP8 Graph capture、TP4DP2/EP8 启动和 smoke 通过；尚未完成题级精度回归 |
| 整模型文本/工具/视觉/长文 | PASS | 文本、工具、单图、260K prompt 和 1,000,016 prompt 单请求均通过；多请求长文 soak 尚未完成 |
| 同负载 A/B 与精度回归 | PARTIAL | 已完成 C64/C128/C256 短请求矩阵和 prefill 矩阵；自然任务质量和完整对照仍未完成 |

当前推荐验收配置为 `tp4dp2 / ep8 + DSpark k5 + stacked-kv + max-seqs=256`。

目标机默认 `python3` 指向 3.6，最初的目标机 CPU 检查因此失败；
改用已安装的 `/usr/bin/python3` 3.10 后完成回归。另修复宿主已有同名 `scripts`
包造成的 namespace 导入冲突，使用显式项目 package。没有修改系统 Python。

目标机回归还复现了同一文件系统时钟粒度内的等长写入未改变 stat 签名：
启动前现在重算所有文件 SHA256，不能仅靠签名复用旧校验。该失败修复后，
后续目标机测试通过。仅进度预检不读取全部权重，明确输出 `NOT_VERIFIED`。

另尝试过相同 digest 的直连地址（等待响应头超时）和现有镜像站 manifest
（拒绝访问），未更换基础版本。重试证据见 `runtime/base-pull-retry-1.log` / `.exit`。
构建及 CPU 导入成功必须以后续成功回执、镜像 ID 及对应源码 revision 为准。
