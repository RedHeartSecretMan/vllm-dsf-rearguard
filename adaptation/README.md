# 适配层

`adaptation/` 保存项目对基础 vLLM 镜像的全部源码适配。镜像构建时，`install.py` 先核对清单中的基础文件和适配文件哈希，全部一致后再将适配文件原子写入运行时；任一文件不一致都会中止安装。

## 目录结构

| 路径 | 说明 |
|---|---|
| `files/vllm/` | 按 vLLM 原始导入路径保存的适配文件 |
| `manifest.json` | 基础文件、适配文件和参考实现的 SHA256 与用途 |
| `install.py` | 校验全部文件后写入运行时，由 `Dockerfile` 调用 |
| `reference/` | ModelScope 1017 版本的编码器、图像处理器和 tokenizer wrapper |

## 维护规则

- 修改 `files/vllm/` 后，需要同步更新 `manifest.json` 中的哈希，否则镜像构建和启动前的源码校验都会失败。
- `reference/` 用于一致性测试，应与上游文件保持逐字一致。
- 适配层不包含模型权重。

许可证和第三方来源见仓库根目录的 [LICENSE](../LICENSE) 与 [NOTICE](../NOTICE)。
