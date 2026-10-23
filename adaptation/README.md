# 统一适配层

`adaptation/` 是项目唯一的 vLLM 源码适配入口。

- `files/vllm/`：按 vLLM 原始导入路径保存最终覆盖文件。
- `manifest.json`：记录基础文件、适配文件和职责分类的 SHA256。
- `install.py`：先校验全部文件，再原子写入运行时。
- `reference/`：编码器、图像处理器和 tokenizer wrapper 的固定参考实现，与 ModelScope 1017 版本逐字相同，哈希记录在 `manifest.json`。

适配层不包含模型权重。许可证和第三方来源统一见仓库根目录的
[LICENSE](../LICENSE) 与 [NOTICE](../NOTICE)。
