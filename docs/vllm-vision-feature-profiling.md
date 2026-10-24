# vLLM Vision Feature Profiling

本文整理 DeepSeek-V4.1-Flash-A100 当前视觉扩展的处理链路、缓存语义、chunk 调度和验证边界。视频能力是服务侧扩展：模型 checkpoint 本身提供图片视觉塔，视频被处理为按时间顺序排列的图片帧。

## 处理链路

```text
OpenAI image/video request
        │
        ├─ renderer：媒体顺序、稳定 video_id / uuid、文本 wrapper
        │
        ├─ pinned vLLM media IO：读取视频元数据并计算目标帧索引
        │       └─ 默认 fps=1、num_frames=32
        │
        ├─ DeepSeek multimodal processor：图片 patch、grid、frame span
        │
        ├─ Vision Tower + aligner：每张图片/每帧生成视觉 embedding
        │
        ├─ scheduler：图片原子、视频按完整帧切分、文字正常跨 chunk
        │
        └─ language model：将视觉 embedding 合并到文本 embedding 序列
```

主要代码位置：

- renderer：`adaptation/files/vllm/renderers/deepseek_v4.py`
- tokenizer 和视频 wrapper：`adaptation/files/vllm/tokenizers/deepseek_v41.py`、`video_prompt.py`
- 图片/视频预处理：`adaptation/files/vllm/models/deepseek_v4_1/common/mm_preprocess.py`
- NVIDIA/AMD 视觉模型：`adaptation/files/vllm/models/deepseek_v4_1/{nvidia,amd}/vl_model.py`
- chunk 调度：`adaptation/files/vllm/v1/core/sched/scheduler.py`

## 图片输入

一张图片经过 resize、padding、patch 化后，计算 ViT grid 和 LLM grid。每张图片生成一个完整 span：

```text
[IMAGE_START]
([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
[IMAGE_END]
```

span 长度为：

```text
n_llm_h * (n_llm_w + 1) + 2
```

图片 span 中的位置在 `input_ids` 中使用同一个 `image_token_id=129264`。`types` tensor 区分 `IMAGE_START`、图片位置、换行和 `IMAGE_END`。模型用 Vision Tower 和 aligner 生成图片位置的 embedding，再用学习到的 start、newline、end 向量补齐整个 span。

图片不会增加自定义 image ID。vLLM 原生媒体 cache 使用图片内容和媒体处理配置生成的 identifier；图片本身是一个独立媒体 item，不需要视频那样的序列身份。

## 视频采样

启动配置为：

```json
{
  "limit-mm-per-prompt": {
    "image": 64,
    "video": {"count": 2, "num_frames": 32}
  },
  "media-io-kwargs": {
    "video": {"num_frames": 32, "fps": 1}
  }
}
```

视频采样由 pinned vLLM 的 `VideoMediaIO` / `VideoBackend` 负责。它先读取总帧数、源 FPS 和时长，再计算目标帧索引，然后把目标帧交给 codec/backend 读取。DeepSeek processor 不再做第二次抽样。

目标帧数的语义是：

```text
min(total_frames, 32, floor(duration * 1 FPS))
```

如果视频超过约 32 秒，vLLM 在全视频时间范围内均匀选择最多 32 个时间点。processor 只接受不超过 32 帧的 media IO 输出；超过上限会报错，避免悄悄改变采样时间关系。

需要区分“索引提前计算”和“codec 实际解码成本”：目标索引在 media IO 层提前确定，processor 只接收选中的帧；具体 codec 是否仍需要顺序扫描 GOP 或完整码流，由 backend 决定。

processor 会保留：

```text
video_timestamps
video_frame_span_lengths
video_vit_grid
video_llm_grid
video_types
```

## 视频视觉 span

视频不是一个新的 temporal encoder。每一帧单独复用图片路径：

```text
frame_1 → 图片 span 1
frame_2 → 图片 span 2
frame_3 → 图片 span 3
```

一个视频 item 最终得到连续的视觉 embedding：

```text
[frame_1 的完整图片 span]
[frame_2 的完整图片 span]
[frame_3 的完整图片 span]
```

`video_frame_span_lengths=[L1,L2,L3]` 记录每帧完整图片 span 的长度，包括该帧的图片局部 wrapper。它不包含视频全局 wrapper 的普通文字 token。

当前模型没有使用额外的 temporal embedding。视频时序信息来自：

1. media IO 返回的帧顺序；
2. 每帧自己的图片 span；
3. wrapper 中的实际采样时间清单；
4. scheduler 对帧边界的保护。

## 视频全局 wrapper

一个视频只保留一个应用层 `<｜deepseek_image｜>` placeholder。renderer 会把实际采样时间写入普通文本 wrapper：

```text
Video ID: vid_a91c7e2b4c8d102f.
The following 3 image spans are these frames in the same order:
Frame 1 at 0.000s.
Frame 2 at 1.935s.
Frame 3 at 3.871s.
Treat them as one video sequence.
Video frame sequence 1.
<｜deepseek_image｜>
End of video vid_a91c7e2b4c8d102f.
```

`Frame N at t` 清单与后续连续图片 span 按相同顺序对应。当前 pinned vLLM replacement 接口将一个视频 item 展开成一个连续视觉 replacement，不能在同一个 video item 内原生交错插入文字和每帧 span，所以 `Video frame sequence N` 是 occurrence 匹配锚点：

- 它帮助 processor 区分 image placeholder 和 video placeholder；
- 它在 replacement 时被消耗，不作为视觉 token 进入模型；
- 时间清单和视频 ID 文本仍保留在模型 prompt 中。

多视频请求会分别生成：

```text
Video frame sequence 1. + placeholder(V1)
Video frame sequence 2. + placeholder(V2)
```

每个视频的所有帧仍然只占一个应用层 placeholder。

## video_id 和 uuid

`video_id` 和 content-part `uuid` 使用同一稳定身份。

生成顺序：

1. 优先使用请求中显式且安全的 `video_id`、`media_id` 或 `uuid`；
2. 没有显式 ID 时，对视频来源和内容块参数做规范化 JSON；
3. 排序 key、移除身份别名；
4. UTF-8 SHA256，取前 16 位十六进制字符；
5. 添加 `vid_` 前缀。

例如同一视频引用在多轮请求中会重复得到：

```text
vid_a91c7e2b4c8d102f
```

`video_id` 的作用是模型上下文中的稳定引用；`uuid` 的作用是 vLLM 多模态 encoder/prefix cache 的机器身份。UUID 不会让模型主动查询外部 cache，第二轮仍需要重放媒体/历史，或服务端已有对应的视觉/KV cache。

fallback ID 基于媒体引用和参数描述，不是解码后视频字节 hash。URL 内容被替换或签名 URL 改变时，稳定性需要由客户端显式 UUID 或服务端内容身份保证。

## chunk 调度

默认配置保持：

```bash
--disable-chunked-mm-input
--chunk 32768
```

原因是单个视频 item 的最大视觉预算为：

```text
32 帧 × 1024 token/帧 = 32768 visual tokens
```

关闭 chunked 时，vLLM 要求一个完整 multimodal item 能够放入单个 batch，因此默认预算不能低于 32768。

显式使用 `--no-disable-chunked-mm-input` 后，项目 scheduler 执行：

```text
文字：可以在任意 chunk 边界跨越
图片：完整图片 span 不可拆
视频：只能在 video_frame_span_lengths 的累计边界拆分
```

例如：

```text
video_frame_span_lengths = [10, 12, 10]
累计边界 = 10、22、32
```

chunk 落到第 15 个视频 token 时，回退到第 10 个；落到第 22 个时，可以在第二帧结束后切分。缺少帧长度元数据时，scheduler 保守地把整个视频当作一个原子 item。

CUDA 平台原本会对未声明支持的 multimodal prefix-LM 强制打开 `disable_chunked_mm_input`。DeepSeek 配置声明了 `supports_chunked_mm_input`，因此只有显式 opt-in 时才使用帧边界调度；默认仍保持原子视觉输入。

## 多轮历史和缓存

第一轮：

```text
视频媒体
  → 视频 encoder output
  → wrapper + 视觉 span
  → Transformer prefill
  → encoder cache / prefix KV cache
```

第二轮重放相同媒体历史时：

```text
相同 uuid
  → 可以查找视频 encoder output
相同 wrapper 和历史 prompt
  → 可以命中 prefix KV
新增问题
  → 只计算新增上下文和生成内容
```

`video_id` 不是媒体数据库查询接口。只发送“继续分析 vid_xxx”而不提供历史视觉内容或服务端 cache，模型无法恢复视频。

## Responses 协议相关

Responses adapter 负责把历史 output item 转成内部 chat message：

- 普通 output message 转 assistant content；
- function/custom-tool call 转 assistant tool call；
- function/custom-tool output 转 tool message；
- `previous_response_id` 重放时保留 tool call 和 tool output；
- reasoning item 仍按当前安全策略过滤，不把隐藏 reasoning 自动重放给模型；
- 多段 output content 和 refusal 不再假设 `content[0].text`。

真实 endpoint 仍需要在 pinned vLLM runtime 中验收。CPU 测试主要覆盖 schema、转换和 streaming 状态机。

## 验证入口

当前 uv Python 3.10 环境的 CPU 合约测试包含：

- 官方 tokenizer parity；
- 视频 ID、UUID、时间清单和多轮 wrapper；
- 视频采样元数据；
- 图片原子调度；
- 视频多帧边界调度；
- Responses custom-tool 转换和 streaming；
- manifest、启动配置和 benchmark contract。

运行：

```bash
UV_NO_CONFIG=1 \
UV_CACHE_DIR=/tmp/dsv41-uv-cache \
uv run --python /tmp/dsv41-uv310/bin/python --no-project \
python -m unittest discover -s tests -v
```

GPU 验收前至少应覆盖：单图、单视频、超过 32 秒视频、双视频、image-video-image 混排、默认非 chunked、帧边界 chunked、多轮视频重放和 Responses endpoint。

## 仍然存在的边界

- 视频仍然复用图片 encoder，没有官方 temporal encoder；
- wrapper 的时间信息是文本提示，不是学习到的 temporal embedding；
- codec 是否顺序扫描完整码流取决于具体 media backend；
- GPU endpoint 和 pinned runtime 的 Responses 测试需要单独归档真实回执；
- processor 不再二次抽样，media IO 返回超过 32 帧会失败而不是静默改写时间映射。
