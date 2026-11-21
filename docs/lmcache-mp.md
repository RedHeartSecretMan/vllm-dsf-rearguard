# LMCache MP 跨 DP 复用

`--lmcache-mp` 在同一台 8×A100 机器上运行两个容器。vLLM 持有 GPU KV，LMCache 从独立进程提供共享的主机 L1，两个 DP 实例都连接它。这一配置参考 [Issue #1 的跨 DP 实测](https://github.com/RedHeartSecretMan/vllm-dsf-rearguard/issues/1#issuecomment-6816255390) 和 [LMCache 的 V4.1 配方](https://docs.lmcache.ai/recipes/deepseek_v41_flash.html)。

## 准备

- 使用和默认部署相同的固定基础镜像及 1017 权重校验流程。开启 LMCache 的镜像额外安装固定版本的 LMCache 0.5.6 及其依赖，检查 wheel 的 SHA256 和混合 KV 连接器的导入。
- [默认配置](../config/runtime/lmcache-mp.json) 使用 512 GiB L1，先锁页 16 GiB，再由 LMCache 在后台逐段锁页至 512 GiB。主机内存按完整 L1 和 Engram 权重一起预算。L1 按 LRU 淘汰，不配置 L2。启动前检查主机可用内存至少覆盖 L1 上限。服务重启后 L1 为空。
- 两个容器共享主机网络和 IPC。vLLM API、LMCache ZMQ 和 LMCache HTTP 都绑定 `127.0.0.1`。两个容器由同一镜像启动，确保 CUDA IPC 两端使用相同的 LMCache 版本。项目通过外部模块路径选择支持混合 KV 的连接器，并保留各组的独立对象。

```bash
ENABLE_LMCACHE_MP=1 bash scripts/build.sh vllm-dsf-rearguard:lmcache-mp
```

## 启动与验证

`$MODEL` 指向已校验的本机权重目录。先运行以下命令检查计划，再加上 `--start`。需要 8 张空闲 A100。启动器会先等待 LMCache 健康检查通过，再启动 vLLM 并在运行目录保存回执。

```bash
python3 launch.py "$MODEL" --image vllm-dsf-rearguard:lmcache-mp \
  --name dsf-lmcache-test --port 16688 --context 262144 --chunk 16384 \
  --parallel tp4dp2 --dspark 5 --max-seqs 256 --memory 0.90 \
  --experimental stacked-kv --experimental bounded-prefill --text-only \
  --lmcache-mp
```

```bash
curl --fail http://127.0.0.1:16688/health
curl --fail http://127.0.0.1:19180/status
python3 -m scripts.benchmarks.lmcache_mp_probe \
  --base-url http://127.0.0.1:16688 --lmcache-url http://127.0.0.1:19180 \
  --prompt-tokens 20992 --output runtime/lmcache-mp-probe.json
```

验收要求 8 个 GPU worker 完成注册，同一文本前缀在 DP0 写入后被 DP1 从外部缓存读出，两次贪心生成的文本 SHA256 一致。探针从 `/status` 读取每个 chunk 的对象组数，等所有完整输入 chunk 的对象写入 L1 后再发热请求。Chunk 请求值为 128，实际尺寸以连接器与服务端协商结果为准。做吞吐 A/B 时对照臂保持相同镜像及参数，去掉 `--lmcache-mp` 并加上 `--host-network-ipc`，在两个臂使用同一批请求。多轮真实容量和多模态缓存需要各自的用例和结果。

停止试验先停 vLLM，再停 LMCache。大容量 L1 释放锁页需要时间。收到停止超时报错时，先用 `docker inspect` 核对容器状态和显存释放情况，再恢复其他服务。

```bash
docker stop --time 90 dsf-lmcache-test
docker stop --time 180 dsf-lmcache-test-lmcache
```

## 8×A100 文本验收

2026-11-21 在 8×A100-SXM4-80GB 上用提交 `7107612` 及镜像 `sha256:5b017f6bfc8a878ba20b342dd36163a066f5ebef4919c560e04752a2fae5272c` 运行。配置为 1017 权重、262K、16K prefill chunk、0.90 显存水位、TP4/DP2/EP8、DSpark k5 和纯文本输入。服务端确认 L1 完成锁页 512 GiB，无 L2，8 个 worker 注册，协商后的 KV chunk 为 128 token。

| 输入 token | 配对轮数 | DP1 外部命中 token | DP0 冷 / DP1 热首 token 延迟中位数 |
|---:|---:|---:|---:|
| 2,049 | 1 | 2,048 | 2.485 / 1.707 s |
| 20,992 | 3 | 20,991 | 1.756 / 0.460 s |

每对使用同一输入和盐值输出 8 个 token。4 对均通过逐引擎外部命中、LMCache 读回计数及文本 SHA256 校验。以上时间是同一服务的冷、热请求配对观察。有无缓存的吞吐和多轮并发容量需要用固定回放另行测量。
