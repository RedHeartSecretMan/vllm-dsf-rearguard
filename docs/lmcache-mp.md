# LMCache MP 跨 DP 复用

`--lmcache-mp` 在同一台 8×A100 机器上运行两个容器。vLLM 持有 GPU KV，LMCache 从独立进程提供共享的主机 L1，两个 DP 实例都连接它。这一配置参考 [Issue #1 的跨 DP 实测](https://github.com/RedHeartSecretMan/vllm-dsf-rearguard/issues/1#issuecomment-6816255390) 和 [LMCache 的 V4.1 配方](https://docs.lmcache.ai/recipes/deepseek_v41_flash.html)。

## 准备

- 使用和默认部署相同的固定基础镜像及 1017 权重校验流程。开启 LMCache 的镜像额外安装固定版本的 LMCache 0.5.6 及其依赖，检查 wheel 的 SHA256 和混合 KV 连接器的导入。
- [默认配置](../config/runtime/lmcache-mp.json) 使用 512 GiB L1 上限，启动分配 16 GiB，按需增长。L1 按 LRU 淘汰，不配置 L2。vLLM 启动前检查主机可用内存至少覆盖 L1 上限。实际部署还应留出 Engram 主机权重和系统内存。服务重启后 L1 为空。
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

验收要求 8 个 GPU worker 完成注册，同一文本前缀在 DP0 写入后被 DP1 从外部缓存读出，两次生成的 token 一致。Chunk 请求值为 128，实际尺寸以连接器与服务端协商结果为准。做吞吐 A/B 时对照臂保持相同镜像及参数，去掉 `--lmcache-mp` 并加上 `--host-network-ipc`，在两个臂使用同一批请求。多轮真实容量和多模态缓存需要各自的用例和结果。

停止试验先停 vLLM，再停 LMCache。容器身份可从启动回执核对。

```bash
docker stop dsf-lmcache-test
docker stop dsf-lmcache-test-lmcache
```
