ARG BASE_IMAGE=lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3
FROM ${BASE_IMAGE}

# Install one unified DeepSeek-V4.1-Flash-1017 adaptation overlay.
COPY adaptation /opt/flash1017/adaptation
RUN python3 /opt/flash1017/adaptation/install.py
COPY launch.py /opt/flash1017/launch.py
COPY scripts /opt/flash1017/scripts
COPY config /opt/flash1017/config
COPY tests /opt/flash1017/tests
COPY manifests /opt/flash1017/manifests
RUN PYTHONPATH=/opt/flash1017 \
    python3 -m unittest discover -s /opt/flash1017/tests -p 'test_*.py' -v
COPY scripts/check_runtime.py /opt/flash1017/check_runtime.py
RUN python3 /opt/flash1017/check_runtime.py
RUN CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void \
    python3 /opt/flash1017/tests/test_cpu_hotpaths.py --native-loader
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="deepseek-v4.1-flash-a100" \
      org.opencontainers.image.source="https://github.com/RedHeartSecretMan/deepseek-v4.1-flash-a100" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.licenses="Apache-2.0"
ENTRYPOINT ["vllm"]
