ARG BASE_IMAGE=lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3
FROM ${BASE_IMAGE}

# Install one unified DeepSeek-V4.1-Flash-1017 adaptation overlay.
COPY adaptation /opt/dsv41-a100/adaptation
RUN python3 /opt/dsv41-a100/adaptation/install.py
COPY launch.py /opt/dsv41-a100/launch.py
COPY VERSION /opt/dsv41-a100/VERSION
COPY scripts /opt/dsv41-a100/scripts
COPY config /opt/dsv41-a100/config
COPY tests /opt/dsv41-a100/tests
COPY manifests /opt/dsv41-a100/manifests
RUN PYTHONPATH=/opt/dsv41-a100 \
    python3 -m unittest discover -s /opt/dsv41-a100/tests -p 'test_*.py' -v
COPY scripts/check_runtime.py /opt/dsv41-a100/check_runtime.py
RUN python3 /opt/dsv41-a100/check_runtime.py
RUN cd /opt/dsv41-a100 && CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void \
    python3 -m scripts.hotpaths.check --native-loader
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="deepseek-v4.1-flash-a100" \
      org.opencontainers.image.source="https://github.com/RedHeartSecretMan/deepseek-v4.1-flash-a100" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.licenses="Apache-2.0"
ENTRYPOINT ["vllm"]
