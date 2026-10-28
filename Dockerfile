ARG BASE_IMAGE=lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3
FROM ${BASE_IMAGE}

# Install one unified DeepSeek-V4.1-Flash-1017 adaptation overlay.
COPY adaptation /opt/vllm-dsf-rearguard/adaptation
RUN python3 /opt/vllm-dsf-rearguard/adaptation/install.py
COPY launch.py /opt/vllm-dsf-rearguard/launch.py
COPY VERSION /opt/vllm-dsf-rearguard/VERSION
COPY scripts /opt/vllm-dsf-rearguard/scripts
COPY config /opt/vllm-dsf-rearguard/config
COPY tests /opt/vllm-dsf-rearguard/tests
COPY manifests /opt/vllm-dsf-rearguard/manifests
RUN PYTHONPATH=/opt/vllm-dsf-rearguard \
    python3 -m unittest discover -s /opt/vllm-dsf-rearguard/tests -p 'test_*.py' -v
COPY scripts/check_runtime.py /opt/vllm-dsf-rearguard/check_runtime.py
RUN python3 /opt/vllm-dsf-rearguard/check_runtime.py
RUN cd /opt/vllm-dsf-rearguard && CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void \
    python3 -m scripts.hotpaths.check --native-loader
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="vllm-dsf-rearguard" \
      org.opencontainers.image.source="https://github.com/RedHeartSecretMan/vllm-dsf-rearguard" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.licenses="Apache-2.0"
ENTRYPOINT ["vllm"]
