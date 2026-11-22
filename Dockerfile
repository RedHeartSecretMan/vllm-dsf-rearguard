ARG BASE_IMAGE=lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3
FROM ${BASE_IMAGE}

# Install one unified DeepSeek-V4.1-Flash-1017 adaptation overlay.
COPY adaptation /opt/vllm-dsf-rearguard/adaptation
RUN python3 /opt/vllm-dsf-rearguard/adaptation/install.py
COPY launch.py /opt/vllm-dsf-rearguard/launch.py
COPY VERSION /opt/vllm-dsf-rearguard/VERSION
COPY native /opt/vllm-dsf-rearguard/native
COPY scripts /opt/vllm-dsf-rearguard/scripts
COPY config /opt/vllm-dsf-rearguard/config
COPY tests /opt/vllm-dsf-rearguard/tests
COPY manifests /opt/vllm-dsf-rearguard/manifests
# The official Linux x86_64/cp312 wheel was built with Torch 2.13 and CUDA 13.
# grpcio-observability pins grpcio; grpcio-status must use the same version.
ARG ENABLE_LMCACHE_MP=0
ARG ENABLE_KV_ONLY_NATIVE=0
RUN case "${ENABLE_LMCACHE_MP}" in \
      0) ;; \
      1) python3 -m pip install --break-system-packages --no-cache-dir --no-deps --require-hashes \
           'https://files.pythonhosted.org/packages/fb/e7/fc3d86192a87d23e00a5e62d1c55a6eaf7129c6624df14a01b86e224672a/lmcache-0.5.6-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl#sha256=c7892944ca91ec965c3448b2e30e0869f86a2499bdd01b70eb988e3436d71e4e' \
           'https://files.pythonhosted.org/packages/54/bf/f4a3b9693e35d25b24b0b39fa46d7d8a3c439e0a3036c3451764678fec20/grpcio-1.78.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl#sha256=12a771591ae40bc65ba67048fa52ef4f0e6db8279e595fd349f9dfddeef571f9' \
           'https://files.pythonhosted.org/packages/be/8d/9f39011d244a8cbe9d8df7c6692d17e06c0f62a80582fafdce97e8f0e707/grpcio_observability-1.78.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl#sha256=80ff6e088d8e21eb6d1228ea9193f7d1ab078ee8fc76ce81fd85f0b780f7a090' \
           'https://files.pythonhosted.org/packages/83/8a/1241ec22c41028bddd4a052ae9369267b4475265ad0ce7140974548dc3fa/grpcio_status-1.78.0-py3-none-any.whl#sha256=b492b693d4bf27b47a6c32590701724f1d3b9444b36491878fb71f6208857f34' \
         && CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void \
           python3 /opt/vllm-dsf-rearguard/scripts/check_lmcache_runtime.py ;; \
      *) echo 'ENABLE_LMCACHE_MP must be 0 or 1' >&2; exit 1 ;; \
    esac
RUN case "${ENABLE_KV_ONLY_NATIVE}" in \
      0) ;; \
      1) VLLM_BUILD_KV_ONLY_NATIVE=1 python3 /opt/vllm-dsf-rearguard/scripts/build_kv_only_native.py ;; \
      *) echo 'ENABLE_KV_ONLY_NATIVE must be 0 or 1' >&2; exit 1 ;; \
    esac
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
