# syntax=docker/dockerfile:1.10
# =============================================================================
# sglang-deepep-v2-efa - SGLang + DeepEP V2 + AWS EFA (Wave 38.2)
# =============================================================================
#
# STATUS: Wave 38.2 — SGLang 0.5.11 from PyPI wheel + in-place patch applying
# PR #24443 (DeepEP V2 ElasticBuffer probe). Pure-Python patch, no Rust
# compiler needed (Wave 38.1 attempted to install from fork git source,
# which triggered outlines_core Rust build + failed).
#
# Upstream PR: sgl-project/sglang#24443 "moe: add DeepEP V2 ElasticBuffer
# support" — OPEN DRAFT, MERGEABLE, 0 maintainer reviews 2026-05-12.
# HEAD ecb4b0e. Touches 1 file: python/sglang/srt/layers/moe/token_dispatcher/deepep.py
#
# The patch adds:
#   - `have_deepep_v2` probe (import deep_ep.ElasticBuffer)
#   - `SGLANG_DEEPEP_USE_V2=1` env gate in `get_deepep_buffer`
#   - `_build_v2_buffer` classmethod that constructs ElasticBuffer (MoE-shape)
#
# Without this patch, `--moe-a2a-backend deepep` instantiates V1 Buffer
# which depends on NVSHMEM — Wave 29 + 38 CUDA illegal memory failures.
#
# Base cascade:
#   nvidia/cuda:13.0.0 -> deepep-v2-efa-base:v0.2.5-sm90a
#   -> sglang-deepep-v2-efa:<tag> (this repo, Wave 38.2)
#
# Build:
#   docker build -f docker/Dockerfile -t sglang-deepep-v2-efa:fast .
#
# Sibling repos:
#   - antonai-work/deepep-v2-efa-base      (base substrate)
#   - antonai-work/vllm-deepep-v2-efa      (PROVEN native V2 via PR #41183)
#   - antonai-work/trtllm-deepep-v2-efa    (shim V2 via api-shim CompatBuffer)
# =============================================================================

ARG BUILD_MODE=fast
ARG BASE_IMAGE_FAST=ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.5-sm90a

FROM ${BASE_IMAGE_FAST} AS base-fast

FROM base-fast AS release

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# SGLang 0.5.11 from PyPI (wheel — no Rust compiler needed).
# -----------------------------------------------------------------------------
ARG SGLANG_VERSION=0.5.11

# Step 1: Install sglang code only (--no-deps) to avoid torch downgrade.
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      "sglang==${SGLANG_VERSION}"

# Step 2: Compute + install filtered deps (exclude torch family + triton).
RUN set -e; \
    python3 -c "\
from importlib.metadata import requires; \
reqs = requires('sglang'); \
bad = {'torch','torchvision','torchaudio','torchcodec','torch_memory_saver','torchao','flashinfer_python','flashinfer_cubin','triton'}; \
good = [r.split(';')[0].strip() for r in reqs if not any(r.lower().startswith(b.lower()) for b in bad)]; \
print('\n'.join(good))" > /tmp/sglang-deps.txt && \
    pip install --no-cache-dir --break-system-packages -r /tmp/sglang-deps.txt

# Step 3: Force-restore v0.2.5 cu13 stack (transitive deps may have downgraded)
ARG NVIDIA_NCCL_PIN=2.30.4
ARG NVSHMEM_PIN=3.6.5
RUN pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
      "nvidia-nccl-cu13==${NVIDIA_NCCL_PIN}" \
      "nvidia-nvshmem-cu13==${NVSHMEM_PIN}"

# -----------------------------------------------------------------------------
# Step 4: Apply PR #24443 patch to installed sglang deepep.py.
# The patch adds `have_deepep_v2` probe + `_build_v2_buffer` classmethod
# + `SGLANG_DEEPEP_USE_V2=1` env gate. Only touches one file.
# -----------------------------------------------------------------------------
COPY docker/deepep-v2-patched.py /tmp/deepep-v2-patched.py
# Whole-file replace: overwrite the installed deepep.py with the PR #24443
# fork-branch version (dmvevents/sglang @ deepep-v2-elasticbuffer-support).
# Simpler + more reliable than `patch` which has path-prefix + multi-hunk
# edge cases. The file is self-contained; no other sglang module needs
# touching since PR #24443 only adds an env-gate + classmethod.
RUN set -eux; \
    DEST=$(python3 -c "import sglang, os; print(os.path.join(os.path.dirname(sglang.__file__), 'srt/layers/moe/token_dispatcher/deepep.py'))"); \
    echo "Installing PR #24443 deepep.py at ${DEST}"; \
    test -f "${DEST}"; \
    cp /tmp/deepep-v2-patched.py "${DEST}"; \
    python3 -c "\
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer; \
assert hasattr(DeepEPBuffer, '_build_v2_buffer'), 'PR #24443 V2 probe missing on DeepEPBuffer'; \
print('[OK] sglang V2 probe installed (PR #24443 whole-file)')"

# Step 5: Verify DeepEP V2 exports are available for the probe.
RUN python3 -c "\
import deep_ep; \
assert hasattr(deep_ep, 'ElasticBuffer'), 'DeepEP V2 ElasticBuffer missing'; \
import sglang; print(f'sglang={sglang.__version__}'); \
print('[OK] sglang V2 ElasticBuffer routing ready')"

# -----------------------------------------------------------------------------
# V2 runtime ENV block (activates PR #24443 gate + inherits EFA/NCCL-Gin
# from base image). SGLANG_DEEPEP_USE_V2=1 is the new opt-in flag from the PR.
# -----------------------------------------------------------------------------
ENV DEEP_EP_BACKEND=nccl \
    DEEP_EP_USE_V2_SHIM=0 \
    SGLANG_DEEPEP_USE_V2=1 \
    NCCL_NET_PLUGIN=ofi \
    NCCL_GIN_TYPE=2 \
    NCCL_GIN_ENABLE=1 \
    NCCL_NVLS_ENABLE=0 \
    FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    FI_EFA_FORK_SAFE=1 \
    FI_EFA_ENABLE_SHM_TRANSFER=0 \
    OFI_NCCL_PROTOCOL=RDMA \
    OFI_NCCL_GIN_MAX_REQUESTS=512 \
    EP_EFA_MAX_QPS=2 \
    EP_EFA_RDMA_GBS=25.0 \
    PYTHONPATH=/opt

# No-shim invariant: the PR #24443 patch puts ElasticBuffer routing
# directly inside SGLang, so we do NOT need api-shim.
RUN test ! -e /opt/api-shim || (echo "ERROR: /opt/api-shim present but this overlay uses native PR #24443 patch"; exit 1)

WORKDIR /opt

LABEL org.opencontainers.image.title="sglang-deepep-v2-efa" \
      org.opencontainers.image.description="SGLang 0.5.11 + PR #24443 patch + DeepEP V2 + AWS EFA (Wave 38.2)." \
      org.opencontainers.image.source="https://github.com/antonai-work/sglang-deepep-v2-efa" \
      org.opencontainers.image.licenses="MIT" \
      status.wave="38.2" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="146cc356aa00c39ac1590c05775e05b0f031e70c" \
      deepep.pr="612" \
      sglang.version="0.5.11" \
      sglang.upstream_pr="24443" \
      sglang.patch="docker/pr24443-elasticbuffer.patch"
