# syntax=docker/dockerfile:1.10
# =============================================================================
# sglang-deepep-v2-efa - SGLang + DeepEP V2 + AWS EFA
# =============================================================================
#
# STATUS: Wave 38.1 — SGLang installed from dmvevents/sglang fork branch
# `deepep-v2-elasticbuffer-support` (HEAD ecb4b0e, upstream PR #24443 HEAD).
# This branch carries the probe-plumbing that lets SGLang select DeepEP V2's
# `ElasticBuffer` at buffer-build time when `SGLANG_DEEPEP_USE_V2=1` is set.
#
# Why the fork: upstream sgl-project/sglang PR #24443 is OPEN DRAFT with 0
# maintainer reviews. Our fork branch is mergeable, rebased to main, and
# is the only way to get native V2 routing in SGLang today.
#
# Base cascade:
#   nvidia/cuda:13.0.0 -> deepep-v2-efa-base:v0.2.5-sm90a  (patched NCCL + EFA + DeepEP V2)
#   -> sglang-deepep-v2-efa:<tag>  (this repo, Wave 38.1)
#
# Build:
#   docker build -f docker/Dockerfile -t sglang-deepep-v2-efa:fast .
#
# Sibling repos:
#   - antonai-work/deepep-v2-efa-base      (base substrate)
#   - antonai-work/vllm-deepep-v2-efa      (PROVEN native V2 via PR #41183)
#   - antonai-work/trtllm-deepep-v2-efa    (shim-based V2, see Wave 30)
# =============================================================================

ARG BUILD_MODE=fast
ARG BASE_IMAGE_FAST=ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.5-sm90a

FROM ${BASE_IMAGE_FAST} AS base-fast

FROM base-fast AS release

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# SGLang overlay — Wave 38.1: install from fork PR #24443.
#
# Upstream state (verified 2026-05-12):
#   - sgl-project/sglang v0.5.11 released 2026-05-05, NO native V2 support
#   - PR #24443 adds `SGLANG_DEEPEP_USE_V2=1` env + `_build_v2_buffer`
#     classmethod in `sglang/srt/layers/moe/token_dispatcher/deepep.py`
#   - PR state: OPEN DRAFT, MERGEABLE, 0 maintainer reviews, 7 days stale
#   - HEAD: ecb4b0e (2026-05-08, lint fix)
#
# Fork repo: https://github.com/dmvevents/sglang (branch
# deepep-v2-elasticbuffer-support) carries the PR #24443 head commit.
#
# The fork is installed from source (git clone + pip install --no-deps) so
# the V2 probe-plumbing lands in the image. Without this, `--moe-a2a-backend
# deepep` routes through V1 `Buffer` which depends on NVSHMEM — and we'd
# hit Wave 29 / Wave 38 CUDA-illegal-memory failures.
# -----------------------------------------------------------------------------
ARG SGLANG_FORK_URL=https://github.com/dmvevents/sglang.git
ARG SGLANG_FORK_BRANCH=deepep-v2-elasticbuffer-support
ARG SGLANG_FORK_SHA=ecb4b0ed9f485ac6376d10facb8e5ded2f95d9b0

# Wave 38.1: install SGLang from fork branch with V2 ElasticBuffer probe.
# --no-deps: skip framework-level torch pin (base image already has torch 2.11 cu130).
# --no-build-isolation: reuse base-image torch so sgl-kernel's C++ ABI matches.
RUN set -eux; \
    git clone --depth 1 --branch "${SGLANG_FORK_BRANCH}" "${SGLANG_FORK_URL}" /tmp/sglang-fork; \
    cd /tmp/sglang-fork; \
    git checkout "${SGLANG_FORK_SHA}"; \
    pip install --no-cache-dir --break-system-packages --no-deps \
      /tmp/sglang-fork/python; \
    rm -rf /tmp/sglang-fork

# Compute + install sglang's transitive deps, filtering out torch/flashinfer/triton
# (would overwrite base-image cu13 stack).
RUN set -e; \
    python3 -c "\
from importlib.metadata import requires; \
reqs = requires('sglang'); \
bad = {'torch','torchvision','torchaudio','torchcodec','torch_memory_saver','torchao','flashinfer_python','flashinfer_cubin','triton'}; \
good = [r.split(';')[0].strip() for r in reqs if not any(r.lower().startswith(b.lower()) for b in bad)]; \
print('\n'.join(good))" > /tmp/sglang-deps.txt && \
    pip install --no-cache-dir --break-system-packages -r /tmp/sglang-deps.txt

# EVIDENCE-LEDGER Wave 7d invariant #1: force-restore v0.2.5 cu13 stack.
# sglang[deps] transitives can silently downgrade nvidia-nccl-cu13 -> cu12 and
# nvidia-nvshmem-cu13 -> 3.4.5, breaking deep_ep _C.so ABI.
ARG NVIDIA_NCCL_PIN=2.30.4
ARG NVSHMEM_PIN=3.6.5
RUN pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
      "nvidia-nccl-cu13==${NVIDIA_NCCL_PIN}" \
      "nvidia-nvshmem-cu13==${NVSHMEM_PIN}"

# Verify deep_ep + sglang imports with V2 probe-plumbing present.
RUN python3 -c "\
import deep_ep; \
assert hasattr(deep_ep, 'ElasticBuffer'), 'DeepEP V2 ElasticBuffer missing'; \
import sglang; print(f'sglang={sglang.__version__}'); \
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher; \
assert hasattr(DeepEPDispatcher, '_build_v2_buffer'), 'PR #24443 V2 probe missing — fork branch may not be applied'; \
print('[OK] sglang V2 ElasticBuffer routing ready')"

# -----------------------------------------------------------------------------
# V2 runtime ENV block (CRITICAL — overrides ambient base defaults when
# the pod is spawned with its own env section that does not include these).
# Mirrors the private integration tree:
#   /home/ubuntu/deepep-v2-integration/integrations/sglang-deepep-v2/docker/Dockerfile
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

# No-shim invariant: since the fork adds native V2 probe, we do NOT need
# api-shim. Assert /opt/api-shim does NOT exist to prevent accidental V1
# shim activation.
RUN test ! -e /opt/api-shim || (echo "ERROR: /opt/api-shim present but this overlay is native-V2"; exit 1)

WORKDIR /opt

LABEL org.opencontainers.image.title="sglang-deepep-v2-efa" \
      org.opencontainers.image.description="SGLang + DeepEP V2 + AWS EFA MoE inference (Wave 38.1: fork PR #24443 native ElasticBuffer routing)." \
      org.opencontainers.image.source="https://github.com/antonai-work/sglang-deepep-v2-efa" \
      org.opencontainers.image.licenses="MIT" \
      status.wave="38.1" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="146cc356aa00c39ac1590c05775e05b0f031e70c" \
      deepep.pr="612" \
      sglang.fork="dmvevents/sglang" \
      sglang.fork_branch="deepep-v2-elasticbuffer-support" \
      sglang.fork_sha="ecb4b0ed9f485ac6376d10facb8e5ded2f95d9b0" \
      sglang.upstream_pr="24443"
