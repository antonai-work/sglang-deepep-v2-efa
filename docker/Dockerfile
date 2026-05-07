# syntax=docker/dockerfile:1.10
# =============================================================================
# sglang-deepep-v2-efa - SGLang + DeepEP V2 + AWS EFA
# =============================================================================
#
# STATUS: SCAFFOLD. The v0.2.5 base child here has NOT been revalidated
# against SGLang yet. Wave 29 in the private dev tree
# (deepep-v2-integration) tracks the revalidation. This Dockerfile lands
# the cascade-symmetric source shape; body will be materialized once
# Wave 29 PROVEN evidence is in hand.
#
# Dual-path build (sibling pattern from vllm-deepep-v2-efa):
#   BUILD_MODE=fast    -> FROM ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.5-sm90a
#   BUILD_MODE=vanilla -> TODO (Wave 29). See sibling repos for inline pattern.
#
# Build:
#   docker build -f docker/Dockerfile --build-arg BUILD_MODE=fast \
#                -t sglang-deepep-v2-efa:fast .
#
# Sibling repos:
#   - antonai-work/deepep-v2-efa-base      (base substrate)
#   - antonai-work/vllm-deepep-v2-efa      (reference inference overlay)
#   - antonai-work/trtllm-deepep-v2-efa    (sibling inference overlay)
# =============================================================================

ARG BUILD_MODE=fast
ARG BASE_IMAGE_FAST=ghcr.io/antonai-work/deepep-v2-efa-base:v0.2.5-sm90a

FROM ${BASE_IMAGE_FAST} AS base-fast

FROM base-fast AS release

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# SGLang overlay — Wave 32 version bump.
#
# Why 0.5.11: Upstream released 0.5.11 on 2026-05-05 with CUDA 13 + torch 2.11
# as the DEFAULT build matrix (tracking issue sgl-project/sglang#21498,
# upgrade PR sgl-project/sglang#21247, Dockerfile prep #23593, sgl-kernel prep
# #24162). Before 0.5.11, sgl_kernel was built against torch 2.9 C10 ABI (the
# `i` variant of c10_cuda_check_implementation). 0.5.11 ships sglang-kernel
# 0.4.2 whose common_ops.abi3.so uses the torch 2.11 ABI (`j` variant —
# empirically verified via nm -D on /tmp/sgl-k4.2/sgl_kernel/sm90/common_ops).
#
# Prior attempt (v0.2.5 wave 29d): sglang 0.5.6.post2 imported but sgl_kernel
# failed at undefined symbol _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
# (the 2.9 ABI) on our torch 2.11 base. Bumping to 0.5.11 resolves this.
#
# Note: `sgl-kernel` was renamed to `sglang-kernel` between 0.5.6 and 0.5.11.
# -----------------------------------------------------------------------------
ARG SGLANG_VERSION=0.5.11

# Recipe (proven via docker-run smoke on v0.2.5 base):
#   1. Install sglang code only (--no-deps), skipping its torch hard pin
#   2. Compute sglang's transitive deps MINUS torch/flashinfer/torchvision/etc
#      via importlib.metadata.requires('sglang')
#   3. Install that filtered dep list normally (pip resolves their transitives
#      — including sglang-kernel==0.4.2 with the torch 2.11 ABI)
#   4. Force-reinstall nvidia-nccl-cu13 + nvidia-nvshmem-cu13 to restore v0.2.5
#      versions (a deep transitive may have pinned older 2.28.9 / 3.4.5)
#
# Why the filter: sglang's requires_dist still hard-pins torch==2.11.0 and
# torchao / torchaudio / torchcodec which would overwrite our base install.
# The cu13 invariant (EVIDENCE-LEDGER Wave 7d #1) forbids this.
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      "sglang==${SGLANG_VERSION}"

RUN set -e; \
    python3 -c "\
from importlib.metadata import requires; \
reqs = requires('sglang'); \
bad = {'torch','torchvision','torchaudio','torchcodec','torch_memory_saver','torchao','flashinfer_python','flashinfer_cubin','triton'}; \
good = [r.split(';')[0].strip() for r in reqs if not any(r.lower().startswith(b.lower()) for b in bad)]; \
print('\n'.join(good))" > /tmp/sglang-deps.txt && \
    pip install --no-cache-dir --break-system-packages -r /tmp/sglang-deps.txt

# Force-restore v0.2.5 cu13 stack (transitive deps may have downgraded)
ARG NVIDIA_NCCL_PIN=2.30.4
ARG NVSHMEM_PIN=3.6.5
RUN pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
      "nvidia-nccl-cu13==${NVIDIA_NCCL_PIN}" \
      "nvidia-nvshmem-cu13==${NVSHMEM_PIN}"

ENV PYTHONPATH=/opt

WORKDIR /opt

LABEL org.opencontainers.image.title="sglang-deepep-v2-efa" \
      org.opencontainers.image.description="SGLang + DeepEP V2 + AWS EFA MoE inference cascade (Wave 29 pending)." \
      org.opencontainers.image.source="https://github.com/antonai-work/sglang-deepep-v2-efa" \
      org.opencontainers.image.licenses="MIT" \
      status.wave="32" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="146cc356aa00c39ac1590c05775e05b0f031e70c" \
      deepep.pr="612" \
      sglang.version="0.5.11" \
      sglang.upstream_pr="24443"
