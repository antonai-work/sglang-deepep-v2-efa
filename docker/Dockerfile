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
# SGLang overlay. Wave 29 will:
#   1. Confirm 0.5.6.post2 still imports cleanly on the v0.2.5 cu13 base.
#   2. Add the DeepEP V2 adapter from upstream sgl-project/sglang#24443.
#   3. Bake the DeepGEMM JIT cache for Qwen3-30B-A3B-FP8 so first
#      /v1/chat/completions returns in <10 s (V1 suffered a 10-20 min stall).
# -----------------------------------------------------------------------------
ARG SGLANG_VERSION=0.5.6.post2

# Minimal overlay for Wave 29 cross-node smoke. Full DeepEP V2 adapter
# (sgl-project/sglang#24443) + DeepGEMM JIT cache baking TBD.
RUN pip install --no-cache-dir --break-system-packages "sglang[all]==${SGLANG_VERSION}"

# Guard torch/nccl/nvshmem to match v0.2.5 base (cu13-unified stack).
# Any torch upgrade silently re-installs nvidia-nccl/nvshmem wheels.
ARG NVIDIA_NCCL_PIN=2.30.4
ARG NVSHMEM_PIN=2.11.2
RUN pip install --no-cache-dir --break-system-packages --no-deps --force-reinstall \
      "nvidia-nccl-cu13==${NVIDIA_NCCL_PIN}" \
      "nvidia-nvshmem-cu13==${NVSHMEM_PIN}"

ENV PYTHONPATH=/opt

WORKDIR /opt

LABEL org.opencontainers.image.title="sglang-deepep-v2-efa" \
      org.opencontainers.image.description="SGLang + DeepEP V2 + AWS EFA MoE inference cascade (Wave 29 pending)." \
      org.opencontainers.image.source="https://github.com/antonai-work/sglang-deepep-v2-efa" \
      org.opencontainers.image.licenses="MIT" \
      status.wave="29-pending" \
      deepep.branch="aws-efa-auto-qp-cap-v2" \
      deepep.sha="146cc356aa00c39ac1590c05775e05b0f031e70c" \
      deepep.pr="612" \
      sglang.upstream_pr="24443"
