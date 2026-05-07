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

# SGLang 0.5.6.post2 hard-pins torch==2.9.1 (cu12 wheel) which shatters
# the v0.2.5 base's cu13 invariant. Install --no-deps and curate the
# runtime dep list to keep torch 2.11.0+cu130 intact.
#
# Invariant (EVIDENCE-LEDGER Wave 7d #1): torch silently pins nvidia-nccl
# + nvidia-nvshmem. Never let sglang's torch 2.9 in. The --no-deps install
# below preserves the base's cu13 stack wholesale.
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      "sglang==${SGLANG_VERSION}"

# Non-torch runtime deps sglang imports. Curated from sgl-project/sglang
# pyproject.toml minus {torch*, flashinfer*, nvidia-*-cu12, triton}.
# Each is pinned to what sglang 0.5.6 expects; no version to let pip pick
# the latest compatible with torch 2.11.
RUN pip install --no-cache-dir --break-system-packages --no-deps \
      tqdm fastapi 'uvicorn[standard]' psutil orjson pydantic pydantic_core \
      pyzmq interegular lark outlines xgrammar requests aiohttp httpx hf-xet \
      urllib3 charset-normalizer idna certifi \
      transformers tokenizers safetensors huggingface_hub datasets blobfile \
      modelscope einops scipy python-multipart uvloop cloudpickle prometheus-client \
      anthropic openai litellm numba compressed-tensors msgpack-numpy msgspec msgpack \
      dill multiprocess llguidance partial_json_parser pybase64 \
      soundfile sentencepiece torchao torch_memory_saver \
      ninja psutil packaging filelock networkx \
      starlette sniffio anyio h11 httptools websockets \
      watchfiles python-dotenv click colorama typing_inspect \
      'email-validator' annotated-types nest-asyncio \
      IPython

# NOTE: no nvidia-nccl-cu13 or nvidia-nvshmem-cu13 guard needed. The
# --no-deps sglang install never touched them; the base's cu13 stack
# is untouched.

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
