# sglang-deepep-v2-efa

SGLang + DeepEP V2 + AWS EFA MoE inference cascade.

**Status:** SCAFFOLD. The v0.2.5 base child has not yet been
revalidated against SGLang. Wave 29 in the private `deepep-v2-integration`
dev tree tracks that revalidation; this repo lands the cascade-symmetric
source shape so CI/CodeBuild pipelines can hydrate as soon as Wave 29
produces PROVEN evidence.

## Image cascade

```
nvidia/cuda:13.0.0-devel-ubuntu24.04                   (public)
  └── deepep-v2-efa-base:v0.2.5-sm90a                  (base: NCCL patched, EFA, DeepEP V2)
        └── sglang-deepep-v2-efa:<tag>                 (this repo, Wave 29 target)
```

Parent base: https://github.com/antonai-work/deepep-v2-efa-base

## Sibling repos (reproducibility cascade)

| Repo | Purpose | Status |
|---|---|---|
| [deepep-v2-efa-base](https://github.com/antonai-work/deepep-v2-efa-base) | Base substrate (EFA + NCCL + DeepEP V2) | v0.2.5-sm90a released |
| [vllm-deepep-v2-efa](https://github.com/antonai-work/vllm-deepep-v2-efa) | vLLM inference stack | Wave 26b PROVEN |
| [megatron-deepep-v2-efa](https://github.com/antonai-work/megatron-deepep-v2-efa) | Megatron-LM training | Wave 27 PROVEN |
| [nemo-rl-deepep-v2-efa](https://github.com/antonai-work/nemo-rl-deepep-v2-efa) | NeMo-RL + Megatron full-stack | Wave 28 PROVEN |
| **sglang-deepep-v2-efa** | **SGLang inference (this repo)** | **Wave 29 pending** |
| [trtllm-deepep-v2-efa](https://github.com/antonai-work/trtllm-deepep-v2-efa) | TRT-LLM inference | Wave 30 pending |

## Upstream PRs consumed

| Upstream | PR | HEAD SHA | Status |
|---|---|---|---|
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | [#24443](https://github.com/sgl-project/sglang/pull/24443) | `f66ab63b` | DRAFT |
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | [#612](https://github.com/deepseek-ai/DeepEP/pull/612) | `146cc356` | OPEN (baked into base) |

Full pinned versions in [`pins.env`](pins.env).

## Build

```bash
docker build -f docker/Dockerfile --build-arg BUILD_MODE=fast \
             -t sglang-deepep-v2-efa:fast .
```

Image will be rebuilt by CodeBuild once Wave 29 lands the SGLang
overlay body.

## Licensing

MIT. SGLang itself is under Apache-2.0. DeepEP under MIT (DeepSeek).
