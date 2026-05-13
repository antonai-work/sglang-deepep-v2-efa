"""Wave 38.9: force ep_moe.layer.EPMoE.deprecate_flag=True for all FP8 configs.

Usage: python3 patch_ep_moe_deprecate_flag.py <path-to-ep_moe/layer.py>

Root cause: ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm() runs at import
time. If libcuda.so.1 isn't yet linkable (pre-driver-injection), `import
deep_gemm` fails → ENABLE_JIT_DEEPGEMM caches False → deprecate_flag=False
→ hits "forward_deepgemm_contiguous is deprecated" assert.

Fix: make deprecate_flag=True whenever quant_config is Fp8Config, regardless
of the cached ENABLE_JIT_DEEPGEMM value. This routes through the modern
super().run_moe_core path which uses flashinfer / cutedsl / triton MoE
runners that don't require deep_gemm.
"""
import sys

SENTINEL = "# [wave38.9 deprecate_flag force-on for FP8]"

OLD = (
    "        if _use_aiter or _is_npu:\n"
    "            self.deprecate_flag = False\n"
    "        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(\n"
    "            quant_config, Fp8Config\n"
    "        ):\n"
    "            self.deprecate_flag = True\n"
    "        else:\n"
    "            self.deprecate_flag = False"
)

NEW = (
    "        " + SENTINEL + "\n"
    "        if _use_aiter or _is_npu:\n"
    "            self.deprecate_flag = False\n"
    "        elif isinstance(quant_config, Fp8Config):\n"
    "            # Force modern super().run_moe_core path even if\n"
    "            # ENABLE_JIT_DEEPGEMM evaluated False at import time\n"
    "            # (pre-driver-injection, libcuda.so.1 not linkable).\n"
    "            self.deprecate_flag = True\n"
    "        else:\n"
    "            self.deprecate_flag = False"
)


def main():
    if len(sys.argv) != 2:
        print("usage: patch_ep_moe_deprecate_flag.py <path>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    src = open(path).read()
    if SENTINEL in src:
        print(f"[wave38.9] already patched: {path}")
        return
    if OLD not in src:
        print(f"[wave38.9] PATCH TARGET NOT FOUND in {path}", file=sys.stderr)
        sys.exit(1)
    open(path, "w").write(src.replace(OLD, NEW, 1))
    print(f"[wave38.9] patched {path}: deprecate_flag=True for Fp8Config")


if __name__ == "__main__":
    main()
