#!/usr/bin/env bash
# cajeta-llama plan 15.1.4 — kernel-level two-backend agreement on REAL
# silicon (spec 12.3/13.21). Unit 7.3.1 bounded the device paths to the
# in-process CPU backend and deferred hardware runs to the Unit 15 gate,
# "which runs on real silicon by definition"; this is that harness.
#
# It builds dev.cajeta.llama.bench.GpuParity with the GPU backend ONLY —
# no `cpu` in --xpu-backend — so a device launch cannot quietly fall back
# to a CPU kernel. A run that reports `result onGpu: yes` therefore proves
# the GEMM executed on the device, which a green portable suite does not:
# Ewise.matmulF32Op routes on Placement.pair and has a host scalar
# fallback, so host-side execution looks identical from the outside.
#
#   tools/gpu/run-gpu-parity.sh                 # amdgpu (default)
#   XPU=nvptx tools/gpu/run-gpu-parity.sh       # NVIDIA
#   XPU=vulkan tools/gpu/run-gpu-parity.sh      # SPIR-V
#
# No CAJETA_GPU_COOPMATRIX_IMPL override: the straddling-tier defect that
# made it necessary is fixed (cajeta, 2026-08-21) — a kernel whose tiles
# straddle tiers now demotes to the portable tile as a GROUP and lowers,
# instead of being skipped. Set the variable by hand only to force the
# portable path on a backend that would otherwise go native.
set -euo pipefail

here="$(cd "$(dirname "$0")/../.." && pwd)"
CAJETA="${CAJETA:-cajeta}"
XPU="${XPU:-amdgpu}"
CODEC_CJA="${CODEC_CJA:-$(ls -t "$here"/../cajeta-codec/build/archive/dev.cajeta.codec-*.cja | head -1)}"
JINJA_CJA="${JINJA_CJA:-$(ls -t "$here"/../cajeta-jinja/build/archive/dev.cajeta.jinja-*.cja | head -1)}"
out="$here/tmp/gpu-parity"
mkdir -p "$out"

echo ">> building GpuParity for --xpu-backend=$XPU (no cpu fallback)"
CAJETA_OWNED_BIND=warn CAJETA_CAPTURED_BORROW=warn \
"$CAJETA" --emit=exe --xpu-backend="$XPU" \
    --classpath="$CODEC_CJA,$JINJA_CJA" \
    -o "$out/gpuparity" \
    dev.cajeta.llama.bench.GpuParity.run "$here/src/main/cajeta" "$out" >/dev/null

echo ">> running on device"
cd "$here"
exec "$out/gpuparity"
