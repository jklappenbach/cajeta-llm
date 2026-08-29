#!/usr/bin/env bash
# cajeta-llm plan 15.1.4 — kernel-level two-backend agreement on REAL
# silicon (spec 12.3/13.21). Unit 7.3.1 bounded the device paths to the
# in-process CPU backend and deferred hardware runs to the Unit 15 gate,
# "which runs on real silicon by definition"; this is that harness.
#
# It builds dev.cajeta.llm.bench.GpuParity with the GPU backend ONLY —
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

# --- artifact discovery -------------------------------------------------
# Where a checkout's .cja is. Prefers `cajeta artifact-path`, which reads
# that project's OWN manifest -- so a project that moves its artifacts with
# settings.output is followed rather than guessed, and the version comes
# from details.version instead of whichever file happens to be newest.
#
# Falls back to the historical build/archive glob only when the toolchain
# does not HAVE the verb (it lands after 0.24.0), so this keeps working on
# an older cajeta and starts using the verb as soon as a newer one is on
# PATH -- no flag day.
#
# The gate is the CAPABILITY, not the outcome. A fallback keyed on "the
# verb failed" would silently mask a verb that ran and answered wrongly,
# which is the very failure this replaces; keyed on "the verb is absent",
# it cannot. An empty result still means "not in this checkout", exactly
# as the glob did, so callers' registry fallbacks are unchanged.
cajeta_artifact_path() {
    local dir="$1" name="$2"
    local cj="${CAJETA:-${CAJETA_BIN:-cajeta}}"
    if [[ -z "${_cajeta_has_ap:-}" ]]; then
        if "$cj" artifact-path --help 2>/dev/null \
                | grep -q 'artifact-path \[options\]'; then
            _cajeta_has_ap=yes
        else
            _cajeta_has_ap=no
        fi
    fi
    if [[ "$_cajeta_has_ap" == yes ]]; then
        # Only report a path that EXISTS. The verb answers where the
        # artifact would be even when nothing has built it, but the glob
        # this replaces returned empty in that case, and every caller
        # reads empty as "not in this checkout" and falls back to the
        # registry. Handing back a path to a missing file instead would
        # turn that into a confusing compile failure.
        local p
        p=$( cd "$dir" 2>/dev/null && "$cj" artifact-path 2>/dev/null ) || return 0
        [[ -n "$p" && -f "$p" ]] && printf '%s\n' "$p"
        return 0
    else
        ls -t "$dir"/build/archive/"$name"-*.cja 2>/dev/null | head -1
    fi
}

XPU="${XPU:-amdgpu}"
CODEC_CJA="${CODEC_CJA:-$(cajeta_artifact_path "$here"/../cajeta-codec dev.cajeta.codec)}"
JINJA_CJA="${JINJA_CJA:-$(cajeta_artifact_path "$here"/../cajeta-jinja dev.cajeta.jinja)}"
out="$here/tmp/gpu-parity"
mkdir -p "$out"

echo ">> building GpuParity for --xpu-backend=$XPU (no cpu fallback)"
CAJETA_OWNED_BIND=warn CAJETA_CAPTURED_BORROW=warn \
"$CAJETA" --emit=exe --xpu-backend="$XPU" \
    --classpath="$CODEC_CJA,$JINJA_CJA" \
    -o "$out/gpuparity" \
    dev.cajeta.llm.bench.GpuParity.run "$here/src/main/cajeta" "$out" >/dev/null

echo ">> running on device"
cd "$here"
exec "$out/gpuparity"
