#!/usr/bin/env bash
# cajeta-llm plan 15.1.2/15.1.3 — build and run the ParityRun instrument
# (dev.cajeta.llm.bench.ParityRun) against the f32-converted reference
# checkpoint and the transformers fp32 fixture. Release + bounded live-set:
# the shipping configuration (§12.10's compiler-mode axis).
#
#   CAJETA=…/build/src/cajeta tools/parity/run-parity.sh
#
# Prereqs on this box: tools/parity/fixtures/llama31-8b (gen_fixture.py),
# /home/julian/models/Meta-Llama-3.1-8B-Instruct-f32 (convert_f32.py),
# sibling checkouts of cajeta-codec and cajeta-jinja (built .cja archives).
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
# The library has not done its ownership-migration pass (run-tests.sh's
# standing note) — warn, don't reject, when compiling it from source.
export CAJETA_OWNED_BIND="${CAJETA_OWNED_BIND:-warn}"
export CAJETA_CAPTURED_BORROW="${CAJETA_CAPTURED_BORROW:-warn}"
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

CODEC_CJA="${CODEC_CJA:-$(cajeta_artifact_path "$here"/../cajeta-codec dev.cajeta.codec)}"
JINJA_CJA="${JINJA_CJA:-$(cajeta_artifact_path "$here"/../cajeta-jinja dev.cajeta.jinja)}"
out="$here/tmp/parity-build"
mkdir -p "$out"

echo ">> building parity runner (release, bounded live-set)"
"$CAJETA" --emit=exe --release --live-set=bounded --xpu-backend=cpu \
    --classpath="$CODEC_CJA,$JINJA_CJA" \
    -o "$out/parity" \
    dev.cajeta.llm.bench.ParityRun.run "$here/src/main/cajeta" "$out" >/dev/null

echo ">> running (8B f32 on host — expect minutes)"
cd "$here"
exec "$out/parity"
