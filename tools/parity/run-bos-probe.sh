#!/usr/bin/env bash
# cajeta-llm plan 18.3.2 — the real-model acceptance for the GGUF rope
# un-permute.
#
# Greedy-decodes the RAW 6-token prompt "<|begin_of_text|>The capital of France
# is" (128000 791 6864 315 9822 374) on the REAL Llama-3.1-8B-Instruct Q4_K_M
# and prints the engine's LOGITS at the reference's top-5 ids.
#
# READ THIS BEFORE TREATING A TOKEN AS A DEFECT. The correct answer for these
# ids is ' a' (264), NOT ' Paris' -- transformers fp32 puts ' a' at +16.0104 and
# ' Paris' at +15.9078. `llama-completion -p "The capital of France is"` looks
# like it disagrees only because it applies the Instruct CHAT TEMPLATE and
# evaluates 15 tokens, a different prompt entirely. Assuming otherwise cost a
# full day on 2026-08-21.
#
# Compare LOGITS, never the token. The GGUF rope un-permute moves the logits by
# ~1.5 and fixes their ORDER (without it ' the' outranks ' Paris'), yet the
# argmax is 264 either way -- a token-level check cannot see it.
#
# Pass `onestep` to feed the prompt one token at a time (stepSeq n=1) instead of
# a prefill chunk; spec 13.14 makes them one forward path, so they must agree.
# Pass `nopack` to run the same checkpoint dequantized.
#
# Env: CAJETA (compiler; needs the ownership fixes on main), MODEL (gguf path).
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

MODEL="${MODEL:-$HOME/models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
CODEC_CJA="${CODEC_CJA:-$(cajeta_artifact_path "$here"/../cajeta-codec dev.cajeta.codec)}"
JINJA_CJA="${JINJA_CJA:-$(cajeta_artifact_path "$here"/../cajeta-jinja dev.cajeta.jinja)}"
out="$here/tmp/bosprobe"
mkdir -p "$out"
export TMPDIR="$here/tmp"

echo ">> building BosProbe"
CAJETA_OWNED_BIND=warn CAJETA_CAPTURED_BORROW=warn \
"$CAJETA" --emit=exe --release \
    --classpath="$CODEC_CJA,$JINJA_CJA" \
    -o "$out/bosprobe" \
    dev.cajeta.llm.bench.BosProbe.run "$here/src/main/cajeta" "$out" >/dev/null

echo ">> $MODEL"
cd "$here"
exec "$out/bosprobe" "$MODEL" "$@"
