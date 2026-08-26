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
MODEL="${MODEL:-$HOME/models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
CODEC_CJA="${CODEC_CJA:-$(ls -t "$here"/../cajeta-codec/build/archive/dev.cajeta.codec-*.cja | head -1)}"
JINJA_CJA="${JINJA_CJA:-$(ls -t "$here"/../cajeta-jinja/build/archive/dev.cajeta.jinja-*.cja | head -1)}"
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
