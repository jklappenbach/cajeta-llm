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
CODEC_CJA="${CODEC_CJA:-$(ls -t "$here"/../cajeta-codec/build/archive/dev.cajeta.codec-*.cja | head -1)}"
JINJA_CJA="${JINJA_CJA:-$(ls -t "$here"/../cajeta-jinja/build/archive/dev.cajeta.jinja-*.cja | head -1)}"
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
