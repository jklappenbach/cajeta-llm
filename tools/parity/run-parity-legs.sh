#!/usr/bin/env bash
# The parity harness (cajeta xpu-kernel-adaptor plan Unit 8): both legs at
# the reference model's decode shapes, under the same residency, on the
# same timer tier, joined into census/parity-<backend>.tsv by ParityTable.
#
#   XPU_BACKEND=nvptx tools/parity/run-parity-legs.sh
#
# Steps:
#   1. build the llama.cpp-side harnesses (build-ggml-leg.sh);
#   2. the CALIBRATION row: ggml-leg at test-backend-ops' hot shape
#      (q4_K 4096x14336, one tensor) against the recorded 13330 ns
#      (plan 1.8, 2026-09-19; 12940 ns on 2026-09-24), within 5%;
#   3. ggml-leg cold at the seven engine shapes;
#   4. the cajeta leg (bench/ParityLeg) on the requested backend;
#   5. ParityJoin reads both, refuses what §7.1 refuses, renders the table.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
source "$here/tools/parity/llama-env.sh"
CAJETA="${CAJETA:-$here/../cajeta/build/src/cajeta}"
XPU_BACKEND="${XPU_BACKEND:-nvptx}"
out="$here/tmp/parity-build"
mkdir -p "$out"
llm_commit="$(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo unknown)"

"$here/tools/parity/build-ggml-leg.sh"

leg() { # type m k copies iters  (synthetic, for the hot calibration row)
    "$out/ggml-leg" --ggml-lib-dir "$LLAMA_BIN" --type "$1" --m "$2" --k "$3" --n 1 \
        --copies "$4" --iters "$5" --build "$LLAMA_COMMIT" --fill pattern
}
real() { # type m k tensor-format layers iters  (real tensors from the reference GGUF)
    "$out/ggml-leg" --ggml-lib-dir "$LLAMA_BIN" --type "$1" --m "$2" --k "$3" --n 1 \
        --iters "$6" --build "$LLAMA_COMMIT" --gguf "$MODEL" --tensor "$4" --layers "$5"
}
reallist() { # type m k tensor-list iters
    "$out/ggml-leg" --ggml-lib-dir "$LLAMA_BIN" --type "$1" --m "$2" --k "$3" --n 1 \
        --iters "$5" --build "$LLAMA_COMMIT" --gguf "$MODEL" --tensor-list "$4"
}
# The layers whose ffn_down / attn_v are q4_K and q6_K in Q4_K_M (read
# from the GGUF 2026-09-24); the other tensors are q4_K in every layer.
Q4_LAYERS=4,5,7,8,9,11,12,14,15,17,18,20,22,23,25,26
Q6_LAYERS=0,1,2,3,6,10,13,16,19,21,24,27,28,29,30,31
ALL16=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

echo ">> llama.cpp leg ($LLAMA_COMMIT, CUDA), real tensors from $MODEL"
[[ -f "$MODEL" ]] || { echo "no reference model at $MODEL" >&2; exit 1; }
rows="$out/llama-rows.tsv"
{
    # calibration: the hot test-backend-ops shape, one tensor, synthetic
    leg q4_K 4096 14336 1 20
    # the engine's seven, cold, real weights. 32 x 2.25 MB is exactly the
    # 4090's L2, so the q4_K k/v pool is every layer's k plus the q4_K v's:
    # 48 tensors, 108 MB.
    kv=""; for l in $(seq 0 31); do kv="$kv,blk.$l.attn_k.weight"; done
    for l in ${Q4_LAYERS//,/ }; do kv="$kv,blk.$l.attn_v.weight"; done
    reallist q4_K  1024  4096 "${kv#,}" 5
    real q6_K  1024  4096 'blk.%d.attn_v.weight'   "$Q6_LAYERS" 5
    real q4_K  4096  4096 'blk.%d.attn_q.weight'   "$ALL16" 5
    real q4_K 14336  4096 'blk.%d.ffn_gate.weight' "0,1,2,3,4,5,6" 5
    real q4_K  4096 14336 'blk.%d.ffn_down.weight' "4,5,7,8,9,11,12" 5
    real q6_K  4096 14336 'blk.%d.ffn_down.weight' "0,1,2,3,6,10,13" 5
    real q6_K 128256 4096 'output.weight' "0" 5
} | tee "$rows"

echo ">> cajeta leg ($llm_commit, $XPU_BACKEND)"
# The dependency archives, the way run-tests.sh resolves them.
olla="${OLLA_HOME:-$HOME/.olla}"
ver() { sed -n "s/.*\"dev\.cajeta\.$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$here/cajeta.json" | head -1; }
codec_cja="$olla/dev.cajeta.codec/$(ver codec)/dev.cajeta.codec-$(ver codec).cja"
jinja_cja="$olla/dev.cajeta.jinja/$(ver jinja)/dev.cajeta.jinja-$(ver jinja).cja"
logging_cja="$olla/dev.cajeta.logging/$(ver logging)/dev.cajeta.logging-$(ver logging).cja"
for f in "$codec_cja" "$jinja_cja" "$logging_cja"; do [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }; done
export CAJETA_OWNED_BIND="${CAJETA_OWNED_BIND:-warn}"
export CAJETA_CAPTURED_BORROW="${CAJETA_CAPTURED_BORROW:-warn}"
if [[ ! -x "$out/parity-leg-$XPU_BACKEND" || "${REBUILD:-0}" == 1 ]]; then
    "$CAJETA" --emit=exe --release --live-set=bounded --xpu-backend="$XPU_BACKEND" \
        --classpath="$codec_cja,$jinja_cja,$logging_cja" \
        -o "$out/parity-leg-$XPU_BACKEND" \
        dev.cajeta.llm.bench.ParityLeg.run "$here/src/main/cajeta" "$out" \
        2> "$out/parity-leg-$XPU_BACKEND.err" >/dev/null \
        || { cat "$out/parity-leg-$XPU_BACKEND.err" >&2; exit 1; }
    "$CAJETA" --emit=exe --release --live-set=bounded --xpu-backend="$XPU_BACKEND" \
        --classpath="$codec_cja,$jinja_cja,$logging_cja" \
        -o "$out/parity-join" \
        dev.cajeta.llm.bench.ParityJoin.run "$here/src/main/cajeta" "$out" \
        2> "$out/parity-join.err" >/dev/null \
        || { cat "$out/parity-join.err" >&2; exit 1; }
fi
crows="$out/cajeta-rows-$XPU_BACKEND.tsv"
( cd "$here" && "$out/parity-leg-$XPU_BACKEND" --build "$llm_commit" --gguf "$MODEL" ) | tee "$crows"

echo ">> the table"
table="$here/census/parity-$XPU_BACKEND.tsv"
( cd "$here" && "$out/parity-join" --llama "$rows" --cajeta "$crows" \
    --recorded-hot-ns 13330 --tolerance 0.05 ) | tee "$table"
echo ">> wrote $table"
