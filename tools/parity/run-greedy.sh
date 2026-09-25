#!/usr/bin/env bash
# Greedy token agreement against llama.cpp on the reference model
# (cajeta xpu-kernel-adaptor plan 8.1.3): llama.cpp's walk with top-2 gaps
# from tools/parity/llama-greedy, cajeta's walk on the SAME prompt ids from
# bench/GreedyVsLlama, judged by GreedyAgreement (rate >= 0.99, every
# divergence at a gap under 1e-3).
#
#   XPU_BACKEND=nvptx DEVICE=cuda tools/parity/run-greedy.sh [prompt] [n]
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
source "$here/tools/parity/llama-env.sh"
CAJETA="${CAJETA:-$here/../cajeta/build/src/cajeta}"
XPU_BACKEND="${XPU_BACKEND:-nvptx}"
DEVICE="${DEVICE:-cuda}"
prompt="${1:-The measured baseline for the parity gate is}"
n="${2:-32}"
out="$here/tmp/parity-build"
mkdir -p "$out"
[[ -x "$out/llama-greedy" ]] || "$here/tools/parity/build-ggml-leg.sh"
[[ -f "$MODEL" ]] || { echo "no reference model at $MODEL" >&2; exit 1; }

echo ">> llama.cpp greedy ($LLAMA_COMMIT, CUDA): $n tokens"
"$out/llama-greedy" --model "$MODEL" --prompt "$prompt" --n "$n" > "$out/llama-greedy.tsv"
grep -c '^greedy' "$out/llama-greedy.tsv" | sed 's/^/   tokens: /'

olla="${OLLA_HOME:-$HOME/.olla}"
ver() { sed -n "s/.*\"dev\.cajeta\.$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$here/cajeta.json" | head -1; }
codec_cja="$olla/dev.cajeta.codec/$(ver codec)/dev.cajeta.codec-$(ver codec).cja"
jinja_cja="$olla/dev.cajeta.jinja/$(ver jinja)/dev.cajeta.jinja-$(ver jinja).cja"
logging_cja="$olla/dev.cajeta.logging/$(ver logging)/dev.cajeta.logging-$(ver logging).cja"
export CAJETA_OWNED_BIND="${CAJETA_OWNED_BIND:-warn}"
export CAJETA_CAPTURED_BORROW="${CAJETA_CAPTURED_BORROW:-warn}"
if [[ ! -x "$out/greedy-$XPU_BACKEND" || "${REBUILD:-0}" == 1 ]]; then
    echo ">> building GreedyVsLlama ($XPU_BACKEND)"
    "$CAJETA" --emit=exe --release --live-set=bounded --xpu-backend="$XPU_BACKEND" \
        --classpath="$codec_cja,$jinja_cja,$logging_cja" \
        -o "$out/greedy-$XPU_BACKEND" \
        dev.cajeta.llm.bench.GreedyVsLlama.run "$here/src/main/cajeta" "$out" \
        2> "$out/greedy-$XPU_BACKEND.err" >/dev/null \
        || { cat "$out/greedy-$XPU_BACKEND.err" >&2; exit 1; }
fi
echo ">> cajeta greedy ($DEVICE)"
cd "$here"
"$out/greedy-$XPU_BACKEND" --model "$MODEL" --ref "$out/llama-greedy.tsv" \
    --device "$DEVICE" --n "$n" | tee "$out/greedy-$XPU_BACKEND.out"
