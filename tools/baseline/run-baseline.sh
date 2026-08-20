#!/usr/bin/env bash
# cajeta-llama plan 15.2.1 / 15.1.1 — the llama.cpp baseline at the spec
# 13.20 reference configuration: Llama-3.1-8B-Instruct Q4_K_M, gfx1151 /
# ROCm, 4096-token context. Every fraction in §12.8/12.9/12.11 divides by
# the numbers this script records; they are meaningless against an
# unmeasured denominator.
#
# Measures:
#   1. batch 1: prefill tok/s at pp4096 and decode tok/s at tg128
#      (llama-bench, 3 repetitions, -ngl 99).
#   2. a representative batched shape: 4 parallel sequences, 512-token
#      prompts, 128 generated each (llama-batched-bench).
#   3. resident memory: peak RSS of a 4096-ctx llama-cli generation
#      (/usr/bin/time -v), plus the model/KV buffer sizes llama.cpp logs.
#
# Emits tools/baseline/results/BASELINE-<date>.md with every §12.10
# identity field, and leaves the raw logs beside it.
set -euo pipefail

MODEL="${MODEL:-$HOME/models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/code/llama.cpp/build/bin}"
CTX="${CTX:-4096}"
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/results"
mkdir -p "$out"
stamp="$(date +%Y%m%d)"
md="$out/BASELINE-$stamp.md"

commit="$(git -C "$(dirname "$LLAMA_BIN")/.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
rocm="$(hipconfig --version 2>/dev/null || echo unknown)"
gpu="$(rocminfo 2>/dev/null | grep -m1 'Marketing Name' | sed 's/.*: *//' || true)"
[ -n "$gpu" ] || gpu="$(rocminfo 2>/dev/null | grep -m1 'gfx' | awk '{print $2}')"

echo ">> llama-bench (batch 1: pp$CTX + tg128, 3 reps)"
"$LLAMA_BIN/llama-bench" -m "$MODEL" -p "$CTX" -n 128 -ngl 99 -r 3 \
    -o md 2> "$out/bench-$stamp.err" | tee "$out/bench-$stamp.md"

echo ">> llama-batched-bench (4 seqs x 512 prompt + 128 gen)"
"$LLAMA_BIN/llama-batched-bench" -m "$MODEL" -c "$CTX" -b 2048 -ub 512 \
    -ngl 99 -npp 512 -ntg 128 -npl 4 \
    > "$out/batched-$stamp.txt" 2>&1 || true
tail -20 "$out/batched-$stamp.txt"

echo ">> resident memory (llama-completion, ctx $CTX, 32 tokens)"
# llama-completion, NOT llama-cli: this llama.cpp build's cli is
# conversation-only — it rejects -no-cnv with a note pointing here, then
# spins its loader animation into the redirect unboundedly (the first two
# runs of this script wrote 36 GB and 12 GB of spinner before being
# killed). llama-completion is the single-shot binary; -n terminates it.
# -st (--single-turn): with a predefined -p prompt this exits after one
# completion instead of waiting on stdin (the interactive banner path).
/usr/bin/time -v "$LLAMA_BIN/llama-completion" -m "$MODEL" -c "$CTX" -n 32 \
    -ngl 99 -st \
    -p "The measured baseline for the parity gate is" \
    < /dev/null > "$out/cli-$stamp.out" 2> "$out/cli-$stamp.err" || true
rss_kb="$(grep 'Maximum resident set size' "$out/cli-$stamp.err" | grep -o '[0-9]*' || echo 0)"
grep -E "buffer size|KV self size|Peak|resident" "$out/cli-$stamp.err" | head -10 || true

{
    echo "# llama.cpp baseline — spec 13.20 reference configuration (plan 15.1.1)"
    echo
    echo "Recorded: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "## §12.10 identity"
    echo
    echo "- model: $(basename "$MODEL")"
    echo "- model sha256: $(sha256sum "$MODEL" | cut -d' ' -f1)"
    echo "- quantization: Q4_K_M"
    echo "- backend: llama.cpp $commit, ROCm (hip $rocm), device: $gpu"
    echo "- contextLength: $CTX"
    echo "- batchSize: 1 (llama-bench) and 4-parallel (llama-batched-bench)"
    echo "- compilerMode / liveSet: n/a — this is the DENOMINATOR engine"
    echo "- host: $(uname -sr), $(nproc) cores"
    echo
    echo "## batch 1 (llama-bench, r=3)"
    echo
    cat "$out/bench-$stamp.md"
    echo
    echo "## batched shape (llama-batched-bench: 4 x pp512 + tg128)"
    echo
    echo '```'
    tail -12 "$out/batched-$stamp.txt"
    echo '```'
    echo
    echo "## resident memory (llama-cli, ctx $CTX)"
    echo
    echo "- Maximum resident set size: ${rss_kb} kB"
    echo '```'
    grep -E "buffer size|KV self size" "$out/cli-$stamp.err" | head -8 || true
    echo '```'
} > "$md"
echo ">> wrote $md"
