#!/usr/bin/env bash
# Builds the two llama.cpp-side harnesses against the CUDA build of
# llama.cpp beside this checkout (plan 8.1.1, 8.1.3):
#   tmp/parity-build/ggml-leg       per-op device-timed MUL_MAT leg
#   tmp/parity-build/llama-greedy   greedy token ids + top-2 gaps
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
source "$here/tools/parity/llama-env.sh"
out="$here/tmp/parity-build"
mkdir -p "$out"
[[ -f "$LLAMA_BIN/libggml-cuda.so" ]] || { echo "no libggml-cuda.so in $LLAMA_BIN (build llama.cpp with -DGGML_CUDA=ON first)" >&2; exit 1; }
inc=(-I"$LLAMA_ROOT/ggml/include" -I"$LLAMA_ROOT/ggml/src" -I"$LLAMA_ROOT/include" -I"$CUDA_HOME/include")
libs=(-L"$LLAMA_BIN" -lggml -lggml-base -L"$CUDA_HOME/lib64" -lcudart -Wl,-rpath,"$LLAMA_BIN" -Wl,-rpath,"$CUDA_HOME/lib64")
echo ">> ggml-leg (llama.cpp $LLAMA_COMMIT)"
g++ -O2 -std=c++17 "${inc[@]}" -o "$out/ggml-leg" "$here/tools/parity/ggml-leg.cpp" "${libs[@]}"
echo ">> llama-greedy"
g++ -O2 -std=c++17 "${inc[@]}" -o "$out/llama-greedy" "$here/tools/parity/llama-greedy.cpp" -L"$LLAMA_BIN" -lllama "${libs[@]}"
echo ">> built into $out"
