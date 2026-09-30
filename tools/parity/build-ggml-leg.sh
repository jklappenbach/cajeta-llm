#!/usr/bin/env bash
# Builds the two llama.cpp-side harnesses against the llama.cpp build beside
# this checkout (plan 8.1.1, 8.1.3), CUDA or HIP by which backend it carries:
#   tmp/parity-build/ggml-leg       per-op device-timed MUL_MAT leg
#   tmp/parity-build/llama-greedy   greedy token ids + top-2 gaps
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
source "$here/tools/parity/llama-env.sh"
out="$here/tmp/parity-build"
mkdir -p "$out"
inc=(-I"$LLAMA_ROOT/ggml/include" -I"$LLAMA_ROOT/ggml/src" -I"$LLAMA_ROOT/include")
if [[ -f "$LLAMA_BIN/libggml-cuda.so" ]]; then
    inc+=(-I"$CUDA_HOME/include")
    libs=(-L"$LLAMA_BIN" -lggml -lggml-base -L"$CUDA_HOME/lib64" -lcudart -Wl,-rpath,"$LLAMA_BIN" -Wl,-rpath,"$CUDA_HOME/lib64")
    defs=()
elif [[ -f "$LLAMA_BIN/libggml-hip.so" ]]; then
    rocm="${ROCM_PATH:-$(hipconfig -R 2>/dev/null || echo /opt/rocm)}"
    inc+=(-I"$rocm/include")
    libs=(-L"$LLAMA_BIN" -lggml -lggml-base -L"$rocm/lib" -lamdhip64 -Wl,-rpath,"$LLAMA_BIN" -Wl,-rpath,"$rocm/lib")
    defs=(-DGGML_LEG_HIP -D__HIP_PLATFORM_AMD__)
else
    echo "no libggml-cuda.so or libggml-hip.so in $LLAMA_BIN (build llama.cpp with -DGGML_CUDA=ON or -DGGML_HIP=ON first)" >&2
    exit 1
fi
echo ">> ggml-leg (llama.cpp $LLAMA_COMMIT, ${defs[*]:-CUDA})"
g++ -O2 -std=c++17 "${defs[@]}" "${inc[@]}" -o "$out/ggml-leg" "$here/tools/parity/ggml-leg.cpp" "${libs[@]}"
echo ">> llama-greedy"
g++ -O2 -std=c++17 "${defs[@]}" "${inc[@]}" -o "$out/llama-greedy" "$here/tools/parity/llama-greedy.cpp" -L"$LLAMA_BIN" -lllama "${libs[@]}"
echo ">> built into $out"
