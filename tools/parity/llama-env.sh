# Sourced by the parity scripts: where llama.cpp is, and how to run it.
#
# THE WSL DRIVER TRAP (2026-09-24). On a WSL2 box with the distro
# `libnvidia-compute-580` package installed, every llama.cpp binary that
# initialises the CUDA backend SIGSEGVs inside
# `/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1` (580.178.04): the
# Windows driver updated to 610 on 2026-09-20 and its libcuda dlopens the
# PTX JIT library by soname, finds the stale distro copy first, and calls a
# mismatched entry point. The driver ships its own matching copy under
# /usr/lib/wsl/drivers/nv_dispsi.inf_amd64_*/, so that directory goes FIRST
# on LD_LIBRARY_PATH. cajeta is unaffected: it assembles PTX with ptxas and
# never loads the JIT library. `--help` on llama-bench crashes the same
# way, since usage text enumerates devices.

LLAMA_ROOT="${LLAMA_ROOT:-$HOME/code/cpp/llama.cpp}"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-$LLAMA_ROOT/build-cuda}"
LLAMA_BIN="$LLAMA_BUILD_DIR/bin"
LLAMA_COMMIT="$(git -C "$LLAMA_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

wsl_driver_dir="$(ls -d /usr/lib/wsl/drivers/nv_dispsi.inf_amd64_* 2>/dev/null | head -1)"
if [[ -n "$wsl_driver_dir" ]]; then
    export LD_LIBRARY_PATH="$wsl_driver_dir:$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

CUDA_HOME="${CUDA_PATH:-/usr/local/cuda}"
MODEL="${MODEL:-$HOME/models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
