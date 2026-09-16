#!/usr/bin/env bash
# Build (when stale) and run the HF -> GGUF converter,
# dev.cajeta.llm.convert.HfToGguf (codebook-quants spec §8.6):
#   tools/convert/hf-to-gguf.sh <checkpoint dir> <out.gguf> f16|tq1_0|tq2_0
# CAJETA names the compiler; DEPS the classpath archives (codec, jinja,
# logging) when the sibling checkouts are not where run-tests.sh looks.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
CAJETA="${CAJETA:-cajeta}"
out="$here/tmp/tools"; mkdir -p "$out"
export TMPDIR="${TMPDIR:-$here/tmp/tmpdir}"; mkdir -p "$TMPDIR"
export CAJETA_OWNED_BIND="${CAJETA_OWNED_BIND:-warn}"
export CAJETA_CAPTURED_BORROW="${CAJETA_CAPTURED_BORROW:-warn}"
if [[ -z "${DEPS:-}" ]]; then
    DEPS=""
    for repo in cajeta-codec cajeta-jinja cajeta-logging; do
        cja="$(ls -t "$here/../$repo"/build/archive/*.cja 2>/dev/null | head -1)"
        [[ -n "$cja" ]] || { echo "hf-to-gguf: no archive under ../$repo/build/archive; set DEPS" >&2; exit 1; }
        DEPS="${DEPS:+$DEPS,}$cja"
    done
fi
bin="$out/hf-to-gguf"
newest="$(find "$here/src/main/cajeta" -name '*.cajeta' -newer "$bin" 2>/dev/null | head -1 || true)"
if [[ ! -x "$bin" || -n "$newest" ]]; then
    echo ">> building $bin"
    "$CAJETA" --emit=exe --classpath="$DEPS" -o "$bin" \
        dev.cajeta.llm.convert.HfToGguf.run "$here/src/main/cajeta" "$out" \
        2> >(grep -v 'mma-tiering\|xpu-kernel-spill\|xpu-kernel-skipped' >&2 || true)
fi
exec "$bin" "$@"
