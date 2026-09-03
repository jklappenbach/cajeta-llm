# mxfp4-extract — reference-fixture generator (Unit 32, plan 6.2.1)

A one-time tool that produces `src/test/fixtures/mxfp4/mxfp4_ref.bin`
from a real MXFP4 GGUF, using **llama.cpp's own `dequantize_row_mxfp4`**
as the numeric oracle. It is C++ (not cajeta) on purpose: it links
llama.cpp's `ggml` to capture the reference values the witness compares
against. The fixture is checked in, so the test needs no llama.cpp at
run time (spec §6.2).

Fixture layout (all little-endian): `int32 ne0`, `int32 nBlocks`, then
`nBlocks*17` raw packed bytes, then `nBlocks*32` float32 reference values.

## Rebuild the fixture

```sh
LL=~/code/llama.cpp                       # a built llama.cpp checkout
g++ -O2 -std=c++17 extract.cpp \
    -I"$LL/ggml/include" -I"$LL/ggml/src" \
    -L"$LL/build/bin" -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,"$LL/build/bin" -o extract
GGUF=~/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf
./extract "$GGUF" ../../src/test/fixtures/mxfp4/mxfp4_ref.bin 9
```

The `9` is the block count (9 blocks = 288 elements, a width that is a
multiple of 32 but not 256 — the width-cliff shape, spec §6.3). It
picks the first MXFP4 tensor in the file (blk.0.ffn_down_exps.weight on
gpt-oss-20b).
