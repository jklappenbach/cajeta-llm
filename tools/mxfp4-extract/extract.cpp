// Extract a small MXFP4 reference fixture from a real GGUF, using
// llama.cpp's own dequantize_row_mxfp4 as the oracle. Fixture layout
// (all little-endian): int32 ne0 (tensor width), int32 nBlocks,
// then nBlocks*17 raw packed bytes, then nBlocks*32 float32 reference.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "ggml.h"
#include "gguf.h"
#include "ggml-quants.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: extract <gguf> <out.bin> [nblocks]\n"); return 2; }
    const char* path = argv[1];
    const char* outp = argv[2];
    int wantBlocks = argc > 3 ? atoi(argv[3]) : 8;

    struct gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/nullptr };
    struct gguf_context* ctx = gguf_init_from_file(path, p);
    if (!ctx) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    int64_t n = gguf_get_n_tensors(ctx);
    int64_t pick = -1;
    for (int64_t i = 0; i < n; i++) {
        if (gguf_get_tensor_type(ctx, i) == GGML_TYPE_MXFP4) {
            // prefer a width that is x32 but not x256 (§6.3); accept first otherwise
            pick = i;
            break;
        }
    }
    if (pick < 0) { fprintf(stderr, "no MXFP4 tensor in %s\n", path); return 1; }

    const char* tname = gguf_get_tensor_name(ctx, pick);
    size_t dataOff = gguf_get_data_offset(ctx);
    size_t tOff = gguf_get_tensor_offset(ctx, pick);
    fprintf(stderr, "tensor[%lld] '%s' type=MXFP4 dataOff=%zu tOff=%zu\n",
            (long long)pick, tname, dataOff, tOff);

    // Read raw bytes for wantBlocks blocks (17 bytes each) from the file.
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "reopen fail\n"); return 1; }
    if (fseek(f, (long)(dataOff + tOff), SEEK_SET) != 0) { fprintf(stderr,"seek fail\n"); return 1; }
    int32_t nBlocks = wantBlocks;
    std::vector<uint8_t> raw(nBlocks * 17);
    if (fread(raw.data(), 1, raw.size(), f) != raw.size()) { fprintf(stderr,"read fail\n"); return 1; }
    fclose(f);

    // llama.cpp reference dequant.
    std::vector<float> ref(nBlocks * 32);
    dequantize_row_mxfp4((const block_mxfp4*)raw.data(), ref.data(), (int64_t)nBlocks * 32);

    // ne0 is unknown here without reading tensor dims; MXFP4 rows tile at 32,
    // gpt-oss hidden widths (e.g. 2880) are x32 not x256 — record 0 and let
    // the test note it. (We deliberately keep the fixture format simple.)
    int32_t ne0 = 0;

    FILE* o = fopen(outp, "wb");
    fwrite(&ne0, 4, 1, o);
    fwrite(&nBlocks, 4, 1, o);
    fwrite(raw.data(), 1, raw.size(), o);
    fwrite(ref.data(), 4, ref.size(), o);
    fclose(o);
    fprintf(stderr, "wrote %s: %d blocks (%zu raw + %zu floats)\n",
            outp, nBlocks, raw.size(), ref.size());
    // Print a few values for a sanity glance.
    for (int i = 0; i < 4 && i < (int)ref.size(); i++) fprintf(stderr, "  ref[%d]=%g\n", i, ref[i]);
    gguf_free(ctx);
    return 0;
}
