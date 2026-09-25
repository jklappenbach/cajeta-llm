// ggml-leg — the llama.cpp leg of the parity harness (cajeta
// xpu-kernel-adaptor plan 8.1.1, spec §7.1).
//
// Times ggml's MUL_MAT for a quantized weight against an f32 activation
// at one shape, on llama.cpp's own backend and stream, with llama.cpp's
// own event mechanism, over a POOL of distinct weights sized past the
// part's L2 — the residency a decode actually sees. test-backend-ops
// repeats one tensor (L2-hot on a 4090) and times with the host clock
// around the whole graph; neither is comparable to a cold, device-timed
// cajeta number, which is why this exists.
//
// The timer: ggml creates its events with cudaEventDisableTiming, so the
// two ggml events are re-armed with timing-enabled CUDA events and
// recorded through ggml_backend_event_record, which puts them on the
// backend's compute stream. The elapsed time is the device's own, and
// the event clock's scale against the host clock is measured on every
// bracket (see the timing loop: the ratio moves with the part's state on
// this box).
//
// Output: one `leg-row` line (LegRow.cajeta documents the columns), or
// `leg-refused` with the reason. Build and run with build-ggml-leg.sh.
//
//   ggml-leg --ggml-lib-dir <dir> --type q4_K --m 4096 --k 14336 --n 1
//            --copies 7 --iters 5 --rounds 3 --build 67a17c1
//            [--gguf model.gguf --tensor blk.%d.ffn_down.weight --layers 4,5,7]
//            [--gguf model.gguf --tensor-list blk.0.attn_k.weight,blk.0.attn_v.weight]
//
// CUDA only for now: the L2 size and the timing events come from cudart.
// A HIP leg is the same code with the hip runtime; a Vulkan leg has no
// event clock and would print tier `unavailable`, which the table refuses.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"   // struct ggml_backend_event: to re-arm its CUDA event
#include "ggml-cpu.h"
#include "gguf.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int64_t host_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static ggml_type type_of(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && s == n) return (ggml_type) t;
    }
    return GGML_TYPE_COUNT;
}

static cudaEvent_t rearm(ggml_backend_event_t ev) {
    // ggml's event was created with cudaEventDisableTiming; replace it.
    cudaEventDestroy((cudaEvent_t) ev->context);
    cudaEvent_t e;
    if (cudaEventCreateWithFlags(&e, cudaEventDefault) != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate failed\n");
        exit(2);
    }
    ev->context = e;
    return e;
}

int main(int argc, char ** argv) {
    std::string lib_dir, type_s = "q4_K", build = "", fill = "quantize", gguf_path, tensor_fmt, layers_s, tensor_list_s;
    int64_t m = 4096, k = 14336, n = 1, copies = 1;
    int iters = 5, rounds = 3;
    bool usage_weights = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](void) -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--ggml-lib-dir") lib_dir = next();
        else if (a == "--type") type_s = next();
        else if (a == "--m") m = atoll(next());
        else if (a == "--k") k = atoll(next());
        else if (a == "--n") n = atoll(next());
        else if (a == "--copies") copies = atoll(next());
        else if (a == "--iters") iters = atoi(next());
        else if (a == "--rounds") rounds = atoi(next());
        else if (a == "--build") build = next();
        else if (a == "--fill") fill = next();
        else if (a == "--gguf") gguf_path = next();
        else if (a == "--tensor") tensor_fmt = next();
        else if (a == "--layers") layers_s = next();
        else if (a == "--tensor-list") tensor_list_s = next();
        else if (a == "--usage-weights") usage_weights = true;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    // --layers 4,5,7: copy c reads the c-th listed layer; copies follows the list.
    std::vector<int> layers;
    if (!layers_s.empty()) {
        size_t p = 0;
        while (p <= layers_s.size()) {
            size_t q2 = layers_s.find(',', p);
            if (q2 == std::string::npos) q2 = layers_s.size();
            if (q2 > p) layers.push_back(atoi(layers_s.substr(p, q2 - p).c_str()));
            p = q2 + 1;
        }
        copies = (int64_t) layers.size();
    }
    // --tensor-list a,b,c: explicit tensor names, one per copy.
    std::vector<std::string> tensor_list;
    if (!tensor_list_s.empty()) {
        size_t p = 0;
        while (p <= tensor_list_s.size()) {
            size_t q2 = tensor_list_s.find(',', p);
            if (q2 == std::string::npos) q2 = tensor_list_s.size();
            if (q2 > p) tensor_list.push_back(tensor_list_s.substr(p, q2 - p));
            p = q2 + 1;
        }
        copies = (int64_t) tensor_list.size();
    }
    ggml_type type = type_of(type_s);
    if (type == GGML_TYPE_COUNT) { fprintf(stderr, "unknown type %s\n", type_s.c_str()); return 2; }

    if (!lib_dir.empty()) ggml_backend_load_all_from_path(lib_dir.c_str());
    else ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        printf("leg-refused\tllama.cpp\t-\tmul_mat\tno GPU device registered (lib dir '%s')\n", lib_dir.c_str());
        return 1;
    }
    const char * backend_name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { printf("leg-refused\tllama.cpp\t%s\tmul_mat\tbackend init failed\n", backend_name); return 1; }
    if (strcmp(backend_name, "CUDA") != 0) {
        printf("leg-refused\tllama.cpp\t%s\tmul_mat\tthis leg times with CUDA events; backend is %s\n",
               backend_name, backend_name);
        return 1;
    }
    int l2 = 0;
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);

    // Tensors: `copies` weights of m rows x k, one activation k x n, one
    // output per weight, all in the backend's buffer.
    const size_t tensors = (size_t) copies * 2 + 1;
    ggml_init_params ip = { ggml_tensor_overhead() * tensors + ggml_graph_overhead_custom(tensors, false) + 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> w(copies), y(copies);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    for (int64_t c = 0; c < copies; c++) {
        w[c] = ggml_new_tensor_2d(ctx, type, k, m);
        y[c] = ggml_mul_mat(ctx, w[c], x);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { printf("leg-refused\tllama.cpp\tCUDA\tmul_mat\tbuffer allocation failed\n"); return 1; }
    if (usage_weights) ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Data: REAL tensors from the reference GGUF when --gguf is given
    // (`--tensor blk.%d.ffn_down.weight`, copy c reads layer c), else a
    // distinct pseudo-random weight per copy quantized by ggml, else a
    // byte pattern. Real data is the engine's case; the other two exist
    // because the q4_K, q5_K and iq4_nl MMVQ kernels on this build read
    // quantized noise 30x slower than a byte ramp, so a synthetic leg can
    // land anywhere (measured 2026-09-24, 4090, llama.cpp 67a17c1).
    gguf_context * gg = nullptr;
    ggml_context * gmeta = nullptr;
    FILE * gf_file = nullptr;
    if (!gguf_path.empty()) {
        gguf_init_params gp = { true, &gmeta };
        gg = gguf_init_from_file(gguf_path.c_str(), gp);
        if (!gg) { printf("leg-refused\tllama.cpp\tCUDA\tmul_mat\tcannot read %s\n", gguf_path.c_str()); return 1; }
        gf_file = fopen(gguf_path.c_str(), "rb");
    }
    std::vector<float> src((size_t) m * k);
    std::vector<uint8_t> q(ggml_nbytes(w[0]));
    for (int64_t c = 0; c < copies; c++) {
        uint32_t seed = 0x9e3779b9u * (uint32_t) (c + 1);
        for (size_t i = 0; i < src.size(); i++) {
            seed = seed * 1664525u + 1013904223u;
            src[i] = ((seed >> 8) & 0xffff) / 65536.0f - 0.5f;
        }
        if (gg) {
            char name[256];
            if (!tensor_list.empty()) snprintf(name, sizeof name, "%s", tensor_list[c].c_str());
            else snprintf(name, sizeof name, tensor_fmt.c_str(), layers.empty() ? (int) c : layers[c]);
            int64_t id = gguf_find_tensor(gg, name);
            if (id < 0) { printf("leg-refused\tllama.cpp\tCUDA\tmul_mat\tno tensor %s in the gguf\n", name); return 1; }
            ggml_tensor * meta = ggml_get_tensor(gmeta, name);
            if (meta->type != type || meta->ne[0] != k || meta->ne[1] != m) {
                printf("leg-refused\tllama.cpp\tCUDA\tmul_mat\t%s is %s %lld x %lld, not %s %lld x %lld\n", name,
                       ggml_type_name(meta->type), (long long) meta->ne[0], (long long) meta->ne[1],
                       ggml_type_name(type), (long long) k, (long long) m);
                return 1;
            }
            size_t off = gguf_get_data_offset(gg) + gguf_get_tensor_offset(gg, id);
            fseek(gf_file, (long) off, SEEK_SET);
            if (fread(q.data(), 1, q.size(), gf_file) != q.size()) { printf("leg-refused\tllama.cpp\tCUDA\tmul_mat\tshort read of %s\n", name); return 1; }
        } else if (fill == "pattern") {
            // Synthetic blocks: sane f16 scales in the header, a byte ramp
            // in the payload — the same shape of data cajeta's probes use.
            const size_t bb = ggml_type_size(type);
            for (size_t i = 0; i < q.size(); i++) q[i] = (uint8_t) ((i * 7 + 13 + c) & 255);
            for (size_t b = 0; b + bb <= q.size(); b += bb) {
                q[b] = 0; q[b + 1] = 0x3C;          // d = 1.0
                if (bb >= 4) { q[b + 2] = 0; q[b + 3] = 0x38; }   // dmin = 0.5
            }
        } else {
            ggml_quantize_chunk(type, src.data(), q.data(), 0, m, k, nullptr);
        }
        ggml_backend_tensor_set(w[c], q.data(), 0, q.size());
    }
    std::vector<float> xs((size_t) k * n);
    for (size_t i = 0; i < xs.size(); i++) xs[i] = 0.02f * (float) ((int) (i % 11) - 5);
    ggml_backend_tensor_set(x, xs.data(), 0, xs.size() * sizeof(float));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, tensors, false);
    for (int64_t c = 0; c < copies; c++) ggml_build_forward_expand(gf, y[c]);

    // Warm-up and the non-zero guard.
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("leg-refused\tllama.cpp\tCUDA\tmul_mat\tgraph compute failed\n");
        return 1;
    }
    std::vector<float> out((size_t) m * n);
    ggml_backend_tensor_get(y[copies - 1], out.data(), 0, out.size() * sizeof(float));
    bool live = false;
    for (float v : out) if (v != 0.0f) { live = true; break; }
    if (!live) { printf("leg-refused\tllama.cpp\tCUDA\tmul_mat\tproduced zeros\n"); return 1; }

    // The timer on the backend's own stream. THE SCALE IS PER BRACKET: on
    // this box the event clock's ratio to the host clock moved between
    // 0.98 and 1.10 across processes minutes apart, busy or idle, so a scale
    // measured once is not a correction. Each round's bracket is long enough
    // (about 200 ms of device time, iters raised to reach it) that the host
    // clock spanning the same submissions to the synchronize bounds it to
    // well under 0.1%; the row's clock_scale is that round's device/host
    // ratio, and the number is the bracket divided by it.
    ggml_backend_event_t e0 = ggml_backend_event_new(dev);
    ggml_backend_event_t e1 = ggml_backend_event_new(dev);
    cudaEvent_t c0 = rearm(e0), c1 = rearm(e1);
    for (int i = 0; i < 3; i++) ggml_backend_graph_compute_async(backend, gf);
    ggml_backend_synchronize(backend);
    int64_t hp0 = host_ns();
    for (int i = 0; i < 3; i++) ggml_backend_graph_compute_async(backend, gf);
    ggml_backend_synchronize(backend);
    int64_t per_iter = (host_ns() - hp0) / 3;
    if (per_iter < 1000) per_iter = 1000;
    int need = (int) (200000000LL / per_iter) + 1;
    if (need > iters) iters = need;
    if (iters > 20000) iters = 20000;

    double best_host = 0, best_scale = 0;
    for (int r = 0; r < rounds; r++) {
        ggml_backend_synchronize(backend);
        int64_t h0 = host_ns();
        ggml_backend_event_record(e0, backend);
        for (int it = 0; it < iters; it++) {
            if (ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS) {
                printf("leg-refused\tllama.cpp\tCUDA\tmul_mat\tgraph compute failed in the bracket\n");
                return 1;
            }
        }
        ggml_backend_event_record(e1, backend);
        ggml_backend_synchronize(backend);
        int64_t h1 = host_ns();
        float ms_r = 0;
        if (cudaEventElapsedTime(&ms_r, c0, c1) != cudaSuccess) {
            printf("leg-refused\tllama.cpp\tCUDA\tmul_mat\tcudaEventElapsedTime refused\n");
            return 1;
        }
        double host = (double) (h1 - h0);
        double scale_r = (ms_r * 1.0e6) / host;
        printf("# round %d: %.1f us per launch over %.0f ms, event clock scale %.4f\n", r,
               host / 1000.0 / (double) (iters * copies), host / 1.0e6, scale_r);
        if (r == 0 || host < best_host) { best_host = host; best_scale = scale_r; }
    }
    double scale = best_scale;
    double ns_total = best_host;

    int64_t launches = (int64_t) iters * copies;
    int64_t per = (int64_t) (ns_total / (double) launches);
    int64_t wbytes = (int64_t) ggml_nbytes(w[0]);
    int64_t pool = wbytes * copies;
    int64_t per_launch_bytes = wbytes + (int64_t) ggml_nbytes(x) + (int64_t) ggml_nbytes(y[0]);
    const char * residency = l2 <= 0 ? "unknown" : (pool > l2 ? "cold" : "hot");
    double flops = 2.0 * (double) m * (double) k * (double) n;

    printf("leg-row\tllama.cpp\t%s\t%s\tmul_mat\t%s\t%lld\t%lld\t%lld\t%lld\t%lld\t%d\t%s\tdevice\t%lld\t%lld\t%.1f\t%.4f\n",
           backend_name, build.c_str(), ggml_type_name(type),
           (long long) m, (long long) k, (long long) n, (long long) launches,
           (long long) pool, l2, residency, (long long) per, (long long) per_launch_bytes,
           flops, scale);

    ggml_backend_event_free(e0);
    ggml_backend_event_free(e1);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
