// cupti-tally — a per-kernel device-time tally for ANY CUDA program, by CUPTI
// injection (cajeta-llm 15.2.15.1.2: llama.cpp has no CUDA per-op logger, and
// nsys is not installed on Phoenix).
//
//   g++ -O2 -shared -fPIC cupti-tally.cpp -o libcupti-tally.so -lcupti -ldl
//   CUDA_INJECTION64_PATH=$PWD/libcupti-tally.so CUPTI_TALLY_OUT=run.csv \
//       llama-batched-bench -m model.gguf -c 4096 -npp 512 -ntg 128 -npl 4 -ngl 99
//
// The driver loads the library at cuInit and calls InitializeInjection, which
// enables CONCURRENT_KERNEL activity (the kind that keeps kernels concurrent)
// and registers the buffer callbacks. Every kernel record is appended to
// CUPTI_TALLY_OUT as `start_ns,end_ns,stream,name [grid/block]` (device clock; copies
// and fills too, as `memcpy HtoD 8.0 MB` and `memset 2.0 MB`), and
// a per-name summary (count, total us, mean us) goes to stderr at exit.
// kernel-tally.py windows the CSV the way `cajeta profile summary` does.
//
// CUPTI_TALLY_HOST_OUT=host.csv also records the host side: every driver API
// call (and, for programs on the CUDA runtime such as llama.cpp, every runtime
// call, prefixed `rt:`) as `start_ns,end_ns,thread,correlation,name` on the
// CPU clock, and `<host.csv>.corr` the `correlation,start_ns` of every kernel.
// The CPU and GPU clocks differ by tens of seconds under WSL and drift by
// percents, so gpu-timeline.py fits one onto the other through the launch
// calls and the kernels they started, then attributes each nanosecond the
// GPU sat idle to the host call overlapping it (sync, alloc, copy, launch) or
// to the program's own code when none did. Each recorded call costs about a
// microsecond, so compare the wall time with and without it before trusting
// a split of launch-bound time.
#include <cupti.h>
#include <algorithm>
#include <cxxabi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct Row { uint64_t count = 0, total = 0, mx = 0; };
std::mutex g_lock;
std::map<std::string, Row> g_rows;
FILE* g_out = nullptr;
FILE* g_host = nullptr;
FILE* g_corr = nullptr;
uint64_t g_hostRecords = 0;
uint64_t g_first = 0, g_last = 0, g_records = 0;

std::string demangle(const char* n) {
    int st = 0;
    char* d = abi::__cxa_demangle(n, nullptr, nullptr, &st);
    std::string s = (st == 0 && d) ? d : n;
    free(d);
    // The template arguments and the parameter list are noise for a tally.
    const size_t lt = s.find('<');
    const size_t lp = s.find('(');
    size_t cut = std::string::npos;
    if (lt != std::string::npos) cut = lt;
    if (lp != std::string::npos && lp < cut) cut = lp;
    if (cut != std::string::npos) s = s.substr(0, cut);
    return s;
}

void CUPTIAPI bufferRequested(uint8_t** buf, size_t* size, size_t* maxRecords) {
    *size = 8 * 1024 * 1024;
    *buf = (uint8_t*) malloc(*size);
    *maxRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
    CUpti_Activity* rec = nullptr;
    std::lock_guard<std::mutex> g(g_lock);
    while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
        uint64_t start = 0, end = 0; unsigned stream = 0; std::string name;
        if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            const CUpti_ActivityKernel9* k = (const CUpti_ActivityKernel9*) rec;
            char shape[64];
            snprintf(shape, sizeof shape, " [%dx%dx%d/%d]", k->gridX, k->gridY, k->gridZ, k->blockX);
            name = demangle(k->name ? k->name : "?") + shape;
            start = k->start; end = k->end; stream = k->streamId;
            // The launch call that started it shares this id: the pairs put the
            // host calls (CPU clock) on the device clock (gpu-timeline.py).
            if (g_corr) fprintf(g_corr, "%u,%llu\n", k->correlationId, (unsigned long long) k->start);
        } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            // Copies and fills are device time a kernel tally cannot see; a
            // prefill that waits on them reads as idle otherwise.
            const CUpti_ActivityMemcpy5* m = (const CUpti_ActivityMemcpy5*) rec;
            static const char* kinds[] = {"?", "HtoD", "DtoH", "HtoA", "AtoH", "AtoA", "AtoD", "DtoA", "DtoD", "HtoH", "PtoP"};
            const unsigned ck = m->copyKind < 11 ? m->copyKind : 0;
            char b[96];
            snprintf(b, sizeof b, "memcpy %s %.1f MB", kinds[ck], m->bytes / 1048576.0);
            name = b; start = m->start; end = m->end; stream = m->streamId;
        } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            const CUpti_ActivityMemset4* m = (const CUpti_ActivityMemset4*) rec;
            char b[96];
            snprintf(b, sizeof b, "memset %.1f MB", m->bytes / 1048576.0);
            name = b; start = m->start; end = m->end; stream = m->streamId;
        } else if (rec->kind == CUPTI_ACTIVITY_KIND_DRIVER || rec->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
            // Host calls go to their own file and never into the device tally.
            const CUpti_ActivityAPI* a = (const CUpti_ActivityAPI*) rec;
            const bool rt = rec->kind == CUPTI_ACTIVITY_KIND_RUNTIME;
            const char* nm = nullptr;
            cuptiGetCallbackName(rt ? CUPTI_CB_DOMAIN_RUNTIME_API : CUPTI_CB_DOMAIN_DRIVER_API, a->cbid, &nm);
            if (g_host) fprintf(g_host, "%llu,%llu,%u,%u,%s%s\n", (unsigned long long) a->start,
                                (unsigned long long) a->end, a->threadId, a->correlationId,
                                rt ? "rt:" : "", nm ? nm : "?");
            g_hostRecords++;
            continue;
        } else {
            continue;
        }
        Row& r = g_rows[name];
        const uint64_t d = end - start;
        r.count++; r.total += d; if (d > r.mx) r.mx = d;
        if (!g_first || start < g_first) g_first = start;
        if (end > g_last) g_last = end;
        g_records++;
        if (g_out) fprintf(g_out, "%llu,%llu,%u,%s\n", (unsigned long long) start,
                           (unsigned long long) end, stream, name.c_str());
    }
    free(buf);
}

void atExit() {
    cuptiActivityFlushAll(1);
    std::lock_guard<std::mutex> g(g_lock);
    if (g_out) { fclose(g_out); g_out = nullptr; }
    if (g_host) {
        fclose(g_host); g_host = nullptr;
        if (g_corr) { fclose(g_corr); g_corr = nullptr; }
        fprintf(stderr, "cupti-tally: %llu host call records\n", (unsigned long long) g_hostRecords);
    }
    std::vector<std::pair<std::string, Row>> rows(g_rows.begin(), g_rows.end());
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second.total > b.second.total; });
    uint64_t sum = 0;
    for (auto& r : rows) sum += r.second.total;
    fprintf(stderr, "cupti-tally: %llu kernel records, %.3f ms of kernel time over a %.3f ms span\n",
            (unsigned long long) g_records, sum / 1e6, (g_last - g_first) / 1e6);
    fprintf(stderr, "%-56s %9s %12s %9s %9s\n", "kernel", "count", "total_us", "mean_us", "max_us");
    for (auto& r : rows)
        fprintf(stderr, "%-56.56s %9llu %12.1f %9.2f %9.1f\n", r.first.c_str(),
                (unsigned long long) r.second.count, r.second.total / 1e3,
                r.second.total / 1e3 / r.second.count, r.second.mx / 1e3);
}

} // namespace

extern "C" int InitializeInjection(void) {
    static bool once = false;
    if (once) return 1;
    once = true;
    if (const char* p = getenv("CUPTI_TALLY_OUT")) g_out = fopen(p, "w");
    if (cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted) != CUPTI_SUCCESS) {
        fprintf(stderr, "cupti-tally: cuptiActivityRegisterCallbacks failed\n");
        return 1;
    }
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS)
        fprintf(stderr, "cupti-tally: could not enable CONCURRENT_KERNEL activity\n");
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    if (const char* h = getenv("CUPTI_TALLY_HOST_OUT")) {
        g_host = fopen(h, "w");
        g_corr = fopen((std::string(h) + ".corr").c_str(), "w");
        if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER) != CUPTI_SUCCESS)
            fprintf(stderr, "cupti-tally: could not enable DRIVER activity\n");
        if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME) != CUPTI_SUCCESS)
            fprintf(stderr, "cupti-tally: could not enable RUNTIME activity\n");
    }
    atexit(atExit);
    return 1;
}
