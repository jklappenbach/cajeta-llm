# Kernel census tables

One file per backend, one row per kernel (xpu-kernel-adaptor plan 3.2.2).
`./run-tests.sh` writes the live table to `build/census-<backend>.tsv` on
every run; the copies here are refreshed by hand from a run whose suite
result is recorded, so a diff against them is a change in what lowers,
what runs, or what describes itself:

```sh
XPU_BACKEND=nvptx ./run-tests.sh && cp build/census-nvptx.tsv census/nvptx.tsv
```

Columns: `backend  kernel  class  launches  manifest  note`.

| class | meaning |
|---|---|
| `RAN` | registered, launched at least once by the suite, not refused |
| `SKIP` | registered, never launched, and tracked by name to the plan item that retires the skip (the note) |
| `PROBE` | a measurement kernel from the bench tree, not a shipped kernel |
| `EXTERNAL` | a cajeta stdlib kernel, gated by cajeta's own suites |
| `UNCOVERED` | registered, never launched, tracked by nothing: fails the suite |
| `STALE-SKIP` | tracked as a skip but it ran: fails the suite, drop the entry |
| `DECLINED` | never registered: the compiler produced no device code for this backend, and the note is its `[xpu-kernel-skipped]` reason |

`manifest` is whether `KernelManifest.of(kernel)` answers on this backend
(plan 4.3.2); `launches` is the runtime's count of launches that were not
refused (the Unit 6 instrument).
