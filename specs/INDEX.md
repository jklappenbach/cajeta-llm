# Active work

| spec | plan | status |
|---|---|---|
| [prefill-routes-spec](prefill-routes-spec.md) | [prefill-routes-plan](../agents/prefill-routes-plan.md) | active |
| [prepared-weights-spec](prepared-weights-spec.md) | — | draft |
| [codebook-quants-spec](codebook-quants-spec.md) | [codebook-quants-plan](../agents/codebook-quants-plan.md) | active |
| [wave-row-geometry](wave-row-geometry-spec.md) | [plan](../agents/wave-row-geometry-plan.md) | **active** — wave width and rows per workgroup are declared per kernel and tuned per part; the 44-kernel conversion is done. Open: 2.3.2, and the NVIDIA sweep 4.1.2–4.1.3. Unit 3 closed as moot 2026-10-16. |
| [host-floor](host-floor-spec.md) | — | **draft** — every format that loads must run on the HOST; device routes are optimizations above that floor. Filed from a measurement: `Quant.supported()` claims 21 types, `Linear.matvecInto`'s host chain serves 16, and the five it misses run on a GPU and fail on the host (IQ4_NL among them, optimized this week). Four of the five have a tested host mat-vec in `Quant` with ZERO production callers; MXFP4 has none. `theFourPartInvariant` cannot catch it — its four parts are `supported` plus three DEVICE predicates, and the host path is not one of them. Retires `supported()` and `decodable()` for `runnable(ty)` = decoder + host mat-vec, replaces the four-part gate with "every runnable type has a `matvecInto` arm", and dissolves Unit 8's "gates last" ordering into "floor first" (rewrites codebook-quants 8.2.1). |
