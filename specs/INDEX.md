# Active work

| spec | plan | status |
|---|---|---|
| [prefill-routes-spec](prefill-routes-spec.md) | [prefill-routes-plan](../agents/prefill-routes-plan.md) | draft |
| [prepared-weights-spec](prepared-weights-spec.md) | — | draft |
| [codebook-quants-spec](codebook-quants-spec.md) | [codebook-quants-plan](../agents/codebook-quants-plan.md) | active |
| [host-floor](host-floor-spec.md) | — | **draft** — every format that loads must run on the HOST; device routes are optimizations above that floor. Filed from a measurement: `Quant.supported()` claims 21 types, `Linear.matvecInto`'s host chain serves 16, and the five it misses run on a GPU and fail on the host (IQ4_NL among them, optimized this week). Four of the five have a tested host mat-vec in `Quant` with ZERO production callers; MXFP4 has none. `theFourPartInvariant` cannot catch it — its four parts are `supported` plus three DEVICE predicates, and the host path is not one of them. Retires `supported()` and `decodable()` for `runnable(ty)` = decoder + host mat-vec, replaces the four-part gate with "every runnable type has a `matvecInto` arm", and dissolves Unit 8's "gates last" ordering into "floor first" (rewrites codebook-quants 8.2.1). |
| [route-table-spec](route-table-spec.md) | — | moved → `cajeta/specs/archive/route-table-spec.md` (xpu layer, CLOSED 2026-09-18; this package is the first registrant, as `codebook-quants-plan.md` 9.2.9) |
