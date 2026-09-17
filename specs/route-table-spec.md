# Route table — spec

Status: draft. Registered in [INDEX.md](INDEX.md). Plan: none yet.

## 1. Definition

The route table is one declaration of which kernel serves which weight
format in which regime, and one test that audits that declaration
against every format the engine admits. It replaces the predicates that
today restate that knowledge from memory, one per route, in whichever
file the route lives.

### 1.1 Problem statement

A format reaches parity in four parts — decoder, host mat-vec, bind
gate, device route — and the fourth part is not one thing. It is a
dozen: the decode wave route, its integer and f32 arms, the grouped id
route, the fused row route, the coop GEMM, the WMMA widen, the batch
kernels, the split at bind. Each has a predicate that admits a format,
and each predicate was written when the route was, naming the formats
that existed then.

The consequence is measured, not argued. On 2026-09-17 alone:

- `wavef`'s guard read `!q8kDims` where it meant "no integer wave route
  engaged"; IQ4_NL answered no to both and decoded item-per-row at a
  256-aligned width. 26.4 → 42.9 t/s once asked properly (plan 9.2.4).
- `ExpertBank.idReady()` listed types 12 and 14. The codebook banks were
  refused the grouped id route and dispatched one launch per expert per
  projection: 384 a token against llama.cpp's ~72 (plan 6.4.3, cause 3).
- `Linear.packedWaveReady()` ended `(q8 && wave) || wave6` — the same
  list, one file over — and its dispatcher fell through a bare `else`
  into the Q6_K decoder. Widening the predicate alone would have fed
  codebook bytes through it: wrong logits, no crash.
- The fix for the second added `codebookId()`: one more list of the
  same shape.

This is the eighth or ninth time the class has been solved, and each
time from scratch, because the knowledge lives in prose and in whoever
last hit the refusal. The half that exists is `theFourPartInvariant`
(codebook-quants 3.1.5): over every `supported()` type, four gates
agree. It has held since Unit 3. It covers four gates; the routes have
grown to a dozen predicates outside it, so it stays green while a
route it does not know refuses.

### 1.2 Scope

Every predicate in the llm package that decides, for a weight, which
kernel runs — and every dispatcher those predicates guard. The table
is data; the selector is one function; the audit is one test.

### 1.3 Non-goals

- **Capability queries.** `Linear.backendIsVulkan()` at ~14 sites asks
  "is WMMA present", "can I `dotSum`", "is this RADV". Those are the
  device's to answer and belong in xpu. They are listed so the table
  does not absorb them by accident, and deferred to the arc in §7.
- **Measured selection.** Choosing among admitted routes by the
  per-device Autotune store. The table makes that possible; it does
  not do it.
- **Layout selection.** The resident split layout is already one
  layout for every format (codebook-quants Unit 9). A layout exists
  because kernels read it; choosing one is choosing a kernel set, and
  the kernels still have to be written.
- **Writing kernels.** The table names which kernel serves a format.
  It does not draft the kernel's body.

## 2. Possible solutions

### 2.1 A sweep onto `qAct` / `intWaveRouted()`

Replace each format list with the union predicate `Linear` already
carries. Fixes every list that exists today. Does nothing for the next
route, whose author writes a new predicate outside the union — which
is how `idReady` came to exist beside `qAct`. Rejected as sufficient;
kept as the first mechanical step of §4.

### 2.2 Extend the four-part invariant to every predicate

Keep the predicates, teach the test about all of them. The test then
has to *know* every predicate, and a new one is invisible to it until
someone adds it — the same failure, moved into the test. Rejected.

### 2.3 A route table (proposed)

Routes become rows of data. There is one selector and one audit, and
the audit walks the table, so a route that exists is a route the test
sees. §3.

### 2.4 Capability / cost / policy registry in xpu

The full separation: capability answered by the device, cost from the
Autotune store, policy a selector over both. Correct, larger, and it
sits *on top of* §2.3 — the rows are the policy half. Deferred to §7
as its own arc.

## 3. Proposed solution

### 3.1 A route is a row

| field | meaning |
|---|---|
| `name` | what the diag prints when it refuses or takes it |
| `regime` | decode-row (M = 1), prefill-batch (M ≥ tile), bind, fused tail |
| `admits(ty, shape)` | the formats and shape constraints (`inDim % 256`, `outDim % tile`, block alignment) |
| `needs` | capability the device must have — a query, never a backend name |
| `dispatch(...)` | the launcher; every admitted format has an arm |
| `priority` | order among rows that admit the same (format, regime) |

`admits` is derived where it can be — an integer-wave row admits what
`qAct` admits — and enumerated only where the route genuinely serves a
named set (the k-quant id route serves Q4_K and Q6_K because those are
the kernels that exist). Either way it is in the row.

### 3.2 One selector

`RouteTable.pick(regime, ty, shape)` returns the first admitting row by
priority, or a refusal that names the last row consulted and the clause
that refused. The `moe-row-route` and `moe-batch-route` records read
the same object. No caller tests a format itself.

### 3.3 One audit

`theRouteTableInvariant`: for every `ty` in `Quant.supported()` and
every row, the row either admits `ty` or refuses it by name; and for
every admitted `(row, ty)`, the row's dispatcher has an arm for `ty` —
which a bare `else` fails on the first run. Per row, a does-fire test
and a does-not-fire test, as codebook-quants 9.2.7 did.

### 3.4 Use cases

- **3.4.1** When a format is added, the audit names every row that
  lacks it, before any model is loaded.
- **3.4.2** When a route is added, the audit runs it against every
  format that exists, in the same commit.
- **3.4.3** When a dispatcher has no arm for a format its row admits,
  the audit fails; the fallthrough of 9.2.4 and `packedWaveReady`
  cannot ship.
- **3.4.4** When a route refuses at runtime, the diagnostic names the
  row and the clause, so a census that shows the old kernels is read
  in one step, not by bisection.
- **3.4.5** When a predicate would name formats, it is a row instead;
  a grep for `== 12 || == 14`, `codebookId`, `(q8 && wave) || wave6`
  returns nothing.
- **3.4.6** When the same (format, regime) is admitted by two rows,
  priority decides and the audit reports the shadowed row — the seed
  of measured selection (§7), without doing it yet.

## 4. Implementation

### 4.1 Where

`dev.cajeta.llm.model.Route` (the row) and `RouteTable` (rows, selector,
refusal), in the llm package. The mechanism is generic; the rows are
not. It moves to xpu when a second package registers rows (§6).

### 4.2 Order

1. §2.1's sweep, so the table starts from predicates that mean what
   they say.
2. Decode-row routes: `launchOne`'s chain, `packedWaveReady` /
   `matvecPackedKeep`, `idReady` / `idRowReady` / `symId`, the
   `zeroSyncReady` clauses. This is where every defect of 2026-09-17
   lived.
3. Prefill routes: `coopRoutedHere`, `hasBatchKernel`, the widen and
   Mw8 gates, the coop tile-divisibility refusals.
4. Bind: `splitOn` (which 9.2.1 already reduces to "nonzero
   `scaleBytes`") and the `deqFor` twin.
5. Attention: the flash decode / prefill tile gates on `hd == 128` —
   the same shape of predicate, recorded under codebook-quants 4.3.5
   as the reason bitnet-large (head dim 96) takes the scalar pair.

Each step replaces predicates with rows and every bare `else` with
arms and a terminal refusal; the audit is extended as rows land, so
the invariant never covers less than it did the commit before.

### 4.3 Tests

`theFourPartInvariant` becomes `theRouteTableInvariant` and grows with
§4.2. Per row: fire / no-fire. The existing route tests
(`MoeRowRouteTest`, `LinearKernelRouteTest`) keep asserting decisions;
they read the table instead of the flags.

### 4.4 Acceptance

- The audit is green over every `supported()` type and every row.
- The grep of 3.4.5 returns nothing outside `RouteTable`.
- Filtered suite green.
- Legs flat on the recorded files (the six of codebook-quants 9.3.2
  and the Qwen1.5-MoE): this is a refactor, and a speed change in
  either direction is a routing change to explain.

## 5. Kernel classes it helps draft

"Helps draft" means: when the kernel exists, the table says where it
goes, what it admits, and what it must not silently replace — and the
audit says what is missing. The classes, by regime:

- **Decode mat-vec** — the wave kernels (integer and f32 arms), the
  grouped id kernels, the fused down + combine + norm tails. The
  densest set of predicates today and the whole of 6.4.3's cause 3.
- **Prefill GEMM** — coop X1 / X3 / N64 / N256, the WMMA Mw / Mw8
  widen routes, MMQ. Their tile-divisibility refusals become row
  constraints instead of scattered `% 128` checks.
- **Bind** — the resident split, the widen slab, the retired repack.
- **Attention** — flash decode / prefill tile against the scalar pair,
  gated on head dimension. Not a weight format, but the same decision
  and the same failure mode (bitnet-large silently on the slow path).
- **Fused tails** — a row whose `admits` is the conjunction of the
  rows it fuses, which is how `idDownCombineTail` should have been
  declared from the start.

What it does not help: the kernel body. An IQ3_XXS id kernel still has
to be written by someone who knows the block layout.

## 6. Beyond language models

The table's shape is (data format × regime × capability) → kernel, with
an audit over the format set. Only the format axis is LLM-specific —
GGUF quantizations. The mechanism is not:

- **cajeta-xgboost** picks histogram kernels by bin count, feature
  count and device (the int64 fixed-point quantiser is one such route),
  with the same "which one runs here" question and no audit.
- **cajeta-ml** picks dense GEMMs by dtype (f32 / f16 / bf16), shape and
  WMMA availability. Training adds a regime axis (forward / backward)
  that maps directly onto §3.1's `regime`.
- Any package that has more than one kernel for one operation has this
  table already, written as if-chains.

So: build it in the llm package first, because that is where the pain
has been measured eight times; extract `Route` / `RouteTable` to xpu
the first time a second package registers rows. xgboost is the
candidate, and the extraction is the point at which capability queries
(§1.3) have to move too.

## 7. The arc this sits under

Capability into xpu (`backendIsVulkan`'s sites become device queries,
driver quirks move with them), cost from the Autotune store, measured
selection among admitted rows. Each is a spec of its own; each assumes
this table exists. This spec is the part that stops the bleeding now.
