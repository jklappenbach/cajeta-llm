# Prepared weights — spec

STATUS: **draft**, and drafted alone. The exploration behind it was
parked on this clone's focus stack from 2026-09-13; this promotes it to
a reviewable document so the findings stop living in a one-line note. It
has NOT been through the design skill's conversation and no plan should
be written from it until it has.

## 1. Definition

### 1.1 What it is

A **prepared-weight cache**: a derived, device-keyed sidecar holding the
de-interleaved, tile-ordered form of a GGUF's tensors that the engine
currently recomputes on every load. Written once, mmapped thereafter,
and ALWAYS safe to delete, because every byte of it is reconstructible
from the source GGUF.

### 1.2 The problem, measured

A k-quant block interleaves its scales with its quants, so every kernel
reads a 16-byte header inside the weight stream and bit-unpacks six-bit
scale fields before it can do arithmetic. Both engines pay to undo that
file layout: llama.cpp stages a de-interleaved LDS tile per dispatch,
and this engine builds an int8 widen copy per tensor.

- The widen kernels cost **~190 ms of device time per 8B load**
  (measured 2026-09-13).
- The whole `prefillWeights=auto` policy exists to decide, per format,
  whether that transform is worth paying for.
- 7.4 (2026-09-14) added a SECOND derived artifact beside the widen —
  the compact f16 scale image — and made a third class of tensor (the
  column-remainder shapes) eligible for the widen at all.
- 8.3.1 (2026-09-14) measured the widen's cost in the other currency:
  on Qwen2.5-VL-72B the full widen is ~70 GB over 45 GB of packed
  weights, and a PARTIAL widen prefilled at 19.95 tok/s against 26.2 for
  no widen at all. The transform is not unambiguously worth it at scale.

A cache does not make the transform cheaper; it makes it **paid once**.
That changes the auto policy's question from "is this transform worth
its cost every load" to "is it worth its disk".

### 1.3 Constraints

- **Not a distributable format.** Its entire value is that nobody has to
  adopt anything. It must never be published, shipped, or treated as a
  model artifact.
- **Reconstructible, therefore disposable.** Deleting the cache must
  never lose information, and a missing or stale entry must degrade to
  the current recompute path silently and correctly.
- **Device-keyed.** Tile geometry and wave width are part of the key,
  because the prepared form is shaped for a specific device's kernels.
- **Source-identified.** The entry must carry a hash of the SOURCE, so a
  changed GGUF cannot be served a stale prepared form.

### 1.4 Container finding (2026-09-13)

- The GGUF spec has **no notion of a derived or cached form** — it
  describes one self-contained file of metadata plus tensor data. A tile
  cache therefore sits OUTSIDE it and must never be mistaken for a
  variant of it.
- But the container is extensible by convention: a dotted key namespace
  with a 315-entry registry in llama.cpp's `gguf-py/gguf/constants.py`.
  So the sidecar can BE a GGUF carrying our own namespaced keys (tile
  geometry, wave width, source hash) — no new container to invent, and
  the cajeta GGUF writer makes it writable from this repo.
- CORRECTION to an earlier note in this exploration that said "GGUF has
  no integrity story": llama.cpp ships `llama-gguf-hash`, per-model AND
  per-tensor, xxh64/sha1/sha256/uuid with `manifest --check`. The hash
  is external to the file rather than absent, so the sidecar should
  record the source hash in THAT form rather than inventing one.

### 1.5 Non-goals

- Not a quantization format, and not a new quant type.
- Not a distribution or sharing mechanism.
- Not a replacement for `prefillWeights`; it changes that policy's
  inputs, it does not remove the policy.
- Not a correctness mechanism — it must be a pure performance cache.

## 2. Identity and invalidation

- **2.1** When a prepared entry's recorded source hash does not match the
  GGUF it is asked to serve, the entry is ignored and the engine
  recomputes.
- **2.2** When a prepared entry's recorded tile geometry or wave width
  does not match the active device, the entry is ignored and the engine
  recomputes.
- **2.3** When a prepared entry is absent, the engine recomputes exactly
  as it does today, with no diagnostic beyond a cache-miss record.
- **2.4** When a prepared entry is corrupt or truncated, it is treated as
  absent rather than trusted.
- **2.5** When the cache directory is deleted while the engine is not
  running, the next load succeeds and produces identical output.

## 3. What is stored

- **3.1** When a tensor takes the int8 widen route, its widened
  `[outDim x colsPad]` copy is a prepared artifact.
- **3.2** When a tensor takes the symmetric deq route, its compact f16
  scale image (7.4.2) is a prepared artifact beside the widen.
- **3.3** When a format needs a device repack before the coop GEMM
  (`coopNeedsRepack`), the repacked form is a prepared artifact.
- **3.4** When a tensor needs no transform for any route it can take,
  nothing is stored for it.

## 4. Cost and policy

- **4.1** When a prepared entry is served, the load does not run that
  tensor's widen kernel.
- **4.2** When the cache is warm, the `auto` policy's per-format decision
  is made against the cache-hit cost, not the recompute cost.
- **4.3** When the cache would exceed a configured disk budget, entries
  are refused rather than evicted mid-load, and the refusal names what
  it refused — the shape 8.3.1 established for the memory budget.
- **4.4** When a prepared form is larger than the source tensor it
  derives from, that fact is recorded, because the int8 widen is
  ~2x the packed bytes and the disk cost is the whole trade.

## 5. Open questions for the design conversation

- **5.1** Does this survive 8.3.1's finding at all? If a partial widen is
  slower than none on a 72B, the cache may be optimizing a transform the
  engine should be doing LESS of. The honest first step may be a
  `widenMb` sweep, not a cache.
- **5.2** Per-model directory or one content-addressed store?
- **5.3** Who writes the entry — the loader on first miss, or an explicit
  `cajeta-llm prepare` verb? The first is invisible, the second is
  predictable.
- **5.4** Does the sidecar-as-GGUF finding (1.4) actually pay, against a
  plain length-prefixed blob plus a small header?
