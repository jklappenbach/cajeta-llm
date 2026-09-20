# Wave-width-agnostic coop kernels — determination & conversion checklist

Status: **draft**. The compiler-side keystone is landed and validated on wave32
(NVIDIA) + wave8 (CPU AVX2). The per-kernel conversions in this document are
**validated on wave64 (AMD/CDNA)** — that is the only width at which they change
behaviour — so they are owned by the AMD/CDNA session. This file is the handoff.

## 1. Definition

Make the cajeta-llm coop/decode kernels compute correctly at **any** hardware
wave width — wave32 (NVIDIA, RDNA) and wave64 (CDNA) — from a single source, in
both `--emit` binary output and the JIT. The mechanism is the AUTO model:
kernels read `Wave.width()` / `Wave.laneId()` (which lower to the device's real
width on every backend) instead of the literal `32`. No `@Wave(width=N)` — that
annotation is currently informational only (consumed at `XpuMirPrinter.cpp:52`,
enforced by no backend), so it is NOT a width contract today.

## 2. What is already done (compiler, this session)

- `Wave.reduceSumF32Segmented(value, segment)` / `reduceMaxF32Segmented` —
  a bounded XOR butterfly on `waveShuffleDivergent`; correct on every backend in
  JIT and AOT; `min(seg,width)` clamp; CPU routes through its vectorizing
  whole-wave reduce. Commit `98837044`. Test `XpuWaveSegmentedDeviceTests`
  (segment 16 in a 32-wave → two independent sums; segment 32 → whole wave).
- `Wave.width()`/`Wave.laneId()` confirmed AUTO: AMD `amdgcn_wavefrontsize`
  (32 RDNA / 64 CDNA), NVPTX `%warpsize` (32), Vulkan `SubgroupSize` (runtime).

## 3. The three-way classification, applied per literal

Every `32` / `64` in a kernel is ONE of:

1. **Wave-coupled** — a lane count or lane-index math (`gid / 32`, `lane & 31`,
   `rl % 4` where rl spans the wave, the per-lane sub-block partition). MUST
   become width-derived. This is the conversion.
2. **Format** — a quant layout constant (`b * 32L` = the MXFP4 32-element quant
   block; `32 * g` = a 32-byte header quarter; `b * 144` = K-quant block bytes).
   MUST stay literal. Blind-swapping these corrupts numerics silently.
3. **Tile** — an output/token tile size independent of the wave (`q4kQ8MmqKernel`'s
   64-row × 32-token tile, its `int32[64*65]` LDS extents). Stays literal; the
   256-thread workgroup covers the tile at any wave width.

## 4. Kernel triage (cajeta-llm/src/main/.../io/QuantKernel.cajeta unless noted)

- **Tiled coops need NO change.** `q4kQ8MmqKernel`, its Q6_K twin, `coopQ8`,
  `mxfp4MatVecKernelCoopQ8`: zero `Wave.*` ops, barrier-synced, private register
  tiles, all extents format/tile. Verified width-agnostic already. Leave them.
- **Pattern A — wave-reduce mat-vecs** (`qkvWaveMatVecKernel` and the ~50
  `Wave.reduceSumF32` sites): NOT a literal swap. `wave = gid/32` → `/Wave.width()`
  is mechanical, but the lane→sub-block partition (`g=rl%4; b=rl/4`, 8 b-values
  for 32 lanes) ALGORITHMICALLY assumes 32 lanes — on 64 lanes the per-lane work
  halves and the partition must be re-derived. Re-tile, then validate numerics
  on wave64.
- **Pattern B — block-scoped reduces** (`mxfp4QuantActKernel` amax over a
  32-block; the k-quant amax at 32/256 granularity): the reduce becomes
  `reduceMaxF32Segmented(amax, <blockLanes>)`. But also audit the lane→element
  mapping: if a wave handles one 32-block via 32 lanes, on wave64 it spans two
  blocks and the element mapping must change too. Reduce swap alone is not
  sufficient unless the element loop is already `for e = lane; e < block; e +=
  width`.

## 5. The correctness landmine (do not skip)

A whole-wave `reduceMaxF32`/`reduceSumF32` over a fixed logical block is CORRECT
only when wave == block. On wave64 over a 32-block it merges two blocks and
hands both the same wrong scale — **silent wrong numbers, not a crash**. Every
wave reduce must be audited against its logical block size; convert to
segmented wherever wave may exceed block. This is why §4 Pattern B cannot be a
find/replace.

## 6. Validation matrix

| Width | Where | Catches |
|---|---|---|
| wave32 | NVIDIA (WSL+Windows), this session | conversions that BREAK the 32 case |
| wave8  | CPU AVX2, this session | Pattern-A width-dependence (W≠32), for kernels that call a wave op |
| wave64 | AMD/CDNA, **the owning session** | the actual fix — the only width where these changes alter behaviour |

wave32 and wave8 are regression guards; **wave64 is the acceptance gate.** A
green wave32 run does NOT confirm a wave64 conversion is correct — 32 is
unchanged there.
