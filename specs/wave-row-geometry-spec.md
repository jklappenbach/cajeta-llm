# Wave row geometry — spec

Status: **draft** 2026-09-18. Registered in [INDEX.md](INDEX.md).
Package: cajeta-llm. Touches `io/QuantKernel.cajeta` and the launchers
beside it.

## 1. Definition

A wave-cooperative kernel should learn its wave width from the device and
its rows per workgroup from a measurement taken on the machine it is
running on. Today most of them assume 32 of each, written as a literal.

The mechanism to do better is already built and already used by exactly
one kernel. This spec is about the other forty-three.

### 1.1 Problem statement

Measured 2026-09-18.

- `QuantKernel.cajeta` has **44 kernels** containing
  `KernelThread.globalIdX() / 32`. A kernel written that way is correct
  only at wave width 32 and only at one row per workgroup.
- **Three** sites in the same file use `Group.width()`. One of them is
  `q2kQ8WaveMatVecKernel`, whose comment records why the change was
  made: "The 32 this used to assume made one workgroup per row and was
  wrong on a wave64 part."
- `QuantKernel.waveRowsPerBlock()` recalls a per-machine tuned value,
  falls back to a default, and clamps the workgroup to
  `Device.maxThreadsPerBlock()`. `setWaveRowsPerBlock(n)` exists "for a
  sweep". `waveRowGrid` and `waveRowBlock` derive the launch from it.
  **Only `q2kQ8Wave` calls them.**
- **Nothing writes the value.** There is a recall path, a sweep hook and
  a default. No production code sweeps and calls `Autotune.remember`, so
  the default is doing all of the work and the tuned path has never
  carried a measured number.
- `waveRowsPerBlock()` calls `Autotune.recall`, not `recallFor`. The
  versioned variant is two functions away in the same class and
  discards a hint measured against different code, reporting XPU-T02.
  As written, a rebuild that changes a kernel keeps the old winner
  silently.

The cost is visible in the launchers. Four different launch strategies
serve one family of kernels with identical block geometry.
`q4k` and `q6k` call `rowsPerWave(cols / 256)` and then override its
answer with a literal 4. `q3k` and `q5k` use a bare
`grid: [rows], block: [32]`. `q2k` uses `waveRowGrid` / `waveRowBlock`.
Nothing about the data distinguishes them.

### 1.2 Scope

The conversion of wave-cooperative kernels in `QuantKernel.cajeta` to
wave-relative indexing, the launchers that feed them, a sweep that
measures rows per workgroup and writes it, and the version check on the
recall.

### 1.3 Non-goals

- **A cost model.** This spec measures rows per workgroup. Predicting it
  is the `xpu-kernel-adaptor` draft's, and the data this produces is
  what that prediction would be fitted to.
- **Kernels outside the wave-cooperative families.** The utility kernels
  that share the literal are listed but converted only where the same
  reasoning applies.
- **The other hardcoded widths.** `QuantKernel.cajeta` has 135 sites
  keyed to a literal 32. This spec converts the ones that are a
  kernel's lane and row derivation. A stride or a payload width that
  happens to be 32 is a different fact.

## 2. A kernel derives its geometry

- **2.1** When a wave-cooperative kernel computes its lane, it derives it
  from `Group.width()` rather than from a literal.
- **2.2** When such a kernel computes which row it serves, it divides the
  global id by the wave width, so a workgroup may carry several rows.
- **2.3** When a kernel strides across its row, the stride is the wave
  width.
- **2.4** When a kernel is converted, it produces bit-identical output to
  the version it replaces at wave width 32 and one row per workgroup.
  This is the gate, and a speed change without it is not a result.

## 3. A launcher derives its shape

- **3.1** When a wave-cooperative kernel is launched, the grid and block
  come from `waveRowGrid` and `waveRowBlock` rather than from literals.
- **3.2** When a family has its own derived rows-per-wave rule, that rule
  and the tuned value agree on one answer rather than one overriding the
  other. `q4k` and `q6k` compute `rowsPerWave` and then discard it.
- **3.3** When a workgroup would exceed the device's thread limit, it is
  clamped, which `waveRowsPerBlock` already does.

## 4. The sweep writes

- **4.1** When no tuned value exists for this machine, the candidates are
  swept once and the winner is remembered.
- **4.2** When the sweep runs, it runs during prewarm rather than on a
  token the user is waiting for. `prewarmPrefillWeights` is that point.
- **4.3** When a candidate is timed, correctness is checked before speed.
  A tuner that can select a fast wrong answer is worse than a constant,
  which is `Autotune`'s own stated contract.
- **4.4** When the sweep has a winner, it is stored with the build id, so
  a later run against changed kernels discards it rather than trusting
  it.
- **4.5** When a stored value is recalled, it is recalled with the build
  id it was measured against.
- **4.6** When the sweep cannot run, the derived default serves, and the
  kernel is correct.

## 5. What it produces beyond the tuning

- **5.1** When a converted kernel runs on a wave64 part, it is correct
  without a rebuild. None of the forty-four are today.
- **5.2** When the sweep has run across the formats, the winners are a
  data set: fourteen kernels against four candidate block sizes. That
  is the evidence the adaptor's cost model would be fitted to, and it
  answers whether four rows per wave generalizes past the two formats
  where it was found by hand.

### 5.3 Measured

Waves per workgroup against kernel time, `bench/ParityLeg
--rows-per-block N` on real weights, min of three, in us. One wave per
workgroup is the default for every format, and Unit 3's sweep was closed
as moot on these numbers (Julian 2026-10-16).

gfx1151 (proton), 2026-09-30, the 8B Q4_K_M's own tensors:

| shape | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| q4_K 1024x4096 | 16.75 | 16.91 | 16.67 | 16.68 |
| q6_K 1024x4096 | 21.56 | 21.50 | 21.15 | 21.64 |
| q4_K 4096x4096 | 47.15 | 47.63 | 47.47 | 47.24 |
| q4_K 14336x4096 | 148.1 | 148.1 | 148.9 | 149.2 |
| q4_K 4096x14336 | 158.4 | 158.9 | 159.3 | 159.7 |
| q6_K 4096x14336 | 223.6 | 222.0 | 222.1 | 222.6 |
| q6_K head | 1907 | 1908 | 1911 | 1927 |

gfx1151, 2026-09-30, `ParityLeg --by-type` on the 8B in Q8_0, Q6_K,
Q5_K_M, Q3_K_M and Q2_K, arm order rotated:

| shape | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| q8_0 1024x4096 | 25.3 | 26.8 | 27.7 | 27.2 |
| q8_0 4096x4096 | 84.0 | 84.4 | 85.8 | 86.8 |
| q8_0 14336x4096 | 279.6 | 280.9 | 280.8 | 281.4 |
| q8_0 4096x14336 | 286.5 | 284.2 | 285.5 | 291.3 |
| q8_0 head | 2448 | 2471 | 2445 | 2448 |
| q6_K 4096x4096 | 68.0 | 67.9 | 67.6 | 67.4 |
| q6_K 14336x4096 | 221.9 | 221.7 | 221.9 | 223.3 |
| q5_K 4096x4096 | 59.8 | 61.3 | 60.9 | 60.7 |
| q5_K 14336x4096 | 186.0 | 188.3 | 189.2 | 188.4 |
| q5_K 4096x14336 | 196.6 | 198.0 | 197.3 | 193.1 |
| q3_K 4096x4096 | 40.9 | 41.0 | 40.6 | 41.2 |
| q3_K 14336x4096 | 118.6 | 119.0 | 118.8 | 120.9 |
| q3_K 4096x14336 | 125.0 | 126.7 | 124.9 | 125.0 |
| q2_K 4096x4096 | 33.5 | 34.5 | 34.3 | 34.3 |
| q2_K 14336x4096 | 100.8 | 100.1 | 100.9 | 98.8 |

- No format prefers more than one wave by more than about 2% at any cold
  shape. q8_0 1024x4096 prefers one wave by 6 to 9%.
- At one wave the large shapes stream 192 to 228 GB/s of weight, 75 to
  89% of the part's 256 GB/s. Four rows per wave, the q4_K and q6_K
  layout, has nothing left to win at decode on this part.
- The one shape where four waves pays is a weight that stays in the 32 MB
  MALL (q4_K 4096x14336 repeated, 45.5 us at one wave and 41.8 at four).
  A decode never reads a weight that way.

sm_89 (RTX 4090), 2026-09-29, one wave against four only (cajeta
xpu-kernel-adaptor 9.3.3):

| shape | 1 | 4 |
|---|---|---|
| q4_K 1024x4096 | 26.2 | 25.1 |
| q6_K 1024x4096 | 19.8 | 25.3 |
| q4_K 4096x4096 | 19.1 | 28.3 |
| q4_K 14336x4096 | 48.3 | 48.8 |
| q4_K 4096x14336 | 43.6 | 44.3 |
| q6_K 4096x14336 | 58.0 | 58.0 |
| q6_K head | 429.7 | 439.4 |

Single samples on that box swing by up to 3x, so only the min of repeats
is a result. The other formats and the two and eight wave columns are not
measured there yet (plan 4.1.2).
