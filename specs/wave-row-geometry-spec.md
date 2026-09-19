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
