# wave-row-geometry — plan

Implements [`specs/wave-row-geometry-spec.md`](../specs/wave-row-geometry-spec.md).
Unit numbers trace to spec sections as *(spec §x.y)*.

## Description

Convert the wave-cooperative kernels in `QuantKernel.cajeta` to derive
their lane, row and stride from `Group.width()`, feed them from
`waveRowGrid` / `waveRowBlock`, sweep rows per workgroup once per machine
and write the winner, and version-check the recall.

`q2kQ8WaveMatVecKernel` is the worked precedent. Its body is three lines
of change and its comment records the reason.

## Systems

- `src/main/cajeta/dev/cajeta/llm/io/QuantKernel.cajeta`, the kernels and
  their launchers.
- `cajeta.xpu.Autotune`: `recallFor`, `rememberFor`, `claimOnce`,
  `deviceKey`, `reportUnderperforming`.
- `cajeta.xpu.Group.width()` in kernel bodies, `Device.waveSize()` and
  `Device.maxThreadsPerBlock()` on the host.
- `CausalLM.prewarmPrefillWeights` as the sweep's window.
- The filtered suite via `tmp/u9/rebuild-tests.sh`.

## Deliverables

- Wave-cooperative kernels correct at any wave width and at more than one
  row per workgroup.
- One launch strategy for the family, replacing the four that exist.
- A sweep that writes a per-machine, build-checked rows-per-workgroup.
- The measured table the adaptor's cost model would be fitted to.

## Unit 1 — the version check (spec §4.5)

Smallest, independent of everything else, and it is a live correctness
hole: a rebuild that changes a kernel keeps the old winner silently.

### 1.1 TDD
- [x] 1.1.1 A value remembered against one build id is NOT recalled
      against a different one, and the discard is reported.
- [x] 1.1.2 A value remembered and recalled against the same build id is
      returned.
- [x] 1.1.3 With no stored value, the derived default serves and the
      clamp still applies.

### 1.2 Coding
- [x] 1.2.1 `waveRowsPerBlock()` calls `Autotune.recallFor` with a build
      id rather than `Autotune.recall`.
- [x] 1.2.2 The build id is derived from something that actually changes
      when the kernels change, not from a hand-bumped constant.

### 1.3 Acceptance
- [x] 1.3.1 Filtered suite green.
- [x] 1.3.2 No behaviour change on a machine with no stored value, which
      is every machine today, because nothing writes one yet.

### 1.4 Found on the way
- [x] 1.4.1 **`Autotune.recallFor` did not honour its own contract.** The
      in-process memo is keyed by `name` alone and was consulted BEFORE
      the build id was compared, so once any value was memoized every
      later `recallFor` returned it whatever build it was measured
      against. Its doc says "a stale hint is exactly the failure that is
      invisible without an alert", and the memo was what made it
      invisible. Fixed in `runtime/src/cajeta/xpu/Autotune.cajeta`: the
      memo carries the build it was verified against and answers only on
      a match.
- [x] 1.4.2 The existing coverage could not have caught it. Each arm of
      `discardsAndReportsAHintTunedForAnotherBuild` runs in its OWN
      process, so the memo is always empty by the time the differing
      build is asked. A sweep that remembers and then resolves in one
      process is the real shape, and it had no test. Added as
      `theMemoDoesNotAnswerAheadOfTheBuildCheck`.
- [x] 1.4.3 The build id is `v<vgpr>s<sgpr>l<lds>p<spill>w<wave>` from
      the kernel manifest. Measured here as `v127s27l0p0w32`. It is a
      weak hash: a semantic change that happens to leave the register
      and LDS counts identical would not be caught. It is derived rather
      than hand-bumped, which is what 1.2.2 asked for, and widening it to
      several kernels is cheap if a collision ever shows up.

## Unit 2 — the conversion, staged (spec §2, §3)

44 kernels carry `globalIdX() / 32`. They are not one shape, so they do
not convert in one commit. Each stage is bit-gated against the kernel it
replaces BEFORE any timing.

### 2.1 TDD
- [x] 2.1.1 Per converted kernel, a bit-identical check against the
      pre-conversion output at wave 32, one row per workgroup. `==`, not
      a tolerance: the arithmetic is unchanged.
      DONE 2026-09-29 (cajeta xpu-kernel-adaptor 9.1.1): `selftest/WaveGolden`
      and `src/test/fixtures/wave/goldens-wave32.tsv`, the output bits of
      22 kernels recorded on the pre-conversion tree and compared `==` by
      every run since; a missing row fails. Not one bit moved.
- [x] 2.1.2 The same kernel at rows-per-workgroup 2 and 4 is bit-identical
      to rows-per-workgroup 1. This is the property the sweep depends on
      and it is worth asserting before the sweep exists.
      DONE 2026-09-29: `WaveGeometryTest.*RowsPerWorkgroupTwoAndFourMatchOne`
      for q2k, q3k, q4k, q5k, q6k, q80 on a 64-row fixture with no two
      rows alike, `==` on the bits.
- [x] 2.1.3 A does-fire check that the converted launcher actually used
      the derived geometry, via a counter, not by reading the source.
      DONE 2026-09-29: `QuantKernel.waveGridDerivations()` and
      `WaveGeometryTest.everyConvertedLauncherDerivesItsGeometry`.

### 2.2 Coding
- [x] 2.2.1 Stage A, the wave mat-vec family: `q3k`, `q5k`, `q4k`, `q6k`,
      `q80`, `q40`, `q50`, `tq10`, `tq20`, `iq4nl`, the five `iq*` and
      `f16F32`. Body to `Group.width()`, launcher to `waveRowGrid` /
      `waveRowBlock`.
      DONE 2026-09-29 as two nvptx-gated stages (one-row and two-row
      kernels; then the q4k/q6k slot kernels and the three fused qkv
      kernels), 23 kernels and 25 launch sites, every launcher through
      `QuantKernel.waveGrid`. On cpu (wave 8, 2026-09-29 leg 07:01-07:30): every converted kernel lowers, runs and is value-checked against its host oracle at width 8 (census 156 ran / 138 value-checked, up from 137 / 116, 0 uncovered); the only cpu reds are the two plan reds 6.4.4 and 6.4.6 plus route tests whose waveMv() guard had meant "a device" and now ask their route (fixed in the same commit). The three fused qkv kernels stay held on cpu (4.2.1.10).
- [x] 2.2.2 Stage A resolves the `rowsPerWave` conflict of §3.2. `q4k`
      and `q6k` compute a derived value and then discard it for a
      literal 4. One of the two is right and the other goes.
      DONE 2026-09-29: `slotRowsPerWave` answers segments for short rows
      and the four slots (`WAVE_ROW_SLOTS`, the kernel's accumulators)
      for long ones; `rowsPerWave` keeps the segment meaning for the Id
      launchers and derives from `waveWidth()`; no launcher overwrites
      anything.
- [~] 2.2.3 Stage B, the grouped-id family: the `*Id*` kernels. Their row
      derivation carries an expert index, so the conversion is the same
      idea with a different decomposition.
      The five grouped-id MAT-VECS are DONE 2026-09-29 (cajeta
      xpu-kernel-adaptor Unit 9, its Stage C): `q4kQ8IdMatVecKernel`,
      `q6kQ8IdMatVecKernel`, `symQ8IdMatVecKernel`, `iq3xxsQ8IdMatVecKernel`,
      `iq4nlQ8IdMatVecKernel` derive lane, wave and the segment packing
      from `Group.width()`, their launchers take `waveGrid`/`waveRowBlock`,
      goldens recorded for the three with direct tests and bit-identical
      after. On cpu (wave 8, legs 11:20-12:20): 520 passed, the two plan reds 6.4.4 and 6.4.6, 88 skipped; census 159 ran / 140 value-checked / 0 stale, the five grouped-id mat-vecs and the two forced-32 Group kernels lowering there for the first time. STILL OPEN here: the grouped-id
      GLU and combine kernels (`q4kQ8IdGateUpGluKernel`,
      `q4kQ8IdDownCombineKernel`, `q6kQ8IdDownCombineKernel`,
      `iq3xxsQ8IdGateUpGluKernel`, `iq4nlQ8IdDownCombineKernel`,
      `symQ8IdDownCombineKernel`), still on `KernelCap.WAVE32_ONLY`.
- [x] 2.2.4 Stage C, the batch and utility kernels: `q4kQ8Batch*`,
      `q6kQ8Batch*`, `q8kPack`, `touchLines`, `moeTopKBatch`,
      `mxfp4QuantAct`. Convert only where the literal is a lane or row
      derivation. A stride or payload width that happens to be 32 stays.
      CLOSED ANOTHER WAY, 2026-09-30. The conversion of the twenty-nine
      remaining wave-32 kernels (the grouped-id GLU and combine family, the
      two GateUpGlu wave kernels, iq2sQ8WaveMatVecLds, the routers and
      top-k, the packs, qkPrep, the flash attend family, the mxfp4 coops,
      and q8kPack, which had been credited at wave 8 by a test whose maximum
      happened to fall in the first quarter block) was not done body by
      body. Each DECLARES its wave, `@Kernel @Wave(width = 32)`, and the
      compiler honors the declaration on the cpu backend (cajeta a538aa77):
      the work-item loop is laid out at 32 lanes, Wave.width() folds to 32,
      the reduces span 32, and the manifest records waveWidth 32 for
      Group.laneBlockOf. The same source runs unchanged on every backend,
      which is the whole point of the annotation the spec had carried since
      the beginning. KernelCap.WAVE32_ONLY, wave32Only and the "written for
      a 32-lane wave" skip are gone; usable() is registered and not measured
      wrong. Found on the way and fixed in the compiler: the cpu segmented
      reduce ignored its segment (right while the host wave was never wider
      than a segment, wrong at 32 with eight-lane segments: the Gqa1 flash
      decode), now a select-guarded butterfly (XpuCpuDeclaredWaveTests).
      Found and fixed in the engine: isDeviceRouted answered for a format
      that could not be launched (a TQ2_0 body on cpu threw from the f32
      launch table once the batched route opened there); the Id GEMM arms
      launched q4kWmmaIdMwKernel without asking; and packed activations were
      off on cpu BY NAME, closing every packed-only route there (11.2.5
      measures which form is faster, 6.4.9 says the gate is a capability).
- [x] 2.2.5 An audit that no converted family still divides by a literal.
      DONE 2026-09-30 as a script over every @Kernel: a kernel that uses a
      wave op either derives its lanes from Group.width() / Wave.width() or
      declares @Wave(width = 32); after q8kPackKernel took the declaration,
      none is left that does neither. The cpu leg: 601 passed, 0 failed, 43
      skipped (from 549 / 2 / 88), census 192 ran / 177 value-checked / 0
      uncovered / 0 stale, 5 cannot and 36 tracked device skips, under
      CAJETA_XPU_POISON=1, 22:07 to 22:26.

### 2.3 Acceptance
- [x] 2.3.1 Bit gate passes per kernel before any speed number is quoted.
      DONE 2026-09-29, see 2.1.1.
- [ ] 2.3.2 Legs flat on the recorded files. This is a refactor, and a
      change in either direction is a geometry change to explain.
- [x] 2.3.3 Filtered suite green at each stage, not only at the end.
      DONE 2026-09-29: the whole nvptx suite, not a filter, after each
      stage (605/0 and 606/0).

## Unit 3 — the sweep that writes (spec §4)

### 3.1 TDD
- [ ] 3.1.1 With no stored value, the sweep runs once and stores one.
- [ ] 3.1.2 With a stored value for this build, the sweep does not run.
- [ ] 3.1.3 A candidate that fails the correctness check is not selected
      even when it is fastest.
- [ ] 3.1.4 The sweep runs during prewarm, asserted by a counter, so no
      token pays for it.

### 3.2 Coding
- [ ] 3.2.1 The sweep over rows-per-workgroup 1, 2, 4, 8, clamped by
      `waveRowsPerBlock`'s existing device limit.
- [ ] 3.2.2 Min-of-repeats per candidate rather than one sample.
- [ ] 3.2.3 `Autotune.claimOnce` so concurrent processes do not both
      sweep, and `rememberFor` with the build id.
- [ ] 3.2.4 Correctness before speed, per `Autotune`'s own contract.

### 3.3 Acceptance
- [ ] 3.3.1 A cold machine sweeps once, a warm one never.
- [ ] 3.3.2 The stored winner survives a restart and is discarded by a
      rebuild.

## Unit 4 — the measured table (spec §5.2)

- [x] 4.1.1 Run the sweep across the formats on gfx1151 and record the
      winners against the hand-found values, including whether four rows
      per wave generalizes past `q4k` and `q6k`.
      q4_K and q6_K SWEPT on proton 2026-09-30 (`bench/ParityLeg
      --rows-per-block N`, the 8B Q4_K_M's own tensors, min of three, us):

      | shape | 1 | 2 | 4 | 8 |
      |---|---|---|---|---|
      | q4_K 1024x4096 | 16.75 | 16.91 | 16.67 | 16.68 |
      | q6_K 1024x4096 | 21.56 | 21.50 | 21.15 | 21.64 |
      | q4_K 4096x4096 | 47.15 | 47.63 | 47.47 | 47.24 |
      | q4_K 14336x4096 | 148.1 | 148.1 | 148.9 | 149.2 |
      | q4_K 4096x14336 | 158.4 | 158.9 | 159.3 | 159.7 |
      | q6_K 4096x14336 | 223.6 | 222.0 | 222.1 | 222.6 |
      | q6_K head | 1907 | 1908 | 1911 | 1927 |

      Flat within about 1% at every cold engine shape, so one wave per
      workgroup stays the default, as on sm_89 (9.3.3). The one place four
      waves pays is a weight that stays in the 32 MB MALL (the q4_K
      4096x14336 control: 45.5 us at 1, 41.8 at 4), which a decode never
      is. BLOCKED for the other formats: ParityLeg reads only the Q4_K_M
      layout and refuses a tensor of another type by name.
      UNBLOCKED and SWEPT 2026-09-30: `ParityLeg --by-type` groups any
      model's projection weights by (type, rows, cols) and runs each group on
      its own tensors. The 8B in Q8_0, Q6_K, Q5_K_M, Q3_K_M and Q2_K, one to
      eight waves per workgroup, three reps with the arm order rotated, min
      of three, us:

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

      No format prefers more than one wave by more than about 2% at any
      cold shape (the largest, q5_K 4096x14336 and q2_K 14336x4096 at 8,
      are 2.2% and 2.0%), and q8_0 1024x4096 prefers one by 6 to 9%. One
      wave per workgroup stays the default for every format.
      Four rows per wave, the q4k and q6k layout, has nothing left to win
      at decode on this part: at one wave the other formats' large shapes
      already stream 192 to 228 GB/s of weight (q8_0 218 to 228, q5_K 206
      to 217, q3_K 202 to 213, q2_K 192), the same band as q4_K and q6_K
      (200 to 226) and 75 to 89% of the 256 GB/s peak. The small shapes
      are lower for every format alike, which is the fixed launch cost.
- [ ] 4.1.2 The same on the NVIDIA box, which is a wave-32 part with
      different everything else.
- [ ] 4.1.3 Record the table in the spec, where the adaptor draft can
      read it.
