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
- [ ] 2.1.1 Per converted kernel, a bit-identical check against the
      pre-conversion output at wave 32, one row per workgroup. `==`, not
      a tolerance: the arithmetic is unchanged.
- [ ] 2.1.2 The same kernel at rows-per-workgroup 2 and 4 is bit-identical
      to rows-per-workgroup 1. This is the property the sweep depends on
      and it is worth asserting before the sweep exists.
- [ ] 2.1.3 A does-fire check that the converted launcher actually used
      the derived geometry, via a counter, not by reading the source.

### 2.2 Coding
- [ ] 2.2.1 Stage A, the wave mat-vec family: `q3k`, `q5k`, `q4k`, `q6k`,
      `q80`, `q40`, `q50`, `tq10`, `tq20`, `iq4nl`, the five `iq*` and
      `f16F32`. Body to `Group.width()`, launcher to `waveRowGrid` /
      `waveRowBlock`.
- [ ] 2.2.2 Stage A resolves the `rowsPerWave` conflict of §3.2. `q4k`
      and `q6k` compute a derived value and then discard it for a
      literal 4. One of the two is right and the other goes.
- [ ] 2.2.3 Stage B, the grouped-id family: the `*Id*` kernels. Their row
      derivation carries an expert index, so the conversion is the same
      idea with a different decomposition.
- [ ] 2.2.4 Stage C, the batch and utility kernels: `q4kQ8Batch*`,
      `q6kQ8Batch*`, `q8kPack`, `touchLines`, `moeTopKBatch`,
      `mxfp4QuantAct`. Convert only where the literal is a lane or row
      derivation. A stride or payload width that happens to be 32 stays.
- [ ] 2.2.5 An audit that no converted family still divides by a literal.

### 2.3 Acceptance
- [ ] 2.3.1 Bit gate passes per kernel before any speed number is quoted.
- [ ] 2.3.2 Legs flat on the recorded files. This is a refactor, and a
      change in either direction is a geometry change to explain.
- [ ] 2.3.3 Filtered suite green at each stage, not only at the end.

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

- [ ] 4.1.1 Run the sweep across the formats on gfx1151 and record the
      winners against the hand-found values, including whether four rows
      per wave generalizes past `q4k` and `q6k`.
- [ ] 4.1.2 The same on the NVIDIA box, which is a wave-32 part with
      different everything else.
- [ ] 4.1.3 Record the table in the spec, where the adaptor draft can
      read it.
