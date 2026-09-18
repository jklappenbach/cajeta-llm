# Host floor — spec

Status: **draft** 2026-09-18. Registered in [INDEX.md](INDEX.md).
Package: cajeta-llm. Touches `io/Quant.cajeta`, `io/GgufFile.cajeta`,
`io/Checkpoint.cajeta`, `model/Linear.cajeta`, and
`selftest/TernaryTest.cajeta`.

## 1. Definition

Every format this package loads must run on the host. The device routes
are optimizations above that floor, and which of them exist is a
separate question from whether the format is usable at all.

Today those two questions are one boolean, `Quant.supported(ty)`, and it
answers the second — a format is "supported" once its DEVICE routes
exist. The floor is not checked by anything.

### 1.1 Problem statement

Measured 2026-09-18, not argued.

- `Quant.supported(ty)` lists **21** types. `Linear.matvecInto`'s host
  chain serves **16**. The other five reach its terminal
  `throw "no host mat-vec for packed type N"`.
- Four of those five — **IQ4_NL, IQ4_XS, Q4_1, Q5_1** — HAVE a host
  mat-vec in `Quant` (`iq4nlMatVecIntoAt`, `iq4xsMatVecIntoAt`,
  `q41MatVecIntoAt`, `q51MatVecIntoAt`). Each is tested by `QuantTest`
  and `LegacyWaveMatVecTest`, and each has **zero production callers**.
  `Linear.matvecInto` simply has no arm for them.
- The fifth, **MXFP4**, has no host mat-vec at all: four decode
  functions in `Quant`, five kernels and seven launchers in
  `QuantKernel`, every one of them called only from `Mxfp4Bench`.
- So those five formats run on a GPU and fail on the host. IQ4_NL is
  one of them, and it was optimized this week.

Nothing caught it because nothing looks. `TernaryTest.theFourPartInvariant`
asserts `supported(ty)` equals `QuantKernel.hasKernel(ty)`,
`QuantKernel.coopSupports(ty)` and `Linear.packedSupported(ty, 4096)` —
**three device predicates**. The host path is not one of the four parts,
so the gate that is supposed to say a format is whole is structurally
incapable of checking the one thing that makes it runnable. It also
exempts MXFP4 by name, which is how it stays green over the
disagreement between `hasKernel` (no MXFP4) and `matVecLaunch` (an
MXFP4 arm).

The gate's shape has a second cost. Because it will not open until
every device route exists, a format cannot ship incrementally, and a
second list grew to escape it:

    static boolean decodable(int32 ty) {
        return Quant.supported(ty)
            || ty == Quant.GG_IQ1_S || ty == Quant.GG_IQ1_M;
    }

whose own comment says `supported` "may not say yes until all four
parts are there". `blockElems` and `blockBytes` already key off
`decodable` rather than `supported`, so half the package has quietly
moved off the gate already.

### 1.2 Scope

What it means for a format to be loadable, where that is refused, and
the invariant that keeps it true. The predicate that answers it, its
call sites, and the retirement of `supported` and the four-part gate.
The five formats that do not meet the floor today are brought to it.

### 1.3 Non-goals

- **Which device route serves a format.** That is the route table
  (codebook-quants 9.2.9). This spec says a format runs; the table says
  what runs it fastest.
- **Kernel performance.** A host mat-vec is a floor, not a target. It
  must be correct, and it must exist.
- **Making wave-cooperative kernels portable.** Recorded separately:
  135 hard-32 sites in `QuantKernel.cajeta`, `Wave.width()` read zero
  times, and `Device.waveSize()` answering 0 on the CPU backend because
  the backend never publishes the width it already computes.

## 2. The floor

A format is **runnable** when its decoder and its host mat-vec both
exist. That is the whole gate, and it is the same on every device.

- **2.1** When a format has a decoder and a host mat-vec, it loads.
- **2.2** When a format lacks either, the file is refused AT OPEN,
  naming the format and which part is missing — not at first use of a
  tensor, which fails mid-generation and far from the cause.
- **2.3** When a format is runnable but has no device route, it loads
  and runs on the host. Slowly is a performance result, not an error.
- **2.4** When a format gains a device route later, nothing about its
  loadability changes.
- **2.5** When a device is absent or refuses, every runnable format
  still produces output.

## 3. The predicate

`Quant.runnable(ty)` replaces both `supported(ty)` and `decodable(ty)`.

- **3.1** When `supported(ty)` is read for a load decision
  (`Checkpoint:175`, `GgufFile:302/325/359/430`, `GgufWriter:442`), it
  is `runnable(ty)`.
- **3.2** When `decodable(ty)` is read for block geometry
  (`blockElems`, `blockBytes`), it is `runnable(ty)`, and the IQ1
  escape hatch goes with it — there is nothing left to escape.
- **3.3** When a caller wants to know which device routes serve a
  format, it asks the route table, not a predicate. `hasKernel`,
  `coopSupports` and `packedSupported` stop being a definition of
  support and become what they describe.

## 4. The invariant

- **4.1** When any type is runnable, `Linear.matvecInto` has an arm for
  it — asserted over every type, and this is the check that does not
  exist today.
- **4.2** When a host mat-vec exists in `Quant`, something other than a
  test calls it, or it is declared dead — the four unused ones are how
  this defect hid.
- **4.3** When the assertion fails, it names the type and the missing
  part, so the fix is one arm and not a bisection.
- **4.4** When `theFourPartInvariant` is replaced by 4.1, no type is
  exempt by name. MXFP4's exemption exists only because the invariant
  asserts something untrue about it.

## 5. Bringing the five to the floor

- **5.1** When IQ4_NL, IQ4_XS, Q4_1 or Q5_1 reaches `matvecInto`, it
  dispatches to the `Quant` function that already exists and is already
  tested.
- **5.2** When MXFP4 reaches `matvecInto`, it dispatches to a host
  mat-vec written for it — the one format whose floor is genuinely
  missing rather than unwired. Correctness against the scalar decode is
  the bar; speed is not.
- **5.3** When the five are wired, 4.1 passes without exemptions.

## 6. Formats ship incrementally

The gate's ordering constraint is what forced codebook-quants Unit 8 to
sequence IQ1_S/IQ1_M as "decoders; host; kernels; registration; gates
last". With the floor as the gate, that ordering has no reason to exist.

- **6.1** When a format has a decoder and a host mat-vec, it ships —
  its device kernels land afterwards, each an optimization measured
  against the floor it already has.
- **6.2** When IQ1_S and IQ1_M reach the floor, they load and generate,
  before any IQ1 kernel is written.
- **6.3** When Unit 8's plan items are rewritten (8.2.1), "gates last"
  is replaced by "floor first", and the `decodable` escape hatch that
  existed for exactly these two formats is deleted rather than extended.

## 7. What follows

- The route table's audit (codebook-quants 9.2.9.7) builds its query
  set from `runnable(ty)`, so it inherits a list that means one thing.
- The four MXFP4 device kernels with no row, and the wave-cooperative
  portability gap, are the route table's and the kernels' — named in
  §1.3 so this spec does not absorb them.
- A kernel-coverage check — every `*Launch*` in `QuantKernel` has a row
  or is declared bench-only — would have caught the MXFP4 stranding.
  It belongs with the route table, not here.
