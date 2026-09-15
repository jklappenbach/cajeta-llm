# Codebook and ternary quants — plan (Unit 34)

Implements [`specs/codebook-quants-spec.md`](../specs/codebook-quants-spec.md).
Approved 2026-09-15 (Julian: "support as many model architectures /
quantizations as we can"; "do it for all — I want simplification where
we can get it").

**Work:** bring the seven codebook types (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, IQ3_S) and the two ternary types (TQ1_0, TQ2_0) through
the four-part stack — decoder, host mat-vec, bind gate, device route —
on one resident layout that every packed format shares; add the
`bitnet` architecture and a cajeta HF-to-GGUF converter so a ternary
file exists to run; migrate the six older formats that hold two device
copies today onto the same layout and delete the repack machinery.

**Systems:** `io/Quant.cajeta` (constants, decoders, host mat-vecs, the
layout table), `io/QuantKernel.cajeta` (split kernel, wave decode
kernels, coop X1/X3 kernels, registration tables, resident grid
tables), `io/WmmaKernel.cajeta` (the widen kernels of the migrated
formats), `io/IqGrid.cajeta` (new; generated), `io/GgufFile.cajeta`
(`typeName`, refusal text), `io/GgufWriter.cajeta`, `io/SafetensorsFile`,
`tok/SpProto`, `model/Linear.cajeta` (`payloadDev`/`scalesDev`, upload
streaming, `packedSupported`, host chain, `coopRouteBit`),
`model/ExpertBank.cajeta` (window split), `model/ModelConfig.cajeta`
and the layer graph (bitnet), `tools/convert/HfToGguf.cajeta` (new);
llama.cpp 5306f4b (`llama-imatrix`, `llama-quantize`,
`llama-perplexity`, the HIP and Vulkan builds) as the oracle; the
cajeta repo's `tmp/llmbench/leg.sh` for every timing leg.

**Deliverables:** nine new formats loading, decoding bit-exactly,
prefilling `batched` and decoding wave-per-row; resident device bytes
equal to file bytes for every packed format, with `coopDev`, the Q6_K
pad copy and both repack kernels gone; the `bitnet` graph; the
converter and a converted `bitnet_b1_58-3B` at f16/tq1_0/tq2_0;
eleven IQ arbiter files with llama.cpp reference numbers; a parity
table per unit in this plan.

**Order:** the layout first (Unit 1), proven on Q8_0 — every later
kernel is written against it once. Reference material next (Unit 2),
because fixtures, tables and arbiter files gate everything after. The
ternary pair (Unit 3) is the four-part template on the new layout with
the smallest kernels; the architecture and converter (Unit 4) give it a
real model. The codebook tier follows by kinship — ksigns family,
raw-sign family, the IQ1 pair — and the migration of the older formats
closes the plan, once the split path has carried nine formats.

**Rules carried:** four parts or an honest refusal (a type enters
`Quant.supported()` and `Linear.packedSupported` last, in the same
commit as its kernels); bit gate before any speed number; announce
every timing leg and wait for the go; filtered suite only
(`tmp/u9/rebuild-tests.sh lib`); no python under `tools/`; scratch under
`tmp/cbq/`; comments only at signatures.

---

## Unit 1 — One resident layout, proven on Q8_0 (spec §10, 1.4)

### 1.1 TDD
- [x] 1.1.1 `ResidentLayoutTest.theLayoutTableIsAPermutationOfTheBlock`:
      for every `Quant.supported()` type, `Quant.payloadBytes(ty) +
      Quant.scaleBytes(ty) == Quant.blockBytes(ty)` and
      `payloadBytes % 4 == 0`; `scaleBytes` is 0 exactly for Q4_1, Q5_1,
      Q2_K, Q4_K, Q5_K, IQ4_XS; `scaleOffset` is 0 for the legacy and
      k formats and `blockBytes - 2` for the ternary pair.
- [x] 1.1.2 `ResidentLayoutTest.theSplitKernelMatchesTheHostSplit`: a
      Q8_0 fixture tensor through `QuantKernel.splitLaunch` yields
      payload bytes equal to file bytes 2..34 of every block and scales
      equal to bytes 0..2, byte for byte against `Quant.splitInto` on
      the host; also at an odd block count and for a tensor larger than
      one staging chunk.
- [x] 1.1.3 The Q8_0 cases of `LegacyWaveMatVecTest` and `QuantKernelTest`
      re-pointed: `q8MatVecKernel`, `q80F32WaveMatVecKernel`,
      `q80Q8WaveMatVecKernel` over `(payload, scales)` equal the host
      mat-vec to the bars they meet today.
- [x] 1.1.4 `q80F16CoopX1/X3` over the split layout equal the host GEMM
      at the existing bar; `coopBlockWords(Q8_0) == 8`; after both routes
      run, `devWBytes == payload + scales` bytes — no `coopDev`.
- [x] 1.1.5 `q80WidenKernel` and `q80WmmaDeqMw8Kernel` read the split
      layout; the Mw8 route's existing parity test passes.
- [x] 1.1.6 `ExpertBankTest`: a window of Q8_0 experts splits on upload
      and the slot Linear's kernels read it.
- [x] 1.1.7 With `CAJETA_XPU_ALLOC_TRACE`, binding a mapped Q8_0 tensor
      allocates payload, scales and one staging chunk; the chunk is
      shared across tensors, so the peak is resident plus one chunk.

### 1.2 Coding
- [x] 1.2.1 `Quant.payloadBytes / scaleBytes / scaleOffset(ty)` — one
      table for every format the engine carries (scale at 0 for legacy
      and k formats, at the end for TQ, MXFP4's e8 byte); `Quant.splitOn(ty)`
      names the formats split today (Q8_0 now; each new format as it
      lands; the five older ones in Unit 8, where the predicate dies);
      host `Quant.splitInto`.
- [x] 1.2.2 `QuantKernel.splitKernel(payload, scales, src, nBlocks,
      blockBytes, payloadBytes, scaleOffset)` — one lane per payload
      dword, coalesced stores; `splitLaunch`.
- [x] 1.2.3 `Linear`: `packedDev` becomes `payloadDev` + `scalesDev`
      (`scalesDev` null when `scaleBytes == 0`); `ensureDevice` and
      `streamPackedFromMap` stream through one shared static staging
      buffer and split; `stageExpertWindowStaged` and `bindExpertSlot`
      split instead of repack; `coopW = payloadDev.wordView()`.
      REVISED 2026-09-15 after 1.3.2: the two-array form lost 30 % of
      Q8_0 decode (a second stream per wave, and a power-of-two row
      stride camping on channels once the scales left the row — see the
      spec's 1.4 and 12.3). The resident layout is PER ROW — the row's
      scales padded to a dword, then its payloads — in ONE buffer:
      `payloadDev` alone, `Quant.rowPrefixBytes/rowResidentBytes`, the
      split kernel placing block `blockBase + b` by row, every Q8_0
      kernel back to its original operand list, the scale read one
      aligned dword (`scaleAtDev`), and `symScaleImageKernel` given the
      row stride. `scalesDev`, `bindExpertSlicePair` and the second
      slab are gone.
- [x] 1.2.4 The eight Q8_0 kernels re-offset to `(payload, scales)`;
      `coopBlockWords(Q8_0) = 8`; `coopNeedsRepack(Q8_0)` false.
- [x] 1.2.5 Formats not yet split keep their kernels and repack path
      reading `payloadDev` as they read `packedDev` — nothing else moves
      in this unit.

### 1.3 Acceptance
- [x] 1.3.1 Filtered suite green.
- [x] 1.3.2 8B Q8_0 legs (announced, quiet box): pp512 / tg128 within
      noise of the 2026-09-14 rows; bit gate first.
      MEASURED 2026-09-15, `leg.sh cajeta <8B Q8_0> 512x128 3`, arms
      alternating new/old/new/old, suite 150/150 before every arm:

        two-array split (7d357eb)  pp 1614..1650   tg 17.2..18.4
        old 95608c6                pp 1470..1657   tg 23.3..25.7
        one-dword scale read       pp 1648..1658   tg 18.3..18.5
        old 95608c6                pp 1633..1661   tg 25.71..25.74
        ROW-SPLIT (shipped)        pp 1654..1661   tg 25.79..25.82
        old 95608c6                pp 1645..1659   tg 25.69..25.74

      The first layout lost 30 % of decode. Not the scale read (one
      aligned dword changed nothing), not registers or ISA
      (`tmp/cbq/isa/kernelisa`: 33 vs 32 VGPRs, no spill, the same load
      and waitcnt mix), but the memory system: the profiler put the
      whole delta inside `q80Q8WaveMatVecKernel` itself (161 -> 226 us
      per launch), and `tmp/cbq/src/.../SplitProbe.cajeta` showed a
      second stream per wave AND a power-of-two row stride camping on
      channels once the scales left the row (file 217 GB/s, split
      arrays 126, payload-only 144, per-row 220 on 4096x4096). The
      per-row split is what ships: one buffer, the file's row stride,
      dword-clean payloads, bit-exact, and 0.3 % faster than the file
      layout in decode.
- [x] 1.3.3 Qwen3-Coder-30B Q8_0: resident weight bytes equal the file's
      (ledger), against the two-copy number measured before the change;
      load time not worse. Both numbers recorded here.
      MEASURED 2026-09-15 with `CAJETA_XPU_ALLOC_TRACE`, prompt 512 /
      gen 8, the 95608c6 bench binary against this tree
      (`tmp/cbq/resident30b.sh`, ledgers under `tmp/cbq/`):

        before  peak 34.62 GB  steady 34.41 GB  load 19.0 s
                prefill 1236 tok/s  decode 55.5 tok/s (8 tokens)
        after   peak 34.66 GB  steady 34.45 GB  load 11.4 s
                prefill 1259 tok/s  decode 48.6 tok/s (8 tokens)

      The resident number did not move because the two-copy case never
      arose on this workload: neither ledger holds an `ensureCoopW`
      allocation. The Q8_0 projections prefill through the int8 widen
      (`deqFor`), and the expert slab is single-copy once its widened
      twin exists (the packed slab is released). The file is 32.48 GB;
      the excess is the int8 widen twin (0.96 GB for the projections;
      the 30.8 GB expert twin replaces the packed slab) plus 0.8 GB of
      batch scratch. That twin is the one-copy rule's remaining case and
      is item 8.2.2. Load fell by 40 % because the split streams the slab
      from the mapping in chunks and no 30 GB host array exists any
      more (a first cut copied it byte by byte and doubled the load
      time; fixed before this record). Host RSS reads 32.3 GB against
      23.8 because the mapping's pages stay resident in place of the
      freed host array. The 8-token decode delta was the two-array
      regression 1.3.2 found and the row-split layout removed.

## Unit 2 — Reference material: fixtures, tables, names, arbiter files (spec §2, §3.5, §4.1, §9)

### 2.1 TDD
- [ ] 2.1.1 `IqGridTest.everyTableChecksumsToTheGeneratorsValue`: the five
      grids, `ksigns_iq2xs`, `kmask_iq2xs` and the packed forms (iq1s
      nibbles, the two-bit IQ2 alphabet) — a position-weighted checksum
      equal to the value `tmp/cbq/grids.c` printed, pinned in the test.
- [ ] 2.1.2 `GgufFileTest.typeNameKnowsTheCodebookAndTernaryIds`: ids 16,
      17, 18, 19, 21, 22, 29, 34, 35 name themselves; the loader's
      refusal text lists every `supported()` type.
- [ ] 2.1.3 `QuantTest.dequantizeRefusesATypeWithNoBranch`, naming it.
- [ ] 2.1.4 `QuantTest.theManifestCoversEveryFixture`: every `.bin` in the
      fixture directory has an entry with a matching block count.

### 2.2 Coding
- [ ] 2.2.1 `tmp/cbq/gen.c` (from `tmp/q1fix/gen.c`): nine fixture pairs
      over the same token_embd values, a synthetic positive importance
      vector handed to all seven IQ quantizers, the TQ1_0 fixture checked
      for distinct values in all three regions; committed with the
      manifest regenerated.
- [ ] 2.2.2 `tmp/cbq/grids.c`: emits `io/IqGrid.cajeta` (byte-per-value
      and packed forms as static arrays) and prints the checksums.
- [ ] 2.2.3 `GgufFile.typeName` six ids; refusal text from `supported()`;
      `Quant.dequantize` throws on a type with no branch.
- [ ] 2.2.4 Arbiter files under `tmp/cbq/`: `llama-imatrix` over the
      Q8_0 8B on a calibration text distinct from the perplexity text;
      `llama-quantize --imatrix --allow-requantize` to IQ1_S, IQ1_M,
      IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M;
      `llama-imatrix` over Qwen1.5-MoE Q4_K_M and an IQ3_XXS of it.
      Reference numbers per file — HIP and Vulkan pp512 / pp2048 /
      tg128 through `leg.sh`, `llama-perplexity` on the README text —
      announced, quiet box, recorded in 2.3.1.
- [ ] 2.2.5 The two compiler findings filed in the cajeta repo:
      `cajeta.xpu.Constant<T>` declared and unwired; `@FastMath` folding
      `fpext(fptrunc x)` to x. Placement per Julian.

### 2.3 Acceptance
- [ ] 2.3.1 Fixtures, manifest and `IqGrid.cajeta` committed with their
      tests; llama.cpp loads and runs all eleven arbiter files; the
      reference table is in this record.

## Unit 3 — TQ2_0 and TQ1_0: the four-part template on the new layout (spec §3.3, §5, §6, §7, §8.4)

### 3.1 TDD
- [ ] 3.1.1 `QuantTest.tq20MatchesReferenceExactly`, `tq10…` — exact.
- [ ] 3.1.2 `QuantTest.tqHostMatVecMatchesTheDequantizedReference`
      (`checkMatVec`, both), and the Q8 twins against the f32 host at the
      Q8 route's bar.
- [ ] 3.1.3 `tq20Q8WaveMatVecKernel` / `tq10Q8WaveMatVecKernel` equal the
      Q8 host twin exactly (the path is integer) over the split layout.
- [ ] 3.1.4 `tq20/tq10F16CoopX1/X3` equal the host GEMM at the IQ4 bar;
      `coopBlockWords` 16 / 13; `coopColsOk` at 256.
- [ ] 3.1.5 `QuantTest.theFourPartInvariant`: over every `supported()`
      type, `packedSupported`, `coopSupports`, `hasKernel` and the host
      chain agree — a type is admitted by all or by none.
- [ ] 3.1.6 `KernelIsa`: no spill; the `v_dot4` count per block matches
      the design.

### 3.2 Coding
- [ ] 3.2.1 `GG_TQ1_0` / `GG_TQ2_0`; `blockBytes` 54 / 66; `blockElems`
      256; `payloadBytes` 52 / 64 with `scaleOffset` at the end;
      decoders; `dequantize` branches.
- [ ] 3.2.2 `tq10MatVecIntoAt` / `tq20MatVecIntoAt` and `…IntoQ8`; the
      `Linear.matvecInto` host chain.
- [ ] 3.2.3 `tq20Q8WaveMatVecKernel` (two-bit fields to int8 lanes,
      `dotAccum`), `tq10Q8WaveMatVecKernel` (×pow3, ×3, >>8 in integer);
      coop X1/X3 staging d·trit to f16; launchers; `matVecLaunch`,
      `hasKernel`, `coopBatchLaunch`, `coopBlockWords`, `coopRouteBit`.
- [ ] 3.2.4 `Quant.supported()`, `splitOn`, `Linear.packedSupported` —
      last.

### 3.3 Acceptance
- [ ] 3.3.1 Filtered suite green; ISA clean. End-to-end waits for Unit 4.

## Unit 4 — The `bitnet` architecture and the converter (spec §8)

### 4.1 TDD
- [ ] 4.1.1 `HfToGgufTest.weightQuantMatchesTheReference`: a hand-built
      4×4 tensor → scale = mean|w|, every value in {−s, 0, s}; the 1e-5
      clamp on an all-zero tensor.
- [ ] 4.1.2 `HfToGgufTest.tqPackersMatchGgmlByteForByte`: ternary × scale
      values through our packers equal ggml's `ggml_quantize_chunk`
      over the same values (new fixture pairs `tq1_0-ternary`,
      `tq2_0-ternary` from `tmp/cbq/gen.c`).
- [ ] 4.1.3 `HfToGgufTest.roundTrip`: a synthetic two-layer
      `BitnetForCausalLM` directory (config.json, one safetensors shard
      written by a test helper, a tiny `tokenizer.model`) converts at
      f16 and tq2_0; `GgufFile` opens both; `ModelConfig.fromGguf` reads
      `bitnet` hparams; every tensor of llama.cpp's `bitnet` list is
      present, `output.weight` absent, norms F32, `token_embd` F16,
      projections TQ2_0.
- [ ] 4.1.4 `CausalLMTest.theBitnetGraphAppliesTheSubNorms`: a one-layer
      bitnet model on the host equals a reference computed in the test
      — `attn_sub_norm` before `attn_output`, `ffn_sub_norm` before
      `ffn_down`, NEOX rope, tied head; with `.scale` tensors bound the
      projections are multiplied.
- [ ] 4.1.5 `ModelConfigTest`: `bitnet` accepted; the refusal text names
      it among the supported architectures.

### 4.2 Coding
- [ ] 4.2.1 `tools/convert/HfToGguf.cajeta` — `HfToGguf <dir> <out.gguf>
      f16|tq1_0|tq2_0`: config → hparams (`bitnet.*`, rope scaling
      linear 1.0), shards through `model.safetensors.index.json`, the
      llama and bitnet name maps, `weight_quant` in f32, the TQ packers,
      `SpProto` → `tokenizer.ggml.*`, `GgufWriter`.
- [ ] 4.2.2 `ModelConfig.fromGguf` bitnet branch; NEOX rope for the arch;
      `attn_sub_norm` / `ffn_sub_norm` bound and applied in the layer;
      tied lm_head when `output.weight` is absent; optional `.scale`
      tensors as a scalar multiply after the projection.

### 4.3 Acceptance
- [ ] 4.3.1 `1bitLLM/bitnet_b1_58-3B` converted at f16, tq1_0 and tq2_0
      under `tmp/cbq/`; llama.cpp loads all three; `llama-quantize`
      from our f16 to TQ1_0 and TQ2_0 gives projection tensors byte
      for byte equal to ours (compared per tensor through `GgufFile`).
- [ ] 4.3.2 Greedy agreement with llama.cpp CPU on both TQ files over
      fixed prompts; perplexity on the README text matches
      `llama-perplexity` at the same window.
- [ ] 4.3.3 Legs (announced): cajeta pp512 / tg128 on both files against
      llama.cpp CPU; decode GB/s beside the Q4_K wave kernel's on the
      8B, so the bandwidth claim of spec 8.5 is a number.
- [ ] 4.3.4 Resident bytes equal file bytes.

## Unit 5 — IQ2_XXS, IQ2_XS, IQ3_XXS: the ksigns family (spec §3.4, §4, §6.2, §7.2)

### 5.1 TDD
- [ ] 5.1.1 Decoders exact, three fixtures.
- [ ] 5.1.2 Host mat-vecs and Q8 twins, three formats.
- [ ] 5.1.3 `IqGridDeviceTest`: a probe kernel gathers every entry of the
      resident device tables (byte and packed forms) and expands all 128
      `ksigns` indices; equal to the host arrays.
- [ ] 5.1.4 Wave decode kernels equal the Q8 twins exactly — 2s+1 per
      sub-block, the 0.125 / 0.25 tails once per block.
- [ ] 5.1.5 Coop X1/X3 with the LDS-staged table equal the host GEMM;
      `coopBlockWords` 16 / 18 / 24.
- [ ] 5.1.6 The four-part invariant extended; ISA: no spill.

### 5.2 Coding
- [ ] 5.2.1 `QuantKernel.ensureIqTables()` — static resident buffers
      uploaded once (the `pfSink` pattern), passed as kernel operands.
- [ ] 5.2.2 Decoders; host; wave kernels (one gather, one sign expand,
      one `dotAccum` per 8); coop staging through the LDS table;
      registration; `supported` / `splitOn` / `packedSupported` last.

### 5.3 Acceptance
- [ ] 5.3.1 IQ2_XXS, IQ2_XS, IQ3_XXS 8B files: `batched`, no
      `batch-refused`; legs against llama.cpp HIP and Vulkan (pp ≥ 1.0×,
      tg ≥ 0.95×); greedy agreement; perplexity within the floor;
      resident bytes equal file bytes.

## Unit 6 — IQ2_S, IQ3_S: raw signs and qh high bits (spec §3.4, 12.1)

### 6.1 TDD
- [ ] 6.1.1 Decoders exact, two fixtures; IQ3_S's `1 + 2s` scale.
- [ ] 6.1.2 Host mat-vecs and Q8 twins.
- [ ] 6.1.3 Wave decode kernels and coop X1/X3; `coopBlockWords` 20 / 27.
- [ ] 6.1.4 The 12.1 harness: the IQ2_S wave kernel with an LDS-table arm
      and an L1-table arm, bit-identical outputs, timed alternating on
      idle; the choice recorded here with both numbers.

### 6.2 Coding
- [ ] 6.2.1 Decoders; host; kernels; registration; gates last.

### 6.3 Acceptance
- [ ] 6.3.1 IQ2_S, IQ2_M, IQ3_S, IQ3_XS, IQ3_M 8B files (the mixes
      exercise Units 5 and 6 together): legs, greedy agreement,
      perplexity, resident bytes.
- [ ] 6.3.2 The IQ3_XXS Qwen1.5-MoE: every expert tensor `batched`,
      perplexity within the MoE floor of llama.cpp's, legs.

## Unit 7 — IQ1_S, IQ1_M (spec §3.2, §6.3)

### 7.1 TDD
- [ ] 7.1.1 Decoders exact: IQ1_M's f16 rebuilt from four nibbles, the qh
      nibble split, the delta signs.
- [ ] 7.1.2 Host mat-vecs and Q8 twins with the delta term through the
      pack's per-32 sums (IQ1_S) and a per-8 sum (IQ1_M).
- [ ] 7.1.3 Wave decode kernels and coop X1/X3; `coopBlockWords` 12 / 14;
      `scaleBytes(IQ1_M) == 0`.
- [ ] 7.1.4 12.1 re-measured on the iq1s table (16 KB bytes, 8 KB nibbles).

### 7.2 Coding
- [ ] 7.2.1 Decoders; host; kernels; registration; gates last.

### 7.3 Acceptance
- [ ] 7.3.1 IQ1_S and IQ1_M 8B files: legs, greedy agreement, perplexity,
      resident bytes.

## Unit 8 — Migrate Q4_0, Q5_0, Q3_K, Q6_K, IQ4_NL; delete the repack machinery (spec §10.2–10.3, 12.3)

### 8.1 TDD
- [ ] 8.1.1 Per format, the existing kernel tests re-pointed to
      `(payload, scales)`: decode, wave f32, coop X1/X3, the widen and
      Mw8 kernels of Q3_K and Q6_K, IQ4_NL's six.
- [ ] 8.1.2 `ResidentLayoutTest` covers every supported type with
      `splitOn` gone; `devWBytes == payload + scales` for each.
- [ ] 8.1.3 A test that the removed names are gone from the tree.

### 8.2 Coding
- [ ] 8.2.1 The 42 kernels re-offset; `coopNeedsRepack`,
      `blockRepack2Kernel`, `blockPadKernel`, `ensureQ6Pad`, `coopDev`
      and `splitOn` removed; the split unconditional; `coopBlockWords`
      the payload stride everywhere.
- [ ] 8.2.2 The int8 widen twin of a Q8_0 weight — `deqFor(Q8_0)` builds
      a tile-major `deqDev` (and `ensureWidenSlab` a `deqSlabDev`) that
      the Mw8 GEMM and the sym id-kernels read — is a second copy of
      int8 data. Those kernels read the split payload in place and
      `deqFor(Q8_0)` turns false. Found by 1.3.3 on the 30B Q8_0: the
      twin is the whole excess over file bytes there, and the coop copy
      the split retires was never built because the widen route served
      prefill.

### 8.3 Acceptance
- [ ] 8.3.1 Filtered suite green.
- [ ] 8.3.2 Legs per format (announced): Q4_K_M 8B (carries Q6_K and
      Q5_0 tensors), Q3_K_M, Q6_K, the iq4_nl file, a Q4_0 8B from
      `llama-quantize`; bit gate then A/B, no decode regression.
- [ ] 8.3.3 Resident bytes equal file bytes for every file above and the
      30B Q8_0 re-checked; the repack code gone.
