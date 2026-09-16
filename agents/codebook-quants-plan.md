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
converter and a converted `bitnet_b1_58-large` at f16/tq1_0/tq2_0;
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
- [x] 2.1.1 `IqGridTest.everyTableChecksumsToTheGeneratorsValue`: the five
      grids, `ksigns_iq2xs`, `kmask_iq2xs` and the packed forms (iq1s
      nibbles, the two-bit IQ2 alphabet) — a position-weighted checksum
      equal to the value `tmp/cbq/grids.c` printed, pinned in the test.
- [x] 2.1.2 `GgufFileTest.typeNameKnowsTheCodebookAndTernaryIds`: ids 16,
      17, 18, 19, 21, 22, 29, 34, 35 name themselves; the loader's
      refusal text lists every `supported()` type.
- [x] 2.1.3 `QuantTest.dequantizeRefusesATypeWithNoBranch`, naming it.
- [x] 2.1.4 `QuantTest.theManifestCoversEveryFixture`: every `.bin` in the
      fixture directory has an entry with a matching block count.

### 2.2 Coding
- [x] 2.2.1 `tmp/cbq/gen.c` (from `tmp/q1fix/gen.c`): nine fixture pairs
      over the same token_embd values, a synthetic positive importance
      vector handed to all seven IQ quantizers, the TQ1_0 fixture checked
      for distinct values in all three regions; committed with the
      manifest regenerated.
- [x] 2.2.2 `tmp/cbq/grids.c`: emits `io/IqGrid.cajeta` (byte-per-value
      and packed forms as static arrays) and prints the checksums.
- [x] 2.2.3 `GgufFile.typeName` six ids; refusal text from `supported()`;
      `Quant.dequantize` throws on a type with no branch.
- [x] 2.2.4 Arbiter files under `tmp/cbq/`: `llama-imatrix` over the
      Q8_0 8B on a calibration text distinct from the perplexity text;
      `llama-quantize --imatrix --allow-requantize` to IQ1_S, IQ1_M,
      IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M;
      `llama-imatrix` over Qwen1.5-MoE Q4_K_M and an IQ3_XXS of it.
      Reference numbers per file — HIP and Vulkan pp512 / pp2048 /
      tg128 through `leg.sh`, `llama-perplexity` on the README text —
      announced, quiet box, recorded in 2.3.1.
      FILES MADE 2026-09-15 (`tmp/cbq/arbiters.sh`, imatrix over 64
      chunks of llama.cpp's docs, `--allow-requantize` from the Q8_0 8B):

        llama8b-iq1_s    2019632544   llama8b-iq3_xxs  3274917280
        llama8b-iq1_m    2161976736   llama8b-iq3_xs   3518752160
        llama8b-iq2_xxs  2399217056   llama8b-iq3_s    3682330016
        llama8b-iq2_xs   2605786528   llama8b-iq3_m    3784828320
        llama8b-iq2_s    2758493600   qwen15moe-iq3_xxs 6345614336
        llama8b-iq2_m    2948285856   (imatrix over the Q4_K_M)

      REFERENCE LEGS 2026-09-15 (`tmp/cbq/reflegs.sh`, quiet box,
      llama.cpp 5306f4b, fa=true, three reps; t/s; PPL = llama-perplexity
      on README.md, -c 2048, four chunks):

        | file | GB | HIP pp512 | HIP tg128@512 | HIP pp2048 | HIP tg64@2048 | VK pp512 | VK tg128@512 | VK pp2048 | VK tg64@2048 | PPL |
        |---|---|---|---|---|---|---|---|---|---|---|
        | llama8b-iq1_s | 2.01 | 1237.6 | 75.1 | 1187.6 | 70.2 | 1188.8 | 90.5 | 1113.2 | 82.7 | 22.80 ± 1.09 |
        | llama8b-iq1_m | 2.15 | 1025.9 | 74.0 | 1011.0 | 68.9 | 1195.7 | 79.8 | 1102.6 | 74.3 | 12.93 ± 0.60 |
        | llama8b-iq2_xxs | 2.39 | 782.3 | 52.0 | 768.0 | 49.5 | 1180.7 | 78.7 | 1089.3 | 73.1 | 8.41 ± 0.38 |
        | llama8b-iq2_xs | 2.60 | 1152.3 | 50.9 | 1117.3 | 48.7 | 1177.9 | 73.4 | 1102.2 | 68.6 | 7.27 ± 0.32 |
        | llama8b-iq2_s | 2.75 | 1055.4 | 49.9 | 1030.6 | 47.6 | 1170.4 | 69.9 | 1110.6 | 64.8 | 6.88 ± 0.30 |
        | llama8b-iq2_m | 2.94 | 1125.5 | 50.1 | 1104.6 | 47.9 | 1157.5 | 67.2 | 1111.8 | 63.3 | 6.47 ± 0.28 |
        | llama8b-iq3_xxs | 3.27 | 761.3 | 48.1 | 752.1 | 46.1 | 1208.1 | 60.3 | 1152.7 | 57.0 | 6.11 ± 0.26 |
        | llama8b-iq3_xs | 3.51 | 705.3 | 47.0 | 699.9 | 45.2 | 1210.6 | 56.2 | 1153.0 | 53.6 | 5.98 ± 0.25 |
        | llama8b-iq3_s | 3.67 | 697.4 | 47.7 | 690.8 | 45.7 | 1257.0 | 54.9 | 1196.5 | 52.2 | 5.97 ± 0.26 |
        | llama8b-iq3_m | 3.78 | 732.4 | 47.5 | 729.5 | 45.4 | 1254.2 | 54.1 | 1193.3 | 51.6 | 5.97 ± 0.25 |
        | qwen15moe-iq3_xxs | 6.34 | 1173.1 | 92.8 | 1159.4 | 76.3 | 2289.3 | 139.6 | 2276.9 | 107.6 | 6.07 ± 0.26 |

      Vulkan decodes every IQ file faster than HIP on this box (the HIP
      IQ dequant kernels are the older scalar family); the parity bars of
      Units 5-7 take the better of the two per cell. Rows are in
      `tmp/llmbench/rows.jsonl` (stamps 20260915-1927..1950).
- [ ] 2.2.5 The compiler findings filed in the cajeta repo:
      `cajeta.xpu.Constant<T>` declared and unwired; `@FastMath` folding
      `fpext(fptrunc x)` to x; and, found here, a static field with an
      array-literal initializer refused as "int32[] not assignable to
      int32[]" (the literal compiles as a local, so `IqGrid` wraps every
      table in a method). Drafted at `tmp/cbq/compiler-findings.md`;
      placement per Julian.

### 2.3 Acceptance
- [x] 2.3.1 Fixtures, manifest and `IqGrid.cajeta` committed with their
      tests; llama.cpp loads and runs all eleven arbiter files; the
      reference table is in this record (under 2.2.4).

## Unit 3 — TQ2_0 and TQ1_0: the four-part template on the new layout (spec §3.3, §5, §6, §7, §8.4)

### 3.1 TDD
- [x] 3.1.1 `QuantTest.tq20MatchesReferenceExactly`, `tq10…` — exact.
- [x] 3.1.2 `QuantTest.tqHostMatVecMatchesTheDequantizedReference`
      (`checkMatVec`, both), and the Q8 twins against the f32 host at the
      Q8 route's bar.
- [x] 3.1.3 `tq20Q8WaveMatVecKernel` / `tq10Q8WaveMatVecKernel` equal the
      Q8 host twin exactly (the path is integer) over the split layout.
- [x] 3.1.4 `tq20/tq10F16CoopX1/X3` equal the host GEMM at the IQ4 bar;
      `coopBlockWords` 16 / 13; `coopColsOk` at 256.
- [x] 3.1.5 `QuantTest.theFourPartInvariant`: over every `supported()`
      type, `packedSupported`, `coopSupports`, `hasKernel` and the host
      chain agree — a type is admitted by all or by none.
- [x] 3.1.6 `KernelIsa`: no spill; the `v_dot4` count per block matches
      the design.
      READ 2026-09-15 off the test exe: tq20 wave 63 VGPRs, 32 v_dot4,
      one b128 payload load; tq10 wave 149 VGPRs, 16 v_dot4 and 64
      `global_load_d16` per block per lane — the scalar base-three
      decode loads every byte it touches. Correct and spill-free; a
      vector unpack (the ×3^n, ×3, >>8 sequence on 16 lanes) is the
      follow-up if TQ1_0's legs fall short of TQ2_0's. Coop X1/X3:
      137–139 VGPRs, 18 KB LDS, no spill, no scratch.

### 3.2 Coding
- [x] 3.2.1 `GG_TQ1_0` / `GG_TQ2_0`; `blockBytes` 54 / 66; `blockElems`
      256; `payloadBytes` 52 / 64 with `scaleOffset` at the end;
      decoders; `dequantize` branches.
- [x] 3.2.2 `tq10MatVecIntoAt` / `tq20MatVecIntoAt` and `…IntoQ8`; the
      `Linear.matvecInto` host chain.
- [x] 3.2.3 `tq20Q8WaveMatVecKernel` (two-bit fields to int8 lanes,
      `dotAccum`), `tq10Q8WaveMatVecKernel` (×pow3, ×3, >>8 in integer);
      coop X1/X3 staging d·trit to f16; launchers; `matVecLaunch`,
      `hasKernel`, `coopBatchLaunch`, `coopBlockWords`, `coopRouteBit`.
- [x] 3.2.4 `Quant.supported()`, `splitOn`, `Linear.packedSupported` —
      last.

### 3.3 Acceptance
- [x] 3.3.1 Filtered suite green; ISA clean. End-to-end waits for Unit 4.

## Unit 4 — The `bitnet` architecture and the converter (spec §8)

### 4.1 TDD
- [x] 4.1.1 `HfToGgufTest.weightQuantMatchesTheReference`: a hand-built
      4×4 tensor → scale = mean|w|, every value in {−s, 0, s}; the 1e-5
      clamp on an all-zero tensor.
- [x] 4.1.2 `HfToGgufTest.tqPackersMatchGgmlByteForByte`: ternary × scale
      values through our packers equal ggml's `ggml_quantize_chunk`
      over the same values (new fixture pairs `tq1_0-ternary`,
      `tq2_0-ternary` from `tmp/cbq/gen.c`).
- [x] 4.1.3 `HfToGgufTest.roundTrip`: a synthetic two-layer
      `BitnetForCausalLM` directory (config.json, one safetensors shard
      written by a test helper, a tiny `tokenizer.model`) converts at
      f16 and tq2_0; `GgufFile` opens both; `ModelConfig.fromGguf` reads
      `bitnet` hparams; every tensor of llama.cpp's `bitnet` list is
      present, `output.weight` absent, norms F32, `token_embd` F16,
      projections TQ2_0.
- [x] 4.1.4 `BitnetTest.theBitnetGraphAppliesTheSubNorms`: a two-layer
      bitnet model on the host equals a reference computed in the test
      — `attn_sub_norm` before `attn_output`, `ffn_sub_norm` before
      `ffn_down`, tied head (one token at position 0, where NEOX rope
      is the identity; the rope is 4.3.2's claim). `BitnetTest.
      theScaleTensorsMultiplyTheProjections`: with `.scale` tensors
      bound the projections are multiplied.
- [x] 4.1.5 `BitnetTest.modelConfigAcceptsBitnetAndNamesItInTheRefusal`
      and `hfConfigAcceptsBitnet`: `bitnet` / `BitnetForCausalLM`
      accepted, tied head from a missing `output.weight`, and both
      refusal texts name it among the supported architectures.

### 4.2 Coding
- [x] 4.2.1 `dev.cajeta.llm.convert.HfToGguf` (in the library, so the
      suite links it; `tools/convert/hf-to-gguf.sh` builds and runs it)
      — `HfToGguf <dir> <out.gguf> f16|tq1_0|tq2_0`: config → hparams
      (`bitnet.*`, rope scaling linear 1.0), shards through
      `model.safetensors.index.json`, the llama and bitnet name maps,
      `weight_quant` in f32, the TQ packers (`Quant.tq10Quantize` /
      `tq20Quantize`, `Quant.f32ToHalfBits`), `SpProto` + added tokens
      → `tokenizer.ggml.*`, `GgufWriter`.
- [x] 4.2.2 `ModelConfig.fromGguf` bitnet branch; NEOX rope for the arch;
      `attn_sub_norm` / `ffn_sub_norm` bound and applied in the layer;
      tied lm_head when `output.weight` is absent; optional `.scale`
      tensors as a scalar multiply after the projection.

### 4.3 Acceptance
- [x] 4.3.1 `1bitLLM/bitnet_b1_58-large` (1536 × 4096, the one member of
      the family a 256-weight block tiles; the 3B is 3200 × 8640)
      converted at f16, tq1_0 and tq2_0 under `tmp/cbq/`; llama.cpp
      loads all three; `llama-quantize` from our f16 to TQ1_0 and TQ2_0
      gives projection tensors byte for byte equal to ours (compared per
      tensor through `GgufFile`).
      DONE 2026-09-15 (`tmp/cbq/bitnet.sh`): 266 tensors each from the
      2.92 GB F32 safetensors; f16 1,458,846,240 B in 81 s (4.3 GB RSS,
      held payloads), tq1_0 243,218,976 B and tq2_0 275,069,472 B in 23 s
      (llama-quantize's are 185 / 217 MB: it packs `token_embd` as Q4_K
      where the converter policy, like llama.cpp's own, keeps F16). `llama-quantize`
      from our f16: `ggufdiff` (bench/GgufTensorDiff) reports 264 of 264
      `blk.*` tensors identical for both TQ types (every projection AND
      every norm). llama-perplexity (CPU, README, c2048 × 4): f16 7.0114,
      tq1_0 7.0138, tq2_0 7.0138 (chunk 1: 7.90). Found on the way:
      cajeta's JSON reader threw on `model_max_length` (an integer wider
      than int64) — fixed in the stdlib (`JsonReader.currentNumberFitsInt64`,
      widens to float64; tests `JsonFloat.wideIntegerWidensToFloat`,
      `int64LimitsStayInteger`), recorded in `tmp/cbq/compiler-findings.md`.
- [x] 4.3.2 Greedy agreement with llama.cpp CPU on both TQ files over
      fixed prompts; perplexity on the README text matches
      `llama-perplexity` at the same window.
      DONE 2026-09-15 (`tmp/cbq/bitnet.sh cajeta`, `ppl-bos.sh`): greedy
      "The capital of France is", 24 tokens at temp 0 — both TQ files
      give llama-completion's text exactly (" Paris. It is the largest
      city in the country and the second largest in Europe. It is also
      the most populous"). Perplexity at llama-perplexity's chunk-1
      window (BOS + README tokens, positions 1024..2046 scored; `pplprobe
      pre=1025 eval=1023`): tq1_0 7.9014, tq2_0 7.9000 against 7.9028
      for both TQ files (f16 chunk 1: 7.8951). Found on the way, all
      fixed: the batched prefill's device-resident attention arm
      (`resAttn`) skipped the attention sub-norm — chunk-4 greedy said
      "the." while chunk 1 was right (`HfToGgufTest.checkBatched` now
      pins it); `pplprobe` tokenized without BOS where llama-perplexity
      heads every chunk with one (9.39 against 7.90 on the same window —
      the probe now prepends BOS iff `Tokenizer.addsBos`, a no-op for the
      qwen2-pre files); the CLI's streamed decode dropped each piece's
      leading space (`Tokenizer.decodePiece`). Left open: the f16 GGUF
      loads in llama.cpp but not in the engine — `prewarmPrefillWeights`
      → `Linear.ensureDevice` refuses an f32-weight linear ("no host
      bytes and no source mapping for a packed weight of 0 bytes"); f16
      is a conversion intermediate here, so it is not on this unit.
- [x] 4.3.3 Legs (announced): cajeta pp512 / tg128 on both files against
      llama.cpp CPU; decode GB/s beside the Q4_K wave kernel's on the
      8B, so the bandwidth claim of spec 8.5 is a number.
      DONE 2026-09-15 (`tmp/cbq/bitnet-legs*.sh`, `tmp/llmbench/leg.sh`
      gained a `cpu` engine = the HIP build at -ngl 0; 3 reps, t/s,
      llama.cpp fa=0/fa=1):

      | file | engine | pp512 | tg128@512 | pp2048 | tg64@2048 |
      |---|---|---|---|---|---|
      | tq1_0 | cajeta | 4072 | 165.5 | 2060 | 98.0 |
      | tq1_0 | llama.cpp CPU | 743 / 763 | 168 / 160 | 716 / 729 | 108 / 88 |
      | tq2_0 | cajeta | 5426 | 211.8 | 2300 | 112.1 |
      | tq2_0 | llama.cpp CPU | 1450 / 1539 | 175 / 163 | 1360 / 1311 | 107 / 86 |
      | q4_k_m (same model, llama-quantize from our f16) | cajeta | 5747 | 222.3 | | |
      | q4_k_m | llama.cpp CPU | 3088 / 5608 | 131 / 127 | | |
      | q4_k_m | llama.cpp HIP | 6745 / 7166 | 169 / 196 | | |
      | q4_k_m | llama.cpp Vulkan | 7544 / 9680 | 202 / 268 | | |
      | Llama-3.1-8B q4_k_m | cajeta | 1546 | 41.9 | | |

      The first pass measured 39-41 t/s decode on EVERY bitnet file, the
      same 24 ms/token as the 8B: the profile put 19.4 ms of it in
      `HostOps.rowsDotRow` — a tied model had no lm_head Linear at all,
      and scored its logits on the host against the f32 embedding table
      (32002 x 1536, scalar). Fixed in this unit: F16 is now a packed
      weight type (eight halves a block, `Quant.f16Block/f16MatVecInto`,
      `QuantKernel.f16F32WaveMatVecKernel` over f32 activations,
      `f16F16CoopX3Kernel` for the batched GEMM, so the four-part
      invariant holds), the head is always a Linear and a tied model
      binds it to `token_embd` (`CausalLM.headLinear`), and the
      embedding table stays packed. Decode went 24 -> 4.7 ms/token on
      tq2_0; the f16 GGUF runs in the engine as well (greedy identical,
      ppl 7.8949 vs llama-perplexity 7.8951). Bandwidth: the 8B Q4_K_M
      streams 4.92 GB in 23.9 ms = 206 GB/s; bitnet tq2_0 moves ~350 MB
      (273 MB weights + 75 MB K/V at depth 512) in 4.7 ms = 74 GB/s, so
      the 700M model's token is attention and launch count, not weight
      bytes — and spec 8.5's bar (TQ2_0 faster than the same model's
      Q4_K_M) is NOT met: 4.7 vs 4.5 ms. The tq2_0 wave kernel runs
      7.3 MB/layer in 7.9 us = 132 GB/s against the Q4_K kernel's ~200;
      that gap is 4.3.5.
- [ ] 4.3.5 TQ1_0/TQ2_0 wave mat-vec at the Q4_K kernel's bandwidth
      (132 -> ~200 GB/s; ISA read first). Deferred to the kernel-tuning
      pass; Unit 5 proceeds.
- [x] 4.3.4 Resident bytes equal file bytes.
      DONE 2026-09-15 (`tmp/cbq/bitnet-ledger.sh`, CAJETA_XPU_ALLOC_TRACE
      through the CLI at ctx 4096): 169 `Linear.allocResident`
      allocations (24 layers × 7 projections + the tied F16 head) sum to
      tq1_0 241,637,376 B and tq2_0 273,487,872 B — exactly 24 × 110,592
      blocks × 54 / 66 B plus 32002 × 1536 × 2 B, the projection and
      token_embd tensors' file bytes; the per-row split pads nothing at
      these widths (1536 = 6 blocks, 4096 = 16). The rest of the 7.15 /
      7.19 GB peak is the KV planes (0.604 GB, f16 at 4096) and 6.241 GB
      of `Linear.ensureBatchOut` from `prewarmPrefillWeights` — every
      layer's batch outputs held at the full 4096 rows, a prefill-design
      cost shared by every model, not this unit's.

## Unit 5 — IQ2_XXS, IQ2_XS, IQ3_XXS: the ksigns family (spec §3.4, §4, §6.2, §7.2)

### 5.1 TDD
- [x] 5.1.1 Decoders exact, three fixtures.
- [x] 5.1.2 Host mat-vecs and Q8 twins, three formats.
- [x] 5.1.3 `IqGridDeviceTest`: a probe kernel gathers every entry of the
      resident device tables (byte and packed forms) and expands all 128
      `ksigns` indices; equal to the host arrays.
      (`IqCodebookTest.theDeviceTablesEqualTheHostTables`, one probe over
      all five tables.)
- [x] 5.1.4 Wave decode kernels equal the Q8 twins exactly — 2s+1 per
      sub-block, the 0.125 / 0.25 tails once per block.
- [x] 5.1.5 Coop X1/X3 with the LDS-staged table equal the host GEMM;
      `coopBlockWords` 16 / 18 / 24. (X3 only: the X1 variant is a
      long-k tuning of the same body and is deferred with 4.3.5.)
- [x] 5.1.6 The four-part invariant extended; ISA: no spill.
      (`TernaryTest.theFourPartInvariant` covers every type by
      construction; the ISA read is deferred with 4.3.5.)

### 5.2 Coding
- [x] 5.2.1 `QuantKernel.ensureIqTables()` — static resident buffers
      uploaded once (the `pfSink` pattern), passed as kernel operands.
- [x] 5.2.2 Decoders; host; wave kernels (one gather, one sign expand,
      one `dotAccum` per 8); coop staging through the LDS table;
      registration; `supported` / `splitOn` / `packedSupported` last.
      DONE 2026-09-16: `Quant.iq2xxsInts/iq2xsInts/iq3xxsInts` feed one
      block decoder, one host mat-vec and one integer twin per format;
      `QuantKernel.iqDot16` expands two ksigns groups into sixteen int8
      lanes for one `dotAccum` (the `(x ^ -m) + m` sign form, no table
      for `kmask` since it is `1 << j`); three wave kernels take one
      32-element sub-block per lane, four lanes per block; three coop X3
      kernels stage `ksigns` and the grid in LDS at entry behind one
      barrier. All three join `splitOn`, so a row is its f16 scale
      prefix then 16 / 18 / 24 payload words per block.

### 5.3 Acceptance
- [~] 5.3.1 IQ2_XXS, IQ2_XS, IQ3_XXS 8B files: `batched`, no
      `batch-refused`; legs against llama.cpp HIP and Vulkan (pp ≥ 1.0×,
      tg ≥ 0.95×); greedy agreement; perplexity within the floor;
      resident bytes equal file bytes.
      PART DONE 2026-09-16 on the two loadable files (`tmp/cbq/iq-accept*.sh`,
      `iq-routes.sh`). Routes: `batch-route coop iq2_xxs 4096x4096` and
      `coop iq2_xs 4096x4096`, no `batch-refused` on any tensor.
      Perplexity at llama-perplexity's chunk-1 window (BOS aligned):
      iq2_xxs 7.4942 against 7.4910 (+0.04%), iq2_xs 6.5934 against
      6.6042 (−0.16%) — within the floor. Resident bytes equal the file
      bytes of every bound Linear EXACTLY: 2,217,934,848 of 2,390,310,912
      and 2,424,504,320 of 2,596,880,384, each short by the Q2_K
      `token_embd` (172,376,064 B) that an untied model keeps in the
      Embedding rather than a Linear. Greedy is NOT token-identical on
      iq2_xs: the openings agree and the fifth token tips ("famous" /
      "full"), which is the near-tie behaviour these f16-accumulating
      kernels document — perplexity over 1023 positions carries the
      claim. BLOCKED for IQ3_XXS: `llama8b-iq3_xxs.gguf` stores
      `token_embd` as IQ3_S (type 21), so the file cannot load until
      Unit 6; it is a composition dependency, not a defect here. The
      HIP/Vulkan legs are ANNOUNCED and wait on the box.

## Unit 6 — IQ2_S, IQ3_S: raw signs and qh high bits (spec §3.4, 12.1)

### 6.1 TDD
- [x] 6.1.1 Decoders exact, two fixtures; IQ3_S's `1 + 2s` scale.
- [x] 6.1.2 Host mat-vecs and Q8 twins.
- [x] 6.1.3 Wave decode kernels and coop X1/X3; `coopBlockWords` 20 / 27.
      (X3 only, as in Unit 5; X1 rides 4.3.5.)
- [~] 6.1.4 The 12.1 harness: the IQ2_S wave kernel with an LDS-table arm
      and an L1-table arm, bit-identical outputs, timed alternating on
      idle; the choice recorded here with both numbers.
      Both arms SHIP and are bit-identical
      (`IqCodebookTest.theTwoTableResidencyArmsAgreeExactly`;
      `QuantKernel.setIqLdsTable` picks one, L1 by default). The TIMING
      half is announced and waits on a quiet box.

### 6.2 Coding
- [x] 6.2.1 Decoders; host; kernels; registration; gates last.
      DONE 2026-09-16: `Quant.iq2sInts` (ten-bit index, the high two bits
      from `qh`, raw sign bytes) and `iq3sInts` (a ninth index bit per
      half group, `1+2s` per 32) join the Unit 5 machinery, which
      generalized into `iqScale` and `iqStep`. Device: `iq2s`/`iq3s`
      tables, two wave kernels plus the LDS-table twin, two coop X3
      kernels. `Linear.routeSaid` widened to int64 — the route-record
      mask had run out of bits and IQ3_S would have aliased `q4 deqMw4`.

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
