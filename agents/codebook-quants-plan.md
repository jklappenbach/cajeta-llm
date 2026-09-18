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
real model. The codebook tier follows by kinship — ksigns family, then
the raw-sign family. Unit 7 breaks that run deliberately: by then eight
files have been measured against llama.cpp and every decode wave in the
family is slow by the same 0.66–0.85×, so the cause is found once, on
the kernels that exist, before the IQ1 pair (Unit 8) copies the shape
again. The migration of the older formats closes the plan, once the
split path has carried nine formats.

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
      is item 9.2.2. Load fell by 40 % because the split streams the slab
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
      table in a method); and `cajeta profile summary` windowing only by
      relative duration, with no way to scope the device tier to a named
      HOST frame — the gap that let 9.2.1 state a whole-run share as a
      prefill share and hold it for a day. Drafted at
      `tmp/cbq/compiler-findings.md`; placement per Julian.

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
      | q4_k_m | llama.cpp (HIP) | 6745 / 7166 | 169 / 196 | | |
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
- [x] 4.3.5 TQ1_0/TQ2_0 wave mat-vec at the Q4_K kernel's bandwidth
      (132 -> ~200 GB/s; ISA read first) — folded into Unit 7 with the
      IQ decode gap, which has the same shape.
      MET 2026-09-16, and spec 8.5's bar with it — once the two files
      are made comparable. The bar is "TQ2_0 faster than Q4_K_M on the
      same model", and it was failing 4.60 ms against 4.29. The cause
      is not the ternary kernel: the default TQ file is TIED and keeps
      `token_embd` in F16, 32002 x 1536 x 2 = 98.3 MB, while Q4_K_M
      quantizes the same tensor to Q6_K at 32.3 MB. The head reads that
      tensor every token, so the TQ file streams 66 MB more per token
      -- 0.287 ms at the ~230 GB/s the profiler measures for that
      kernel, against a total gap of 0.31 ms. Holding the embedding
      constant (`bitnet-large-lq-*`, Q6_K embeddings, same
      projections), five reps each:

      | file | ms/token | t/s |
      |---|---|---|
      | q4_k_m | 4.288 | 233.2 |
      | lq-tq2_0 | 4.179 | 239.3 MEETS |
      | lq-tq1_0 | 5.451 | 183.5 |

      TQ1_0 stays slower and the reason is already recorded: it is
      cleanly VALU-bound at 16.5 VALU per value, 108% of the
      single-issue ceiling, not bandwidth-bound like every other format
      in this plan. Spec 8.5 names TQ2_0.
      ALSO FOUND, and not ours to fix in this unit: bitnet-large is
      16 heads over 16 kv heads at head dim 96, and every flash-decode
      gate in `AttnKernel` requires `hd == 128`, so its decode attend
      takes the scalar `attnScore` + `attnCombine` pair -- 3072
      launches each, 44.85 us and 23.50 us, 31% of the token. It
      affects both arms of the table above equally, so it does not move
      the bar; it is the same "a fast path exists and this model cannot
      reach it" shape as the two fusions of 7.2.2, and it is the
      largest single item left on any model we run.
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
- [x] 5.3.1 IQ2_XXS, IQ2_XS, IQ3_XXS 8B files: `batched`, no
      `batch-refused`; legs against llama.cpp (HIP) and Vulkan (pp ≥ 1.0×,
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
      HIP/Vulkan legs are MEASURED — see the legs table below
      Unit 6; the bar is not met and the gap is item 6.4.1.

## Unit 6 — IQ2_S, IQ3_S: raw signs and qh high bits (spec §3.4, 12.1)

### 6.1 TDD
      DECODE HALF NOW CLEARS, 2026-09-16 night after Unit 7 (which
      carries 6.4.1, this item's blocker): iq2_xxs 0.952, iq2_xs 0.954,
      iq3_xxs 0.997 of llama.cpp (Vulkan) against the 0.95 bar, and iq3_xxs — the
      file Unit 5 could not open — is measured for the first time.
      Prefill: iq2_xxs 1.071 and iq2_xs 1.053 clear 1.0x; iq3_xxs is
      0.992, and its coop kernel is the IQ3 family of 6.4.2. CLOSED
      2026-09-17 when 6.4.2's doubled token tile reached every IQ coop
      kernel: prefill 1.331 / 1.341 / 1.273, decode 0.952 / 0.952 /
      0.993. Both halves met on all three.
- [x] 6.1.1 Decoders exact, two fixtures; IQ3_S's `1 + 2s` scale.
- [x] 6.1.2 Host mat-vecs and Q8 twins.
- [x] 6.1.3 Wave decode kernels and coop X1/X3; `coopBlockWords` 20 / 27.
      (X3 only, as in Unit 5; X1 rides 4.3.5.)
- [x] 6.1.4 The 12.1 harness: the IQ2_S wave kernel with an LDS-table arm
      and an L1-table arm, bit-identical outputs, timed alternating on
      idle; the choice recorded here with both numbers.
      Both arms ship and are bit-identical
      (`IqCodebookTest.theTwoTableResidencyArmsAgreeExactly`;
      `QuantKernel.setIqLdsTable` picks one). MEASURED 2026-09-16 on a
      quiet box, `llama8b-iq2_m` (156 IQ2_S tensors), tg128 at depth
      512, five repeats with the arm order alternating
      (`tmp/cbq/u6-lds-ab.sh`):

      | rep | L1 t/s | LDS t/s |
      |---|---|---|
      | 1 | 46.63 | 36.93 |
      | 2 | 46.78 | 36.98 |
      | 3 | 46.81 | 36.96 |
      | 4 | 46.70 | 37.10 |
      | 5 | 46.83 | 37.13 |

      DECIDED: L1, and it is not close — 46.8 against 37.1, a 26% gap
      with under 0.5% spread inside each arm. The decode wave is one
      workgroup of 32 lanes per row, so an 8 KB stage is paid per row
      and read by 32 lanes; Vulkan's LDS choice is for a GEMM workgroup
      of 256 that reuses the table across a whole tile. `iqLdsTable`
      stays false and the LDS kernel stays as the control.

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
- [x] 6.3.1 IQ2_S, IQ2_M, IQ3_S, IQ3_XS, IQ3_M 8B files (the mixes
      exercise Units 5 and 6 together): legs, greedy agreement,
      perplexity, resident bytes.
      DONE 2026-09-16, legs included (`tmp/cbq/u6-accept.sh`,
      `u56-legs.sh`; the legs miss their bar — item 6.4.1). Every
      one of these files carries IQ3_S or IQ2_S, so all six — including
      the `llama8b-iq3_xxs` that Unit 5 could not open — load only now.
      Perplexity at llama-perplexity's chunk-1 window, BOS aligned:

      | file | cajeta | llama.cpp | delta | greedy (16 tokens) |
      |---|---|---|---|---|
      | iq2_s | 6.2280 | 6.2346 | −0.11% | identical |
      | iq2_m | 5.7347 | 5.7365 | −0.03% | tips at token 1 |
      | iq3_s | 5.1634 | 5.1717 | −0.16% | identical |
      | iq3_xs | 5.1719 | 5.1743 | −0.05% | tips at token 2 |
      | iq3_m | 5.1521 | 5.1721 | −0.39% | identical |
      | iq3_xxs | 5.3515 | 5.3598 | −0.15% | identical |

      Routes: `coop` per codebook type, `dotAccum ty=12` for the Q4_K
      tensors, no `batch-refused`. Resident bytes equal the bound
      tensors' file bytes EXACTLY on all six (ledger against the GGUF
      table: 2,523,856,896 / 2,713,649,152 / 3,447,693,312 /
      3,284,115,456 / 3,550,191,616 / 3,040,280,576, each the linear
      bytes less the `token_embd` an untied model keeps in the
      Embedding). Worth a look later: all seven perplexities here sit
      BELOW llama.cpp's, between 0.03% and 0.39% — a consistent sign,
      not scatter.
      DECODE HALF NOW CLEARS on all five, same round: iq2_s 0.972,
      iq2_m 0.961, iq3_s 0.998, iq3_xs 1.007, iq3_m 0.995. Prefill
      splits exactly on whether the file carries IQ3_S — iq2_s 1.055
      and iq2_m 1.103 clear, iq3_xs 0.876, iq3_m 0.856 and iq3_s 0.828
      do not, which is 6.4.2 and nothing else. CLOSED 2026-09-17 by
      6.4.2's rollout: prefill 1.357 / 1.348 / 1.130 / 1.204 / 1.135,
      decode 0.976 / 0.968 / 0.999 / 1.009 / 0.995. Both halves met on
      all five.
      Perplexity moved within the floor and 7.3.2 carries the table and
      the control that names the cause.
- [x] 6.3.2 The IQ3_XXS Qwen1.5-MoE: every expert tensor `batched`,
      perplexity within the MoE floor of llama.cpp's, legs.
      DONE 2026-09-16, legs included (they miss badly — item 6.4.3):
      all 24 layers report
      `moe-batch-route resident` and the four codebook types take the
      coop GEMM, with no refusal; perplexity 5.4687 against 5.4781
      (−0.17%), inside the MoE routing-flip floor. `DenseRouteProbe`
      set no expert residency budget, so its first answer was 24 ×
      "an expert is not admitted" — the trap `PplProbe` already
      documents; the probe now sets the budget the engine's AUTO would.
      RE-VERIFIED 2026-09-17 on the post-Unit-9 build, because none of
      the above carried: this file's experts include IQ4_NL, which the
      migration moved onto the split layout, and the binaries that gave
      those answers predate it. Routes: 24 `batch-route resident` and 4
      `batch-route coop`, no `batch-refused`. Perplexity 5.47719 against
      llama.cpp's 5.4781, -0.017% — inside the floor, and tighter than
      2026-09-16's -0.17%. Greedy agrees with llama.cpp's CPU stream on
      the tokens it shares.
      THE LEDGER LEG OF THAT SCRIPT MEASURES NOTHING ON A MoE and is not
      quoted: `allocResident` over a 2-token generation reads 678 MB of
      a 6.35 GB file, which is the same artifact 6.4.3 opened on — a
      short generation admits a handful of experts. The residency figure
      that means something is `residentKb` under a 512-token prompt,
      recorded there.
      The legs ride 6.4.3's re-measurement; the gap has been that item's
      since it opened.

### 5.4 / 6.4 Legs (measured 2026-09-16, quiet box)

      UNCHANGED BY UNIT 7: the Qwen MoE's perplexity is 5.46277 now
      against 5.4687 recorded (-0.11%, inside the floor), the routes
      still report `moe-batch-route resident` on all 24 layers, and the
      leg is still 6.4.3's. Unit 7's decode work reaches this file
      through the same wave kernels, but its legs are dominated by
      expert residency, which is what 6.4.3 is about.
- [x] The announced legs for 5.3.1, 6.3.1 and 6.3.2, one table
      (`tmp/cbq/u56-legs.sh`; 3 reps, engine order alternating per file,
      max of reps, llama.cpp best of fa=0/fa=1; pp512 and tg128 at depth
      512). llama.cpp reproduces 2.2.4's reference legs within noise
      (MoE HIP 1178/93.1 against 1173.1/92.8, Vulkan 2308/139.4 against
      2289.3/139.6), so the box is the same one and the gap is ours.

      | file | cajeta pp | hip | vulkan | pp × | cajeta tg | hip | vulkan | tg × |
      |---|---|---|---|---|---|---|---|---|
      | iq2_xxs | 1311 | 783 | 1178 | 1.11 | 61.7 | 52.3 | 78.8 | 0.78 |
      | iq2_xs | 1235 | 1155 | 1182 | 1.04 | 48.7 | 51.3 | 73.4 | 0.66 |
      | iq3_xxs | 1198 | 758 | 1199 | 1.00 | 50.7 | 48.2 | 59.4 | 0.85 |
      | iq2_s | 1233 | 1057 | 1164 | 1.06 | 47.8 | 50.0 | 69.9 | 0.68 |
      | iq2_m | 1288 | 1129 | 1157 | 1.11 | 46.8 | 50.2 | 67.0 | 0.70 |
      | iq3_s | 1038 | 698 | 1255 | 0.83 | 42.8 | 47.7 | 55.1 | 0.78 |
      | iq3_xs | 1088 | 706 | 1212 | 0.90 | 46.0 | 47.1 | 56.4 | 0.82 |
      | iq3_m | 1082 | 740 | 1260 | 0.86 | 43.0 | 47.6 | 54.2 | 0.79 |
      | qwen15moe-iq3_xxs | 613 | 1178 | 2308 | 0.27 | 36.6 | 93.1 | 139.4 | 0.26 |

      THE BAR IS NOT MET. Prefill clears 1.0× on the five IQ2-family
      files and misses on the three the IQ3_S coop GEMM dominates
      (0.83–0.90); decode misses 0.95× on every file (0.66–0.85),
      though it beats or matches HIP everywhere except the MoE. The MoE
      is 0.27× / 0.26× — its own item below. These are first
      measurements, not regressions: none of these files could load
      before Units 5 and 6.
- [x] 6.4.1 The IQ codebook decode gap — promoted to Unit 7, which
      folds in 4.3.5's ternary gap as the same shape.
- [x] 6.4.2 The IQ3_S coop GEMM prefill gap: 0.83–0.90× of llama.cpp (Vulkan) where
      the IQ2 family clears 1.0×. The three files that miss are exactly
      the IQ3_S-heavy ones (193 / 157 / 81 tensors).
      MET 2026-09-17, and the cause was weight REUSE, not the kernel's
      work. `iq3sF16CoopX3Kernel` is 79.8% of an IQ3_S prefill at 2.22 ms
      a launch against `iq2xsF16CoopX3Kernel`'s 1.74 — and the two
      disassemble to 728 and 733 instructions with near-identical mixes,
      so the 1.28x is not work. Normalising the launch counts and fitting
      total coop time against BYTES PER BLOCK over the three files gives
      `t = 302.6 + 4.951 x bytes` ms, which predicts iq3_s at 837 against
      833 measured — 0.5%. So 64% of the IQ3_S coop time is weight
      traffic and 36% is the shared activation-plus-MMA floor, and the
      A side is read once per 128-token tile.
      THE FIX WAS ALREADY WRITTEN FOR ANOTHER FORMAT. Unit 32's
      `q4kF16CoopN256Kernel` doubles the token tile for exactly this
      reason and records the decomposition that motivated it — "mma
      ceiling 57%, A-side load+dequant 27%, B-side 17%... the lever was
      weight REUSE, not arithmetic". It was worth +6% on Q4_K because
      Q4_K's A side is 27%; IQ3_S's is 64%. `iq3sF16CoopN256Kernel` is
      the X3 body with the token tile at 256: `warpR = sg % 2` over 64
      rows and `warpC = sg / 2` over 64 tokens, sixteen accumulators
      instead of eight, B left in global (staging it would cost 36 KB of
      LDS on top of the codebook, and the A side is the lever). vgpr 212,
      no spill, LDS unchanged at 20480.

      | file | before | after | vulkan | was | now |
      |---|---|---|---|---|---|
      | iq3_s | 1036.6 | 1410.4 | 1266.1 | 0.828 | 1.114 MEETS |
      | iq3_xs | 1061.8 | 1312.3 | 1209.4 | 0.876 | 1.085 MEETS |
      | iq3_m | 1079.0 | 1445.3 | 1258.9 | 0.856 | 1.148 MEETS |

      The kernel itself went 2.22 -> 1.51 ms a launch, -32%, against a
      predicted -32%. DECODE IS UNTOUCHED, measured in the same window:
      54.99 / 56.92 / 53.91 against 55.03 / 56.97 / 53.97.
      iq3_xs gains least because only 157 of its tensors are IQ3_S; the
      rest still take the 128-token tile, which is the rollout below.
      ROLLED OUT TO ALL FIVE IQ COOP KERNELS, because the four the item
      did not name are the same body with the same 128-token tile, and
      iq3_xs gained least (+23.6%) precisely because only 157 of its
      tensors were IQ3_S while the rest still took the old tile. Same
      transformation each time, 212-219 vgpr, no spill anywhere, LDS
      unchanged. Every file re-measured on the shipped code, five reps a
      side, means:

      | file | prefill | vulkan | was | now | decode | now |
      |---|---|---|---|---|---|---|
      | iq2_xxs | 1597.7 | 1200.3 | 1.071 | 1.331 | 75.39 | 0.952 |
      | iq2_xs | 1578.6 | 1177.3 | 1.053 | 1.341 | 70.05 | 0.952 |
      | iq2_s | 1589.5 | 1171.6 | 1.055 | 1.357 | 68.44 | 0.976 |
      | iq2_m | 1569.4 | 1164.3 | 1.103 | 1.348 | 65.08 | 0.968 |
      | iq3_xxs | 1532.9 | 1204.4 | 0.992 | 1.273 | 59.93 | 0.993 |
      | iq3_xs | 1464.1 | 1216.0 | 0.876 | 1.204 | 56.96 | 1.009 |
      | iq3_s | 1419.6 | 1255.9 | 0.828 | 1.130 | 54.99 | 0.999 |
      | iq3_m | 1435.1 | 1264.5 | 0.856 | 1.135 | 53.92 | 0.995 |

      EIGHT OF EIGHT ON BOTH BARS, prefill 1.13-1.36x and decode
      0.952-1.009x. Decode is untouched by construction — no decode
      kernel was edited — and the table above is the evidence rather
      than the claim. This closes 5.3.1's last number (iq3_xxs prefill,
      which sat at 0.992) and 6.3.1's three.
      A METHOD NOTE PAID FOR TWICE TONIGHT: read the numbers from
      `rows.jsonl`, never from a monitor's notification. The tail-based
      watcher re-emitted lines from the PREVIOUS round twice, once with
      a prefill 4% off and once with a whole stale file's row, and both
      times the durable per-rep record settled it. Reps within a round
      agree to 0.6%.
      A NOTE ON THE TEST, because it failed first for the right reason.
      `CoopQuantGemmTest.checkN256` staged its weights with
      `blockRepack2Launch`, but every IQ codebook format is `splitOn` —
      the coop GEMM reads the split resident layout, an f16 scale prefix
      then payload words. The kernel read garbage and the test said so
      (-473248 against -0.0937). It now takes `splitLaunch` and
      `coopView` like `checkFormatAt` does, and the red-to-green
      transition is the evidence that the case is a live gate rather
      than a shape the suite never reaches.
- [x] 6.4.3 The Qwen1.5-MoE at 0.27× / 0.26×. Expert residency is the
      first suspect (the CLI ledger held 678 MB of 6.3 GB of expert
      bytes), so measure what the bank admits before touching a kernel.
      MEASURED FIRST, as the item says, and THE NAMED SUSPECT IS
      REFUTED. `residentKb` reads 6,969,468 — 6.97 GB — against a 6.3 GB
      expert set, so the bank now admits essentially everything and
      residency is not the cause. Re-measured baseline 2026-09-17 after
      Unit 7 and 6.4.2, which reached this file not at all: prefill
      615.3 against llama.cpp (Vulkan)'s 2297.5 (0.268x), decode 36.75 against
      138.11 (0.266x).
      WHAT THE CENSUS SAYS (512 tokens, device time). CORRECTED
      2026-09-17 under 9.2.6: these shares are of the WHOLE RUN's device
      time, not of prefill, and `splitKernel` is a LOAD row that does
      not belong in this table at all — windowed to the measured prefill
      its count is zero.

      | kernel | ms | share | what it is |
      |---|---|---|---|
      | iq4nlF16CoopX3 | 299.3 | 29.1% | per-expert GEMM |
      | blockRepack2Kernel | 189.7 | 18.4% | layout conversion |
      | iq3xxsF16CoopX3 | 184.9 | 18.0% | per-expert GEMM |
      | iq3xxsF16CoopN256 | 102.3 | 9.9% | the shared expert |
      | splitKernel | 95.7 | 9.3% | layout conversion |

      FOUR CAUSES, none of them residency:
      1. THE EXPERT GEMM PAYS FOR ROWS IT DOES NOT HAVE.
         `Linear.cajeta` padded every coop batch to 128 tokens
         (`(rows + 127) / 128 * 128`), and with top-4 over 60 experts a
         512-token prefill gives each expert about 34. That is 3.7x of
         dead rows, and 3.7x is the size of the whole gap. Confirmed by
         arithmetic: the expert GEMMs run at ~11 TFLOP/s on the rows
         they compute and ~2.9 effective.
      2. LAYOUT CONVERSION AT RUNTIME, and the two halves of it fall
         in DIFFERENT PHASES — which the 27.7% first written here hid
         by summing them. `blockRepack2Kernel`, 1420 launches at one per
         expert per layer on first admission, is 176.9 ms and 25.5% OF
         PREFILL: that one is real, and the migration collected it.
         `splitKernel`, 264 launches and 96.9 ms, is bind-time work in
         LOAD and was never part of prefill. This is Unit 9's item 9.2.1
         (`coopNeedsRepack`, the repack machinery) showing up as a
         quarter of the MoE's prefill, not a third. llama.cpp charges
         both to LOAD; we charged the repack to the first prefill that
         touches an expert, and a fresh process per leg rep pays it
         every time.
      3. DECODE STILL DISPATCHES PER EXPERT: 288 `iq3xxsQ8WaveMatVec`
         and 96 `iq4nlMatVec` launches a token, 384 where llama.cpp
         issues about 72.
      4. IQ4_NL DECODES THROUGH `iq4nlMatVecKernel`, the ONE-ITEM-PER-ROW
         kernel 7.1.4 measured at 1.8-3.7x worse than a wave — 3072
         launches, 90.4 ms, 7.2%. IQ4_NL has no resident wave mat-vec
         because it is one of Unit 9's unmigrated formats.
      CAUSE 1 IS FIXED HERE, because it is not Unit 9's and it helps any
      ragged batch anywhere. A 64-token tile for all five codebook coop
      kernels and IQ4_NL's (`*F16CoopN64Kernel`: the X3 body with
      `warpC` over 32 tokens and four accumulators instead of eight),
      and `Linear` now pads to the finest tile the format has
      (`QuantKernel.coopPadGrain`). 86-148 vgpr, no spill — LIGHTER than
      the X3 kernels they relieve, so occupancy improves too. Prefill
      612 -> 657 t/s, 0.268 -> 0.286; `iq3xxsF16Coop` 184.9 -> 126.8 ms
      (-31%), `iq4nlF16Coop` 299.3 -> 283.2 (-5%). Dense prefill is
      untouched: a 512-token chunk still pads to 512 and takes N256
      (iq3_s 1420.7 t/s against 1419.6, decode 54.78 against 54.99).
      IQ4_NL gained least and the reason is worth chasing later — its
      per-launch time barely moved (197 -> 180 us) where IQ3_XXS's fell
      with the tile, so that kernel is not padding-bound.
      UNIT 9 PULLED FORWARD 2026-09-17, and its first format paid: with
      IQ4_NL migrated the runtime repack is gone and MoE prefill is
      902.1 against llama.cpp (Vulkan)'s 2277.8 — 0.396, from 0.268 when this item
      opened and 0.286 after the halved tile alone. Completing the other
      four formats leaves it there (896 t/s in the profile): this model
      carries no Q6_K, Q3_K, Q4_0 or Q5_0 tensors, so the rest of the
      unit is worth nothing HERE and everything to the files that do. Decode is unmoved at
      0.262, which the census predicts: its 384 launches a token are
      cause 3, the grouped id-GEMM, untouched.
      RE-OPENED AND RE-MEASURED 2026-09-17, Unit 9 having landed. THREE
      OF FOUR CAUSES ARE COLLECTED and ONE remains, which is now the
      whole gap.

      | leg | opened | after migration | now | llama.cpp (Vulkan) |
      |---|---|---|---|---|
      | prefill t/s | 615.3 | 902.1 | 908.5 | 2306.9 |
      | | 0.268x | 0.396x | 0.394x | |
      | decode t/s | 36.75 | ~36.4 | 39.86 | 139.60 |
      | | 0.266x | 0.262x | 0.285x | |

      CAUSE 4 IS COLLECTED BY 9.2.4, not 9.2.5, and the distinction is
      worth keeping: `iq4nlMatVecKernel` is gone from the profile — 3072
      launches and 94.5 ms became `iq4nlF32WaveMatVecKernel` at 2957 and
      35.4 ms, -62%. I predicted the INTEGER wave on the ground that
      gate/up carry inDim 2048 and clear `q8kDims`. Wrong: this file's
      IQ4_NL tensors are all `ffn_down_exps` at inDim 1408, four a
      layer, which fails `% 256` and takes the f32 wave 9.2.4 built.
      9.2.5's integer route does not reach this model at all.
      CAUSE 2 IS GONE FROM PREFILL ENTIRELY. The corrected census (see
      below on how it had to be windowed) reads:

      | phase | device self | top items |
      |---|---|---|
      | load/bind [0,1650ms) | 260.9 ms | splitPayload 64.2 + splitScale 45.7 = 109.9 (42%) |
      | prefill [1650,2230ms) | 522.4 ms | iq4nlF16CoopN64 231.9 (44%), iq3xxsF16CoopN64 126.2 (24%) |
      | decode [2230ms,end) | 231.4 ms | iq3xxsQ8Wave 109.1, iq4nlF32Wave 35.4 |

      No layout conversion appears in prefill at any count. The split's
      291 ms is 109.9 ms after 9.2.6, which is the number 9.2.6's A/B
      implied and could not see directly.
      CAUSE 3 IS THE WHOLE REMAINING GAP, and the census states it
      without inference: decode spends 231.4 ms of DEVICE time inside
      944.4 ms of wall. Three quarters of decode is not GPU work. That
      is the 384 launches a token against llama.cpp's ~72, and it needs
      the grouped id-GEMM over the expert dimension that the codebook
      formats do not have. Nothing else in the decode column is worth
      touching before it: the two mat-vec kernels that dominate the
      device time together are 145 ms of a 944 ms token stream.
      HOW THE CENSUS HAD TO BE WINDOWED, because the first cut was
      wrong: anchoring the load window on the `LlmEngine.load` HOST
      frame put all 1270 expert-GEMM launches inside load, which is
      impossible. The two tiers have different origins and the offset is
      not recoverable from the footers. Re-cut on kernel populations
      instead — the split's last launch, the first N64 GEMM — the
      windows self-evidence: prefill wall 574.81 ms against the
      harness's 574.933, decode 944.39 against 950.59. This is finding 5
      under 2.2.5, and it is the second attribution this profiler
      ergonomic has cost this unit.
      CAUSE 3 COLLECTED 2026-09-17. The grouped id mat-vec existed for
      Q4_K and Q6_K; the codebook formats were not admitted to it.

      | leg | before | after | llama.cpp (Vulkan) |
      |---|---|---|---|
      | decode t/s | 39.86 | **122.1** | 140.1 |
      | | 0.285x | **0.871x** | |
      | prefill t/s | 908.5 | 908.3 | 2309 |
      | | 0.394x | 0.393x | |

      Two kernels, `iq3xxsQ8IdMatVecKernel` and `iq4nlQ8IdMatVecKernel`
      — the dense wave kernels with the slab row through `sel[kk]`, the
      activation base through `xRowBlocks` and the output row changed,
      the dot loop untouched. Bit-identical to the per-expert wave
      launches over 192 rows on both (`MoeCodebookIdMatVecTest`), each
      expert a distinct block rotation so a kernel ignoring `sel`
      could not agree by accident. The IQ4_NL one reads the INTEGER
      route on the caller's padded q8_K activation, where 9.2.4's dense
      path took the f32 wave because 1408 is not a multiple of 256.
      TWO GATES REFUSED, AND I FOUND ONLY ONE BY READING. `idReady()`
      listed 12 and 14; widened, the census was UNCHANGED — the id
      kernels absent, decode 33.65 as before. I had written that the
      kernels "flip `zeroSyncReady()`"; that was the format clause
      assumed to be the only clause. `DenseRouteProbe`, given two
      decode steps, named the real one: "shared expert not row-routed",
      all 24 layers, clause ONE of seven. `Linear.packedWaveReady()`
      ended `(q8 && wave) || wave6` — the same two-format list, one file
      over — and its dispatcher fell through a bare `else` into the
      Q6_K decoder, which would have fed codebook bytes through it the
      moment the predicate widened: wrong logits, no crash. Explicit
      arms now and a terminal `return false`. THIRD INSTANCE TODAY of a
      predicate naming formats where it means "on an integer wave
      route": `wavef`'s guard (9.2.4), `idReady`, `packedWaveReady`.
      Sweep opened as 9.2.9.
      THE CENSUS, decode window (the boundary is the `LlmEngine.load`
      frame's span, which on this trace lands after the prefill; the
      window's 267.96 ms wall against the harness's 279.87 decode ms
      says it is the decode window and nothing else): device self
      214.67 ms in 267.96 ms of wall, 80% busy, from 231 in 944 (24%).
      `iq3xxsQ8IdMatVecKernel` 1424 launches (2 a layer: gate, up),
      `iq4nlQ8IdMatVecKernel` 712 (1: down), `rmsnormRouterTopKKernel`
      712 (the fused device router, new to this file). Per-token
      mat-vec launches ~438 -> ~222; the shared expert (72) and the
      attention projections (~90) stay dense, which is correct.
      PERPLEXITY MOVED AND IS NOT WAVED THROUGH: 5.50046 against
      llama.cpp's 5.4781, +0.41%, from -0.02% on the previous build.
      Inside the ~0.5% MoE routing-flip floor 7.3.2 records, and the
      id kernels are bit-identical to what they replace, so the
      movement is the route's OTHER changes — the fused device router,
      the shared-expert arms, the down+combine. Greedy agrees with
      llama.cpp on the shared prefix and ends at eos where the old
      route continued. OWED before this is called settled: the
      `[diag] route` record diff between the two routes (7.3.2's
      arbiter), so the +0.41% is shown to be top-k flips and not a
      numeric fault in one of the three changed paths.
      WHAT REMAINS IS PREFILL: 0.393x with the expert coop GEMMs
      dominating device time — `iq4nlF16CoopN64` 1270 launches / 231 ms
      at 182 us each, `iq3xxsF16CoopN64` 2540 / 127 ms — inside a
      prefill window that is ~91% device-busy. I first wrote "GEMM-bound,
      not dispatch-bound" from that 91%, and the ARITHMETIC REFUTES IT:
      one down launch is one expert, 2048 x 1408 at 18/32 B = 1.62 MB,
      in 182 us = 8.9 GB/s, 4% of the ceiling. Its grid is
      `(outDim/128) * (rows/64)` = 16 x 1 = SIXTEEN workgroups for ~34
      routed tokens padded to 64; gate/up get 11. The device is "busy"
      running grids that cannot fill it, 3810 of them in series. That
      is cause 3's shape on the prefill side — per-expert dispatch
      where llama.cpp issues one grouped `mul_mat_id` — and it is CAUSE
      5, named.
      AND IT IS THE FOURTH FORMAT LIST: the grouped prefill id-GEMM
      exists (`gemmIdBatchMw` / `gemmIdBatchWide`, the WMMA Mw family
      with an expert map), and `ExpertBank.idGemmReady()` reads
      `(packedTy == 12 || packedTy == 14) && colsN % 256 == 0`. The
      codebook banks are refused it and fall to the per-expert coop
      launch above. Route-table spec §1.1 gains a line.
      THE FIX IS BIGGER THAN CAUSE 3's: the id-GEMMs are WMMA kernels
      (`q4kWmmaIdMwKernel`, `q6kWmmaIdMwKernel`, `symWmmaDeqIdMw8`) with
      an expert map, not wave mat-vecs with a `sel` index — a codebook
      variant is a new kernel family, or an id variant of the coop
      family that reads the same map. THE ISA READ (7.2.1's method,
      `bench/KernelIsa`): `iq4nlF16CoopN64Kernel` 89 VGPRs / 18 SGPRs,
      `iq3xxsF16CoopN64Kernel` 89 / 19 — "spillBytes": 0; "spillBytes": 0. Spill 0 bytes on both,
      18432 B static LDS, and the manifest's own occupancy figure is 3
      resident groups a CU — 24 waves, six a SIMD. The kernels are not
      the problem; the sixteen-workgroup grid is. Nothing in either
      body needs touching for cause 5 — the fix is the launch shape.
      THE ARBITER RAN (`bench/MoeRowArbiter`, `pplprobe ... norowmoe`):
      both routes from ONE binary, `setSharedRow(false)` reproducing
      the old refusal at the same clause. WHAT IT SETTLED: arm B reads
      5.47719 — the old build's figure to five places — so the +0.41%
      is the route, not the build. Eight decode steps on a FIXED token
      sequence (identical inputs to both arms whatever they predict):
      argmax identical on 8 of 8; expert selections differ on 8 of 8,
      by 1-3 experts of 96 a step (2/4/4/2/4/6/2/2 cells of 1440);
      max|A-B| logit 0.61-0.87 every step, which is one swapped
      expert's contribution, not reassociation noise.
      WHAT IT DID NOT SETTLE, stated plainly: no flip-free step
      occurred, so the numerics of the three changed paths are NOT
      isolated — the logit delta at every step carries a flip. And the
      flip RATE is the open question: 1-3% of selections a token is
      more than "near-tied top-k" intuition, and the fused device router
      (`rmsnormRouterTopKKernel`, new to this file's decode) is the
      suspect for a router-logit-level difference rather than a tie.
      THE DECISIVE PROBE, owed: the router logits of both arms BEFORE
      top-k, same layer, same step. ~1e-6 apart means the flips are
      genuine ties and the route is faithful; ~1e-3 or worse means the
      fused router computes something different and that is a finding
      against it, floor or no floor. Not done here — it needs an
      accessor the engine does not expose, and it is a decision whether
      it precedes 9.2.9.
      THE PROBE RAN, with a router tap (`MoeFfn.setRouterTap`, null
      guarded, fired at both arms' router sites) and the arbiter
      printing each layer's router-logit delta per step. VERDICT: the
      fused norm+router is FAITHFUL. Layer 0 — no MoE block upstream,
      attention identical in both arms — differs by 5.2e-6 / 6.7e-6 /
      8.6e-6 at steps 0 / 3 / 7: f32 reassociation. Layer 1 is already
      0.014 / 0.027 / 0.031, three to four orders up after exactly ONE
      MoE block, and the profile sits at 0.03-0.13 through the depth.
      Every flip's margin is under its layer's delta.
      SO THE DIVERGENCE IS INJECTED INSIDE THE MoE BLOCK, and there is
      one place in it where the route changes precision by
      construction: the DOWN projection. Arm B ran it through the f32
      wave (1408 is not a multiple of 256, so 9.2.4's dense path took
      `iq4nlF32WaveMatVec`); arm A runs it through the integer id
      kernel on the caller's q8_K-packed gate*up. The gate/up id kernels
      are bit-identical to the wave kernels they replaced (the test),
      and the shared expert takes the same wave kernels either way, so
      the int8 rounding of the down activation is the change. That is
      inherent to the integer route and it is what llama.cpp's
      `mul_mat_vec_q` does too (q8_1 activations), so it is a precision
      CHOICE, not a fault. Not pinned to the last decimal — pinning it
      means an f32-activation id kernel for down, which is also the
      other arm of the choice.
      THE CHOICE, Julian's: keep 3.06x and +0.41% (inside the floor,
      mechanism named), or an f32-activation id kernel for IQ4_NL down
      that pays the delta back in bandwidth. The tap stays: null
      guarded, two short methods, and it is the instrument that turned
      "inside the floor" into a named cause in one afternoon.
      CAUSE 5 COLLECTED under 6.4.4: prefill 0.393x -> 0.856x. What
      remains on this item is the precision choice (6.4.5) and two
      kernel-rate residues — the grouped coop bodies at ~50-60 GB/s on
      prefill, and decode's last 12% — both under 7.2.1's method, and
      neither a launch count any more.
      RE-OPENED 2026-09-17 (Julian: "Let's get back to 6.4.3. Get it
      done"; then "Idle."): prefill 1.21x, decode 0.878x. The residue is
      decode, and the decode residue is measured below before any kernel
      is touched.
      - [x] 6.4.3.1 MEASURED 2026-09-17: the idle anatomy of a decode
            token. Device busy 228 of 280 ms over 32 tokens (81%). The
            host issues a token's ~600 launches in 1.31 ms and waits 6.83
            ms in the stream sync (`MoeBenchProbe`'s dec-tail timers); the
            ROCm wait-policy knobs (`HSA_ENABLE_INTERRUPT=0`,
            `ROC_ACTIVE_WAIT_TIMEOUT`) move nothing, ABBA. A new
            `cajeta profile summary --gaps` (the idle between consecutive
            slices, histogram + largest, cajeta repo) partitions the
            52.6 ms: 16134 gaps of 2.54 us between consecutive launches =
            40.9 ms, and 29 gaps of ~336 us at the token tail (head kernel
            -> next token's first norm: logits fetch 0.10, argmax, embed
            0.07, first launch) = 9.7 ms. `LaunchFloorProbe` puts the
            floor at 2.0-2.4 us a launch, host submit and device gap
            alike. So decode's idle IS launch count. THE TALLY, per layer
            (24 layers, 25 launches): rmsnormPack 1, q k v 3, bias adds
            3, qkPrep 1, flash 1, reducePack 1, o 1 (accumulate form),
            rmsnormRouterTopK 1, gate 1, up 1, glu 1, pack 1, down 1,
            combine 1, shared gate 1, up 1, glu 1, pack 1, down 1, gate z
            1, sigmoid-add 1. Thirteen of these are epilogue work of a
            neighbour; each fused one saves its ~2.5 us gap plus the tiny
            kernel's own 2-5 us. The fusions exist for Q4_K/Q6_K
            (`idGateUpGlu`, `idDownCombineTail`, `qkvWaveMatVecKernel`)
            and refuse the IQ family, which is why this file pays 25.
      - [x] 6.4.3.2 Bias in the IQ wave mat-vec epilogue (q, k, v carry
            biases on this model): `accum` mode 2 stores `tot + bias[row]`
            on the five IQ wave kernels and the IQ4_NL wave kernel;
            `Linear.launchOne` folds when the staged launch has no output
            scale. -3 launches a layer. TDD: `MoeCodebookIdMatVecTest.
            iq3xxsWaveLaunchAddsItsBias` / `iq4nlWaveLaunchAddsItsBias`
            (fused == plain launch + `addBiasRowsDevice`, exact).
            PREDICTION to check before going on: 72 launches x ~4.5 us =
            ~0.33 ms a token (4%). MEASURED 2026-09-17: bit gate exact
            (greedy stream md5 56a6a744f993ddc2 both binaries); decode
            8.21 -> 8.04 ms a token, 121.9 -> 124.4 t/s (ABBA x3), so
            2.4 us a removed launch, not 4.5: the gap goes, the tiny
            kernel's own time was partly hidden. The remaining fusions
            are re-costed at 2.4 us a launch: 11 launches a layer =
            ~0.63 ms, plus the tail's ~0.25 = 7.16 ms a token = ~140 t/s
            against 138.4 — parity by a hair if every item lands.
      - [x] 6.4.3.3 IQ4_NL down + combine + next-layer norm tail: an
            IQ4_NL arm of `idDownCombineTailLaunchNoSync`. -2 (combine,
            next rmsnormPack). The combine keeps `moeCombineAddKernel`'s
            order (each expert's total reduced as the id mat-vec reduces
            it, then hs += sum_k w_k tot_k) so the fused launch is
            bit-identical to the chain; the Q4_K twin walks (expert,
            block) pairs and its test carries a tolerance. TDD:
            `MoeCodebookIdMatVecTest.iq4nlDownCombineTailMatchesTheChain`
            (hs, normed output and packed bytes exact; counter reset).
            LANDED 2026-09-17. Two lowering lessons on the way: the tail
            copied from the Q4_K twin declares `i`, and a same-named
            int64 in the dot loop had the whole kernel SKIPPED silently
            (`rebuild-tests.sh` filters the note; the cycle script now
            greps it back); and the Q4_K tail sums squares by wave
            reduce + eight partials where `rmsnormPackRowF32` uses a
            256-lane tree, a last-bit difference in `scale` — this tail
            takes the tree so the normed row is bit-identical to the
            launch it replaces. Bit gate exact; decode 8.04 -> 7.97 ms a
            token, 125.4 t/s, against the pre binary's 8.25 in the same
            session (ABBA x3).
      - [x] 6.4.3.4 IQ3_XXS gate + up + GLU in one id launch: an IQ3_XXS
            arm of `ExpertBank.idGateUpGlu`, both dots exactly as the id
            mat-vec takes them, `gluF32`'s activation in the epilogue.
            -2 (the q8_K pack of the GLU output stays a launch; folding
            it needs a 256-row block per workgroup or a last-arriving
            counter, a later item if the tail leaves room). TDD:
            `MoeCodebookIdMatVecTest.iq3xxsGateUpGluMatchesTheChain`
            (exact, gate and up from different fixture offsets). LANDED
            2026-09-17: bit gate exact; decode 7.97 -> 7.76 ms a token,
            128.9 t/s (ABBA x3, pre binary 8.25).
      - [x] 6.4.3.5 The shared expert: gate + up + GLU in one IQ2_S wave
            launch (`Linear.gateUpGluPacked`, two rows a wave as the IQ2_S
            kernel takes them); the IQ3_S down accumulating into the
            residual scaled by sigmoid(z) in its epilogue (`accum` mode
            3, `Linear.matvecPackedAccumGated`; the wave kernels' extra
            buffer is now `aux`: a bias in mode 2, the gate logit in
            mode 3). The gate logit's own mat-vec and the pack stay. -3.
            TDD: `MoeCodebookIdMatVecTest.iq2sWaveGateUpGluMatchesTheChain`
            (63 rows, so the odd-row tail is covered) and
            `iq3sWaveAccumGatedMatchesTheChain`, both exact. LANDED
            2026-09-17: bit gate exact; decode 7.76 -> 7.66 ms a token,
            130.5 t/s (ABBA x3, pre binary 8.24). CORRECTED the same
            evening by a census of the new binary: only the gated down
            had taken. THIS FILE'S RECIPE (printed from the gates): attn
            q, k IQ2_S; attn v, o IQ3_XXS; shared gate, up IQ3_XXS;
            shared down IQ3_S; expert gate, up IQ3_XXS; expert down
            IQ4_NL; output Q5_K — the census had q/k and the shared
            gate/up swapped. Added `iq3xxsQ8WaveGateUpGluKernel` and a
            by-type dispatch in `gateUpGluPacked` (the IQ2_S kernel stays,
            tested, for recipes that quantize a shared expert so). TDD:
            `iq3xxsWaveGateUpGluMatchesTheChain`. A route that is not in
            the census did not take: the census, not the gate's code, is
            the proof.
      - [x] 6.4.3.6 Fused QKV for IQ3_XXS: `iq3xxsQkvWaveMatVecKernel`,
            one body per weight parameter (the lowering knows buffer
            parameters only), the biases in the epilogue as 6.4.3.2
            folds them; `Linear.matvecQkvStagedKeep` takes the arm when
            all three are IQ3_XXS wave routes. -2. TDD:
            `MoeCodebookIdMatVecTest.iq3xxsQkvLaunchMatchesThreeBiasedLaunches`
            (exact). First cut measured FLAT (7.66 -> 7.69) because this
            file's q and k are IQ2_S and the arm never took; the census
            said so. Added `iq2sIq3xxsQkvWaveMatVecKernel` for the
            recipe: IQ2_S row pairs for q and k, IQ3_XXS rows for v, the
            per-type wave bodies verbatim, dispatched by the type triple.
            TDD: `iq2sIq3xxsQkvLaunchMatchesThreeBiasedLaunches` (odd row
            counts, so both pair tails are covered). LANDED 2026-09-17
            with 6.4.3.5's correction in the same cycle: bit gate exact;
            decode 7.69 -> 7.38 ms a token, 135.5 t/s (ABBA x3, pre
            binary 8.22). The census of the new binary: one fused QKV and
            one fused shared gate-up-GLU launch a layer, the plain
            IQ3_XXS wave kernel down to the o-projection; ~340 launches a
            token (was ~600), decode idle 13.4% of the window (was
            18.7%). 0.979x of llama.cpp (Vulkan)'s 138.4 from here.
      - [x] 6.4.3.7 The token tail (~0.3 ms): the argmax on the device
            (`argmaxRowKernel`, the host scan's first-occurrence answer),
            one word downloaded instead of the 608 KB logits row; the row
            is fetched on demand by `logits()` when a sampler or a probe
            asks. The embed gather stays on the host: the table is packed
            on the host and a device copy is a load-time item of its own.
            TDD: `QuantKernelTest.argmaxRowKernelMatchesTheHostScan`
            (ties, the last index, index 0). LANDED 2026-09-17: bit gate
            exact; decode 7.38 -> 7.29 ms a token, 137.2 t/s (ABBA x3,
            pre binary 8.23) — 0.99x of llama.cpp (Vulkan)'s 138.4.
            The census then showed the argmax itself at 194 us: one
            workgroup walking 152k logits. Rewritten as 149 workgroups of
            1024 elements with a last-arriving reduction of the partials,
            the same first-occurrence rule at both levels (the test now
            runs each case twice, so the counter's reset is covered).
            LANDED 2026-09-17: bit gate exact; decode 7.29 -> 7.11 ms a
            token, 140.6 t/s (ABBA x3, pre binary 8.22).
      - [x] 6.4.3.8 Bit gate after each item: the greedy stream of
            `schedthroughput qwen15moe prompt=512 gen=128` hashes
            identically to `tmp/cbq/st-pre643` (the V-d binary); the
            legs ABBA x3 at the end; the item closes at decode >= 1.0x of
            llama.cpp (Vulkan) or with the residue named. HELD on every
            item (md5 56a6a744f993ddc2 throughout). CLOSING LEGS
            2026-09-17 (`tmp/cbq/u6438-legs.log`, engine order
            alternating, 3 reps each, quiet box):

            | 512x128 | cajeta | llama.cpp (Vulkan) | llama.cpp (HIP) |
            |---|---|---|---|
            | prefill t/s | 2758 | 2295 | 1177 |
            | | 1.20x | | 2.34x |
            | decode t/s | 140.8 | 138.4 | 92.9 |
            | | 1.017x | | 1.52x |

            Decode 0.878x -> 1.017x in one evening, every step
            bit-exact; the residue of 6.4.3.1's tally is three launches a
            layer (6.4.3.9), not taken.
      - [x] 6.4.3.9 TAKEN (Julian, 2026-09-17: ".2ms x 1M tokens is 3
            minutes. Go for it."): the last three fusable launches a
            layer — the shared expert's gate logit into its gate-up-GLU
            launch (one extra 256-lane workgroup running
            `routerF32MatVecKernel`'s tree), and the two q8_K packs of
            the GLU outputs by a last-arriving-wave counter per 256-row
            block (`q8kPackKernel`'s arithmetic verbatim; the routed pack
            pads 1408 to 1536). ~72 launches x 2.5 us = ~0.2 ms a token,
            ~1.04x. The gate logit's 256-lane tree is reproduced by ONE
            32-lane wave (eight virtual lanes a lane, the 128/64/32
            levels folded in registers, the last five through LDS) so
            the logit is bit-identical to `routerF32MatVecKernel`'s; the
            pack tails are `q8kPackKernel`'s arithmetic verbatim under a
            release/acquire counter a block. TDD:
            `MoeCodebookIdMatVecTest.iq3xxsGateUpGluPackMatchesTheChain`
            (1408 rows, so the padded sixth block is covered) and
            `iq3xxsWaveGateUpGluPackGateMatchesTheChain` (packed bytes,
            GLU rows and the logit exact; counters reset; two passes).
            FIRST CUT MEASURED SLOWER (7.11 -> 7.27, gate exact): with one
            row a workgroup, every one of ~11,000 waves a layer paid a
            device-scope fence and a counter atomic, and 256 atomics on
            one word serialize. Second cut: eight rows a workgroup, the
            fence and the atomic once a workgroup, wave 0 packing with
            the same 32-lane arithmetic. STILL 7.30: the census put the
            shared launch at 62 us (was 43) — the logit's one-wave
            2048-term loop, dispatched LAST, extended the launch's tail.
            Third cut: the logit on all 256 threads of its workgroup
            with `routerF32MatVecKernel`'s own lane mapping and tree,
            exact by construction and simpler than the eight-virtual-lane
            emulation it replaces. LANDED 2026-09-17: bit gate exact;
            decode 7.11 -> 6.94 ms a token, 144.2 t/s (ABBA x3, pre
            binary 8.21). Census: the packs and the logit launch gone
            from decode, the two fused kernels at 47.9 / 46.6 us where the
            four launches they absorbed cost 61 us of kernel time plus
            three gaps; ~270 launches a token; decode idle 10.3% of the
            window. CLOSING LEGS again (`tmp/cbq/u6439-close.log`):

            | 512x128 | cajeta | llama.cpp (Vulkan) | llama.cpp (HIP) |
            |---|---|---|---|
            | prefill t/s | 2752 | 2286 | 1174 |
            | | 1.20x | | 2.34x |
            | decode t/s | 144.2 | 138.7 | 92.9 |
            | | 1.039x | | 1.55x |


- [x] 6.4.4 Cause 5: the grouped prefill id-GEMM for the codebook
      banks. `ExpertBank.idGemmReady()` admits Q4_K and Q6_K (and the
      widened symmetric slab); the codebook banks fall to one coop
      launch per expert — sixteen workgroups, 8.9 GB/s, 3810 in series
      — and prefill sits at 0.393x of llama.cpp (Vulkan). The existing
      id-GEMMs are WMMA kernels driven by an expert map
      (`q4kWmmaIdMw`, `q6kWmmaIdMw`, `symWmmaDeqIdMw8`); the codebook
      coop bodies clear the IQ4 bar and have no map. SPIKE FIRST: an id
      variant of the coop family reading `mapE / mapT0 / mapM1`, IQ3_XXS
      only, measured against the per-expert launches on the MoE before
      the second format is written. Bit gate: the grouped GEMM equals
      the per-expert GEMMs over the same slab. ISA read is done (89
      VGPRs, no spill, 3 groups/CU) — the launch shape is the whole
      cause. Admission is a route-table row, not a fifth list (9.2.9).
      SPIKED AND MEASURED 2026-09-17, IQ3_XXS (gate, up):
      `iq3xxsF16CoopIdN64Kernel` — the N64 body verbatim, a prologue
      that reads expert / t0 / mEnd from the 64-row chunk map, weight
      rows from the bank's whole slab, activation rows from ONE f16
      stage of the gathered batch, and a guarded tail store (spill to a
      per-wave LDS tile, land rows below mEnd) because in one launch the
      next chunk's rows are live. Bit-identical to the per-expert
      launches over ragged groups of 34 / 70 / 0 / 5 rows, canary rows
      past the batch untouched (`MoeCodebookIdGemmTest`).
      THE GATE WENT PER BANK: `idPath` had required all three banks
      ready, so an IQ4_NL down would have refused the whole layer — the
      same trap as cause 3. Gate/up grouped, down per-expert, the route
      record names the mix. And the 64-row maps were uploaded only
      under `anyMw` (WMMA banks), which would have left the coop-id
      branch reading a stale `meta` — zero chunks, nothing launched,
      suite green. Found by reading; widened.

      | | before | after | llama.cpp (Vulkan) |
      |---|---|---|---|
      | prefill t/s | 908 | **1107** | 2302 |
      | | 0.393x | **0.481x** | |
      | decode t/s | 122.1 | 122.3 | 140.1 |

      Census: `iq3xxsF16CoopN64` 2540 launches / 127 ms became
      `iq3xxsF16CoopIdN64` 96 / 94 ms (982 us a launch, ~600
      workgroups); the per-expert copy-outs went with them. Down is
      untouched at 1270 / 231.66 ms and is now the largest item at 31%
      of the after-load window: the IQ4_NL twin is the next step, same
      shape, the N64 iq4nl body. After that the grouped launch itself
      reads ~59 GB/s — the coop body's rate, a kernel question, not a
      launch one. `hasCoopIdKernel(ty)` is a registration list in
      `QuantKernel`, the shape the audit walks; it becomes a row.
      THE IQ4_NL TWIN LANDED the same afternoon: the iq4nl N64 body
      under the same prologue and guarded tail, `coopIdReady` asking the
      format's own column rule (`coopColsOk`: 64 for IQ4_NL, so down's
      1408 is admitted), one f16 stage of the GLU output before the
      down GEMM. Bit-identical to the per-expert launches at 1408 wide,
      canary intact. `iq4nlF16CoopN64` 1270 launches / 232 ms became
      `iq4nlF16CoopIdN64` 48 / 121 ms (2.52 ms a launch), and the route
      record reads `resident: id GEMMs` with no mix.

      | 512x128, ABBA x3 | opened | after cause 3 | **after cause 5** | llama.cpp (Vulkan) | llama.cpp (HIP) |
      |---|---|---|---|---|---|
      | prefill t/s | 615.3 | 908.5 | **1966** | 2296 | 1180 |
      | | 0.268x | 0.394x | **0.856x** | | 1.67x |
      | decode t/s | 36.75 | 122.1 | 122.4 | 139.0 | 93.0 |
      | | 0.266x | 0.878x | 0.881x | | 1.32x |

      Prefill 2.16x on this item, 3.2x since 6.4.3 opened; cajeta now
      clears llama.cpp's HIP build on both legs of this model. DONE.
      What the census leaves for the next item is not a launch count:
      the two grouped launches read ~60 and ~50 GB/s of slab, the coop
      body's own rate on these shapes, so the remaining 14% to
      llama.cpp (Vulkan) on prefill is a kernel-rate question under
      7.2.1's method (ISA read first, one variable at a time).
- [x] 6.4.6 The grouped coop bodies' rate: prefill past 0.856x. The
      6.4.4 census left two kernels at 61% of the prefill window, down's
      `iq4nlF16CoopIdN64` 2.4x slower a launch than gate/up's iq3xxs
      twin at equal FLOPs. ISA READ FIRST (KernelIsa manifests):

      | kernel | vgpr | spill | LDS | groups/CU |
      |---|---|---|---|---|
      | `iq4nlF16CoopIdN64` | 90 | 0 | 26624 | 2 |
      | `iq3xxsF16CoopIdN64` | 148 | 0 | 28160 | 2 |
      | `iq4nlF16CoopN64` (dense) | 89 | 0 | 18432 | 3 |

      Two variables, one at a time. A: the iq4nl body's expansion is
      `iq4KvS`, a sixteen-branch if-chain, called sixteen times a lane a
      k-step, where the decode kernel does the same remap in two
      `v_perm_b32` through `Vector.lut4`. B: the guarded tail's 8 KB LDS
      tile (`fs`) is exactly the difference between 2 and 3 resident
      groups on both grouped kernels.
      A LANDED 2026-09-17: `packed.vload<4>(ro).asBytes()`, `lut4` on
      the low and high nibbles, thirty-two constant-lane stores into the
      f16 stage (a runtime lane index allocas). Bit gate unchanged
      (`MoeCodebookIdGemmTest`, 309 green, the one pre-existing 9.3.1
      failure). `iq4nlF16CoopIdN64` 48 launches / 121 ms became 48 /
      43.4 ms (2.52 -> 0.904 ms a launch); the iq3xxs twin sits at 0.978
      ms, so the two bodies now run within 8% of each other.

      | 512x128, ABBA x3 | after cause 5 | **after A** | llama.cpp (Vulkan) | llama.cpp (HIP) |
      |---|---|---|---|---|
      | prefill t/s | 1966 | **2424** | 2301 | 1178 |
      | | 0.856x | **1.053x** | | 2.06x |
      | decode t/s | 122.4 | 122.1 | 139.3 | 93.2 |
      | | 0.881x | 0.877x | | 1.31x |

      Prefill now clears llama.cpp (Vulkan) on this model. B is open:
      the two grouped kernels are 137 ms of a ~215 ms prefill at 2
      groups/CU, and the layout that removes the tail entirely is a
      chunk-major batch (every chunk owns its 64 rows, so the coop store
      never crosses into a neighbour and `fs` and the copy loops go).
      B REFUTED BY THE CEILING PROBE: both kernels with `fs` and the
      guarded tail removed (timing only, wrong output) manifest at
      19968 / 18432 bytes of LDS, 3 groups/CU, and run at 995 / 913 us
      a launch against 978 / 904 — no change. Occupancy is not the
      binding constraint; the chunk-major layout is not worth a
      refactor for it. Reverted.
      THE ISA READ NAMES THE CAUSE: per k-step the body pays SIX
      serialized global round trips — the scale word (wait), the weight
      words (wait), then a `while (ks < 4)` loop the compiler does not
      unroll, each pass loading two activation tiles and waiting
      `vmcnt(0)` before its four `v_wmma`. Weight bytes in flight per
      lane per step: 16. The kernel is a latency chain, which is why a
      third resident group bought nothing and why it reads 60 GB/s.
      C: the eight activation tile loads issued at the top of the
      k-step, before the expansion, and the four sub-steps written out
      (t00..t31 / u00..u31) so the loads overlap the expansion and the
      barrier. MEASURED FLAT: the ISA shows the loads hoisted and the
      waits spread (`vmcnt(16)`, `vmcnt(3)`), 88 -> 189 VGPRs, no
      spill, and the launch times do not move (962 / 975 us). So the
      global round trips were not the chain either. What the ISA
      proved instead: the 96 / 48 launch counts are the bench's warm-up
      prefill plus the measured one, so a launch covers the whole
      512-token batch, ~70 chunks x 11 row tiles, 25.8 GFLOP in 978 us
      = 26 TFLOPS against a 59 TFLOPS dense-WMMA ceiling — 44% of
      peak, half of it spent on rows past each group's end (34 rows a
      group on average, computed 64 at a time). Not a bandwidth
      kernel; the "60 GB/s" was the wrong lens.
      E LANDED 2026-09-17, on C's unrolled body: per wave, the two
      16-token tiles are live only while `cbw < mEnd` / `cbw + 16 <
      mEnd` (wave-uniform), and a dead tile skips its activation loads
      and its two `mma`s. The expansion and the barriers stay
      workgroup-wide. Bit gate unchanged. 962 -> 778 us (iq3xxs), 975
      -> 753 us (iq4nl); 211 / 139 VGPRs, no spill.

      | 512x128, ABBA x3 | after A | **after E** | llama.cpp (Vulkan) | llama.cpp (HIP) |
      |---|---|---|---|---|
      | prefill t/s | 2424 | **2749** | 2303 | 1179 |
      | | 1.053x | **1.194x** | | 2.33x |
      | decode t/s | 122.1 | 121.8 | 139.0 | 93.0 |
      | | 0.877x | 0.876x | | 1.31x |

      D (weight words one k-step ahead) is dropped: C proved the
      round trips are hidden already. What the census leaves, per
      prefill: the grouped bodies 55 ms, the shared expert's dense
      N256 bodies 47 ms (same expansion, 16-19 TFLOPS, no ragged
      waste), the f32 router GEMM 10 ms (203 us a launch for 126
      MFLOP), and the device idle ~10% of the window (a 50 ms scan of
      the device tier: the prefill sits at 90% busy, decode at 80%).
      The scan also shows where the other 48 grouped launches live:
      not a second prefill but one full-size launch a layer spread
      through the LOAD phase, [250, 1550) ms at 17% device-busy — a
      per-layer warm-up forward interleaved with the upload, 1.3 s of
      the 2.1 s load. Not this item's; noted for the load leg.
      NEXT PROBE: every wave in a workgroup loads its own activation
      tiles from global — the four row-waves sharing one token slab
      fetch it four times, and every row tile refetches it — ~800 MB
      of L2 traffic a launch against 66 MB of slab. PROBED (all tile
      loads pointed at one cache-resident block, timing only): 778 ->
      640 us (iq3xxs), 753 -> 665 us (iq4nl). Activation traffic is
      ~15% of a launch, so staging the token slab through LDS (9 KB
      more, one more barrier, likely one group/CU) is bounded at a few
      percent of prefill and is NOT built. Reverted.
      WHERE THIS LEAVES 6.4.6: prefill 1.194x of llama.cpp (Vulkan),
      decode 0.876x (untouched by this item). The ~640 us that remain
      in a grouped launch are the body itself — expansion, the LDS
      round trip, two barriers a k-step — at 23 TFLOPS effective
      against 59; the shared expert's dense N256 bodies (47 ms a
      prefill, 24%) sit at the same 28% of peak with no ragged waste.
      A body-level redesign (weights AND activations staged, a wider
      token tile per wave) is the next lever and a new item, not a
      residue of this one. Also noted: the 1.3 s per-layer warm-up
      inside load, for the load leg.
- [x] 6.4.7 THE COOP BODY REDESIGN (Julian, 2026-09-17: "let's redesign
      the body, then"). The grouped bodies run at ~23 TFLOPS effective
      and the shared expert's dense N256 bodies at ~16, against 59
      dense-WMMA; 6.4.6's probes refuted occupancy, global-load latency
      and activation traffic (15%) as the bound. What is left is the
      body's own per-k-step structure: expand a 128x64 weight tile into
      LDS, barrier, 16 LDS tile loads + 8 global tile loads + 16 mma a
      wave, barrier. The same rule as 7.2.1: no tile shape is chosen
      before the cost of each phase is measured and llama.cpp's
      factoring is read.
      - [x] 6.4.7.1 MEASURED 2026-09-17, timing-only probes on both
            grouped bodies, one phase varied each (two of them crashed
            the run on non-finite garbage before they were made
            finite — a timing probe still has to keep the model's
            numbers finite, or the trace it leaves is the OLD one and
            reads as "no change"):

            | us a launch | iq3xxs | iq4nl | varied |
            |---|---|---|---|
            | as committed | 778 | 753 | — |
            | P1 expansion -> one store a lane | 595 | 590 | expansion ALU + 31 stores |
            | P2 A tiles splatted, LDS gone | 477 | 524 | + the LDS loads |
            | P3 every mma twice | 944 | 923 | MMA marginal +166 / +170 |
            | (6.4.6) tiles from one cached block | 640 | 665 | activation traffic |

            So of 778: expansion ~183, LDS A loads ~118, activation
            loads ~120, the MMA at peak ~247 (14.6 GFLOP effective at
            59 TFLOPS), and ~110 of barriers, prologue and epilogue.
            The MMA is a third of the launch and the two phases the
            barriers serialize around it are half. The expansion's
            cost is nearly the same for the light iq4nl body (two
            perms + 32 cvt/mul) as for the heavy iq3xxs one, so it is
            the phase, not the arithmetic, that costs.
      - [x] 6.4.7.2 READ llama.cpp's Vulkan `mul_mm` for `mul_mat_id`
            on this device class (2026-09-17, an Opus read of
            `ggml-vulkan.cpp` and `mul_mm.comp`, file:line kept in the
            session). On RADV with KHR coopmat there is NO integer/MMQ
            GEMM for `mul_mat_id` for any type (every `CREATE_MMQ` sits
            in the non-coopmat branch) and no `mul_mmq` variant for
            any IQ type on any device — so their MoE path is the same
            f16-WMMA-with-dequant-into-LDS shape as ours. Tile choice
            uses n = total tokens (512), not rows per expert, so the
            LARGE config runs: BLOCK 256, BM 128, BN 128, BK 32, 4
            warps of 64x64 (16 accumulators a warp), LDS pitch BK/2+4
            f16vec2 (the coopmat bank pad), ~26 KB. Per k-block: A
            dequantized into `buf_a` (16 values a thread), B gathered
            through `row_ids` and converted f32->f16 at the LDS write,
            ONE barrier, then 8 A loads + 32 B loads + 32 mma a warp
            (B re-loaded for every A row; A reuse 4x, B reuse 1x), ONE
            barrier. Single-buffered, no prefetch. Ragged: a
            `count_experts` prepass kills whole workgroups past the
            expert's count, but inside a surviving tile DEAD WORK IS
            NOT SKIPPED — at 34 rows in a 128-wide tile 73% of the mma
            is dead and the `warp_c == 1` warps produce nothing; only
            the store is guarded, through an LDS stage with 32
            workgroup barriers. Three things theirs does that ours
            does not: the token gather lands in LDS as one dense tile
            before the K loop; B is staged in LDS (converted once,
            read by all warps); and a runtime selection layer (three
            tiles x aligned x accumulator type, an LDS-budget
            feasibility pass, a vendor/driver warptile override) — the
            route-table shape of 9.2.9. What ours does that theirs
            does not: 6.4.6 E's per-tile dead-work skip, and 64-token
            chunks against their 128 — which is why we lead them on
            this model despite the same body family.
      - [x] 6.4.7.3 DESIGN (2026-09-17). Keep the workgroup (128 rows x
            64 tokens, 8 waves of 32x32, E's per-tile skip, the guarded
            tail through `fs`) and change the k-loop's PHASE STRUCTURE:
            the expansion of k-step i+1 runs in the same phase as the
            MMA of k-step i, into a second LDS buffer, with ONE barrier
            a k-step. BK drops from 64 to 32 columns so two A buffers
            fit: 2 x 128 x 40 halfs (32 + the 8-half coopmat pad) = 20
            KB, plus `fs` 8 KB and the IQ3 tables 1.5 KB = 29.5 KB ->
            2 groups/CU, the count the manifest gave the current body.
            Each lane expands HALF a block a k-step (16 values; the
            half is `aHalf`, wave-uniform, so the constant-lane stores
            sit in a uniform branch); per wave per k-step 4 A loads, 4
            activation loads, 8 mma. B is NOT staged (bounded at 15%
            by 6.4.6's probe; the LDS it needs is the group we would
            lose). Barrier count per column is unchanged. Prediction
            from 6.4.7.1: the expansion's ~170 us hides behind the
            MMA; the LDS loads and the activation loads stay; iq3xxs
            778 -> ~600 us, iq4nl 753 -> ~590 us, prefill ~185 -> ~167
            ms, ~1.32x of llama.cpp (Vulkan).
      - [x] 6.4.7.4 TDD: `MoeCodebookIdGemmTest` (grouped == per-expert,
            canary intact) was the bit gate for every variant above,
            green on all five and on V-d; the test now builds the
            slab's `halfView()` beside its word view.
      - [x] 6.4.7.5 CODED AND MEASURED 2026-09-17 — the design is REFUTED
            on this hardware. Every variant passed the bit gate; none
            beat E's body:

            | us a launch | iq3xxs | iq4nl | LDS | groups/CU |
            |---|---|---|---|---|
            | E (committed) | 778 | 753 | 28160 / 26624 | 2 |
            | pipelined, BK=32, one barrier | 1170 | 1190 | 30208 / 28672 | 2 |
            | + second tile's row loop-invariant (V-a) | 1070 | 1010 | same | 2 |
            | + scales through `halfView` (V-b) | 1070 | 1020 | same | 2 |
            | pipelined, BK=64, one barrier (V-c) | 844 | 835 | 46592 / 45056 | 1 |

            The ISA of the BK=32 body showed the software half->float
            (40 instructions, six branches, a subnormal loop) once a
            lane a step and the second activation tile of each pair
            split into five loads plus thirteen half-word moves; fixing
            both (V-a, V-b) recovered 100-180 us and left the body a
            third slower than E. At BK=64 the same single-barrier
            pipeline (V-c) runs within 9% of E — from ONE resident
            group, since two 18 KB buffers plus the 8 KB tail stage
            exceed the 32 KB that two groups allow. So the phase
            structure buys about what the lost group costs, and no LDS
            budget reaches both: with `fs` gone (chunk-major output,
            a MoeFfn change) the double buffer is still 38 KB.
            Instruction counts do not explain V-c's 844 either — its
            issue work sums to ~290 us — so the stall is latency:
            each k-step still exposes the weight-load wait before the
            expansion and the activation-load wait before the mma,
            back to back. Hiding them means loads two steps ahead in
            registers (+64 VGPRs: iq3xxs's 212 cannot take it) or the
            token slab in LDS (over budget). The tile route llama.cpp
            and hipBLASLt take on RDNA3 — bigger per-wave tiles at
            half the VGPR cost — is wave64, and the amdgpu backend
            pins wave32 (`oclc_wavefrontsize64_off`): a compiler item,
            not a kernel one.
            KEPT: E's bodies with the scales read through the new
            `KernelBuffer.halfView()` (cajeta stdlib, one load and a
            hardware convert instead of the branchy decode; measured
            flat, bit-identical, 125 more call sites can follow).
            V-d, that final state: 752 / 737 us a launch (-3%), bit
            gate green, legs in 6.4.7.7.
      - [x] 6.4.7.6 ROLL OUT: nothing to roll out — 6.4.7.5 did not
            win. The `halfView` scale read is a separate cleanup across
            the 125 `halfBitsToF32` kernel call sites, item 9.2.11.
      - [x] 6.4.7.7 ACCEPTED 2026-09-17 on V-d (E's bodies, scales through
            the half view):

            | 512x128, ABBA x3 | after E | **V-d** | llama.cpp (Vulkan) | llama.cpp (HIP) |
            |---|---|---|---|---|
            | prefill t/s | 2749 | **2769** | 2288 | 1175 |
            | | 1.194x | **1.210x** | | 2.36x |
            | decode t/s | 121.8 | 121.5 | 138.4 | 93.0 |
            | | 0.876x | 0.878x | | 1.31x |

            Decode unmoved. THE ITEM'S RESULT is negative on its
            question: under a 128x64 / 8-wave workgroup at two groups
            per CU, the body's phase structure cannot be pipelined
            within the LDS the group count allows, and the per-step
            costs that remain (two exposed load latencies a step) need
            registers the wave32 tile does not have. The ceiling for
            this body family on gfx1151 is where E sits; the next
            level is wave64 tiles, a compiler item.
- [x] 6.4.5 The precision choice on the down projection. The zero-sync
      row runs down through the integer id kernel on q8_K-packed
      gate*up; the route it replaced ran the f32 wave. Router faithful
      to 5e-6; perplexity +0.41% (5.50046 against 5.4781), inside the
      floor, mechanism named. Either: KEEP, and record the trade beside
      llama.cpp's own q8_1 activations; or an f32-activation id kernel
      for IQ4_NL down, measured for the bandwidth it costs. Julian's
      call; the arbiter (`bench/MoeRowArbiter`, `MoeFfn.setRouterTap`)
      is the instrument either way.
      THE DECIMAL, MEASURED 2026-09-17 (Julian: "I want the decimal";
      I had predicted rounding alone would come in "well under 0.1%"
      and that is REFUTED). `MoeFfn.setRoutePin` records one route's
      expert SELECTIONS and replays them on the other, leaving each
      route its own logits and its own weights, so a pinned pair
      differs in arithmetic only; `bench/MoePinnedPpl` runs the four
      passes over `llama.cpp/README.md`, pre=1025 eval=1023, the same
      corpus and shape as 7.3.2's figures.

      | route | own routes | pinned to the other's |
      |---|---|---|
      | A zero-sync row (integer down) | 5.50046 | 5.48918 |
      | B per-expert (f32 down) | 5.47719 | 5.48325 |

      THE INSTRUMENT IS VALIDATED BEFORE THE RESULT IS READ: both own-
      route figures reproduce the published ones to five places (5.50046,
      5.47719); the pin covered 24552 rows (1023 tokens x 24 layers) with
      0 overflow; and the weight rule it replays with (`weightsFor`)
      matched `routeFromLogits`'s own weights on every one of those rows,
      0 misses, which is what stops a replayed route being weighted
      differently from the route that recorded it.
      THE DECOMPOSITION of the +0.425% gap: rounding alone is +0.219%
      held on B's routes and +0.314% held on A's, so flips are the
      remaining +0.11 to +0.19%. Rounding is the MAJORITY of the gap,
      not the minority — "inside the routing-flip floor" was the wrong
      reading of it. The flip rate is also settled and it is not a
      handful of near-ties: 4562 and 4703 of 24552 rows carry at least
      one differing expert, 19% of rows.
      THE MECHANISM, measured by `MoeFfn.setActErrTap` (the q8_K round
      trip simulated on the row's real activations, host-side, read-only
      — the four perplexities are unchanged with it on): the relative
      RMS error is 1.12% on the layer input, which BOTH routes quantize
      and which therefore cancels, against 1.70% on the GLU output,
      which only the integer down quantizes. The cause is block width,
      not the format: the mean block outlier max|x|/rms is 4.45 on the
      layer input and 6.82 on the GLU output — a SwiGLU product carries
      outliers, and q8_K spends one scale on 256 elements. Simulated in
      32-element blocks the same GLU output costs 0.91%, BELOW what the
      layer input already costs at 256.
      AND THE REFERENCE DOES NOT MAKE THIS TRADE. `ggml_vk_should_use_mmvq`
      (llama.cpp 5306f4b, `ggml-vulkan.cpp`) refuses the integer
      activation path on AMD when the contraction dimension is under
      2048; the routed down projection's is 1408 and gate/up's is 2048,
      so at decode llama.cpp quantizes the gate/up activations on this
      file and leaves the down's in f16. The plan's earlier note that
      "llama.cpp's `mul_mat_vec_q` does too (q8_1 activations)" is true
      of its CUDA/HIP dense path and not of the Vulkan arm this unit
      benchmarks against, which is why its 5.4781 sits at our f32 arm's
      5.47719 rather than at our integer arm's 5.50046.
      SO THE CHOICE WAS THREE-WAY: keep the integer route and record the
      trade; an f32-activation id kernel for the down, which costs the
      fused down+combine+tail launch of 6.4.3.3 and the int8 dot; or
      quantize the down's activation in 32-element blocks, which the tap
      said recovers roughly half the error for a new pack in the
      gate-up-GLU epilogue and a changed inner loop in
      `iq4nlQ8IdDownCombineKernel`, keeping the fusion and the dot.

      THE COST, MEASURED 2026-09-18 (Julian: "Measure the cost. Explore
      the block width"), and it settles the item: KEEP.
      `MoeFfn.setActBlockSim(w)` rounds the down projection's
      activations to `w`-element int8 blocks by the pack kernel's own
      rule, inside the per-expert route whose down reads f32 — so a
      width costs what its kernel would cost, with no kernel written.
      `bench/MoeBlockWidthPpl` scores 24 independent 2048-token windows
      of `tmp/cbq/calib.txt`, 24552 positions, each width pinned to the
      control's expert selections.

      | down activations | rounding RMS | ppl vs f32 |
      |---|---|---|
      | q8_K, blocks of 256 (ships) | 1.72% | +0.101% +- 0.054% |
      | blocks of 64 | 1.15% | +0.106% +- 0.054% |
      | blocks of 32 | 0.91% | -0.022% +- 0.053% |

      THE COMPARISON IS PAIRED, position by position, and that is the
      whole reason the table means anything. The first cut compared
      aggregate perplexities over 1023 positions and read +0.02% at 256,
      +0.50% at 128 and +0.54% at 64 — non-monotone against a rounding
      error that falls monotonically, and at one point blocks of 32 came
      out NEGATIVE. The per-position scatter is +-0.08 nats against an
      effect of 0.001, so an aggregate difference at that length is
      trajectory noise. The null control — a pinned pass with no
      rounding at all — returns dNLL exactly 0 +- 0, which is what says
      the pairing and the pin are sound before any width is read.
      WHAT IT SAYS. The shipping route's activation rounding costs
      +0.10% +- 0.05%, about 1.9 sigma from zero. Blocks of 64 cost the
      SAME +0.11% on a third less rounding error, so the perplexity is
      not tracking the error magnitude; blocks of 32 land at zero, but
      256 - 32 is 1.6 sigma and NOT resolved. Nothing here buys the
      f32 kernel, and a 32-wide kernel would chase at most 0.12% on 1.6
      sigma of evidence for no throughput gain — the activation is 1760
      bytes against the expert's whole IQ4_NL weight read, so width
      cannot move decode.
      AND THE +0.25% DECIMAL WAS OVER-READ. The pinned A-vs-B figures
      (+0.219%, +0.314%) are differences of the same 0.003-nat size and
      carry the same +-0.26% at 1023 positions, which nobody quoted. The
      down's activation rounding is +0.10% of the +0.425% gap; the rest
      is flips plus every other arithmetic difference between two
      different implementations, not the activation quantization the
      mechanism note attributed it to. My recommendation reversed twice
      and lands where Julian started: KEEP.

## Unit 7 — The decode bandwidth gap (spec §6, §8.5, 12.1; folds 4.3.5 and 6.4.1)

Every ternary and codebook decode wave mat-vec runs below llama.cpp's
Vulkan on the same file and the same device: tg 0.66–0.85× across eight
IQ files (6.3's legs) and 132 GB/s for TQ2_0 against the Q4_K wave
kernel's ~200 on this GPU (4.3.3's). One kernel family, one device, and
the Q4_K kernel sitting beside them as a worked example of the same
shape going fast — so this is a cause to FIND, not a rewrite to guess
at. Nothing in this unit changes a kernel before 7.2.1 records why.

### 7.1 TDD
- [x] 7.1.1 `bench/CodebookBandwidth`: achieved GB/s per kernel at the real
      shapes (4096×4096, 4096×14336, 14336×4096, 1024×4096) for Q4_K,
      TQ1_0, TQ2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S — resident
      bytes over the min of five interleaved repeats, one table. The
      Q4_K row is the CONTROL: if it does not reproduce ~200 GB/s the
      probe is wrong and nothing below it may be read.
- [x] 7.1.2 The ISA read per kernel (`RADV_DEBUG=shaderstats`): VGPR
      count, scratch bytes, occupancy; a guard that asserts no scratch
      spill on any of them. Spilling is invisible to wall-clock, so a
      flat A/B without this read proves nothing.
      GREEN on all sixteen wave kernels (`bench/KernelIsa`, which
      already existed for exactly this); the counts are in 7.2.1.
- [x] 7.1.3 The exactness gate: every Q8-twin, coop and fixture test in
      `TernaryTest` and `IqCodebookTest` still passes after any rewrite.
      A faster kernel that is not the same kernel is not a fix.
      HELD THROUGH EVERY VARIABLE of 7.2.2 — the filtered suite was run
      after each edit and never went green on a shortcut: it grew 229 ->
      240 -> 253 as the fused paths that the unit reached needed cases
      of their own (`NormPackTest`'s four fused-width cases,
      `Gqa8FlashDecodeTest.reducePackMatchesReduceThenPackGqa4`). The
      suite now carries `Gqa8FlashDecodeTest`, `ResidentMoeDecodeTest`,
      `QkNormBatchTest` and `BackendParityTest` as well. The seventh
      variable also passes the end-to-end form of the same gate: the
      greedy stream of 128 tokens on llama8b-iq2_xs is byte-identical
      across the change.
- [x] 7.1.5 The CPU control: would either family decode faster on the
      CPU? This box is a Zen 5 with AVX512-VNNI and VBMI sharing one
      LPDDR5X pool with the iGPU, so neither side wins on the bus.
      MEASURED 2026-09-16 (`tmp/cbq/cpu-vs-gpu.sh`, llama.cpp `-ngl 0`,
      tg128 at depth 512), converted to achieved weight bandwidth:

      | file | GB | cpu t/s | cpu GB/s | cajeta GB/s | vulkan GB/s |
      |---|---|---|---|---|---|
      | iq2_xxs | 2.22 | 32.1 | 71 | 137 | 175 |
      | iq2_xs | 2.42 | 31.7 | 77 | 118 | 178 |
      | iq2_s | 2.52 | 30.9 | 78 | 121 | 176 |
      | iq3_xxs | 3.04 | 25.1 | 76 | 154 | 181 |
      | iq3_s | 3.45 | 21.2 | 73 | 148 | 190 |

      NO for the codebook family: the CPU is half our rate and 40% of
      Vulkan's. And it is FLAT — 71-78 GB/s on every format, whatever
      the table work — which says the CPU is simply at its memory
      ceiling on an 8B model and the format never enters into it. The
      one ternary model we have is 700M, where CPU (175 t/s) and cajeta
      (212) both sit near 50 GB/s, far under either ceiling: that model
      is overhead-bound and answers nothing about the format.
      What this DOES settle is the target. Vulkan reaches 175-190 GB/s
      end to end, above this machine's CPU ceiling and near its memory
      ceiling, while our best ISOLATED IQ2_XS kernel in 7.1.1 runs at
      123 GB/s — below Vulkan's whole-model rate on the same file. So
      the gap is the kernels, not the engine around them, and the
      roofline to aim at is ~190 GB/s rather than "beat the CPU".
- [x] 7.1.4 A lane-mapping control at one fixed shape — one item per
      row against one wave per row — so the mapping is measured rather
      than assumed to be the ceiling (it was, at 208 against 162 GB/s,
      the last time this question came up on this device).
      MEASURED 2026-09-16 on IQ2_XS, all four probe shapes, with a
      throwaway `iq2xsQ8ItemMatVecKernel` that runs the SAME
      `iq2xsSub` body with one work item per row — a lane walks its
      whole row, eight sub-blocks a block, instead of eight lanes
      sharing a block. Only the mapping changes. Q4_K, TQ and the IQ3
      rows are the in-run control and move less than 0.5%:

      | shape | wave | item | wave / item |
      |---|---|---|---|
      | 4096x4096 | 136.7 | 77.5 | 1.76x |
      | 14336x4096 | 189.1 | 50.8 | 3.72x |
      | 4096x14336 | 166.2 | 86.7 | 1.92x |
      | 1024x4096 | 73.2 | 29.2 | 2.51x |

      Far wider than the 208/162 this question got on Q4_K, and widest
      exactly where the unit's models live: at 14336x4096 the item form
      gives up 3.7x. Two reasons compound in the IQ body that do not in
      Q4_K's -- a lane must walk eight sub-blocks serially instead of
      one, and the 32 lanes of a wave read 32 DIFFERENT rows, so every
      weight load is a 32-way scatter where the wave form reads one
      row's 288-byte span across the wave. The wave mapping is not an
      assumption; it is worth more than every variable in 7.2.2 put
      together. The control kernel was deleted after the reading.
      The same run is the clean arm for two other questions. The
      `accum` parameter of the eighth variable is FREE: iq2xs reads
      189.13 GB/s against the 189.1 of the round before the flag
      existed. And the two-bit revert is exact -- every IQ row is back
      within 0.6% of its pre-variable value.

### 7.2 Coding
- [x] 7.2.1 Record 7.1.1's table and 7.1.2's ISA read in this plan, and
      name the cause each row points at, BEFORE editing a kernel.
      DONE 2026-09-16 (`bench/CodebookBandwidth`, `bench/KernelIsa`).
      The control reproduces: Q4_K reads 206 GB/s at 14336×4096, so the
      probe is sound. But GB/s is the WRONG METRIC here, and that is the
      first finding — normalized per weight VALUE at 14336×4096:

      | kernel | GB/s | G values/s | v_dot4 | vmcnt(0) | dot4 per drain | VGPR |
      |---|---|---|---|---|---|---|
      | tq2_0 | 193 | 805 | 32 | 1 | 32.0 | 63 |
      | iq2xxs | 143 | 596 | 8 | 1 | 8.0 | 101 |
      | tq1_0 | 95 | 483 | 16 | 1 | 16.0 | 149 |
      | iq3xxs | 168 | 470 | 8 | 1 | 8.0 | 104 |
      | iq2xs | 123 | 457 | 4 | 3 | 1.3 | 70 |
      | iq2s | 133 | 447 | 4 | 3 | 1.3 | 66 |
      | q4_k | 207 | 394 | 80 | 10 | 8.0 | 127 |
      | iq3s | 157 | 392 | 4 | 4 | 1.0 | 71 |

      Q4_K and IQ3_S decode values at the SAME rate (394 / 392 G/s);
      Q4_K only looks faster in GB/s because it carries 4.5 bits per
      weight where IQ3_S carries 3.4. So none of these kernels is
      bandwidth-bound and spec 8.5's "bandwidth ceiling" framing does
      not hold below 4 bits: a format that halves its bytes at the same
      value rate takes the SAME time, and decode t/s cannot improve by
      reading less. Time tracks values decoded.
      THE CAUSE, and it is one cause: `dot4 per drain`. Every
      `vmcnt(0)` is a full memory drain, and the three slowest kernels
      issue 4 dot4s between drains while TQ2_0 issues 32. IQ2_XS, IQ2_S
      and IQ3_S interleave single-byte loads with the table gather that
      DEPENDS on them (the ISA shows `global_load_d16` and `u8` in
      exactly those three), so each group of 8 stalls the wave on its
      own index. IQ2_XXS and IQ3_XXS read the whole 8-byte descriptor
      as dwords first, compute four indices in registers, then issue
      their gathers together — one drain, 8 dot4s, and 30-50% more
      throughput on strictly more table work. No kernel spills
      (7.1.2 green: `vgpr_spill 0 scratch 0` on all sixteen), so
      occupancy and pressure are not the story; issue order is.
      TQ1_0 is its own row: no gather at all, yet 483 G/s and 149 VGPRs,
      because 64 `global_load_d16` per body decode the base-three trits
      a half-word at a time.
- [x] 7.2.2 The change the evidence asks for, one variable at a time,
      each re-measured against 7.1.1's table and gated on 7.1.3.
      FIRST VARIABLE (from 7.2.1): give IQ2_XS, IQ2_S and IQ3_S the
      descriptor read IQ2_XXS already has — the sub-block's index and
      sign bytes as whole dwords, all indices computed in registers,
      then the gathers issued together. Predicted: dot4 per drain 1.0
      → 8, and the three slowest rows up toward IQ2_XXS's 596 G/s.
      DONE 2026-09-16. The ISA half of the prediction landed exactly:
      `vmcnt(0)` 3/3/4 → 1/1/1, `v_dot4` 4 → 8 on all three, and every
      `global_load_u8`/`d16` gone; VGPRs 66/70/71 → 104/104/117 with no
      spill, which is values in flight rather than pressure. The
      throughput half landed PARTLY — at 14336×4096:

      | kernel | GB/s before → after | G values/s before → after |
      |---|---|---|
      | iq3s | 157 → 181 (+15.1%) | 392 → 451 |
      | iq2s | 133 → 149 (+11.5%) | 447 → 498 |
      | iq2xs | 123 → 141 (+14.6%) | 457 → 524 |
      | iq2xxs (control, untouched) | 143 → 144 (+1.0%) | 596 → 601 |
      | q4_k (control, untouched) | 207 → 206 (−0.2%) | 394 → 394 |

      Both controls held flat, so the 11–15% is the change and not
      drift. But none of the three reached IQ2_XXS's 596: one drain per
      body was necessary and is not sufficient, so a second cost
      remains — most likely `iqDot16`'s 32 ALU ops per 16 lanes
      building the weight vector, which IQ2_XXS pays too and which is
      now the tallest thing left. Exactness gate green throughout
      (229/229).
      A METHOD NOTE worth more than the 15%: the first attempt at this
      change reported "no ISA difference" twice. The edit had never
      been applied — its script aborted on a bad assertion, in a
      backgrounded call whose output nothing read — and the flat result
      was a faithful measurement of unchanged code. Then the edit
      landed and STILL changed nothing, because `iqU32` was itself four
      byte loads. Read the instrument's own output before believing a
      null result twice.
      SECOND VARIABLE: `iqDot16`'s vector build, then TQ1_0's 64
      half-word loads per body.
      PREDICTION for the second variable, written before the edit:
      `iqDot16` applies the sign one value at a time — per value a
      shift, a mask, a negate, an xor and an add, then a truncation and
      an insert into a 16-lane byte vector, about 130 VALU ops per
      call. Every grid byte is nonzero and at most 62 (checked across
      all five tables), so a per-byte two's complement never carries
      out of its own byte and the whole thing can run a dword at a
      time: `t = s | s<<7 | s<<14 | s<<21` puts sign bit j at bit 8j,
      one AND takes the low four and `(t>>4)` the high four, `q = p<<7`
      then `q | (q-p)` widens 0x01 to 0xFF per byte, and
      `(g ^ m) + p` negates four values at once. Four dwords then
      bitcast into the dot through `asBytes`, and `dotSum` replaces
      `dotAccum` plus its four extracts and three adds. About 38 VALU
      ops for the same 16 values.
      Expected: `global_load` and `vmcnt(0)` UNCHANGED (the memory
      shape is untouched), `v_dot4` unchanged, total VALU down roughly
      threefold, VGPRs down. The discriminating observation is
      IQ2_XXS: it was flat for the first variable because it already
      had the descriptor read, but it pays this cost in full, so if the
      cause is named right it moves this time. Q4_K stays flat either
      way.
      DONE 2026-09-16. Both halves landed. The ISA half exactly as
      written — `v_dot4` 8, `vmcnt(0)` 1 and every load count
      unchanged on all five, VALU down about 40% and VGPRs halved:

      | kernel | VALU before → after | VGPR before → after |
      |---|---|---|
      | iq2xxs | 288 → 168 | 100 → 48 |
      | iq2xs | 297 → 176 | 104 → 52 |
      | iq2s | 298 → 181 | 104 → 52 |
      | iq3xxs | 299 → 179 | 103 → 50 |
      | iq3s | 322 → 206 | 117 → 54 |

      And the throughput half, at 14336×4096:

      | kernel | GB/s before → after | G values/s before → after |
      |---|---|---|
      | iq2s | 149 → 183 | 498 → 613 (+22.9%) |
      | iq2xs | 141 → 162 | 524 → 604 (+15.2%) |
      | iq3xxs | 171 → 195 | 478 → 547 (+14.4%) |
      | iq3s | 181 → 202 | 451 → 505 (+11.9%) |
      | iq2xxs | 144 → 161 | 601 → 669 (+11.2%) |
      | q4_k (control) | 206 → 205 | 394 → 392 (−0.5%) |
      | tq2_0 (control) | 193 → 193 | 805 → 804 (−0.1%) |
      | tq1_0 (control) | 96 → 96 | 489 → 488 (−0.4%) |

      IQ2_XXS moved, which is what separates this result from a drift:
      it was the untouched control for the first variable and stayed
      flat there, and it pays this cost in full. Three controls flat,
      five treated kernels up 11–23%. Exactness gate green (229/229).
      End to end the gain arrives DAMPED: 11–23% in the kernel became
      3.6–6.6% in the model, where the first variable's 11–15% became
      14–17%. Solving Amdahl backwards from that says the kernels hold
      only a third of decode — and that inference is WRONG, which is
      why the profiler ran instead of a third edit. `CAJETA_PROFILER=1`
      with `CAJETA_PROFILER_GPU_RING=262144` (the default ring drops
      897 per mille on a decode of this length and the summary then
      silently covers only the tail) on llama8b-iq2_s, 128 tokens at
      depth 512:

      | decode kernel | ms/token | share |
      |---|---|---|
      | iq2xs wave (156 calls) | 10.00 | 60.1% |
      | iq3s wave (36) | 1.67 | 10.0% |
      | q5k wave — the output head (1) | 1.60 | 9.6% |
      | q4k wave (32) | 0.46 | 2.7% |
      | **weight mat-vec** | **13.73** | **82.5%** |
      | rmsnorm (65) | 0.69 | 4.1% |
      | attn decode reduce (32) | 0.65 | 3.9% |
      | attn decode gqa4 (32) | 0.50 | 3.0% |
      | q8k pack (129) | 0.44 | 2.6% |
      | glu, add, qkPrep (129) | 0.64 | 3.8% |
      | **everything else** | **2.91** | **17.5%** |

      2130 ms of device work over 128 tokens is 16.64 ms/token against
      16.9 measured, so decode is device-bound with about 1 ms/token of
      gaps across ~612 launches. The mat-vec is 82.5%, not 33%. The
      damping is composition, not Amdahl: an `iq2_s` file is mostly
      IQ2_XS tensors (+15.2%, not the +22.9% IQ2_S got), and 12.3% of
      its decode is q5_K and q4_K tensors this change never touched —
      the Q5_K output head alone is 9.6%, reading 361 MB per token at
      230 GB/s, already at the bandwidth ceiling.
      Weighting the probe's per-kernel gains by that census predicts
      +9.0%, and +6.5 to +8.6% arrived — see 7.3.1.
      The first leg run said +3.6 to +6.6% and was WRONG: a browser was
      playing video. On this APU that costs 1.9–2.8% of decode, and
      the tell was LOAD TIME, which is host I/O and cannot be touched
      by a kernel edit — it had risen 2–7% while prefill, whose
      kernels are byte-identical, fell 1.8–7.3%. Read load time on
      every leg; a move over ~1.5% is the box, not the code.
      THIRD VARIABLE, named from a clean disassembly: 64-bit address
      arithmetic. `KernelIsa`'s bundled disassembler is stale — it
      prints `.long` for VOP3 opcodes it cannot name AND a phantom
      `v_cndmask` from each one's second dword, so its totals are right
      by accident and its opcode names are not. Re-read with
      `llvm-objdump --mcpu=gfx1151`:

      | kernel | VALU | addr64 | quarter-rate | % of VALU ceiling |
      |---|---|---|---|---|
      | iq2xxs | 168 | 38 | 14 | 47% |
      | iq2xs | 176 | 40 | 17 | 45% |
      | iq2s | 181 | 40 | 17 | 47% |
      | iq3xxs | 179 | 41 | 16 | 41% |
      | iq3s | 206 | 43 | 16 | 44% |
      | tq10 | 1055 | 154 | 7 | 108% |
      | q4k | 871 | 73 | 26 | 14% |
      | tq20 | 170 | 31 | 5 | 19% |

      The ceiling column is wave-instructions per second against 40 CUs
      x 2 SIMD32 at 2.9 GHz. It sorts the set cleanly: TQ1_0 is over
      100%, so it is dual-issuing and genuinely instruction-bound —
      its 16.5 VALU per value is the whole story and the plan's other
      named target. Q4_K and TQ2_0 sit at 14–19% of VALU and
      193–205 GB/s: bandwidth-bound. The IQ family saturates NEITHER
      wall, 41–47% of VALU and 63–79% of bandwidth.
      What it does spend: 38–43 instructions per body on 64-bit byte
      offsets, of which 14–17 are quarter-rate `v_mul_lo_u32` /
      `v_mad_u64_u32` (see the standing note that RDNA integer multiply
      is quarter-rate). Counted at 4x that is about a third of the
      body's VALU cycles, and it produces no weight values. The
      indices are 32-bit and the bases are loop-invariant, so the
      multiplies belong outside the loop — reduce first, then scale.
      PREDICTED: addr64 down by two thirds, quarter-rate ops toward
      zero, VALU per value from 5.5 to about 4.3 on iq2xs, and at that
      count the IQ2 family reaches Q4_K's 205 GB/s, which is 764 G
      values/s and would clear 7.3.1 for every file. Loads, `vmcnt(0)`
      and `v_dot4` must not move. Q4_K and TQ2_0 stay flat.
      FOURTH VARIABLE — from READING llama.cpp rather than our own ISA,
      at Julian's direction, and it displaces the third as the next
      edit. Their Vulkan IQ2_XS mat-vec, decode path on RDNA:

      | | cajeta | llama.cpp |
      |---|---|---|
      | lanes per row | 32 | 16 |
      | rows per workgroup | 1 | 4, sharing one activation load |
      | sign bits | `ksigns[q>>9]`, an L1 gather | `q>>9` + `bitCount` |
      | codebook | L1 | LDS, staged per workgroup |
      | activations | packed to q8_K | read as f32 |
      | arithmetic | int8 `v_dot4` | f32/f16 `fma` |

      `ggml_vk_create_pipeline` passes `rm_iq = 2 * rm_kq = 4` rows and
      `wg_size_subgroup16` threads; `init_iq_shmem` stages the grid.
      THE ONE THAT MATTERS FIRST: **ksigns is not a table, it is
      arithmetic.** Checked against all 128 entries of `IqGrid.ksigns()`
      — `ksigns[s] == s | ((popcount(s) & 1) << 7)`, no exceptions. We
      gather it four times per body from L1; they compute it. Our body
      issues 14 loads per 32 values where Q4_K issues 26 per 320, and
      the kernel saturates neither the VALU ceiling (45%) nor DRAM
      (63%), so LOAD ISSUE is the limiter and four of the fourteen
      loads are arithmetic.
      Their decode is 13.69 ms/token against our 16.01. Holding
      everything outside this kernel equal, their IQ2_XS mat-vec runs
      8.54 ms against our 10.86 — 27% faster, where 16.7% clears the
      bar.
      CHANGE: `iqSign7(s) = s | ((Bits.count(s) & 1) << 7)`, replacing
      `ks[...]` in the IQ2_XXS, IQ2_XS and IQ3_XXS wave mat-vecs.
      PREDICTED: `global_load_b32` down by 4 per body, `vmcnt(0)`
      unchanged at 1 (the gathers were already in one drain), VALU
      roughly flat (a popcount, an and, a shift and an or replace the
      index scaling the gather needed), and 10–20% on those three
      kernels if load issue is really the limiter.
      THE CONTROL IS INTERNAL AND BETTER THAN Q4_K: IQ2_S and IQ3_S
      read raw sign bytes and never touch ksigns. They share `iqDot16`,
      the split layout, the lane mapping and the launch path, and
      differ only in this. They must not move. Q4_K and TQ2_0 as the
      outer controls.
      DONE 2026-09-16. ISA exactly as predicted: `global_load_b32`
      down by 4 on each of the three, `vmcnt(0)` still 1, `v_dot4`
      still 8, VALU up by exactly 8 with 4 `v_bcnt`. The two internal
      controls came back BYTE-IDENTICAL (iq2s 181 VALU, iq3s 206).

      | kernel | loads/body | G values/s |
      |---|---|---|
      | iq2xs | 14 -> 10 | 604 -> 666 (+10.3%) |
      | iq2xxs | 13 -> 9 | 669 -> 696 (+4.0%) |
      | iq3xxs | 18 -> 14 | 547 -> 555 (+1.5%) |
      | iq2s (internal control) | 12 | 613 -> 614 (+0.2%) |
      | iq3s (internal control) | 16 | 505 -> 507 (+0.4%) |
      | q4_k / tq2_0 / tq1_0 | - | +0.9 / -0.5 / +1.3% |

      Four loads traded for eight VALU ops bought 10.3%, which settles
      that LOAD ISSUE binds this kernel, not arithmetic. Exactness gate
      229/229 — the table identity holds for all 128 entries, so the
      kernels are bit-identical.
      The two small movers say where the rest is. IQ3_XXS is at 198
      GB/s against Q4_K's 207, so it is nearly at the bandwidth wall
      and has little left. IQ2_XXS at 167 GB/s is not, and it is one of
      the two kernels the FIRST variable never touched — it and
      IQ3_XXS still carry the old `while (pp < 2)` body with gathers
      inside the loop. Giving them variable 1's treatment is the cheap
      follow-on.
      REMAINING FROM THE llama.cpp READ, in order of expected size:
      (a) FOUR ROWS PER WORKGROUP. We run one row per 32-lane wave and
      re-read the whole activation row for every output row; they run
      four rows per workgroup and load it once. Activations are 4x our
      weight traffic at IQ2 rates, so this is the largest structural
      difference left.
      (b) The grid in LDS. Spec 12.1 decided against it at 32 lanes
      serving ONE row. At four rows per workgroup the amortization is
      4x better, and with ksigns gone the grid is the only table left,
      so 12.1 must be RE-DECIDED after (a), not before.
      (c) The 64-bit address arithmetic of the old third variable,
      still 38–41 instructions per body and untouched by this change.
      END TO END, quiet box, alternating arms, mean of three:

      | file | before | after | gain | vulkan | ratio | was |
      |---|---|---|---|---|---|---|
      | iq2_xs | 61.56 | 65.14 | +5.8% | 73.23 | 0.890 | 0.843 |
      | iq2_xxs | 66.44 | 69.48 | +4.6% | 78.14 | 0.889 | 0.850 |
      | iq2_s | 60.37 | 62.51 | +3.5% | 69.80 | 0.896 | 0.865 |
      | iq3_xxs | 56.44 | 56.26 | −0.3% | 58.56 | 0.961 | 0.939 |
      | iq3_s | 51.97 | 51.96 | −0.0% | 54.84 | 0.947 | 0.947 |

      `iq3_s` is the model-level control and it is textbook: an
      IQ3_S-dominated file whose kernel was not touched, −0.0% with
      the ratio unchanged to three places. `iq3_xxs` moved −0.3% while
      ITS Vulkan arm slowed 2.5% in the same slot, so read its 0.961 as
      "about at the bar, re-measure", not as a gain.
      A TRAP THAT COST A RUN: the first A/B after this change reported
      +0.3%. `tmp/cbq/u7-ab.sh` called `leg.sh`, which executes whatever
      `schedthroughput` is on disk, and that exe predated the commit by
      seventeen minutes. `stat -c %y` against the commit time found it.
      The script now builds first and prints the artifact's mtime. When
      the isolated probe moves and the model does not, check the
      binary's timestamp BEFORE theorising.
      WHERE THE IQ2 FAMILY STANDS: 0.889–0.896, needing about 6.7%
      more decode, which is ~10% more from the IQ2_XS kernel since it
      holds 68% of decode. Of the 10 loads now left per body, FIVE are
      grid gathers and TWO are activations. That makes (a) and (b)
      above one change rather than two, and it is what llama.cpp
      actually does: a wider workgroup covering four rows, activations
      loaded once for all four, and the grid staged in LDS where the
      stage is amortised over eight waves instead of one. Spec 12.1
      must be re-decided as part of it, not before it.
      FIFTH VARIABLE: rows per wave. DONE 2026-09-16, and the first
      prediction was WRONG in a way worth keeping.
      FOUR rows per wave, the number llama.cpp uses, made IQ2_XS 11.8%
      SLOWER — 179.5 → 158.4 GB/s, reproducible to 0.2% with every
      other kernel identical between the two binaries. The ISA was
      exactly as designed: the activation `global_load_b128` stayed at
      2 for four rows instead of becoming 8, loads per row fell 10 →
      7.75, VGPRs 52 → 100 with no spill.
      THE ERROR WAS READING THEIR ROW COUNT WITHOUT THEIR THREAD COUNT.
      llama.cpp puts 16 threads on a row in a 64-thread workgroup, so
      four rows arrive with four times the threads: they REDISTRIBUTE
      parallelism at `rows x 16` threads. Keeping our 32 lanes and
      quartering the workgroup count took us from `rows x 32` to
      `rows x 8`. Waves in flight fell 14336 → 3584 while loads fell
      only 22%, and this kernel hides gather latency by having many
      waves resident. TWO rows per wave lands on their thread count,
      `rows x 16`, and that is where the optimum is:

      | rows/wave | loads/row | VGPR | vmcnt(0) | GB/s |
      |---|---|---|---|---|
      | 1 | 10 | 52 | 1 | 179.5 |
      | 2 | 8.5 | 66 | 1 | 190.1 |
      | 4 | 7.75 | 100 | 3 | 158.4 |

      Applied to all five, the answer splits BY FORMAT:

      | kernel | before | after | |
      |---|---|---|---|
      | iq2xxs | 173.9 | 184.8 | +6.3% |
      | iq2xs | 179.1 | 189.1 | +5.6% |
      | iq2s | 182.9 | 188.6 | +3.1% |
      | iq3xxs | 199.5 | 199.7 | +0.1% (reverted to one row) |
      | iq3s | 202.9 | 203.2 | +0.1% (reverted to one row) |
      | q4_k / tq2_0 / tq1_0 | - | - | +0.4 / +0.8 / +0.2% |

      At two rows IQ3_XXS and IQ3_S measured −2.0% and −1.3%: both
      were already at 199–203 GB/s against Q4_K's 206, so fewer loads
      buy a kernel at the byte ceiling nothing while the lost
      parallelism still costs. They keep one row per wave. This is a
      per-format launch policy, the same shape as Q4_K's GEMM.
      Exactness gate 229/229 at every step.
      12.1 (L1 vs LDS) STILL STANDS UNCHANGED: the LDS case needed a
      wider workgroup to amortise the stage, and the measurement above
      says a wider workgroup is not affordable here — parallelism is
      worth more than the loads it would save. Re-testing LDS would now
      have to come with a bigger BLOCK, not more rows per wave.
      END TO END, five reps each side, arm order flipped from the
      three-rep round (which had the Vulkan arm bouncing 3% between
      rounds on unchanged code — read ratios from the five-rep run):

      | file | cajeta | vulkan | ratio |
      |---|---|---|---|
      | iq3_s | 52.20 | 54.88 | 0.951 MEETS |
      | iq3_xxs | 56.96 | 60.15 | 0.947 |
      | iq2_s | 64.24 | 69.35 | 0.926 |
      | iq2_xxs | 70.68 | 78.22 | 0.904 |
      | iq2_xs | 65.96 | 73.06 | 0.903 |

      (Superseded by the pack-fusion rows below: iq2_xs reaches 68.08
      and 0.931.) Unit 7 has taken iq2_xs from 48.7 t/s and 0.66 to
      68.08 and 0.931, +40%.
      WHAT REMAINS IS THE IQ2 FAMILY, and the target is exact. Every
      kernel on this box tops out near 206 GB/s — q4_k 207, iq3s 203,
      tq2_0 193 — against a ~256 GB/s LPDDR5X peak, so ~206 is the
      practical streaming ceiling and IQ3_S clearing the bar is a
      consequence of reaching it. iq2xs is at 189 and iq2xxs at 185.
      +8.8% on iq2xs is 0.75 ms/token off a kernel holding ~65% of
      decode, which is exactly the 4.9% that turns 0.903 into 0.95:
      "become bandwidth-bound like the others" and "clear 7.3.1" are
      the same statement. Re-profile and re-read the ISA before
      choosing the next variable — two changes have landed since the
      last read and the binding constraint has moved each time.
      SECOND CANDIDATE, worth raising because it is now the same size
      as the gap: 17% of decode is not a mat-vec, and `q8kPackKernel`
      is 0.44 ms/token of it (129 launches at 3.2 us). llama.cpp pays
      none of it — it reads f32 activations. The count is already
      minimal (four per layer, one per distinct activation vector, plus
      the head; the "we pack once per linear" hypothesis is REFUTED, it
      would be 225), so it can only be FUSED into the kernel that
      produces each vector — the norm, the attention output, the GLU.
      That is ~2.8% on every format, not just IQ2.
      IQ2 RE-PROFILE, 2026-09-16, quiet box, on llama8b-iq2_xs after
      both changes. Decode is 14.62 ms/token of device work (was
      16.01): iq2xs wave 9.69 (66.3%), the Q5_K head 1.58 (10.8%),
      q4k+q2k 0.80, everything else 2.55 (17.4%) — still 82.6%
      mat-vec. −0.75 ms/token clears the bar, which is +7.7% on the
      iq2xs kernel, 189 → 204 GB/s.
      WHAT BINDS IT IS THE GATHER WIDTH, and the ISA sorts the whole
      set on one column:

      | kernel | `global_load_b64` per body | GB/s |
      |---|---|---|
      | iq2xs | 10 | 189 |
      | iq2xxs | 10 | 185 |
      | iq2s | 8 | 189 |
      | iq3xxs | 1 | 200 |
      | iq3s | 1 | 203 |
      | q4k | 0 | 207 |

      The three formats below the ceiling are exactly the three whose
      codebook entry is EIGHT bytes, so every lookup is a divergent
      `global_load_b64` with 32 unrelated lane addresses. IQ3's entry
      is four bytes and gathers as `b32`; Q4_K's weights are contiguous
      `b128`. Nothing else binds: DRAM is at 74% of peak, load ISSUE at
      2.3% of the ceiling, VALU at 47%. Loads per value REFUTES itself
      as the explanation — iq3s has the most at 0.50 and sits at the
      ceiling, iq2xs has 0.266 and does not.
      SIXTH VARIABLE, ready to build: `IqGrid.iq2xxs2b()`,
      `iq2xs2b()` and `iq2s2b()` already hold one `int32` per grid
      entry — eight 2-bit codes over the magnitude set {8, 25, 43}
      (verified against the int64 tables). Gathering those makes every
      IQ2 lookup a `b32`, halves the codebook's footprint, and turns
      the expansion into a 4-entry LUT, which is the `Vector.lut4`
      pattern that beat llama.cpp on MXFP4. PREDICTED: `b64` per body
      10 → 0, `b32` up by the same count, byte rate toward 204, and
      IQ3/Q4_K/TQ flat.
      PACK FUSION, 2026-09-16, at Julian's direction after the
      re-profile. THE FINDING IS NOT THAT FUSION WAS MISSING — it was
      built and unreachable. Three fused fast paths exist and none ran
      on llama-3-8B:
      - `Prim.rmsnormPackDevice` (Unit 48) declined `dim > 2048`; the
        model is 4096 wide. The cap was structural — one workgroup of
        256 threads, eight waves, ONE 256-element pack block per wave.
        A wave now takes block `wid`, then `wid + 8`.
      - the dense route's post-attention norm only tried the fused path
        when a model had EXPERTS; it now takes it like the
        attention-input norm.
      - `attendDecodePartialsLaunchNoSync` (Unit 50, reduce + pack in
        one launch) allow-lists `nH == nKv` and `nH == 8 * nKv`.
        Llama-3-8B is GQA-4. STILL UNREACHED — the remaining item.
      Extending the norm kernel forced a numerics decision: its Unit 54
      per-wave reduction cannot match `rmsnormRowF32`'s 256-element LDS
      tree bit for bit, because the tree's first three levels pair
      lanes 128, 64 and 32 apart, which cross waves. Since the fused
      kernel REPLACES the pair whenever the width allows, a last-bit
      difference would make the normed row depend on which path the
      width selected — the 4096 test caught exactly that, at 2048 and
      512 it had matched by luck. The tree is back; eight barriers
      instead of three, and the kernel is still half the cost of the
      pair it replaces.
      Per token: `rmsnormRowF32` 65 calls → 1, `q8kPackKernel` 129 →
      65.5, `rmsnormPackRowF32` 64 at 5.58 us where the pair cost
      12.20 — it beats even the norm alone, because it keeps the
      normed values in registers instead of re-reading the row twice.
      Decode 14.62 → 14.21 ms/token. End to end, five reps, arms
      alternating:

      | file | before | after | ratio | was |
      |---|---|---|---|---|
      | iq3_s | 52.20 | 52.97 | 0.970 MEETS | 0.951 |
      | iq3_xxs | 56.96 | 57.92 | 0.968 MEETS | 0.947 |
      | iq2_s | 64.24 | 65.77 | 0.944 | 0.926 |
      | iq2_xxs | 70.68 | 72.58 | 0.925 | 0.904 |
      | iq2_xs | 65.96 | 67.70 | 0.925 | 0.903 |

      BOTH IQ3 FILES NOW CLEAR THE BAR. `gluPackF32` then took the last
      of the four packs a dense layer pays; unlike the norm it needs no
      cross-block reduction, only the per-block max, so it keeps the
      many-workgroup shape.
      With the GLU fused too:

      | file | before | after | ratio | was |
      |---|---|---|---|---|
      | iq3_s | 52.97 | 53.68 | 0.980 MEETS | 0.970 |
      | iq3_xxs | 57.92 | 58.66 | 0.973 MEETS | 0.968 |
      | iq2_s | 65.77 | 66.56 | 0.956 MEETS | 0.944 |
      | iq2_xxs | 72.58 | 73.47 | 0.940 | 0.925 |
      | iq2_xs | 67.70 | 68.08 | 0.931 | 0.925 |

      THREE OF FIVE NOW MEET 7.3.1. Only the two thinnest IQ2 formats
      remain, at 0.931 and 0.940, and the sixth variable (the `b32`
      codebook gather) is aimed exactly at them.
      A dense layer now pays TWO packs, not four: the attention output
      and — on the sub-norm shapes only — the GLU. Fusing the
      attention one needs GQA-4 in Unit 50's reduce+pack allow-list.
      THE PRICE OF BIT-EXACTNESS, measured rather than assumed. A
      throwaway variant of the fused kernel carrying Unit 54's per-wave
      reduction, alternated against the shipping tree, min of five over
      500 launches:

      | dim | tree | wave | cost |
      |---|---|---|---|
      | 2048 | 3.76 us | 3.45 us | +9.1% |
      | 4096 | 5.19 us | 4.92 us | +5.5% |
      | 14336 | 12.60 us | 12.27 us | +2.7% |

      A flat ~0.3 us per launch whatever the width — the barriers, not
      the data. At 64 fused norms a token that is 17 us on a 14 ms
      token, 0.12% of decode, below the A/B's noise floor. Unit 54's
      optimisation was worth a tenth of a percent here and interchange-
      ability costs exactly that, so the tree stays. The third option
      — give `rmsnormRowF32` the wave form too, so both shift together
      — would recover 0.12% while touching a kernel every model runs,
      and is not worth it. The variant and its bench were deleted.
      SIXTH VARIABLE MEASURED AND REVERTED, 2026-09-16. The two-bit
      codebook was built exactly as predicted and is FLAT, so the
      gather-width reading of the table above is REFUTED. The three
      IQ2 kernels gathered `IqGrid.iq2xxs2b/iq2xs2b/iq2s2b` (one
      `int32` per entry, verified byte-for-byte against the int64
      tables), spread the eight two-bit codes to byte lanes, OR'd the
      sign bit in at bit 2 and took magnitude AND sign from one
      `Vector.lut4` over {8, 25, 43, 0, -8, -25, -43, 0}. `b64` per
      body went 10 -> 0 and the codebook halved, exactly as written
      down. Same box, `iq3xxs`/`iq3s`/`q4_k` as in-run controls
      (198.9 / 202.6 / 205.8 against the reference round's 199.7 /
      203.2 / 207, so the box is 0.3-0.6% slower and the arms are
      comparable):

      | kernel | 8-byte grid | 2-bit grid | control-adjusted |
      |---|---|---|---|
      | iq2xxs | 184.8 | 183.7 | -0.2% |
      | iq2xs | 189.1 | 187.5 | -0.5% |
      | iq2s | 188.6 | 190.5 | +1.4% |

      A COMPILER FIX CAME OUT OF IT AND STAYS. `Vector.lut4` lowered to
      THREE `v_perm_b32` per dword unconditionally — one per table
      half plus a select off the index's bit 3 — even when the index
      cannot reach the high half. `AmdgpuKernelLowering::byteLut16` now
      peels the receiver's slot load and bitcasts, and when the index
      is visibly `& <constant with bit 3 clear in every byte>` it emits
      the low perm alone: 48 perms per body -> 16, VGPR 60. LLVM will
      not do this for us — the selector DOES fold to the constant
      `0x03020100` under `-O2`, but nothing simplifies
      `llvm.amdgcn.perm(a, b, 0x03020100)` to `b` (checked directly
      against llvm-22). Worth 0 to 3% here and inert for `& 15` users
      (mxfp4, iq4_nl), which is every other caller today.
      THEN THE THREE-PROBE SWEEP THAT SAID WHERE THE TIME IS. Each
      probe keeps the loop shape, the load count and the launch
      geometry and varies ONE mechanism, at 14336x4096:

      | probe | iq2xxs | iq2xs | iq2s |
      |---|---|---|---|
      | shipping (2-bit) | 183.7 | 187.5 | 190.5 |
      | every gather masked to 8 entries | 181.2 | 187.7 | 190.4 |
      | all decode arithmetic deleted | 190.5 | 195.1 | 195.3 |
      | four rows per wave (iq2xs only) | - | 149.2 | - |

      The codebook gather COSTS NOTHING: collapse all 32 lanes onto one
      cache line and the number does not move. Deleting every sign,
      spread, permute and expansion — leaving the loads, the four
      `v_dot4` and the scales — buys 2.5-4.1%. And four rows per wave
      reproduces its earlier regression (-20%) now that the codebook is
      half the size, so it is register pressure, not the table.
      WHAT IS LEFT IS A FIXED PER-VALUE COST, and it fits every format
      on the box with two constants. Let w be weight bytes per value;
      time per value is `w/229 + 0.000267` (us-scale), so the reported
      weight-GB/s is `w / (w/229 + b)`:

      | kernel | w | predicted | measured |
      |---|---|---|---|
      | q4_k | 0.563 | 207 (fit) | 206-207 |
      | iq3s | 0.410 | 200 | 202.6 |
      | iq3xxs | 0.365 | 196 | 198.9 |
      | iq2s | 0.305 | 191 | 190.5 |
      | iq2xs | 0.289 | 189 | 187.5 |
      | iq2xxs | 0.246 | 184 (fit) | 183.7 |

      THE IQ2 KERNELS ARE ALREADY AT THE SAME MACHINE LIMIT AS Q4_K.
      Their lower weight-GB/s is arithmetic, not slack: the same fixed
      cost divided by fewer weight bytes. 1/229 GB/s is the streaming
      rate and `b` is the per-value overhead every format pays — of
      which the measured arithmetic is 0.000061, about a quarter. A
      format at 0.289 bytes per value cannot reach 207 while `b` is
      that size, whatever the kernel does with the codebook.
      SEVENTH VARIABLE, and the one that paid: GQA x4 in unit 50's
      fused reduce+pack allow-list — the last unfused pack per dense
      layer, named as open in the pack-fusion record above.
      `attendDecodePartialsLaunchNoSync` served `nH == nKv` and
      `nH == 8 * nKv`; llama-3-8B is `nH == 4 * nKv`, so every dense
      layer paid a separate `attnFlashDecodeReduceKernel` AND a
      separate `q8kPackKernel`. Both GQA kernels already write `part`
      indexed by q-head — the comment on the x8 kernel says so — and
      `attnReducePackQ8Kernel` reads it that way, so the fused form
      needed only the ratio in the gate and the x4 kernel in the
      branch. The test oracle `attendDecodePairLaunchNoSync` picks the
      same kernel by ratio, so the new case isolates the FUSION.
      Profiled on llama8b-iq2_xs, 128 tokens:

      | kernel | before | after |
      |---|---|---|
      | attnFlashDecodeReduceKernel | 4096 x 14.58 us = 59.7 ms | gone |
      | q8kPackKernel | 4288 x 3.54 us = 15.2 ms | 192 x 20.0 us = 3.8 ms |
      | attnReducePackQ8Kernel | - | 4096 x 6.10 us = 25.0 ms |

      46 ms of 1.79 s and 8192 fewer launches a run; decode 15.288 ->
      14.852 ms/token under the profiler, -2.9%. The reduce was the
      one non-mat-vec kernel whose cost was latency rather than work:
      14.58 us to combine ~131 KB of partials is 9 GB/s.
      THE DECODE CENSUS AFTER IT, same run, device time per token:
      iq2xs wave 9.69 ms (71%), the Q5_K head 1.55 (11.4%), attention
      0.67 (4.9%), q4k+q2k 0.81 (6.0%), norm+pack 0.36, add 0.24,
      glu+pack 0.12, qkPrep 0.09, rope 0.04. Weight mat-vec is 88% and
      the model above says that 88% is at the machine limit, so what
      remains to win is the 12% and it is already down to launch-sized
      pieces.
      EIGHTH VARIABLE, the same shape of finding once more: the
      accumulate form existed and could not be reached. Unit 48 built
      `Linear.matvecStagedAccum` so "the residual add rides the
      o-projection's store", and it served the Q4_K and Q6_K wave
      routes only — an IQ model fell through to a separate `addF32`
      launch. The MLP tail never even asked: its residual was an
      unconditional `Prim.addDevice` after `matvecStagedKeep`. So a
      dense layer paid TWO whole launches of a 5-VGPR kernel, 65 a
      token, to add 4096 floats twice.
      The five IQ wave kernels take the `accum` flag the Q4_K and Q6_K
      kernels already carried, `matvecStagedAccum` serves `waveIq`, and
      the down projection asks for it. `addF32` 8320 launches -> 640,
      30.2 ms -> 11.7 ms over 128 tokens; decode 14.852 -> 14.709
      ms/token under the profiler. The greedy stream is byte-identical
      across both variables — the add is the same f32 add in the same
      order, only the launch is gone.
- [x] 7.2.3 `@Occupancy(maxThreads)` wherever a launch block is not a
      literal — an unpinned block is budgeted for 1024 threads and caps
      VGPRs at 192, which is a despill the ISA read will show.
      AUDITED AND NOT NEEDED, which the ISA read settles rather than
      assumes. 27 launch sites pass a non-literal block; only an integer
      LITERAL pins `amdgpu-flat-work-group-size` (`Compiler.cpp`'s
      `constBlockThreads` takes `IntegerLiteralExpression` alone, and a
      non-constant dim at ANY site erases the kernel from the map), and
      `@Occupancy(maxThreads)` overrides it where present. Of those
      sites exactly ONE reaches a live decode kernel —
      `q2kQ8WaveMatVecKernel` through `waveRowBlock()`, 4 launches a
      token on this file. It builds at vgpr 68, sgpr 21, no spill and
      no scratch, so the 192 cap is nowhere near it and pinning would
      change nothing. Across all 240 kernels in the exe NOT ONE spills
      or sits at 192; the only scratch in the tree is the stdlib's
      portable `CooperativeMatrix` tile (`matmulBf16/F32/F64`), which
      the compiler already names as the tile rather than pressure. The
      rest of the non-literal sites are the one-item-per-row f32
      mat-vec family and the WMMA prefill paths, none of them on this
      unit's route. Pinning a kernel that does not want the registers
      buys nothing, so nothing was pinned: recorded as measured.

### 7.3 Acceptance
- [x] 7.3.1 Decode ≥ 0.95× the better llama.cpp backend on the eight
      codebook files and both TQ files; prefill not regressed.
      AFTER 7.2.2's first variable (tg128 at depth 512, same box):

      | file | before | after | gain | GB/s | × vulkan |
      |---|---|---|---|---|---|
      | iq2_xs | 48.7 | 56.8 | +16.6% | 138 | 0.77 |
      | iq2_s | 47.8 | 55.7 | +16.5% | 141 | 0.80 |
      | iq3_s | 42.8 | 48.9 | +14.3% | 169 | 0.89 |
      | iq3_xxs | 50.7 | 52.3 | +3.2% | 159 | 0.88 |
      | iq2_xxs (control) | 61.7 | 61.5 | −0.3% | 136 | 0.78 |

      The kernel gain reaches the model nearly one to one — 11–15% in
      the probe, 14–17% end to end — which says the decode path around
      these kernels adds no overhead worth hunting. The two controls
      behave exactly as the file census predicts: `iq2_xxs`, whose
      kernel was untouched, does not move; `iq3_xxs`, whose file also
      carries 64 IQ2_S and 33 IQ3_S tensors, gains a partial 3.2%.
      NOT MET: 0.77–0.89× against the 0.95 bar, and prefill is
      unchanged. The gap is now ~1.2× rather than ~1.5×.
      AFTER 7.2.2's second variable. Measured on a QUIET box against
      llama.cpp Vulkan (`-fa`) in ONE window with the arm order
      alternating by file, mean of three reps on both sides:

      | file | before | after | gain | vulkan | ratio |
      |---|---|---|---|---|---|
      | iq2_xxs | 61.44 | 66.65 | +8.5% | 78.19 | 0.850 |
      | iq2_xs | 56.77 | 61.61 | +8.5% | 73.06 | 0.843 |
      | iq2_s | 55.65 | 60.46 | +8.6% | 69.78 | 0.865 |
      | iq3_xxs | 52.28 | 56.44 | +8.0% | 60.09 | 0.939 |
      | iq3_s | 48.82 | 51.98 | +6.5% | 54.88 | 0.947 |

      Two quiet runs of the same binary agree to 0.6%, and tonight's
      Vulkan numbers land within 1% of the reference table above, so
      the older reference was sound. Prefill is untouched by
      construction — the coop GEMMs never call `iqDot16`.
      STILL NOT MET: 0.843–0.947 against 0.95, but iq3_s misses by
      three tenths of a point and iq3_xxs by one, against 0.66–0.85
      at the start of the unit. The IQ2 family is the gap now, 8–16
      points short, and it is the family with bandwidth headroom left
      (161–183 GB/s against Q4_K's 205 on the same box).
      MET, on all five files, after 7.2.2's seventh and eighth
      variables (the GQA x4 reduce+pack fusion and the accumulate form
      on both residual adds). Five reps a side, arm order alternating
      by file, one window, quiet box:

      | file | cajeta | vulkan | ratio | was |
      |---|---|---|---|---|
      | iq3_s | 55.03 | 55.12 | 0.998 MEETS | 0.980 |
      | iq3_xxs | 60.33 | 60.52 | 0.997 MEETS | 0.973 |
      | iq2_s | 68.38 | 70.36 | 0.972 MEETS | 0.956 |
      | iq2_xs | 70.27 | 73.65 | 0.954 MEETS | 0.931 |
      | iq2_xxs | 75.49 | 79.26 | 0.952 MEETS | 0.940 |

      PREFILL IS NOT REGRESSED, measured in the same window (pp512,
      `-fa`): 1279.7 / 1240.2 / 1235.4 / 1202.3 / 1036.6 against
      Vulkan's 1195.1 / 1177.6 / 1171.3 / 1211.8 / 1252.9, so
      1.07 / 1.05 / 1.06 / 0.99 / 0.83. The IQ2 files keep the
      1.04-1.11x they had; the last row is IQ3_S's coop GEMM, which is
      6.4.2 and untouched by this unit -- none of the eight variables
      edited a coop kernel.
      WHERE THE UNIT STARTED: 0.66-0.85x, 0 of 5 over the bar. Where it
      ends: 0.952-0.998, 5 of 5. iq2_xs went 48.7 -> 70.27 t/s, +44%.
      THE OTHER FIVE FILES the item names, measured the same way the
      next evening. The mixes carry the same five kernels and follow;
      for the ternary pair the better llama.cpp backend is the CPU, not
      Vulkan (Vulkan runs TQ at 70 t/s, HIP at 117, the CPU at 168-173):

      | file | cajeta | llama.cpp | backend | ratio |
      |---|---|---|---|---|
      | iq2_m | 64.78 | 67.40 | vulkan | 0.961 MEETS |
      | iq3_xs | 56.97 | 56.55 | vulkan | 1.007 MEETS |
      | iq3_m | 53.97 | 54.25 | vulkan | 0.995 MEETS |
      | tq1_0 | 169.74 | 168.38 | cpu | 1.008 MEETS |
      | tq2_0 | 217.64 | 172.61 | cpu | 1.261 MEETS |

      TEN OF TEN. The item is met on every file it names.
- [x] 7.3.2 Perplexity and the Q8 twins unchanged on the files of 5.3.1
      and 6.3.1 — the numbers this unit may not move.
      THE TWINS ARE UNCHANGED: every Q8-twin, coop and fixture test in
      `TernaryTest` and `IqCodebookTest` passes, filtered suite 253/253
      (7.1.3).
      PERPLEXITY MOVED, BY BETWEEN -0.27% AND +0.37%, AND THE CAUSE IS
      NAMED AND DEMONSTRATED. `PplProbe` prefills 1025 tokens and then
      DECODES 1023 one at a time, so it measures the path this unit
      rewrote; every number below is a decode number, not a prefill one.

      | file | recorded | now | delta | llama.cpp | now vs lc |
      |---|---|---|---|---|---|
      | iq2_xxs | 7.4942 | 7.50027 | +0.08% | 7.4910 | +0.12% |
      | iq2_xs | 6.5934 | 6.59635 | +0.04% | 6.6042 | -0.12% |
      | iq3_xxs | 5.3515 | 5.34233 | -0.17% | 5.3598 | -0.33% |
      | iq2_s | 6.2280 | 6.22051 | -0.12% | 6.2346 | -0.23% |
      | iq2_m | 5.7347 | 5.75566 | +0.37% | 5.7365 | +0.33% |
      | iq3_s | 5.1634 | 5.14954 | -0.27% | 5.1717 | -0.43% |
      | iq3_xs | 5.1719 | 5.17733 | +0.11% | 5.1743 | +0.06% |
      | iq3_m | 5.1521 | 5.14281 | -0.18% | 5.1721 | -0.57% |
      | qwen-moe | 5.4687 | 5.46277 | -0.11% | 5.4781 | -0.28% |

      THE CONTROL, and it is exact. `PplProbe` gained a `noreducepack`
      arm that calls `AttnKernel.setReducePack(false)`, so the decode
      attend takes the separate reduce launch again and the split
      partials are summed in the serial order the fused kernel
      replaced. Same binary, same corpus:

      | file | fused | noreducepack | recorded |
      |---|---|---|---|
      | iq3_s | 5.14954 | 5.16341 | 5.1634 |
      | iq2_m | 5.75566 | 5.74079 | 5.7347 |

      iq3_s returns to its recorded value to five figures. So the mover
      is `attnReducePackQ8Kernel`'s summation order: it gives each of
      eight waves a quarter of the splits and meets them in LDS, where
      the kernel it replaced walked every split serially in one wave.
      Float addition is not associative, the difference is in the last
      bits, and the KV cache carries it forward, which is why a 1e-7
      difference reads as 0.3% after 1023 steps. iq2_m keeps +0.11%
      with the fusion off, from the norm+pack and GLU+pack fusions
      earlier in this unit, inside the same floor.
      THIS IS NOT A NEW TOLERANCE, it is an existing one reaching one
      more shape. Unit 50/57 built `attnReducePackQ8Kernel` knowing it
      could not match the serial reduce bit for bit — its own test
      (`Gqa8FlashDecodeTest.reducePackMatchesReduceThenPack`) sets the
      bar at the dequantized value within one quantum for exactly this
      reason — and shipped it for the GQA x1 and x8 shapes. Unit 7 put
      GQA x4 in the same allow-list; llama-3-8B is the model that
      changed. Every number above stays inside the +/-0.5% routing-flip
      floor 6.3.2 documents, and the spread against llama.cpp (-0.57%
      to +0.33%) is the same size as the one already recorded (-0.39%
      to +0.04%).
      ONE THING THE MOVE DID SETTLE. 6.3.1's watch item -- "all seven
      perplexities sit BELOW llama.cpp's, between 0.03% and 0.39%, a
      consistent sign, not scatter" -- does not survive a change of
      summation order: six of nine are below now and three above. The
      consistent sign was an artifact of one reduction order, not a
      quality edge, and it should not be read as one.
      GREEDY, 16 tokens at temp 0, against llama.cpp on the same
      prompt: iq3_s still identical token for token; iq2_m still tips
      early, as recorded; iq2_xs now tips at about token 6 where it
      matched before. Argmax on a near-tie is the most sensitive thing
      in the engine ([[moe-ppl-routing-flip-floor]]) and it is the
      first thing a last-bit change shows.
      THE PRICE OF THE ALTERNATIVE, for the record: keeping the serial
      reduce costs the launch this unit removed -- 4096 launches at
      14.58 us over 128 tokens, 3.3% of decode, and iq2_xs drops from
      0.954 back under the bar. JULIAN'S CALL if he wants the older
      numbers back; the arm to flip is `AttnKernel.setReducePack`.
- [x] 7.3.3 Legs re-run and recorded (announced).
      DONE: the ten-file table under 7.3.1, the ternary pair against all
      three llama.cpp backends, and 4.3.5's embedding-controlled bitnet
      round. Box gated on `/proc/loadavg` and `gpu_busy_percent` before
      each, arms alternating by file, and load time read on every leg as
      the box witness ([[load-time-is-the-box-witness]]) — it sits
      within 1% across the two rounds on every 8B file.
- [x] 7.3.4 If a gap survives with a measured, named cause that is not
      ours (a hardware or compiler limit), it is recorded here with its
      number and the unit closes on that — an explained gap is a
      result, an unexplained one is not.
      NO GAP SURVIVES ON THIS UNIT'S BAR: ten of ten files clear 7.3.1.
      Two limits were measured on the way and both are recorded above
      rather than worked around:
      1. `w/229 + 0.000267` per value fits every format on the box, so
         the IQ2 kernels' lower weight-GB/s is a fixed per-value cost
         divided by fewer weight bytes, not slack. Three probes closed
         the alternatives: the codebook gather costs nothing, all the
         decode arithmetic is worth 2.5-4.1%, four rows a wave costs
         20%. THAT is why the last 5% came from launches, not bytes.
      2. TQ1_0 stays VALU-bound at 16.5 VALU per value, 108% of the
         single-issue ceiling (4.3.5). It still clears 7.3.1 at 1.008x
         because the llama.cpp ceiling it is measured against is the
         CPU, which is bound by the same LPDDR5X pool (7.1.5).
      ONE NAMED GAP CARRIES FORWARD and is NOT this unit's: every
      flash-decode gate requires `hd == 128`, so bitnet-large (16 heads,
      head dim 96) takes the scalar `attnScore` + `attnCombine` pair for
      31% of its decode token. It is the same shape as the two fusions
      of 7.2.2 — a fast path the model cannot reach — and it is the
      largest single item left on any model in this plan. Recorded under
      4.3.5; it belongs to whichever unit generalises the flash kernels
      off head dim 128.

## Unit 8 — IQ1_S, IQ1_M (spec §3.2, §6.3)

### 8.1 TDD
- [x] 8.1.1 Decoders exact: IQ1_M's f16 rebuilt from four nibbles, the qh
      nibble split, the delta signs.
      DONE 2026-09-17, and both passed against ggml's own `.f32`
      fixtures on the first run.
      THE DELTA NEEDS NO TERM OF ITS OWN, which is simpler than this
      unit assumed. Spec 6.3 expected `+/-0.125` to ride the Q8 pack's
      per-32 sums (IQ1_S) and a per-8 sum (IQ1_M), because a delta is
      not an integer and the IQ family decodes through an INTEGER path
      (`iqInts` fills 256 signed `t` and the per-sub-block multipliers,
      then `d * ls8[k/step] * t[k]`). But the iq1s grid's values are
      -1/0/+1 and the delta is exactly an eighth, so `8*g +/- 1` IS an
      integer — it lands in {-9,-7,-1,1,7,9} — and the whole family
      keeps the integer path at a 0.125 scale, which `iqScale` already
      returns by default. `Quant.iq1Group8` writes `8*v + 1` or
      `8*v - 1` and nothing else changes: host mat-vec, Q8 twin and
      block decode all inherit it. The device kernels will too — the
      values fit int8 for `dp4a` and a 256-value block sums to at most
      292k, well inside int32 — so the bsums trick is not needed
      anywhere.
      TWO SMALLER THINGS the spec names and the decoders confirm:
      IQ1_M has NO leading f16 (`Quant.iqBlockScale` rebuilds it from
      the high nibble of each of four scale words, and every caller of
      the old `f16At(raw, ro)` now goes through it), and its `qh`
      carries the index's high three bits and the delta sign a NIBBLE at
      a time where IQ1_S packs multiplier, sign and three index triples
      into one `qh` word per 32.
      GATES HELD, per 8.2.1's "gates last". Adding IQ1 to
      `Quant.supported` before the kernels exist fails
      `TernaryTest.theFourPartInvariant`, which demands supported ==
      hasKernel == coopSupports == packedSupported — exactly as it
      should. `Quant.decodable` carries the decode-only state so
      `blockElems` still answers for IQ1, and `supported` stays false
      until 8.1.3 lands.
- [~] 8.1.2 Host mat-vecs and Q8 twins with the delta term through the
      pack's per-32 sums (IQ1_S) and a per-8 sum (IQ1_M).
      HOST HALF DONE: `theIq1HostMatVecsMatchTheDequantized` runs both
      formats' `iqMatVecIntoAt` against their own dequantized weights.
      `iqMatVecIntoQ8` (the twin) inherits the same `iqInts` and needs
      no IQ1 code either. The twin is only CHECKED against the wave
      kernel (`IqCodebookTest.checkQ8` launches it), so that half waits
      on 8.1.3. Per 8.1.1, the delta term the item describes does not
      exist — it is folded into the integer value.
- [ ] 8.1.3 Wave decode kernels and coop X1/X3; `coopBlockWords` 12 / 14;
      `scaleBytes(IQ1_M) == 0`.
- [ ] 8.1.4 12.1 re-measured on the iq1s table (16 KB bytes, 8 KB nibbles).

### 8.2 Coding
- [ ] 8.2.1 Decoders; host; kernels; registration; gates last.

### 8.3 Acceptance
- [ ] 8.3.1 IQ1_S and IQ1_M 8B files: legs, greedy agreement, perplexity,
      resident bytes.

## Unit 9 — Migrate Q4_0, Q5_0, Q3_K, Q6_K, IQ4_NL; delete the repack machinery (spec §10.2–10.3, 12.3)

### 9.1 TDD
- [ ] 9.1.1 Per format, the existing kernel tests re-pointed to
      `(payload, scales)`: decode, wave f32, coop X1/X3, the widen and
      Mw8 kernels of Q3_K and Q6_K, IQ4_NL's six.
- [ ] 9.1.2 `ResidentLayoutTest` covers every supported type with
      `splitOn` gone; `devWBytes == payload + scales` for each.
- [ ] 9.1.3 A test that the removed names are gone from the tree.

### 9.2 Coding
- [~] 9.2.1 The 42 kernels re-offset; `coopNeedsRepack`,
      `blockRepack2Kernel`, `blockPadKernel`, `ensureQ6Pad`, `coopDev`
      and `splitOn` removed; the split unconditional; `coopBlockWords`
      the payload stride everywhere.
      PULLED FORWARD 2026-09-17 at Julian's direction, because 6.4.3 is
      blocked on it. THREE OF FIVE FORMATS MIGRATED: IQ4_NL, Q4_0, Q5_0.
      `coopNeedsRepack` is down to Q3_K and Q6_K.
      IQ4_NL FIRST, because it is 6.4.3's blocker: the Qwen MoE's 1420
      `blockRepack2Kernel` launches were its IQ4_NL experts. Its three
      coop kernels and its item-per-row mat-vec now read the split
      layout (`rowWords = pw + bpr*4`, the scale from the prefix word,
      payload at `ro` with no `+1` word) and `coopBlockWords` is the
      payload stride, 4. MoE PREFILL 657 -> 895 t/s, and
      `blockRepack2Kernel` is gone from the profile entirely.
      Q4_0 AND Q5_0 followed, same shape: six kernels each (Q8 wave, f32
      wave, item-per-row, coop X1/X3, widen), `coopBlockWords` 5 -> 4 and
      6 -> 5. The host side needed nothing — `Linear.runScaleImage`
      already branches on `splitOn`, so the scale image follows the
      format in automatically.
      COVERAGE, honestly: the Q8-wave and f32 routes of all three are
      gated by `LegacyWaveMatVecTest` and `CoopQuantGemmTest` against
      real fixtures, and that gate FIRED on IQ4_NL (a wrong dispatch
      order and three stale staging sites, both caught). The WIDEN route
      (`q40WidenKernel`/`q50WidenKernel` feeding `symWmmaDeqMw8`) has no
      format-specific test and no file in the tree carries Q4_0 or Q5_0
      tensors — the 8B Q4_K_M here is q4_K + q6_K only. It is migrated
      but UNVERIFIED until 9.3.2's Q4_0 file exists, which that item
      already calls for.
      SPLITKERNEL IS NOW THE MoE's TOP ITEM at 336 launches and
      290.95 ms — 65.4% of the LOAD phase's device time, 23.2% of the
      whole run's (the figure first recorded here read the `avg` column
      as a total and called the share prefill's; re-derived under 9.2.6)
      — and the CAUSE IS ITS LANE MAPPING, not the work. It runs
      ONE WAVE PER BLOCK — `b = globalIdX() / 64`, then
      `i = lane; while (i < blockBytes) { ...; i = i + 64; }` — so for
      IQ4_NL's 18-byte block lanes 0..17 copy ONE BYTE each and lanes
      18..63 idle: 28% lane occupancy and a byte at a time. A
      256-element block (108-210 bytes) fills the wave but still copies
      bytewise. Two fixes, either cheap: pack several small blocks per
      wave, or copy dwords with a byte tail (every payload is a whole
      number of dwords by construction, and the scale field is two
      bytes). It is one-time work per tensor that a fresh process per
      leg rep pays every time and llama.cpp charges to load — and it is
      two thirds of this model's LOAD and the largest single item left
      in 6.4.3 after the repack. NOT DONE: outside this unit's items, and
      recorded here so it is a short job rather than a rediscovery.
      Q3_K FOLLOWED, and it is cheaper than the byte count suggests
      because of where its scale lives. Q3_K is `hmask, qs, scales, d`
      and Q6_K is `ql, qh, scales, d` — the f16 is at the END of the
      block, like the ternary pair — so `Quant.scaleOffset` returns
      `blockBytes - 2` for both and THE PAYLOAD OFFSETS INSIDE A BLOCK
      DO NOT MOVE. Only the row stride and the scale read change, which
      is three lines a kernel instead of a re-offset of every field.
      `splitInto` and `splitKernel` were already general over
      `scaleOffset`, so neither needed a line.
      FOUR OF FIVE MIGRATED. `coopNeedsRepack` is Q6_K alone.
      THREE GATES FIRED ON Q3_K and each was right to:
      `deviceBlockStridesAgreeWithGgmls` caught that Q3_K was still in
      `coopNeedsRepack` while its `coopBlockWords` had already become
      the payload stride; `ResidentLayoutTest` asserted every type keeps
      its scale at offset 0, which Q3_K and Q6_K break — it now asserts
      the invariant the split actually needs, that the scale sits at ONE
      END so the payload either side stays contiguous; and the Q3_K Mw8
      probe was still uploading file blocks.
      ONE ERROR WORTH RECORDING, because it is the failure mode of a
      long mechanical pass: the first attempt at that probe fix matched
      `wDev #= heap ... rawW.count()` by SHAPE and landed in
      `compareInto`, a Q4_K probe 3000 lines away, which the filtered
      suite does not exercise — so it would have shipped silently. Match
      by enclosing FUNCTION NAME, never by a line shape that repeats.
      Reverted and re-applied by name; `compareInto` is absent from the
      diff.
      Q6_K LANDED TOO — ALL FIVE FORMATS ARE MIGRATED and
      `coopNeedsRepack` returns false for every type. Its 22 kernels
      went three lines each as predicted (row stride, block base, the
      scale through `scaleAtDev`), scripted by ENCLOSING FUNCTION NAME
      after the earlier slip. Four needed hand treatment:
      `q6kQ8WaveMatVecKernel` reads its scale as `vload<4>(ro + 206L)`
      bytes [2],[3] for four rows at once; `q6kWmmaIdMwKernel` has a
      SECOND block base (`qlo`/`qho`) that no `int64 ro = ` regex
      matches; the scale image and the widen index from the row start;
      and `qkvWaveMatVecKernel` carries its own Q6 branch behind
      `vQ6 != 0`, which only the fused-QKV test reaches.
      THE PAD MACHINERY IS DELETED: `q6kPadKernel`, `q6kPadLaunch`,
      `Linear.ensureQ6Pad`, `q6PadDev` and `q6PadW`. It existed only
      because 210 is not dword-aligned; the 208-byte payload is, so the
      coop GEMM reads `payloadDev.wordView()` like every other format
      and the `mmq q6` route takes `packedW`. `coopBlockWords(Q6_K)` is
      52, the payload stride.
      SEVEN GATES FIRED ACROSS THE Q6_K PASS and every one was a real
      miss: the coop tile dispatch had to match IQ4_NL and Q6_K by TYPE
      before the `splitOn` predicate (both are now split but neither
      uses the codebook launcher); `q6kQ8WaveMatVecIntoDevice`, the
      fused-QKV test's `bv6`, `MoeIdMw8Test.q6kSlab` and `MmqProbe`'s
      `checkMmq6` all staged file blocks; and the missed `qlo`/`qho`
      base showed up as the MoE id test alone.
      VERIFIED: 267/267 including `q6kKernelIsBitIdenticalToSerial` and
      thirteen other Q6_K gates (Wmma, Mw4, Mw8, Epi, coop, N256), and
      the 8B Q4_K_M — whose 33 Q6_K tensors include the output head —
      loads and generates, matching llama.cpp's greedy for 14 of 16
      tokens before the usual near-tie divergence.
      REMAINING FOR THIS ITEM: the predicate cleanup `coopNeedsRepack`
      (now constant false), `blockRepack2Kernel`, `blockPadKernel`,
      `coopDev` and `splitOn` itself — `splitOn` is now every format
      with a nonzero `scaleBytes` except MXFP4, so it can become that
      question — plus 9.2.2's `deqFor(Q8_0)` int8 twin.
- [ ] 9.2.2 The int8 widen twin of a Q8_0 weight — `deqFor(Q8_0)` builds
      a tile-major `deqDev` (and `ensureWidenSlab` a `deqSlabDev`) that
      the Mw8 GEMM and the sym id-kernels read — is a second copy of
      int8 data. Those kernels read the split payload in place and
      `deqFor(Q8_0)` turns false. Found by 1.3.3 on the 30B Q8_0: the
      twin is the whole excess over file bytes there, and the coop copy
      the split retires was never built because the widen route served
      prefill.

- [x] 9.2.3 The fused reduce+pack leaves no f32 attention output, and
      an o-projection on the f32 route has nothing to read. Found by
      9.3.2's iq4_nl file, which THREW on both arms —
      `matvecStagedKeep: no staged device activation for packed type 20`
      — where `nofd` generates, so the cause is the flash-decode
      staging and not the migration. It is a live regression from Unit
      7's GQA-4 widening (020f8ae): before that, llama-3-8B took the
      scalar attend pair, which stages a real f32 row. It reaches every
      format OUTSIDE `Linear.qAct` — IQ4_NL, IQ4_XS, Q4_1, Q5_1, IQ1_S,
      IQ1_M, F16 — on any GQA-4 or GQA-8 model, which is why no file in
      the suite saw it: the toy fixtures and every 8B in the tree carry
      only `qAct` types.
      FIX: `Linear.ensureStagedF32` materializes the f32 form lazily —
      the remembered source when there is one, else a reduce of the same
      split partials through `AttnKernel.reduceF32LaunchNoSync`. The
      fused path keeps its saved launch for the packed consumers that
      are the common case, and the f32 route pays one reduce only when
      it actually asks.
      GATED BOTH WAYS: the lazy reduce must reproduce the attend pair's
      row at GQA x8 and at GQA x4, and — the does-not-fire half — an
      ordinary device staging must still copy its remembered source, so
      the partials branch cannot hijack the path every other layer
      takes. 270/270.


- [x] 9.2.4 `wavef`'s guard asks the real question. It read `!q8kDims`,
      which stands in for "the integer wave route did not engage" and
      gets it wrong for a type that has NO integer route: IQ4_NL
      answers no to both, so at a 256-aligned width it decoded
      item-per-row. It is now `!(wave || wave6 || ... || waveIq)`,
      computed after `waveIq` for that reason.
      THE GUARD ALONE WOULD HAVE BEEN INERT, and this is the part worth
      remembering. `launchOne`'s `wavef` arm dispatches Q8_0, Q5_0 and
      Q4_0 only — IQ4_NL fell through to `matVecLaunch`, the same
      item-per-row kernel — while `hasF32WaveKernel` answered TRUE for
      IQ4_NL, IQ4_XS, Q4_1 and Q5_1, none of which had one. A lying
      predicate over a dispatch that silently falls through is how a
      routing fix measures as "no change".
      So: `iq4nlF32WaveMatVecKernel` written (Q4_0's wave geometry with
      the nibble through `iq4Kv` and lut4, exactly as the one-item
      kernel decodes it), the dispatch arm added, and
      `hasF32WaveKernel` cut to the four that exist.
      LATENT CONSEQUENCE, unmeasured: at a NON-256-aligned width
      IQ4_XS/Q4_1/Q5_1 now get the item kernel's row chunking back,
      like every other item-per-row type. No file in the tree has that
      shape.
      MEASURED on the iq4_nl 8B, quiet box, ABBA arm order, rep 1 kept
      because all three reps of every pass agreed inside 1.5%:

      | arm | decode t/s | weight GB/s |
      |---|---|---|
      | item-per-row (pre) | 26.22 - 26.94 | 124 |
      | f32 wave (post) | 41.39 - 42.90 | 202 |

      +59% DECODE, and 202 GB/s is the wall (206 practical). Prefill
      unmoved at 833-842 both arms, which is right: this is a decode
      mat-vec route. Load 3020-3067 across all twelve runs, so the box
      held ([[load-time-is-the-box-witness]]).
      THE GREEDY STREAM IS IDENTICAL over all 128 tokens. A wave
      reduction reassociates what the item kernel summed serially, so
      the tokens were expected to move and did not — the difference is
      below the decision threshold everywhere in this window. The route
      change is gated by the kernel-against-host test; this is a bonus.
      AND IT ERASES 9.3.2's REGRESSION: 42.90 against the 28.60 that
      file read BEFORE the migration is +50%, so the shipping path is
      half again faster than it was, not 7.7% slower.
      NOT DONE, and worth more still: IQ4_NL has no INTEGER wave route either.
      Q4_0 runs at 213 GB/s on that route against this one's ceiling.
      IQ4_NL is a 16-entry 4-bit codebook — the MXFP4 shape, lut4 then
      dotSum over per-32 Q8 — so the kernel is a known quantity. It
      changes the answer, so it belongs behind `q8Route` with its own
      exactness budget, which is a unit and not a line.


- [x] 9.2.5 IQ4_NL's INTEGER wave route. Q4_0's twin — one lane per
      32-element block, the same 16-byte payload and f16 scale — with
      the nibble through `iq4Kv` and lut4 instead of `q - 8`.
      `dotSum` TAKES A SIGNED RECEIVER, which the codebook family
      (`iqDot16`) already relied on and I did not check first: the first
      cut carried a +128 bias in the table so the codebook would be the
      unsigned operand of a mixed dp4a, and paid it back off the q8_K
      block's stored sub-block sum. Wrong by up to 140% — the shifted
      values above 127 read back as negative. The signed table feeds it
      directly: no bias, no sum load, and cheaper than the Q4_0 twin,
      which needs its -8 only because its quants are naturally unsigned.
      Read the proven idiom in the same family BEFORE inventing one.
      `packedAct` turns true for IQ4_NL as a consequence, so the
      o-projection reads the packed stage and 9.2.3's lazy f32 reduce
      stops firing on this format — a launch a layer a token.
      NOT ADDED: the accum form. `matvecStagedAccum` has branches for
      waveIq, Q4_K and Q6_K, and none for Q4_0/Q5_0, so IQ4_NL keeps
      their behaviour rather than growing a sixth.
      MEASURED on the iq4_nl 8B, ABBA, three reps a pass:

      | arm | decode t/s | weight GB/s |
      |---|---|---|
      | f32 wave | 42.55 - 42.82 | 202 |
      | integer wave | 44.12 - 44.93 | 211 |

      +4.9%, AND THAT IS THE WHOLE HEADROOM: Q4_0 measures 213 GB/s on
      this route, so IQ4_NL has arrived at the same wall. The f32 wave
      was already at 202, which is why this leg is worth 5% where the
      route fix before it was worth 59%. Both f32 passes agree to 0.3%,
      both integer passes to 1.8%; load 2977-3075 across twelve runs
      with no systematic split between arms.
      The greedy streams agree for 4 of 128 tokens and then diverge —
      the activation quantization, which is why the route is behind
      `q8Route`. The kernel is exact to 2.0e-06 against a host
      reference carrying the same quantization, and that is the gate.
      NOISE WORTH NAMING: prefill read 814/796 in the integer arm's
      second pass against 833-841 everywhere else. Prefill is the coop
      GEMM, which this change does not touch, so it is the box, not the
      code — recorded rather than averaged away.
      CUMULATIVE for this file: 28.60 before the migration, 26.39 after
      it, 42.90 with the route fix, 44.93 here. 1.57x over where it
      started.


- [x] 9.2.6 `splitKernel`'s lane mapping, recorded under 9.2.1 and 6.4.3
      and deferred twice. It gave every block a 64-lane wave and walked
      it a byte at a time, so an 18-byte IQ4_NL block left 46 lanes idle.
      Now one work item per payload DWORD: the scale sits at one end by
      construction, so a block's payload is a single contiguous run in
      the source and its destination is dword-aligned because the row
      prefix is padded to one. The scales go in a second kernel, one
      item per block, which keeps the word view unaliased. The byte
      kernel stays for a payload that is not dword-clean — every split
      type is, but Unit 8's IQ1 pair is not migrated yet.
      MEASURED, ABBA, three reps a pass, on LOAD MS because that is
      where the split runs:

      | file | block | pre | post | delta |
      |---|---|---|---|---|
      | iq4_nl 8B | 18 B (4 dw) | 2883 | 2578 | -10.6% |
      | MoE iq3_xxs | 98 B (24 dw) | 2265 | 2090 | -7.7% |
      | Q6_K 8B | 210 B (52 dw) | 3333 | 3326 | -0.2% |

      Prefill and decode flat on all three, which is right for bind-time
      work.
      Q6_K IS THE CONTROL AND IT COULD HAVE REFUTED THIS. A 210-byte
      block already fills the wave (3.3 iterations, ~77%) and still
      issues 210 byte-writes against 52 dword-writes — so if the DWORD
      conversion were doing the work, Q6_K would have gained. It did
      not. The gain is the lane mapping alone, and the 16x cut in
      dispatched threads that comes with it (one 64-thread workgroup per
      block becomes four items).
      RECONCILED 2026-09-17 off the SAVED TRACES — `prof-moe2/3/4`
      under `tmp/cbq/` through `cajeta profile summary`, so no new run.
      9.2.1's figure was wrong twice and both errors were in the
      READING, not the profiler.
      THE UNIT: "866 us" is the `avg` column. The row is `splitKernel |
      count 336 | self 290.95 ms | total 290.95 ms | avg 865.92 us | max
      3.54 ms | 23.2%`. The total is 291 ms; 866 us is ONE launch of
      336. A whole-model split moves ~7 GB in and 7 GB out, so 866 us
      would have been 14 TB/s — the reading should have refused itself.
      THE DENOMINATOR: 23.2% is of the RUN's device self time, 1.25 s
      over load, prefill and decode. Windowed to the measured prefill
      the count is ZERO. All 336 launches fall in LOAD, where they are
      65.4% of the phase. The saving landing in load is not the anomaly;
      load is the only place it could have landed.
      THE WORK IS CONSERVED ACROSS THE MIGRATION, which is the
      independent check that neither number is invented. Before IQ4_NL
      moved (`prof-moe2`): `blockRepack2Kernel` 1324 launches / 176.9 ms
      IN PREFILL, 25.5% of it, and `splitKernel` 264 / 96.9 ms IN LOAD.
      After (`prof-moe4`): no repack at all, `splitKernel` 336 /
      291.0 ms, all of it in load. Prefill gave up 207 ms of wall
      (778.8 -> 572.0, 657 -> 895 t/s) for the 176.9 ms of repack it
      stopped doing, and load took the same work as split. 9.2.1's claim
      was about load from the beginning.
      AND THE A/B CLOSES ON IT: 291 ms before this item, 175 ms saved,
      ~116 ms left — 14 GB of traffic at 48 GB/s becoming 121, against
      this device's 206 GB/s ceiling. Both ends are physical.
      WHAT THE ERROR WAS: a per-launch average read as a total, and a
      whole-run share read as a phase share. Both are columns of the
      same row, and the arithmetic refutes the misreading in one line —
      bytes over time against a known ceiling. The 6.4.3 census carries
      the same defect and is corrected there.


- [x] 9.2.7 `splitScaleKernel`'s lane mapping, which 9.2.6 left behind on
      the half it did not touch. The 6.4.3 census reads splitPayload
      64.23 ms and splitScale 45.70 ms: 42% of the split's cost to move
      TWO BYTES a block, about 1% of its bytes. It is one work item per
      block issuing two byte-writes — the same shape 9.2.6 replaced on
      the payload half, and the scales of a row are contiguous in the
      destination by construction, so a wave can take a row's whole
      scale prefix as dwords. Gated by the device-vs-host split tests
      that 9.2.6 added.
      DONE 2026-09-17, AND THE STATED MECHANISM IS REFUTED. The gain is
      real but small, and the reason it is small is the finding.

      | arm | launches | self | per launch |
      |---|---|---|---|
      | pre, `splitScaleKernel` | 336 | 45.70 ms | 136.01 us |
      | post, `splitScaleWordKernel` | 264 | 26.15 ms | 99.04 us |
      | post, `splitScaleKernel` (mid-row chunks) | 72 | 17.55 ms | 243.69 us |

      43.70 ms against 45.70, -4.4% overall and about -7% on the 264
      chunks that actually changed kernel. The load A/B is FLAT on all
      three shapes (iq4_nl ~2600 both arms, MoE 2105 -> 2087, Q6_K 3336
      -> 3333), which is the right answer rather than a disappointing
      one: 45.70 ms inside a 2.1-3.3 s load is 1-2%, under a +/-3%
      spread. A flat A/B here means "smaller than the noise floor", never
      "no change" — 9.2.6's 175 ms was four times larger and DID resolve.
      WHY THE MAPPING BOUGHT SO LITTLE: the scale split is not store
      bound or item bound, it is READ AMPLIFICATION bound, and the new
      mapping does not change the reads. Block scales sit `blockBytes`
      apart, so a wave pulls one cache line per block to take two bytes:
      64 lanes over 128 blocks of iq3_xxs span 12544 bytes, 196 lines,
      to use 256 bytes — 49x. At 99.04 us a 32 MB chunk that is 21.9 MB
      of lines at ~221 GB/s, which is AT this device's ceiling. Both
      arms were already there; the dword store bought the 7% left over.
      There is no lane mapping that improves this further.
      WHAT WOULD: fold the scale into the payload pass, which already
      reads those exact lines. Opened as 9.2.8 — and it would retire
      `splitScaleWordKernel` rather than build on it.
      THE GATE FALLS BACK MORE THAN EXPECTED: 72 of 336 chunks end
      mid-row and take the per-block kernel, and they are the expensive
      ones (243.69 us against 99.04) because they are the 32 MB chunks of
      the big tensors. 21% of launches, 40% of the remaining cost.

- [ ] 9.2.8 Fold the scale into the payload pass. 9.2.7 measured the
      scale split at ~221 GB/s of CACHE LINES for 2 useful bytes a
      block — at the device ceiling for an access pattern that is 49x
      amplified, so no mapping of a separate pass can improve it. The
      payload kernel already reads those same lines: for a head-scale
      format the scale is the two bytes before the payload, for a
      tail-scale one the two after. A merged pass makes the scale
      approximately free and retires both `splitScaleWordKernel` and
      `splitScaleKernel`, worth the whole 43.70 ms on the MoE.
      THE OBSTACLE IS THE WRITE, not the read: the payload kernel stores
      through `outW` (`KernelBuffer<int32>`) and a scale is two bytes, so
      either the kernel takes a second int8 view of the same buffer — two
      descriptors onto one allocation, which is what 9.2.6 split the
      kernels to avoid — or a lane assembles a prefix dword from two
      blocks whose lines it does not both hold. Settle which by probe
      before writing the kernel.

- [ ] 9.2.9 THE ROUTE TABLE — this package's registration. Specified
      as the xpu layer's selection contract at
      `cajeta/specs/archive/route-table-spec.md` (**CLOSED 2026-09-18**,
      plan at `cajeta/agents/archive/route-table-plan.md`; architecture
      note `docs/specification/xpu/CajetaXPU-Routing.md`, guide
      `docs/guide/25-xpu-kernels.md` §25.6). The layer half SHIPPED —
      `Route` is one kernel variant per row with `shapeRefusal` /
      `readyRefusal` naming their own gates, `RouteTable` resolves at
      bind (`admissible` / `pick` / `candidates`), `whyNotPicked` names
      the row that got furthest, and `audit` walks format / regime /
      shape only, reporting what a row needs rather than ruling on it.
      This item is the llm half: every predicate below becomes a row,
      every dispatcher gets explicit arms and a terminal refusal, and
      `theFourPartInvariant` becomes the audit's walker. NO LONGER
      BLOCKED — was blocked on that plan's Units 1-3
      (`Route` / `RouteTable` / the audit) in the runtime; it follows
      the spec's §4.2 order and owns the spec's §4.4 acceptance — the
      §3.4.5 grep returning nothing, and flat legs on the recorded
      files. Not a sweep.
      This class of defect has now been solved eight times and each
      time from scratch, because the knowledge of which format may take
      which route lives in prose and in whoever last hit the refusal,
      not in the code. Today alone: `wavef`'s guard (9.2.4),
      `ExpertBank.idReady()`, `Linear.packedWaveReady()` — each a
      predicate naming formats where it meant "on an integer wave
      route", each costing a whole route, and then `codebookId()` was
      added as one more list of the same shape.
      THE HALF THAT EXISTS: `theFourPartInvariant` (3.1.5) already
      asserts, over every `supported()` type, that `packedSupported`,
      `coopSupports`, `hasKernel` and the host chain agree. It has held
      since Unit 3. It covers four parts; the routes have grown to a
      dozen predicates outside it — `idReady`, `idRowReady`, `symId`,
      `packedWaveReady`, `coopRoutedHere`, the `wavef` guard, the
      `zeroSyncReady` clauses — so it stays green while a route it does
      not know refuses.
      THE TABLE: one row per route — name, the regime it serves (decode
      row / prefill batch / bind), the predicate that admits a format,
      the dispatcher that owns it. Every predicate above becomes a row
      or a derivation from `qAct` / `intWaveRouted()`; `codebookId()`
      and every `== 12 || == 14` goes. `backendIsVulkan()`'s ~14 sites
      are NOT this item (they are capability, and belong in xpu); they
      are listed so the table does not absorb them by accident.
      THE TEST: walk every `supported()` format against every row and
      assert each is admitted or refused BY NAME. Adding a format is
      then: write its kernels, add it to the rows that apply, and the
      test names the rows you forgot. Adding a route is: add a row, and
      the test audits it against every format that exists. A does-fire
      and a does-not-fire test per row, as 9.2.7 did. (Drafted with a
      dispatcher-arm probe too — "an admitted format has an ARM, a bare
      `else` fails on day one". Route-table spec §3.0 REMOVED that: at
      one row per kernel variant there is no arm to omit, and no
      dry-run flag for a dispatcher to forget to honour.)
      NOT IN SCOPE: the capability / cost / policy split (capability
      into xpu, measured selection from the Autotune store). That is
      the arc this table sits under, and it is a spec. This item is the
      part that stops the bleeding now.

      READ FIRST, 2026-09-18 — the call sites, before any code. Three
      decisions fall out of them and they are settled here rather than
      discovered halfway.

      **A row class per KERNEL, instantiated once per format.** `waveIq`
      is ONE flag serving five formats (IQ2_XXS, IQ2_XS, IQ3_XXS,
      IQ2_S, IQ3_S) and its launcher already takes `packedTy`, so it is
      five instances of one row class, not five classes and not one row
      admitting a set. `matVecLaunch`'s eleven arms are eleven
      instances of another. Rows are data; classes are kernels.

      **What is NOT a row, and the plan must not turn into one.**
      `oneSync` and `foldBias` are LAUNCH parameters — the same kernel
      submitted differently — so they ride on the call and a row's
      `dispatch` may branch on them. It may never branch on the FORMAT.
      That is the line that keeps "no arms" meaning something.
      `q8`, `packed1`, `qAct` and `packedAct` are not routes either:
      they are consequences. `qAct` is "did an integer row win", which
      after this item is a property of the row that was picked, read
      once at bind — not an OR of sixteen booleans recomputed in
      `finishDeviceSetup`. The shared stage (`stageHit`) keys on it, so
      it has to survive the move as one question with one answer.

      **`wavef`'s negation DISSOLVES, and this is the proof the table
      is the right shape.** Today it reads `waveMv() && inDim % 32 == 0
      && !(wave || wave6 || wave2 || wave3 || wave5 || wave8 || wave40
      || wave50 || waveT1 || waveT2 || waveIq || wave4nl) &&
      hasF32WaveKernel(ty)` — a twelve-term negation meaning "no
      integer wave route engaged", and 9.2.4 is what happened when
      `!q8kDims` stood in for it and IQ4_NL decoded item-per-row at a
      256-aligned width. Under the table the integer row for a format
      simply OUTRANKS its f32 wave row by `priority`, and the negation
      is never written down at all. Nothing to widen, nothing to forget.

- [ ] 9.2.9.1 The sweep that precedes the table (route-table spec
      §4.2 step 1): every predicate still standing in for another
      question says what it means, so the rows start from predicates
      that are true. Small — 9.2.4 already did `wavef` — but do it
      first and separately, so a row that turns out to admit differently
      from the flag it replaced is a ROW bug and not an inherited one.

- [ ] 9.2.9.2 Decode-row rows, and `launchOne` retires. The sixteen
      booleans latched in `finishDeviceSetup` become rows resolved once
      by `RouteTable.pick` at bind and stored; `launchOne`'s ~16-arm
      chain becomes one virtual call. Q4_K is THREE rows at DecodeRow
      (wave, packed, and the three-buffer form) ordered by priority —
      exactly spec §3.1's worked example, and the code confirms it.
      `shapeRefusal` carries `inDim % 256` (the q8_K superblock, whose
      absence walked the pack off its buffer and poisoned a HIP
      context) and `inDim % 32` for the f32 wave rows. `readyRefusal`
      carries what `ensureDevice` has or has not done. The terminal
      `matVecLaunch` arm keeps its throw; it becomes the lowest-priority
      row per format it serves.

- [ ] 9.2.9.3 MoE decode rows: `ExpertBank.idReady` / `idRowReady` /
      `symId` / `codebookId`, and `MoeFfn.zeroSyncReady`'s nine gates.
      The split is already read gate by gate in the xpu guide §25.6.2 —
      three per-bank format tests become the `format()` of three rows,
      four shape tests go to `shapeRefusal`, two slab tests to
      `readyRefusal`, and `widenSlabOn` goes to `readyRefusal` too
      because it is a mutable A/B arm. Each keeps its own sentence in
      the `moe-row-route` record: `whyNotPicked().text()` replaces
      `sayRowRoute`, and 2.3.2's rule is that no refusal gets coarser.

- [ ] 9.2.9.4 Prefill rows: `coopRoutedHere`, `hasBatchKernel`, the
      widen and Mw8 gates, the coop tile-divisibility refusals.
      `coopRoutedHere` mixes all three questions in one expression —
      `coopOn` (a switch), `!backendIsVulkan()` (a capability, 9.2.10's,
      and it must NOT be absorbed here), `!hasBatchKernel(ty)` (a
      priority ordering, not a test), `coopColsOk` and `outDim % 128`
      (shape). Splitting it is most of this item.

- [ ] 9.2.9.5 Bind rows: `Quant.splitOn` and the `deqFor` twin, at
      `Regime.Bind`. BLOCKED on 9.2.1, which reduces `splitOn` to
      "nonzero `scaleBytes`" — doing this first would row-ify a list
      that is about to stop being one.

- [ ] 9.2.9.6 Attention rows: the flash decode / prefill tile gates on
      `hd == 128` (`AttnKernel.cajeta:2508` and the kernels above it).
      Not a weight format — the `ty` axis here is HEAD DIMENSION, which
      is the genericity the route-table plan's 1.1.8 proved on a dtype
      axis and a bin-count axis. Recorded under 4.3.5 as the reason
      bitnet-large (hd 96) takes the scalar pair, silently.

- [ ] 9.2.9.7 The audit, and the acceptance. `theFourPartInvariant`
      (3.1.5) becomes `theRouteTableInvariant`: queries built by walking
      `ty` 0..63 and filtering `Quant.supported(ty)`, then
      `RouteTable.audit(queries)` asserting `unservedCount()`,
      `neverFiringCount()` and the shadowed pairs. Per row a does-fire
      and a does-not-fire test via `auditRow`. Then spec §4.4: the
      §3.4.5 grep (`== 12 || == 14`, `codebookId`, `(q8 && wave) ||
      wave6`) returns nothing outside the rows, the filtered suite is
      green, and LEGS FLAT on the six recorded files of 9.3.2 and the
      Qwen1.5-MoE — this is a refactor, and a speed change in either
      direction is a routing change to explain.

      THE GREP, BASELINE 2026-09-18: the §3.4.5 patterns
      (`packedTy == 12`, `packedTy == 14`, `codebookId`) match **14
      lines in one file**, `model/ExpertBank.cajeta`. That is the number
      9.2.9.7 drives to zero, and it is smaller than the prose suggests
      — the format lists concentrated in the expert banks while the
      `Linear` side spread into sixteen named booleans instead. Two
      shapes of the same defect, and only one of them greps.

      FOUND WHILE READING, both to fix under 9.2.9.7 rather than
      quietly: `QuantKernel.hasKernel(ty)` omits MXFP4 while
      `matVecLaunch(ty, ...)` serves it, so the two lists already
      disagree — and `theFourPartInvariant` exempts MXFP4 by name,
      which is how it stays green over the disagreement. And
      `ExpertBank.idGemmReady()` is `idReady`'s defect a second time:
      `packedTy == 12 || == 14` in the same expression as `deqSlabDev`
      and `slabDev`, so widening the format list also claims the slab.

- [ ] 9.2.10 `Linear.backendIsVulkan()`'s ~14 sites (and `MoeFfn`'s
      two) become `Device.supports(Capability)` queries, with the
      `Capability` values they need added in the runtime — WMMA
      present, `dotSum` usable, the RADV context-loss quirk. Three
      questions under one name today: capability, driver quirk, route
      policy. The first two move; the third is a row's `needs`. After
      9.2.9; route-table spec §7.

- [ ] 9.2.11 `GgufFile.halfBitsToF32` in kernels -> `KernelBuffer.halfView()`
      reads. 125 call sites decode an f16 scale by bit manipulation
      with a subnormal loop (40 instructions, six branches a call);
      through a float16 view it is one load and `v_cvt_f32_f16`. Done
      for the two grouped coop bodies in 6.4.7 (bit-identical, -3% a
      launch); the wave mat-vecs and the dense coop bodies remain.
      Each kernel gains a `packedH` parameter; each launcher a half
      view kept beside the word view (`ExpertBank.slabH` is the
      pattern). Bit gate per kernel, the format's existing test.
### 9.3 Acceptance
- [~] 9.3.1 Filtered suite green.
      `LinearKernelRouteTest` was never IN the filtered suite, and
      adding it for 9.2.4 found `legacyWidensAreExact` red since
      e160e69 — it staged FILE blocks into a kernel that reads the
      split layout, the same miss the Q6_K pass hit seven times. With
      the staging fixed, Q4_0 and Q5_0 widen EXACT over 16384 bytes, so
      9.2.1's unverified widen route is now verified and correct.
      IT ALSO COST AN HOUR TO A USE-AFTER-FREE IN THE TEST ITSELF, and
      the shape is worth naming: `int8[] dev #= raw;` then, inside an
      `if`, `int8[] sp #= heap int8[...]; dev #= sp;` — `#=` records a
      BORROW, `sp` keeps title and frees at the end of the block, and
      the upload after it read reused memory. The tell was that the
      mismatch count MOVED between two runs that differed only in a
      print statement (32 then 31), with the first bytes wrong and
      everything after exact: allocator bookkeeping in the freed block's
      head. Hoisting the allocation to the enclosing scope made it
      exact. THREE MORE SITES had the same shape in
      `LegacyWaveMatVecTest` — the gates for every migrated format —
      passing on luck; all three hoisted.
      ONE FAILURE LEFT, and it is not this unit's:
      `tunesThePartitionWidthOnThisDevice` submits sixteen 512-token
      prefills to an engine with ctx 1024 and maxSeqs 1, and the third
      runs off the end.
      CAUSE READ 2026-09-17, and it is NOT "the slot is never released"
      as this item first recorded: the slot IS released — the finish
      path in `Scheduler` calls `cache.release(r.slot)` and clears
      `slotReq` — but it never resets the DEVICE-RESIDENT KV planes.
      `DeviceKv` keeps its own `len`/`seq`, `owns()` stays true because
      the planes are still ahead of the host cache, and
      `CausalLM.seqLength` asks the planes first by design (a resident
      generation must not read the deliberately-behind paged length).
      So the next request admitted to that slot inherits the finished
      one's position: prefill 1 leaves the planes at 512, prefill 2
      takes `startPos = 512` and `adopt`s 1024, prefill 3 is refused —
      which is the message's "512 tokens at position 1024" exactly.
      FIXED 2026-09-17 (Julian: "Fix it"), and it was three sites, not
      one: `CausalLM.releaseSeq(seq)` now releases the blocks AND the
      planes together — `resetSeq` is it plus `pos = 0` — and the
      scheduler's FINISH, CANCEL and PREEMPT paths all go through it
      where each had reached past the model for `kvCache().release`.
      TDD: `SchedulerTest.aRecycledSlotStartsTheNextRequestAtZero` runs
      two 64-token requests through one slot of a 128-position context
      on the routable fixture, so a leaked position runs off the end.
      IT WAS RED FIRST, with the predicted message ("64 tokens at
      position 65 exceeds maxSeq 128"), and the fix turns it AND
      `tunesThePartitionWidthOnThisDevice` green. The suite's other
      eight scheduler tests — cancel, preemption, resume, admission,
      all paths this touches — stay green. Decode 144.3 t/s and the
      greedy stream hash unchanged.
      IT WAS NOT A TEST ARTEFACT: the same path serves a second request
      on a recycled slot, so its RoPE positions and KV writes would have
      landed at the previous request's offsets. The suite had no test
      that submits two requests on a device-routed model, which is how
      it survived.
      A SECOND RED CAME WITH THE HELPER, surfaced by pulling
      `SchedulerTest` (and its `ForwardTest` helper) into the filtered
      suite: `ForwardTest.tiedEmbeddingAndExplicitHeadDim` expects 11
      parameters from a 1-layer tied llama and the tree reported 12.
      FIXED 2026-09-17, and it WAS `lmHead` — this item first recorded
      it as null under tie, which it has not been since `43fbdf1` put
      the tied head on the device (4.3.3). The head is always a Linear
      now and a tied checkpoint binds it to the embedding table, so the
      reflective walk emitted an `lm_head.weight` key that the
      checkpoint does not have and `parameterTensors` counted the
      embedding twice. The key list is torch's `state_dict()` contract
      and torch omits a tied head, so the walk omits it: `InfModule`
      gains `tiedParameters()`, `Linear.bindTied` sets it, and
      `CausalLM.bind` uses that spelling for the tied branch. The
      tied test carries a new assertion that the name and tensor walks
      agree in length, so guarding only one of them fails loudly, and
      the untied `parameterNamesMatchTorchStateDictKeys` is the leg
      that asserts the hook does NOT over-fire.
      NOT a resident-bytes finding after all: `43fbdf1` measured that
      cost and took it deliberately ("the resident ledger grows by
      exactly the head's file bytes") in exchange for 19.4 ms a token
      of host `rowsDotRow`. Only the key list was wrong.
      Filtered suite 339/339.
- [x] 9.3.2 Legs per format (announced): Q4_K_M 8B (carries Q6_K and
      Q5_0 tensors), Q3_K_M, Q6_K, the iq4_nl file, a Q4_0 8B from
      `llama-quantize`; bit gate then A/B, no decode regression.
      THREE OF FIVE DONE 2026-09-17, against a binary built from the
      commit BEFORE the Q6_K migration (353087e) in a worktree, so the
      arms differ only by the change under test. Quiet box, arm order
      alternating by file, three reps, rep 1 discarded as cold
      ([[bench-first-model-runs-cold]] — it read 27.1 against 32.6 on
      the 6.6 GB file).
      THE BIT GATE IS EXACT: the greedy stream of all three files
      hashes IDENTICALLY before and after. The split moves bytes; it
      does not touch a value, and this is the proof rather than the
      claim.

      | file | pre | post | delta |
      |---|---|---|---|
      | Q6_K 8B | 32.61 | 32.87 | +0.8% |
      | Q4_K_M 8B | 43.18 | 43.42 | +0.6% |
      | Q3_K_M 8B | 49.80 | 49.88 | +0.2% |

      NO REGRESSION ANYWHERE, and all three are marginally FASTER — the
      Q6_K row largest, which is what the retired 212-byte pad predicts:
      that copy is 2 bytes a block of 6.59 GB, about 63 MB of write and
      read that no longer happens.
      THE OTHER THREE, 2026-09-17. The iq4_nl file DID already exist —
      `tmp/q1fix/llama8b-iq4_nl.gguf` from the IQ1 work, 189 iq4_nl
      tensors + 36 q5_K + the q6_K head. `llama8b-q4_0.gguf` (225 q4_0)
      and `llama8b-q5_0.gguf` (225 q5_0) were quantized from the Q8_0 8B
      with the same recipe. Q5_0 is here because 9.2.1 records its widen
      route as unverified with no file either, and it is the same
      commit as Q4_0.
      PRE IS b0bd58a — the parent of fa11822 (IQ4_NL), itself the parent
      of e160e69 (Q4_0/Q5_0) — so one control sits before all three. The
      Q6_K leg's binary could not serve: it was built at 353087e, which
      already carries them. 9.2.3's fix is patched into BOTH arms, so
      the iq4_nl file runs on the control and the arms differ only by
      the migration.

      THE BIT GATE IS EXACT on all three: iq4_nl e79402903475603a,
      q4_0 03a9ff9be0b1cb5c, q5_0 a5ce1a7a3dc4f2a8, identical pre and
      post. The q4_0/q5_0 hashes also match the run taken BEFORE 9.2.3
      existed, which is a free control on that fix.

      | file | pre | post | delta | post GB/s |
      |---|---|---|---|---|
      | iq4_nl 8B | 28.60 | 26.39 | -7.7% | 124 |
      | q4_0 8B | 46.01 | 45.78 | -0.5% | 213 |
      | q5_0 8B | 38.05 | 38.16 | +0.3% | 213 |

      Q4_0 AND Q5_0 COST NOTHING and are AT THE MACHINE WALL — 213
      weight-GB/s each, which is where Q4_K sits. Their widen route is
      now exercised by a file, closing half of 9.2.1's coverage gap.
      IQ4_NL REGRESSED 7.7% ON DECODE (and 2.4% on prefill), and the
      cause is not the layout but WHAT KERNEL READS IT. IQ4_NL sets no
      wave flag at all on a 4096-wide model: `waveIq` covers
      IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S and not IQ4_NL, and `wavef` —
      which would serve it, since `hasF32WaveKernel(IQ4_NL)` is true —
      is gated on `!q8kDims`, false whenever inDim % 256 == 0. So it
      decodes on the ITEM-PER-ROW f32 kernel, and the measurement says
      so: 124 GB/s against Q4_0's 213 on a file of the same size and
      the same bit width. The split costs 7.7% there because that
      kernel is latency-exposed, not bandwidth-bound; on the two
      formats at the wall it costs nothing.
      THE FIFTH UNREACHABLE FAST PATH OF THIS UNIT. `wavef`'s guard
      means "the integer wave route did not engage", and `!q8kDims` is
      the wrong way to ask it — a type with no integer route and a
      256-aligned width answers no to both. Fixing it is a routing
      change with its own bit-gate baseline (an f32 reduction reorder),
      so it is NOT folded in here. NOT DONE, and worth more than the
      regression it would erase.
      THE TRADE WAS NEVER TAKEN: 9.2.4 routed IQ4_NL off the
      item-per-row kernel and its decode is now 42.90 against the 28.60
      this file read BEFORE the migration. CLOSED — five formats
      measured, bit gate exact on all of them, and no decode regression
      on any shipping path.
- [~] 9.3.3 Resident bytes equal file bytes for every file above and the
      30B Q8_0 re-checked; the repack code gone.
      EXACT ON THE PURE Q6_K 8B, which is the file this unit most had to
      prove: `CAJETA_XPU_ALLOC_TRACE` reports 6,156,165,120 bytes in 225
      `allocResident` allocations. The file carries 226 Q6_K tensors
      totalling 6,587,105,280, and `token_embd` — 128256 x 4096 / 256 x
      210 = 430,940,160 — stays in the Embedding rather than a Linear.
      6,587,105,280 - 430,940,160 = 6,156,165,120. To the byte.
      THE REPACK CODE IS GONE: `q6kPadKernel`, `q6kPadLaunch`,
      `Linear.ensureQ6Pad`, `q6PadDev` and `q6PadW` deleted;
      `coopNeedsRepack` returns false for every type, so `coopDev` is
      never allocated. Those two copies were the whole excess over file
      bytes for these formats.
      OWED: the 30B Q8_0 re-check, and the two files 9.3.2 still needs.
