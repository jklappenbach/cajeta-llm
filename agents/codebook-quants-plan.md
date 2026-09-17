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
      HIP/Vulkan legs are MEASURED — see the legs table below
      Unit 6; the bar is not met and the gap is item 6.4.1.

## Unit 6 — IQ2_S, IQ3_S: raw signs and qh high bits (spec §3.4, 12.1)

### 6.1 TDD
      DECODE HALF NOW CLEARS, 2026-09-16 night after Unit 7 (which
      carries 6.4.1, this item's blocker): iq2_xxs 0.952, iq2_xs 0.954,
      iq3_xxs 0.997 of Vulkan against the 0.95 bar, and iq3_xxs — the
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
- [~] 6.3.2 The IQ3_XXS Qwen1.5-MoE: every expert tensor `batched`,
      perplexity within the MoE floor of llama.cpp's, legs.
      DONE 2026-09-16, legs included (they miss badly — item 6.4.3):
      all 24 layers report
      `moe-batch-route resident` and the four codebook types take the
      coop GEMM, with no refusal; perplexity 5.4687 against 5.4781
      (−0.17%), inside the MoE routing-flip floor. `DenseRouteProbe`
      set no expert residency budget, so its first answer was 24 ×
      "an expert is not admitted" — the trap `PplProbe` already
      documents; the probe now sets the budget the engine's AUTO would.

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
- [x] 6.4.2 The IQ3_S coop GEMM prefill gap: 0.83–0.90× of Vulkan where
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
- [~] 6.4.3 The Qwen1.5-MoE at 0.27× / 0.26×. Expert residency is the
      first suspect (the CLI ledger held 678 MB of 6.3 GB of expert
      bytes), so measure what the bank admits before touching a kernel.
      MEASURED FIRST, as the item says, and THE NAMED SUSPECT IS
      REFUTED. `residentKb` reads 6,969,468 — 6.97 GB — against a 6.3 GB
      expert set, so the bank now admits essentially everything and
      residency is not the cause. Re-measured baseline 2026-09-17 after
      Unit 7 and 6.4.2, which reached this file not at all: prefill
      615.3 against Vulkan's 2297.5 (0.268x), decode 36.75 against
      138.11 (0.266x).
      WHAT THE PREFILL CENSUS SAYS (512 tokens, device time):

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
      2. LAYOUT CONVERSION AT RUNTIME is 27.7% of prefill —
         `blockRepack2Kernel` 1420 launches (one per expert per layer,
         on first admission) and `splitKernel` 264. This is Unit 9's
         item 9.2.1 (`coopNeedsRepack`, the repack machinery) showing up
         as a third of the MoE's prefill. llama.cpp charges the same
         work to LOAD; we charge it to the first prefill that touches an
         expert, and a fresh process per leg rep pays it every time.
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
      902.1 against Vulkan's 2277.8 — 0.396, from 0.268 when this item
      opened and 0.286 after the halved tile alone. Completing the other
      four formats leaves it there (896 t/s in the profile): this model
      carries no Q6_K, Q3_K, Q4_0 or Q5_0 tensors, so the rest of the
      unit is worth nothing HERE and everything to the files that do. Decode is unmoved at
      0.262, which the census predicts: its 384 launches a token are
      cause 3, the grouped id-GEMM, untouched.
      BLOCKED ON UNIT 9 for the rest. Causes 2 and 4 are Unit 9's items
      by definition — the repack machinery and the IQ4_NL migration —
      and cause 3 needs a grouped id-GEMM over the expert dimension that
      the codebook formats do not have. Re-open this item when Unit 9
      lands; nothing else here is worth doing before it, because a third
      of the prefill is work Unit 9 deletes outright.


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
      SPLITKERNEL IS NOW THE MoE's TOP ITEM at 23.2%, 336 launches at
      866 us, and the CAUSE IS ITS LANE MAPPING, not the work. It runs
      ONE WAVE PER BLOCK — `b = globalIdX() / 64`, then
      `i = lane; while (i < blockBytes) { ...; i = i + 64; }` — so for
      IQ4_NL's 18-byte block lanes 0..17 copy ONE BYTE each and lanes
      18..63 idle: 28% lane occupancy and a byte at a time. A
      256-element block (108-210 bytes) fills the wave but still copies
      bytewise. Two fixes, either cheap: pack several small blocks per
      wave, or copy dwords with a byte tail (every payload is a whole
      number of dwords by construction, and the scale field is two
      bytes). It is one-time work per tensor that a fresh process per
      leg rep pays every time and llama.cpp charges to load — but it is
      23% of this model's prefill and the largest single item left in
      6.4.3 after the repack. NOT DONE: outside this unit's items, and
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
      prefills to an engine with ctx 1024 and maxSeqs 1, and the
      sequence slot is never released between requests, so the third
      runs off the end. `e.cancel(id)` does not release it. It is an
      autotune TIMING test in a correctness suite and it wants the
      scheduler's request lifecycle, which is a different unit.
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
