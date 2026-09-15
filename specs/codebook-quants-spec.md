# Codebook and ternary quants — IQ1/IQ2/IQ3 and TQ1_0/TQ2_0 route like every other format

Status: draft 2026-09-15 (Julian: "support as many model architectures /
quantizations as we can"; TQ1_0/TQ2_0 in, and the BitNet architecture
with them, since nothing else produces a ternary file).
Evidence: prefill-routes plan 7.5.2, which shipped IQ4_NL/IQ4_XS and
found the family splits there; ggml at llama.cpp 5306f4b
(`ggml-common.h`, `ggml-quants.c`, `ggml-cpu/quants.c`,
`src/llama-quant.cpp`, `src/models/bitnet.cpp`) read for every layout,
formula and number below. Nothing here is inferred from memory.

## 1. Definition

### 1.1 Problem

Nine ggml block types refuse to load. Seven are the codebook tier —
IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S — where a block
stores indices into a static grid of lattice points, with sign and
scale bits kept apart. Two are ternary — TQ1_0, TQ2_0 — where every
weight is −1, 0 or +1 times one block scale. `Quant.supported()`
rejects all nine, so `GgufFile.loadF32` throws "unsupported ggml type"
before a byte moves. That is an honest refusal and, with only a
decoder, it would be the only correct behaviour.

What refuses with them:

- Every llama.cpp file whose ftype is IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
  IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S or IQ3_M — the 1.5 to 3.5
  bit-per-weight tier, which is how a 70B fits a 24 GB card and what
  most published quantizations below Q4 are.
- The mixes inside those files. An IQ3_XXS file carries IQ2_S `attn_k`
  and `attn_q`; an IQ1_S file carries IQ2_XXS `attn_output`; an IQ2_S
  file is bulk IQ2_XS with IQ3_S embeddings; an IQ3_XS file is bulk
  IQ3_S with IQ3_XXS gate/up. One format alone opens nothing.
- Every BitNet b1.58 checkpoint. TQ1_0/TQ2_0 are the only ggml types
  for ternary-trained weights, and those files also carry
  `general.architecture = bitnet`, which `ModelConfig.fromGguf`
  refuses.
- Diagnostics. `GgufFile.typeName` knows IQ2_XXS, TQ1_0 and TQ2_0 and
  prints `type_17`, `type_18`, `type_19`, `type_21`, `type_22`,
  `type_29` for the other six.

### 1.2 What the formats are (read from ggml, not inferred)

Every one is a 256-weight block.

| type | id | bytes | bpw | grid table (entries × values) | signs | scale, per group | dot's float tail |
|---|---|---|---|---|---|---|---|
| IQ1_S | 19 | 50 | 1.5625 | iq1s_grid 2048 × 8 int8 (16 KB; 8 KB as nibbles) | none, values are −1/0/+1 | d·(2s+1), s 3-bit per 32; delta ±0.125 per 32 | 1 |
| IQ1_M | 29 | 56 | 1.75 | iq1s_grid, shared | none | d is an f16 rebuilt from four nibbles; d·(2s+1), s 3-bit per 16; delta ±0.125 per 8 | 1 |
| IQ2_XXS | 16 | 66 | 2.0625 | iq2xxs_grid 256 × 8 (2 KB) | ksigns index, 7-bit per 8 | d·(0.5+s)·0.25, s 4-bit per 32 | 0.125 |
| IQ2_XS | 17 | 74 | 2.3125 | iq2xs_grid 512 × 8 (4 KB) | ksigns index, 7-bit per 8 | d·(0.5+s)·0.25, s 4-bit per 16 | 0.125 |
| IQ2_S | 22 | 82 | 2.5625 | iq2s_grid 1024 × 8 (8 KB) | raw byte per 8 | d·(0.5+s)·0.25, s 4-bit per 16 | 0.125 |
| IQ3_XXS | 18 | 98 | 3.0625 | iq3xxs_grid 256 × 4 (1 KB) | ksigns index, 7-bit per 8 | d·(0.5+s)·0.5, s 4-bit per 32 | 0.25 |
| IQ3_S | 21 | 110 | 3.4375 | iq3s_grid 512 × 4 (2 KB) | raw byte per 8 | d·(1+2s), s 4-bit per 32 | 1 |
| TQ1_0 | 34 | 54 | 1.6875 | none | none | d per 256; five trits per byte, base 3; d at the END | 1 |
| TQ2_0 | 35 | 66 | 2.0625 | none | none | d per 256; two bits per weight; d at the END | 1 |

Facts that shape the design:

- **The grids are small magnitudes.** IQ2 grid bytes are {8, 25, 43};
  IQ3_XXS {4, 12, …, 62}; IQ3_S odd 1..15; IQ1 {−1, 0, +1}. Every value
  fits int8, so a grid entry is eight (or four) int8 lanes and the dot
  against int8 activations is an integer.
- **Every dot is float × integer.** ggml's `vec_dot_*_q8_K` sums
  int8×int8 products, weights each sub-block by the odd integer 2s+1,
  and multiplies once per block by d·d_q8 and the constant tail. The
  IQ1 delta is a second integer sum — the activations of the group,
  times ±1 — folded with the constant 0.125. No float arithmetic
  happens inside a block.
- **Signs come two ways.** IQ2_XXS, IQ2_XS and IQ3_XXS store a 7-bit
  index into `ksigns_iq2xs` (128 bytes; bit 7 completes parity).
  IQ2_S and IQ3_S store the eight sign bits raw. IQ1 stores none.
- **Two block layouts have no leading scale.** IQ1_M rebuilds its f16
  d from the top nibble of each of its four scale words; TQ1_0 and
  TQ2_0 put d after the payload.
- **The static tables are the whole codebook.** ggml-common.h holds
  them as literals, 33 KB in the byte-per-value form. ggml's runtime
  `iq2xs_init_impl` builds search structures for the quantizer only;
  inference needs the literals, and the index order of the literals
  matches the seeds (checked entry by entry for the first five).
- **Ternary is only for ternary weights.** ggml's TQ quantizer is a
  plain absmax round to {−1, 0, +1}; on any other model it destroys the
  weights. llama.cpp's converter writes BitNet weights as ternary ×
  scale, so a TQ block carries the scale in d and files from it have
  no separate scale tensor. Upstream runs TQ on CPU only — no CUDA,
  HIP, Vulkan, Metal or SYCL kernel exists.
- **The seven IQ types all have HIP and Vulkan kernels upstream**, so
  same-file, same-backend parity numbers exist for them.

### 1.3 Scope

In: the nine types through the four-part stack prefill-routes 7.5
defined — block decoder, host mat-vec, bind gate, device route with a
decode mat-vec and a coop GEMM — plus the two things this tier adds
that no earlier format needed: device-resident grid tables, and Q8
activation staging with per-group sums for the integer dots. The six
missing type names. The `bitnet` architecture: two sub-norms, NEOX
rope, tied output, optional per-tensor scale. A cajeta converter from a
`BitnetForCausalLM` checkpoint to a `bitnet` GGUF at f16, tq1_0 or
tq2_0 — a ternary block packs exactly, so this is packing, not a
quantization search. One resident layout rule for every packed format
(§10), which also migrates the six older formats that hold two device
copies today and retires the repack kernels, the pad copy and
`coopDev`. Arbiter checkpoints made on this box, IQ with llama.cpp's
own tools and TQ with ours, and the parity bar every other format met.

Out: quantizing to the seven codebook types (cajeta-llm consumes them;
the importance-matrix search structures stay in ggml). bitnet.cpp's
`i2_s` format, which is not a ggml type. The retired ids 31–33 and 36–38,
NVFP4 (40) and Q1_0 (41). Mixture-of-experts checkpoints in these
formats are in scope for the route — expert slabs use the same kernels
— and one is made for the measured set (12.4).

### 1.4 Constraints

- **Four parts or an honest refusal.** A type enters
  `Quant.supported()` and `Linear.packedSupported` in the same commit
  as its host mat-vec and both device kernels. A decoder alone turns
  today's honest refusal into the 2026-08-26 `loadF32` trap.
- **Bit-exact decoders.** `Quant.<fmt>Block` reproduces ggml's
  `to_float` exactly over blocks ggml's own quantizer wrote. The 7.5.1
  fixture rule stands: a throwaway C program under `tmp/` linking
  libggml-base, only the output committed, nothing non-cajeta in
  `tools/`.
- **Integer dots.** Host and device mat-vecs accumulate int8 products
  in int32 with the 2s+1 sub-block weights and touch float once per
  block. The IQ1_S delta rides the per-32 sums the 320-byte Q8 pack
  already carries; IQ1_M's per-8 sums are reduced from the loaded
  lanes.
- **Tables are generated, then pinned.** The grids and `ksigns` are
  generated from ggml-common.h into one cajeta source, and a test pins
  the XXH3 of every table to the value the generator printed from the
  same header.
- **One resident copy, at the file's byte count.** This tier exists to
  fit. The only field in any ggml block that breaks dword alignment is
  the 2-byte f16 block scale, so the resident layout de-interleaves it
  PER ROW: each row is its block scales back to back, padded to a
  dword, then its payloads, each a whole number of dwords. One array,
  one stream per wave, the file's row stride. Every device kernel of a
  format reads that array, the decode mat-vec and the GEMM read the
  same one, and no weight holds a `packedDev` and a `coopDev` copy at
  once. The split happens on the device at upload through a bounded
  staging chunk (blocks are independent, so a tensor streams through in
  pieces), so the peak is resident plus one chunk, never two copies.
  Measured on the Q8_0 wave mat-vec (gfx1151, 2026-09-15): two separate
  arrays cost 11–42 % against the file layout because a wave walks two
  streams, and a bare 32-byte payload alone still lost up to 34 %
  because a 4096-column row then strides by a power of two and waves
  camp on the same channels; the per-row form matched or beat the file
  layout on every shape.
- **Optimal and measured.** Wave-per-row coalesced decode, one gather
  per 8 weights, no per-lane byte reads. Parity is the same-file
  llama.cpp number on the same backend, never a Q4_K comparison.
- **No python tools; no /tmp.** As everywhere in this repo. The
  converter is a `.cajeta` program under `tools/`, built on
  `SafetensorsFile`, `SpProto`, `Tokenizer` and `GgufWriter`, which
  exist for this.

## 2. Loading and naming

- **2.1** When a tensor's ggml type is 16, 17, 18, 19, 21, 22, 29, 34
  or 35, `GgufFile.typeName` names it (`iq2_xxs`, `iq2_xs`, `iq3_xxs`,
  `iq1_s`, `iq3_s`, `iq2_s`, `iq1_m`, `tq1_0`, `tq2_0`) in every route
  record, diagnostic and refusal.
- **2.2** When a type has all four parts, `Quant.supported()`,
  `Quant.blockBytes`, `Quant.blockElems` (256), `Quant.dequantize`,
  `Linear.matvecInto`'s host chain and `Linear.packedSupported` all
  admit it; when it has fewer, none does, and the load refuses with
  the type's name.
- **2.3** When `Quant.dequantize` meets a supported type it has no
  branch for, it throws naming the type. Today's bare `else` decodes
  it as Q6_K.
- **2.4** When the loader refuses a type, its "supported:" list names
  every type `supported()` admits; today it stops at Q6_K.
- **2.5** When the fixture manifest is read, it lists every fixture in
  the directory with type, block count and generator; today it is six
  formats behind.

## 3. Block decoders

- **3.1** When any of the nine decoders runs over a fixture ggml's
  quantizer wrote, every f32 equals ggml's `to_float` bit for bit
  (`QuantTest.checkFormat`, exact equality).
- **3.2** When IQ1_M is decoded, the block scale is the f16 rebuilt
  from the top nibble of each of its four scale words, and each qh byte
  supplies two groups of 8: three high index bits plus a delta sign in
  its low nibble, the same in its high nibble.
- **3.3** When TQ1_0 is decoded, its three regions land where ggml puts
  them — 32 bytes × 5 trits to weights 0..159, 16 bytes × 5 to
  160..239, 4 qh bytes × 4 to 240..255 — and a fixture with distinct
  values in every region proves it.
- **3.4** When IQ2_S or IQ3_S is decoded, the sign bytes are read raw;
  when IQ2_XXS, IQ2_XS or IQ3_XXS is decoded, through `ksigns`.
- **3.5** When the fixtures are made, they are nine pairs over the same
  Llama-3.1-8B `token_embd` values the directory already uses, from
  `ggml_quantize_chunk`, with a synthetic positive importance vector
  for the types whose quantizer asserts one (IQ2_XXS, IQ2_XS, IQ1_S).

## 4. Codebook tables

- **4.1** When the engine starts, the five grids, `ksigns_iq2xs` and
  `kmask_iq2xs` exist as static cajeta arrays generated from
  ggml-common.h, and a test pins each table's XXH3 to the value the
  generator printed.
- **4.2** When a kernel needs a table, it reads a device copy uploaded
  once per process and shared by every layer and expert — a static
  `KernelBuffer` behind an ensure guard, the `pfSink` pattern — never
  a per-launch or per-layer upload.
- **4.3** When a coop GEMM workgroup stages codebook blocks, the table
  it reads is in LDS, filled by the workgroup at entry behind one
  barrier (llama.cpp's Vulkan path stages 2·2048 + 4·2048 bytes for
  IQ1); when a decode mat-vec wave reads it, LDS or the L1 path is
  chosen by measurement (12.1).
- **4.4** When a table has a packed form — `iq1s_grid_gpu` at a nibble
  per value, the three-letter IQ2 alphabet at two bits per value — the
  device copy is the packed form and a kernel decodes an entry in
  registers.
- **4.5** When `cajeta.xpu.Constant<T>` (declared, not wired) becomes
  usable, the table binding moves under it without changing any
  kernel's arithmetic. The compiler finding is filed; this spec does
  not wait on it — CUDA ships the same tables in global memory.

## 5. Host mat-vec

- **5.1** When a codebook or ternary weight is multiplied on the host,
  `Quant.<fmt>MatVecIntoAt(packed, pOff, rows, cols, x, xOff, y, yOff)`
  computes it straight off the blocks in ggml's loop order, and
  matches dequantize-then-multiply to the existing `checkMatVec` bar.
- **5.2** When the host serves as the device oracle, it also has the Q8
  form consuming the 320-byte pack, so the integer route is checked
  against an integer twin, not only against f32.

## 6. Device decode mat-vec

- **6.1** When a bound weight of one of the nine types is decoded, the
  kernel is one wave per row — four lanes per block, four rows per
  wave, as the Q4_K Q8 wave kernel — over the Q8 pack, `dotAccum` for
  the products, the 2s+1 weight applied in int32 per sub-block, d·d_q8
  once per block. The 256-block already implies the Q8 route's
  `inDim % 256 == 0`.
- **6.2** When a group of 8 is decoded, it costs one table gather, one
  sign expansion (a `ksigns` byte or a raw byte to eight ±1 lanes
  through a permute or mask) and one `dotAccum`.
- **6.3** When IQ1_S runs, the delta term is ±(2s+1) times the pack's
  per-32 activation sum; when IQ1_M runs, the per-8 sum is one
  `dotAccum` of the loaded lanes against ones.
- **6.4** When TQ2_0 runs, the two-bit fields expand to −1/0/+1 int8
  lanes with shifts and masks and no table; when TQ1_0 runs, five trits
  per byte come out with the ×pow3, ×3, >>8 sequence ggml uses, all in
  integer.
- **6.5** When the kernel is compared with the Q8 host twin (5.2), it
  matches to the bar the Q4_K Q8 kernels meet in
  `LegacyWaveMatVecTest`; against the f32 host, to the Q8 route's bar.
- **6.6** When the kernel's ISA is read (`KernelIsa`), it shows no spill
  and a `v_dot4` count that matches the design.
- **6.7** When the backend has no Q8 wave route, the host mat-vec of §5
  serves and the route record says so; there is no second, f32-
  activation device kernel per format unless a measured backend needs
  one (12.6).

## 7. Prefill: the coop GEMM route

- **7.1** When a codebook or ternary projection is prefilled with the
  default `packed` weights at a shape the tile divides, it is batched
  through `<fmt>F16CoopX1/X3`, `prefill-mode` says `batched`, and
  `batch-refused` never fires for a supported type at a dividing
  shape.
- **7.2** When staging decodes a block into the f16 LDS tile, grid,
  sign and scale are applied in integer and converted once; the table
  is the LDS copy of 4.3.
- **7.3** When `coopBlockWords` is asked, it returns the payload's
  dword stride in the de-interleaved layout — IQ2_XXS 16, IQ2_XS 18,
  IQ2_S 20, IQ3_XXS 24, IQ3_S 27, IQ1_S 12, IQ1_M 14, TQ1_0 13,
  TQ2_0 16 — and a row is its scale prefix (the row's f16 scales,
  padded to a dword) followed by that many words per block (IQ1_M keeps
  its scale words in the payload; it has no f16 field).
  `coopNeedsRepack` is replaced by the one de-interleave every format
  with an f16 field takes at upload. The decode kernel of §6 reads the
  same array (1.4).
- **7.4** When a route record prints, each type has its own bit in
  `Linear.coopRouteBit`, so "print once" is per type; today every type
  past Q3_K shares one bit.
- **7.5** When the GEMM is compared with the host GEMM, it matches to
  the bar the IQ4 coop kernels pass.
- **7.6** When an expert slab in one of these formats is staged for a
  MoE window, the same repack and the same kernels serve it.

## 8. Ternary formats and the BitNet architecture

- **8.1** When `general.architecture` is `bitnet`, `ModelConfig.fromGguf`
  accepts it, reads hparams under `bitnet.`, and the graph is llama's
  with an RMS `attn_sub_norm` on the attention output before
  `attn_output`, an RMS `ffn_sub_norm` (size n_ff) on silu(gate)·up
  before `ffn_down`, NEOX rope, and SwiGLU — llama.cpp's
  `src/models/bitnet.cpp` graph, not the model card.
- **8.2** When `output.weight` is absent, the lm_head reuses
  `token_embd.weight`.
- **8.3** When a `blk.N.<proj>.scale` one-element tensor is present,
  the projection's output is multiplied by it; when absent, by
  nothing. Files from llama.cpp's converter carry none — the scale is
  folded into the ternary values and lands in each block's d.
- **8.4** When a TQ file is loaded, its `token_embd` (Q4_K from
  `llama-quantize`, F16 from the converter) and output (Q6_K, or tied)
  route as today, and the ternary tensors route through §6 and §7.
- **8.5** When a ternary checkpoint is benchmarked, the reference is
  llama.cpp's CPU number on the same file, and the honest bar is the
  bandwidth ceiling: TQ2_0 reads less than half of Q4_K_M's bytes per
  weight and should decode faster than the Q4_K_M of the same model.
- **8.6** When the converter is pointed at a `BitnetForCausalLM`
  checkpoint directory (config.json, sharded safetensors,
  tokenizer.model), it writes one `bitnet` GGUF that llama.cpp loads:
  `general.architecture = bitnet`; `bitnet.context_length`,
  `embedding_length`, `block_count`, `feed_forward_length`,
  `attention.head_count`, `attention.head_count_kv`,
  `attention.layer_norm_rms_epsilon`, `rope.dimension_count`,
  `rope.freq_base` from config.json; `rope.scaling.type = linear` with
  factor 1.0 as llama.cpp's converter writes; and
  `tokenizer.ggml.model = llama` with `tokens`, `scores`, `token_type`,
  `bos_token_id`, `eos_token_id` from `tokenizer.model` through
  `SpProto`.
- **8.7** When a projection weight (q, k, v, o, gate, up, down) is
  converted, it first goes through BitNet's `weight_quant` in f32 —
  scale = mean|w| clamped at 1e-5, w' = clamp(round(w/scale), −1, 1) ×
  scale — and then, at outtype tq1_0 or tq2_0, into TQ blocks whose d is
  the block's absmax (the scale, or 0 for an all-zero block). No q/k
  permute: `bitnet` ropes NEOX, and llama.cpp's converter does none.
- **8.8** When a tensor is `token_embd.weight` at a ternary outtype, it
  is written F16; norms and every 1-D tensor F32; at outtype f16 every
  2-D weight is F16 after `weight_quant`. This is llama.cpp's policy
  for the same outtype (`conversion/base.py`), so the two files compare
  tensor by tensor.
- **8.9** When our f16 file is given to `llama-quantize` for TQ1_0 and
  TQ2_0, ggml's packer over our ternary × scale values produces
  projection tensors byte for byte equal to ours. That comparison is
  the packers' bit-exact test, beside the fixture test of §3.
- **8.10** When llama.cpp loads the converted file, its CPU greedy
  tokens on a fixed prompt are the reference of 9.4, and the engine on
  the same file matches them.

## 9. Arbiter checkpoints and evidence

- **9.1** When the IQ set is needed, it is made on this box:
  `llama-imatrix` over the Q8_0 8B on a calibration text, then
  `llama-quantize --imatrix --allow-requantize` to each of IQ1_S,
  IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S and
  IQ3_M — ten files, every ftype mix — parked under `tmp/` like the
  IQ4 set; and an IQ3_XXS Qwen1.5-MoE-A2.7B from its Q4_K_M with a
  fresh imatrix, for the expert route (7.6).
- **9.2** When the ternary set is needed, `1bitLLM/bitnet_b1_58-3B`
  (f32 safetensors in three shards, `BitnetForCausalLM`, SiLU, tied
  embeddings, 26 layers of 3200 × 8640, a 32002-entry LLaMA-2
  SentencePiece vocab — the family llama.cpp's `bitnet` graph matches)
  is converted with our converter at tq1_0 and tq2_0, and its f16
  output is requantized by `llama-quantize` for the cross-check (8.9).
  Microsoft's 2B-4T (`BitNetForCausalLM`, relu²) stays out until
  mainline llama.cpp carries it.
- **9.3** When parity is measured, it is `leg.sh` on a quiet box, pp512
  / pp2048 / tg against llama.cpp on the same file — HIP and Vulkan for
  IQ, CPU for TQ — announced first, per the standing rule.
- **9.4** When live correctness is measured, it is on the dense 8B —
  greedy agreement with llama.cpp on the same file, and teacher-forced
  perplexity within the routing-flip floor of `llama-perplexity` on
  the same text and window — not on a mixture, per the prefill-routes
  11.2 finding (the routing-flip floor).

## 10. One resident layout for every packed format

- **10.1** When any packed weight is uploaded, its resident layout is
  the file's blocks with the lone scale field — the f16 d, wherever in
  the block it sits, or MXFP4's e8 byte — de-interleaved per row: the
  row's scales first, padded to a dword, then the row's payloads, each
  a whole number of dwords. The row stride is the file's unless the row
  has an odd block count, when it is two bytes longer. A format whose
  block is already a dword multiple (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K,
  IQ4_XS, IQ1_M) keeps the file layout as the degenerate case, with an
  empty prefix.
- **10.2** When one of the six formats that hold two device copies
  today — Q4_0, Q5_0, Q8_0, Q3_K, Q6_K, IQ4_NL — is uploaded, it takes
  the same split, and `coopDev`, `ensureQ6Pad`, `blockRepack2Kernel`,
  `blockPadKernel` and `coopNeedsRepack` are removed, not kept beside
  it. MXFP4 holds one copy today (no GEMM route) and migrates when it
  gains one.
- **10.3** When a decode, coop, widen or wave kernel of one of those
  six reads a block, it reads `(payload, scales)`; each migrated kernel
  passes its existing bit gate against the host oracle before its
  timing A/B, and the A/B shows no decode regression.
- **10.4** When the split runs, it is one device kernel fed by the
  existing mapped-upload stream in bounded chunks, so the peak is
  resident plus one chunk; an expert window splits on upload where it
  repacked before.
- **10.5** When the Qwen3-Coder-30B Q8_0 on this box is loaded, its
  weights are resident at the file's bytes, not twice that, and
  `CAJETA_XPU_ALLOC_TRACE` shows one payload and one scales buffer per
  weight.
- **10.6** When `coopBlockWords` is asked for any format, it is the
  payload's dword stride — the one number the split kernel, the decode
  kernels and the GEMM agree on.

## 11. Acceptance (whole spec)

- Every packed format is resident at its file bytes; the repack
  kernels, the pad copy and `coopDev` are gone; every migrated kernel
  is bit-gated and A/B'd with no decode regression.
- Every one of the nine types loads, decodes bit-exactly, prefills
  `batched` and decodes wave-per-row on HIP, with no `batch-refused` at
  dividing shapes; the filtered suite is green; the grid hash test is
  in it.
- On the ten IQ 8B files: prefill ≥ 1.0× and decode ≥ 0.95× llama.cpp's
  best of HIP and Vulkan on the same file — the bar every earlier
  format met — and perplexity within the floor.
- On the IQ3_XXS Qwen1.5-MoE: every expert tensor routes `batched`
  with no `batch-refused`, and perplexity is within the MoE floor of
  llama.cpp's on the same file.
- On the two TQ files: the converter's projection tensors equal ggml's
  byte for byte, the `bitnet` graph produces llama.cpp's greedy tokens,
  and decode beats llama.cpp CPU.
- Resident device bytes per weight equal the file's for that weight;
  the peak during a load is resident plus one staging chunk; no host
  f32 copy at any point.
- `typeName` complete for the family; the manifest current.

## 12. Decisions (2026-09-15)

- **12.1** Table residency for the decode wave — LDS per workgroup
  (Vulkan's choice; 8 KB for IQ2_S costs occupancy) or global through
  L1 (CUDA's choice; one gather per 8 weights). DECIDED: measure both
  on IQ2_S and IQ1_S, the two largest tables, and choose per kernel.
- **12.2** Packed table forms. DECIDED: generate both the byte-per-value
  and the packed literals from the header once; the test pins both.
- **12.3** The one-copy rule. Q4_0/Q5_0/Q8_0/Q3_K/Q6_K/IQ4_NL hold
  `packedDev` (read by the decode mat-vec) and `coopDev` (read by the
  GEMM) at once, because the repack pads the payload to a dword
  stride. The dedup is the de-interleave of §10: split the scale field
  into its own array and every ggml payload is dword-clean at exactly
  its file bytes — Q4_0 16, Q5_0 20, Q8_0 32, IQ4_NL 16, Q3_K 108,
  Q6_K 208. It is a memory question, not a tokens-per-second one: each
  kernel reads only its own copy today, so the split changes no
  kernel's traffic and removes the 2-byte-per-block padding. It pays
  where the second copy decides fit — a 30B Q8_0 holds 32 GB twice on
  this UMA box — and where it eats the widen budget that admits int8
  prefill copies. DECIDED (Julian: "do it for all; I want
  simplification where we can get it"): one rule for every format, the
  six older formats migrated in this plan, the repack machinery
  removed rather than kept beside the new path. The FORM of the split
  was then decided by measurement, not taken from this text: the first
  cut used two arrays and lost 30 % of Q8_0 decode; the per-row form of
  1.4 restored it (`tmp/cbq/src/.../SplitProbe.cajeta`, four layouts,
  three shapes, alternating arms).
- **12.4** Mixture-of-experts in the measured set. DECIDED: an IQ3_XXS
  Qwen1.5-MoE from the Q4_K_M with a fresh imatrix (9.1).
- **12.5** Ternary arbiter. DECIDED: `1bitLLM/bitnet_b1_58-3B` through
  our own converter (8.6–8.10, 9.2); Microsoft 2B-4T when mainline
  llama.cpp gains relu² and the `BitNetForCausalLM` entry. The local
  torch install does not import (`hipsparselt`), which is one more
  reason the converter is ours.
- **12.6** Decode kernels per format. DECIDED: one, the Q8 wave kernel;
  the f32-activation twins exist only for formats that predate the Q8
  route.
