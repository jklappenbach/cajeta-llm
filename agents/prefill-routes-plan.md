# Prefill routes — plan (Unit 33)

Implements [`specs/prefill-routes-spec.md`](../specs/prefill-routes-spec.md).
Draft 2026-09-06; Julian: "let's fix the issues we know about first".

**Work:** give every quant format a batched prefill route on HIP, make a
batched refusal name itself, remove the three GEMM spills, and bring the
two MoE checkpoints that prefill per-row or time out onto the batched path.
Every route change is gated by the sweep recipe that found the problem.

**Systems:** `model/Linear.cajeta` (`isBatchRouted`, `isBatchRoutedFor`,
`matmulBatchKeep` route chain, `sayRoute`), `model/CausalLM.cajeta`
(`batchReady`, `prefill-mode` diag), `io/QuantKernel.cajeta`
(`coopBatchLaunchNoSync`, `coopColsOk`, `hasBatchKernel`),
`io/WmmaKernel.cajeta` (the Mw8 kernels), `model/ExpertBank.cajeta`,
`DiagRecord.cajeta`; stdlib `cajeta.xpu.KernelManifest` (spill oracle);
the cajeta repo's `tmp/llmbench/leg.sh` + `summary.sh` (the sweep recipe)
and its `rows.jsonl` of 2026-09-06 as the before rows.

**Deliverables:** `batch-refused` diagnostic; coop GEMM route on HIP for
Q2_K/Q3_K/Q5_K/Q8_0 with a padded tail chunk; Mixtral and Qwen1.5-MoE
batched; three kernels at `spillBytes = 0`; a before/after table in this
plan's acceptance and in the bench memory.

---

## Unit 1 — Name the refusal (spec §2)

### 1.1 TDD
- [x] 1.1.1 `DiagTest`: a `Linear` whose format has no batched route on the
      active backend makes `batchReady` emit one `batch-refused` record
      naming layer, projection, format and predicate; a second layer with
      the same (projection, format, reason) does not emit again.
      WRITTEN 2026-09-06; BLOCKED on a cajeta compiler regression: cajeta
      67690686 SIGSEGVs in LLVM RAGreedy (`SplitEditor::deleteRematVictims`,
      fault +0x8) codegen-ing the test module (`--emit=exe --profile=test
      --xpu-backend=cpu`), with or without this test; the Sep 5 compiler
      (cajeta-llama-u34, 0.26.0 46b1337d) builds it. Repro:
      cajeta-llm `tmp/u1/build.log`; per-class `--emit=ir` modules all
      pass `llc -O2 -mcpu=znver5`, so the crashing function is in a module
      that snapshot never sees — `CAJETA_DUMP_CODEGEN_BC` added to name it.
      NAMED + BISECTED 2026-09-06: `opt -passes=verify` on the dumped
      WmmaKernel module: `use of undefined value '%wi.tx'` in the launch
      wrapper's inlined copy of `q4kWmmaDeqEpiKernel`'s block function —
      the work-item latch adds (`wi.next`, `wi.y.next`) still reference the
      block function's original PHIs (a detached/foreign value the inliner's
      map never saw). Reverting cajeta 66041f35 (CpuBarrierFission: latch as
      scaffold after the last barrier — the fix that newly ACCEPTS these WMMA
      kernels) makes the module compile. Fix lands in cajeta.
      FIXED 2026-09-07, cajeta 057f4fe9: the cause was not the inliner but
      the fission walk regioning the `if (t0 < rows && i0 < outDim)` join
      block twice (pre-loop and post-loop region); a barrier loop under
      divergent control flow is now DECLINED by name (the host-stub
      fallback that ran before 66041f35). Suite on that compiler, cpu:
      359 passed / 0 failed / 1 skipped, both DiagTest tests green.
- [x] 1.1.2 `DiagTest`: the `prefill-mode per-row` record carries the
      refusal count in `v1`.

### 1.2 Coding
- [x] 1.2.1 `CausalLM.batchReady` collects the first failing predicate per
      projection through a `Linear.batchRefusal()` string (`"deqPrefill
      off"`, `"no batch kernel for <ty>"`, `"rows % 128"`, `"coop cols"`,
      `"packedTy < 0"`) and emits `batch-refused` once per distinct triple.
- [x] 1.2.2 `DiagRecord` documents the two records; `LoggingDiagCallback`
      prints them.

### 1.3 Acceptance
- [x] 1.3.1 `schedthroughput <Mixtral> prompt=128 gen=1 trace` names the
      refusing tensor(s) — the measured answer to spec §4.1's first half.
      MEASURED 2026-09-06 (bench built with the Unit 1 engine): one record,
      `batch-refused attn_k q8_0: only route is the int8 Mw8 GEMM
      (prefillWeights=int8); prefillWeights=packed` (layer 0, rows 128) —
      Mixtral-8x7B Q4_K_M carries Q8_0 attention keys, so it is the same
      missing route as the 8B Q8_0; Unit 2 closes both.

## Unit 2 — Coop GEMM route on HIP for the formats without a batch kernel (spec §3)

### 2.1 TDD
- [x] 2.1.1 Spike first (no half-measures): force the coop route on HIP for
      Q8_0 and measure `schedthroughput` prefill at 512 on the 8B Q8_0
      against the 13.2 tok/s before row. The unit proceeds only if the
      spike is batched and faster; if the coop kernels misbehave on HIP the
      finding is recorded here and Unit 2 re-plans around the int8 Mw8
      route instead.
      SPIKE 2026-09-06 (`schedthroughput <8B Q8_0> prompt=512 gen=32 trace
      coophip`, bench built with cajeta 67690686): the route ENGAGES —
      `prefill-mode batched 128`, `batch-route coop ty=8 4096 4096` — and the
      GPU faults: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on the ROCm
      queue, after which every number the process prints is garbage (first
      token 0). The coop kernels have never run on HIP before; the Vulkan
      driver would have hidden an out-of-range read (robust buffer access),
      HIP does not. Next: `CoopQuantGemmTest` on gfx1151 (the kernels' own
      parity tests) to split kernel indexing from the AMD `Tile.load` lowering.
      RESULT: the whole selftest suite built with the Sep 5 compiler runs
      on gfx1151 at 359 passed / 0 failed / 1 skipped, every
      `CoopQuantGemmTest` (Q8_0, Q2_K, Q3_K, Q5_K, Q4_0, Q5_0, and the
      non-256 widths) matching the host — the coop kernels and the AMD tile
      lowering are sound at the test shapes. The fault is in the ENGINE's
      HIP plumbing of the route (repack `wordView`, f16 staging, pad rows,
      or a missing sync) or a shape the tests never reach (128x4096x4096).
      SERIALIZED (`AMD_SERIALIZE_KERNEL=3`): NO fault, first token 77 = the
      per-row baseline's, prefill 512 in 3169 ms = 161.6 tok/s (12x the
      per-row 13.2) even with a device sync after every one of ~3300
      kernels — the coop route on HIP is CORRECT and fast; the fault is a
      RACE (a queued kernel against a buffer's lifetime or an unfinished
      transfer), deterministic unserialized (2/2), gone under
      `AMD_LOG_LEVEL=4` alone (dispatch slowed enough).
      BRACKETS (bench arms, 2026-09-07 00:xx): `pfsync` (sync at phase marks)
      → still faults; `coopsync1` (sync right after `ensureBtXh`, before the
      GEMM) → still faults; `coopsync2` (sync right after the GEMM) → the
      process printed nothing (aborted). So the fault is raised BY the coop
      GEMM dispatch (or its repack) itself when it is not preceded by a
      device-wide sync — its inputs at launch time are the suspects: the
      `packedDev` the repack reads (weight prefetch stream? handle not yet
      assigned?), `coopW` = `coopDev.wordView()`, `btXh`. Under serialization
      the same launch computes the right tokens.

      NIGHT 2 (2026-09-07, after the compiler fix landed as cajeta 057f4fe9)
      — CORRECTED: the "race" is HOST HEAP CORRUPTION on the coop route.
      Five runs of `tmp/llmbench-spike/schedthroughput <8B Q8_0> prompt=512
      gen=2 trace coophip` (same args) gave three different failures: (a)
      four runs exit 0 with every later launch saying `no registered kernel
      'gluF32' / 'q80F16CoopX1Kernel'` and first token 0 — the kernels ARE
      in the binary (`strings` shows both `.kd` descriptors; a first
      "tree-shake pruned them" reading was wrong), so the registry lookups
      themselves went bad; (b) two of those runs also page-faulted on the
      GPU (dmesg 00:24:08 / 00:24:14, client TCP, a READ of
      0x576a86f02000 — a host-heap-shaped address, i.e. a kernel handed a
      corrupted pointer); (c) the fifth run: host SIGSEGV, exit 139, fault
      addr 0x80000, symbolized against the binary:
      `__cajeta_string_drop_claimed` ← `Diag.emit` +0x11c ← `Linear.sayRoute`
      ← `matmulBatchKeep` (coop branch) — an owned string dropped inside
      `Diag.emit`, right after `prefill-mode batched` and BEFORE any coop
      launch. Heap-layout dependence is why `AMD_SERIALIZE_KERNEL=3` and
      `AMD_LOG_LEVEL=4` "fixed" it last night. Leading suspect: `sayRoute`'s
      coop call is the one that passes a HEAP TEMPORARY name (`"coop ty=" +
      this.packedTy`; dotAccum at Linear:2166 is the other), and `Diag.emit`
      does `r.name #= name` into its static scratch record and drops the
      previous value on the next emit — measure whether the temp's title is
      both tendered to the callee AND dropped by the caller (double drop),
      or a dangling borrow is later dropped as owned. `DiagTest` passes the
      same shape on CPU, so it is layout-dependent; write the probe as a
      Diag test that emits two heap-temp names back to back under a
      callback. GPU is HEALTHY (AttentionTest device tests pass, no new
      fault during the crash run). Still true: the coop route needs
      `outDim % 128 == 0` and `cols % 64|256 == 0`, so the [out=8,in=256]
      fixtures never engage it. The coopsync1/2 bench arms in cc05c21 set
      flags with NO use site in Linear (dead) — ignore their earlier
      "brackets".
      MEASURED (2026-09-07, host, no GPU): (1) `Diag.emit` is the VICTIM —
      `DiagTest.emitWithHeapTemporaryNamesLeavesTheHeapIntact` relays 256
      heap-temporary names through a plain param with same-size-class bait
      allocated between emits: every bait intact, live count balanced
      (360/0/1 both profiles, commit 457d4cf). Mechanism read in the IR +
      runtime: the caller passes the temp with transfer word 0 and drops it
      right after the call; the callee's `#=` on a String field routes a
      lend through `__cajeta_string_resolve`, which returns a FRESH wrapper
      the field owns (owned copy for len <= 256, shared stake above), then
      sets the own-bit — safe. (2) `#=` on a CLASS field from a lent param
      (`probe.Q.keep` IR): reads the arriving title flag, displaces the old
      value only if it was owned, stores, and sets own-bit = title flag —
      i.e. records a BORROW for a lend, exactly CLAUDE.md §2.3. A static
      `#=` from a param (`probe.Q.park`) just stores the pointer. The
      caller frees the object at scope end, so `Linear.btXhSrc #= src`
      (stageBatchFromDevice) DANGLES if `src` dies before the coop
      route's `ensureBtXh` reads it; the callers pass `this.pfXnDev`,
      `dkv.prefillAttendResident(...)` (= the DeviceKv FIELD `aoDev`, long-
      lived) and `d.mlp.gateProj.yBatch` (a Linear field that
      `ensureBatchOut` can REALLOCATE — check that growth path). (3) The
      bench builds cleanly THROUGH run-tests.sh's enumeration
      (`tmp/u4/build-bench.sh`: its prologue + `--emit=cja` + `--emit=exe
      --tree-shake=off --xpu-backend=amdgpu`), which is the order-dependence
      of the ownership check demonstrated by construction; note the copied
      prologue's `trap rm -rf $out EXIT` — delete it or the exe vanishes.
      (4) The 2026-09-06 leg logs and the working spike runs never printed
      `no registered kernel`; it was new to tonight's four runs.
      NAMED (2026-09-07, fresh bench = run-tests enumeration + tree-shake
      off, cajeta 057f4fe9): `no registered kernel` is NOT corruption — the
      runtime resolves kernels LAZILY (`hipModuleGetFunction` at first
      launch, cajeta_xpu_launch.c) and ignores HIP return codes; after a GPU
      fault every HIP call returns `hipErrorIllegalAddress`, so every
      not-yet-resolved kernel prints that line and the process exits 0 with
      garbage. With `AMD_LOG_LEVEL=4 AMD_SERIALIZE_KERNEL=3` (`tmp/u4/coop-
      log4.log`) the dispatch order before the first error is: blockRepack2
      → f32ToF16 (grid 2048 = 128x4096) → q80F16CoopX3Kernel (grid 32 =
      the 4096x4096 **o_proj** GEMM, first use) → addF32 launch returns
      hipErrorIllegalAddress. The q/k/v GEMMs (same kernel, same shape) ran
      fine just before. The kernel log for that pid: READ faults (client
      TCP) walking HOST-HEAP pages 0x5fe7c2551000..0x5fe7c2567000 — the
      region holding the process's HIP objects (its stream is at
      0x5fe7c2574740). So the o_proj coop launch marshals a HANDLE that is
      a host pointer: a KernelBuffer wrapper freed and its memory reused
      by libamdhip, read back as `deviceHandle` at launch — layout- and
      timing-dependent by nature, and it faults SERIALIZED too now. Diag
      trace `coop-args` (bench arm `coopargs`) prints the four handles +
      lengths per coop launch to name the argument.
      ROOT CAUSE (2026-09-07, measured in IR, `cajeta/tmp/probe-emit/src/probe/T.cajeta`):
      a local bound from a TERNARY of a borrowed field and a literal —
      `String nm = r.name != null ? r.name : "-";` — gets an ARMED drop entry
      (`__cajeta_drop_push … __cajeta_string_drop`) and frees the field's
      wrapper at scope end; `String nm = r.name;` gets none. cajeta's
      LocalVariableDeclaration classifies an initializer as a borrow only for
      the shapes it recognises (literal, identifier, field read, array
      element, plain-return call); a BooleanSwitchExpression matches none and
      defaults to OWNED. The bench's `StdoutTrace.onRecord` (trace arm) and
      the CLI's `LoggingDiagCallback.cajeta:102` both have that exact line,
      so every diag record under trace/debug logging double-frees the scratch
      record's name wrapper; the block is reused (a KernelBuffer wrapper,
      an arena tensor carrying 524288 = 128x4096 — the deterministic fault
      addr 0x80000), and the next emit frees the impostor. THAT is the whole
      "GPU race": a freed KernelBuffer wrapper's handle read back as a host
      pointer. Same class as [[kernel-ternary-mislowers]], host side. Fix in
      the COMPILER (ternary arms classified; mixed arms carry a runtime title
      flag like flagged calls); the two cajeta-llm lines are legal as written.
      FIXED IN CAJETA (2026-09-07, `fix(ownership): a local bound from a
      ternary owns exactly what the taken arm produced`, on main; tests
      TernaryOwnershipTests 4/4, 48 ownership suites 288/0/2). VERIFIED HERE:
      bench rebuilt with that compiler (`tmp/u4/build-bench.sh`, tree-shake
      off), `<8B Q8_0> prompt=512 gen=8 trace coophip`: first token 77 both
      SERIALIZED and UNSERIALIZED, no GPU fault (kernel log clean), no `no
      registered kernel`, no SIGSEGV (`tmp/u4/fix-ser3.log`, `fix-unser.log`).
      Unserialized: prefill 512 in 2013 ms = 254 tok/s, decode 27.6 tok/s —
      INDICATIVE ONLY (the cpu suite ran beside it; box not quiet); the
      per-row row of record is 13.2 tok/s. The coop route on HIP is correct;
      2.2.x (structural default) and 2.1.2–2.1.5 can proceed; 2.3.1 is the
      announced acceptance leg. The `coopargs` arm + `coop-args` Diag record
      (Linear/SchedThroughput) stay as a launch-handle instrument.
      LATER THE SAME NIGHT: a direct `cajeta --emit=cja` of the library
      (same fixed compiler, same sources, same classpath as run-tests.sh)
      FAILS with `CAJETA_ERROR_OWNED_RESULT_NEEDS_TRANSFER` at
      SafetensorsFile.cajeta:199 (`t = Tensor.zeros<float32>`), :214/:221/
      :228 (`t = this.loadF16/Bf16/F32(name)` in the *Device loaders), and
      after those are spelled `#=` a further `this.x = Tensor.zeros(...)`
      site — while run-tests.sh builds the identical library CLEAN (its
      log, line 10 → 359 passed). The ownership checker is ORDER-DEPENDENT
      (a cajeta defect: false negatives in the default order); the sites
      are genuine Producer results bound with `=`. Not fixed tonight — a
      partial one-file migration was reverted; needs a focused sweep with
      the compiler as oracle. Consequence for the instrument: build the
      bench through run-tests.sh's enumeration (or fix the sweep first),
      with `--tree-shake=off` so the coop kernels survive.
- [x] 2.1.2 `LinearKernelRouteTest`: on gfx1151, a Q8_0 / Q2_K / Q3_K /
      Q5_K `Linear` built from the `kquant/` fixture blocks reports
      `isBatchRoutedFor(128) == true` with `prefillWeights=packed`, and the
      batched output matches the per-row (forced serial) output within the
      coop route's Vulkan tolerance, per format.
- [x] 2.1.3 `LinearKernelRouteTest`: Q4_K and Q6_K still take their native
      int8 routes (`batch-route q4 plain` / `mmq q6`); the coop route is not
      chosen where a native kernel exists.
- [x] 2.1.4 `ForwardTest`: a 200-token prompt (128 + 72 tail) prefills
      `batched` end to end — the tail chunk is padded, not sent per-row —
      and the logits of the last real row equal the unpadded per-row
      logits within tolerance.
- [x] 2.1.5 `EngineTest`: `prefillWeights=int8` still selects the Mw8
      routes for the formats that have them (spec §3.4 — nothing regresses).

### 2.2 Coding
- [x] 2.2.1 `Linear.isBatchRouted`: open the coop branch on every backend
      (not only Vulkan) for formats where `!hasBatchKernel(ty)`, behind
      `coopColsOk` and `outDim % 128`; `isBatchRoutedFor` stops requiring
      `deqPrefill` for those formats when the coop route is open.
- [x] 2.2.2 `matmulBatchKeep`: the coop branch is reachable on HIP;
      `sayRoute` records `coop <fmt>`; the f32→f16 activation conversion and
      the coop scratch (`yBatch`, staging) are allocated on the HIP device
      path exactly as on Vulkan.
- [x] 2.2.3 Tail padding: `CausalLM.forwardRowsBatched` (or the chunk
      planner) rounds the last chunk's rows up to the route's tile
      (`fitsMw8`/coop tile) with zero rows, and the epilogue ignores them.
- [x] 2.2.4 `EngineOptions` doc: `packed` now batches every format; the
      `int8` note names the Mw8 route as the alternative, not the only one.

### 2.3 Acceptance
- [x] 2.3.1 Sweep legs (`leg.sh cajeta <8B Q2_K|Q3_K_M|Q5_K_M|Q8_0>
      512x128 3` and `2048x64 3`): `prefill-mode batched`, prefill tok/s
      ≥ 0.17x the best llama.cpp row for each, decode within noise of the
      2026-09-06 rows. Rows appended to the bench memory table.
      PASS 2026-09-12 (amdgpu/gfx1151): prefill 512 tok/s Q2_K 926 (0.82x
      llama), Q3_K_M 423 (0.31x), Q5_K_M 558 (0.42x), Q8_0 654 (0.72x,
      3-rep median; rep-1 cold outlier 263). All batched; all ≥ 0.17x. Was
      per-row 13–18 tok/s → 25–137x over old cajeta. Rows in rows.jsonl.
- [x] 2.3.2 Teacher-forced perplexity (`PplProbe`) on the 8B Q8_0 within
      noise of the per-row run.
      PASS 2026-09-12: coop route ppl 7.467 (meanNll 2.0105) vs nocoop
      control 7.448 (2.0079); Δ 0.25%, within f16-accum noise. Coop prefill
      2.88 s vs nocoop 89.75 s (31x). Corpus tmp/u3/ppl-corpus.txt.

## Unit 3 — The MoE checkpoints (spec §4)

### 3.1 TDD
- [ ] 3.1.1 `MoeForwardTest` on a fixture whose expert format had no
      batched route: prefill is `batched`; expert-group dispatch records
      `device` for the groups the budget admits.
      NOTE (2026-09-12): no checked-in MoE fixture witnesses this —
      `toy-moe.gguf` and `toy-routable-moe.gguf` both carry Q4_K experts,
      which always routed. Build a Q8_0-expert toy MoE (the real
      Qwen1.5-MoE `ffn_down_exps` is Q8_0 — the exact Unit-2 case). The
      `records device` half is amdgpu-only (cpu has no device grouped
      path); on cpu assert prefill `batched` and skip the device half.
- [x] 3.1.2 A load-time test on `toy-moe.gguf`: `LlmEngine.load` emits a
      phase breakdown under `trace` (`load-phase` record: open / bind /
      pack / warm-up / total nanos), so a slow load has a named component
      before it is fixed. `LoadPhaseTest` GREEN on cpu; read-vs-upload
      split inside `bind` is deferred to 3.2.2's profiler pass.

### 3.2 Coding
- [x] 3.2.1 Fix whatever Unit 1's diagnostic names on Mixtral (expected:
      an attention projection format without a HIP route, closed by
      Unit 2 — verify, do not assume).
      VERIFIED 2026-09-12 (amdgpu/gfx1151, no fix needed): Mixtral-8x7B
      Q4_K_M now prefills `batched 128 0` (zero refusals). The previously
      refusing `q8_0` projection takes Unit 2's coop route
      (`batch-route coop q8_0 1024 4096`). Exactly as predicted. Load 22.2 s.
- [x] 3.2.2 Qwen1.5-MoE: profile the load (60 experts × 24 layers of small
      slabs; shared expert; `attn_*.bias`) and remove the component that
      scales with expert count rather than bytes; then the 512 prefill.
      RESOLVED-BY-REALITY 2026-09-12: the 31.6 s→5.4 s improvement already
      landed (prior coop/residency work). No expert-count component remains
      — bind is bytes-proportional (0.57 s/8.8 GB ≈ 15 GB/s warm);
      cross-checked against Mixtral (load scales with bytes 26→22 s, not
      experts 8 vs 60). Residual is the deliberate warm-up prefill (cold
      4.36 s vs warm 2.52 s). No fix warranted; a warm-up cut would only
      move cost to first inference. 512 prefill confirmed batched at 3.3.1.
      Instrument: the `load-phase` record (3.1.2) names the coarse phase;
      build the engine `--profiler=instrument` and read the headless
      `TraceSummary` for the per-slab method scaling 60×24 (cajeta-profiler).
      FINDING 2026-09-12 (amdgpu): the 31.6 s load is GONE — Qwen1.5-MoE
      now loads in 5.4 s and prefills `batched 128 0`. load-phase nanos:
      open 0.257 / bind 0.574 / pack ~0 / warm-up 4.359 / total 5.19.
      The expert-count blowup is NOT in bind (0.57 s ≈ 15 GB/s, bytes-
      proportional); load is now dominated by the deliberate warm-up
      prefill (standalone prefill(128) = 2.52 s). Profiling the warm-up
      to confirm no per-expert component hides inside it before closing.
- [~] 3.2.3 Qwen2.5-VL-72B Q4_K_L: with Unit 2 in place, confirm every
      tensor format routes; fix the remaining refusal if the diagnostic
      names one.
      BLOCKED 2026-09-12: no Qwen2.5-VL-72B on the box (~40 GB download,
      not started unprompted). Mixtral (8 experts) + Qwen1.5-MoE (60
      experts) already witness mixed-format MoE routing; the 72B adds the
      Q4_K_L mixed Q4/Q5/Q6/Q8 dense-attn case. Needs the model fetched.

### 3.3 Acceptance
- [~] 3.3.1 Sweep legs for Mixtral, Qwen1.5-MoE (load ×3 + 512x128) and
      the 72B (512x128 ×1): no per-row, no timeout, load ratio vs
      llama.cpp recorded.
      PASS 2026-09-12 for the two available models (72B blocked, see 3.2.3):
      Mixtral 512 prefill 130 tok/s batched (was per-row 16.9), load 14.5 s;
      Qwen1.5-MoE 512 prefill 71 tok/s batched (was timeout), load ~4.3 s
      (×3 stable, was 31.6 s). Neither per-row, neither timeout. Load ratios
      recorded in rows.jsonl. Blocked only on the 72B leg (no model).

## Unit 4 — No shipped GEMM kernel spills (spec §5)

### 4.1 TDD
- [ ] 4.1.1 `QuantKernelTest`: `KernelManifest.of("q4kWmmaDeqMw8Kernel")`,
      `("q2kWmmaDeqMw8Kernel")`, `("q6kF16CoopN256GKernel")` report
      `spillBytes == 0` on gfx1151 (skips where the backend has no
      footprint); the same test lists every registered GEMM kernel and
      fails on any non-zero spill, so the check cannot silently narrow.
- [ ] 4.1.2 Each kernel's existing correctness test still passes
      bit-for-bit (they are int8/f16 tile kernels with exact references).

### 4.2 Coding
- [x] 4.2.1 RESOLVED 2026-09-13 by the 2x4 wave re-shape, not by a despill:
      the kernel now reports vgpr=211 spill=0 and is FASTER (688 -> 877),
      so the trade below never had to be made.
      `q4kWmmaDeqMw8Kernel` (256 VGPR, 124 B): cut live registers —
      the eight persistent f32 accumulators (~64 VGPRs) are the named
      price; drain half per chunk or narrow the N tile — ISA-verified
      (`cajeta --xpu-emit=isa`, `vgpr_spill_count = 0`).
      FINDING 2026-09-12 (REVERTED): cutting int32 accumulators 4→2 (four
      2-tile sub-chunks) DID reach spill=0, but REGRESSED prefill 639→581
      tok/s (−9%): halving accumulators forced 2× more weight-fragment
      reloads, which cost MORE than the spill traffic removed. For this
      kernel the 124 B spill is CHEAPER than despilling — it is near its
      register/reload optimum at 639. Reverted (kept the spilling 639
      version). A tile-narrowing despill (fewer facc without more reloads)
      would change the tile + launcher + the 6.1.1 reference — deferred.
      Lesson: spill≠slow; the despill must not trade spill for reloads.
- [x] 4.2.2 RESOLVED 2026-09-13 by the same re-shape: spill=0, LDS unchanged.
      `q2kWmmaDeqMw8Kernel` (256 VGPR, 108 B, 25 KB LDS): same
      treatment; LDS is its occupancy limiter, so the register cut must
      not move work into LDS.
- [ ] 4.2.3 `q6kF16CoopN256GKernel` (192 VGPR, 68 B): the coop route
      Unit 2 puts on HIP — fix before Unit 2's acceptance leg on Q6_K-
      bearing formats, or record that Q6_K keeps its native route.

### 4.3 Acceptance
- [ ] 4.3.1 Before/after duration on each kernel's own route (`MmqProbe` /
      `DecodeProbe` shape it runs at), not worse beyond noise; prefill
      tok/s row for the routes that use them.

## Unit 5 — Whole-spec acceptance (spec §6)

### 5.1 TDD
- [ ] 5.1.1 `run-tests.sh` green on CPU and gfx1151.

### 5.2 Coding
- [ ] 5.2.1 Update `llm-vs-llamacpp-bench-2026-09-06` memory and the cajeta
      repo's bench page with the after rows; archive this plan.

### 5.3 Acceptance
- [ ] 5.3.1 The 2026-09-06 sweep recipe rerun: no `per-row` on any
      checkpoint; every prefill ratio ≥ 0.17x; decode within noise; load
      not worse. Table in this section.

## Unit 6 — Multi-wave packed Q4_K GEMM for prefill parity

NEW ARC opened by Julian 2026-09-12 ("optimize to reach parity or beat").
This is the GEMM arithmetic-intensity work spec §1.3 deferred — now IN scope.
Diagnosis (profiler, amdgpu/gfx1151): `q4kWmmaKernel` is 81% of Q4_K prefill
device time (q6kWmmaEpiKernel 14.5%, attn 1.3%). Root cause: SINGLE-WAVE
16×16 tile (32 threads/wg) with a per-sub-block barrier-bracketed scalar
epilogue (~16 barriers/block) — the int8 matrix cores drain and idle on
scalar f32 scaling, no sibling waves to hide it. llama `mul_mat_q` uses
`MMQ_NWARPS 8` on a large tile, int32 in-register, scales folded. The code's
own note (Linear.cajeta:1224) already defers the win to "multi-wave tiling";
the register-accum epi variant (q4EpiOn) measured a wash. Q4_K_M 218 tok/s vs
llama 1320 = 0.165x.

Design decisions (Julian 2026-09-12):
- PORTABLE GEOMETRY: kernel reads wave width via `Group.width()` (compile-time
  fold, 32/64), never a literal 32 / `globalIdX()/32`. Launcher derives
  block/grid from `Device.*` (waveSize/simdCount/dispatchBlocks/
  sharedBytesPerBlock/maxThreadsPerBlock), mirroring `Linear.targetBlocks()`,
  with a measured-literal fallback on any 0-return. Tile MATRIX shapes stay
  compile-time type params (no device shape enumeration); the multi-wave tile
  must be correct for both wave widths.
- SCHEDULER DEPLOYMENT (shipped path): launcher reads `kernel.manifest()
  .feasibleBlocks()[0]` (compiler occupancy-optimal block) + `occupancyLimiter`
  /`residentGroupsPerCu` diagnostics; grid from `Device.dispatchBlocks`; routes
  through `Scheduler.submit()`+`bind()` for forward-compat + access-set
  validation. The auto-scheduler (submit→resolved launch) is unbuilt (9/135);
  this is the shipped occupancy-resolution surface.

HYPOTHESIS CONFIRMED 2026-09-12 (zero new kernel): routing Q4_K_M prefill
through the EXISTING multi-wave kernel via the int8-deq path (`deq` bench flag
→ `q4 deqMw8Part`, q4kWmmaDeqMw8Kernel) = 636.8 tok/s vs packed single-wave
217.8 = **2.9×**, 0.165x→0.48x llama. Multi-wave tiling IS the lever. That
kernel still spills 124B (headroom). 6.2.1 = the same multi-wave win on the
PACKED route (inline nibble widen, no 2× int8 weight memory). Template =
q4kWmmaDeqMw8Kernel (8 waves/wg, 128×128 tile, register accumulators facc0-7,
pre-folded cf/cg, rowBase grid-sweep).

PROGRESS 2026-09-12: `q4kWmmaMwKernel` + `q4kWmmaMwLaunch` + `setQ4MwPacked`
flag (route bit 2^29 "q4 mw") written (WmmaKernel.cajeta:1492/1808,
Linear.cajeta:2089) and BIT-CORRECT (6.1.1 PASS, matches deq-Mw8 exactly).
Measured 8B Q4_K_M prefill 512: 341 tok/s = 1.57x over single-wave 218
(0.165→0.26x llama), packed memory. BELOW the deq path's 637 — occupancy-
bound: the inline-widen `bt` is 32 KB LDS (~46 KB/wg → ~1 WG/CU) and the
kernel spills 236 B (vgpr=192).
FINDING 2026-09-12: two-k-half widen (bt 32→16 KB, ~2 WG/CU, bit-gate still
PASS) gave only 341→357 tok/s — the packed path is NOT occupancy-bound, it is
COMPUTE-bound on the per-launch inline nibble-widen (the +2 barriers/block
offset the occupancy gain). The deq path's 639 comes from NOT widening per
launch (pre-dequantizes once at prewarm). So the packed path is capped near
~639 by widen cost; closing it needs a cheaper widen (vectorized lut4/v_perm,
cf [[mxfp4-kernel-perf-decode-bound]]), not more occupancy. PARITY ROADMAP:
(1) multi-wave DONE (218→357 packed / 639 deq); (2) despill the DEQ kernel
(4.2.1, 124 B; despill historically +55-70%) → ~1000+, the fastest parity
lever; (3) vectorized packed widen → 357 toward 639 at packed memory; (4)
deeper GEMM AI tuning toward 1320. Reaching true parity (1320, 6x) is
multi-iteration.
LLAMA STUDY 2026-09-12 (decisive; reframes the roadmap): tile (128×128, 8
wave32), int8 WMMA 16×16×16, 64 f32 acc/thread, and the d·sc/-dmin·m epilogue
are ALL already identical to llama's mmq. The 2× gap is WEIGHT-READ BANDWIDTH:
llama reads Q4_K at native 4.5 bpw and expands nibbles IN-REGISTER
(`load_tiles_q4_K`, mmq.cuh:2120); our deq path reads an int8 copy at 8.5 bpw;
4.5/8.5 = 0.53x ≈ measured 0.48x. llama does NOT double-buffer/async on AMD
(sync K-loop, mmq.cuh:3485-3518); it hides latency via `__launch_bounds__
(256,2)` = 2 blocks/CU. RANKED LEVERS: (1) feed the B fragment from nibbles via
`CooperativeMatrix.fromWords` (proven in q4kWmmaIdMwKernel :1085-1160, "4 loads
+ 4 ALU"/half, no LDS/barrier) into q4kWmmaMwKernel — reads packed 4.5 bpw,
drops the deq copy; expected ~2x → ~1300. (2) stage activations to LDS once
with a LITERAL stride — all 8 waves re-read the same A fragment from global (8x
redundant) AND the runtime `xRowBytes` stride degrades every A load to 16
byte-loads. (3) AsyncCopy prefetch NOT viable — inert on gfx1151 (no LDS-DMA in
RDNA3 silicon, AmdgpuKernelLowering.cpp:106-113). (4) retry @Occupancy
(minResident=2) after (1)/(2) lighten registers. (5) keep both K-halves' mbA
resident.
PROGRESS 2026-09-12 (e25805b): RANK 1 applied — q4kWmmaMwKernel B-feed is now
`fromWords` from packed nibbles (bt tile/widen loops/widen barriers gone, flat
j-loop, 2 barriers/block, LDS 12.5 KB, +KernelBuffer<int32> packedW param via
ensurePackedW). 6.1.1 bit-gate PASSES. But prefill stayed 359 — three B-feeds
(LDS widen 341 / two-k-half 357 / fromWords 359) all ~360 ⇒ the B-feed was
NOT the packed bottleneck. LAUNCHER WAS: q4kWmmaMwLaunch was the only launcher
slicing at dispatchBlocks(8)≈20 WGs (one wave/SIMD, under-fills); every tuned
launcher uses partWgs=32. Switching to partWgs → 416 tok/s (+16%). RESIDUAL vs
deq 637: fromWords = 8 scalar word-loads + ~12 ALU per B fragment vs deq's ONE
wide <4 x i32> load ⇒ packed-mw is INSTRUCTION-bound at 4.5 bpw, deq is
BANDWIDTH-bound at 8.5. NEXT: wide-load the 8 packed words per j-pair
(Vector<int32,4>×2) and reuse across even/odd nibble + both halves (16× fewer
B loads at 4.5 bpw); then RANK 2 (A-side LDS staging, literal stride).
6.3.2 CAVEAT: the naive dispatchBlocks saturation law under-fills ~16%; the
measured 32-WG slice needs a derivable oversubscription factor (test 2×).
PROGRESS 2026-09-12 (2b2fab4): vload B-feed (8× dwordx4/block, reuse across the
j-pair) → 431 tok/s (+3.5%), bit-gate PASS. RANK 2 (A staged in LDS, literal
stride 144, 4 barriers/block) WRITTEN and ran 512 (+19%, spill 204 B) but
FAILS the bit-gate (element 0 = 1.4e6 vs 0.3 — a scale-sized error; prime
suspect the folded staging barrier) → 512 is VOID; fix in flight, not
committed. MODEL CORRECTION: measured on two BIT-CORRECT kernels, packed
(4.5 bpw) 431 < deq (8.5 bpw) 637 — halving weight bytes gave ~0%, so the
family is INSTRUCTION- and LATENCY/OCCUPANCY-bound, NOT bandwidth-bound; the
study's 0.53x ratio was coincidental. Roadmap: deq+A-staging is the SPEED
base (no fromWords VALU), packed-mw the MEMORY base; next levers = partWgs
sweep (32 caps WGs in flight <2/CU on 40 CUs) + @Occupancy(minResident=2),
then port A-staging to deq-Mw8.
PROGRESS 2026-09-12: RANK 2 FIXED, bit-gate PASS → 507 tok/s (+18% over 431;
deq control 637 unchanged in the same alternating A/B, prompt=512). The
bit-wrong cause was NOT the barrier: the LDS tile `Shared<int8> at` shadowed
the deq template's `int64 at` (the rg-staging q8 partial-sum address) — blocks
share the method scope map, so a differently-typed shadow ships corrupt IR
silently (second instance of the known compiler defect). Renamed `aTile`; the
explicit staging barrier is restored (5/block) until a fold is proven. Spill
204 B at vgpr=192 (the LDS budget lowered the VGPR cap). Kernel now: B at 4.5
bpw via fromWords + A staged once per block in LDS (30.5 KB, literal stride
144, wide fragment loads). NEXT: partWgs sweep (32/40/64/80, both arms).
PROGRESS 2026-09-12 (9ab9739): barrier fold PASSES the gate (the shadow was the sole
cause; 4 barriers/block, gain within noise). partWgs SWEEP, two replicates
≤1% apart: packed-mw 32/40/60/64/72/80 → 510/534/557/550/555/555; deq →
638/688/633/628/693/690 (48 → 591). The pattern is WHOLE ROUNDS of resident
capacity: gfx1151 = 20 WGPs × 8 SIMD32, 768 VGPR/lane/SIMD, 64 KB LDS per
CU; packed-mw (192 VGPR, 31 KB LDS) holds 2 groups/CU = 80 slots, deq (256
VGPR) 1/CU = 40. The manifest's residentGroupsPerCu says 3 for deq (total
waves / group waves, ignoring per-SIMD placement inside one CU) and is ABSENT
for the unpinned packed kernel, so the law is computed from shipped surfaces
(`sliceWgsFor`/`residentGroupsPerMp`: manifest vgpr+ldsStaticBytes × Device
registers/simds/wave/lds/threads); `partitionSliceFollowsResidentCapacity`
pins 80/40 on gfx1151 and the `launch-geom` Diag record shows it under
`trace`. No override: packed-mw 554 (+9%), deq 688 (+8%) = 0.52x llama —
6.3.1's first gate is crossed by the deq route. HELD compiler follow-ups:
manifest residency granularity, residency for feasible-block kernels, a
CUs-per-multiprocessor surface. NEXT: port A-staging to the deq kernel.
PROGRESS 2026-09-12: A-staging ported to q4kWmmaDeqMw8Kernel per K-half (2→4
barriers/block, spill 124→0): bit-gate PASS but 666 vs 688 = −3% (three runs,
packed-mw control steady 553). On this kernel the two extra barriers cost
more than the wide A loads gain (its B loads were already wide global reads).
Trying ONE copy per block (128×256 tile, stride 272, 36 KB, barriers back to
2); patch of the two-half form kept at tmp/u3/deq-astage-two-half.patch.
RESULT: one-copy variant (2 barriers, 47.6 KB LDS, spill 8 B) bit-gate PASS,
685/686/678 vs 688 = parity. A-staging is REFUTED on the deq kernel: its A
loads were never the bottleneck (the +18% on packed-mw came from relieving
the fromWords VALU path, not from A bytes). REVERTED to the plain kernel;
patches kept in tmp/u3/deq-astage-{two-half,one-copy}.patch. NEXT:
re-profile the deq route at 688 to find where the remaining 2x lives.
PROGRESS 2026-09-12: profile of the deq route (whole run): q4kWmmaDeqMw8 66.8%,
q4k/q6k widen 13% (LOAD-time, ensureDeq builds once), q6kWmmaDeqMw8 9.7%,
attention 3.2% — inside the measured prefill the Q4_K GEMM is ~77%, Q6_K
~11%. Q6_K partitioned launcher moved onto the slice law (derives 40):
neutral, 688→689. @Occupancy(minResident=8) on the deq kernel RE-MEASURED
under the 80-slot slice the old note asked for: 192 VGPR / 212 B spill →
519 at 80, 497 at 40, vs 688 unpinned — spill dominates at any partition;
REVERTED, note extended in the kernel. 6.2.3 (Q6_K): the slice law
transfers; the A-staging technique does not (refuted on deq).
PROGRESS 2026-09-12 (ISA analysis of q4kWmmaDeqMw8Kernel): llama.cpp on gfx1151
runs the SAME v_wmma_i32_16x16x16_iu8 (RDNA3 → AMD_WMMA_AVAILABLE), the same
128×128 tile, 8 waves, and the same per-32-k float epilogue — per-element
arithmetic is at parity. Per 32-k step, one wave: 16 WMMA = 256 VALU-port
cycles, scaledAccumInto2S = 229 (43% of the port), addressing 43, 18
global_load_b128, 33 ds ops (64 rgAll dwords), 44 blocking s_waitcnt; issue
floor caps matrix use at 46%, measured 24%. STRUCTURAL difference: our wave
owns 8 token tiles × 1 column tile (18 loads, 66 LDS dwords, no row-scale
reuse) vs llama's 2×4 (12 loads, 24 dwords, 4× reuse). Ranked cuts: (1)
re-shape to 2×4 [source-only, in flight]; (2) integer-fold the 6-bit
sub-block scale via v_mad_i32_i24 into an int32 accumulator, float drain
once per block: −776 of 2414 VALU slots/block (−17% of the floor), exact
(|dot| < 2^23, Σ ≤ 3.1e7 < 2^31) but needs a NEW CooperativeMatrix verb
`scaledAccumI32(iacc, scale)` lowered to llvm.amdgcn.mad.i24 (a plain mul
lands quarter-rate v_mul_lo and is a net loss) — HELD for the compiler;
(3) staging: software f16 decode loop (needs an f16→f32 intrinsic, HELD),
flatten xsA..xsH → xsAll[128] (~85 slots/block) and mask the `ps` top byte
so the 4 byte loads combine (~40 VALU + 9 VMEM/block) [both in flight].
PROGRESS 2026-09-12: cuts (1)+(3ii)+(3iii) LANDED on q4kWmmaDeqMw8Kernel —
each wave now owns 2 token tiles × 4 column tiles (4 wave-rows × 2
wave-columns tile the 128×128), 12 operand loads per 32-k step (4 A + 8 B,
was 18), 4 (cf,cg) pairs + 2 rg/xs slices (was 1 + 8), xsA..xsH → xsAll[128]
(no divergent 8-arm store), `ps` top byte masked so the 4 byte loads combine.
Bit-gate PASS, SPILL-FREE (was 124 B), 8B Q4_K_M prefill 512: 688 → 877
tok/s (+27%, three runs 872-877; packed-mw control 556) = 0.66x llama 1320.
PROGRESS 2026-09-12: the same 2×4 re-shape on q6kWmmaDeqMw8Kernel (Q6 has 16
one-step j iterations and no min term: 2 A + 4 B loads + 8 mma per j, cf
stride 16 → column-tile offset 256; xsA..xsH → xsAll[128]; no unmasked byte
assembly to fix): bit-identical to the Mw4 sibling (new gate, exact), route
877 → 884 (+1%, its 11% share), no spill.
PROGRESS 2026-09-12: re-profile at 884: q4kWmmaDeqMw8 409 ms (was 668), q6k
88 (was 97), attention 33, widen 135 (load-time). Remaining gap to llama
1320 = 1.49x, almost all inside the two GEMMs. The 32-k step loop is NOT
unrolled (llama's is; that is where its load batching comes from) and the
kernel language has no unroll directive — third HELD compiler item beside
scaledAccumI32 (mad.i24 int-fold, −17% floor) and a native f16→f32.
Packed-mw kernel: 4×2 re-shape (its B unpack is VALU per column tile, so
2×4 would quadruple it) LANDED: bit-gate PASS, spill 204 → 168 B, 555 → 566 (+2%);
the `ps` top-byte mask on it: 566 → 573 (+1%). Packed route 573 = 0.43x, deq
route 884 = 0.67x llama 1320. NEXT (compiler-side; the compiler tree is on main
and clean of anything related — a stale session-start snapshot was misread as
uncommitted ownership work): scaledAccumI32 int-fold verb (−17% of the issue floor), f16→f32
intrinsic, loop-unroll directive; then re-profile.

### 6.1 TDD
- [x] 6.1.1 Bit-correctness gate: pin the current `q4kWmmaKernel` Q4_K prefill
      output (a fixed shape, e.g. 512×4096×4096 on a routable fixture) as a
      reference; the new multi-wave kernel matches it bit-for-bit (int8 MMA is
      exact — no f16-noise tolerance). Host-parity check too.
- [x] 6.1.2 (slice law + block = 8 waves asserted in `packedMwSubmissionMatchesManifest`; the `32` grep gate run by hand — clean) Portability: a GPU-free `LaunchGeometryTest`-style assertion that
      the launcher's derived block == `Group.laneBlock()`-consistent value and
      grid covers all tiles; and that no `block:[32]`/`/32` literal remains in
      the new kernel+launcher (grep gate).
- [x] 6.1.3 (`partitionSliceFollowsResidentCapacity` + `packedMwSubmissionMatchesManifest`: manifest block accepted, submission derives the access sets, no refusal, descriptor 4×256 / 31232 B / 32 waves) Deployment: the launcher reads a non-null `manifest().feasibleBlocks()`
      on gfx1151 and launches with `feasibleBlocks()[0]`; `Scheduler.submit`
      access sets match the manifest (no refusal).

### 6.2 Coding
- [~] 6.2.1 (built and bit-correct at 554; spill 204 B, not zero — despill regressed on deq, held) `q4kWmmaMwKernel` (packed route): multi-wave tile (N waves/wg
      cooperating on a larger output tile), int32 in-register accumulation
      across sub-blocks, Q4_K sub-block scales + dmin folded in the epilogue
      WITHOUT per-sub-block LDS round-trips/barriers. Wave width via
      `Group.width()`. ISA-verified zero spill (`--xpu-emit=isa`).
- [x] 6.2.2 (launcher + slice law + every slice routed through `Scheduler.submit` via `mwSubmit`, geometry read back from the descriptor, no measurable cost 554→555; the default stays OFF: the packed route is the memory base and 19% behind the int8-deq route) Portable launcher `q4kWmmaMwLaunch`: manifest `feasibleBlocks` +
      `Device` geometry → block/grid (measured-literal fallback); route through
      `Scheduler.submit`; wire into Linear's packed Q4_K route behind a flag
      (`setQ4Mw`), default off until 6.3 passes, then default on.
- [x] 6.2.3 (slice law on the Q6_K deq launcher, neutral; the 2×4 wave re-shape transfers: q6kWmmaDeqMw8Kernel bit-identical to its Mw4 sibling under `multiWaveQ6kDeqMw8AgreesWithDeqMw4`, route 877 → 884; A-staging does not transfer) Apply the same to the Q6_K packed route (`q6kWmmaEpiKernel`,
      14.5%) if the technique transfers, or record why Q6_K differs.

### 6.3 Acceptance
- [x] 6.3.1 (PARITY PASSED 2026-09-13 on every dense format: 1.04x-1.59x llama.cpp's best backend at matched micro-batch; decode unchanged) Prefill parity (amdgpu/gfx1151, idle-gated A/B, announce first):
      8B Q4_K_M prefill 512 ≥ 0.5x llama (first gate, from 0.165x), target
      parity ≥ 1.0x (stretch). Re-profile: q4kWmmaMwKernel share + occupancy
      (`residentGroupsPerCu` up vs the single-wave kernel). ppl unchanged
      (6.1.1), decode within noise, no new spill.
- [x] 6.3.2 (80/40 reproduced from the profile; SM expectation: simds=4 → the unit is the SM, 65536 regs) Portability re-derive: the launcher reproduces gfx1151's measured
      block/grid from the profile (not a literal); note the NVIDIA-path
      expectation (wave32, different simdCount) without a device to run it.

PROGRESS 2026-09-13: EVERY FORMAT. The 2×4 wave re-shape applied to the three
remaining int8 GEMMs (Q8_0, Q3_K, Q2_K), all spill-free, all six formats
bit-identical to a pinned pre-change fingerprint (new `deqMw8Fingerprints-
AcrossFormats`, a position-weighted fold — an XOR fold reads 0 here because
these outputs repeat and XOR cancels). Q8_0 deq 567 → 804 (+42%), Q3_K
749 → 778, Q2_K 676 → 721. Separately, the SHIPPED default was the wrong arm
on four of six formats: `prefillWeights` now defaults to "auto", a per-format
policy (int8 widen for every k-quant but Q2_K, which keeps the packed
cooperative GEMM), with "packed"/"int8" still forcing globally. 8B prefill
512, shipped default before → after, vs llama.cpp's best backend:
Q2_K 919 → 915 (0.81x), Q3_K 422 → 775 (0.57x), Q4_K 218 → 889 (0.67x),
Q5_K 555 → 833 (0.63x), Q6_K 216 → 781 (0.76x), Q8_0 640 → 803 (0.88x, and
1.29x llama's own ROCm backend). Decode unchanged at 25.8-55.4, which is
0.86-0.97 of llama's best and already beats its ROCm backend on Q3_K/Q4_K.
llama study per format: their MMQ does all unpacking once in `load_tiles`
(our `deq` widen is the same decision one level out); their per-warp shape is
2 weight-row × 4 token tiles for Q8_0/Q4_K/Q3_K — the same rule as our 2×4
under a transposed C fragment — and Q2_K's 1×5 is an LDS-overflow artifact
(16 of its 100 ints per row are dead), not a model to copy. Their Q2_K also
pays an extra all-ones MMA for the tail min-sums, which our staging avoids.

PROGRESS 2026-09-13 (partition law + int fold): TWO further levers.
(1) The partition width was a per-kernel resident-capacity derivation (40 or
80) and four of the six int8 launchers ignored it entirely, still slicing at
the 32 literal. MEASURED: 60 WGs is the optimum for EVERY format, sharply —
50 and 70 are 3-6% worse in two reps — and it does NOT track each kernel's
own residency (60 wins whether the kernel holds 2 groups/CU or 4). So the
law is now three slices per driver multiprocessor, shared by all five int8
deq launchers; the capacity number is still computed and reported in the
`launch-geom` Diag record, but no longer sizes the partition. Mechanism for
the sharp peak at exactly 3/mp is NOT established (tile divisibility is an
untested alternative), and one device cannot separate the two.
(2) `scaledAccumI32` (the compiler verb shipped today) applied to
q3kWmmaDeqMw8Kernel: the raw 6-bit scale folds into an int32 accumulator via
mul.i24 per 16-k sub-block and the float drain runs once per 256-k block
instead of 16 times. Staging splits `d*sc` into `scAll` (int32) + `dAll`
(float per column). No spill (~241 VGPR estimate fit). +4.7%. Output moves
by 2.4e-10 relative — it now rounds ONCE per block instead of 16 times, so
it is strictly more accurate; the other five formats stay bit-identical.
8B prefill 512, shipped default at session start -> now, vs llama's best:
Q2_K 919 -> 943 (0.84x), Q3_K 422 -> 898 (0.67x), Q4_K 218 -> 915 (0.69x),
Q5_K 555 -> 900 (0.68x), Q6_K 216 -> 799 (0.77x), Q8_0 640 -> 876 (0.96x,
and 1.40x llama's own ROCm backend). Decode unchanged.
NEXT: int-fold the shared q4kWmmaDeqMw8Kernel (52% of Q5_K's prefill, 21% of
Q3_K's), then Q2_K/Q6_K. MoE is the far larger gap: Qwen1.5-MoE measures 75
tok/s prefill against llama's 2342 (0.03x) and is a different route entirely
(slots never take the deq path).

MoE FINDING 2026-09-13 (Qwen1.5-MoE-A2.7B Q4_K_M, profiled): the MoE prefill
gap is NOT a kernel problem. 512-token prefill takes 7095 ms of wall for
**183 ms of total device time** — 97% of it is host/latency. The device work
itself is the wrong shape too: 4096 calls of `q4kQ8WaveMatVecKernel` (the
DECODE matvec, 11 us each) and 1036 of `q4kQ8BatchMatMulKernel` (66 us), i.e.
the expert path runs many tiny launches instead of grouping tokens by expert
into one GEMM per (layer, expert) the way llama.cpp's mul_mat_id does.
Host profile shows `ExpertBank.admitAndRun` at 799 ms total and a
`Prim.gluRowHost` CPU fallback inside prefill, plus per-expert upload/
download round trips. Chunk size confirms the per-chunk overhead: 128 -> 512
takes 79 -> 94 tok/s (+19%), still 0.04x llama's 2342. THE UNIT: group the
chunk's tokens by expert once, then one batched GEMM per (layer, expert) on
the existing Mw8 kernels, with the gather/scatter on device and no host round
trip per expert. Everything the dense arc built (2x4 tiling, the partition
law, the int fold) applies unchanged once the launches are the right shape.
Mixtral 130 vs llama 542 (0.24x) and Qwen3-Coder-30B 337-689 vs 1295 are the
same story at smaller ratios.

PROGRESS 2026-09-13 — PARITY PASSED ON EVERY DENSE FORMAT. Two more levers.
(1) The DEFAULT PREFILL MICRO-BATCH was 128 while llama.cpp's own default is
512, so every comparison so far ran at a 4x batching handicap. Raising it is
worth ~46% (Q4_K 910 -> 1325 at a stroke) and is purely a batching choice:
new gate `greedyOutputIsIndependentOfPrefillChunk` runs a ~300-token prompt
through the real 8B engine at chunk 128 and 512 and requires identical greedy
text. 1024 is worse than 512 on five of six formats.
(2) `scaledAccumI32` applied to the remaining int8 GEMMs. Q4_K/Q5_K share a
kernel that also carries a dmin term, so only the dot term int-folds and the
min stays per-sub-block on the new `rank1AccumS`; that kernel needed its mma
temporaries halved to 4 with two token chunks (est. 239 VGPR vs 275) to stay
spill-free. Q6_K and Q2_K likewise.
8B prefill 512 vs llama.cpp's BEST backend (hip or vulkan, whichever wins):
  Q2_K   1274 vs 1123 = 1.13x
  Q3_K_M 1396 vs 1349 = 1.04x
  Q4_K_M 1465 vs 1320 = 1.11x
  Q5_K_M 1415 vs 1314 = 1.08x
  Q6_K   1318 vs 1033 = 1.28x
  Q8_0   1452 vs  914 = 1.59x
Session start, shipped default, was 218-919. Decode unchanged at 25.8-55.5,
which is 0.86-0.97x llama's best and beats its ROCm backend on Q3_K/Q4_K.
The two bit-identity gates between kernel pairs became tolerance gates (3e-8
and 6e-8): the folded routes no longer share a summation order with their
unfolded siblings, by design. Per-format fingerprints moved 1e-9 to 1e-10
(fewer roundings, so strictly more accurate); Q8_0, which was not folded,
stayed bit-identical.
6.3.1 is MET on every dense format. Remaining: decode 0.86-0.97x (bandwidth
bound, at the measured coalesced ceiling), and MoE, which is a launch-shape
problem (see the MoE FINDING above), not a kernel one.

PROGRESS 2026-09-13 (adaptive by construction). Three pieces so a kernel sizes
itself on hardware the toolchain has never met, rather than inheriting
gfx1151's constants:
(1) MEASURED footprint (cajeta 3e628b40): `KernelManifest.measuredRegsPerThread/
SpillBytes/LdsStaticBytes/MaxThreads` read the LOADED code object through
cuFuncGetAttribute/hipFuncGetAttribute. Static, because on unknown hardware
there is no manifest to hang them on. `residentGroupsPerMp` now asks the
driver first and falls back to the manifest, so the occupancy law answers
where the compile-time model is absent entirely. On gfx1151 measured and
modelled agree exactly (192/192 and 234/234 regs, 168/0 spill), which is the
validation, not a no-op.
(2) `cajeta.xpu.Autotune` (cajeta 9b28e72b): a per-device store keyed by a
fingerprint of the machine shape, so a cache copied between machines is
ignored rather than believed. One file per (device, knob); a corrupt entry
reads as absent. The SWEEP stays with the caller by design — only the caller
knows what a candidate costs and how to check it is still correct.
(3) The partition width is now DISCOVERED here: `tunesThePartitionWidth-
OnThisDevice` times 40/50/60/72/80 on real prefills and records the winner;
`sliceWgsFor` prefers a tuned value over the derived rule (launch-geom source
3 vs 1). METHOD MATTERS: one sample per candidate gave a 60% spread with no
curve and picked 72; three passes, alternating direction, keeping the MIN per
candidate gives 400/379/367/372/374 — a clean minimum at 60, independently
reproducing the hand sweep. A tuner that can pick a bad value is worse than
a constant.

HEAD-TO-HEAD 2026-09-13, same window, quiet box (load ~1.5), llama.cpp build
5306f4b. Both engines measured minutes apart rather than against the recorded
2026-09-06 rows, because llama's ROCm numbers today read 10-13% BELOW what the
same build produced that day (Q4_K 1160 vs 1320) while ours moved ~1% — so the
stale rows would have flattered us. 8B Llama-3.1, prompt 512, gen 64:

  format   cajeta pp/tg     llama ROCm pp/tg   pp ratio  tg ratio
  Q2_K     1281 / 55.36     989 / 55.58        1.29x     1.00x
  Q3_K_M   1392 / 47.86    1186 / 46.01        1.17x     1.04x
  Q4_K_M   1449 / 41.80    1160 / 39.70        1.25x     1.05x
  Q5_K_M   1409 / 35.65    1146 / 35.28        1.23x     1.01x
  Q6_K     1320 / 31.84     926 / 31.42        1.42x     1.01x
  Q8_0     1460 / 25.78     585 / 25.41        2.49x     1.01x

Against the RECORDED llama best-of-backends (hip or vulkan, 2026-09-06) the
prefill claim still holds on every format: 1.14x / 1.03x / 1.10x / 1.07x /
1.28x / 1.60x. DECODE now matches or beats llama's ROCm on all six, and the
comparison is CONSERVATIVE: llama's tg64 generates from an empty context while
ours decodes after a 512-token prefill, so ours carries a 512-token KV
attention cost per token that llama's does not.
CAVEAT: no Vulkan number today. Both local llama builds register only a ROCm
device (build-vulkan/bin/llama-bench reports backend ROCm and --list-devices
shows ROCm0 alone), so the recorded Vulkan decode figures (Q2_K 64.6) could
not be re-measured and we are likely still behind Vulkan on decode.

VULKAN RESOLVED 2026-09-13. The old build-vulkan binary never ran Vulkan: a
system-wide ggml install in /usr/local/lib carries a HIP backend and the loader
resolves libggml-*.so.0 from there ahead of the build's own, so it reported
ROCm. Every "vulkan" figure taken from it was the HIP path. A clean clone at
/home/julian/code/llama.cpp.vulkan (same commit 5306f4b, GGML_VULKAN=ON with
HIP and CUDA OFF, needs glslc + spirv-headers) registers a real RADV device
with KHR_coopmat once run with LD_LIBRARY_PATH pointed at its own bin.
Settled box (load 1.5), best of passes: Q4_K llama-vulkan 1193 vs cajeta 1441
= 1.21x; Q8_0 llama-vulkan 921 vs cajeta 1455 = 1.58x. Vulkan today reproduces
the recorded 2026-09-06 figures (Q8_0 921 vs 914), so those rows are sound and
the earlier ROCm shortfall was thermal/power: DECODE (bandwidth bound) is
unchanged across both days while PREFILL (power bound) sags 7-13% after hours
of benching.
FINAL, vs llama's BEST EVER on this box (any backend, any day):
  prefill  Q2_K 1.14x  Q3_K 1.03x  Q4_K 1.10x  Q5_K 1.07x  Q6_K 1.28x  Q8_0 1.59x
  decode   0.87x-0.97x — BEHIND llama's Vulkan on every format, and that is the
           number to quote; the earlier "decode matches or beats" was against
           llama's ROCm path only. Ours also carries a 512-token KV context
           where llama's tg64 starts empty, so the gap is smaller than it looks.
The 2.47x Q8_0 figure was against llama's ROCm backend, which handles Q8_0
badly (585 vs its own 1160 on Q4_K). Against its best, Q8_0 is 1.59x.

DECODE, APPLES-TO-APPLES 2026-09-13. The "behind on every format" reading was
a DEPTH MISMATCH, not a deficit: llama-bench's tg64 generates from an EMPTY
context while ours decodes after a 512-token prefill, so we were paying a
512-token KV attention per token and llama was not. llama-bench takes `-d`
(n-depth), which prefills before measuring generation. Both engines at depth
512, quiet box (load 0.3-1.2), cajeta best of 2:
  format   cajeta   llama vulkan   ratio
  Q2_K     55.29    59.43          0.93x
  Q3_K_M   47.82    47.56          1.01x
  Q4_K_M   41.97    40.12          1.05x
  Q5_K_M   35.67    36.68          0.97x
  Q6_K     31.87    32.12          0.99x
  Q8_0     25.78    25.87          1.00x
So decode is at PARITY (0.93x-1.05x), not 0.87x-0.97x behind. One real gap
remains: Q2_K at 0.93x, the most bandwidth-bound format, where the dequant
cost per weight byte is highest. Q5_K at 0.97x is marginal.
LESSON: llama-bench's tg figure is depth 0 by default. Any decode comparison
must pass -d to match the context the other engine is carrying, or it measures
two different workloads.

LANE MAPPING 2026-09-13 — one real bug fixed, one hypothesis REFUTED.
The wave-per-row mat-vec family (24 kernels) hardcoded `globalIdX() / 32` and
launched `block: [32]`, so every one of them is WAVE32-ONLY and wrong on a
wave64 part, and each row got its own workgroup. q2kQ8WaveMatVecKernel is now
wave-relative (`Group.width()` for both the lane mask and the row divisor) and
its launcher takes geometry from a new shared helper — `QuantKernel.
waveWidth/waveRowsPerBlock/waveRowBlock/waveRowGrid` — which DERIVES the wave
width and the workgroup ceiling from the device and leaves exactly one free
parameter, rows per workgroup, measured per machine through Autotune.
REFUTED: rows per workgroup does not matter here. Swept 1/2/4/8 on Q2_K
decode: 55.29 / 55.40 / 55.36 / 55.35 tok/s — flat within 0.2%. Launch count
was not the limiter, so the 7% gap to llama's Vulkan decode (176 vs 189 GB/s
against a 212 GB/s roofline) is read efficiency inside the kernel, not launch
geometry. Decode output is byte-identical across the change.
NEXT for the Q2_K gap: the remaining structural difference is that llama's
Vulkan shaders run a 64-wide subgroup on RADV while ours is one wave32 per
row, so each of their rows is reduced by twice the lanes. Testing that needs
a wave64 launch path, not a re-mapping of the wave32 one.
OWED: the other 23 kernels of this family still hardcode 32. The fix is
mechanical now that the helper exists, but each needs its own decode
fingerprint before and after.

CORRECTION 2026-09-13 — the sweep in daa0d20 WAS NOT A SWEEP. The `mvrpb=`
bench flag it claimed to use had silently failed to apply (a perl replace that
did not match), and the check that should have caught it, `grep -n ... | head`,
exits on HEAD and so reported success on zero matches. All four "swept" runs
used the same default, and the 0.2% spread was pure noise. The conclusion
happened to be right; the evidence for it did not exist.
RE-RUN with a verified flag (an absurd value must move the number, checked
first), best of alternating passes, Q2_K decode tok/s:
  rows per WORKGROUP  1: 55.19   2: 54.75   4: 54.91   8: 54.83
  rows per WAVE       1: 54.41   2: 54.56   4: 54.49
Both flat-to-negative; the original one-row-per-wave, one-row-per-workgroup
geometry is best. So the activation-amortization theory is REFUTED here too,
despite predicting the cross-format pattern correctly: the activation vector
is ~4 KB and lives in L1/L2, so re-reading it per row costs cache bandwidth,
not DRAM, and this kernel is DRAM-bound on weights. The rule N = ceil(8A/B)
is sound as a TRAFFIC argument and wrong as a THROUGHPUT prediction on a part
whose cache holds the activations.
KEPT: the wave-relative fix (the family was wave32-only), the shared geometry
helper, and the knobs — with the derived default now 1, which is what this
device measures. The multi-row kernel is reverted and saved at
tmp/u3/q2k-multirow.patch; it is bit-identical and neutral, worth retrying on
a part with less cache or with fp16 activations (A=2 doubles every N).
STILL OPEN: q2kQ8WaveMatVecKernel's `b = lane / 4` with a literal `+ 8L`
stride is correct only at wave32 — on wave64 lanes 32-63 re-walk blocks lanes
0-31 already covered. The lane/row derivation is now wave-relative but this
stride is not; the family is not yet genuinely portable.

STRIDE IS PROFILE-DRIVEN 2026-09-13. The rule: a wave-cooperative kernel has
TWO numbers, and only one of them is a literal.
  lanesPerItem   = ALGORITHMIC. Four lanes share one 16-byte Q2_K group
                   because the group holds four 2-bit planes. A property of
                   the decomposition, so it stays written down.
  itemsInFlight  = waveWidth / lanesPerItem = THE MACHINE'S. It must come
                   from Group.width(), and it is the loop stride.
q2kQ8WaveMatVecKernel's stride was the literal 8, which is 32/4 — wave32 only.
At wave64 lanes 32-63 re-walked blocks lanes 0-31 had already done, double
counting. Now `bStride = wv / 4`. Bit-identical on wave32 (8), correct at
wave64 (16), decode unchanged at 55.2-55.3.
The same rewrite is owed to the rest of the family: `b = b + 32L` appears
twice (lanesPerItem = 1, so stride is simply `wv`) and `b = b + 8L` once more.
COMPILER FINDING, needs a minimal repro before filing: while editing, a sed
deleted `int64 b = (int64) (lane / 4);` from the kernel body. The build
SUCCEEDED with zero errors and the kernel ran on device, producing wrong
output and a fake 70% decode speedup (it was skipping work). An UNDECLARED
name in a @Kernel body appears to resolve to something rather than failing to
compile. Same family as the differently-typed shadow fixed in cajeta 706f2117,
but that check only fires on a REDECLARATION; this was no declaration at all.
