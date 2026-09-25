# tools/parity — the parity harness

The instruments behind `agents/xpu-kernel-adaptor-plan.md` Unit 8 (spec
§7.1 "Par, defined"): every number that judges a cajeta kernel against
its llama.cpp counterpart is produced here, with the four fields the spec
demands on the same line as the number.

| file | what |
|---|---|
| `llama-env.sh` | where llama.cpp is, and the WSL driver-library fix every llama.cpp binary on this box needs (see the file) |
| `build-ggml-leg.sh` | builds `ggml-leg` and `llama-greedy` against `llama.cpp/build-cuda` |
| `ggml-leg.cpp` | the llama.cpp leg: one MUL_MAT shape, device-timed on ggml's own stream through its own events, over a pool of real tensors from the reference GGUF, one `leg-row` per run |
| `llama-greedy.cpp` | llama.cpp's greedy walk with the top-2 logit gap per position, the reference for GreedyVsLlama |
| `run-parity-legs.sh` | both legs at the reference model's seven decode shapes, joined by `bench/ParityJoin` into `census/parity-<backend>.tsv` |
| `run-greedy.sh` | greedy token agreement, llama.cpp against `bench/GreedyVsLlama` on the same prompt ids |
| `run-parity.sh`, `gen_fixture.py`, `convert_f32.py` | the older fp32-fixture parity run (plan 15 of the llm plan), untouched |

The cajeta side lives in `src/main/cajeta/dev/cajeta/llm/bench/`:
`LegRow` (the line format), `ParityLeg` (the cajeta leg), `ParityTable`
(the refusals and the judgement), `ParityJoin` (the table writer),
`OracleMap` (`census/oracle-map.tsv`, every shipped kernel to its
counterpart or NONE), `GreedyVsLlama`. `selftest/ParityTableTest` and
`selftest/OracleMapTest` pin the rules; the numbers are never asserted
in the suite.

## The rules a row must satisfy before it is a comparison

1. same op, type and shape on both legs;
2. the llama.cpp leg labelled `CUDA`, `HIP`, `Vulkan` or `CPU` with its
   build hash, never bare `cuda`;
3. the same residency, `cold` (pool past L2) or `hot`, never `unknown`;
4. the `device` timer tier on both legs;
5. a calibration row that reproduced a recorded number within tolerance,
   else the table refuses every pair and says so.

`of_peak` is the binding roofline axis against the part's THEORETICAL
bus bandwidth (1008.1 GB/s on the 4090), a floor on the true fraction.

## Run

```sh
XPU_BACKEND=nvptx tools/parity/run-parity-legs.sh    # census/parity-nvptx.tsv
XPU_BACKEND=nvptx DEVICE=cuda tools/parity/run-greedy.sh
```

Both need the reference model at `$MODEL`
(`~/models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`)
and a CUDA build of llama.cpp at `$LLAMA_ROOT/build-cuda`.

## Three things measured on the way (2026-09-24, RTX 4090, llama.cpp 67a17c1)

- **The event clock's scale moves with the part's state.** The ratio of
  CUDA event time to host time read 0.98 to 1.10 across processes minutes
  apart, busy or idle, on both legs, so a scale measured once (KernelTimer's
  first-use calibration, or a sleep-bracketed pair) is not a correction.
  Both legs now raise every bracket to about 200 ms of device time and
  measure the ratio on that same bracket: the host clock spanning the
  submissions to the synchronize bounds the interval to well under 0.1%,
  the row's `clock_scale` column is that bracket's ratio, and the number
  is the host-bounded bracket. Across the eight shapes of one run it reads
  1.0534 to 1.0543 on both legs. xpu-kernel-adaptor plan 5.4.3.
- **The ggml q4_K / q5_K / iq4_nl MMVQ kernels read quantized noise 30x
  slower than real weights.** A weight quantized by `ggml_quantize_chunk`
  from uniform noise streams at 28 GB/s cold; the same shape from the
  model's own tensors, or a byte ramp, at 860 GB/s; q4_0, q6_K, q8_0 and
  f16 do not care. Not chased (it is llama.cpp's kernel and not the
  engine's case), but it is why both legs measure REAL tensors and why
  `--fill pattern`/`--fill quantize` exist on `ggml-leg`.
- **The hot calibration row reproduces `test-backend-ops`.** One q4_K
  4096x14336 tensor repeated, device-timed over a long bracket: 13.83 us
  against test-backend-ops' 12.94 us the same day and the recorded 13.33
  us of 2026-09-19 (3.7% apart, tolerance 5%). A short bracket (20
  launches) had read 20 us: launch and capture latency, not the kernel.
