# cajeta-llm

A decoder-only LLM inference engine for the cajeta ecosystem
(`dev.cajeta.llm`): load Llama-family open weights from Hugging Face
safetensors and GGUF, run KV-cached incremental decode on any XPU backend,
sample, detokenize.

llama.cpp supplies the architecture reference; none of its code is reused —
cajeta's `@Native` marshals no C structs or callbacks, so the engine is
written against `cajeta.math.Tensor` + `cajeta.xpu` end to end.

## Related projects

- **[cabra](../cajeta-cabra)** (`dev.cajeta.cabra`) — the resident-model
  serving harness: loads a model once through this engine's public API,
  receives prompts, streams responses, and stops each one on the model's
  end-of-generation set (`LlmEngine.eogTokens`). In the reference
  ecosystem's terms, `dev.cajeta.llm` is the llama.cpp role, cabra the
  llama-server role, and olla the distribution half of the ollama role.
  cabra deliberately consumes only this library's public surface — it is
  the engine's first real embedder, so an API gap shows up there first.
- **olla** — the package registry; distributes this library (and, once
  application install lands, cabra itself).

Design and plan live in the cajeta workspace:
`specs/cajeta-llm-spec.md` (requirements + decisions) and
`agents/cajeta-llm-plan.md` (19 units, dependency-ordered). Units 1–4, 10
and 17 are stdlib work in the `cajeta` repo; this repo is the engine
(Units 5–16, 18–19).

## Status

| Area | State |
|------|-------|
| Safetensors loading (mmap-backed, native dtype, sharded) | Unit 5 — in progress |
| Tape-free forward path | Unit 6 — pending |
| Transformer primitives, ragged attention | Unit 7 — pending |
| Paged KV cache, scheduler, prefix store | Units 8–11 — pending |
| Tokenizer, chat templates, sampling | Units 12–14 — pending |
| GGUF + k-quants | Units 16, 18 — pending |
| Parity + perf gates (vs transformers / llama.cpp) | Unit 15 — pending |

## Targets

Llama 3.x, Mistral, Qwen2/3, Gemma 3 (text). Perf gates: decode ≥60% /
prefill ≥50% of llama.cpp on Llama-3.1-8B-Instruct Q4_K_M at gfx1151/ROCm,
measured baseline first.

## Building

```
cajeta build        # emits build/archive/dev.cajeta.llm-<version>.cja
./run-tests.sh      # cajeta-unit suite (CAJETA=<compiler> to override)
```

The loader needs `MappedFile` and the int64 file path (cajeta main ≥
2026-08-13); until that ships in a release, point `CAJETA` at a main build.
