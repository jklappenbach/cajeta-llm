# cajeta-llm

A decoder-only LLM inference engine for the cajeta ecosystem
(`dev.cajeta.llm`): load Llama-family open weights from Hugging Face
safetensors and GGUF, run KV-cached incremental decode on any XPU backend,
sample, detokenize.

llama.cpp supplies the architecture reference; none of its code is reused —
cajeta's `@Native` marshals no C structs or callbacks, so the engine is
written against `cajeta.math.Tensor` + `cajeta.xpu` end to end.

## Kernels

Every device kernel the engine ships, how the families serve decode, prefill,
batched decode, mixture-of-experts and attention, how a kernel is chosen for a
weight format and a batch shape on the device the process landed on, and what
is measured at load and per problem: the website's
[Language models on the device](https://cajeta.dev/kernels/llm/) and the
complete [catalog](https://cajeta.dev/kernels/catalog-llm/), one row per
registered kernel with its source, its llama.cpp counterpart and its census
standing on each backend. The machinery underneath is
[One kernel, any device](https://cajeta.dev/kernels/). In this repository the
sources are `src/main/cajeta/dev/cajeta/llm/io` (quantised kernels, tiles,
WMMA GEMMs) and `model` (attention, primitives, embedding); the census tables
and the llama.cpp oracle map are under `census/`
([census/README.md](census/README.md)).

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
`agents/cajeta-llm-plan.md` (dependency-ordered units; the early stdlib units
landed in the `cajeta` repo, the engine here).

## Documentation

The user documentation is on the website: [Language models](https://cajeta.dev/llm/)
(quickstart, the engine API, models and formats, inside the engine,
configuration, running and serving with cabra, measuring and where it stands
against llama.cpp). Status is kept there; the plan
(`agents/cajeta-llm-plan.md` in the cajeta workspace) is the record of what
is done and open.

## Building

```
cajeta build        # emits build/archive/dev.cajeta.llm-<version>.cja
./run-tests.sh      # cajeta-unit suite (CAJETA=<compiler> to override)
```

The loader needs `MappedFile` and the int64 file path (cajeta main ≥
2026-08-13); until that ships in a release, point `CAJETA` at a main build.
