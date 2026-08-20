#!/usr/bin/env python3
# cajeta-llama plan Unit 18 — k-quant fixtures from REAL blocks.
#
# Pulls raw packed blocks of every required type out of the downloaded
# Llama-3.1-8B GGUFs (Q4_K_M mixes Q4_K/Q5_K/Q6_K by tensor sensitivity —
# spec 9.5's whole point; Q8_0 comes from the Q8_0 build; Q2_K/Q3_K/Q4_0
# are synthesized from the same real f32 data via gguf-py's own quantizers,
# llama.cpp's reference implementation), dequantizes each with gguf-py, and
# writes per-type fixture pairs:
#
#   src/test/fixtures/gguf/kquant/<type>.bin   raw packed blocks
#   src/test/fixtures/gguf/kquant/<type>.f32   reference dequant (LE f32)
#   src/test/fixtures/gguf/kquant/manifest.json  type ids, counts, sources
#
# 8 blocks per type — enough to exercise every sub-block scale position —
# a few KB committed per type.
import json, os, struct, sys
sys.path.insert(0, "/home/julian/code/llama.cpp/gguf-py")
import numpy as np
import gguf
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize, quantize

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "src", "test", "fixtures", "gguf", "kquant")
os.makedirs(OUT, exist_ok=True)

Q4KM = "/home/julian/models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
Q80 = "/home/julian/models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"

BLOCKS = 8
manifest = {}

def emit(name, ty, raw, deq):
    open(os.path.join(OUT, f"{name}.bin"), "wb").write(raw)
    open(os.path.join(OUT, f"{name}.f32"), "wb").write(
        deq.astype("<f4").tobytes())
    manifest[name] = {"ggml_type": int(ty), "elements": int(deq.size),
                      "bytes": len(raw)}
    print(name, "type", int(ty), len(raw), "bytes ->", deq.size, "f32")

def from_real(path, ty, name):
    r = gguf.GGUFReader(path)
    for t in r.tensors:
        if t.tensor_type == ty:
            bs, es = gguf.GGML_QUANT_SIZES[ty]   # (block elems, block bytes)
            nbytes = BLOCKS * es
            raw = bytes(t.data.reshape(-1)[:nbytes].tobytes())
            deq = dequantize(
                np.frombuffer(raw, dtype=np.uint8).reshape(BLOCKS, es), ty)
            emit(name, ty, raw, deq.reshape(-1))
            manifest[name]["source_tensor"] = str(t.name)
            return
    raise SystemExit(f"no {name} tensor in {path}")

def from_quantizer(ty, name, elems):
    rng = np.random.default_rng(7)
    data = (rng.standard_normal(elems) * 0.3).astype(np.float32)
    raw = quantize(data, ty).tobytes()
    deq = dequantize(np.frombuffer(raw, dtype=np.uint8), ty)
    emit(name, ty, raw, deq.reshape(-1))
    manifest[name]["source_tensor"] = "gguf-py quantize (seed 7)"

MODELS = "/home/julian/models/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-%s.gguf"
# Real blocks from the real builds (a Q4_K_M mixes Q4_K/Q6_K — spec 9.5):
from_real(Q4KM, T.Q4_K, "q4_k")
from_real(Q4KM, T.Q6_K, "q6_k")
from_real(Q80, T.Q8_0, "q8_0")
from_real(MODELS % "Q5_K_M", T.Q5_K, "q5_k")
from_real(MODELS % "Q2_K", T.Q2_K, "q2_k")
from_real(MODELS % "Q3_K_M", T.Q3_K, "q3_k")
# Q4_0 has no published build for this model — gguf-py's own quantizer is
# still llama.cpp's reference implementation:
from_quantizer(T.Q4_0, "q4_0", BLOCKS * 32)

json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print("manifest written")
