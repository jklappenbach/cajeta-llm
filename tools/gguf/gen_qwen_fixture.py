#!/usr/bin/env python3
# cajeta-llm plan Unit 20 — the Qwen2-family GGUF fixtures:
#
#   src/test/fixtures/gguf/toy-qwen.gguf     arch `qwen2`, 2 layers, with
#                                            the q/k/v BIASES the Llama
#                                            path has nowhere to put
#   src/test/fixtures/gguf/toy-badarch.gguf  arch `mamba` — the 20.1.1
#                                            error fixture
#
# Deliberately NOT an edit of gen_fixture.py's toy.gguf: a dozen tests
# assert against that file's exact contents, and a second architecture is
# additive, not a variation of the first.
#
# Two things this fixture pins that toy.gguf cannot:
#   * hyperparameters live under `qwen2.*`, so a reader that hard-codes
#     `llama.*` reads nothing and the config comes out empty;
#   * biases exist at all. A dropped bias still produces fluent text,
#     which is why QwenArchTest checks the number, not the output.
#
# Qwen2 is NOT rope-permuted by convert_hf_to_gguf.py (only the Llama
# family is), so q/k are stored in HF layout here — matching what the
# loader's ropePermuteHeads already does for a non-llama architecture.
#
# Deterministic: numpy default_rng(20); regenerating is byte-stable.
import os, struct
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "src", "test", "fixtures", "gguf")

H, L, NH, NKV, HD, IT, V, CTX = 16, 2, 4, 2, 4, 32, 64, 64

GGUF_MAGIC = 0x46554747
T_U32, T_F32, T_STR, T_ARR, T_U64, T_I32 = 4, 6, 8, 9, 10, 5
GG_F32 = 0


def s(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v):   return s(k) + struct.pack("<I", T_STR) + s(v)
def kv_u32(k, v):   return s(k) + struct.pack("<II", T_U32, v)
def kv_f32(k, v):   return s(k) + struct.pack("<I", T_F32) + struct.pack("<f", v)


def kv_arr_u32(k, xs):
    return (s(k) + struct.pack("<I", T_ARR)
            + struct.pack("<IQ", T_U32, len(xs))
            + struct.pack("<%dI" % len(xs), *xs))


rng = np.random.default_rng(20)


def W(*shape, scale=0.05):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


# (gguf_name, array)
tensors = []
def add(gg, arr): tensors.append((gg, arr.astype(np.float32)))

add("token_embd.weight", W(V, H))
for i in range(L):
    g = f"blk.{i}."
    add(g + "attn_norm.weight", 1.0 + W(H, scale=0.01))
    add(g + "attn_q.weight", W(NH * HD, H))
    add(g + "attn_k.weight", W(NKV * HD, H))
    add(g + "attn_v.weight", W(NKV * HD, H))
    # The whole point of this fixture: biases, and non-trivial ones —
    # a bias of zeros would pass a test that drops it.
    add(g + "attn_q.bias", W(NH * HD, scale=0.5))
    add(g + "attn_k.bias", W(NKV * HD, scale=0.5))
    add(g + "attn_v.bias", W(NKV * HD, scale=0.5))
    add(g + "attn_output.weight", W(H, NH * HD))
    add(g + "ffn_norm.weight", 1.0 + W(H, scale=0.01))
    add(g + "ffn_gate.weight", W(IT, H))
    add(g + "ffn_up.weight", W(IT, H))
    add(g + "ffn_down.weight", W(H, IT))
add("output_norm.weight", 1.0 + W(H, scale=0.01))
add("output.weight", W(V, H))


def kvs_for(arch):
    p = arch + "."
    kv = [
        kv_str("general.architecture", arch),
        kv_str("general.name", "cajeta-toy-qwen"),
        kv_u32(p + "embedding_length", H),
        kv_u32(p + "block_count", L),
        kv_u32(p + "attention.head_count", NH),
        kv_u32(p + "attention.head_count_kv", NKV),
        kv_u32(p + "feed_forward_length", IT),
        kv_u32(p + "context_length", CTX),
        kv_u32(p + "rope.dimension_count", HD),
        kv_u32(p + "vocab_size", V),
        kv_f32(p + "attention.layer_norm_rms_epsilon", 1e-6),
        kv_f32(p + "rope.freq_base", 1000000.0),
    ]
    if arch == "qwen2vl":
        # M-RoPE sections, as Qwen2.5-VL ships them. Text-only inference
        # must be unaffected (spec 4.13).
        kv.append(kv_arr_u32(p + "rope.dimension_sections", [1, 1, 2, 0]))
    return kv


def build(path, arch):
    kvs = kvs_for(arch)
    kv_blob = b"".join(kvs)
    ALIGN = 32
    dirs, off, payloads = [], 0, []
    for gg, a in tensors:
        pay = a.astype("<f4").tobytes()
        payloads.append(pay)
        ne = list(reversed(a.shape))       # ggml ne: fastest dim first
        d = (s(gg) + struct.pack("<I", len(ne))
             + b"".join(struct.pack("<Q", n) for n in ne)
             + struct.pack("<IQ", GG_F32, off))
        dirs.append(d)
        off += (len(pay) + ALIGN - 1) // ALIGN * ALIGN
    head = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(kvs))
    body = head + kv_blob + b"".join(dirs)
    blob = body + b"\x00" * ((-len(body)) % ALIGN)
    for pay in payloads:
        blob += pay + b"\x00" * ((-len(pay)) % ALIGN)
    with open(path, "wb") as f:
        f.write(blob)
    print(f"wrote {path} ({len(blob)} bytes, {len(tensors)} tensors, {arch})")


build(os.path.join(OUT, "toy-qwen.gguf"), "qwen2")
build(os.path.join(OUT, "toy-qwenvl.gguf"), "qwen2vl")
build(os.path.join(OUT, "toy-badarch.gguf"), "mamba")
