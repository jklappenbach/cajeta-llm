#!/usr/bin/env python3
# threaded-forward-path plan Unit 4 — generate
#
#   src/test/fixtures/gguf/toy-q4k.gguf
#
# A four-tensor GGUF carrying a Q4_K and a Q6_K weight, each with its
# dequantized F32 twin,
# so a test can bind a PACKED Linear and a reference Linear from the same
# numbers and compare them.
#
# The Q4_K bytes are not quantized here: they are the REAL blocks already
# checked in at src/test/fixtures/gguf/kquant/q4_k.bin, pulled from the
# published Llama-3.1-8B Q4_K_M build, and q4_k.f32 is gguf-py's reference
# dequantization of exactly those bytes. Re-quantizing would introduce a
# second implementation of Q4_K into the fixtures whose only job would be
# to agree with the first.
#
# 1152 bytes = 8 blocks = 2048 elements, laid out [out=8, in=256] so every
# row is exactly one block.
#
# Deterministic: it copies bytes. Regenerating is byte-stable.
import os, struct
import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "src", "test", "fixtures", "gguf")
KQ = os.path.join(OUT, "kquant")

GGUF_MAGIC = 0x46554747
T_U32, T_F32, T_STR = 4, 6, 8
GG_F32, GG_Q4_K, GG_Q6_K = 0, 12, 14
OUT_DIM, IN_DIM = 8, 256

def s(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b

def kv_str(k, v):  return s(k) + struct.pack("<I", T_STR) + s(v)
def kv_u32(k, v):  return s(k) + struct.pack("<II", T_U32, v)
def kv_f32(k, v):  return s(k) + struct.pack("<I", T_F32) + struct.pack("<f", v)

q4k = open(os.path.join(KQ, "q4_k.bin"), "rb").read()
deq4 = np.frombuffer(open(os.path.join(KQ, "q4_k.f32"), "rb").read(),
                     dtype="<f4")
q6k = open(os.path.join(KQ, "q6_k.bin"), "rb").read()
deq6 = np.frombuffer(open(os.path.join(KQ, "q6_k.f32"), "rb").read(),
                     dtype="<f4")
assert len(q4k) == 8 * 144, len(q4k)
assert len(q6k) == 8 * 210, len(q6k)
assert deq4.size == OUT_DIM * IN_DIM, deq4.size
assert deq6.size == OUT_DIM * IN_DIM, deq6.size

# Q4_K and Q6_K in ONE file, which is the shape a real Q4_K_M ships:
# measured on Meta-Llama-3.1-8B-Instruct-Q4_K_M, 74.4% of bytes are Q4_K
# and 25.6% are Q6_K (attn_v and output.weight), with no Q5_K and no Q8_0.
# A mix is therefore the normal case, not an edge case.
#
# (gguf_name, ne (fastest dim first), ggml type, payload)
tensors = [
    ("blk.0.ffn_down.weight", [IN_DIM, OUT_DIM], GG_Q4_K, q4k),
    ("blk.0.ffn_up.weight",   [IN_DIM, OUT_DIM], GG_F32,
     deq4.astype("<f4").tobytes()),
    ("blk.0.attn_v.weight",   [IN_DIM, OUT_DIM], GG_Q6_K, q6k),
    ("blk.0.attn_output.weight", [IN_DIM, OUT_DIM], GG_F32,
     deq6.astype("<f4").tobytes()),
]

kvs = [
    kv_str("general.architecture", "llama"),
    kv_str("general.name", "cajeta-q4k-linear"),
    kv_u32("llama.embedding_length", IN_DIM),
    kv_u32("llama.block_count", 1),
    kv_u32("llama.attention.head_count", 1),
    kv_u32("llama.attention.head_count_kv", 1),
    kv_u32("llama.feed_forward_length", OUT_DIM),
    kv_u32("llama.context_length", 64),
    kv_u32("llama.rope.dimension_count", IN_DIM),
    kv_u32("llama.vocab_size", 1),
    kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
    kv_f32("llama.rope.freq_base", 10000.0),
]

ALIGN = 32
dirs, off = [], 0
for gg, ne, ty, pay in tensors:
    dirs.append(s(gg) + struct.pack("<I", len(ne))
                + b"".join(struct.pack("<Q", n) for n in ne)
                + struct.pack("<IQ", ty, off))
    off += (len(pay) + ALIGN - 1) // ALIGN * ALIGN

head = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(kvs))
body = head + b"".join(kvs) + b"".join(dirs)
blob = body + b"\x00" * ((-len(body)) % ALIGN)
for _gg, _ne, _ty, pay in tensors:
    blob += pay + b"\x00" * ((-len(pay)) % ALIGN)

path = os.path.join(OUT, "toy-q4k.gguf")
open(path, "wb").write(blob)
print(path, len(blob), "bytes")
