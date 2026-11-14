#!/usr/bin/env python3
# cajeta-llm plan 15.2.21.5: gemma3 GGUF fixtures, written the way llama.cpp's converter
# writes them (conversion/gemma.py, src/models/gemma3.cpp at 5306f4b):
#
#   toy-gemma3.gguf      FourNormDeviceTest's Gemma toy (2 layers, H 512, hd 128, no window)
#                        with ToyModel.valAt values: every *norm.weight carries the converter's
#                        +1, projections are lossless Q8_0 (scale 1/64), q/k are not permuted
#                        (NEOX rope), the embedding is tied, ffn_norm is the PRE-feedforward norm.
#   toy-gemma3-27b.gguf  metadata only, the 27B's keys: sliding_window 1024 with the default
#                        period of 6, linear rope scaling 8 on global layers, freq_base_swa,
#                        and the 27B's attention scale (1/sqrt(n_embd/n_head), not head_dim).
import os, struct
import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "src", "test", "fixtures", "gguf")

H, L, NH, NKV, HD, IT, V, CTX = 512, 2, 4, 1, 128, 512, 256, 128

GGUF_MAGIC = 0x46554747
T_U32, T_F32, T_STR, T_ARR, T_I32 = 4, 6, 8, 9, 5
GG_F32, GG_Q8_0 = 0, 8

def s(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b

def kv_str(k, v):  return s(k) + struct.pack("<I", T_STR) + s(v)
def kv_u32(k, v):  return s(k) + struct.pack("<II", T_U32, v)
def kv_f32(k, v):  return s(k) + struct.pack("<I", T_F32) + struct.pack("<f", v)
def kv_arr_str(k, xs):
    return s(k) + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_STR, len(xs)) \
        + b"".join(s(x) for x in xs)
def kv_arr_f32(k, xs):
    return s(k) + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_F32, len(xs)) \
        + struct.pack("<%df" % len(xs), *xs)
def kv_arr_i32(k, xs):
    return s(k) + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_I32, len(xs)) \
        + struct.pack("<%di" % len(xs), *xs)

def val_at(k, n):
    """ToyModel.valAt(k, j) for j in [0, n): tensor k of the HF key order, torch layout."""
    j = np.arange(n, dtype=np.int64)
    m = (k * 7 + (j % 9973)) % 13 - 6
    return m.astype(np.float32) * np.float32(0.03125)

def q8_0_exact(vals, rows, cols):
    """Q8_0 with d = 1/64 (f16 0x2400): every toy value m/32 (|m| <= 6) is q = 2m exactly."""
    assert cols % 32 == 0
    q = np.rint(vals.reshape(rows, cols // 32, 32) * 64.0).astype(np.int8)
    assert np.all(q.astype(np.float32) / 64.0 == vals.reshape(rows, cols // 32, 32))
    d = np.full((rows, cols // 32, 1), 0x2400, dtype="<u2").view(np.int8).reshape(rows, cols // 32, 2)
    return np.concatenate([d, q], axis=2).tobytes()

def build_toy():
    hf = ["model.embed_tokens.weight"]
    for i in range(L):
        p = f"model.layers.{i}."
        hf += [p + n for n in ["input_layernorm.weight", "self_attn.q_proj.weight",
               "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
               "self_attn.q_norm.weight", "self_attn.k_norm.weight",
               "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
               "post_feedforward_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
               "mlp.down_proj.weight"]]
    hf.append("model.norm.weight")
    k_of = {n: k for k, n in enumerate(hf)}

    gg_of = {"input_layernorm.weight": "attn_norm.weight", "self_attn.q_proj.weight": "attn_q.weight",
             "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight",
             "self_attn.o_proj.weight": "attn_output.weight", "self_attn.q_norm.weight": "attn_q_norm.weight",
             "self_attn.k_norm.weight": "attn_k_norm.weight",
             "post_attention_layernorm.weight": "post_attention_norm.weight",
             "pre_feedforward_layernorm.weight": "ffn_norm.weight",
             "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
             "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
             "mlp.down_proj.weight": "ffn_down.weight"}
    dims = {"self_attn.q_proj.weight": (NH * HD, H), "self_attn.k_proj.weight": (NKV * HD, H),
            "self_attn.v_proj.weight": (NKV * HD, H), "self_attn.o_proj.weight": (H, NH * HD),
            "mlp.gate_proj.weight": (IT, H), "mlp.up_proj.weight": (IT, H), "mlp.down_proj.weight": (H, IT)}

    tensors = []
    tensors.append(("token_embd.weight", [H, V], GG_F32, val_at(k_of["model.embed_tokens.weight"], V * H).tobytes()))
    for i in range(L):
        p = f"model.layers.{i}."
        for rest, gg in gg_of.items():
            k = k_of[p + rest]
            name = f"blk.{i}.{gg}"
            if rest in dims:
                rows, cols = dims[rest]
                tensors.append((name, [cols, rows], GG_Q8_0, q8_0_exact(val_at(k, rows * cols), rows, cols)))
            else:
                n = HD if "_norm.weight" in rest and "self_attn" in rest else H
                tensors.append((name, [n], GG_F32, (val_at(k, n) + np.float32(1.0)).tobytes()))
    tensors.append(("output_norm.weight", [H], GG_F32, (val_at(k_of["model.norm.weight"], H) + np.float32(1.0)).tobytes()))

    toks = [f"<0x{b:02X}>" for b in range(V)]
    kp = "gemma3."
    kvs = [
        kv_str("general.architecture", "gemma3"),
        kv_str("general.name", "cajeta-toy-gemma3"),
        kv_u32(kp + "block_count", L),
        kv_u32(kp + "context_length", CTX),
        kv_u32(kp + "embedding_length", H),
        kv_u32(kp + "feed_forward_length", IT),
        kv_u32(kp + "attention.head_count", NH),
        kv_u32(kp + "attention.head_count_kv", NKV),
        kv_f32(kp + "rope.freq_base", 10000.0),
        kv_f32(kp + "rope.freq_base_swa", 10000.0),
        kv_f32(kp + "attention.layer_norm_rms_epsilon", 1e-6),
        kv_u32(kp + "attention.key_length", HD),
        kv_u32(kp + "attention.value_length", HD),
        kv_str("tokenizer.ggml.model", "llama"),
        kv_arr_str("tokenizer.ggml.tokens", toks),
        kv_arr_f32("tokenizer.ggml.scores", [0.0] * V),
        kv_arr_i32("tokenizer.ggml.token_type", [6] * V),
    ]
    return kvs, tensors

def build_27b():
    kp = "gemma3."
    kvs = [
        kv_str("general.architecture", "gemma3"),
        kv_str("general.name", "cajeta-toy-gemma3-27b-keys"),
        kv_u32(kp + "block_count", 62),
        kv_u32(kp + "context_length", 131072),
        kv_u32(kp + "embedding_length", 5376),
        kv_u32(kp + "feed_forward_length", 21504),
        kv_u32(kp + "attention.head_count", 32),
        kv_u32(kp + "attention.head_count_kv", 16),
        kv_str(kp + "rope.scaling.type", "linear"),
        kv_f32(kp + "rope.scaling.factor", 8.0),
        kv_f32(kp + "rope.freq_base", 1000000.0),
        kv_f32(kp + "rope.freq_base_swa", 10000.0),
        kv_f32(kp + "attention.layer_norm_rms_epsilon", 1e-6),
        kv_u32(kp + "attention.key_length", 128),
        kv_u32(kp + "attention.value_length", 128),
        kv_u32(kp + "attention.sliding_window", 1024),
        kv_u32(kp + "vocab_size", 262208),
    ]
    tensors = [("output_norm.weight", [16], GG_F32, np.ones(16, dtype="<f4").tobytes())]
    return kvs, tensors

def write(name, kvs, tensors):
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
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    open(path, "wb").write(blob)
    print(path, len(blob), "bytes,", len(tensors), "tensors")

kvs, tensors = build_toy()
write("toy-gemma3.gguf", kvs, tensors)
kvs, tensors = build_27b()
write("toy-gemma3-27b.gguf", kvs, tensors)
