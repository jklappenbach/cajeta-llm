#!/usr/bin/env python3
# cajeta-llama plan Unit 16 — generate the GGUF test fixtures:
#
#   src/test/fixtures/gguf/toy.gguf          the toy model, four storage
#                                            types (F32/F16/BF16/Q8_0),
#                                            full llama metadata, embedded
#                                            tokenizer + chat template
#   src/test/fixtures/gguf/toy.safetensors   the SAME weights (already
#   src/test/fixtures/gguf/toy-config.json   rounded to each tensor's GGUF
#                                            storage precision), for the
#                                            16.3.1 format-parity gate:
#                                            both loaders must reach
#                                            IDENTICAL f32 weights, so the
#                                            two logits agree exactly
#   src/test/fixtures/gguf/toy-bad.gguf      one tensor typed Q4_K (12) —
#                                            unsupported until Unit 18;
#                                            the 16.1.6 error fixture
#
# Deterministic: numpy default_rng(42); regenerating is byte-stable.
import json, os, struct
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "src", "test", "fixtures", "gguf")

H, L, NH, NKV, HD, IT, V, CTX = 16, 2, 4, 2, 4, 32, 96, 64

GGUF_MAGIC = 0x46554747
T_U32, T_F32, T_STR, T_ARR, T_U64, T_I32 = 4, 6, 8, 9, 10, 5
GG_F32, GG_F16, GG_Q8_0, GG_Q4_K, GG_BF16 = 0, 1, 8, 12, 30

def s(x):  # gguf string
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b

def kv_str(k, v):   return s(k) + struct.pack("<I", T_STR) + s(v)
def kv_u32(k, v):   return s(k) + struct.pack("<II", T_U32, v)
def kv_f32(k, v):   return s(k) + struct.pack("<I", T_F32) + struct.pack("<f", v)
def kv_arr_str(k, xs):
    return s(k) + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_STR, len(xs)) + b"".join(s(x) for x in xs)
def kv_arr_f32(k, xs):
    return s(k) + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_F32, len(xs)) + struct.pack("<%df" % len(xs), *xs)
def kv_arr_i32(k, xs):
    return s(k) + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_I32, len(xs)) + struct.pack("<%di" % len(xs), *xs)

def f16_round(a):  return a.astype(np.float16).astype(np.float32)
def bf16_round(a):
    b = a.astype(np.float32).view(np.uint32)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)  # RNE
    return b.view(np.float32)

def q8_quant(a):
    # ggml Q8_0: blocks of 32, f16 scale d = amax/127, q = round(x/d) int8.
    flat = a.reshape(-1)
    assert flat.size % 32 == 0
    blocks = flat.reshape(-1, 32)
    out_q, out_d, deq = [], [], []
    for blk in blocks:
        amax = np.max(np.abs(blk))
        d = np.float16(amax / 127.0) if amax > 0 else np.float16(0)
        df = np.float32(d)
        q = np.round(blk / df).astype(np.int8) if df != 0 else np.zeros(32, np.int8)
        q = np.clip(q, -127, 127).astype(np.int8)
        out_d.append(d); out_q.append(q); deq.append(q.astype(np.float32) * df)
    payload = b"".join(struct.pack("<e", d) + q.tobytes() for d, q in zip(out_d, out_q))
    return payload, np.concatenate(deq).reshape(a.shape).astype(np.float32)

def bf16_bytes(a):
    return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()

rng = np.random.default_rng(42)
def W(*shape, scale=0.05):
    return (rng.standard_normal(shape) * scale).astype(np.float32)

# (hf_name, gguf_name, array, ggml_type)
tensors = []
def add(hf, gg, arr, ty): tensors.append([hf, gg, arr, ty])

add("model.embed_tokens.weight", "token_embd.weight", W(V, H), GG_F32)
for i in range(L):
    p, g = f"model.layers.{i}.", f"blk.{i}."
    add(p+"input_layernorm.weight", g+"attn_norm.weight", 1.0 + W(H, scale=0.01), GG_F32)
    add(p+"self_attn.q_proj.weight", g+"attn_q.weight", W(NH*HD, H), GG_F16 if i == 0 else GG_F32)
    add(p+"self_attn.k_proj.weight", g+"attn_k.weight", W(NKV*HD, H), GG_BF16 if i == 0 else GG_F32)
    add(p+"self_attn.v_proj.weight", g+"attn_v.weight", W(NKV*HD, H), GG_F32)
    add(p+"self_attn.o_proj.weight", g+"attn_output.weight", W(H, NH*HD), GG_F32)
    add(p+"post_attention_layernorm.weight", g+"ffn_norm.weight", 1.0 + W(H, scale=0.01), GG_F32)
    add(p+"mlp.gate_proj.weight", g+"ffn_gate.weight", W(IT, H), GG_F32)
    add(p+"mlp.up_proj.weight", g+"ffn_up.weight", W(IT, H), GG_F32)
    add(p+"mlp.down_proj.weight", g+"ffn_down.weight", W(H, IT), GG_Q8_0 if i == 0 else GG_F32)
add("model.norm.weight", "output_norm.weight", 1.0 + W(H, scale=0.01), GG_F32)
add("lm_head.weight", "output.weight", W(V, H), GG_F32)

# Round every tensor to its storage precision FIRST, so the safetensors
# twin carries the identical dequantized values (the 16.3.1 exactness).
payloads = []
for t in tensors:
    hf, gg, a, ty = t
    if ty == GG_F16:
        a = f16_round(a); pay = a.astype(np.float16).tobytes()
    elif ty == GG_BF16:
        a = bf16_round(a); pay = bf16_bytes(a)
    elif ty == GG_Q8_0:
        pay, a = q8_quant(a)
    else:
        pay = a.astype("<f4").tobytes()
    t[2] = a
    payloads.append(pay)

# Tiny SP tokenizer: unk/bos/eos, 256 byte tokens, a few pieces.
# SP BPE merges adjacent pieces only when the MERGED piece exists in the
# vocab, so reaching "▁hello" from characters needs the whole chain of
# intermediates — a real SP vocab has them; the first draft didn't, and
# encode byte-fell-back (caught by the round-trip test as verbatim ▁).
pieces = ["▁", "h", "e", "l", "o", "w", "r", "d",
          "▁h", "▁he", "▁hel", "▁hell", "▁hello",
          "▁w", "▁wo", "▁wor", "▁worl", "▁world"]
toks = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)] + pieces
types = [2, 3, 3] + [6]*256 + [1]*len(pieces)
# Longer pieces score higher (less negative), the SP convention.
scores = [0.0, 0.0, 0.0] + [0.0]*256 \
       + [-30.0 + len(x) for x in pieces]
TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}"
            "<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}")

kvs = [
    kv_str("general.architecture", "llama"),
    kv_str("general.name", "cajeta-toy"),
    kv_u32("llama.embedding_length", H),
    kv_u32("llama.block_count", L),
    kv_u32("llama.attention.head_count", NH),
    kv_u32("llama.attention.head_count_kv", NKV),
    kv_u32("llama.feed_forward_length", IT),
    kv_u32("llama.context_length", CTX),
    kv_u32("llama.rope.dimension_count", HD),
    kv_u32("llama.vocab_size", V),
    kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
    kv_f32("llama.rope.freq_base", 10000.0),
    kv_str("tokenizer.ggml.model", "llama"),
    kv_arr_str("tokenizer.ggml.tokens", toks),
    kv_arr_f32("tokenizer.ggml.scores", scores),
    kv_arr_i32("tokenizer.ggml.token_type", types),
    kv_str("tokenizer.chat_template", TEMPLATE),
]

def build(path, bad=False):
    kv_blob = b"".join(kvs)
    ALIGN = 32
    dirs, off = [], 0
    for (hf, gg, a, ty), pay in zip(tensors, payloads):
        ne = list(reversed(a.shape))        # ggml ne: fastest dim first
        t = GG_Q4_K if (bad and gg == "blk.1.ffn_up.weight") else ty
        d = s(gg) + struct.pack("<I", len(ne)) \
            + b"".join(struct.pack("<Q", n) for n in ne) \
            + struct.pack("<IQ", t, off)
        dirs.append(d)
        off += (len(pay) + ALIGN - 1) // ALIGN * ALIGN
    head = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(kvs))
    body = head + kv_blob + b"".join(dirs)
    pad = (-len(body)) % ALIGN
    blob = body + b"\x00" * pad
    for pay in payloads:
        blob += pay + b"\x00" * ((-len(pay)) % ALIGN)
    open(path, "wb").write(blob)
    print(path, len(blob), "bytes")

os.makedirs(OUT, exist_ok=True)
build(os.path.join(OUT, "toy.gguf"))
build(os.path.join(OUT, "toy-bad.gguf"), bad=True)

# Safetensors twin + config.json.
st_meta, st_data, off = {}, b"", 0
for hf, gg, a, ty in tensors:
    b = a.astype("<f4").tobytes()
    st_meta[hf] = {"dtype": "F32", "shape": list(a.shape),
                   "data_offsets": [off, off + len(b)]}
    st_data += b; off += len(b)
hdr = json.dumps(st_meta, sort_keys=True).encode()
open(os.path.join(OUT, "toy.safetensors"), "wb").write(
    struct.pack("<Q", len(hdr)) + hdr + st_data)
json.dump({
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": H, "intermediate_size": IT, "num_hidden_layers": L,
    "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
    "vocab_size": V, "max_position_embeddings": CTX,
    "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
    "tie_word_embeddings": False, "hidden_act": "silu",
}, open(os.path.join(OUT, "toy-config.json"), "w"), indent=1)
print("safetensors twin + config written")
