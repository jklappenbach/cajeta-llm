#!/usr/bin/env python3
# cajeta-llm plan Unit 26 (26.1.1) — generate
#
#   src/test/fixtures/gguf/toy-moe.gguf           the MoE instrument
#   src/test/fixtures/gguf/toy-moe-split-bad.gguf 15.17's reject arm
#   src/test/fixtures/gguf/toy-moe-badwidth.gguf  15.3's reject arm
#
# toy-moe is Unit 24's routable dimensions with the dense FFN replaced
# by 4 experts (top-2) plus a GATED shared expert, shaped like the
# qwen2moe witness on this box — one fixture witnesses both 15.9's
# routed path and 15.18's sigmoid gate. Faithfulness notes, each
# MEASURED against the downloaded files 2026-08-31 (tmp/gguf_dump.py):
#
#   - qwen2moe carries NO expert_feed_forward_length key: the expert
#     width exists only in the slab shape (Qwen1.5-MoE-A2.7B, 60
#     experts, width 1408). The fixture matches — a loader that needs
#     the key would bind the fixture and then fail the witness.
#   - the slabs are rank-3, fastest-dim first:
#       ffn_gate_exps/ffn_up_exps  ne=[H, IT_E, E]
#       ffn_down_exps              ne=[IT_E, H, E]
#     and the router ffn_gate_inp ne=[H, E] is F32.
#   - the shared expert is dense (ffn_*_shexp) with its own learned
#     gate VECTOR ffn_gate_inp_shexp ne=[H] (sigmoid on x·w — 15.18).
#   - in the real Q4_K_M witness ffn_down_exps is Q8_0, because its
#     width 1408 is not a whole number of Q4_K superblocks per row —
#     the exact 15.3 constraint. The fixture's width IS Q4_K-clean
#     (256), so every slab here stays Q4_K and the existing kernels
#     serve expert GEMVs unchanged; the badwidth fixture below carries
#     the constraint's reject arm.
#
# Routing preconditions are inherited from gen_routable_fixture.py and
# re-asserted at the bottom (the Unit 24 rule: a dimension edit fails
# HERE, not as a mysteriously vacuous test).
import os, struct
import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "src", "test", "fixtures", "gguf")
KQ = os.path.join(OUT, "kquant")

# Unit 24's routable dimensions...
H, L, NH, NKV, HD, V, CTX = 512, 2, 4, 1, 128, 256, 256
# ...with the FFN replaced by experts. IT_E is one Q4_K superblock per
# expert row-run on purpose: the smallest width that keeps every slice
# block-aligned (15.3) and every per-expert GEMV coop-routable
# (rows 256 % 128, cols 512 % 256).
E, USED, IT_E, IT_S = 4, 2, 256, 512

GGUF_MAGIC = 0x46554747
T_U32, T_F32, T_STR, T_ARR, T_I32 = 4, 6, 8, 9, 5
GG_F32, GG_Q4_K = 0, 12

BLOCK_ELEMS = 256
BLOCK_BYTES = 144
SRC_BLOCKS = 8

def s(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b

def kv_str(k, v):  return s(k) + struct.pack("<I", T_STR) + s(v)
def kv_u32(k, v):  return s(k) + struct.pack("<II", T_U32, v)
def kv_f32(k, v):  return s(k) + struct.pack("<I", T_F32) + struct.pack("<f", v)
def kv_bool(k, v): return s(k) + struct.pack("<I", 7) + struct.pack("<B", 1 if v else 0)
def kv_arr_str(k, xs):
    return s(k) + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_STR, len(xs)) \
        + b"".join(s(x) for x in xs)
def kv_arr_f32(k, xs):
    return s(k) + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_F32, len(xs)) \
        + struct.pack("<%df" % len(xs), *xs)
def kv_arr_i32(k, xs):
    return s(k) + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_I32, len(xs)) \
        + struct.pack("<%di" % len(xs), *xs)

q4k = open(os.path.join(KQ, "q4_k.bin"), "rb").read()
assert len(q4k) == SRC_BLOCKS * BLOCK_BYTES, len(q4k)

# Unit 28: the raw source blocks decode to values whose FFN outputs land
# around 1e-5 — f16-SUBNORMAL, so the coop path's f16 activation staging
# lost ~1% relative and device parity failed on numbers no real model
# produces (real checkpoints put activations in f16-normal range, which
# is why the dense toy-routable passes the same staging at 0.99999).
# Scale each block's d/dmin f16 halves x16: gate/up outputs x16, the
# GLU product x256, the expert output x4096 — into normal range. The
# router, the oracle constants, and every byte-count assertion are
# untouched; the shexp ratio is scale-invariant by construction.
def scale_q4k_blocks(raw, k):
    # The source d/dmin are THEMSELVES f16-subnormal (measured: block
    # 0's d has a zero exponent) — decode, scale, re-encode via numpy
    # so subnormals promote to normal cleanly. x16 is exact in binary
    # fp, so no quantization drift beyond the promotion itself.
    out = bytearray(raw)
    for b in range(len(raw) // BLOCK_BYTES):
        for half in (0, 2):   # d, dmin
            o = b * BLOCK_BYTES + half
            v = np.frombuffer(bytes(out[o:o + 2]), dtype=np.float16)[0]
            w = np.float16(np.float32(v) * np.float32(k))
            assert np.isfinite(w), (b, half, v)
            out[o:o + 2] = w.tobytes()
    return bytes(out)

q4k = scale_q4k_blocks(q4k, 16)

def q4k_tile_blocks(nblocks, phase=0):
    """`nblocks` real Q4_K superblocks, tiled cyclically starting at
    `phase` — the phase keeps different experts' slices from being the
    same bytes, so a mis-sliced expert view cannot accidentally agree
    with the oracle."""
    pay = []
    for i in range(nblocks):
        k = (i + phase) % SRC_BLOCKS
        pay.append(q4k[k * BLOCK_BYTES:(k + 1) * BLOCK_BYTES])
    return b"".join(pay)


def q4k_rank2(rows, cols, phase=0):
    assert cols % BLOCK_ELEMS == 0, (rows, cols)
    return q4k_tile_blocks(rows * (cols // BLOCK_ELEMS), phase)

def q4k_rank3(cols, rows, experts):
    """One merged slab: expert-major, each expert a rank-2 [rows, cols]
    Q4_K payload with its own tiling phase."""
    assert cols % BLOCK_ELEMS == 0, (cols, rows, experts)
    return b"".join(q4k_rank2(rows, cols, phase=e) for e in range(experts))

def f32_ramp(rows, cols=None, scale=0.02, bias=0.0):
    n = rows if cols is None else rows * cols
    a = ((np.arange(n) % 25) - 12).astype(np.float32) / 64.0 * (scale * 64.0)
    a = a + bias
    shape = (rows,) if cols is None else (rows, cols)
    return a.reshape(shape).astype(np.float32)

def router_weights(rows, cols):
    """Router rows that PREFER DIFFERENT experts for different tokens:
    each expert's row peaks on a different residual phase, so top-2
    identity varies by token and a top-k bug cannot hide behind a
    constant selection. Values on the 1/64 grid like every fixture."""
    a = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        idx = (np.arange(cols) + 7 * r) % 25
        a[r] = ((idx - 12).astype(np.float32) / 64.0) * 1.28
    return a.astype(np.float32)


def write_gguf(path, kvs, tensors):
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
    open(path, "wb").write(blob)
    print(path, len(blob), "bytes,", len(tensors), "tensors")


def base_kvs(arch, extra):
    toks = [f"<0x{b:02X}>" for b in range(V)]
    return [
        kv_str("general.architecture", arch),
        kv_str("general.name", "cajeta-toy-moe"),
        kv_u32(arch + ".embedding_length", H),
        kv_u32(arch + ".block_count", L),
        kv_u32(arch + ".attention.head_count", NH),
        kv_u32(arch + ".attention.head_count_kv", NKV),
        kv_u32(arch + ".feed_forward_length", IT_S),
        kv_u32(arch + ".context_length", CTX),
        kv_u32(arch + ".rope.dimension_count", HD),
        kv_u32(arch + ".vocab_size", V),
        kv_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5),
        kv_f32(arch + ".rope.freq_base", 10000.0),
    ] + extra + [
        kv_str("tokenizer.ggml.model", "llama"),
        kv_arr_str("tokenizer.ggml.tokens", toks),
        kv_arr_f32("tokenizer.ggml.scores", [0.0] * V),
        kv_arr_i32("tokenizer.ggml.token_type", [6] * V),
    ]


# ── toy-moe.gguf ────────────────────────────────────────────────────────
tensors = []
def add(name, ne, ty, pay):
    tensors.append((name, ne, ty, pay))
def add_q4k(name, rows, cols, phase=0):
    add(name, [cols, rows], GG_Q4_K, q4k_rank2(rows, cols, phase))
def add_f32(name, arr):
    add(name, list(reversed(arr.shape)), GG_F32, arr.astype("<f4").tobytes())

add_f32("token_embd.weight", f32_ramp(V, H))
for i in range(L):
    g = f"blk.{i}."
    add_f32(g + "attn_norm.weight", f32_ramp(H, scale=0.01, bias=1.0))
    add_q4k(g + "attn_q.weight",      NH * HD, H)
    add_q4k(g + "attn_k.weight",      NKV * HD, H)
    add_q4k(g + "attn_v.weight",      NKV * HD, H)
    # qwen2moe carries attention biases (measured on the witness).
    add_f32(g + "attn_q.bias", f32_ramp(NH * HD, scale=0.005))
    add_f32(g + "attn_k.bias", f32_ramp(NKV * HD, scale=0.005))
    add_f32(g + "attn_v.bias", f32_ramp(NKV * HD, scale=0.005))
    add_q4k(g + "attn_output.weight", H, NH * HD)
    add_f32(g + "ffn_norm.weight", f32_ramp(H, scale=0.01, bias=1.0))
    # The routed bank: merged rank-3 slabs, expert-major.
    add(g + "ffn_gate_exps.weight", [H, IT_E, E], GG_Q4_K,
        q4k_rank3(H, IT_E, E))
    add(g + "ffn_up_exps.weight",   [H, IT_E, E], GG_Q4_K,
        q4k_rank3(H, IT_E, E))
    add(g + "ffn_down_exps.weight", [IT_E, H, E], GG_Q4_K,
        q4k_rank3(IT_E, H, E))
    add_f32(g + "ffn_gate_inp.weight", router_weights(E, H))
    # The gated shared expert (15.18).
    add_q4k(g + "ffn_gate_shexp.weight", IT_S, H, phase=1)
    add_q4k(g + "ffn_up_shexp.weight",   IT_S, H, phase=2)
    add_q4k(g + "ffn_down_shexp.weight", H, IT_S, phase=3)
    add_f32(g + "ffn_gate_inp_shexp.weight", f32_ramp(H, scale=0.02))
add_f32("output_norm.weight", f32_ramp(H, scale=0.01, bias=1.0))
add_q4k("output.weight", V, H)

kvs = base_kvs("qwen2moe", [
    kv_u32("qwen2moe.expert_count", E),
    kv_u32("qwen2moe.expert_used_count", USED),
    # Deliberately NO qwen2moe.expert_feed_forward_length: the witness
    # ships none; the width must come from the slab.
])
os.makedirs(OUT, exist_ok=True)
write_gguf(os.path.join(OUT, "toy-moe.gguf"), kvs, tensors)

# Every routing precondition, restated as asserts (the Unit 24 rule),
# including the per-expert VIEW shapes the bank will hand the kernels.
assert HD <= 128 and HD % 2 == 0
assert NH % NKV == 0
for gg, ne, ty, _pay in tensors:
    if ty != GG_Q4_K:
        continue
    cols, rows = ne[0], ne[1]
    assert cols % 256 == 0, (gg, "coopColsOk: cols % 256", cols)
    assert rows % 128 == 0, (gg, "coop route: outDim % 128", rows)
assert (IT_E * H) % BLOCK_ELEMS == 0     # expert stride lands on blocks
assert IT_E % 128 == 0 and H % 256 == 0  # each VIEW routes like a Linear
print("routing preconditions hold, per-expert views included")

# ── unit 27 arms ────────────────────────────────────────────────────────
# Same model, one lever moved per file (controls vary the mechanism):
#   noshexp — the shared expert absent entirely (27.1.2's control);
#   shexp0  — the shexp gate VECTOR zeroed, so its sigmoid is exactly 0.5
#             (with the real vector the multiply is sigma(g.x); the ratio
#             of the two shexp contributions is the sigmoid, measured
#             without dequantizing anything in the oracle);
#   norm    — expert_weights_norm=true carried IN METADATA (15.6: the
#             reader honors a present key; no witness carries one);
#   sigmoid — expert_gating_func=2, which must REFUSE (15.4/15.13).
t_noshexp = [t for t in tensors if "shexp" not in t[0]]
write_gguf(os.path.join(OUT, "toy-moe-noshexp.gguf"), kvs, t_noshexp)

t_shexp0 = [(n, ne, ty,
             (np.zeros(H, dtype="<f4").tobytes()
              if n.endswith("ffn_gate_inp_shexp.weight") else pay))
            for (n, ne, ty, pay) in tensors]
write_gguf(os.path.join(OUT, "toy-moe-shexp0.gguf"), kvs, t_shexp0)

kvs_norm = kvs + [kv_bool("qwen2moe.expert_weights_norm", True)]
write_gguf(os.path.join(OUT, "toy-moe-norm.gguf"), kvs_norm, tensors)

kvs_sig = kvs + [kv_u32("qwen2moe.expert_gating_func", 2)]
write_gguf(os.path.join(OUT, "toy-moe-sigmoid.gguf"), kvs_sig, tensors)

# ── the 27.1.1 router oracle ────────────────────────────────────────────
# Computed here from the SAME formulas that wrote the file, in f32, for
# the probe row the test constructs: x[j] = ((3j % 25) - 12) / 64.
xp = ((np.arange(H) * 3 % 25) - 12).astype(np.float32) / np.float32(64.0)
R = router_weights(E, H)
logits = (R.astype(np.float32) @ xp).astype(np.float32)
pr = np.exp((logits - logits.max()).astype(np.float32)).astype(np.float32)
pr = (pr / pr.sum().astype(np.float32)).astype(np.float32)
order = np.argsort(-pr, kind="stable")
sel = order[:USED]
w_raw = pr[sel]
w_norm = (w_raw / max(w_raw.sum(), np.float32(6.103515625e-5))).astype(np.float32)
gvec = f32_ramp(H, scale=0.02)
gs = 1.0 / (1.0 + np.exp(-float(np.dot(gvec.astype(np.float32), xp))))
print("ORACLE 27.1.1 (x[j] = ((3j %% 25) - 12)/64):")
print("  logits  =", " ".join("%.8f" % v for v in logits))
print("  probs   =", " ".join("%.8f" % v for v in pr))
print("  top%d    =" % USED, " ".join(str(int(e)) for e in sel))
print("  w_raw   =", " ".join("%.8f" % v for v in w_raw))
print("  w_norm  =", " ".join("%.8f" % v for v in w_norm))
print("  shexp gate sigma(g.x) = %.8f" % gs)
assert len(set(np.round(pr, 6))) == E, "router probs must be distinct"

# Second probe: the NEGATED row must flip the selection (an oracle a
# stuck always-[0,1] top-k cannot pass).
xn2 = (-xp).astype(np.float32)
l2 = (R.astype(np.float32) @ xn2).astype(np.float32)
p2 = np.exp((l2 - l2.max()).astype(np.float32)).astype(np.float32)
p2 = (p2 / p2.sum().astype(np.float32)).astype(np.float32)
o2 = np.argsort(-p2, kind="stable")[:USED]
w2 = p2[o2]
print("ORACLE 27.1.1 probe 2 (x2 = -x):")
print("  top%d    =" % USED, " ".join(str(int(e)) for e in o2))
print("  w_raw   =", " ".join("%.8f" % v for v in w2))
assert set(o2) != set(int(e) for e in sel), "probe 2 must select differently"


# ── toy-moe-split-bad.gguf (15.17) ──────────────────────────────────────
# TheBloke's pre-2024-04 Mixtral layout, reproduced small: plain llama
# arch, MoE declared ONLY by expert_count, and one tensor PER EXPERT
# (ffn_gate.0.weight ...). The loader must reject this naming the fix
# (a merged-slab requant), never a bare missing-tensor throw. F32
# everywhere and one layer: this file exists to be refused.
BH, BNH, BIT = 64, 2, 64
bt = []
def badd_f32(name, arr):
    bt.append((name, list(reversed(arr.shape)), GG_F32,
               arr.astype("<f4").tobytes()))
badd_f32("token_embd.weight", f32_ramp(V, BH))
g = "blk.0."
badd_f32(g + "attn_norm.weight", f32_ramp(BH, scale=0.01, bias=1.0))
badd_f32(g + "attn_q.weight", f32_ramp(BH, BH))
badd_f32(g + "attn_k.weight", f32_ramp(BH, BH))
badd_f32(g + "attn_v.weight", f32_ramp(BH, BH))
badd_f32(g + "attn_output.weight", f32_ramp(BH, BH))
badd_f32(g + "ffn_norm.weight", f32_ramp(BH, scale=0.01, bias=1.0))
badd_f32(g + "ffn_gate_inp.weight", router_weights(E, BH))
for e in range(E):
    badd_f32(g + f"ffn_gate.{e}.weight", f32_ramp(BIT, BH))
    badd_f32(g + f"ffn_up.{e}.weight", f32_ramp(BIT, BH))
    badd_f32(g + f"ffn_down.{e}.weight", f32_ramp(BH, BIT))
badd_f32("output_norm.weight", f32_ramp(BH, scale=0.01, bias=1.0))
badd_f32("output.weight", f32_ramp(V, BH))

def small_kvs(arch, h, nh, it, extra):
    toks = [f"<0x{b:02X}>" for b in range(V)]
    return [
        kv_str("general.architecture", arch),
        kv_str("general.name", "cajeta-toy-moe-bad"),
        kv_u32(arch + ".embedding_length", h),
        kv_u32(arch + ".block_count", 1),
        kv_u32(arch + ".attention.head_count", nh),
        kv_u32(arch + ".attention.head_count_kv", nh),
        kv_u32(arch + ".feed_forward_length", it),
        kv_u32(arch + ".context_length", 64),
        kv_u32(arch + ".vocab_size", V),
        kv_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-5),
        kv_f32(arch + ".rope.freq_base", 10000.0),
    ] + extra + [
        kv_str("tokenizer.ggml.model", "llama"),
        kv_arr_str("tokenizer.ggml.tokens", toks),
        kv_arr_f32("tokenizer.ggml.scores", [0.0] * V),
        kv_arr_i32("tokenizer.ggml.token_type", [6] * V),
    ]

write_gguf(os.path.join(OUT, "toy-moe-split-bad.gguf"),
           small_kvs("llama", BH, BNH, BIT, [
               kv_u32("llama.expert_count", E),
               kv_u32("llama.expert_used_count", USED),
           ]), bt)

# ── toy-moe-badwidth.gguf (15.3) ────────────────────────────────────────
# Merged slabs whose DOWN projection reads 320 columns per row — not a
# whole number of Q4_K superblocks, so a slice cannot land on a block
# boundary. The real Q4_K_M witness dodges exactly this by shipping its
# 1408-wide down_exps as Q8_0; a file that claims Q4_K here is
# malformed and must be rejected WITH THE WIDTH NAMED, never
# mis-sliced. Payload is sized as ceil and zero-filled: the loader must
# refuse on shape before it ever reads a block.
WBAD = 320
WH = 512   # hidden stays Q4_K-clean so ONLY down_exps is malformed:
           # gate/up are VALID Q4_K (rows 320 is fine, cols 512 is
           # whole blocks); down's inner dim 320 is the one defect.
wt = []
def wadd(name, ne, ty, pay):
    wt.append((name, ne, ty, pay))
wadd("token_embd.weight", [WH, V], GG_F32,
     f32_ramp(V, WH).astype("<f4").tobytes())
g = "blk.0."
wadd(g + "attn_norm.weight", [WH], GG_F32,
     f32_ramp(WH, scale=0.01, bias=1.0).astype("<f4").tobytes())
wadd(g + "attn_q.weight", [WH, WH], GG_F32,
     f32_ramp(WH, WH).astype("<f4").tobytes())
wadd(g + "attn_k.weight", [WH, WH], GG_F32,
     f32_ramp(WH, WH).astype("<f4").tobytes())
wadd(g + "attn_v.weight", [WH, WH], GG_F32,
     f32_ramp(WH, WH).astype("<f4").tobytes())
wadd(g + "attn_output.weight", [WH, WH], GG_F32,
     f32_ramp(WH, WH).astype("<f4").tobytes())
wadd(g + "ffn_norm.weight", [WH], GG_F32,
     f32_ramp(WH, scale=0.01, bias=1.0).astype("<f4").tobytes())
wadd(g + "ffn_gate_inp.weight", [WH, E], GG_F32,
     router_weights(E, WH).astype("<f4").tobytes())
wadd(g + "ffn_gate_exps.weight", [WH, WBAD, E], GG_Q4_K,
     q4k_rank3(WH, WBAD, E))
wadd(g + "ffn_up_exps.weight", [WH, WBAD, E], GG_Q4_K,
     q4k_rank3(WH, WBAD, E))
zb = b"\x00" * BLOCK_BYTES
wadd(g + "ffn_down_exps.weight", [WBAD, WH, E], GG_Q4_K,
     zb * (((WBAD + 255) // 256) * WH * E))
wadd("output_norm.weight", [WH], GG_F32,
     f32_ramp(WH, scale=0.01, bias=1.0).astype("<f4").tobytes())
wadd("output.weight", [WH, V], GG_F32,
     f32_ramp(V, WH).astype("<f4").tobytes())

write_gguf(os.path.join(OUT, "toy-moe-badwidth.gguf"),
           small_kvs("qwen2moe", WH, 4, 512, [
               kv_u32("qwen2moe.expert_count", E),
               kv_u32("qwen2moe.expert_used_count", USED),
           ]), wt)
