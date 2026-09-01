#!/usr/bin/env python3
# cajeta-llm plan Unit 24 — generate
#
#   src/test/fixtures/gguf/toy-routable.gguf
#
# The FIRST fixture in this repo that can drive a forward pass through
# DEVICE code. Every other model fixture is H=16/HD=4, and the device
# routes reject it three ways over — so until now no automated test could
# reach the device path at all, and one written against a toy model was
# vacuously green. That gap cost Unit 22 (host and device attention
# disagreed for an unknown period, found by a bench arm rather than a
# gate) and has held 15.1.4 blocked since 2026-08-20.
#
# EVERY DIMENSION HERE IS FORCED BY A CONSTRAINT READ FROM THE ENGINE,
# not chosen for convenience. Change one and the fixture silently stops
# routing, which is why the test asserts routing before it asserts
# anything else:
#
#   Linear.deviceLaunchable()  packedTy >= 0        -> weights are Q4_K
#   Linear coop route          outDim % 128 == 0    -> every out below
#   QuantKernel.coopColsOk     cols % 256 == 0      -> every in below
#   DeviceKv.supports          headDim <= 128, even -> HD = 128
#                              nH % nKv == 0        -> 4 % 1
#                              no windowed layers   -> plain llama
#   attnFlashDecodeGqa4Kernel  hd == 128 && nH == 4*nKv
#
# THAT LAST ONE IS WHY nH IS 4 AND NOT 2, and it costs 1.2 MB. The decode
# attend has two implementations, and the flash one is gated on a GQA
# ratio of exactly 4 — which Llama-3.1-8B has (32 heads / 8 kv) and which
# a 2:1 fixture would MISS, quietly exercising the scalar path instead.
# A fixture that routes down kernels production never runs is a fixture
# that tests nothing; matching the shape is the entire point.
#
#     token_embd  [V=256, H=512]      F32 (an embedding lookup, not a GEMM)
#     attn_q      [NH*HD=512, H=512]  attn_k/v [NKV*HD=128, H=512]
#     attn_output [H=512, NH*HD=512]
#     ffn_gate/up [IT=512, H=512]     ffn_down [H=512, IT=512]
#     output      [V=256, H=512]
#
# THE Q4_K BYTES ARE NOT QUANTIZED HERE. They are the REAL blocks already
# checked in at kquant/q4_k.bin, pulled from the published
# Llama-3.1-8B-Instruct-Q4_K_M build, tiled cyclically to fill the larger
# tensors. gen_q4k_linear_fixture.py gives the reason and it holds here:
# re-quantizing "would introduce a second implementation of Q4_K into the
# fixtures whose only job would be to agree with the first".
#
# NO F32 TWIN FILE. The host/device comparison uses THIS model down both
# paths — same bytes, same weights, routing toggled — which is a tighter
# control than a separate f32 checkpoint (that one varies quantization
# and compute path together) and keeps the fixture under a megabyte.
#
# Deterministic: it copies bytes. Regenerating is byte-stable.
import os, struct
import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "src", "test", "fixtures", "gguf")
KQ = os.path.join(OUT, "kquant")

H, L, NH, NKV, HD, IT, V, CTX = 512, 2, 4, 1, 128, 512, 256, 256

GGUF_MAGIC = 0x46554747
T_U32, T_F32, T_STR, T_ARR, T_I32 = 4, 6, 8, 9, 5
GG_F32, GG_Q4_K = 0, 12

BLOCK_ELEMS = 256          # Q4_K superblock
BLOCK_BYTES = 144
SRC_BLOCKS = 8             # what kquant/q4_k.bin holds

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

q4k = open(os.path.join(KQ, "q4_k.bin"), "rb").read()
deq4 = np.frombuffer(open(os.path.join(KQ, "q4_k.f32"), "rb").read(),
                     dtype="<f4")
assert len(q4k) == SRC_BLOCKS * BLOCK_BYTES, len(q4k)
assert deq4.size == SRC_BLOCKS * BLOCK_ELEMS, deq4.size

def q4k_tile(rows, cols, phase=0):
    """`rows x cols` of real Q4_K blocks, tiled cyclically.

    A Q4_K row is a whole number of 256-element superblocks laid out
    along the row — which is exactly why `cols % 256 == 0` is a routing
    precondition rather than a nicety. Returns the packed payload and
    its exact dequantization, so an oracle never re-derives the values.
    `phase` offsets the tiling so different experts' slabs are not the
    same bytes (gen_moe_fixture's rule: a mis-sliced expert view must
    not accidentally agree with the oracle). phase=0 keeps toy-routable
    and toy-routable-qk BYTE-IDENTICAL to their calibrated builds.
    """
    assert cols % BLOCK_ELEMS == 0, (rows, cols)
    bpr = cols // BLOCK_ELEMS
    pay, deq = [], []
    for i in range(rows * bpr):
        k = (i + phase) % SRC_BLOCKS
        pay.append(q4k[k * BLOCK_BYTES:(k + 1) * BLOCK_BYTES])
        deq.append(deq4[k * BLOCK_ELEMS:(k + 1) * BLOCK_ELEMS])
    return b"".join(pay), np.concatenate(deq).reshape(rows, cols)

def f32_ramp(rows, cols=None, scale=0.02, bias=0.0):
    """Small, exactly-representable values on a 1/64 grid — the same
    reasoning as the other fixtures: a toy forward pass must stay
    finite, and a value that survives f32 round-tripping keeps any
    disagreement attributable to the PATH rather than to storage."""
    n = rows if cols is None else rows * cols
    a = ((np.arange(n) % 25) - 12).astype(np.float32) / 64.0 * (scale * 64.0)
    a = a + bias
    shape = (rows,) if cols is None else (rows, cols)
    return a.reshape(shape).astype(np.float32)

# (gguf_name, ne fastest-dim-first, ggml type, payload)
#
# Unit 29 adds a SECOND fixture from the same builder: toy-routable-qk
# is toy-routable plus per-head q/k RMS norms under arch `qwen3` — the
# dense Qwen3 shape, whose loader path sets qkNorm from the arch the
# same way qwen3moe does. Same Q4_K bytes, same dims, one variable: the
# norm pair between projection and rope. toy-routable itself must stay
# BYTE-IDENTICAL (its oracles and parity thresholds are calibrated), so
# the builder writes it first and the md5 is asserted in the repo test
# suite by simply not regenerating expectations.

def build(arch, qk_norm):
    tensors = []
    def add_q4k(name, rows, cols):
        pay, _deq = q4k_tile(rows, cols)
        tensors.append((name, [cols, rows], GG_Q4_K, pay))
    def add_f32(name, arr):
        ne = list(reversed(arr.shape))
        tensors.append((name, ne, GG_F32, arr.astype("<f4").tobytes()))

    add_f32("token_embd.weight", f32_ramp(V, H))
    for i in range(L):
        g = f"blk.{i}."
        add_f32(g + "attn_norm.weight", f32_ramp(H, scale=0.01, bias=1.0))
        add_q4k(g + "attn_q.weight",      NH * HD, H)
        add_q4k(g + "attn_k.weight",      NKV * HD, H)
        add_q4k(g + "attn_v.weight",      NKV * HD, H)
        add_q4k(g + "attn_output.weight", H, NH * HD)
        if qk_norm:
            # Per-head norms, [HD]. bias=1.0 keeps them near identity so
            # the fixture's activations stay in the regime the parity
            # thresholds were calibrated in; scale=0.01 makes them NOT
            # identity, so a path that skips the norm fails parity
            # instead of passing vacuously.
            add_f32(g + "attn_q_norm.weight",
                    f32_ramp(HD, scale=0.01, bias=1.0))
            add_f32(g + "attn_k_norm.weight",
                    f32_ramp(HD, scale=0.01, bias=1.05))
        add_f32(g + "ffn_norm.weight", f32_ramp(H, scale=0.01, bias=1.0))
        add_q4k(g + "ffn_gate.weight", IT, H)
        add_q4k(g + "ffn_up.weight",   IT, H)
        add_q4k(g + "ffn_down.weight", H, IT)
    add_f32("output_norm.weight", f32_ramp(H, scale=0.01, bias=1.0))
    add_q4k("output.weight", V, H)

    # A byte vocabulary, exactly V tokens: no merges to get wrong, and the
    # vocab size is then the same constraint-satisfying 256 the output
    # projection needs rather than a second number to keep in step.
    toks = [f"<0x{b:02X}>" for b in range(V)]
    types = [6] * V
    scores = [0.0] * V

    kp = arch + "."
    kvs = [
        kv_str("general.architecture", arch),
        kv_str("general.name", "cajeta-toy-routable"
               + ("-qk" if qk_norm else "")),
        kv_u32(kp + "embedding_length", H),
        kv_u32(kp + "block_count", L),
        kv_u32(kp + "attention.head_count", NH),
        kv_u32(kp + "attention.head_count_kv", NKV),
        kv_u32(kp + "feed_forward_length", IT),
        kv_u32(kp + "context_length", CTX),
        kv_u32(kp + "rope.dimension_count", HD),
        kv_u32(kp + "vocab_size", V),
        kv_f32(kp + "attention.layer_norm_rms_epsilon", 1e-5),
        kv_f32(kp + "rope.freq_base", 10000.0),
        kv_str("tokenizer.ggml.model", "llama"),
        kv_arr_str("tokenizer.ggml.tokens", toks),
        kv_arr_f32("tokenizer.ggml.scores", scores),
        kv_arr_i32("tokenizer.ggml.token_type", types),
    ]
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

    # Restate the routing preconditions as ASSERTS, so a future dimension
    # edit fails HERE rather than as a mysteriously vacuous test.
    assert HD <= 128 and HD % 2 == 0, "DeviceKv.supports: headDim"
    assert NH % NKV == 0, "DeviceKv.supports: head_count % head_count_kv"
    for gg, ne, ty, _pay in tensors:
        if ty != GG_Q4_K:
            continue
        cols, rows = ne[0], ne[1]
        assert cols % 256 == 0, (gg, "coopColsOk: cols % 256", cols)
        assert rows % 128 == 0, (gg, "coop route: outDim % 128", rows)
    print("routing preconditions hold for all",
          sum(1 for t in tensors if t[2] == GG_Q4_K), "Q4_K tensors")

kvs, tensors = build("llama", False)
write("toy-routable.gguf", kvs, tensors)
kvs, tensors = build("qwen3", True)
write("toy-routable-qk.gguf", kvs, tensors)

# Unit 38 adds the THIRD fixture: toy-routable-moe is the routable dims
# under arch `qwen3moe` — MoE (4 experts, top-2, expert width 512) plus
# the per-head q/k norms the arch implies. This is the 30B's SHAPE
# CLASS (the shape `deviceResidentReady` refused before unit 38) at
# fixture scale: every tensor still satisfies the routing preconditions
# above, so the resident decode path, the grouped expert dispatch and
# the slot mat-vecs all engage on it.
E, USED, IT_E = 4, 2, 512

def build_moe():
    tensors = []
    def add_q4k(name, rows, cols, ne=None, phase=0):
        pay, _deq = q4k_tile(rows, cols, phase)
        tensors.append((name, ne if ne else [cols, rows], GG_Q4_K, pay))
    def add_f32(name, arr):
        ne = list(reversed(arr.shape))
        tensors.append((name, ne, GG_F32, arr.astype("<f4").tobytes()))

    def router_weights(rows, cols):
        # gen_moe_fixture's rule: each expert's row peaks on a
        # different residual phase, so top-2 identity varies by token
        # and a top-k bug cannot hide behind a constant selection.
        a = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            idx = (np.arange(cols) + 7 * r) % 25
            a[r] = ((idx - 12).astype(np.float32) / 64.0) * 1.28
        return a.astype(np.float32)

    add_f32("token_embd.weight", f32_ramp(V, H))
    for i in range(L):
        g = f"blk.{i}."
        add_f32(g + "attn_norm.weight", f32_ramp(H, scale=0.01, bias=1.0))
        add_q4k(g + "attn_q.weight",      NH * HD, H)
        add_q4k(g + "attn_k.weight",      NKV * HD, H)
        add_q4k(g + "attn_v.weight",      NKV * HD, H)
        add_q4k(g + "attn_output.weight", H, NH * HD)
        add_f32(g + "attn_q_norm.weight", f32_ramp(HD, scale=0.01, bias=1.0))
        add_f32(g + "attn_k_norm.weight", f32_ramp(HD, scale=0.01, bias=1.05))
        add_f32(g + "ffn_norm.weight", f32_ramp(H, scale=0.01, bias=1.0))
        add_f32(g + "ffn_gate_inp.weight", router_weights(E, H))
        # Merged rank-3 slabs, expert-major (the ONLY layout the loader
        # accepts); per-expert AND per-bank phases keep every slice's
        # bytes distinct.
        for bank, base in (("gate", 0), ("up", 2), ("down", 4)):
            rows, cols = (H, IT_E) if bank == "down" else (IT_E, H)
            pay = b"".join(
                q4k_tile(rows, cols, phase=base + e)[0] for e in range(E))
            tensors.append((f"{g}ffn_{bank}_exps.weight",
                            [cols, rows, E], GG_Q4_K, pay))
    add_f32("output_norm.weight", f32_ramp(H, scale=0.01, bias=1.0))
    add_q4k("output.weight", V, H)

    toks = [f"<0x{b:02X}>" for b in range(V)]
    kp = "qwen3moe."
    kvs = [
        kv_str("general.architecture", "qwen3moe"),
        kv_str("general.name", "cajeta-toy-routable-moe"),
        kv_u32(kp + "embedding_length", H),
        kv_u32(kp + "block_count", L),
        kv_u32(kp + "attention.head_count", NH),
        kv_u32(kp + "attention.head_count_kv", NKV),
        kv_u32(kp + "feed_forward_length", IT),
        kv_u32(kp + "context_length", CTX),
        kv_u32(kp + "rope.dimension_count", HD),
        kv_u32(kp + "vocab_size", V),
        kv_u32(kp + "expert_count", E),
        kv_u32(kp + "expert_used_count", USED),
        kv_u32(kp + "expert_feed_forward_length", IT_E),
        kv_f32(kp + "attention.layer_norm_rms_epsilon", 1e-5),
        kv_f32(kp + "rope.freq_base", 10000.0),
        kv_str("tokenizer.ggml.model", "llama"),
        kv_arr_str("tokenizer.ggml.tokens", toks),
        kv_arr_f32("tokenizer.ggml.scores", [0.0] * V),
        kv_arr_i32("tokenizer.ggml.token_type", [6] * V),
    ]
    return kvs, tensors

kvs, tensors = build_moe()
write("toy-routable-moe.gguf", kvs, tensors)
