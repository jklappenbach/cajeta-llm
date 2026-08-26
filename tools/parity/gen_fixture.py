#!/usr/bin/env python3
# cajeta-llm plan 15.2.2 — parity-fixture generation from the pinned
# reference (spec 13.21): Hugging Face transformers at fp32 ON CPU, so the
# reference contributes no noise of its own. Run under the pinned venv:
#
#   /home/julian/code/ml/venv-llama-ref/bin/python gen_fixture.py \
#       --model /home/julian/models/Meta-Llama-3.1-8B-Instruct \
#       --out   fixtures/llama31-8b
#
# Emits into --out:
#   meta.json        model id/path, sha256 of the safetensors index,
#                    transformers/torch versions, prompt text + token ids
#   logits_last.f32  final-layer logits of the LAST prompt position,
#                    raw little-endian float32, one vocab row
#   greedy.json      the 32-token greedy continuation (argmax loop)
#
# The prompt is FIXED and recorded; changing it invalidates the fixture.
import argparse, hashlib, json, os, struct, sys

PROMPT = ("The three complementary detectors of the parity gate are cosine "
          "similarity, top-1 agreement, and softmax KL divergence, measured")
GREEDY_TOKENS = 32

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cpu")
    model.eval()

    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=True)["input_ids"]
    with torch.no_grad():
        out = model(ids)
        logits = out.logits[0, -1, :].to(torch.float32).contiguous()

        # Greedy continuation, one token at a time (KV-cache-free on
        # purpose: the reference stays the simplest possible arithmetic).
        seq = ids
        greedy = []
        for _ in range(GREEDY_TOKENS):
            step = model(seq).logits[0, -1, :]
            nxt = int(torch.argmax(step).item())
            greedy.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)

    with open(os.path.join(args.out, "logits_last.f32"), "wb") as f:
        f.write(struct.pack("<%df" % logits.numel(),
                            *logits.tolist()))

    idx = os.path.join(args.model, "model.safetensors.index.json")
    sha = hashlib.sha256(open(idx, "rb").read()).hexdigest() if os.path.exists(idx) else None

    meta = {
        "model_path": args.model,
        "index_sha256": sha,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": "float32",
        "device": "cpu",
        "prompt": PROMPT,
        "add_special_tokens": True,
        "prompt_token_ids": ids[0].tolist(),
        "vocab_size": int(logits.numel()),
        "greedy_tokens": GREEDY_TOKENS,
    }
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    json.dump({"greedy": greedy}, open(os.path.join(args.out, "greedy.json"), "w"))
    print("fixture written:", args.out, "vocab", logits.numel(),
          "prompt tokens", ids.shape[1], file=sys.stderr)

if __name__ == "__main__":
    main()
