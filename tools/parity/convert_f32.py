#!/usr/bin/env python3
# cajeta-llama Unit 15 support — offline bf16 -> f32 safetensors
# conversion. The engine's forward path is f32-only today (SafetensorsFile
# .loadF32 requires stored dtype F32), so the parity run feeds it an f32
# copy of the reference checkpoint. Sharded output mirrors the input
# shard layout with a rewritten index.
import argparse, json, os
from safetensors.torch import load_file, save_file
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    for sh in shards:
        t = load_file(os.path.join(args.src, sh))
        t32 = {k: v.to(torch.float32).contiguous() for k, v in t.items()}
        save_file(t32, os.path.join(args.dst, sh))
        print("converted", sh, len(t32), "tensors")
    json.dump(idx, open(os.path.join(args.dst, "model.safetensors.index.json"), "w"))
    for extra in ("config.json", "generation_config.json"):
        p = os.path.join(args.src, extra)
        if os.path.exists(p):
            open(os.path.join(args.dst, extra), "w").write(open(p).read())
    print("done:", args.dst)

if __name__ == "__main__":
    main()
