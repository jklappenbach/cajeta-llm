#!/usr/bin/env python3
"""Reference tokenizations for the llama-bpe pre-tokenizer (plan 15.1.14).

llama-bpe and qwen2 are both byte-level BPE and differ essentially in how
they group digits: LLAMA3 splits runs of up to THREE digits (\\p{N}{1,3}),
qwen2 splits every digit singly (\\p{N}). Cases below are chosen to make
that difference visible — anything with 1..4+ digit runs tokenizes
differently under the two rules, so a blind "accept llama-bpe as qwen2"
would silently mis-tokenize every number.

Emits a TSV the cajeta suite reads: text<TAB>id,id,id
"""
import json, sys
from transformers import AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/julian/models/Meta-Llama-3.1-8B-Instruct"
OUT = sys.argv[2] if len(sys.argv) > 2 else "src/test/fixtures/tok/llama_bpe.tsv"

CASES = [
    "hello world",
    "The capital of France is",
    # digit runs: 1, 2, 3, 4, 5 and 6 digits — the whole point
    "1", "12", "123", "1234", "12345", "123456",
    "in 2026 the answer was 42",
    "pi is 3.14159",
    "v0.22.0",
    "1,000,000",
    # the contraction alternation both regexes share
    "it's don't we're I've I'm we'll he'd",
    # whitespace and newline runs
    "a  b\tc\nd\n\ne",
    "   leading",
    "trailing   ",
    # non-latin + punctuation
    "café — naïve",
    "日本語のテキスト",
    "emoji 🙂 and 🇯🇵",
    "mixed42text99here",
]

tok = AutoTokenizer.from_pretrained(MODEL)
pre = json.load(open(f"{MODEL}/tokenizer.json"))["pre_tokenizer"]
rows = []
for c in CASES:
    ids = tok.encode(c, add_special_tokens=False)
    rows.append((c, ids))

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for c, ids in rows:
        esc = c.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
        f.write(esc + "\t" + ",".join(str(i) for i in ids) + "\n")
print(f"wrote {len(rows)} cases -> {OUT}")
print("pre_tokenizer:", json.dumps(pre)[:200])
