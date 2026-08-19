#!/usr/bin/env python3
"""Build the h13 per-user device/Mac corpora from the OPPU ground-truth texts.

Input  : data/oppu_movie/h13_movie_texts.json  (dumped BY THEIR CODE on conduit)
Output : data/oppu_movie/device/<user_id>/{train.jsonl,eval.jsonl,meta.json}

Training records replicate run_oppu.py's `generate_and_tokenize_prompt` exactly:
  full_ids  = tok(full_prompt, truncation, max_len=2048); append EOS if absent
  prompt_len= len(tok(prompt, no EOS))
  labels    = -100 * prompt_len ++ full_ids[prompt_len:]
so loss_start = prompt_len, loss_end = len(full_ids).

The file order IS the training order: 3 epochs, each a seeded permutation
(numpy default_rng(SEED)). The cluster's own shuffle is UNSEEDED, so byte-order
parity with it is unobtainable; what this buys is exact device-vs-Mac
comparability, which is what the fidelity check needs.

Eval records carry pre-tokenized prompts so the device never tokenizes.
"""
import argparse, json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CUTOFF = 2048          # run_oppu.py --cut_off
EPOCHS = 3             # R5 recipe
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=ROOT / "data/oppu_movie/h13_movie_texts.json")
    ap.add_argument("--tokenizer", default=ROOT / "data/models/SmolLM3-3B-oppu-movie-mlx-4bit")
    ap.add_argument("--out", default=ROOT / "data/oppu_movie/device")
    ap.add_argument("--users", default=None, help="comma-separated user_ids (default: all)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    eos = tok.eos_token_id
    data = json.load(open(args.texts))
    want = set(args.users.split(",")) if args.users else None
    out_root = Path(args.out)

    n_prefix_fail = n_trunc = 0
    manifest = []
    for u in data:
        if want and u["user_id"] not in want:
            continue
        recs = []
        for j, e in enumerate(u["train"]):
            full = tok(e["full_prompt"], truncation=True, max_length=CUTOFF,
                       padding=False)["input_ids"]
            if full[-1] != eos and len(full) < CUTOFF:
                full.append(eos)
            else:
                n_trunc += 1
            plen = len(tok(e["prompt"], truncation=True, max_length=CUTOFF,
                           padding=False)["input_ids"])
            if full[:plen] != tok(e["prompt"], truncation=True, max_length=CUTOFF,
                                  padding=False)["input_ids"]:
                n_prefix_fail += 1
            if not (1 <= plen < len(full)):
                raise SystemExit(f"user {u['user_id']} ex {j}: bad span {plen}/{len(full)}")
            recs.append({"id": f"{u['user_id']}_p{j:04d}", "input_ids": full,
                         "loss_start": plen, "loss_end": len(full)})

        rng = np.random.default_rng(SEED)
        order = [i for _ in range(EPOCHS) for i in rng.permutation(len(recs))]

        d = out_root / u["user_id"]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "train.jsonl", "w") as f:
            for i in order:
                f.write(json.dumps(recs[i]) + "\n")

        with open(d / "eval.jsonl", "w") as f:
            for q in u["queries"]:
                ids = tok(q["prompt"], truncation=True, max_length=CUTOFF,
                          padding=False)["input_ids"]
                f.write(json.dumps({"id": q["id"], "input_ids": ids,
                                    "n_prompt_tokens": len(ids)}) + "\n")

        n_per_epoch = len(recs)
        steps = EPOCHS * ((n_per_epoch + 7) // 8)
        meta = {"user_id": u["user_id"], "user_index": u["user_index"],
                "n_examples": n_per_epoch, "epochs": EPOCHS, "accum_window": 8,
                "total_steps": steps, "n_queries": len(u["queries"]),
                "train_tokens_per_epoch": sum(len(r["input_ids"]) for r in recs),
                "max_example_tokens": max(len(r["input_ids"]) for r in recs),
                "seed": SEED, "cutoff": CUTOFF}
        json.dump(meta, open(d / "meta.json", "w"), indent=1)
        manifest.append(meta)
        print(f"{u['user_id']}: {n_per_epoch} ex x{EPOCHS} -> {steps} steps, "
              f"{len(u['queries'])} queries, max {meta['max_example_tokens']} tok")

    json.dump(manifest, open(out_root / "manifest.json", "w"), indent=1)
    print(f"\n{len(manifest)} users written -> {out_root}")
    print(f"prefix-property violations: {n_prefix_fail} (must be 0)")
    print(f"examples hitting the {CUTOFF} cutoff (no EOS appended): {n_trunc}")


if __name__ == "__main__":
    main()
