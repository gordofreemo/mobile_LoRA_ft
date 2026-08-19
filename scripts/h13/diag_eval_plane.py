#!/usr/bin/env python3
"""Diagnose the h13 eval plane. Two questions, one script:

  Q1 (level)  the cluster RAG arm scores 0.4933 on bf16/HF; the 4-bit MLX plane
              scores ~0.36. Is that quantisation, or decoding?
  Q2 (delta)  the personalization effect is +0.0763 (R5 recipe) on bf16/HF.
              Does it reproduce on bf16 MLX? If yes, the h13 pipeline is sound
              and any shrinkage at 4-bit is a real deployment finding. If no,
              the pipeline differs from theirs somewhere and must be fixed
              BEFORE any device night is spent.

Arms: {bf16, 4bit} x {rag, cluster} x {their sampler, greedy}, same queries.
"""
import argparse, json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]
LORA_CFG = {"rank": 8, "scale": 2.0, "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
LABELS = ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia",
          "dark comedy", "classic", "psychology", "fantasy", "romance",
          "thought-provoking", "social commentary", "violence", "true story"]


def score(pred, gold):
    def m(x):
        try:
            return LABELS.index(str(x).strip())
        except ValueError:
            return -1
    return float(m(pred) == m(gold) and m(gold) != -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=25)
    ap.add_argument("--greedy", action="store_true", help="also run the greedy arm")
    ap.add_argument("--out", default=str(ROOT / "results/ondevice/h13_diag_eval_plane.json"))
    args = ap.parse_args()

    data = json.load(open(ROOT / "data/oppu_movie/h13_movie_texts.json"))[: args.n_users]
    nq = sum(len(u["queries"]) for u in data)
    print(f"{args.n_users} users, {nq} queries")
    adapters = ROOT / "data/oppu_movie/mlx_adapters"

    results = {}
    for mtag, mpath in (("bf16", "SmolLM3-3B-oppu-movie-mlx-bf16"),
                        ("4bit", "SmolLM3-3B-oppu-movie-mlx-4bit")):
        p = ROOT / "data/models" / mpath
        if not p.exists():
            print(f"SKIP {mtag}: missing {p}")
            continue
        model, tok = load(str(p))
        linear_to_lora_layers(model, 36, LORA_CFG)

        def set_arm(user_id):
            if user_id is None:      # rag: lora_b = 0 is exactly the base model
                upd = [(f"{n}.lora_b", mx.zeros(m.lora_b.shape))
                       for n, m in model.named_modules() if hasattr(m, "lora_b")]
            else:
                upd = [(k, mx.array(v)) for k, v in
                       load_file(str(adapters / user_id / "adapters.safetensors")).items()]
            model.load_weights(upd, strict=False)
            mx.eval(model.parameters())

        decodings = [("their_sampler", make_sampler(temp=0.1, top_p=0.9, top_k=10))]
        if args.greedy:
            decodings.append(("greedy", make_sampler(temp=0.0)))
        for dtag, sampler in decodings:
            per_arm = {}
            for arm in ("rag", "cluster"):
                acc = inv = 0
                for u in data:
                    set_arm(None if arm == "rag" else u["user_id"])
                    for q in u["queries"]:
                        mx.random.seed(0)
                        o = generate(model, tok, prompt=q["prompt"], max_tokens=200,
                                     sampler=sampler, verbose=False).strip()
                        acc += score(o, q["gold"])
                        inv += (o not in LABELS)
                per_arm[arm] = {"accuracy": acc / nq, "invalid_rate": inv / nq}
                print(f"  {mtag}/{dtag}/{arm:8s} acc={acc/nq:.4f} invalid={inv/nq:.3f}",
                      flush=True)
            per_arm["delta_cluster_minus_rag"] = (
                per_arm["cluster"]["accuracy"] - per_arm["rag"]["accuracy"])
            per_arm["n_queries"] = nq
            results[f"{mtag}/{dtag}"] = per_arm
            print(f"  {mtag}/{dtag} DELTA = {per_arm['delta_cluster_minus_rag']:+.4f}",
                  flush=True)
        del model

    results["_reference_bf16_hf_cluster"] = {
        "rag": 0.4933, "oppu_hot": 0.5845, "delta_hot": 0.0912, "delta_r5": 0.0763}
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
