#!/usr/bin/env python3
"""Mac-side PRE-CHECK of the h13 eval plane (NOT the deliverable).

The paper's eval plane is the phone. This script runs the same MLX code path,
the same 4-bit merged base and the same sampler on the M3, purely to answer
Phase 1's question -- "does the OPPU movie effect survive 4-bit MLX deployment
at all" -- before spending device nights on it. Any number it prints is labelled
mac-plane and never mixed with device results.

Loads the base once, installs the LoRA structure once, and swaps only the
per-user lora_a/lora_b weights. The RAG arm is the identical graph with
lora_b = 0, which is exactly the base model (y + scale*(x@A@0) == y).
"""
import argparse, json, time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]
LORA_CFG = {"rank": 8, "scale": 2.0, "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
NUM_LAYERS = 36
LABELS = ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia",
          "dark comedy", "classic", "psychology", "fantasy", "romance",
          "thought-provoking", "social commentary", "violence", "true story"]


def score(pred, gold):
    """third_party/OPPU/eval/evaluation.py _get_labels + exact index match."""
    def m(x):
        try:
            return LABELS.index(str(x).strip())
        except ValueError:
            return -1
    return float(m(pred) == m(gold) and m(gold) != -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data/models/SmolLM3-3B-oppu-movie-mlx-4bit"))
    ap.add_argument("--texts", default=str(ROOT / "data/oppu_movie/h13_movie_texts.json"))
    ap.add_argument("--adapters", default=str(ROOT / "data/oppu_movie/mlx_adapters"))
    ap.add_argument("--out", default=str(ROOT / "results/ondevice/h13_mac_precheck.jsonl"))
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, tok = load(args.model)
    linear_to_lora_layers(model, NUM_LAYERS, LORA_CFG)
    model.eval()
    sampler = make_sampler(temp=0.1, top_p=0.9, top_k=10)

    def set_arm(adapter_dir):
        """adapter_dir=None -> RAG arm (lora_b zeroed == exact base)."""
        if adapter_dir is None:
            upd = []
            for name, mod in model.named_modules():
                if hasattr(mod, "lora_b"):
                    upd.append((f"{name}.lora_b", mx.zeros(mod.lora_b.shape)))
        else:
            w = load_file(str(Path(adapter_dir) / "adapters.safetensors"))
            upd = [(k, mx.array(v)) for k, v in w.items()]
        model.load_weights(upd, strict=False)
        mx.eval(model.parameters())

    data = json.load(open(args.texts))[: args.n_users]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "w")
    totals = {"rag": [0.0, 0], "cluster": [0.0, 0]}
    changed = 0
    t0 = time.time()

    for u in data:
        preds = {}
        for arm in ("rag", "cluster"):
            set_arm(None if arm == "rag" else Path(args.adapters) / u["user_id"])
            got = []
            for q in u["queries"]:
                mx.random.seed(args.seed)
                got.append(generate(model, tok, prompt=q["prompt"], max_tokens=200,
                                    sampler=sampler, verbose=False).strip())
            preds[arm] = got
            totals[arm][0] += sum(score(p, q["gold"]) for p, q in zip(got, u["queries"]))
            totals[arm][1] += len(got)
        for j, q in enumerate(u["queries"]):
            r, c = preds["rag"][j], preds["cluster"][j]
            changed += (r != c)
            fh.write(json.dumps({"user_id": u["user_id"], "id": q["id"], "gold": q["gold"],
                                 "rag": r, "cluster": c,
                                 "rag_score": score(r, q["gold"]),
                                 "cluster_score": score(c, q["gold"])}) + "\n")
        fh.flush()
        a = totals["rag"][0] / max(1, totals["rag"][1])
        b = totals["cluster"][0] / max(1, totals["cluster"][1])
        print(f"[{u['user_index']:3d}] {u['user_id']} n={totals['rag'][1]:5d} "
              f"rag={a:.4f} cluster={b:.4f} delta={b-a:+.4f} changed={changed} "
              f"({time.time()-t0:.0f}s)", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
