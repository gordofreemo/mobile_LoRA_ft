#!/usr/bin/env python3
"""Mac control arm for h13 — train per-user OPPU movie-tagging User-LoRAs on the
EXACT 4-bit model the phone loads (a quantisation of THEIR merged movie task
adapter), consuming the SAME side-loaded pre-tokenized JSONL, with the same
recipe as the device harness:

  r=8, scale 2.0 (alpha 16), q+v only, all 36 layers, no dropout;
  effective batch 8 = 8 microbatches of batch 1, token-weighted window
  normalization (sum CE / sum masked tokens); global-norm grad clip 1.0;
  AdamW beta 0.9/0.999 eps 1e-8 weight_decay 1e-2 bias-corrected;
  cosine LR 1e-5 with ceil(0.03 x total) warmup; 3 epochs, windows never
  straddling an epoch boundary (HF restarts accumulation each epoch).

The corpus file holds 3 epochs in their baked-in permuted order, so the device
and this control consume byte-identical sequences -- which is what makes the
device-vs-Mac loss comparison a fidelity check rather than a data-order
comparison.

Run:
  .venv-mlx/bin/python train/train_user_mlx_h13.py --users 8000865
  .venv-mlx/bin/python train/train_user_mlx_h13.py --all
"""

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))

BASE_LR = 1e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 1e-2
ACCUM_WINDOW = 8
EPOCHS = 3
GRAD_CLIP = 1.0
LORA_RANK = 8
LORA_SCALE = 2.0            # alpha 16 / r 8
LORA_KEYS = ["self_attn.q_proj", "self_attn.v_proj"]
NUM_LORA_LAYERS = 36


def cosine_lr(step, total_steps):
    """HF cosine-with-warmup, 0-based step (same function the device uses)."""
    warmup = math.ceil(total_steps * WARMUP_RATIO)
    if step < warmup:
        return BASE_LR * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def windows(n_per_epoch):
    """(start, count) into the flat 3-epoch corpus; never straddles an epoch."""
    out = []
    for e in range(EPOCHS):
        i = 0
        while i < n_per_epoch:
            out.append((e * n_per_epoch + i, min(ACCUM_WINDOW, n_per_epoch - i)))
            i += ACCUM_WINDOW
    return out


def git_info():
    def run(a):
        try:
            return subprocess.check_output(a, cwd=PROJECT_ROOT, text=True).strip()
        except Exception:
            return None
    return run(["git", "rev-parse", "--short", "HEAD"]), bool(run(["git", "status", "--porcelain"]))


def train_user(model, reset_state, user_dir, out_dir, commit, dirty, max_steps=0):
    examples = []
    with open(user_dir / "train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            assert 1 <= r["loss_start"] < r["loss_end"] <= len(r["input_ids"])
            examples.append((r["id"], r["input_ids"], r["loss_start"], r["loss_end"]))
    n = len(examples)
    assert n % EPOCHS == 0, f"{user_dir}: {n} examples is not {EPOCHS} whole epochs"
    n_per_epoch = n // EPOCHS
    wins = windows(n_per_epoch)
    schedule_total_steps = len(wins)
    total_steps = min(schedule_total_steps, max_steps) if max_steps > 0 else schedule_total_steps

    reset_state()
    opt = optim.AdamW(learning_rate=BASE_LR, betas=[0.9, 0.999], eps=1e-8,
                      weight_decay=WEIGHT_DECAY, bias_correction=True)

    def loss_fn(model, inputs, targets, mask):
        logits = model(inputs).astype(mx.float32)
        return (nn.losses.cross_entropy(logits, targets) * mask).sum(), mask.sum()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    out_dir.mkdir(parents=True, exist_ok=True)
    mf = open(out_dir / "metrics.jsonl", "w")
    meta = {
        "run_name": out_dir.name, "task": "LaMP_2M", "protocol": "oppu_k1_r5",
        "arm": "mac_control", "user_id": user_dir.name,
        "n_examples": n_per_epoch, "epochs": EPOCHS,
        "total_steps": total_steps, "schedule_total_steps": schedule_total_steps,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_RANK * LORA_SCALE,
        "lora_scale": LORA_SCALE, "lora_keys": LORA_KEYS,
        "num_lora_layers": NUM_LORA_LAYERS,
        "base_learning_rate": BASE_LR, "warmup_ratio": WARMUP_RATIO,
        "accum_window": ACCUM_WINDOW, "grad_clip_norm": GRAD_CLIP,
        "weight_decay": WEIGHT_DECAY, "adam_bias_correction": True,
        "first_example_ids": [e[0] for e in examples[:3]],
        "total_corpus_tokens": sum(len(e[1]) for e in examples),
        "git_commit": commit, "git_dirty": dirty, "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mlx_version": mx.__version__,
    }
    json.dump(meta, open(out_dir / "adapter_meta.json", "w"), indent=2)
    mf.write(json.dumps({"record_type": "run_start", **meta}) + "\n")

    t0 = time.time()
    for step in range(total_steps):
        step_start = time.time()
        start, micro_count = wins[step]
        accum = None
        window_ce = window_tokens = 0.0
        micro_losses, micro_seq_lens = [], []
        for j in range(micro_count):
            _id, ids, ls, le = examples[start + j]
            ntok = len(ids)
            full = mx.array(ids, dtype=mx.int32)[None, :]
            inputs, targets = full[:, :-1], full[:, 1:]
            mask = mx.zeros((1, ntok - 1), dtype=mx.float32)
            mask[:, ls - 1:le - 1] = 1.0
            (ce_sum, ntoks), grads = loss_and_grad(model, inputs, targets, mask)
            accum = grads if accum is None else tree_map(lambda a, b: a + b, accum, grads)
            mx.eval(accum, ce_sum, ntoks)
            ce_f, nt_f = float(ce_sum), float(ntoks)
            window_ce += ce_f
            window_tokens += nt_f
            micro_losses.append(ce_f / nt_f if nt_f else 0.0)
            micro_seq_lens.append(ntok)

        normalized = tree_map(lambda g: g / window_tokens, accum)
        clipped, grad_norm = optim.clip_grad_norm(normalized, GRAD_CLIP)
        lr = cosine_lr(step, schedule_total_steps)
        opt.learning_rate = lr
        opt.update(model, clipped)
        mx.eval(model.trainable_parameters(), opt.state)

        mf.write(json.dumps({
            "record_type": "opt_step", "step": step + 1,
            "epoch": step // max(1, schedule_total_steps // EPOCHS),
            "loss": window_ce / window_tokens if window_tokens else 0.0,
            "learning_rate": lr, "grad_norm_preclip": float(grad_norm),
            "n_micro": micro_count, "window_masked_tokens": int(window_tokens),
            "window_seq_tokens": sum(micro_seq_lens),
            "window_s": time.time() - step_start, "elapsed_s": time.time() - t0,
            "micro_losses": micro_losses, "micro_seq_lens": micro_seq_lens,
        }) + "\n")
    mf.flush()

    adapter = out_dir / "adapters.safetensors"
    mx.save_safetensors(str(adapter), dict(tree_flatten(model.trainable_parameters())))
    elapsed = time.time() - t0
    mf.write(json.dumps({"record_type": "run_end", "elapsed_s": elapsed,
                         "adapter": str(adapter)}) + "\n")
    mf.close()
    return total_steps, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data/oppu_movie/device"))
    ap.add_argument("--model", default=str(PROJECT_ROOT / "data/models/SmolLM3-3B-oppu-movie-mlx-4bit"))
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "train/checkpoints_mlx/h13_mac_control"))
    ap.add_argument("--users", default=None, help="comma-separated user_ids")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--queue", default=str(PROJECT_ROOT / "data/oppu_movie/h13_queue.json"),
                    help="--all follows this frozen queue order")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    data_root, out_root = Path(args.data_root), Path(args.out_root)
    if args.users:
        users = args.users.split(",")
    elif args.all:
        users = [e["user_id"] for e in json.load(open(args.queue))["queue"]]
    else:
        sys.exit("pass --users <ids> or --all")

    commit, dirty = git_info()
    print(f"[h13_mac] model={args.model} users={len(users)} commit={commit} dirty={dirty}",
          flush=True)

    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, _ = load(args.model)
    model.freeze()
    linear_to_lora_layers(
        model, NUM_LORA_LAYERS,
        {"rank": LORA_RANK, "scale": LORA_SCALE, "dropout": 0.0, "keys": LORA_KEYS})
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"[h13_mac] LoRA applied: {n_trainable} trainable params", flush=True)

    # Every user starts from the SAME fresh adapter (lora_a at its init draw,
    # lora_b zero) -- the cluster's per-user independence guarantee (P12).
    pristine = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}

    def reset_state():
        model.update(tree_unflatten([(k, mx.array(v)) for k, v in pristine.items()]))
        mx.eval(model.trainable_parameters())

    for i, uid in enumerate(users):
        out_dir = out_root / uid
        if args.skip_existing and (out_dir / "adapters.safetensors").exists():
            print(f"[h13_mac] {i+1}/{len(users)} {uid} SKIP (exists)", flush=True)
            continue
        steps, elapsed = train_user(model, reset_state, data_root / uid, out_dir,
                                    commit, dirty, args.max_steps)
        print(f"[h13_mac] {i+1}/{len(users)} {uid} {steps} steps in {elapsed:.0f}s "
              f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
