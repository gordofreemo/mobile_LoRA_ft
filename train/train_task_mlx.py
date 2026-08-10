#!/usr/bin/env python3
"""Mac control arm for h12 — train the Per-Task-LoRA (LaMP-7) on the EXACT
4-bit model the phone loads (mlx-community/SmolLM3-3B-4bit), consuming the
SAME pre-tokenized JSONL (data/build_task_device_data.py output), with the
same recipe as the device harness:

  r=4, scale 2.0 (alpha 8), q/k/v/o/gate/up/down, all 36 layers, no dropout;
  effective batch 32 = 32 microbatches of batch 1, token-weighted window
  normalization (sum CE / sum masked tokens); global-norm grad clip 1.0;
  AdamW beta 0.9/0.999 eps 1e-8 wd 0.0 bias-corrected; cosine LR 3e-4 with
  ceil(0.03 x total) warmup steps; ONE pass over the file in its baked-in
  (seed-0 shuffled) order.

This isolates the h12 deviation bundle {4-bit base, no dropout, accumulation
ordering} on known-good silicon, and its first-window micro losses are the
smoke reference for the device (same model, same data, same math -> the
device's first-microbatch losses must match within float tolerance).

Run:
  .venv-mlx/bin/python train/train_task_mlx.py                     # full
  .venv-mlx/bin/python train/train_task_mlx.py --max-steps 20      # smoke ref
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
from mlx.utils import tree_flatten, tree_map

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))

BASE_LR = 3e-4
WARMUP_RATIO = 0.03
ACCUM_WINDOW = 32
GRAD_CLIP = 1.0
LORA_RANK = 4
LORA_SCALE = 2.0  # alpha 8 / r 4
LORA_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]
NUM_LORA_LAYERS = 36


def cosine_lr(step, total_steps):
    """HF cosine-with-warmup, 0-based step; verified against the cluster
    reference's metrics.jsonl (step 10 -> 2.7e-4, step 20 -> 2.9940e-4)."""
    warmup = math.ceil(total_steps * WARMUP_RATIO)
    if step < warmup:
        return BASE_LR * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return BASE_LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def git_info():
    def run(args):
        try:
            return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()
        except Exception:
            return None
    return run(["git", "rev-parse", "--short", "HEAD"]), bool(run(["git", "status", "--porcelain"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data/ondevice_task_data/lamp7_task.jsonl"))
    ap.add_argument("--model", default="mlx-community/SmolLM3-3B-4bit")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "train/checkpoints_mlx/pt_lamp7_mac_control"))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap optimizer steps (smoke: 20); LR schedule always uses the full count")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="mx.checkpoint per block (matches device GC; numerically identical)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.max_steps > 0:
        out_dir = out_dir.with_name(out_dir.name + f"_steps{args.max_steps}")
    metrics_path = out_dir / "metrics.jsonl"
    adapter_path = out_dir / "adapters.safetensors"

    commit, dirty = git_info()
    print(f"[train_task_mlx] data={args.data} model={args.model} out={out_dir} "
          f"max_steps={args.max_steps} grad_checkpoint={args.grad_checkpoint} "
          f"commit={commit} dirty={dirty} host={socket.gethostname()}", flush=True)

    if adapter_path.exists() and not args.overwrite:
        print(f"REFUSING to overwrite {adapter_path} (pass --overwrite)", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, _tokenizer = load(args.model)
    model.freeze()
    linear_to_lora_layers(
        model, NUM_LORA_LAYERS,
        {"rank": LORA_RANK, "scale": LORA_SCALE, "dropout": 0.0, "keys": LORA_KEYS})
    n_trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"[train_task_mlx] LoRA applied: {n_trainable} trainable params", flush=True)

    if args.grad_checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint
        grad_checkpoint(model.layers[0])

    examples = []
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            assert 1 <= r["loss_start"] < r["loss_end"] <= len(r["input_ids"])
            examples.append((r["id"], r["input_ids"], r["loss_start"], r["loss_end"]))
    n = len(examples)
    schedule_total_steps = (n + ACCUM_WINDOW - 1) // ACCUM_WINDOW
    total_steps = schedule_total_steps
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    print(f"[train_task_mlx] n={n} steps={total_steps}/{schedule_total_steps} "
          f"first_ids={[e[0] for e in examples[:3]]}", flush=True)

    def loss_fn(model, inputs, targets, mask):
        logits = model(inputs).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, targets) * mask
        return ce.sum(), mask.sum()

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    opt = optim.AdamW(learning_rate=BASE_LR, betas=[0.9, 0.999], eps=1e-8,
                      weight_decay=0.0, bias_correction=True)

    mf = open(metrics_path, "w")
    meta = {
        "run_name": out_dir.name, "task": "LaMP_7", "arm": "mac_control",
        "model": args.model, "data": args.data,
        "n_examples": n, "total_steps": total_steps,
        "schedule_total_steps": schedule_total_steps,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_RANK * LORA_SCALE,
        "lora_scale": LORA_SCALE, "lora_keys": LORA_KEYS,
        "num_lora_layers": NUM_LORA_LAYERS, "n_trainable_params": n_trainable,
        "base_learning_rate": BASE_LR, "warmup_ratio": WARMUP_RATIO,
        "accum_window": ACCUM_WINDOW, "grad_clip_norm": GRAD_CLIP,
        "weight_decay": 0.0, "adam_bias_correction": True,
        "grad_checkpoint": args.grad_checkpoint,
        "git_commit": commit, "git_dirty": dirty,
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mlx_version": mx.__version__,
    }
    with open(out_dir / "adapter_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    mf.write(json.dumps({"record_type": "run_start", **meta}) + "\n")
    mf.flush()

    t0 = time.time()
    idx = 0
    for step in range(total_steps):
        step_start = time.time()
        micro_count = min(ACCUM_WINDOW, n - idx)
        accum = None
        window_ce = 0.0
        window_tokens = 0.0
        micro_losses = []
        micro_seq_lens = []
        for _ in range(micro_count):
            ex_id, ids, ls, le = examples[idx]
            idx += 1
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

        window_loss = window_ce / window_tokens if window_tokens else 0.0
        rec = {
            "record_type": "opt_step", "step": step + 1,
            "loss": window_loss, "learning_rate": lr,
            "grad_norm_preclip": float(grad_norm),
            "n_micro": micro_count, "window_masked_tokens": int(window_tokens),
            "window_seq_tokens": sum(micro_seq_lens),
            "window_s": time.time() - step_start,
            "elapsed_s": time.time() - t0,
            "micro_losses": micro_losses, "micro_seq_lens": micro_seq_lens,
        }
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[train_task_mlx] step {step + 1}/{total_steps} loss={window_loss:.4f} "
                  f"lr={lr:.3g} grad_norm={float(grad_norm):.4f} "
                  f"({rec['window_s']:.1f}s/step)", flush=True)

    mx.save_safetensors(str(adapter_path), dict(tree_flatten(model.trainable_parameters())))
    elapsed = time.time() - t0
    mf.write(json.dumps({"record_type": "run_end", "elapsed_s": elapsed,
                         "adapter": str(adapter_path)}) + "\n")
    mf.close()
    print(f"[train_task_mlx] done in {elapsed / 3600:.2f} h -> {adapter_path}", flush=True)


if __name__ == "__main__":
    main()
