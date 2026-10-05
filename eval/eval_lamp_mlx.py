#!/usr/bin/env python3
"""
LaMP evaluation harness — MLX backend, for on-device-trained User-LoRA adapters.

eval_lamp.py's model-loading/generation path is HF+PEFT-only and can't load the
adapters MLX-swift training saves on-device (different key convention, no
PEFT adapter_config.json). This script reuses eval_lamp.py's BM25 retrieval,
prompt construction, and rating-parsing logic verbatim (import, not copy) and
swaps only the model-loading/generation backend to mlx_lm, so an on-device
adapter can be evaluated on the Mac without any HF/PEFT conversion step.

Each user in LaMP-3's time split contributes exactly one test record (matches
the R5/R6 per-user eval convention: 100 users x 1 record = the n=100 test
set) — so a single user's "accuracy" here is one correct/incorrect, not a
rate. Report the per-user predictions and the aggregate over all requested
users; don't over-read any individual user's 0/1.

Usage:
    LAMP_DIR=data/lamp_time RESULTS_DIR=results \
    .venv-mlx/bin/python eval/eval_lamp_mlx.py \
        --model-dir data/models/SmolLM3-3B-a1lamp-mlx-4bit \
        --adapter-root /tmp/devpull/eval_adapters \
        --users u00008075,u00005228,u00011077,u00005020,u00012502
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# eval_lamp.py reads LAMP_DIR / RESULTS_DIR / USER_STATS_DIR at import time —
# must be set (env or default) before the import below.
sys.path.insert(0, str(Path(__file__).parent))
import eval_lamp  # noqa: E402


def collect_provenance_mlx() -> dict:
    """Same shape as eval_lamp.collect_provenance() but for the MLX/Mac path
    (no torch/transformers/peft/Condor on this host)."""
    import datetime
    import platform
    import socket
    import subprocess

    import mlx.core as mx
    import mlx_lm

    def _git(*a):
        try:
            return (
                subprocess.check_output(["git", *a], cwd=eval_lamp.PROJECT_ROOT,
                                         stderr=subprocess.DEVNULL)
                .decode().strip()
            )
        except Exception:
            return None

    porcelain = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if porcelain is None else bool(porcelain),
        "python_version": platform.python_version(),
        "mlx_version": mx.__version__,
        "mlx_lm_version": getattr(mlx_lm, "__version__", None),
        "backend": "mlx",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="LaMP_3", choices=["LaMP_3"],
                        help="only rating (LaMP_3) is wired up — that's what the "
                             "on-device E2E adapters were trained on")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--model-dir", required=True,
                        help="fused base model (Task-LoRA already merged), MLX 4-bit dir")
    parser.add_argument("--adapter-root", required=True,
                        help="directory containing one subdir per user fingerprint, "
                             "each with adapter_config.json + adapters.safetensors")
    parser.add_argument("--users", required=True,
                        help="comma-separated user fingerprints, e.g. u00008075,u00005020")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    users = args.users.split(",")

    provenance = collect_provenance_mlx()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(f"[run] backend=mlx task={args.task} split={args.split} "
          f"users={len(users)} k={args.k} seed={args.seed} "
          f"commit={commit_short} host={provenance['hostname']}", flush=True)

    results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
    stem = f"{args.task}_{args.split}_ondevice_e2e_bm25k{args.k}_seed{args.seed}_n{len(users)}"
    out_path = results_dir / f"{stem}.json"
    pred_path = results_dir / f"{stem}.predictions.jsonl"
    if not args.overwrite and (out_path.exists() or pred_path.exists()):
        print(f"ERROR: refusing to overwrite existing results ({out_path}, {pred_path}). "
              f"Pass --overwrite.", file=sys.stderr)
        sys.exit(1)

    user_records = json.loads(
        (Path(eval_lamp.USER_STATS_DIR) / f"{args.task}_user_records.json").read_text()
    )
    questions, golds = eval_lamp.load_split(args.task, args.split)
    questions_by_id = {str(q["id"]): q for q in questions}

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)  # greedy — matches eval_lamp.py's do_sample=False
    max_new = eval_lamp.TASKS[args.task]["max_new_tokens"]

    per_user = []
    all_preds, all_golds, all_ids = [], [], []
    t0 = time.time()

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    with pred_path.open("w") as pred_file:
        for fp in users:
            if fp not in user_records:
                print(f"ERROR: {fp} not in {args.task}_user_records.json", file=sys.stderr)
                sys.exit(1)
            test_ids = user_records[fp].get(args.split, [])
            if not test_ids:
                print(f"ERROR: {fp} has 0 records in split {args.split}", file=sys.stderr)
                sys.exit(1)

            adapter_dir = Path(args.adapter_root) / fp
            if not adapter_dir.exists():
                print(f"ERROR: no adapter dir for {fp} at {adapter_dir}", file=sys.stderr)
                sys.exit(1)

            print(f"[user {fp}] loading model + adapter ({len(test_ids)} test record(s))...",
                  flush=True)
            model, tokenizer = load(args.model_dir, adapter_path=str(adapter_dir))

            for qid in test_ids:
                q = questions_by_id.get(str(qid))
                if q is None:
                    print(f"  WARN: {fp} test id {qid} not found in {args.task}/{args.split} "
                          f"questions, skipping", file=sys.stderr)
                    continue
                if q["id"] not in golds:
                    print(f"  WARN: id {q['id']} has no gold, skipping", file=sys.stderr)
                    continue

                messages = eval_lamp.build_messages(args.task, q, args.k, no_profile=False)
                prompt = eval_lamp.render_prompt(tokenizer, messages)
                text = eval_lamp.clean_output(
                    generate(model, tokenizer, prompt, max_tokens=max_new, sampler=sampler)
                )
                gold = golds[q["id"]]

                pred_rating = eval_lamp.parse_rating(text)
                correct = (pred_rating == str(gold).strip()) if pred_rating is not None else False

                per_user.append({
                    "user_fingerprint": fp,
                    "id": q["id"],
                    "pred": text,
                    "pred_rating": pred_rating,
                    "gold": gold,
                    "correct": correct,
                })
                all_preds.append(text)
                all_golds.append(gold)
                all_ids.append(q["id"])

                pred_file.write(json.dumps({
                    "id": q["id"], "user_fingerprint": fp, "pred": text, "gold": gold,
                }) + "\n")
                pred_file.flush()
                print(f"  [{fp}][{q['id']}] pred={text!r} parsed={pred_rating!r} "
                      f"gold={gold!r} correct={correct}", flush=True)

    elapsed = time.time() - t0
    metrics = eval_lamp.score_rating(all_preds, all_golds)

    record = {
        "schema_version": 1,
        "backend": "mlx",
        "task": args.task,
        "split": args.split,
        "model_dir": args.model_dir,
        "adapter_root": args.adapter_root,
        "users": users,
        "n_users": len(users),
        "k": args.k,
        "seed": args.seed,
        "decoding": "greedy",
        "max_new_tokens": max_new,
        "metric_name": "accuracy",
        "metric_value": metrics["accuracy"],
        "accuracy": metrics["accuracy"],
        "mae": metrics["mae"],
        "parse_fail_rate": metrics["parse_fail_rate"],
        "n": metrics["n"],
        "seconds": round(elapsed, 1),
        "command": "python " + " ".join(sys.argv),
        "per_user": per_user,
        **provenance,
    }
    out_path.write_text(json.dumps(record, indent=2))

    print("=" * 60)
    for r in per_user:
        print(f"  {r['user_fingerprint']}: pred={r['pred']!r} gold={r['gold']!r} "
              f"correct={r['correct']}")
    print("=" * 60)
    print(f"aggregate accuracy over {metrics['n']} users: {metrics['accuracy']:.4f} "
          f"(mae={metrics['mae']}, parse_fail_rate={metrics['parse_fail_rate']:.2f})")
    print(f"written -> {out_path}")
    print(f"           {pred_path}")


if __name__ == "__main__":
    main()
