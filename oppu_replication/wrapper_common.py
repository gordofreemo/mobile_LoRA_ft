"""Shared wrapper plumbing for the OPPU faithful-replication scripts.

Everything here is OUR convention layer (provenance, refuse-to-overwrite,
meta sidecars, output paths) — none of it touches OPPU's training/eval logic.
See PATCHES.md for the full deviation ledger vs third_party/OPPU @ 87f8c69.
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

# their task_name -> the file holding the 100 test users
TEST_FILE = {
    "citation": "user_top_100_history.json",
    "movie_tagging": "user_top_100_history.json",
    "news_categorize": "user_top_100_history.json",
    "news_headline": "user_top_100_history.json",
    "product_rating": "user_top_100_history.json",
    "scholarly_title": "user_top_100_history.json",
    # PATCH P5: their release names this file differently for tweets;
    # their code would crash on the name their own README implies.
    "tweet_paraphrase": "user_more_100_history.json",
}


def provenance():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return "unknown"
    return {
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "condor_cluster_id": os.environ.get("CONDOR_CLUSTER_ID"),
        "condor_proc_id": os.environ.get("CONDOR_PROC_ID"),
    }


def banner(script, args):
    p = provenance()
    print(f"[{script}] task={args.task_name} k={args.k} model={args.model_name} "
          f"seed={args.seed} commit={p['git_commit']} dirty={p['git_dirty']} "
          f"host={p['hostname']} condor={p['condor_cluster_id']}.{p['condor_proc_id']}",
          flush=True)


def refuse_overwrite(paths, overwrite):
    for p in paths:
        p = Path(p)
        if p.exists() and (not p.is_dir() or any(p.iterdir())) and not overwrite:
            print(f"REFUSING to overwrite existing {p} (pass --overwrite)")
            sys.exit(1)


def write_meta(path, args, extra):
    meta = {"args": {k: str(v) for k, v in vars(args).items()}}
    meta.update(extra)
    meta.update(provenance())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def lora_delta_stats(model):
    """Sum of |lora_B| over all adapter params — zero means nothing trained."""
    total_b = 0.0
    n_b = 0
    for name, param in model.named_parameters():
        if "lora_B" in name:
            total_b += float(param.detach().abs().sum())
            n_b += 1
    return {"n_lora_B_tensors": n_b, "lora_B_abs_sum": total_b}


def base_weights_hash(model, limit_tensors=4):
    """Hash a few base (non-LoRA) weight tensors — detects cross-user drift."""
    h = hashlib.md5()
    seen = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            continue
        if "q_proj" in name or "v_proj" in name:
            h.update(param.detach().float().cpu().numpy().tobytes()[:65536])
            seen += 1
            if seen >= limit_tensors:
                break
    return h.hexdigest()


def export_predictions(their_json_path, jsonl_path, task_id, model_name, pred_all,
                       gold_by_id):
    """Their {task, golds, model} format + our per-query {id, pred, gold} JSONL."""
    Path(their_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(their_json_path, "w") as f:
        json.dump({"task": task_id, "golds": pred_all, "model": model_name}, f, indent=4)
    with open(jsonl_path, "w") as f:
        for rec in pred_all:
            f.write(json.dumps({"id": rec["id"], "pred": rec["output"],
                                "gold": gold_by_id.get(str(rec["id"]))}) + "\n")
