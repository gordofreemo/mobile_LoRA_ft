#!/usr/bin/env python3
"""
Round 6 T3 sizing: tokenize all 100 LaMP-4 per-user JSONLs via SmolLM3-3B's
chat template (enable_thinking=False), compute the global token-length
distribution, and pin `max_seq_length = round_up_to_256(min(global_max, 8192))`
for the Step-6 OPPU training config.

Mirrors `round5_t3_sizing.py` (LaMP-3, Round 5), task-swapped to LaMP-4.

Per the Round-6 plan
(experiments/2026-06-19-user-lora-round6-lamp4-multi-plan.md §Step 4):
  - Reads each user listed in data/lamp_user_stats/LaMP_4_top100_users.json,
    loads the user's bm25k4 JSONL, applies the chat template per example.
  - Output JSON: token_length stats + the pinned max_seq_length value.
  - STOP if global max > 8192 (SmolLM3-3B positional ceiling) — do not
    auto-truncate; surface truncation_pct and decide jointly.

Usage (CPU-only):
    python data/lamp_user_stats/round6_t3_sizing.py
    python data/lamp_user_stats/round6_t3_sizing.py --overwrite

Output:
    data/lamp_user_stats/LaMP_4_round6_t3_sizing.json
"""

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/home/ange00008/projects/mobileFT_distill")
)
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"
DATA_DIR = PROJECT_ROOT / "data"
TOKENIZER_DIR = PROJECT_ROOT / "data" / "models" / "SmolLM3-3B"

POSITIONAL_CEILING = 8192
SIZING_STEP = 256


def round_up_to_256(x: int) -> int:
    return ((x + SIZING_STEP - 1) // SIZING_STEP) * SIZING_STEP


def collect_provenance() -> dict:
    def _git(*a):
        try:
            return (
                subprocess.check_output(
                    ["git", *a], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            return None

    porcelain = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "hostname": socket.gethostname(),
        "condor_cluster_id": os.environ.get("CONDOR_CLUSTER_ID") or None,
        "condor_proc_id": os.environ.get("CONDOR_PROC_ID") or None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if porcelain is None else bool(porcelain),
        "python_version": platform.python_version(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top-users",
                        type=Path,
                        default=USER_STATS_DIR / "LaMP_4_top100_users.json",
                        help="path to top-100 users JSON (Step-2 output)")
    parser.add_argument("--out", type=Path,
                        default=USER_STATS_DIR / "LaMP_4_round6_t3_sizing.json",
                        help="output JSON path")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing output (default: refuse)")
    parser.add_argument("--limit-users", type=int, default=0,
                        help="if >0, restrict to first N users (smoke)")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        sys.exit(f"ERROR: refusing to overwrite {args.out}. Pass --overwrite.")

    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] round6_t3_sizing commit={commit_short} "
        f"dirty={provenance.get('git_dirty')} "
        f"host={provenance.get('hostname')} "
        f"cluster.proc={provenance.get('condor_cluster_id')}."
        f"{provenance.get('condor_proc_id')}",
        flush=True,
    )

    # --- Load the top-100 user list --------------------------------------
    if not args.top_users.exists():
        sys.exit(f"ERROR: missing {args.top_users}; run Step 2 first.")
    top = json.loads(args.top_users.read_text())
    users = top["users"]
    if args.limit_users > 0:
        users = users[: args.limit_users]
    fps = [u["user_fingerprint"] for u in users]
    print(f"[users] {len(fps)} fingerprints from {args.top_users}", flush=True)

    # --- Verify each user's bm25k4 JSONL is on disk ----------------------
    missing = [fp for fp in fps
               if not (DATA_DIR / f"lamp_user_train_LaMP_4_{fp}_bm25k4.jsonl").exists()]
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} users have no bm25k4 JSONL "
            f"(e.g. {missing[:3]}). Run Step 3 first."
        )

    # --- Load tokenizer --------------------------------------------------
    print(f"[tokenize] loading SmolLM3-3B tokenizer from {TOKENIZER_DIR} ...",
          flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    if tok.chat_template is None:
        sys.exit("ERROR: tokenizer has no chat_template.")

    # --- Per-user tokenization + global stats ----------------------------
    print(f"[tokenize] applying chat template to all examples "
          f"(enable_thinking=False) ...", flush=True)
    lengths = []
    per_user = {}
    n_examples_seen = 0
    t0 = time.time()
    for ui, fp in enumerate(fps):
        path = DATA_DIR / f"lamp_user_train_LaMP_4_{fp}_bm25k4.jsonl"
        user_lengths = []
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                messages = []
                if r.get("system"):
                    messages.append({"role": "system", "content": r["system"]})
                messages.append({"role": "user", "content": r["user"]})
                messages.append({"role": "assistant", "content": r["assistant"]})
                try:
                    out = tok.apply_chat_template(
                        messages, tokenize=True, return_dict=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    out = tok.apply_chat_template(
                        messages, tokenize=True, return_dict=True,
                    )
                ids = out["input_ids"]
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                user_lengths.append(len(ids))
        per_user[fp] = {
            "n_examples": len(user_lengths),
            "min": min(user_lengths) if user_lengths else None,
            "mean": (sum(user_lengths) / len(user_lengths)) if user_lengths else None,
            "max": max(user_lengths) if user_lengths else None,
        }
        lengths.extend(user_lengths)
        n_examples_seen += len(user_lengths)
        if (ui + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = n_examples_seen / max(elapsed, 1e-6)
            print(f"  {ui+1}/{len(fps)} users "
                  f"({n_examples_seen} examples, {rate:.0f}/s, "
                  f"running max={max(lengths)})", flush=True)
    print(f"[tokenize] done: {n_examples_seen} examples across {len(fps)} users "
          f"in {time.time()-t0:.0f}s", flush=True)

    # --- Global stats ----------------------------------------------------
    sorted_lens = sorted(lengths)

    def pct(p):
        idx = int(round((len(sorted_lens) - 1) * p))
        return sorted_lens[idx]

    stats = {
        "n_examples": len(lengths),
        "n_users": len(fps),
        "min": min(lengths),
        "mean": sum(lengths) / len(lengths),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": max(lengths),
    }
    max_seq_length = round_up_to_256(min(stats["max"], POSITIONAL_CEILING))
    n_over_ceiling = sum(1 for x in lengths if x > POSITIONAL_CEILING)
    truncation_pct = 100.0 * n_over_ceiling / len(lengths)
    print(
        f"[stats] n={stats['n_examples']}, min={stats['min']}, mean={stats['mean']:.0f}, "
        f"p50={stats['p50']}, p95={stats['p95']}, p99={stats['p99']}, "
        f"max={stats['max']} -> max_seq_length={max_seq_length}",
        flush=True,
    )

    # --- Acceptance: if global max exceeds the positional ceiling, STOP --
    # (per plan decision #12: "do not auto-modify methodology" — surface
    # truncation_pct and require a joint decision before Step 5.)
    if stats["max"] > POSITIONAL_CEILING:
        max_users = [fp for fp, st in per_user.items() if st["max"] and st["max"] > POSITIONAL_CEILING]
        out_doc = {
            "schema_version": 1,
            "task": "LaMP_4",
            "round": 6,
            "n_users": stats["n_users"],
            "n_examples_total": stats["n_examples"],
            "token_length": {
                "min": stats["min"],
                "mean": stats["mean"],
                "p50": stats["p50"],
                "p95": stats["p95"],
                "p99": stats["p99"],
                "max": stats["max"],
            },
            "sizing_rule": "max_seq_length = round_up_to_256(min(max(input_ids_len), 8192))",
            "max_seq_length_pinned": None,
            "positional_ceiling": POSITIONAL_CEILING,
            "max_seq_length_le_positional_ceiling": False,
            "n_over_ceiling": n_over_ceiling,
            "truncation_pct": truncation_pct,
            "max_users_over_ceiling": max_users,
            "per_user_stats": per_user,
            "inputs": {
                "top_users": str(args.top_users),
                "tokenizer_dir": str(TOKENIZER_DIR),
                "data_dir": str(DATA_DIR),
            },
            "command": "python " + " ".join(sys.argv),
            "provenance": provenance,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_doc, indent=2))
        sys.exit(
            f"ERROR: global max token length {stats['max']} exceeds positional "
            f"ceiling {POSITIONAL_CEILING}. {n_over_ceiling} example(s) "
            f"({truncation_pct:.2f}%) over ceiling across "
            f"{len(max_users)} user(s) (first few: {max_users[:5]}). "
            f"STOP — full stats written to {args.out} for review; do not "
            f"pin Step-6 config until this is decided jointly."
        )

    # --- Identify the example(s) at max length (helpful for debugging) ---
    max_users = [fp for fp, st in per_user.items() if st["max"] == stats["max"]]
    print(f"[stats] max={stats['max']} reached by {len(max_users)} user(s) "
          f"(first few: {max_users[:5]})", flush=True)

    out_doc = {
        "schema_version": 1,
        "task": "LaMP_4",
        "round": 6,
        "n_users": stats["n_users"],
        "n_examples_total": stats["n_examples"],
        "token_length": {
            "min": stats["min"],
            "mean": stats["mean"],
            "p50": stats["p50"],
            "p95": stats["p95"],
            "p99": stats["p99"],
            "max": stats["max"],
        },
        "sizing_rule": "max_seq_length = round_up_to_256(min(max(input_ids_len), 8192))",
        "max_seq_length_pinned": max_seq_length,
        "positional_ceiling": POSITIONAL_CEILING,
        "max_seq_length_le_positional_ceiling": True,
        "n_over_ceiling": n_over_ceiling,
        "truncation_pct": truncation_pct,
        "max_users": max_users,
        "per_user_stats": per_user,
        "inputs": {
            "top_users": str(args.top_users),
            "tokenizer_dir": str(TOKENIZER_DIR),
            "data_dir": str(DATA_DIR),
        },
        "command": "python " + " ".join(sys.argv),
        "provenance": provenance,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2))
    print(f"[write] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
