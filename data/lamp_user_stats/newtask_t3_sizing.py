#!/usr/bin/env python3
"""
T3 sizing for the five new User-LoRA rounds (R10/PT3 LaMP-2-news, R11/PT4
LaMP-1, R12/PT5 LaMP-7, R13/PT6 LaMP-2-movies, R14/PT7 LaMP-5).

Tokenizes every per-user training corpus for a task, computes the global
token-length distribution, and pins
`max_seq_length = round_up_to_256(min(global_max, 8192))` for that task's
OPPU training configs.

Generalizes `round5_t3_sizing.py` (LaMP-3) / `round6_t3_sizing.py` (LaMP-4)
along the two axes that actually differ between rounds:

  --task     picks the pool JSON and the per-user corpus filenames
  --framing  picks BOTH the corpus filename suffix and the tokenization:
               profile / records -> SmolLM3 chat template, enable_thinking=False
                                    (what train/train.py will do)
               unsupervised      -> RAW tokenization, no chat template
                                    (what train/train_unsupervised_clm.py will do)

Getting that second axis right matters: measuring the unsupervised corpora
through a chat template would over-count by the template's wrapper tokens and
pin a max_seq_length that doesn't correspond to anything the trainer sees.

Acceptance (unchanged from R5/R6): STOP if the global max exceeds SmolLM3-3B's
8192 positional ceiling — do not auto-truncate; write the full stats and
surface truncation_pct for a joint decision.

`--pin-max-seq-length N` records the outcome of that joint decision. It never
fires on its own; a human passes it after reading the blocked run's stats. It
still computes and writes every statistic, but pins N instead of the derived
value and records `max_seq_length_override: true` plus how many examples that
N truncates, so the deviation is visible in the artifact rather than living
only in a chat log or a hand-edited JSON.

Used 2026-07-30 for LaMP-1 (1792) and LaMP-5 (2048). Both had exactly ONE
example above the 8192 ceiling (14,925 and 15,544 tokens) against
next-largest per-user maxima of 1,597 and 2,048 and p95 of 322 and 918. Since
that single outlier truncates at 8192 just as surely as at 1792/2048, pinning
the ceiling would have preserved nothing while costing headroom on every
batch the outlier landed in — so the pin was set from the real distribution
(round_up_to_256 of the second-largest per-user max) instead of from one
anomaly.

Usage (CPU-only; needs the per-user corpora on disk already):
    python data/lamp_user_stats/newtask_t3_sizing.py --task LaMP_2_news --framing records
    python data/lamp_user_stats/newtask_t3_sizing.py --task LaMP_7 --framing unsupervised

Output:
    data/lamp_user_stats/<task>_t3_sizing.json
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

# task -> pool JSON basename (written by data/select_top_users_newtasks.py)
POOL_FILES = {
    "LaMP_2_news": "LaMP_2_news_top27_users.json",
    "LaMP_1": "LaMP_1_top100_users.json",
    "LaMP_2_movies": "LaMP_2_movies_top100_users.json",
    "LaMP_5": "LaMP_5_top100_users.json",
    "LaMP_7": "LaMP_7_top100_users.json",
}


def corpus_suffix(framing: str, bm25_k: int) -> str:
    """Mirror build_user_dataset.py's output-filename rule exactly."""
    if framing == "records":
        return f"records_bm25k{bm25_k}"
    if framing == "unsupervised":
        return "unsup"
    return "bare" if bm25_k == 0 else f"bm25k{bm25_k}"


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
    parser.add_argument("--task", required=True, choices=sorted(POOL_FILES))
    parser.add_argument("--framing", required=True,
                        choices=["profile", "records", "unsupervised"])
    parser.add_argument("--bm25-k", type=int, default=4)
    parser.add_argument("--pool", type=Path, default=None,
                        help="override the pool JSON path")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-users", type=int, default=0,
                        help="if >0, restrict to first N users (smoke)")
    parser.add_argument("--pin-max-seq-length", type=int, default=0,
                        help="record an explicit human decision instead of the "
                             "derived value (see module docstring). Only for "
                             "resolving a run this script already blocked.")
    args = parser.parse_args()
    if args.pin_max_seq_length and args.pin_max_seq_length > POSITIONAL_CEILING:
        sys.exit(f"ERROR: --pin-max-seq-length {args.pin_max_seq_length} exceeds "
                 f"the {POSITIONAL_CEILING} positional ceiling.")

    pool_path = args.pool or USER_STATS_DIR / POOL_FILES[args.task]
    out_path = args.out or USER_STATS_DIR / f"{args.task}_t3_sizing.json"
    if out_path.exists() and not args.overwrite:
        sys.exit(f"ERROR: refusing to overwrite {out_path}. Pass --overwrite.")

    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] newtask_t3_sizing task={args.task} framing={args.framing} "
        f"commit={commit_short} dirty={provenance.get('git_dirty')} "
        f"host={provenance.get('hostname')} "
        f"cluster.proc={provenance.get('condor_cluster_id')}."
        f"{provenance.get('condor_proc_id')}",
        flush=True,
    )

    if not pool_path.exists():
        sys.exit(f"ERROR: missing {pool_path}; run "
                 f"data/select_top_users_newtasks.py --task {args.task} first.")
    pool = json.loads(pool_path.read_text())
    users = pool["users"]
    if args.limit_users > 0:
        users = users[: args.limit_users]
    fps = [u["user_fingerprint"] for u in users]
    print(f"[users] {len(fps)} fingerprints from {pool_path}", flush=True)

    suffix = corpus_suffix(args.framing, args.bm25_k)

    def corpus_path(fp):
        return DATA_DIR / f"lamp_user_train_{args.task}_{fp}_{suffix}.jsonl"

    missing = [fp for fp in fps if not corpus_path(fp).exists()]
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} users have no {suffix} corpus "
            f"(e.g. {missing[:3]}). Build the per-user corpora first."
        )

    print(f"[tokenize] loading SmolLM3-3B tokenizer from {TOKENIZER_DIR} ...",
          flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    if args.framing != "unsupervised" and tok.chat_template is None:
        sys.exit("ERROR: tokenizer has no chat_template.")

    def length_of(rec) -> int:
        if args.framing == "unsupervised":
            # Must mirror train_unsupervised_clm.build_example: raw text,
            # add_special_tokens=True, plus the appended EOS.
            ids = tok(rec["text"], add_special_tokens=True)["input_ids"]
            eos = tok.eos_token_id
            if eos is not None and (not ids or ids[-1] != eos):
                ids = list(ids) + [eos]
            return len(ids)
        messages = []
        if rec.get("system"):
            messages.append({"role": "system", "content": rec["system"]})
        messages.append({"role": "user", "content": rec["user"]})
        messages.append({"role": "assistant", "content": rec["assistant"]})
        try:
            out = tok.apply_chat_template(
                messages, tokenize=True, return_dict=True, enable_thinking=False,
            )
        except TypeError:
            out = tok.apply_chat_template(
                messages, tokenize=True, return_dict=True,
            )
        ids = out["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)

    print(f"[tokenize] measuring {suffix} corpora "
          f"({'raw, no chat template' if args.framing == 'unsupervised' else 'chat template, enable_thinking=False'}) ...",
          flush=True)
    lengths = []
    per_user = {}
    n_examples_seen = 0
    t0 = time.time()
    for ui, fp in enumerate(fps):
        user_lengths = []
        with corpus_path(fp).open() as f:
            for line in f:
                user_lengths.append(length_of(json.loads(line)))
        per_user[fp] = {
            "n_examples": len(user_lengths),
            "min": min(user_lengths) if user_lengths else None,
            "mean": (sum(user_lengths) / len(user_lengths)) if user_lengths else None,
            "max": max(user_lengths) if user_lengths else None,
        }
        lengths.extend(user_lengths)
        n_examples_seen += len(user_lengths)
        if (ui + 1) % 10 == 0:
            rate = n_examples_seen / max(time.time() - t0, 1e-6)
            print(f"  {ui+1}/{len(fps)} users ({n_examples_seen} examples, "
                  f"{rate:.0f}/s, running max={max(lengths)})", flush=True)
    print(f"[tokenize] done: {n_examples_seen} examples across {len(fps)} users "
          f"in {time.time()-t0:.0f}s", flush=True)

    if not lengths:
        sys.exit("ERROR: 0 examples across all users — corpora are empty.")

    sorted_lens = sorted(lengths)

    def pct(p):
        return sorted_lens[int(round((len(sorted_lens) - 1) * p))]

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
    n_over_ceiling = sum(1 for x in lengths if x > POSITIONAL_CEILING)
    truncation_pct = 100.0 * n_over_ceiling / len(lengths)
    over_ceiling = stats["max"] > POSITIONAL_CEILING
    if args.pin_max_seq_length:
        max_seq_length = args.pin_max_seq_length
        over_ceiling = False  # decision recorded; don't re-block on it
    else:
        max_seq_length = None if over_ceiling else round_up_to_256(stats["max"])
    n_over_pinned = (sum(1 for x in lengths if x > max_seq_length)
                     if max_seq_length else None)

    print(
        f"[stats] n={stats['n_examples']}, min={stats['min']}, "
        f"mean={stats['mean']:.0f}, p50={stats['p50']}, p95={stats['p95']}, "
        f"p99={stats['p99']}, max={stats['max']} -> "
        f"max_seq_length={max_seq_length or 'BLOCKED'}",
        flush=True,
    )

    flagged = [fp for fp, st in per_user.items()
               if st["max"] and st["max"] > POSITIONAL_CEILING] if over_ceiling else \
              [fp for fp, st in per_user.items() if st["max"] == stats["max"]]

    out_doc = {
        "schema_version": 1,
        "task": args.task,
        "framing": args.framing,
        "bm25_k": args.bm25_k if args.framing != "unsupervised" else None,
        "corpus_suffix": suffix,
        "n_users": stats["n_users"],
        "n_examples_total": stats["n_examples"],
        "token_length": {k: stats[k] for k in
                         ("min", "mean", "p50", "p95", "p99", "max")},
        "sizing_rule": "max_seq_length = round_up_to_256(min(max(input_ids_len), 8192))",
        "max_seq_length_pinned": max_seq_length,
        "max_seq_length_override": bool(args.pin_max_seq_length),
        "positional_ceiling": POSITIONAL_CEILING,
        "max_seq_length_le_positional_ceiling": not over_ceiling,
        "n_over_ceiling": n_over_ceiling,
        "truncation_pct": truncation_pct,
        "n_over_pinned": n_over_pinned,
        "truncation_pct_at_pinned": (100.0 * n_over_pinned / len(lengths)
                                     if n_over_pinned is not None else None),
        "max_users": flagged,
        "per_user_stats": per_user,
        "inputs": {
            "pool": str(pool_path),
            "tokenizer_dir": str(TOKENIZER_DIR),
            "data_dir": str(DATA_DIR),
        },
        "command": "python " + " ".join(sys.argv),
        "provenance": provenance,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, indent=2))
    print(f"[write] -> {out_path}", flush=True)

    if over_ceiling:
        sys.exit(
            f"ERROR: global max token length {stats['max']} exceeds positional "
            f"ceiling {POSITIONAL_CEILING}. {n_over_ceiling} example(s) "
            f"({truncation_pct:.2f}%) over ceiling across {len(flagged)} "
            f"user(s) (first few: {flagged[:5]}). STOP — full stats written to "
            f"{out_path}; decide jointly before pinning the training config."
        )


if __name__ == "__main__":
    main()
