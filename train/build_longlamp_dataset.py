#!/usr/bin/env python3
"""
Build the LongLaMP Product Review (user split) training corpus for
LongLaMP-LoRA (Review) — LL1's Task-LoRA (see
experiments/2026-07-17-longlamp-review-ll1-plan.md).

Reads data/longlamp/product_review_user/train.json (written by
data/download_longlamp.py), does BM25 top-k retrieval of each reviewer's
profile entries, and writes a compact JSONL ready for train/train.py.

BM25 + role layout here MUST match eval/eval_longlamp.py exactly (same
cardinal rule as LaMP's build_dataset.py / eval_lamp.py pair) — the relevant
blocks (BM25, tokenize, retrieve_profile, PPEP formatting) are duplicated
from eval_longlamp.py rather than imported, same rationale as LaMP's harness
(this script needs to run standalone in a Condor sandbox with no
import-path gymnastics).

Unlike LaMP, LongLaMP's `input` field is already a fully-templated
instruction (verified via the HF datasets-server API, not guessed — see the
plan doc). So the query for BM25 retrieval is the full `input` string, and
the `user` turn is `input` verbatim — no task-instruction template is built
here.

Each line is one training example:
    {"task": "LongLaMP_review_user", "id": "<reviewerId>-<index>",
     "system": "<PPEP-formatted retrieved profile context>",
     "user":   "<input>",
     "assistant": "<output>"}

Usage (CPU is enough — no GPU needed for BM25 + JSONL write):
    python train/build_longlamp_dataset.py --k 4 --seed 0
    python train/build_longlamp_dataset.py --limit 100   # smoke
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

LONGLAMP_DIR = os.environ.get(
    "LONGLAMP_DIR",
    "/home/ange00008/projects/mobileFT_distill/data/longlamp",
)
DATA_OUT_DIR = os.environ.get(
    "DATA_OUT_DIR",
    "/home/ange00008/projects/mobileFT_distill/data",
)
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/home/ange00008/projects/mobileFT_distill",
)

TASK = "product_review_user"
TASK_TAG = "LongLaMP_review_user"

# Same preamble every other task in this project uses ahead of the retrieved
# profile context (see train/build_dataset.py / eval/eval_lamp.py). LongLaMP's
# `input` already carries the task instruction, so this preamble is the only
# framing text the profile context needs.
SYSTEM_PREAMBLE = (
    "The following are examples of this user's past activity. "
    "Use them to match this user's preferences and writing style.\n\n"
)


def index_text(entry: dict) -> str:
    """BM25-indexed text for one profile entry: description + summary + reviewText,
    concatenated. See plan decision #17 / "BM25 query construction" — LongLaMP's
    `input` has no separable raw fields to query against cleanly, so we index
    the profile entry's full text rather than trying to parse the templated
    instruction."""
    return " ".join([
        str(entry.get("description", "")),
        str(entry.get("summary", "")),
        str(entry.get("reviewText", "")),
    ])


def format_entry(entry: dict) -> str:
    """PPEP template, verbatim from the LongLaMP paper (plan decision #8):
    "{overall} is a rating for the product with description {description}.
    {summary} is summary for {reviewText}" """
    return (
        f'{entry.get("overall", "?")} is a rating for the product with '
        f'description {entry.get("description", "")}. '
        f'{entry.get("summary", "")} is summary for {entry.get("reviewText", "")}'
    )


# --- BM25 (identical algorithm to eval/eval_longlamp.py) ---------------------
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return _WORD.findall(str(text).lower())


class BM25:
    def __init__(self, docs_tokens: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.doc_freqs = [Counter(d) for d in docs_tokens]
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        df = Counter()
        for d in docs_tokens:
            for w in set(d):
                df[w] += 1
        self.idf = {
            w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()
        }

    def top_k(self, query_tokens: list, k: int) -> list:
        if self.N == 0:
            return []
        scored = []
        for i in range(self.N):
            freqs, dl = self.doc_freqs[i], self.doc_len[i]
            s = 0.0
            for w in query_tokens:
                tf = freqs.get(w)
                if not tf:
                    continue
                idf = self.idf.get(w, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (tf * (self.k1 + 1)) / denom
            scored.append((s, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scored[:k]]


def retrieve_profile(query: str, profile: list, k: int) -> list:
    if not profile or k <= 0:
        return []
    docs = [tokenize(index_text(it)) for it in profile]
    bm25 = BM25(docs)
    idxs = bm25.top_k(tokenize(query), k)
    return [profile[i] for i in idxs]


def build_example(record: dict, idx: int, k: int) -> dict:
    query = record["input"]
    retrieved = retrieve_profile(query, record.get("profile", []), k)
    if retrieved:
        lines = "\n".join(format_entry(it) for it in retrieved)
        system = SYSTEM_PREAMBLE + lines
    else:
        system = ""
    return {
        "task": TASK_TAG,
        "id": f'{record.get("reviewerId", "unk")}-{idx}',
        "system": system,
        "user": query,
        "assistant": str(record["output"]),
    }


# --- Provenance --------------------------------------------------------------
def collect_provenance() -> dict:
    import datetime
    import platform
    import socket
    import subprocess

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4, help="BM25 profile entries per example")
    parser.add_argument("--seed", type=int, default=0, help="seed for the shuffle (deterministic)")
    parser.add_argument("--limit", type=int, default=0, help="cap examples (smoke testing)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="overwrite existing JSONL / meta files (default: refuse)",
    )
    args = parser.parse_args()

    provenance = collect_provenance()
    suffix = f"bm25k{args.k}"
    out_path = Path(DATA_OUT_DIR) / f"longlamp_train_review_user_{suffix}.jsonl"
    meta_path = Path(DATA_OUT_DIR) / f"longlamp_train_review_user_{suffix}.meta.json"

    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] build_longlamp_dataset task={TASK} k={args.k} seed={args.seed} "
        f"limit={args.limit} commit={commit_short} "
        f"condor={provenance.get('condor_cluster_id') or '-'}."
        f"{provenance.get('condor_proc_id') or '-'} "
        f"host={provenance.get('hostname')}",
        flush=True,
    )

    existing = [p for p in (out_path, meta_path) if p.exists()]
    if existing and not args.overwrite:
        print("ERROR: refusing to overwrite existing files:", file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nPass --overwrite to replace, or change --k / --seed to write a "
            "new path.",
            file=sys.stderr,
        )
        sys.exit(1)

    train_path = Path(LONGLAMP_DIR) / TASK / "train.json"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found — run data/download_longlamp.py first.",
              file=sys.stderr)
        sys.exit(1)

    records = json.loads(train_path.read_text())
    if args.limit > 0:
        records = records[: args.limit]

    Path(DATA_OUT_DIR).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    lines = []
    for i, r in enumerate(records):
        ex = build_example(r, i, args.k)
        lines.append(json.dumps(ex) + "\n")
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  {i + 1}/{len(records)} ({rate:.0f}/s)", flush=True)

    # Deterministic shuffle, matching build_dataset.py's mixed-corpus convention
    # (this corpus is single-task, but the shuffle still decorrelates training
    # order from the source file's on-disk reviewer ordering).
    import random

    rng = random.Random(args.seed)
    rng.shuffle(lines)

    with out_path.open("w") as out:
        for line in lines:
            out.write(line)

    size_mb = out_path.stat().st_size / 1e6
    print(f"{TASK}: {len(lines)} examples ({size_mb:.1f} MB) -> {out_path}")

    meta = {
        "schema_version": 1,
        "task": TASK,
        "k": args.k,
        "retriever": "bm25",
        "seed": args.seed,
        "limit": args.limit,
        "total_count": len(lines),
        "output_file": str(out_path),
        "command": "python " + " ".join(sys.argv),
        **provenance,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"meta:  {meta_path}")
    print(f"total: {len(lines)} examples in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
