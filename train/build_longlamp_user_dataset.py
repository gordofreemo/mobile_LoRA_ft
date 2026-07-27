#!/usr/bin/env python3
"""
Build a single-user training corpus for LongLaMP User-LoRA fine-tuning
(LL4 Review, LL5 Abstract, LL6 Topic Writing).

Per-user companion to `train/build_longlamp_dataset.py`, mirroring
`train/build_user_dataset.py`'s "records" framing (LaMP's Round-4/R5/R6/R8
convention) -- pinned design #8 in
experiments/2026-07-27-longlamp-user-lora-ll4-ll6-plan.md: training examples
are the user's own real (input, output) records from the `_temporal`-split
TRAIN partition (there may be several per user), each with a `system` slot
populated by BM25 top-k retrieval over THAT RECORD'S OWN embedded profile
(LongLaMP profiles aren't a single stable per-user snapshot the way LaMP's
are -- each record carries its own `profile` field, presumed by the dataset's
own "temporal" partitioning to already be the historically-appropriate one
for that record; see plan decision #4). This is simpler than LaMP's records
framing (train/build_user_dataset.py's emit_record_bm25), which asserts a
single shared profile across all of a user's records -- no such assertion is
made or needed here.

BM25 index/query/format/connector per task is duplicated from
train/build_longlamp_dataset.py's TASKS dict (itself ported verbatim from
the LongLaMP authors' training code) -- MUST stay byte-identical to that
copy AND to eval/eval_longlamp.py's copy (same cardinal train/eval
consistency rule as everywhere else in this project). Kept duplicated
rather than imported so this script remains standalone in a Condor sandbox.

User record indices come from data/longlamp_user_stats.py's
<tag>_user_records.json (file-order indices into the `_temporal` split's
train.json -- the same indexing scheme eval/eval_longlamp.py's `load_split`
uses for its synthetic record ids, so no translation is needed at eval time).

Output format matches train/train.py's build_example (flat dict with
`system`/`user`/`assistant` strings), same as build_longlamp_dataset.py.

Output:
    data/longlamp_user_train_<tag>_<user_tag>_bm25k<K>.jsonl
    (+ matching .meta.json sidecar)

Usage (CPU-only -- BM25 + JSONL write, no GPU needed):
    python train/build_longlamp_user_dataset.py --tag review --user "A1B2C3" --k 4
    python train/build_longlamp_user_dataset.py --tag abstract --user "Jane Doe" --k 4 --limit 5   # smoke
"""

import argparse
import datetime
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT", "/home/ange00008/projects/mobileFT_distill"
))
LONGLAMP_DIR = Path(os.environ.get("LONGLAMP_DIR", str(PROJECT_ROOT / "data" / "longlamp")))
USER_STATS_DIR = Path(os.environ.get(
    "LONGLAMP_USER_STATS_DIR", str(PROJECT_ROOT / "data" / "longlamp_user_stats")
))
DATA_OUT_DIR = Path(os.environ.get("DATA_OUT_DIR", str(PROJECT_ROOT / "data")))


def _first_750_words(text: str) -> str:
    """Verbatim duplicate of build_longlamp_dataset.py's helper -- see that
    module's docstring for provenance (paper's extract_first_750_words)."""
    return " ".join(str(text).split()[:750])


def _extract_after_items(input_string: str):
    """Verbatim duplicate of build_longlamp_dataset.py's helper (paper's
    misleadingly-named extract_before_bullets)."""
    idx = input_string.find("items:")
    if idx == -1:
        return input_string
    return input_string[idx + len("items:"):].strip()


# --- Per-task configuration (duplicated from train/build_longlamp_dataset.py
# and eval/eval_longlamp.py -- keep all three byte-identical) ----------------
TASK_CFG = {
    "review": {
        "temporal_task": "product_review_temporal",
        "id_field": "reviewerId",
        "index_field": lambda p: " ".join([
            str(p.get("overall", "")), str(p.get("summary", "")),
            str(p.get("description", "")), str(p.get("reviewText", "")),
        ]),
        "query": lambda inp: inp,
        "format": lambda p: (
            f'"{p.get("overall", "?")}" is a rating for the product with '
            f'description "{p.get("description", "")}". '
            f'"{p.get("summary", "")}" is summary for "{p.get("reviewText", "")}"'
        ),
        "connector": ". Following the given patterns ",
    },
    "abstract": {
        "temporal_task": "abstract_generation_temporal",
        "id_field": "name",
        "index_field": lambda p: " ".join([str(p.get("title", "")), str(p.get("abstract", ""))]),
        "query": _extract_after_items,
        "format": lambda p: (
            f'"{_first_750_words(p.get("abstract", ""))}" is the abstract for '
            f'the title "{p.get("title", "")}"'
        ),
        "connector": (
            ". Use the above abstracts as context to understand the style "
            "and language of the user and, "
        ),
    },
    "topic": {
        "temporal_task": "topic_writing_temporal",
        "id_field": "author",
        "index_field": lambda p: " ".join([str(p.get("summary", "")), str(p.get("content", ""))]),
        "query": lambda inp: inp,
        "format": lambda p: (
            f'"{p.get("summary", "")}" is a summary for "{p.get("content", "")}"'
        ),
        "connector": ". Following the given patterns, ",
    },
}


# --- BM25 (identical algorithm to build_longlamp_dataset.py / eval_longlamp.py)
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


def retrieve_profile(cfg: dict, query: str, profile: list, k: int) -> list:
    if not profile or k <= 0:
        return []
    docs = [tokenize(cfg["index_field"](it)) for it in profile]
    bm25 = BM25(docs)
    idxs = bm25.top_k(tokenize(query), k)
    return [profile[i] for i in idxs]


def build_example(cfg: dict, record: dict, k: int) -> dict:
    raw_input = record["input"]
    query = cfg["query"](raw_input)
    retrieved = retrieve_profile(cfg, query, record.get("profile", []), k)
    if retrieved:
        lines = ", and ".join(cfg["format"](it) for it in retrieved)
        system = lines + cfg["connector"]
    else:
        system = ""
    return {"system": system, "user": raw_input, "assistant": str(record["output"])}


# Filesystem-safe tag for a raw user id (reviewerId/name/author) -- names in
# particular can contain spaces/punctuation. Collision risk is negligible at
# K=100 scale; raw id is kept verbatim in the meta sidecar regardless.
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_user_tag(user_id: str) -> str:
    tag = _UNSAFE.sub("_", user_id).strip("_")
    return tag or "user"


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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tag", required=True, choices=list(TASK_CFG.keys()),
                        help="review / abstract / topic")
    parser.add_argument("--user", required=True,
                        help="raw user id (reviewerId/name/author) as it appears "
                             "in <tag>_user_records.json")
    parser.add_argument("--k", type=int, default=4,
                        help="BM25 top-k profile entries per record (default 4, "
                             "matching the Task-LoRA recipe's k -- but not yet "
                             "locked in, see plan decision #5)")
    parser.add_argument("--limit", type=int, default=0, help="cap records (smoke testing)")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing JSONL / meta (default: refuse)")
    args = parser.parse_args()

    cfg = TASK_CFG[args.tag]
    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] build_longlamp_user_dataset tag={args.tag} user={args.user!r} "
        f"k={args.k} limit={args.limit} commit={commit_short} "
        f"host={provenance.get('hostname')}",
        flush=True,
    )

    user_tag = safe_user_tag(args.user)
    limit_tag = f"_limit{args.limit}" if args.limit > 0 else ""
    out_path = DATA_OUT_DIR / f"longlamp_user_train_{args.tag}_{user_tag}_bm25k{args.k}{limit_tag}.jsonl"
    meta_path = DATA_OUT_DIR / f"longlamp_user_train_{args.tag}_{user_tag}_bm25k{args.k}{limit_tag}.meta.json"
    existing = [p for p in (out_path, meta_path) if p.exists()]
    if existing and not args.overwrite:
        print("ERROR: refusing to overwrite existing files:", file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        print("Pass --overwrite to replace.", file=sys.stderr)
        sys.exit(1)

    records_json = USER_STATS_DIR / f"{args.tag}_user_records.json"
    if not records_json.exists():
        sys.exit(f"ERROR: {records_json} missing -- run data/longlamp_user_stats.py first.")
    user_records = json.loads(records_json.read_text())
    if args.user not in user_records:
        sys.exit(f"ERROR: user {args.user!r} not in {records_json}.")
    train_idxs = set(user_records[args.user]["train"])
    test_idxs = set(user_records[args.user]["test"])
    if not train_idxs:
        sys.exit(f"ERROR: user {args.user!r} has 0 temporal-TRAIN records -- "
                 f"nothing to build a training corpus from.")
    print(f"[user] {args.user!r}: {len(train_idxs)} temporal-train records, "
          f"{len(test_idxs)} temporal-test records", flush=True)

    train_path = LONGLAMP_DIR / cfg["temporal_task"] / "train.json"
    if not train_path.exists():
        sys.exit(f"ERROR: {train_path} not found -- run data/download_longlamp.py "
                 f"--task {cfg['temporal_task']} first.")
    print(f"[load] {train_path}", flush=True)
    all_records = json.loads(train_path.read_text())

    missing = [i for i in train_idxs if i >= len(all_records)]
    if missing:
        sys.exit(f"ERROR: {len(missing)} train indices out of range for "
                 f"{train_path} (len={len(all_records)}); user_records.json and "
                 f"the temporal split disagree -- e.g. {sorted(missing)[:5]}.")

    t0 = time.time()
    lines = []
    n_skipped_empty = 0
    for idx in sorted(train_idxs):
        rec = all_records[idx]
        # Sanity: this record really belongs to this user.
        rid = str(rec.get(cfg["id_field"]))
        if rid != args.user:
            sys.exit(f"ERROR: record index {idx} in {train_path} has "
                     f"{cfg['id_field']}={rid!r}, expected {args.user!r} -- "
                     f"user_records.json is stale (rebuild via "
                     f"data/longlamp_user_stats.py --overwrite).")
        if args.limit > 0 and len(lines) >= args.limit:
            break
        ex = build_example(cfg, rec, args.k)
        if not ex["user"].strip() or not ex["assistant"].strip():
            n_skipped_empty += 1
            continue
        ex["task"] = f"LongLaMP_{cfg['temporal_task']}"
        ex["id"] = f"{args.user}-{idx}"
        lines.append(json.dumps(ex) + "\n")

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for line in lines:
            f.write(line)

    n_written = len(lines)
    print(f"[write] {n_written} examples ({n_skipped_empty} skipped empty "
          f"input/output) -> {out_path}", flush=True)

    meta = {
        "schema_version": 1,
        "tag": args.tag,
        "temporal_task": cfg["temporal_task"],
        "user_id": args.user,
        "user_tag": user_tag,
        "framing": "records_bm25",
        "bm25_k": args.k,
        "n_user_train_records": len(train_idxs),
        "n_user_test_records": len(test_idxs),
        "n_examples": n_written,
        "n_skipped_empty": n_skipped_empty,
        "limit": args.limit,
        "output_jsonl": str(out_path),
        "user_records_json": str(records_json),
        "command": "python " + " ".join(sys.argv),
        **provenance,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[meta] -> {meta_path}", flush=True)
    print(f"total: {n_written} examples in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
