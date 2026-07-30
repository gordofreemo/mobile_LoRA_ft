#!/usr/bin/env python3
"""
Build a single-user training corpus for LongLaMP User-LoRA fine-tuning
(LL4 Review, LL5 Abstract, LL6 Topic Writing).

REVISED 2026-07-28 (see the addendum in
experiments/2026-07-27-longlamp-user-lora-ll4-ll6-plan.md, re-grilled via
/grill_me): the original "records" framing (train on the user's distinct
temporal-TRAIN records) turned out to be non-viable -- real data from
data/longlamp_user_stats.py showed n_train_records==1 for literally every
eligible user on Review and Abstract, not "several per user" as first
assumed. Training a 3-epoch LoRA on one example isn't personalization
training. Fixed the same way LaMP-2-movies/5 fixed an analogous dead end:
**profile-entry reframing** -- each user's snapshot record (the one temporal-
TRAIN record with the largest profile, per
data/longlamp_user_stats.py's `train_snapshot_idx`) has a `profile` field
with many entries (up to 823 for Review, 2458 for Abstract, in the eligible
pools seen so far). Each profile entry is reframed into its own training
example, using THAT TASK'S OWN input/output template applied to the entry's
fields -- verified structurally viable for all 3 tasks by reading real data:

  Review:   entry {overall, description, summary, reviewText} -> input =
            task's own template (rating + description + summary) verbatim,
            output = reviewText. Exact field-for-field match to the real
            task's input shape.
  Topic:    entry {summary, content} -> input = "Generate the content for a
            reddit post {summary}", output = content. Exact match.
  Abstract: entry {title, abstract} -> input = 'Generate an abstract for the
            title "{title}"', output = abstract. The real task's input also
            has a keyword ("items:") list that profile entries don't carry --
            deliberately OMITTED from training input (accepted minor
            train/eval input-shape deviation, decided during the grill;
            eval itself is untouched, still uses the real held-out record's
            full input). cfg["query"]'s "items:"-extraction gracefully falls
            back to the whole input string when "items:" isn't found, so no
            special-casing is needed for the BM25 query side of this.

Context (system slot) construction: **leave-one-out BM25** -- for profile
entry e_j being trained on, retrieve top-k over the snapshot's OTHER profile
entries (all i != j), no temporal ordering (LongLaMP profile entries carry
no per-entry date field, unlike LaMP's time split -- see plan decision #4).
Query = cfg["query"](reconstructed input text) -- same query-construction
function used everywhere else in this project, applied to the input text we
just built rather than a real record's `input`, for train/eval byte-shape
consistency. Empty pool (only possible if profile_size==1, which eligibility
now excludes via profile_size>=2) falls back to bare (system="").

index_field / format / connector / query are duplicated from
train/build_longlamp_dataset.py's TASKS dict (itself ported verbatim from
the LongLaMP authors' training code) -- MUST stay byte-identical to that
copy AND to eval/eval_longlamp.py's copy (same cardinal train/eval
consistency rule as everywhere else in this project). Kept duplicated
rather than imported so this script remains standalone in a Condor sandbox.

Snapshot record indices come from data/longlamp_user_stats.py's
<tag>_user_records.json (`train_snapshot_idx` -- the temporal-train record,
by file-order index, with this user's largest profile).

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

# 3.5GB -- matches data/longlamp_user_stats.py's cap (same rationale: at
# least one task's profile `content` field is unusually large; see that
# script's comment for the confirmed-real-text investigation).
MAX_PARSER_BUF_BYTES = 3584 * 1024 * 1024


def _first_750_words(text: str) -> str:
    """Verbatim duplicate of build_longlamp_dataset.py's helper -- see that
    module's docstring for provenance (paper's extract_first_750_words)."""
    return " ".join(str(text).split()[:750])


def _extract_after_items(input_string: str):
    """Verbatim duplicate of build_longlamp_dataset.py's helper (paper's
    misleadingly-named extract_before_bullets). Falls back to the whole
    input when "items:" isn't present -- exactly what happens for this
    script's reconstructed Abstract training inputs, which deliberately
    omit the keyword list (see module docstring)."""
    idx = input_string.find("items:")
    if idx == -1:
        return input_string
    return input_string[idx + len("items:"):].strip()


# --- Per-task configuration (duplicated from train/build_longlamp_dataset.py
# and eval/eval_longlamp.py -- keep index_field/query/format/connector byte-
# identical across all three copies). target_field / build_input are NEW
# for this script's profile-entry reframing. ---------------------------------
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
        "target_field": "reviewText",
        "build_input": lambda p: (
            f'Generate the review text written by a reviewer who has a given '
            f'an overall rating of "{p.get("overall", "?")}" for a product '
            f'with description "{p.get("description", "")}". The summary of '
            f'the review text is "{p.get("summary", "")}".'
        ),
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
        "target_field": "abstract",
        # Real task input also has a "using the following items: ..." keyword
        # list that profile entries don't carry -- deliberately omitted here
        # (accepted minor train/eval deviation, decided 2026-07-28 grill).
        "build_input": lambda p: f'Generate an abstract for the title "{p.get("title", "")}"',
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
        "target_field": "content",
        "build_input": lambda p: f'Generate the content for a reddit post {p.get("summary", "")}',
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


def retrieve_leave_one_out(cfg: dict, query: str, profile: list, exclude_idx: int, k: int) -> list:
    """BM25 top-k over `profile` excluding index `exclude_idx` (the entry
    being trained on) -- decision #8's revised context construction (no
    date field to enforce strict-prior, so leave-one-out is the safest
    simple analog)."""
    pool_idxs = [i for i in range(len(profile)) if i != exclude_idx]
    if not pool_idxs or k <= 0:
        return []
    docs = [tokenize(cfg["index_field"](profile[i])) for i in pool_idxs]
    bm25 = BM25(docs)
    top = bm25.top_k(tokenize(query), k)
    return [profile[pool_idxs[t]] for t in top]


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


def find_one_record(train_path: Path, wanted_idx: int) -> dict:
    """Stream `train_path` (a JSON array, potentially GBs) and return only
    the record at file-order index `wanted_idx`, without materializing the
    rest of the array. Early-exits the moment it's found. Only one record is
    ever needed per user under profile-entry reframing (the snapshot), so
    this is much cheaper than the earlier multi-index version this script
    used under "records" framing."""
    decoder = json.JSONDecoder()
    with open(train_path, "r", encoding="utf-8") as f:
        buf = ""
        while "[" not in buf:
            chunk = f.read(65536)
            if not chunk:
                raise RuntimeError(f"{train_path}: no top-level '[' found.")
            buf += chunk
        buf = buf[buf.index("[") + 1:]
        idx = 0
        while True:
            buf = buf.lstrip()
            if buf.startswith(","):
                buf = buf[1:].lstrip()
            if buf.startswith("]") or not buf:
                chunk = f.read(1 << 20)
                if not chunk:
                    raise RuntimeError(
                        f"{train_path}: reached end of array without finding "
                        f"index {wanted_idx} (array has {idx} records)."
                    )
                buf += chunk
                continue
            try:
                obj, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = f.read(1 << 20)
                if not chunk:
                    raise
                buf += chunk
                if len(buf) > MAX_PARSER_BUF_BYTES:
                    raise RuntimeError(
                        f"{train_path}: buf grew past {MAX_PARSER_BUF_BYTES:,} "
                        f"bytes without a successful decode at index {idx}."
                    )
                continue
            if idx == wanted_idx:
                return obj
            buf = buf[end:]
            idx += 1


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
                        help="BM25 top-k profile entries per example (default 4, "
                             "matching the Task-LoRA recipe's k)")
    parser.add_argument("--limit", type=int, default=0, help="cap examples (smoke testing)")
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
    snapshot_idx = user_records[args.user].get("train_snapshot_idx")
    if snapshot_idx is None:
        sys.exit(f"ERROR: user {args.user!r} has no train_snapshot_idx (0 "
                 f"temporal-train records with a profile) -- should have been "
                 f"excluded by eligibility filtering upstream.")
    print(f"[user] {args.user!r}: training snapshot = temporal-train index {snapshot_idx}",
          flush=True)

    train_path = LONGLAMP_DIR / cfg["temporal_task"] / "train.json"
    if not train_path.exists():
        sys.exit(f"ERROR: {train_path} not found -- run data/download_longlamp.py "
                 f"--task {cfg['temporal_task']} first.")
    print(f"[scan] streaming {train_path} for index {snapshot_idx} ...", flush=True)
    snapshot = find_one_record(train_path, snapshot_idx)

    rid = str(snapshot.get(cfg["id_field"]))
    if rid != args.user:
        sys.exit(f"ERROR: record index {snapshot_idx} in {train_path} has "
                 f"{cfg['id_field']}={rid!r}, expected {args.user!r} -- "
                 f"user_records.json is stale (rebuild via "
                 f"data/longlamp_user_stats.py --overwrite).")

    profile = snapshot.get("profile", []) or []
    print(f"[snapshot] profile_size={len(profile)}", flush=True)
    if len(profile) < 2:
        sys.exit(f"ERROR: snapshot profile_size={len(profile)} < 2 -- leave-one-out "
                 f"BM25 needs >=2 entries; eligibility should have excluded this user.")

    t0 = time.time()
    lines = []
    n_skipped_empty = 0
    target_field = cfg["target_field"]
    for j, entry in enumerate(profile):
        if args.limit > 0 and len(lines) >= args.limit:
            break
        user_text = cfg["build_input"](entry)
        gold = str(entry.get(target_field, "")).strip()
        if not user_text.strip() or not gold:
            n_skipped_empty += 1
            continue
        query = cfg["query"](user_text)
        retrieved = retrieve_leave_one_out(cfg, query, profile, j, args.k)
        if retrieved:
            retrieved_ids = [str(r.get("id", "")) for r in retrieved]
            this_id = str(entry.get("id", ""))
            assert this_id not in retrieved_ids or not this_id, (
                f"self-retrieval at entry {j} (id={this_id}): {retrieved_ids}"
            )
            lines_ctx = ", and ".join(cfg["format"](it) for it in retrieved)
            system = lines_ctx + cfg["connector"]
        else:
            system = ""
        ex = {
            "task": f"LongLaMP_{cfg['temporal_task']}",
            "id": f"{args.user}-snap{snapshot_idx}-entry{j}",
            "system": system,
            "user": user_text,
            "assistant": gold,
        }
        lines.append(json.dumps(ex) + "\n")

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for line in lines:
            f.write(line)

    n_written = len(lines)
    print(f"[write] {n_written} examples ({n_skipped_empty} skipped empty "
          f"input/output) -> {out_path}", flush=True)

    meta = {
        "schema_version": 2,
        "tag": args.tag,
        "temporal_task": cfg["temporal_task"],
        "user_id": args.user,
        "user_tag": user_tag,
        "framing": "profile_entry_bm25_leave_one_out",
        "bm25_k": args.k,
        "snapshot_train_idx": snapshot_idx,
        "snapshot_profile_size": len(profile),
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
