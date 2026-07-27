#!/usr/bin/env python3
"""
Build a LongLaMP task's training corpus (user split) for its Task-LoRA.

Reads data/longlamp/<task>/train.json (written by data/download_longlamp.py),
does BM25 top-k retrieval of each user's profile entries, and writes a
compact JSONL ready for train/train.py.

Supports three tasks (LL1/LL2/LL3): product_review_user, abstract_generation_user,
topic_writing_user. Per-task BM25 index/query construction and PPEP profile-entry
templates are sourced from the LongLaMP authors' own training code
(github.com/LongLaMP-Benchmark/LongLaMP-Benchmark, longLaMP/prompts/prompts.py),
fetched directly rather than guessed — see per-task comments below. LL1's
product_review_user template originally omitted the literal quote marks the
real template wraps every interpolated value in (found only once this repo
was located, after LL1 had already trained/evaluated); this is now corrected
here for LL2/LL3 but LL1's already-built corpus/checkpoint were not rebuilt
-- the effect size that mattered for LL1 (the repetition-loop regression) is
far too large to plausibly hinge on quote marks. See
experiments/2026-07-25-longlamp-review-ll1.md.

BM25 + role layout here MUST match eval/eval_longlamp.py exactly (same
cardinal rule as LaMP's build_dataset.py / eval_lamp.py pair) — the relevant
blocks (BM25, tokenize, retrieve_profile, PPEP formatting) are duplicated
from eval_longlamp.py rather than imported, same rationale as LaMP's harness
(this script needs to run standalone in a Condor sandbox with no
import-path gymnastics).

Every task's `input` field is already a fully-templated instruction
(verified via the HF datasets-server API) — the `user` turn is `input`
verbatim, no task-instruction template is built here. The one exception is
BM25 *query* construction for abstract_generation_user, which uses only the
text after "items:" in `input` (the bulleted keyword list), matching the
paper's `extract_before_bullets` exactly -- product_review_user and
topic_writing_user both query on the full `input` string (also matching the
paper's own `generate_query_for_product_review_writing` /
`generate_query_for_topic_writing`, the latter via an identity function).

Each line is one training example:
    {"task": "LongLaMP_<task>", "id": "<user_id>-<index>",
     "system": "<PPEP-formatted retrieved profile context>",
     "user":   "<input>",
     "assistant": "<output>"}

Usage (CPU is enough — no GPU needed for BM25 + JSONL write):
    python train/build_longlamp_dataset.py --task product_review_user --k 4 --seed 0
    python train/build_longlamp_dataset.py --task abstract_generation_user --limit 100   # smoke
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


def _first_750_words(text: str) -> str:
    """Verbatim port of the paper's `extract_first_750_words` -- applied only
    to the abstract text injected into the PPEP prompt (create_abstract_paper_prompt),
    NOT to the BM25 index (generation_abstract_query_corpus_maker indexes the
    full untruncated abstract)."""
    return " ".join(str(text).split()[:750])


def _extract_after_items(input_string: str):
    """Verbatim port of the paper's `extract_before_bullets` (its name is
    misleading -- despite the name, it returns the text AFTER "items:", i.e.
    the bulleted keyword list itself, not the title sentence before it).
    Falls back to the full input if "items:" isn't found (shouldn't happen
    on real data, but avoids a None query crashing BM25 on a malformed row)."""
    idx = input_string.find("items:")
    if idx == -1:
        return input_string
    return input_string[idx + len("items:"):].strip()


# --- Per-task configuration ---------------------------------------------------
# id_field:     the on-disk record field holding the user id (differs per task
#               on purpose -- see data/download_longlamp.py, which keeps each
#               task's original HF field name rather than a renamed alias).
# index_field:  profile-entry text BM25 indexes against (paper's own corpus
#               construction, ported verbatim).
# query:        how to build the BM25 query string from `input` (paper's own
#               generate_query_for_* functions, ported verbatim).
# format:       PPEP template per profile entry (paper's own create_*_prompt
#               functions, ported verbatim, quote marks included).
# connector:    trailing text joining the profile block to `input` (paper's
#               own per-task wording, verbatim including punctuation).
# Short filename tag per task -- kept as an explicit table (not derived from
# the task key) so `--task product_review_user` reuses the exact path LL1
# already built (`longlamp_train_review_user_*`), not a renamed one.
FILE_TAGS = {
    "product_review_user": "review",
    "abstract_generation_user": "abstract",
    "topic_writing_user": "topic",
}

TASKS = {
    "product_review_user": {
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
    "abstract_generation_user": {
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
    "topic_writing_user": {
        "id_field": "author",
        # The paper's raw per-profile-entry fields are named input/output
        # internally (generate_query_for_topic_writing indexes on
        # f'{x["input"]} {x["output"]}'); the public HF release renames
        # these to summary/content respectively (matched by semantic role --
        # see the download script's docstring and the LL2/LL3 build commit
        # message for the reasoning).
        "index_field": lambda p: " ".join([str(p.get("summary", "")), str(p.get("content", ""))]),
        "query": lambda inp: inp,
        "format": lambda p: (
            f'"{p.get("summary", "")}" is a summary for "{p.get("content", "")}"'
        ),
        "connector": ". Following the given patterns, ",
    },
}


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


def retrieve_profile(task: str, query: str, profile: list, k: int) -> list:
    if not profile or k <= 0:
        return []
    index_field = TASKS[task]["index_field"]
    docs = [tokenize(index_field(it)) for it in profile]
    bm25 = BM25(docs)
    idxs = bm25.top_k(tokenize(query), k)
    return [profile[i] for i in idxs]


def build_example(task: str, record: dict, idx: int, k: int) -> dict:
    cfg = TASKS[task]
    raw_input = record["input"]
    query = cfg["query"](raw_input)
    retrieved = retrieve_profile(task, query, record.get("profile", []), k)
    if retrieved:
        lines = ", and ".join(cfg["format"](it) for it in retrieved)
        system = lines + cfg["connector"]
    else:
        system = ""
    user_id = record.get(cfg["id_field"], "unk")
    return {
        "task": f"LongLaMP_{task}",
        "id": f"{user_id}-{idx}",
        "system": system,
        "user": raw_input,
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
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
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
    task_short = FILE_TAGS[args.task]
    limit_tag = f"_limit{args.limit}" if args.limit > 0 else ""
    out_path = Path(DATA_OUT_DIR) / f"longlamp_train_{task_short}_user_{suffix}{limit_tag}.jsonl"
    meta_path = Path(DATA_OUT_DIR) / f"longlamp_train_{task_short}_user_{suffix}{limit_tag}.meta.json"

    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] build_longlamp_dataset task={args.task} k={args.k} seed={args.seed} "
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

    train_path = Path(LONGLAMP_DIR) / args.task / "train.json"
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
        ex = build_example(args.task, r, i, args.k)
        lines.append(json.dumps(ex) + "\n")
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"  {i + 1}/{len(records)} ({rate:.0f}/s)", flush=True)

    # Deterministic shuffle, matching build_dataset.py's mixed-corpus convention
    # (this corpus is single-task, but the shuffle still decorrelates training
    # order from the source file's on-disk user ordering).
    import random

    rng = random.Random(args.seed)
    rng.shuffle(lines)

    with out_path.open("w") as out:
        for line in lines:
            out.write(line)

    size_mb = out_path.stat().st_size / 1e6
    print(f"{args.task}: {len(lines)} examples ({size_mb:.1f} MB) -> {out_path}")

    meta = {
        "schema_version": 1,
        "task": args.task,
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
