#!/usr/bin/env python3
"""
Build a single-user training corpus for User-LoRA fine-tuning.

This is the per-user companion to `train/build_dataset.py`. Three
training-data *framings* are supported via `--framing`:

  --framing profile  (default; Round 1 + Round 2 paths):
      Uses the user's `profile` (a chronological list of past interactions)
      from their **latest** time-split train record as the training corpus.
      LaMP_4 profile entries are (article_body, user_headline) pairs. For
      u00000011 specifically the snapshot record's profile is ~1100 entries.

      Sub-modes via `--bm25-k`:
        --bm25-k 0  (Round 1):
            "bare" profile-entry framing. system="", user=text, assistant=title.
            No retrieval at train time.
        --bm25-k K  (K>0; Round 2 variant B):
            Each profile entry e_j is emitted with a system slot populated by
            BM25 top-K retrieval over the user's own strictly-prior profile
            entries (matches eval_lamp.py's inference prompt shape).

      Tasks supported under this framing: LaMP_3, LaMP_4, LaMP_2_movies,
      LaMP_5 — i.e. every task whose profile entries carry both the task's
      input field and its target field. LaMP_2_movies entries are
      (description, tag) pairs; LaMP_5 entries are (abstract, title) pairs.

  --framing records  (Round 4 path, requires --bm25-k K>0):
      Uses the user's time-split train *records* — one (input, gold) pair per
      LaMP_4 train-period record (241 for u00000011, disjoint from 21 dev /
      25 test). For each record: user = record.input byte-for-byte (e.g.,
      "Generate a headline for the following article: ..."), assistant =
      train_outputs[id].output, system = SYSTEM_PREAMBLE + BM25 top-K over
      the snapshot profile (built once outside the per-record loop because
      this user's profile is identical across their 241 train records).
      Same shape as the eval-time task; the User-LoRA's training distribution
      matches the eval distribution. Used by LaMP_4 (Round 4/6) and
      LaMP_2_news (Round 10 / PT3) — the two tasks with real per-user
      train-record volume.

  --framing unsupervised  (R11/PT4 + R12/PT5 path, LaMP_1 and LaMP_7 only):
      OPPU's recipe for tasks whose history does NOT align with the task
      format (arXiv:2402.04401 §3: "we replace the user history output y_u in
      personal PEFT training objectives with right-shifted history x_u' for
      unsupervised next token prediction"). Emits one raw-text example per
      profile entry — no chat template, no system/user/assistant roles, no
      BM25 retrieval, no target field. Output rows are `{"task", "id",
      "text"}` and are consumed by `train/train_unsupervised_clm.py`, NOT by
      `train/train.py`. `--bm25-k` must be 0 (there is no system slot to
      retrieve into). Eval is unaffected: inference still uses the ordinary
      supervised prompt shape with BM25 retrieval.

        LaMP_1: text = f"{entry['title']}\\n\\n{entry['abstract']}"
        LaMP_7: text = entry['text']  (a raw past tweet)

Per the pinned designs (see
experiments/2026-06-14-user-lora-round1-plan.md,
experiments/2026-06-16-user-lora-round2-B-plan.md, and
experiments/2026-06-17-user-lora-round4-plan.md):
  - profile framing: target = `title` (the user's headline) for LaMP_4
  - Round 2 / variant B: BM25 retrieval pool is strictly-prior by ISO date
    (decision A1b). Entries missing a `date` field are dropped from the
    training set; entries with equal date strings are not retrievable for
    each other (strict-prior tie-breaking). Query string = e_j's raw `text`
    field (decision Q.i / 9a). System slot byte-matches build_dataset.py's
    formatting (same TASKS lambda, same SYSTEM_PREAMBLE).
  - Round 4 / records framing: no strict-prior date filter (records have no
    `date` field; A1-lamp didn't filter either). Query = record.input.
    Retrieval pool = full profile (identical across the user's records;
    fingerprint-asserted at build time).

The BM25 / tokenize / SYSTEM_PREAMBLE / TASKS format blocks are duplicated
from `train/build_dataset.py` rather than imported so the script stays
standalone in a Condor sandbox. If you change retrieval here, change it
there (and in eval_lamp.py) too — the byte-for-byte match is the cardinal
train/eval consistency rule.

Profile-snapshot sourcing (profile + unsupervised framings)
-----------------------------------------------------------
The snapshot is normally the user's largest-profile TRAIN-split record. For
LaMP_1, LaMP_2_movies and LaMP_5 that lookup returns nothing: **zero** of
their users hold both a train record and a test record (each user appears in
exactly one split), so every user we can actually evaluate on has no train
record at all. For those users the snapshot falls back to the user's own
TEST record's `profile` field.

That is not leakage: a record's `profile` is by construction the user's
history *prior to* that record, never the record's own query or gold — it is
the same field `eval_lamp.py` reads for BM25 retrieval at eval time, and
profile entries already do double duty as training data plus retrieval
context for LaMP_3/LaMP_4. As belt-and-braces, any profile entry whose `id`
collides with the snapshot record's own `id` is dropped and counted
(`n_dropped_self_record` in the meta sidecar).

Output format for the profile/records framings matches `train/train.py`'s
`build_example` (flat dict with `system`/`user`/`assistant` strings); the
unsupervised framing emits `{"task", "id", "text"}` for
`train/train_unsupervised_clm.py`.

Output:
    data/lamp_user_train_<task>_<user>_bare.jsonl            (profile, --bm25-k 0)
    data/lamp_user_train_<task>_<user>_bm25k<K>.jsonl        (profile, --bm25-k K>0)
    data/lamp_user_train_<task>_<user>_records_bm25k<K>.jsonl (records, --bm25-k K>0)
    data/lamp_user_train_<task>_<user>_unsup.jsonl            (unsupervised)
    (with matching .meta.json sidecar)

Usage (CPU-only):
    python train/build_user_dataset.py --task LaMP_4 --user u00000011
    python train/build_user_dataset.py --task LaMP_4 --user u00000011 --bm25-k 4
    python train/build_user_dataset.py --task LaMP_4 --user u00000011 --framing records --bm25-k 4
    python train/build_user_dataset.py --task LaMP_2_news --user u00004181 --framing records --bm25-k 4
    python train/build_user_dataset.py --task LaMP_2_movies --user u00008123 --bm25-k 4
    python train/build_user_dataset.py --task LaMP_5 --user u00012345 --bm25-k 4
    python train/build_user_dataset.py --task LaMP_7 --user u00000042 --framing unsupervised
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

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/home/ange00008/projects/mobileFT_distill")
)
USER_RECORDS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"
TIME_SPLIT_DIR = PROJECT_ROOT / "data" / "lamp_time"
DATA_OUT_DIR = PROJECT_ROOT / "data"

# Per-task: (input_field, target_field) on profile entries — the tasks whose
# profile entries are shaped like the task's own input -> output pair, so a
# profile entry can be reframed as a supervised training example.
#   LaMP_3        (text, score)         review -> rating
#   LaMP_4        (text, title)         article -> headline
#   LaMP_2_movies (description, tag)    blurb -> tag
#   LaMP_5        (abstract, title)     abstract -> title
# LaMP_2_news uses `--framing records` instead (it has real per-user train
# records, so no reframing is needed). LaMP_1 and LaMP_7 are the two
# genuinely history-misaligned tasks (OPPU arXiv:2402.04401 §3) and use
# `--framing unsupervised`; neither has a profile-level target field.
PROFILE_FRAMING = {
    "LaMP_4": ("text", "title"),
    "LaMP_3": ("text", "score"),
    "LaMP_2_movies": ("description", "tag"),
    "LaMP_5": ("abstract", "title"),
}

# Per-task: profile entry -> raw text, for `--framing unsupervised`. These are
# the tasks with no usable profile-level target, where OPPU substitutes
# next-token prediction on the raw history.
UNSUPERVISED_TEXT = {
    "LaMP_1": lambda e: (
        f'{str(e.get("title", "")).strip()}\n\n{str(e.get("abstract", "")).strip()}'
    ).strip(),
    "LaMP_7": lambda e: str(e.get("text", "")).strip(),
}

# -----------------------------------------------------------------------------
# BM25 retrieval + per-task formatting (duplicated from train/build_dataset.py
# and eval/eval_lamp.py). CRITICAL: must stay byte-equal with those copies —
# any drift breaks the train/eval prompt-shape match that is variant B's whole
# point. The duplication is intentional so this script stays standalone in a
# Condor sandbox; the byte-match is verified at smoke time (Step 3).
# -----------------------------------------------------------------------------
ENTRY_CHARS = 600
TITLE_CHARS = 200


def trim(text: str, n: int = ENTRY_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


TASKS = {
    "LaMP_3": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- Review: "{trim(it.get("text", ""))}" — the user rated it {it.get("score", "?")}/5',
    },
    "LaMP_4": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- Article: "{trim(it.get("text", ""))}" — the user\'s headline: "{trim(it.get("title", ""), TITLE_CHARS)}"',
    },
    "LaMP_7": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- "{trim(it.get("text", ""))}"',
    },
    "LaMP_1": {
        "index_field": lambda it: it.get("abstract", "") or it.get("title", ""),
        "format": lambda it: f'- "{trim(it.get("title", ""), TITLE_CHARS)}": {trim(it.get("abstract", ""))}',
    },
    "LaMP_2_movies": {
        "index_field": lambda it: it.get("description", ""),
        "format": lambda it: f'- Movie: "{trim(it.get("description", ""))}" — tagged "{it.get("tag", "?")}"',
    },
    "LaMP_2_news": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- Article: "{trim(it.get("title", ""), TITLE_CHARS)}": "{trim(it.get("text", ""))}" — categorized "{it.get("category", "?")}"',
    },
    "LaMP_5": {
        "index_field": lambda it: it.get("abstract", "") or it.get("title", ""),
        "format": lambda it: f'- "{trim(it.get("title", ""), TITLE_CHARS)}": {trim(it.get("abstract", ""))}',
    },
}

# Closed label vocabulary for LaMP-2-movies, duplicated byte-for-byte from
# eval/eval_lamp.py's LAMP2_MOVIES_LABELS (same standalone-in-a-Condor-sandbox
# rationale as the BM25/TASKS duplication above). Only movies needs it here:
# its profile-framing question string has to enumerate the tag universe the
# way the real eval-time `input` does. LaMP-2-news never synthesizes a
# question (records framing uses the real `input` verbatim), so its label list
# is not needed in this script.
LAMP2_MOVIES_LABELS = [
    "action", "based on a book", "classic", "comedy", "dark comedy",
    "dystopia", "fantasy", "psychology", "romance", "sci-fi",
    "social commentary", "thought-provoking", "true story", "twist ending",
    "violence",
]

# The order the tags actually appear in inside real LaMP-2-movies `input`
# strings. This is a SINGLE FIXED ORDER, not a per-example shuffle: verified
# against every one of the 1,410 dev records plus the first 1,500 test and
# 1,500 train records — 4,410 records, exactly 1 distinct order.
#
# The 2026-07-27 plan (Piece 1 #4) pinned a deterministic per-example shuffle
# here, on the premise that the real order was arbitrary per example. That
# premise was wrong — what the original check established was only that the
# order isn't alphabetical (i.e. doesn't match LAMP2_MOVIES_LABELS above).
# Since the order is fixed and observable, reproducing it verbatim is the
# train/eval-consistency-preserving choice and shuffling would introduce a
# gratuitous mismatch: at eval time `eval_lamp.py` feeds the record's real
# `input` through unchanged, so the model always sees THIS order.
LAMP2_MOVIES_PROMPT_ORDER = [
    "sci-fi", "based on a book", "comedy", "action", "twist ending",
    "dystopia", "dark comedy", "classic", "psychology", "fantasy",
    "romance", "thought-provoking", "social commentary", "violence",
    "true story",
]
assert set(LAMP2_MOVIES_PROMPT_ORDER) == set(LAMP2_MOVIES_LABELS), (
    "LaMP-2-movies prompt order and label universe disagree"
)

SYSTEM_PREAMBLE = (
    "The following are examples of this user's past activity. "
    "Use them to match this user's preferences and writing style.\n\n"
)


def wrap_user_text(task: str, profile_entry: dict) -> str:
    """Render a profile entry as the eval-time question string for `task`.

    The point is train/eval prompt-shape parity: the User-LoRA should see the
    same `user` slot at training time that `eval_lamp.py` will hand it at
    inference. LaMP_4 is the one task that stays bare (raw article text),
    matching the Round-1/2 pinned design.
    """
    if task == "LaMP_3":
        text = str(profile_entry.get("text", "")).strip()
        return (
            "What is the score of the following review on a scale of 1 to 5? "
            "just answer with 1, 2, 3, 4, or 5 without further explanation. "
            f"review: {text}"
        )
    if task == "LaMP_2_movies":
        # Raw, NOT trim()'d: `trim` truncates at 600 chars and collapses
        # whitespace, which rewrote 18/400 real dev questions in testing.
        # `trim` exists to bound the *retrieved system context*, not the
        # question itself — the real `input` carries the description verbatim.
        description = str(profile_entry.get("description", ""))
        return (
            "Which tag does this movie relate to among the following tags? "
            "Just answer with the tag name without further explanation. "
            f"tags: [{', '.join(LAMP2_MOVIES_PROMPT_ORDER)}] "
            f"description: {description}"
        )
    if task == "LaMP_5":
        # Raw, NOT .strip()'d: 7/400 real abstracts carry leading whitespace
        # and 49/400 trailing, all of which the real `input` preserves.
        abstract = str(profile_entry.get("abstract", ""))
        return f"Generate a title for the following abstract of a paper: {abstract}"
    return str(profile_entry.get("text", "")).strip()

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


MAX_PARSER_BUF_BYTES = 64 * 1024 * 1024


def stream_json_array(path):
    """Yield each top-level object from a JSON-array file without loading the
    whole array. Mirrors the implementation in train/build_dataset.py — kept
    duplicated rather than imported so this script remains standalone in a
    Condor sandbox."""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        while "[" not in buf:
            chunk = f.read(65536)
            if not chunk:
                return
            buf += chunk
        buf = buf[buf.index("[") + 1 :]
        while True:
            buf = buf.lstrip()
            if buf.startswith(","):
                buf = buf[1:].lstrip()
            if buf.startswith("]"):
                return
            if not buf:
                chunk = f.read(65536)
                if not chunk:
                    return
                buf += chunk
                continue
            try:
                obj, idx = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = f.read(65536)
                if not chunk:
                    if buf.strip(" \t\r\n,]"):
                        raise
                    return
                buf += chunk
                if len(buf) > MAX_PARSER_BUF_BYTES:
                    raise RuntimeError(
                        f"stream_json_array: buf grew past "
                        f"{MAX_PARSER_BUF_BYTES:,} chars without a successful "
                        f"decode at offset {f.tell():,}."
                    )
                continue
            yield obj
            buf = buf[idx:]


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


def find_latest_record(task: str, ids: set, split: str) -> dict:
    """Stream one time-split questions file; return the user's record with the
    largest profile (= latest in chronological time-split ordering). Holds
    only the current best record in memory.

    `split` is "train" normally. LaMP_1 / LaMP_2_movies / LaMP_5 have zero
    users holding both a train record and a test record, so for those tasks
    the caller falls back to split="test" — see the module docstring's
    "Profile-snapshot sourcing" section for why that is not leakage.
    """
    q_path = TIME_SPLIT_DIR / task / f"{split}_questions.json"
    best = None
    best_size = -1
    n_seen = 0
    t0 = time.time()
    for r in stream_json_array(str(q_path)):
        if str(r.get("id")) not in ids:
            continue
        n_seen += 1
        prof = r.get("profile", [])
        if len(prof) > best_size:
            best = r
            best_size = len(prof)
        if n_seen % 50 == 0:
            print(f"  scanned: {n_seen}/{len(ids)} {split} records found "
                  f"(best profile size so far {best_size})", flush=True)
        if n_seen == len(ids):
            # Early-exit: nothing left to find; don't scan the rest of a
            # multi-hundred-MB file.
            break
    if best is None or n_seen != len(ids):
        raise RuntimeError(
            f"Found {n_seen} of {len(ids)} expected {split} records — "
            f"records JSON and time-split disagree."
        )
    print(f"  done: {n_seen} {split} records scanned in {time.time()-t0:.0f}s; "
          f"snapshot record_id={best['id']} profile_size={best_size}", flush=True)
    return best


def resolve_snapshot(task: str, train_ids: set, test_ids: set, dev_ids: set):
    """Pick the record whose `profile` becomes this user's training corpus.

    Prefers a train-split record. Falls back to test, then dev, for the tasks
    where users appear in exactly one split (LaMP_1, LaMP_2_movies, LaMP_5) —
    without the fallback those tasks have zero buildable users, since every
    user we can evaluate on has no train record at all.

    Returns (record, split_name).
    """
    for ids, split in ((train_ids, "train"), (test_ids, "test"), (dev_ids, "dev")):
        if ids:
            if split != "train":
                print(
                    f"[snapshot] no train-split record for this user; falling "
                    f"back to their {split}-split record's profile "
                    f"(history prior to that record — see module docstring)",
                    flush=True,
                )
            print(f"[scan] streaming {TIME_SPLIT_DIR / task / f'{split}_questions.json'} ...",
                  flush=True)
            return find_latest_record(task, ids, split), split
    raise RuntimeError(
        "User has no records in any split — nothing to build a snapshot from."
    )


def emit_bm25_records(profile: list, task: str, k: int, out_f,
                      exclude_id: str = "") -> dict:
    """Variant-B emission: per profile entry, BM25-retrieve over strictly-prior
    entries, format with the task's lambda + SYSTEM_PREAMBLE, write JSONL.

    Pool rule (decision A1b + strict-prior tie-breaking from the Round-2 plan):
    for entry $e_j$ with date $d_j$, pool = $\\{e_i : i \\neq j$ AND $e_i$ has a
    date AND $d_i < d_j\\}$. Entries missing a `date` field are dropped from
    the training set (no JSONL row); under strict-prior they are also not
    retrievable into anyone's pool, so the dropped count is recorded but the
    profile is otherwise untouched.

    `exclude_id`, when set, is the snapshot record's own id: any profile entry
    carrying that id is dropped from BOTH the training set and every
    retrieval pool. Belt-and-braces for the test-record snapshot fallback —
    a record's profile is history prior to that record and should never
    contain the record itself, so this is expected to drop 0 entries.

    Returns a stats dict for the meta sidecar.
    """
    input_field, target_field = PROFILE_FRAMING[task]

    if exclude_id:
        n_dropped_self_record = sum(
            1 for e in profile if str(e.get("id", "")) == exclude_id
        )
        profile = [e for e in profile if str(e.get("id", "")) != exclude_id]
    else:
        n_dropped_self_record = 0

    # Tokenize each profile entry once for BM25 (avoid re-tokenizing per query).
    profile_tokens = [tokenize(TASKS[task]["index_field"](e)) for e in profile]

    # Eligible-for-training = has a non-empty `date` field.
    eligible = [i for i, e in enumerate(profile) if e.get("date")]
    n_dropped_no_date = len(profile) - len(eligible)
    eligible.sort(key=lambda i: str(profile[i]["date"]))

    n_written = 0
    n_skipped_empty = 0
    n_examples_with_empty_pool = 0

    for j in eligible:
        e_j = profile[j]
        d_j = str(e_j["date"])
        user_text = wrap_user_text(task, e_j)
        gold = str(e_j.get(target_field, "")).strip()
        # Test the SOURCE field, not the wrapped string: for the tasks whose
        # question is synthesized (LaMP_3, LaMP_2_movies, LaMP_5) the wrapper
        # prefix is always non-empty, so an entry with an empty body would
        # otherwise slip through as a degenerate example. Real data has these
        # (1 empty abstract per ~2,600 LaMP_5 profile entries).
        if not str(e_j.get(input_field, "")).strip() or not gold:
            n_skipped_empty += 1
            continue

        # Strict-prior pool: equal-date entries are NOT in each other's pool.
        pool_idxs = [
            i for i in range(len(profile))
            if i != j and profile[i].get("date") and str(profile[i]["date"]) < d_j
        ]

        if not pool_idxs:
            system = ""
            n_examples_with_empty_pool += 1
        else:
            docs = [profile_tokens[i] for i in pool_idxs]
            bm25 = BM25(docs)
            top = bm25.top_k(tokenize(user_text), k)
            retrieved = [profile[pool_idxs[t]] for t in top]

            e_j_id = str(e_j.get("id", ""))
            retrieved_ids = [str(r.get("id", "")) for r in retrieved]
            assert e_j_id not in retrieved_ids, (
                f"self-retrieval at entry id={e_j_id}: retrieved {retrieved_ids}"
            )

            lines = "\n".join(TASKS[task]["format"](it) for it in retrieved)
            system = SYSTEM_PREAMBLE + lines

        rec = {
            "task": task,
            "id": str(e_j.get("id", "")),
            "system": system,
            "user": user_text,
            "assistant": gold,
        }
        out_f.write(json.dumps(rec) + "\n")
        n_written += 1

    return {
        "n_written": n_written,
        "n_skipped_empty": n_skipped_empty,
        "n_dropped_no_date": n_dropped_no_date,
        "n_examples_with_empty_pool": n_examples_with_empty_pool,
        "n_dropped_self_record": n_dropped_self_record,
    }


def emit_unsupervised(profile: list, task: str, out_f,
                      exclude_id: str = "") -> dict:
    """OPPU's right-shifted-history path (LaMP_1, LaMP_7): one raw-text row
    per profile entry, no prompt shape at all.

    The "right shift" is the causal-LM objective itself — predicting token
    t+1 from tokens ≤t over the user's own history — so there is nothing to
    do here beyond emitting the history text; the shift happens in
    `train/train_unsupervised_clm.py`'s label construction. No BM25, no
    system slot, no target field, and no date filter (there is no
    strictly-prior leakage concern when the entry IS the target).
    """
    render = UNSUPERVISED_TEXT[task]

    n_dropped_self_record = 0
    n_written = 0
    n_skipped_empty = 0
    for e in profile:
        eid = str(e.get("id", ""))
        if exclude_id and eid == exclude_id:
            n_dropped_self_record += 1
            continue
        text = render(e)
        if not text:
            n_skipped_empty += 1
            continue
        out_f.write(json.dumps({"task": task, "id": eid, "text": text}) + "\n")
        n_written += 1

    return {
        "n_written": n_written,
        "n_skipped_empty": n_skipped_empty,
        "n_dropped_self_record": n_dropped_self_record,
    }


def find_train_records(task: str, train_ids: set) -> list:
    """Stream the time-split train_questions file once; return ALL records
    whose ID is in `train_ids`. Used by the Round-4 records-framing path.

    Each returned dict has the full record shape (id, input, profile, ...).
    Order is the stream order, which is the file order. The 241 records for
    u00000011 are a tiny fraction of the file so total memory pressure is
    bounded.
    """
    q_path = TIME_SPLIT_DIR / task / "train_questions.json"
    found = {}
    n_seen = 0
    t0 = time.time()
    for r in stream_json_array(str(q_path)):
        rid = str(r.get("id"))
        if rid not in train_ids:
            continue
        if rid in found:
            raise RuntimeError(
                f"Duplicate train record id {rid} in {q_path}; corpus invariant broken."
            )
        found[rid] = r
        n_seen += 1
        if n_seen % 50 == 0:
            print(f"  scanned: {n_seen}/{len(train_ids)} train records found",
                  flush=True)
        if n_seen == len(train_ids):
            # Early-exit: we've found every record we need; no point scanning
            # the rest of the 863 MB file.
            break
    if n_seen != len(train_ids):
        missing = train_ids - set(found)
        raise RuntimeError(
            f"Found {n_seen}/{len(train_ids)} train records; "
            f"{len(missing)} missing (e.g. {sorted(list(missing))[:5]}). "
            f"User records JSON and time-split train_questions disagree."
        )
    print(f"  done: {n_seen} train records scanned in {time.time()-t0:.0f}s",
          flush=True)
    return list(found.values())


def emit_record_bm25(
    records: list, outputs_by_id: dict, task: str, k: int, out_f
) -> dict:
    """Round-4 records-framing emission: one JSONL line per train-period
    record. user = record.input verbatim; assistant = outputs_by_id[record.id];
    system = SYSTEM_PREAMBLE + BM25 top-K over the snapshot profile (built
    once outside the loop because this user's 241 records all share the same
    profile — fingerprint-asserted below).

    Pre-asserts:
      - All record IDs are present in outputs_by_id (else fail listing missing).

    Profile stability: R4/R6 relied on every one of a user's train records
    carrying an identical profile, which let a single BM25 index serve the
    whole loop. That invariant was verified for LaMP_4 but was never tested
    against LaMP_2_news, whose users hold up to 211 train records. Rather
    than fail on drift, we check the fingerprint and pick the strategy: one
    shared index when stable (LaMP_4's fast path, unchanged), otherwise a
    per-record index over that record's own profile — which is the
    semantically correct thing regardless, since a record's profile is its
    own history-prior snapshot. Which path ran is recorded as
    `profile_stable` in the meta sidecar, so it is never a silent difference.

    Returns a stats dict for the meta sidecar.
    """
    if k <= 0:
        raise ValueError("emit_record_bm25 requires k>0 (records framing).")

    # --- pre-asserts -------------------------------------------------------
    missing_outputs = [str(r.get("id")) for r in records
                       if str(r.get("id")) not in outputs_by_id]
    if missing_outputs:
        raise RuntimeError(
            f"{len(missing_outputs)} train records missing from outputs: "
            f"e.g. {missing_outputs[:5]}"
        )

    # Profile-equality fingerprint across all records (cheap: len + first-entry id).
    first_profile = records[0].get("profile", []) or []
    fp = (
        len(first_profile),
        str(first_profile[0].get("id", "")) if first_profile else "",
    )
    profile_stable = True
    for r in records[1:]:
        prof = r.get("profile", []) or []
        rfp = (
            len(prof),
            str(prof[0].get("id", "")) if prof else "",
        )
        if rfp != fp:
            profile_stable = False
            print(
                f"[profile] drift detected: record {r.get('id')} has "
                f"(len={rfp[0]}, profile[0].id={rfp[1]}) vs reference "
                f"(len={fp[0]}, profile[0].id={fp[1]}) — building one BM25 "
                f"index per record over that record's own profile.",
                flush=True,
            )
            break

    # --- build BM25 once over the shared profile (stable fast path) -------
    profile = first_profile
    if profile_stable:
        profile_tokens = [tokenize(TASKS[task]["index_field"](e)) for e in profile]
        shared_bm25 = BM25(profile_tokens)
    else:
        shared_bm25 = None

    # --- per-record loop --------------------------------------------------
    n_written = 0
    n_skipped_empty = 0
    profile_sizes = []

    for r in records:
        rid = str(r.get("id"))
        user_text = str(r.get("input", "")).strip()
        gold = str(outputs_by_id.get(rid, "")).strip()
        if not user_text or not gold:
            n_skipped_empty += 1
            continue

        if shared_bm25 is not None:
            rec_profile, bm25 = profile, shared_bm25
        else:
            rec_profile = r.get("profile", []) or []
            bm25 = BM25([tokenize(TASKS[task]["index_field"](e)) for e in rec_profile])
        profile_sizes.append(len(rec_profile))

        top = bm25.top_k(tokenize(user_text), k)
        retrieved = [rec_profile[t] for t in top]
        lines = "\n".join(TASKS[task]["format"](it) for it in retrieved)
        system = SYSTEM_PREAMBLE + lines

        rec = {
            "task": task,
            "id": rid,
            "system": system,
            "user": user_text,
            "assistant": gold,
        }
        out_f.write(json.dumps(rec) + "\n")
        n_written += 1

    return {
        "n_written": n_written,
        "n_skipped_empty": n_skipped_empty,
        "profile_size": len(profile),
        "profile_stable": profile_stable,
        "profile_size_min": min(profile_sizes) if profile_sizes else None,
        "profile_size_max": max(profile_sizes) if profile_sizes else None,
        "n_train_records_resolved": len(records),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--user", required=True,
                        help="user fingerprint, e.g. u00000011")
    parser.add_argument("--framing",
                        choices=["profile", "records", "unsupervised"],
                        default="profile",
                        help="training-example unit: profile-entry pairs "
                             "(Rounds 1+2; default), train-period record "
                             "(input, gold) pairs (Round 4/6/10, requires "
                             "--bm25-k>0), or raw profile-entry text for "
                             "OPPU's unsupervised path (R11/R12, LaMP_1 and "
                             "LaMP_7 only, requires --bm25-k 0)")
    parser.add_argument("--bm25-k", type=int, default=0,
                        help="BM25 top-k retrieval over the user's profile "
                             "into the system slot at train time. "
                             "profile framing: 0 = bare (Round 1), K>0 = "
                             "strictly-prior pool (Round 2 / variant B). "
                             "records framing: K>0 required, full profile pool. "
                             "unsupervised framing: must be 0 (no system slot).")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing JSONL / meta (default: refuse)")
    args = parser.parse_args()
    if args.bm25_k < 0:
        sys.exit("ERROR: --bm25-k must be >= 0")
    if args.framing == "records" and args.bm25_k <= 0:
        sys.exit("ERROR: --framing records requires --bm25-k > 0 "
                 "(bare records framing is not a Round-4 axis; re-open the "
                 "design if you want it).")
    if args.framing == "profile" and args.task not in PROFILE_FRAMING:
        sys.exit(
            f"ERROR: --framing profile is not supported for {args.task} — its "
            f"profile entries carry no target field. Supported: "
            f"{sorted(PROFILE_FRAMING)}. Use --framing records "
            f"(LaMP_2_news) or --framing unsupervised "
            f"({sorted(UNSUPERVISED_TEXT)})."
        )
    if args.framing == "unsupervised":
        if args.task not in UNSUPERVISED_TEXT:
            sys.exit(
                f"ERROR: --framing unsupervised is only defined for "
                f"{sorted(UNSUPERVISED_TEXT)} (the two history-misaligned "
                f"tasks per OPPU §3); {args.task} has an aligned profile and "
                f"should use supervised framing."
            )
        if args.bm25_k != 0:
            sys.exit("ERROR: --framing unsupervised requires --bm25-k 0 — "
                     "there is no system slot to retrieve into on this path.")

    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] build_user_dataset task={args.task} user={args.user} "
        f"framing={args.framing} bm25_k={args.bm25_k} commit={commit_short} "
        f"dirty={provenance.get('git_dirty')} "
        f"host={provenance.get('hostname')}",
        flush=True,
    )

    if args.framing == "records":
        suffix = f"records_bm25k{args.bm25_k}"
    elif args.framing == "unsupervised":
        suffix = "unsup"
    else:
        suffix = "bare" if args.bm25_k == 0 else f"bm25k{args.bm25_k}"
    out_path = DATA_OUT_DIR / f"lamp_user_train_{args.task}_{args.user}_{suffix}.jsonl"
    meta_path = DATA_OUT_DIR / f"lamp_user_train_{args.task}_{args.user}_{suffix}.meta.json"

    existing = [p for p in (out_path, meta_path) if p.exists()]
    if existing and not args.overwrite:
        print("ERROR: refusing to overwrite existing files:", file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        print("Pass --overwrite to replace.", file=sys.stderr)
        sys.exit(1)

    # --- Resolve the user's time-split train record IDs --------------------
    records_json = USER_RECORDS_DIR / f"{args.task}_user_records.json"
    if not records_json.exists():
        sys.exit(f"ERROR: {records_json} missing — run data/lamp_user_stats.py first.")
    users = json.loads(records_json.read_text())
    if args.user not in users:
        sys.exit(f"ERROR: user {args.user} not in {records_json}.")
    train_ids = set(users[args.user]["train"])
    dev_ids = set(users[args.user]["dev"])
    test_ids = set(users[args.user]["test"])
    print(f"[user] {args.user}: {len(train_ids)} train records, "
          f"{len(dev_ids)} dev, "
          f"{len(test_ids)} test", flush=True)

    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # records framing (Round 4)
    # =========================================================================
    if args.framing == "records":
        # Plan §"Why this exists" + decision #5: disjoint splits asserted up-front.
        if train_ids & dev_ids or train_ids & test_ids:
            sys.exit(
                f"ERROR: train/dev/test IDs not disjoint for {args.user}: "
                f"train∩dev={len(train_ids & dev_ids)}, "
                f"train∩test={len(train_ids & test_ids)}"
            )
        # Load outputs (~1.5 MB for LaMP_4 → safe to read whole).
        out_path_in = TIME_SPLIT_DIR / args.task / "train_outputs.json"
        print(f"[load] {out_path_in}", flush=True)
        out_doc = json.loads(out_path_in.read_text())
        outputs_by_id = {str(g["id"]): g["output"] for g in out_doc.get("golds", [])}
        print(f"[load] {len(outputs_by_id)} train outputs loaded", flush=True)

        # Stream the 863 MB train_questions file; early-exit after finding all 241.
        print(f"[scan] streaming {TIME_SPLIT_DIR / args.task / 'train_questions.json'} "
              f"for {len(train_ids)} train records ...", flush=True)
        records = find_train_records(args.task, train_ids)

        with out_path.open("w") as f:
            stats = emit_record_bm25(records, outputs_by_id, args.task,
                                     args.bm25_k, f)
        n_written = stats["n_written"]
        n_skipped_empty = stats["n_skipped_empty"]
        print(f"[write] {n_written} examples "
              f"({n_skipped_empty} skipped empty input/gold) -> {out_path}",
              flush=True)

        meta = {
            "schema_version": 1,
            "task": args.task,
            "user_fingerprint": args.user,
            "framing": "records_bm25",
            "bm25_k": args.bm25_k,
            "n_user_train_records": len(train_ids),
            "n_user_dev_records": len(dev_ids),
            "n_user_test_records": len(test_ids),
            "n_train_records_resolved": stats["n_train_records_resolved"],
            "profile_size": stats["profile_size"],
            "profile_stable": stats["profile_stable"],
            "profile_size_min": stats["profile_size_min"],
            "profile_size_max": stats["profile_size_max"],
            "n_examples": n_written,
            "n_skipped_empty": n_skipped_empty,
            "output_jsonl": str(out_path),
            "user_records_json": str(records_json),
            "outputs_json": str(out_path_in),
            "command": "python " + " ".join(sys.argv),
            **provenance,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[meta] -> {meta_path}", flush=True)
        return

    # =========================================================================
    # profile framing (Rounds 1 + 2) and unsupervised framing (R11/R12)
    # =========================================================================
    # --- Find the user's latest (largest-profile) snapshot record ----------
    # Prefers train; falls back to test/dev for the single-split tasks
    # (LaMP_1, LaMP_2_movies, LaMP_5) — see the module docstring.
    snapshot, snapshot_split = resolve_snapshot(
        args.task, train_ids, test_ids, dev_ids
    )
    snapshot_id = str(snapshot["id"])
    # Only guard against self-inclusion when the snapshot came from the very
    # record we will later evaluate on.
    exclude_id = snapshot_id if snapshot_split == "test" else ""

    profile = snapshot.get("profile", [])

    # --- Unsupervised (OPPU right-shifted history): raw text, no pairs -----
    if args.framing == "unsupervised":
        with out_path.open("w") as f:
            stats = emit_unsupervised(profile, args.task, f, exclude_id=exclude_id)
        print(
            f"[write] {stats['n_written']} raw-text examples "
            f"({stats['n_skipped_empty']} skipped empty, "
            f"{stats['n_dropped_self_record']} dropped self-record) -> {out_path}",
            flush=True,
        )
        meta = {
            "schema_version": 1,
            "task": args.task,
            "user_fingerprint": args.user,
            "framing": "unsupervised_clm",
            "target_field": None,
            "n_user_train_records": len(train_ids),
            "n_user_dev_records": len(dev_ids),
            "n_user_test_records": len(test_ids),
            "snapshot_record_id": snapshot_id,
            "snapshot_source_split": snapshot_split,
            "snapshot_profile_size": len(profile),
            "n_examples": stats["n_written"],
            "n_skipped_empty": stats["n_skipped_empty"],
            "n_dropped_self_record": stats["n_dropped_self_record"],
            "output_jsonl": str(out_path),
            "user_records_json": str(records_json),
            "command": "python " + " ".join(sys.argv),
            **provenance,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[meta] -> {meta_path}", flush=True)
        return

    # --- Emit one JSONL line per profile entry -----------------------------
    input_field, target_field = PROFILE_FRAMING[args.task]
    n_dropped_self_record = 0
    if args.bm25_k > 0:
        with out_path.open("w") as f:
            stats = emit_bm25_records(profile, args.task, args.bm25_k, f,
                                      exclude_id=exclude_id)
        n_dropped_self_record = stats["n_dropped_self_record"]
        n_written = stats["n_written"]
        n_skipped_empty = stats["n_skipped_empty"]
        n_dropped_no_date = stats["n_dropped_no_date"]
        n_examples_with_empty_pool = stats["n_examples_with_empty_pool"]
        framing = "profile_entries_bm25"
        print(
            f"[write] {n_written} examples ({n_skipped_empty} skipped empty, "
            f"{n_dropped_no_date} dropped no-date, "
            f"{n_examples_with_empty_pool} with empty pool) -> {out_path}",
            flush=True,
        )
    else:
        n_written = 0
        n_skipped_empty = 0
        n_dropped_no_date = None
        n_examples_with_empty_pool = None
        framing = "profile_entries_bare"
        with out_path.open("w") as f:
            for entry in profile:
                if exclude_id and str(entry.get("id", "")) == exclude_id:
                    n_dropped_self_record += 1
                    continue
                user_text = wrap_user_text(args.task, entry)
                gold = str(entry.get(target_field, "")).strip()
                # Source-field test, same rationale as emit_bm25_records.
                if not str(entry.get(input_field, "")).strip() or not gold:
                    n_skipped_empty += 1
                    continue
                rec = {
                    "task": args.task,
                    "id": str(entry.get("id", "")),
                    "system": "",
                    "user": user_text,
                    "assistant": gold,
                }
                f.write(json.dumps(rec) + "\n")
                n_written += 1
        print(f"[write] {n_written} examples ({n_skipped_empty} skipped for empty "
              f"input/target) -> {out_path}", flush=True)

    # --- Date span of the snapshot's profile (informational) ---------------
    dates = [str(e.get("date", "")) for e in profile if e.get("date")]
    dates = [d for d in dates if d]
    date_lo = min(dates) if dates else None
    date_hi = max(dates) if dates else None

    meta = {
        "schema_version": 1,
        "task": args.task,
        "user_fingerprint": args.user,
        "framing": framing,
        "input_field": input_field,
        "target_field": target_field,
        "n_user_train_records": len(train_ids),
        "n_user_dev_records": len(users[args.user]["dev"]),
        "n_user_test_records": len(users[args.user]["test"]),
        "snapshot_record_id": snapshot_id,
        "snapshot_source_split": snapshot_split,
        "snapshot_profile_size": len(profile),
        "snapshot_profile_date_min": date_lo,
        "snapshot_profile_date_max": date_hi,
        "n_examples": n_written,
        "n_skipped_empty": n_skipped_empty,
        "n_dropped_self_record": n_dropped_self_record,
        "output_jsonl": str(out_path),
        "user_records_json": str(records_json),
        "command": "python " + " ".join(sys.argv),
        **provenance,
    }
    if args.bm25_k > 0:
        meta["bm25_k"] = args.bm25_k
        meta["n_dropped_no_date"] = n_dropped_no_date
        meta["n_examples_with_empty_pool"] = n_examples_with_empty_pool
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[meta] -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
