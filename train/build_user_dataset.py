#!/usr/bin/env python3
"""
Build a single-user training corpus for User-LoRA fine-tuning.

This is the per-user companion to `train/build_dataset.py`. Two
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

  --framing records  (Round 4 path, requires --bm25-k K>0):
      Uses the user's time-split train *records* — one (input, gold) pair per
      LaMP_4 train-period record (241 for u00000011, disjoint from 21 dev /
      25 test). For each record: user = record.input byte-for-byte (e.g.,
      "Generate a headline for the following article: ..."), assistant =
      train_outputs[id].output, system = SYSTEM_PREAMBLE + BM25 top-K over
      the snapshot profile (built once outside the per-record loop because
      this user's profile is identical across their 241 train records).
      Same shape as the eval-time task; the User-LoRA's training distribution
      matches the eval distribution.

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

Output format matches `train/train.py`'s `build_example` (flat dict with
`system`/`user`/`assistant` strings).

Output:
    data/lamp_user_train_<task>_<user>_bare.jsonl            (profile, --bm25-k 0)
    data/lamp_user_train_<task>_<user>_bm25k<K>.jsonl        (profile, --bm25-k K>0)
    data/lamp_user_train_<task>_<user>_records_bm25k<K>.jsonl (records, --bm25-k K>0)
    (with matching .meta.json sidecar)

Usage (CPU-only):
    python train/build_user_dataset.py --task LaMP_4 --user u00000011
    python train/build_user_dataset.py --task LaMP_4 --user u00000011 --bm25-k 4
    python train/build_user_dataset.py --task LaMP_4 --user u00000011 --framing records --bm25-k 4
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

# Per-task: (input_field, target_field) on profile entries. Only LaMP_4 has a
# usable (text, title) shape; LaMP_3 would need (text, score) and LaMP_7 has
# no profile-level target at all — they're not supported here.
PROFILE_FRAMING = {
    "LaMP_4": ("text", "title"),
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
    "LaMP_4": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- Article: "{trim(it.get("text", ""))}" — the user\'s headline: "{trim(it.get("title", ""), TITLE_CHARS)}"',
    },
}

SYSTEM_PREAMBLE = (
    "The following are examples of this user's past activity. "
    "Use them to match this user's preferences and writing style.\n\n"
)

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


def find_latest_train_record(task: str, train_ids: set) -> dict:
    """Stream the time-split train file; return the user's record with the
    largest profile (= latest in chronological time-split ordering). Holds
    only the current best record in memory."""
    q_path = TIME_SPLIT_DIR / task / "train_questions.json"
    best = None
    best_size = -1
    n_seen = 0
    t0 = time.time()
    for r in stream_json_array(str(q_path)):
        if str(r.get("id")) not in train_ids:
            continue
        n_seen += 1
        prof = r.get("profile", [])
        if len(prof) > best_size:
            best = r
            best_size = len(prof)
        if n_seen % 50 == 0:
            print(f"  scanned: {n_seen}/{len(train_ids)} train records found "
                  f"(best profile size so far {best_size})", flush=True)
    if best is None or n_seen != len(train_ids):
        raise RuntimeError(
            f"Found {n_seen} of {len(train_ids)} expected train records — "
            f"records JSON and time-split disagree."
        )
    print(f"  done: {n_seen} train records scanned in {time.time()-t0:.0f}s; "
          f"snapshot record_id={best['id']} profile_size={best_size}", flush=True)
    return best


def emit_bm25_records(profile: list, task: str, k: int, out_f) -> dict:
    """Variant-B emission: per profile entry, BM25-retrieve over strictly-prior
    entries, format with the task's lambda + SYSTEM_PREAMBLE, write JSONL.

    Pool rule (decision A1b + strict-prior tie-breaking from the Round-2 plan):
    for entry $e_j$ with date $d_j$, pool = $\\{e_i : i \\neq j$ AND $e_i$ has a
    date AND $d_i < d_j\\}$. Entries missing a `date` field are dropped from
    the training set (no JSONL row); under strict-prior they are also not
    retrievable into anyone's pool, so the dropped count is recorded but the
    profile is otherwise untouched.

    Returns a stats dict for the meta sidecar.
    """
    input_field, target_field = PROFILE_FRAMING[task]

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
        user_text = str(e_j.get(input_field, "")).strip()
        gold = str(e_j.get(target_field, "")).strip()
        if not user_text or not gold:
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
      - All records share the same profile (len + profile[0].id fingerprint),
        so a single BM25 index serves the per-record loop.

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
    for r in records[1:]:
        prof = r.get("profile", []) or []
        rfp = (
            len(prof),
            str(prof[0].get("id", "")) if prof else "",
        )
        if rfp != fp:
            raise RuntimeError(
                f"Profile fingerprint mismatch: record {r.get('id')} has "
                f"(len={rfp[0]}, profile[0].id={rfp[1]}) vs reference "
                f"(len={fp[0]}, profile[0].id={fp[1]}). Records-framing "
                f"assumes a stable per-user profile — re-open the design "
                f"if this user's profile drifts across records."
            )

    # --- build BM25 once over the shared profile --------------------------
    profile = first_profile
    profile_tokens = [tokenize(TASKS[task]["index_field"](e)) for e in profile]
    bm25 = BM25(profile_tokens)

    # --- per-record loop --------------------------------------------------
    n_written = 0
    n_skipped_empty = 0

    for r in records:
        rid = str(r.get("id"))
        user_text = str(r.get("input", "")).strip()
        gold = str(outputs_by_id.get(rid, "")).strip()
        if not user_text or not gold:
            n_skipped_empty += 1
            continue

        top = bm25.top_k(tokenize(user_text), k)
        retrieved = [profile[t] for t in top]
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
        "n_train_records_resolved": len(records),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=sorted(PROFILE_FRAMING))
    parser.add_argument("--user", required=True,
                        help="user fingerprint, e.g. u00000011")
    parser.add_argument("--framing", choices=["profile", "records"],
                        default="profile",
                        help="training-example unit: profile-entry pairs "
                             "(Rounds 1+2; default) or train-period record "
                             "(input, gold) pairs (Round 4, requires --bm25-k>0)")
    parser.add_argument("--bm25-k", type=int, default=0,
                        help="BM25 top-k retrieval over the user's profile "
                             "into the system slot at train time. "
                             "profile framing: 0 = bare (Round 1), K>0 = "
                             "strictly-prior pool (Round 2 / variant B). "
                             "records framing: K>0 required, full profile pool.")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing JSONL / meta (default: refuse)")
    args = parser.parse_args()
    if args.bm25_k < 0:
        sys.exit("ERROR: --bm25-k must be >= 0")
    if args.framing == "records" and args.bm25_k <= 0:
        sys.exit("ERROR: --framing records requires --bm25-k > 0 "
                 "(bare records framing is not a Round-4 axis; re-open the "
                 "design if you want it).")

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
    # profile framing (Rounds 1 + 2, unchanged)
    # =========================================================================
    # --- Find the user's latest (largest-profile) train record -------------
    print(f"[scan] streaming {TIME_SPLIT_DIR / args.task / 'train_questions.json'} ...",
          flush=True)
    snapshot = find_latest_train_record(args.task, train_ids)

    # --- Emit one JSONL line per profile entry -----------------------------
    input_field, target_field = PROFILE_FRAMING[args.task]
    profile = snapshot.get("profile", [])
    if args.bm25_k > 0:
        with out_path.open("w") as f:
            stats = emit_bm25_records(profile, args.task, args.bm25_k, f)
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
                user_text = str(entry.get(input_field, "")).strip()
                gold = str(entry.get(target_field, "")).strip()
                if not user_text or not gold:
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
        "snapshot_record_id": str(snapshot["id"]),
        "snapshot_profile_size": len(profile),
        "snapshot_profile_date_min": date_lo,
        "snapshot_profile_date_max": date_hi,
        "n_examples": n_written,
        "n_skipped_empty": n_skipped_empty,
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
