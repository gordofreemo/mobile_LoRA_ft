#!/usr/bin/env python3
"""
Per-user viability analysis + top-K selection for LongLaMP `_temporal` splits.

New scaffolding for the LL4-LL6 User-LoRA round
(experiments/2026-07-27-longlamp-user-lora-ll4-ll6-plan.md), mirroring
`data/lamp_user_stats.py` in *purpose* but simpler in *method*: LongLaMP
records carry an explicit user-id field per task (reviewerId / name /
author), so there is no need for LaMP's profile-text union-find clustering
-- users are already identified directly. This script folds LaMP's two-step
pipeline (lamp_user_stats.py for eligibility + a per-task select_top_users_*
script for top-K ranking) into one, since the plan names a single output
artifact per task.

Method (REVISED 2026-07-28, see the addendum in
experiments/2026-07-27-longlamp-user-lora-ll4-ll6-plan.md -- re-grilled via
/grill_me after the ORIGINAL "records" framing turned out to be non-viable:
real data showed n_train_records==1 for literally every eligible user on
Review and Abstract, not "several per user" as decision #8 assumed. The
fix, mirroring LaMP-2-movies/5's own record-count dead end: train on the
user's PROFILE ENTRIES (reframed as (input,output) pairs matching the
task's own shape) instead of on distinct temporal-train records.)
------
For each task:
  1. Load the `_user`-split TRAIN partition (data/longlamp/<task>_user/train.json)
     -- the corpus the task's Task-LoRA (LL1/LL2/LL3) was fine-tuned on --
     and collect the set of user ids appearing there ("seen").
  2. Load the `_temporal`-split TRAIN and TEST partitions
     (data/longlamp/<task>_temporal/{train,test}.json). For each record,
     record its file-order index (0-based) under its user id -- this index
     is exactly the synthetic "id" eval/eval_longlamp.py's `load_split`
     assigns (`str(i)` in file order), so indices computed here line up
     with eval-time filtering without any extra translation.
  3. `profile_size` per user = max(len(profile)) over that user's
     temporal-TRAIN records (NOT test -- profile-entry training draws from
     a TRAIN record's profile, so that's the number that determines
     training-corpus richness). `train_snapshot_idx` = the temporal-train
     index that produced that max -- this record's `profile` field is what
     train/build_longlamp_user_dataset.py trains on (a single snapshot, not
     pooled across the user's train records, mirroring LaMP's original
     profile-framing convention).
  4. Eligible = user NOT in the `_user`-split TRAIN id set (decision #7:
     "not seen by Task-LoRA training") AND has >=1 temporal-TEST record
     (something to evaluate) AND profile_size (from TRAIN, per #3) >= 2
     (bare minimum so leave-one-out BM25 -- decision #8's revised context
     construction -- always has >=1 retrievable entry; NOT "n_train>=1",
     which is now irrelevant to training-corpus size). Rank eligible users
     by profile_size descending, take top-K.

Outputs (per task, in data/longlamp_user_stats/):
    <tag>_users.csv            one row per discovered temporal-split user
                                (user_id, n_train, n_test, seen_by_task_lora,
                                profile_size, train_snapshot_idx)
    <tag>_user_records.json    {user_id: {"train": [idx,...], "test": [idx,...],
                                "train_snapshot_idx": int}}
                                -- consumed by train/build_longlamp_user_dataset.py
                                and eval/eval_longlamp.py's --user-records filter.
    <tag>_top100_users.json    top-K eligible users by profile_size, shape
                                mirroring LaMP_3_top100_users.json (see
                                docstring in data/select_top_users_lamp3.py).

Usage (CPU-only; NOTE: these files are NOT small -- confirmed at run time
2026-07-27 that `_temporal` train.json reaches 3.1 GB (abstract_generation)
and the `_user` train.json files are already 1.2-1.9 GB each -- same order
of magnitude as LaMP's giant per-task files, so this script streams rather
than json.loads()'ing whole arrays, matching data/lamp_user_stats.py's own
stream_array pattern (duplicated here so this script stays standalone in a
Condor sandbox)):
    python data/longlamp_user_stats.py
    python data/longlamp_user_stats.py --tasks product_review --k 100
"""

import argparse
import csv
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# 3.5GB, not 64/256MB -- topic_writing_user's train.json has at least one
# profile `content` field (a per-user aggregated text blob, plausibly one
# very prolific Reddit user's whole post history) that overflowed both the
# original 64MB cap and a 256MB retry (confirmed 2026-07-28 via a bounded
# `dd` byte-range sample, not a full-file load: the "unterminated string" was
# real text, not corrupted JSON -- json.loads() on the whole file succeeded
# previously during LL3's Task-LoRA corpus build). Set safely above the
# largest known file size across all 3 tasks (~3.1GB, abstract_generation's
# _temporal train.json) so no single record can ever exceed it -- still far
# lighter than a whole-file json.loads(), which needs several times the file
# size to hold every record simultaneously; this bounds peak growth to one
# record at a time.
MAX_PARSER_BUF_BYTES = 3584 * 1024 * 1024


def stream_json_array(path):
    """Yield each top-level object from a JSON-array file without loading the
    whole array. Duplicated from data/lamp_user_stats.py's stream_array /
    train/build_user_dataset.py's stream_json_array -- keep behavior in sync."""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        while "[" not in buf:
            chunk = f.read(65536)
            if not chunk:
                return
            buf += chunk
        buf = buf[buf.index("[") + 1:]
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

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    str(Path(__file__).parent.parent),
))
LONGLAMP_DIR = Path(os.environ.get("LONGLAMP_DIR", str(PROJECT_ROOT / "data" / "longlamp")))
OUT_DIR = Path(os.environ.get("LONGLAMP_USER_STATS_OUT_DIR",
                               str(PROJECT_ROOT / "data" / "longlamp_user_stats")))

# tag -> (user-split dir name, temporal-split dir name, id field). Tags match
# build_longlamp_dataset.py's FILE_TAGS (review/abstract/topic) so filenames
# stay consistent with the rest of the LongLaMP scaffolding.
TASKS = {
    "review": {
        "user_task": "product_review_user",
        "temporal_task": "product_review_temporal",
        "id_field": "reviewerId",
    },
    "abstract": {
        "user_task": "abstract_generation_user",
        "temporal_task": "abstract_generation_temporal",
        "id_field": "name",
    },
    "topic": {
        "user_task": "topic_writing_user",
        "temporal_task": "topic_writing_temporal",
        "id_field": "author",
    },
}


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


def _require_path(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run data/download_longlamp.py first "
                  f"(--task {path.parent.name}).")
    return path


def build_seen_ids(user_task: str, id_field: str) -> set:
    path = _require_path(LONGLAMP_DIR / user_task / "train.json")
    print(f"  streaming _user-split TRAIN ({path}) ...", flush=True)
    seen = set()
    n = 0
    for r in stream_json_array(str(path)):
        seen.add(str(r[id_field]))
        n += 1
    print(f"    -> {n} records, {len(seen):,} unique user ids", flush=True)
    return seen


def analyse_task(cfg: dict, seen_ids: set) -> dict:
    """Returns {user_id: {"train": [idx,...], "test": [idx,...],
    "profile_size": int, "train_snapshot_idx": int|None}}.

    profile_size / train_snapshot_idx are computed over TRAIN records only
    (not test) -- see the module docstring's 2026-07-28 revision."""
    temporal_dir = LONGLAMP_DIR / cfg["temporal_task"]
    id_field = cfg["id_field"]
    users = defaultdict(lambda: {
        "train": [], "test": [], "profile_size": 0, "train_snapshot_idx": None,
    })

    for split, fname in (("train", "train.json"), ("test", "test.json")):
        path = _require_path(temporal_dir / fname)
        print(f"  streaming _temporal-split {split} ({path}) ...", flush=True)
        n = 0
        for idx, r in enumerate(stream_json_array(str(path))):
            uid = str(r[id_field])
            users[uid][split].append(idx)
            if split == "train":
                psize = len(r.get("profile", []) or [])
                if psize > users[uid]["profile_size"]:
                    users[uid]["profile_size"] = psize
                    users[uid]["train_snapshot_idx"] = idx
            n += 1
        print(f"    -> {n} records", flush=True)

    for uid, u in users.items():
        u["seen_by_task_lora"] = int(uid in seen_ids)

    print(f"  -> {len(users):,} unique temporal-split users discovered", flush=True)
    return dict(users)


MIN_ELIGIBLE_PROFILE_SIZE = 2  # bare minimum for non-degenerate leave-one-out BM25


def is_eligible(u: dict) -> bool:
    return (not u["seen_by_task_lora"] and len(u["test"]) >= 1
            and u["profile_size"] >= MIN_ELIGIBLE_PROFILE_SIZE)


def write_csv(tag: str, users: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tag}_users.csv"
    fields = ["user_id", "n_train", "n_test", "seen_by_task_lora", "profile_size",
              "train_snapshot_idx"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for uid, u in sorted(
            users.items(),
            key=lambda kv: (kv[1]["seen_by_task_lora"], -kv[1]["profile_size"]),
        ):
            w.writerow({
                "user_id": uid,
                "n_train": len(u["train"]),
                "n_test": len(u["test"]),
                "seen_by_task_lora": u["seen_by_task_lora"],
                "profile_size": u["profile_size"],
                "train_snapshot_idx": u["train_snapshot_idx"],
            })
    print(f"  wrote {out} ({len(users)} rows)", flush=True)
    return out


def write_records_json(tag: str, users: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tag}_user_records.json"
    payload = {
        uid: {
            "train": u["train"], "test": u["test"],
            "train_snapshot_idx": u["train_snapshot_idx"],
        }
        for uid, u in users.items()
    }
    out.write_text(json.dumps(payload))
    print(f"  wrote {out} ({len(payload)} users)", flush=True)
    return out


def write_top_k(tag: str, users: dict, k: int, provenance: dict, command: str) -> Path:
    eligible = {uid: u for uid, u in users.items() if is_eligible(u)}
    ranked = sorted(eligible.items(), key=lambda kv: (-kv[1]["profile_size"], kv[0]))
    top = ranked[:k]
    if len(top) < k:
        print(f"  WARNING: only {len(top)} eligible users for {tag} "
              f"(asked for k={k}) -- surfacing shortfall, not silently under-filling.",
              file=sys.stderr)

    out = OUT_DIR / f"{tag}_top{k}_users.json"
    payload = {
        "schema_version": 2,
        "k": k,
        "selection_criterion": (
            "top_by_profile_size (max over user's temporal-TRAIN records' "
            "embedded profile length -- that record is the training snapshot, "
            "see train_snapshot_idx), eligible = (NOT seen by this task's "
            "Task-LoRA _user-split TRAIN partition) AND (n_temporal_test>=1) "
            f"AND (profile_size>={MIN_ELIGIBLE_PROFILE_SIZE}). Revised "
            "2026-07-28 -- see experiments/2026-07-27-longlamp-user-lora-"
            "ll4-ll6-plan.md addendum: the original n_train>=1 criterion "
            "assumed 'records' framing, since replaced by profile-entry "
            "reframing after real data showed n_train_records==1 for every "
            "eligible user."
        ),
        "eligible_pool_size": len(eligible),
        "n_selected": len(top),
        "min_profile_size_in_top_k": top[-1][1]["profile_size"] if top else None,
        "max_profile_size_in_top_k": top[0][1]["profile_size"] if top else None,
        "users": [
            {
                "user_id": uid,
                "profile_size": u["profile_size"],
                "train_snapshot_idx": u["train_snapshot_idx"],
                "test_record_ids": [str(i) for i in u["test"]],
            }
            for uid, u in top
        ],
        "command": command,
        "provenance": provenance,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out} ({len(top)}/{k} users, eligible pool {len(eligible)})",
          flush=True)
    return out


def summarise(tag: str, users: dict) -> None:
    seen = [u for u in users.values() if u["seen_by_task_lora"]]
    unseen = [u for u in users.values() if not u["seen_by_task_lora"]]
    print(f"\n==== {tag} per-user distribution ====")
    print(f"  total temporal-split users: {len(users):>7,}")
    print(f"    seen by Task-LoRA:       {len(seen):>7,} "
          f"({100 * len(seen) / max(1, len(users)):5.1f}%)")
    print(f"    NOT seen by Task-LoRA:   {len(unseen):>7,} "
          f"({100 * len(unseen) / max(1, len(users)):5.1f}%)")
    eligible = [u for u in users.values() if is_eligible(u)]
    print(f"    eligible (unseen, >=1 test, profile_size>={MIN_ELIGIBLE_PROFILE_SIZE} "
          f"from TRAIN): {len(eligible):>7,}")
    if eligible:
        xs = sorted(u["profile_size"] for u in eligible)
        n = len(xs)
        print(f"    profile_size (TRAIN-based) over eligible: min {xs[0]} "
              f"p50 {xs[n // 2]} p90 {xs[int(0.9 * n)]} max {xs[-1]}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        choices=list(TASKS.keys()))
    parser.add_argument("--k", type=int, default=100,
                        help="top-K users to select per task (default 100)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run even if outputs already exist")
    args = parser.parse_args()

    provenance = collect_provenance()
    command = "python " + " ".join(sys.argv)
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(f"[run] longlamp_user_stats tasks={args.tasks} k={args.k} "
          f"commit={commit_short} host={provenance.get('hostname')}", flush=True)

    for tag in args.tasks:
        cfg = TASKS[tag]
        out_csv = OUT_DIR / f"{tag}_users.csv"
        out_records = OUT_DIR / f"{tag}_user_records.json"
        out_top = OUT_DIR / f"{tag}_top{args.k}_users.json"
        if (out_csv.exists() and out_records.exists() and out_top.exists()
                and not args.overwrite):
            print(f"{tag}: outputs already exist, skipping (use --overwrite to redo)")
            continue
        print(f"\n############ {tag} ({cfg['temporal_task']}) ############", flush=True)
        t0 = time.time()
        seen_ids = build_seen_ids(cfg["user_task"], cfg["id_field"])
        users = analyse_task(cfg, seen_ids)
        del seen_ids
        write_csv(tag, users)
        write_records_json(tag, users)
        write_top_k(tag, users, args.k, provenance, command)
        summarise(tag, users)
        print(f"  ({tag} took {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
