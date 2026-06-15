#!/usr/bin/env python3
"""
Build a single-user training corpus for User-LoRA fine-tuning.

This is the per-user companion to `train/build_dataset.py`. For one user on
one LaMP task, the user's `profile` (a chronological list of the user's past
interactions) on their **latest** time-split train record is used directly as
the User-LoRA training corpus, with no BM25 retrieval and no system context —
the bare profile-entry framing per the pinned User-LoRA Round 1 design.

LaMP_4 profile entries are (article_body, user_headline) pairs, so this yields
one (input=article, target=headline) training example per profile entry. For
u00000011 specifically the latest train record's profile is ~1100 entries.

Per the pinned design (see experiments/2026-06-14-user-lora-round1-plan.md):
  - profile-entry framing, not record framing
  - no system message (bare)
  - no BM25 retrieval at training time
  - target = `title` (the user's headline) for LaMP_4

Output format matches `train/train.py`'s `build_example` (flat dict with
`system`/`user`/`assistant` strings), not the `{"messages": [...]}` form
sketched in the plan — train.py builds the HF message list itself. The plan
spec was a higher-level description of the content; this is the concrete
on-disk encoding that train.py already understands.

Output:
    data/lamp_user_train_<task>_<user>_bare.jsonl
    data/lamp_user_train_<task>_<user>_bare.meta.json

Usage (CPU-only):
    python train/build_user_dataset.py --task LaMP_4 --user u00000011
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
USER_RECORDS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"
TIME_SPLIT_DIR = PROJECT_ROOT / "data" / "lamp_time"
DATA_OUT_DIR = PROJECT_ROOT / "data"

# Per-task: (input_field, target_field) on profile entries. Only LaMP_4 has a
# usable (text, title) shape; LaMP_3 would need (text, score) and LaMP_7 has
# no profile-level target at all — they're not supported here.
PROFILE_FRAMING = {
    "LaMP_4": ("text", "title"),
}

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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=sorted(PROFILE_FRAMING))
    parser.add_argument("--user", required=True,
                        help="user fingerprint, e.g. u00000011")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing JSONL / meta (default: refuse)")
    args = parser.parse_args()

    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] build_user_dataset task={args.task} user={args.user} "
        f"commit={commit_short} dirty={provenance.get('git_dirty')} "
        f"host={provenance.get('hostname')}",
        flush=True,
    )

    out_path = DATA_OUT_DIR / f"lamp_user_train_{args.task}_{args.user}_bare.jsonl"
    meta_path = DATA_OUT_DIR / f"lamp_user_train_{args.task}_{args.user}_bare.meta.json"

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
    print(f"[user] {args.user}: {len(train_ids)} train records, "
          f"{len(users[args.user]['dev'])} dev, "
          f"{len(users[args.user]['test'])} test", flush=True)

    # --- Find the user's latest (largest-profile) train record -------------
    print(f"[scan] streaming {TIME_SPLIT_DIR / args.task / 'train_questions.json'} ...",
          flush=True)
    snapshot = find_latest_train_record(args.task, train_ids)

    # --- Emit one JSONL line per profile entry -----------------------------
    input_field, target_field = PROFILE_FRAMING[args.task]
    profile = snapshot.get("profile", [])
    n_written = 0
    n_skipped_empty = 0
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
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
        "framing": "profile_entries_bare",
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
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[meta] -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
