#!/usr/bin/env python3
"""
Select the top-K users for the five LaMP tasks that have no User-LoRA round
yet: LaMP-2-news (R10/PT3), LaMP-1 (R11/PT4), LaMP-7 (R12/PT5),
LaMP-2-movies (R13/PT6), LaMP-5 (R14/PT7).

Generalizes `data/select_top_users_lamp4.py` (R6, LaMP-4) — same output
schema, same streaming reader, same "rank by profile size" rule — with the
eligibility predicate and K made per-task, because the five tasks split into
two very different shapes:

  LaMP_2_news   K=27,  eligible = unseen AND n_train>=4 AND n_dev>=4.
      The only one of the five with real per-user train-record volume (top
      user: 211 train / 36 dev / 35 test). K=27 IS the entire qualifying
      pool, not a cut — and the qualifying set's actual minimum n_train is
      18, since the distribution has a real gap between 4 and 17, so the
      loose-looking threshold admits no thin users. Users hold 3-35 test
      records each, so downstream eval must use the grouped per-user pattern
      (R6/PT2), never the single-job --user-records-from-file shortcut.

  LaMP_1 / LaMP_2_movies / LaMP_5 / LaMP_7   K=100, eligible = unseen AND
      n_test>=1. Every eligible user holds EXACTLY 1 test record (verified:
      1500 / 1557 / 1500 / 331 users respectively, all n_test==1), so eval
      uses the flat record-level pattern (LaMP-3/PT1). Their per-user
      training volume lives in profile entries, not records — which is why
      ranking by profile size is the right ordering here.

Note on the three single-split tasks (LaMP_1, LaMP_2_movies, LaMP_5): ZERO of
their users hold both a train record and a test record, so `--user-records`
eligibility and the training-corpus snapshot both key off the TEST record.
`train/build_user_dataset.py` handles that with its documented test-record
profile fallback; nothing extra is needed here.

Usage (CPU-only, streams one test_questions.json per task):
    python data/select_top_users_newtasks.py --task LaMP_2_news
    python data/select_top_users_newtasks.py --task LaMP_5 --overwrite

Output:
    data/lamp_user_stats/<task>_top<K>_users.json   (schema identical to
    LaMP_4_top100_users.json, so every downstream consumer works unchanged)
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
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/home/ange00008/projects/mobileFT_distill")
)
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"
TIME_SPLIT_DIR = Path(
    os.environ.get("LAMP_DIR", str(PROJECT_ROOT / "data" / "lamp_time"))
)

# Per-task: (default_K, eligibility predicate over a users.csv row).
# `seen_by_a1_lamp == 0` means "not seen by the Task-LoRA that trained on this
# task" — for these five tasks that is One-LoRA FT, NOT A1-lamp (A1-lamp never
# trained on any of them). The column name is a historical artifact of
# data/lamp_user_stats.py; see the note in its 2026-07-20 docstring.
TASK_POOLS = {
    "LaMP_2_news": (
        27,
        lambda r: int(r["n_train"]) >= 4 and int(r["n_dev"]) >= 4,
        "unseen AND n_train>=4 AND n_dev>=4",
    ),
    "LaMP_1": (100, lambda r: int(r["n_test"]) >= 1, "unseen AND n_test>=1"),
    "LaMP_2_movies": (100, lambda r: int(r["n_test"]) >= 1, "unseen AND n_test>=1"),
    "LaMP_5": (100, lambda r: int(r["n_test"]) >= 1, "unseen AND n_test>=1"),
    "LaMP_7": (100, lambda r: int(r["n_test"]) >= 1, "unseen AND n_test>=1"),
}

MAX_PARSER_BUF_BYTES = 64 * 1024 * 1024


def stream_json_array(path):
    """Yield each top-level object from a JSON-array file without loading the
    whole array. Duplicated from `data/select_top_users_lamp4.py` /
    `train/build_user_dataset.py` so this script remains standalone in a
    Condor sandbox. CRITICAL not to change behavior without updating copies."""
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=sorted(TASK_POOLS))
    parser.add_argument("--k", type=int, default=0,
                        help="how many users to select (default: the task's "
                             "pinned K from TASK_POOLS)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing output (default: refuse)")
    args = parser.parse_args()

    default_k, is_eligible, criterion_text = TASK_POOLS[args.task]
    k = args.k if args.k > 0 else default_k

    csv_path = USER_STATS_DIR / f"{args.task}_users.csv"
    records_json = USER_STATS_DIR / f"{args.task}_user_records.json"
    test_questions = TIME_SPLIT_DIR / args.task / "test_questions.json"

    out_path = args.out or USER_STATS_DIR / f"{args.task}_top{k}_users.json"
    if out_path.exists() and not args.overwrite:
        sys.exit(
            f"ERROR: refusing to overwrite {out_path}. Pass --overwrite to replace."
        )

    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] select_top_users_newtasks task={args.task} k={k} "
        f"commit={commit_short} dirty={provenance.get('git_dirty')} "
        f"host={provenance.get('hostname')} "
        f"cluster.proc={provenance.get('condor_cluster_id')}."
        f"{provenance.get('condor_proc_id')}",
        flush=True,
    )

    # --- Step 1: eligibility from the user-stats CSV ----------------------
    if not csv_path.exists():
        sys.exit(f"ERROR: missing {csv_path}; run data/lamp_user_stats.py first.")
    eligible_fps = set()
    n_rows = 0
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            n_rows += 1
            if int(row["seen_by_a1_lamp"]) != 0:
                continue
            if int(row["n_test"]) < 1:
                # Every round evaluates on test records; a user without one
                # can't contribute a paired observation no matter what else
                # they have.
                continue
            if is_eligible(row):
                eligible_fps.add(row["user_fingerprint"])
    print(f"[csv] {len(eligible_fps)} eligible users ({criterion_text}) "
          f"out of {n_rows} total", flush=True)
    if len(eligible_fps) < k:
        sys.exit(
            f"ERROR: only {len(eligible_fps)} eligible users, asked for K={k}. "
            f"Lower --k or revisit the eligibility rule."
        )

    # --- Step 2: reverse map test_record_id -> fingerprint ----------------
    if not records_json.exists():
        sys.exit(f"ERROR: missing {records_json}; run data/lamp_user_stats.py first.")
    records = json.loads(records_json.read_text())
    test_id_to_fp = {}
    n_test_ids_total = 0
    for fp in eligible_fps:
        if fp not in records:
            sys.exit(
                f"ERROR: fingerprint {fp} is eligible per CSV but not in "
                f"{records_json} — CSV and records JSON disagree."
            )
        for rid in records[fp]["test"]:
            rid_str = str(rid)
            if rid_str in test_id_to_fp:
                sys.exit(
                    f"ERROR: test record id {rid_str} maps to multiple "
                    f"fingerprints ({test_id_to_fp[rid_str]} and {fp}); "
                    f"records JSON invariant broken."
                )
            test_id_to_fp[rid_str] = fp
            n_test_ids_total += 1
    print(f"[records] reverse map: {n_test_ids_total} test_record_id -> "
          f"fingerprint entries (avg "
          f"{n_test_ids_total / max(len(eligible_fps), 1):.2f} per eligible user)",
          flush=True)

    # --- Step 3: stream test_questions.json, capture profile sizes --------
    if not test_questions.exists():
        sys.exit(f"ERROR: missing {test_questions}.")
    fp_to_best = {}  # fp -> (profile_size, test_record_id)
    fp_to_n_test = {}
    n_scanned = n_matched = 0
    t0 = time.time()
    print(f"[stream] reading {test_questions} ...", flush=True)
    for rec in stream_json_array(str(test_questions)):
        n_scanned += 1
        rid = str(rec.get("id", ""))
        fp = test_id_to_fp.get(rid)
        if fp is None:
            continue
        psize = len(rec.get("profile", []) or [])
        cur = fp_to_best.get(fp)
        if cur is None or psize > cur[0]:
            fp_to_best[fp] = (psize, rid)
        fp_to_n_test[fp] = fp_to_n_test.get(fp, 0) + 1
        n_matched += 1
        if n_matched % 200 == 0:
            print(f"  matched {n_matched}/{n_test_ids_total} test records "
                  f"({n_scanned} scanned, {time.time()-t0:.0f}s elapsed)",
                  flush=True)
        if n_matched == n_test_ids_total:
            break  # early-exit: found everything we need
    print(f"[stream] done: {n_scanned} records scanned, "
          f"{n_matched}/{n_test_ids_total} matches in {time.time()-t0:.0f}s",
          flush=True)

    if n_matched != n_test_ids_total:
        missing_fps = [fp for fp in eligible_fps if fp not in fp_to_best]
        sys.exit(
            f"ERROR: scanned the whole stream but only matched "
            f"{n_matched}/{n_test_ids_total} test record IDs. "
            f"{len(missing_fps)} eligible users have no test record found "
            f"(e.g. {missing_fps[:5]}). records JSON and test_questions disagree."
        )

    # --- Step 4: rank by profile size desc, take top K --------------------
    ranked = sorted(
        ((fp, psize, rid) for fp, (psize, rid) in fp_to_best.items()),
        key=lambda x: (-x[1], x[0]),  # stable tiebreak by fp asc
    )
    top = ranked[:k]
    min_profile_size = top[-1][1]
    max_profile_size = top[0][1]
    n_test_values = sorted({fp_to_n_test[fp] for fp, _, _ in top})
    print(f"[rank] top-{k} profile sizes: max={max_profile_size}, "
          f"min={min_profile_size}; users dropped below the cut: "
          f"{len(fp_to_best) - k}", flush=True)
    print(f"[rank] per-user test-record counts in the top-{k}: "
          f"{n_test_values[0]}..{n_test_values[-1]} "
          f"({'FLAT record-level eval' if n_test_values[-1] == 1 else 'GROUPED per-user eval required'})",
          flush=True)

    if min_profile_size == 0:
        sys.exit(
            f"ERROR: at least one selected user has an EMPTY profile — there "
            f"would be nothing to train their User-LoRA on. Tighten the "
            f"eligibility rule or lower K."
        )

    payload = {
        "schema_version": 1,
        "k": k,
        "task": args.task,
        "selection_criterion": f"top_by_profile_size, eligible = ({criterion_text})",
        "eligible_pool_size": len(eligible_fps),
        "min_profile_size_in_top_k": min_profile_size,
        "max_profile_size_in_top_k": max_profile_size,
        "max_test_records_per_user_in_top_k": n_test_values[-1],
        "eval_pattern": "flat_record_level" if n_test_values[-1] == 1 else "grouped_per_user",
        "n_test_records_scanned": n_scanned,
        "users": [
            {"user_fingerprint": fp, "profile_size": psize,
             "test_record_id": rid, "n_test_records": fp_to_n_test[fp]}
            for fp, psize, rid in top
        ],
        "inputs": {
            "csv": str(csv_path),
            "records_json": str(records_json),
            "test_questions": str(test_questions),
        },
        "command": "python " + " ".join(sys.argv),
        "provenance": provenance,
    }
    USER_STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[write] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
