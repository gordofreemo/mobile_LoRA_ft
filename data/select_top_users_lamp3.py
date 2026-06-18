#!/usr/bin/env python3
"""
Select the top-K LaMP-3 users by profile size for the Round 5 User-LoRA
multi-user replication of OPPU.

Per `experiments/2026-06-18-user-lora-round5-lamp3-plan.md` Step 3:

  - Eligible filter: `n_test >= 1` AND `seen_by_a1_lamp == 0`
    (~1,817 LaMP-3 users qualify; pool size verified at run time).
  - Rank: descending by profile size (read from the user's test record's
    `profile` field in `data/lamp_time/LaMP_3/test_questions.json`).
  - Output: `data/lamp_user_stats/LaMP_3_topK_users.json` (K=100 default).

The 414 MB `test_questions.json` is streamed via `stream_json_array` — kept
duplicated from `train/build_user_dataset.py` so this script remains
standalone in a Condor sandbox (the cardinal Condor-sandbox rule for the
project: scripts don't import each other across `data/` / `train/` / `eval/`
boundaries; see [[feedback-large-files]]).

For multi-test-record users (rare in LaMP-3) the `profile_size` reported is
the max over the user's test records; the recorded `test_record_id` is the
one that produced that max. This avoids ambiguity downstream — Step 8 will
eval each user on the test record listed here.

Usage (CPU-only, ~tens of seconds):
    python data/select_top_users_lamp3.py
    python data/select_top_users_lamp3.py --k 100 --overwrite

Output schema:
    {
      "schema_version": 1,
      "k": 100,
      "selection_criterion": "top_by_profile_size, eligible = "
                             "(seen_by_a1_lamp==0 AND n_test>=1)",
      "eligible_pool_size": <int>,
      "min_profile_size_in_top_k": <int>,
      "users": [
        {"user_fingerprint": "u...", "profile_size": <int>,
         "test_record_id": "<id>"},
        ...
      ],
      "provenance": {...}
    }
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
TIME_SPLIT_DIR = PROJECT_ROOT / "data" / "lamp_time"

DEFAULT_CSV = USER_STATS_DIR / "LaMP_3_users.csv"
DEFAULT_RECORDS_JSON = USER_STATS_DIR / "LaMP_3_user_records.json"
DEFAULT_TEST_QUESTIONS = TIME_SPLIT_DIR / "LaMP_3" / "test_questions.json"

MAX_PARSER_BUF_BYTES = 64 * 1024 * 1024


def stream_json_array(path):
    """Yield each top-level object from a JSON-array file without loading the
    whole array. Duplicated from `train/build_user_dataset.py` /
    `train/build_dataset.py` so this script remains standalone in a Condor
    sandbox. CRITICAL not to change behavior without updating the copies."""
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
    parser.add_argument("--k", type=int, default=100,
                        help="how many users to select (default 100)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="path to LaMP_3_users.csv")
    parser.add_argument("--records-json", type=Path, default=DEFAULT_RECORDS_JSON,
                        help="path to LaMP_3_user_records.json")
    parser.add_argument("--test-questions", type=Path,
                        default=DEFAULT_TEST_QUESTIONS,
                        help="path to time-split test_questions.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (default: "
                             "data/lamp_user_stats/LaMP_3_top<k>_users.json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing output (default: refuse)")
    args = parser.parse_args()

    if args.k <= 0:
        sys.exit("ERROR: --k must be > 0")

    out_path = args.out or USER_STATS_DIR / f"LaMP_3_top{args.k}_users.json"
    if out_path.exists() and not args.overwrite:
        sys.exit(
            f"ERROR: refusing to overwrite {out_path}. Pass --overwrite to replace."
        )

    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] select_top_users_lamp3 k={args.k} commit={commit_short} "
        f"dirty={provenance.get('git_dirty')} "
        f"host={provenance.get('hostname')} "
        f"cluster.proc={provenance.get('condor_cluster_id')}."
        f"{provenance.get('condor_proc_id')}",
        flush=True,
    )

    # --- Step 1: load eligibility from the user-stats CSV ----------------
    if not args.csv.exists():
        sys.exit(f"ERROR: missing {args.csv}; run data/lamp_user_stats.py first.")
    eligible_fps = set()
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["n_test"]) >= 1 and int(row["seen_by_a1_lamp"]) == 0:
                eligible_fps.add(row["user_fingerprint"])
    print(f"[csv] {len(eligible_fps)} eligible users "
          f"(n_test>=1 AND seen_by_a1_lamp==0) out of {reader.line_num - 1} total",
          flush=True)

    # --- Step 2: load the records JSON, build test_record_id -> fp map ---
    if not args.records_json.exists():
        sys.exit(f"ERROR: missing {args.records_json}; run data/lamp_user_stats.py first.")
    records = json.loads(args.records_json.read_text())
    test_id_to_fp = {}
    n_test_ids_total = 0
    for fp in eligible_fps:
        if fp not in records:
            sys.exit(
                f"ERROR: fingerprint {fp} is eligible per CSV but not in "
                f"{args.records_json} — CSV and records JSON disagree."
            )
        for rid in records[fp]["test"]:
            rid_str = str(rid)
            if rid_str in test_id_to_fp:
                sys.exit(
                    f"ERROR: test record id {rid_str} maps to multiple fingerprints "
                    f"({test_id_to_fp[rid_str]} and {fp}); records JSON invariant broken."
                )
            test_id_to_fp[rid_str] = fp
            n_test_ids_total += 1
    print(f"[records] reverse map: {n_test_ids_total} test_record_id -> fingerprint "
          f"entries (avg {n_test_ids_total / max(len(eligible_fps), 1):.2f} per eligible user)",
          flush=True)

    # --- Step 3: stream test_questions.json, capture profile sizes -------
    if not args.test_questions.exists():
        sys.exit(f"ERROR: missing {args.test_questions}.")
    fp_to_best = {}  # fp -> (profile_size, test_record_id)
    n_scanned = 0
    n_matched = 0
    t0 = time.time()
    print(f"[stream] reading {args.test_questions} ...", flush=True)
    for rec in stream_json_array(str(args.test_questions)):
        n_scanned += 1
        rid = str(rec.get("id", ""))
        fp = test_id_to_fp.get(rid)
        if fp is None:
            continue
        psize = len(rec.get("profile", []) or [])
        cur = fp_to_best.get(fp)
        if cur is None or psize > cur[0]:
            fp_to_best[fp] = (psize, rid)
        n_matched += 1
        if n_matched % 200 == 0:
            print(f"  matched {n_matched}/{n_test_ids_total} test records "
                  f"({n_scanned} scanned, {time.time()-t0:.0f}s elapsed)",
                  flush=True)
        if n_matched == n_test_ids_total:
            break  # early-exit: we've found every record we need
    elapsed = time.time() - t0
    print(f"[stream] done: {n_scanned} records scanned, "
          f"{n_matched}/{n_test_ids_total} matches in {elapsed:.0f}s",
          flush=True)

    if n_matched != n_test_ids_total:
        missing_fps = [fp for fp in eligible_fps if fp not in fp_to_best]
        sys.exit(
            f"ERROR: scanned the whole stream but only matched {n_matched}/{n_test_ids_total} "
            f"test record IDs. {len(missing_fps)} eligible users have no test record found "
            f"(e.g. {missing_fps[:5]}). records JSON and test_questions disagree."
        )

    # --- Step 4: sort by profile size desc, take top K -------------------
    ranked = sorted(
        ((fp, psize, rid) for fp, (psize, rid) in fp_to_best.items()),
        key=lambda x: (-x[1], x[0]),  # stable tiebreak by fp asc
    )
    top = ranked[: args.k]
    if len(top) < args.k:
        sys.exit(
            f"ERROR: only {len(top)} eligible users found, asked for K={args.k}."
        )
    min_profile_size = top[-1][1]
    max_profile_size = top[0][1]
    print(f"[rank] top-{args.k} profile sizes: "
          f"max={max_profile_size}, min={min_profile_size}, "
          f"floor users dropped: {len(fp_to_best) - args.k}", flush=True)

    # --- Step 5: write output --------------------------------------------
    payload = {
        "schema_version": 1,
        "k": args.k,
        "selection_criterion": (
            "top_by_profile_size, eligible = "
            "(seen_by_a1_lamp==0 AND n_test>=1)"
        ),
        "eligible_pool_size": len(eligible_fps),
        "min_profile_size_in_top_k": min_profile_size,
        "max_profile_size_in_top_k": max_profile_size,
        "n_test_records_scanned": n_scanned,
        "users": [
            {"user_fingerprint": fp, "profile_size": psize,
             "test_record_id": rid}
            for fp, psize, rid in top
        ],
        "inputs": {
            "csv": str(args.csv),
            "records_json": str(args.records_json),
            "test_questions": str(args.test_questions),
        },
        "command": "python " + " ".join(sys.argv),
        "provenance": provenance,
    }
    USER_STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[write] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
