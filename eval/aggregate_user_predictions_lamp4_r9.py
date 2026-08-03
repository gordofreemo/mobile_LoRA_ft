#!/usr/bin/env python3
"""
R9 — consolidate the 200 per-user prediction files from
condor/eval_lamp_user_lamp4_r9.sub into two condition-level JSONL files
(C2-R9 and C3-R9) for the per-user-mean paired comparison.

Clone of eval/aggregate_user_predictions_lamp4_ptl.py (PT2) with the base tag
swapped from Per-Task-LoRA(LaMP-4) to One-LoRA FT and the per-user adapter
suffix swapped from `_ptl` to `_r9`.

Reads `data/lamp_user_stats/LaMP_4_top100_users.json` to know which users are
in the pool (R6's exact K=100 pool, reused unchanged). For each user, locates
their C2-R9 and C3-R9 predictions JSONL files in `results/` by their canonical
eval_lamp.py naming, asserts each has exactly `n_test` lines for that user
(from `data/lamp_user_stats/LaMP_4_user_records.json`), and writes:

    results/LaMP_4_test_r9_C2.predictions.jsonl  (sum(n_test) lines)
    results/LaMP_4_test_r9_C3.predictions.jsonl  (sum(n_test) lines)

Each output line: `{"id": ..., "pred": ..., "gold": ..., "user_fingerprint": ...}`.
The `user_fingerprint` field is metadata for traceability; paired_compare_per_user.py
uses it to group records into per-user means before the paired-t.

Final leakage check: asserts gold byte-match between C2-R9 and C3-R9 for every
record_id.

No accuracy/exact-match descriptive here — LaMP-4 is headline generation scored
by ROUGE-1, not exact match, same as R6/PT2's aggregators.

Usage (CPU, sub-second):
    python eval/aggregate_user_predictions_lamp4_r9.py
    python eval/aggregate_user_predictions_lamp4_r9.py --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"

BASE_TAG = "a2_lamp_1ep_seed0_final"


def c2_pred_path(fp: str) -> Path:
    return RESULTS_DIR / f"LaMP_4_test_{BASE_TAG}_bm25k4_seed0_user{fp}.predictions.jsonl"


def c3_pred_path(fp: str) -> Path:
    return (
        RESULTS_DIR
        / f"LaMP_4_test_{BASE_TAG}_user_lora_lamp4_{fp}_r9_seed0_final_bm25k4_seed0_user{fp}.predictions.jsonl"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-users",
        type=Path,
        default=USER_STATS_DIR / "LaMP_4_top100_users.json",
    )
    parser.add_argument(
        "--user-records",
        type=Path,
        default=USER_STATS_DIR / "LaMP_4_user_records.json",
    )
    parser.add_argument(
        "--out-c2",
        type=Path,
        default=RESULTS_DIR / "LaMP_4_test_r9_C2.predictions.jsonl",
    )
    parser.add_argument(
        "--out-c3",
        type=Path,
        default=RESULTS_DIR / "LaMP_4_test_r9_C3.predictions.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing consolidated files (default: refuse)",
    )
    args = parser.parse_args()

    if not args.top_users.exists():
        sys.exit(f"ERROR: missing {args.top_users}.")
    if not args.user_records.exists():
        sys.exit(f"ERROR: missing {args.user_records}.")

    existing = [p for p in (args.out_c2, args.out_c3) if p.exists()]
    if existing and not args.overwrite:
        print("ERROR: refusing to overwrite existing consolidated files:",
              file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        print("Pass --overwrite to replace.", file=sys.stderr)
        sys.exit(1)

    top = json.loads(args.top_users.read_text())
    fps = [u["user_fingerprint"] for u in top["users"]]
    user_records = json.loads(args.user_records.read_text())
    print(f"[agg] {len(fps)} users from {args.top_users}", flush=True)

    # --- Pre-flight: every file present + n_lines == expected n_test --------
    missing = []
    count_violators = []
    c2_lines = []
    c3_lines = []
    expected_total = 0
    for fp in fps:
        if fp not in user_records:
            sys.exit(f"ERROR: user {fp} not in {args.user_records}.")
        expected_n_test = len(user_records[fp]["test"])
        expected_total += expected_n_test

        for cell, path_fn, lines_acc in (
            ("C2", c2_pred_path, c2_lines),
            ("C3", c3_pred_path, c3_lines),
        ):
            p = path_fn(fp)
            if not p.exists():
                missing.append((fp, cell, str(p)))
                continue
            recs = [json.loads(l) for l in p.open() if l.strip()]
            if len(recs) != expected_n_test:
                count_violators.append((fp, cell, expected_n_test, len(recs)))
                continue
            for r in recs:
                lines_acc.append({
                    "id": r["id"],
                    "pred": r["pred"],
                    "gold": r["gold"],
                    "user_fingerprint": fp,
                })

    if missing:
        for m in missing[:5]:
            print(f"  MISSING: {m}", file=sys.stderr)
        sys.exit(f"ERROR: {len(missing)} prediction files missing.")
    if count_violators:
        for v in count_violators[:5]:
            print(f"  LINE-COUNT-MISMATCH (user, cell, expected, actual): {v}",
                  file=sys.stderr)
        sys.exit(f"ERROR: {len(count_violators)} prediction files with "
                 f"n_lines != expected n_test.")

    assert len(c2_lines) == expected_total, (len(c2_lines), expected_total)
    assert len(c3_lines) == expected_total, (len(c3_lines), expected_total)

    # --- Final leakage check: id paired + gold byte-match --------------------
    c2_by_id = {r["id"]: r for r in c2_lines}
    c3_by_id = {r["id"]: r for r in c3_lines}
    c2_ids = set(c2_by_id)
    c3_ids = set(c3_by_id)
    if c2_ids != c3_ids:
        only_c2 = c2_ids - c3_ids
        only_c3 = c3_ids - c2_ids
        sys.exit(
            f"ERROR: id set mismatch between C2-R9 and C3-R9. "
            f"only_C2={sorted(only_c2)[:5]}, only_C3={sorted(only_c3)[:5]}"
        )
    if len(c2_ids) != expected_total:
        sys.exit(f"ERROR: unique id count {len(c2_ids)} != {expected_total} "
                 f"— duplicate ids.")
    gold_diffs = [
        (rid, c2_by_id[rid]["gold"], c3_by_id[rid]["gold"])
        for rid in c2_ids
        if c2_by_id[rid]["gold"] != c3_by_id[rid]["gold"]
    ]
    if gold_diffs:
        for d in gold_diffs[:5]:
            print(f"  GOLD DRIFT: id={d[0]}, C2-R9_gold={d[1]!r}, C3-R9_gold={d[2]!r}",
                  file=sys.stderr)
        sys.exit(
            f"ERROR: {len(gold_diffs)} record(s) have gold drift between "
            f"C2-R9 and C3-R9. Data corruption — STOP."
        )

    # --- Write consolidated JSONLs ------------------------------------------
    args.out_c2.parent.mkdir(parents=True, exist_ok=True)
    with args.out_c2.open("w") as f:
        for r in c2_lines:
            f.write(json.dumps(r) + "\n")
    with args.out_c3.open("w") as f:
        for r in c3_lines:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {args.out_c2}  ({len(c2_lines)} lines)", flush=True)
    print(f"[write] {args.out_c3}  ({len(c3_lines)} lines)", flush=True)


if __name__ == "__main__":
    main()
