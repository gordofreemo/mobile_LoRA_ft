#!/usr/bin/env python3
"""
Round 5 Step 9 — consolidate the 200 per-user prediction files from Step 8
into two condition-level JSONL files (C2 and C3) for the paired-compare gate.

Reads `data/lamp_user_stats/LaMP_3_top100_users.json` to know which users are
in the pool. For each user, locates their C2 and C3 predictions JSONL files
in `results/` by their canonical eval_lamp.py naming, asserts each has
exactly one line, and writes:

    results/LaMP_3_test_round5_C2.predictions.jsonl  (100 lines)
    results/LaMP_3_test_round5_C3.predictions.jsonl  (100 lines)

Each output line: `{"id": ..., "pred": ..., "gold": ..., "user_fingerprint": ...}`.
The `user_fingerprint` field is metadata for traceability; paired_compare.py
ignores it and joins on `id`.

Final leakage check: asserts gold byte-match between C2 and C3 for every
record_id. This is the same check Step 8's audit ran on raw files; running
it again on the consolidated files guards against aggregation bugs.

Plan reference: experiments/2026-06-18-user-lora-round5-lamp3-plan.md §Step 9.

Usage (CPU, sub-second):
    python eval/aggregate_user_predictions.py
    python eval/aggregate_user_predictions.py --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"

A1_TAG = "a1_lamp_1ep_seed0_checkpoint-1000"


def c2_pred_path(fp: str) -> Path:
    return RESULTS_DIR / f"LaMP_3_test_{A1_TAG}_bm25k4_seed0_user{fp}.predictions.jsonl"


def c3_pred_path(fp: str) -> Path:
    return (
        RESULTS_DIR
        / f"LaMP_3_test_{A1_TAG}_user_lora_lamp3_{fp}_seed0_final_bm25k4_seed0_user{fp}.predictions.jsonl"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-users",
        type=Path,
        default=USER_STATS_DIR / "LaMP_3_top100_users.json",
    )
    parser.add_argument(
        "--out-c2",
        type=Path,
        default=RESULTS_DIR / "LaMP_3_test_round5_C2.predictions.jsonl",
    )
    parser.add_argument(
        "--out-c3",
        type=Path,
        default=RESULTS_DIR / "LaMP_3_test_round5_C3.predictions.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing consolidated files (default: refuse)",
    )
    args = parser.parse_args()

    if not args.top_users.exists():
        sys.exit(f"ERROR: missing {args.top_users}. Run Step 3 first.")

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
    print(f"[agg] {len(fps)} users from {args.top_users}", flush=True)

    # --- Pre-flight: every file present + exactly 1 line --------------------
    missing = []
    one_line_violators = []
    c2_lines = []
    c3_lines = []
    for fp in fps:
        for cell, path_fn, lines_acc in (
            ("C2", c2_pred_path, c2_lines),
            ("C3", c3_pred_path, c3_lines),
        ):
            p = path_fn(fp)
            if not p.exists():
                missing.append((fp, cell, str(p)))
                continue
            recs = [json.loads(l) for l in p.open() if l.strip()]
            if len(recs) != 1:
                one_line_violators.append((fp, cell, len(recs)))
                continue
            r = recs[0]
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
    if one_line_violators:
        for v in one_line_violators[:5]:
            print(f"  LINE-COUNT-NEQ-1: {v}", file=sys.stderr)
        sys.exit(f"ERROR: {len(one_line_violators)} prediction files with != 1 line.")

    assert len(c2_lines) == 100, len(c2_lines)
    assert len(c3_lines) == 100, len(c3_lines)

    # --- Final leakage check: id paired + gold byte-match -------------------
    c2_by_id = {r["id"]: r for r in c2_lines}
    c3_by_id = {r["id"]: r for r in c3_lines}
    c2_ids = set(c2_by_id)
    c3_ids = set(c3_by_id)
    if c2_ids != c3_ids:
        only_c2 = c2_ids - c3_ids
        only_c3 = c3_ids - c2_ids
        sys.exit(
            f"ERROR: id set mismatch between C2 and C3. "
            f"only_C2={sorted(only_c2)[:5]}, only_C3={sorted(only_c3)[:5]}"
        )
    if len(c2_ids) != 100:
        sys.exit(f"ERROR: unique id count {len(c2_ids)} != 100 — duplicate ids.")
    gold_diffs = [
        (rid, c2_by_id[rid]["gold"], c3_by_id[rid]["gold"])
        for rid in c2_ids
        if c2_by_id[rid]["gold"] != c3_by_id[rid]["gold"]
    ]
    if gold_diffs:
        for d in gold_diffs[:5]:
            print(f"  GOLD DRIFT: id={d[0]}, C2_gold={d[1]!r}, C3_gold={d[2]!r}",
                  file=sys.stderr)
        sys.exit(
            f"ERROR: {len(gold_diffs)} record(s) have gold drift between "
            f"C2 and C3. Data corruption — STOP."
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

    # --- Brief descriptive (transparency; NOT the gate) --------------------
    c2_correct = sum(1 for r in c2_lines if r["pred"] == r["gold"])
    c3_correct = sum(1 for r in c3_lines if r["pred"] == r["gold"])
    print(f"\n[descriptive] C2 raw accuracy: {c2_correct}/100  "
          f"C3 raw accuracy: {c3_correct}/100  "
          f"(the gate is paired-t MAE in Step 10)", flush=True)


if __name__ == "__main__":
    main()
