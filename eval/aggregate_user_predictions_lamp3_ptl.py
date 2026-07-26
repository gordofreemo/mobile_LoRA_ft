#!/usr/bin/env python3
"""
PT1 Step 11 — consolidate the 100 per-user C3''' prediction files from
Step 10 into one condition-level JSONL for the descriptive comparison.

Reads `data/lamp_user_stats/LaMP_3_top100_users.json` to know which users are
in the pool (R5's exact K=100 pool, reused unchanged). For each user, locates
their C3''' predictions JSONL file in `results/` by eval_lamp.py's canonical
naming, asserts each has exactly one line, and writes:

    results/LaMP_3_test_pt1_C3.predictions.jsonl  (100 lines)

Each output line: `{"id": ..., "pred": ..., "gold": ..., "user_fingerprint": ...}`.

Clone of eval/aggregate_user_predictions_lamp3_r8.py with BASE_TAG swapped
to Per-Task-LoRA(LaMP-3) and the per-user adapter suffix swapped to _ptl.
C2''' is user-invariant and already lives in ONE batch file
(results/LaMP_3_test_per_task_lamp3_1ep_seed0_final_bm25k4_seed0_topK100.predictions.jsonl,
produced by condor/eval_lamp_user_lamp3_ptl_baseline.sub) rather than 100
per-user files — so this script only consolidates the C3''' side, then loads
the C2''' batch file and runs the same leakage check R5/R8's aggregators ran:
gold byte-match between C2''' and the consolidated C3''' for every shared id.

Plan reference: experiments/2026-07-25-per-task-lora-pt1-lamp3-pilot-plan.md
"Scaffolding inventory" item 11.

Usage (CPU, sub-second):
    python eval/aggregate_user_predictions_lamp3_ptl.py
    python eval/aggregate_user_predictions_lamp3_ptl.py --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"

BASE_TAG = "per_task_lamp3_1ep_seed0_final"

C2_BATCH_PRED_PATH = RESULTS_DIR / f"LaMP_3_test_{BASE_TAG}_bm25k4_seed0_topK100.predictions.jsonl"


def c3_pred_path(fp: str) -> Path:
    return (
        RESULTS_DIR
        / f"LaMP_3_test_{BASE_TAG}_user_lora_lamp3_{fp}_ptl_seed0_final_bm25k4_seed0_user{fp}.predictions.jsonl"
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
        "--c2-batch",
        type=Path,
        default=C2_BATCH_PRED_PATH,
    )
    parser.add_argument(
        "--out-c3",
        type=Path,
        default=RESULTS_DIR / "LaMP_3_test_pt1_C3.predictions.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing consolidated file (default: refuse)",
    )
    args = parser.parse_args()

    if not args.top_users.exists():
        sys.exit(f"ERROR: missing {args.top_users}.")
    if not args.c2_batch.exists():
        sys.exit(f"ERROR: missing C2''' batch predictions {args.c2_batch}. "
                  f"Run condor/eval_lamp_user_lamp3_ptl_baseline.sub first.")

    if args.out_c3.exists() and not args.overwrite:
        sys.exit(f"ERROR: refusing to overwrite existing {args.out_c3}. "
                  f"Pass --overwrite to replace.")

    top = json.loads(args.top_users.read_text())
    fps = [u["user_fingerprint"] for u in top["users"]]
    print(f"[agg] {len(fps)} users from {args.top_users}", flush=True)

    # --- Pre-flight: every C3''' file present + exactly 1 line --------------
    missing = []
    one_line_violators = []
    c3_lines = []
    for fp in fps:
        p = c3_pred_path(fp)
        if not p.exists():
            missing.append((fp, str(p)))
            continue
        recs = [json.loads(l) for l in p.open() if l.strip()]
        if len(recs) != 1:
            one_line_violators.append((fp, len(recs)))
            continue
        r = recs[0]
        c3_lines.append({
            "id": r["id"],
            "pred": r["pred"],
            "gold": r["gold"],
            "user_fingerprint": fp,
        })

    if missing:
        for m in missing[:5]:
            print(f"  MISSING: {m}", file=sys.stderr)
        sys.exit(f"ERROR: {len(missing)} C3''' prediction files missing.")
    if one_line_violators:
        for v in one_line_violators[:5]:
            print(f"  LINE-COUNT-NEQ-1: {v}", file=sys.stderr)
        sys.exit(f"ERROR: {len(one_line_violators)} C3''' prediction files with != 1 line.")

    assert len(c3_lines) == 100, len(c3_lines)

    # --- Load C2''' batch predictions ----------------------------------------
    c2_lines = [json.loads(l) for l in args.c2_batch.open() if l.strip()]
    if len(c2_lines) != 100:
        sys.exit(f"ERROR: C2''' batch file has {len(c2_lines)} lines, expected 100.")

    # --- Leakage check: id paired + gold byte-match --------------------------
    c2_by_id = {r["id"]: r for r in c2_lines}
    c3_by_id = {r["id"]: r for r in c3_lines}
    c2_ids = set(c2_by_id)
    c3_ids = set(c3_by_id)
    if c2_ids != c3_ids:
        only_c2 = c2_ids - c3_ids
        only_c3 = c3_ids - c2_ids
        sys.exit(
            f"ERROR: id set mismatch between C2''' and C3'''. "
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
            print(f"  GOLD DRIFT: id={d[0]}, C2'''_gold={d[1]!r}, C3'''_gold={d[2]!r}",
                  file=sys.stderr)
        sys.exit(
            f"ERROR: {len(gold_diffs)} record(s) have gold drift between "
            f"C2''' and C3'''. Data corruption — STOP."
        )

    # --- Write consolidated JSONL --------------------------------------------
    args.out_c3.parent.mkdir(parents=True, exist_ok=True)
    with args.out_c3.open("w") as f:
        for r in c3_lines:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {args.out_c3}  ({len(c3_lines)} lines)", flush=True)

    # --- Brief descriptive (transparency; NOT a gate — PT1 has no pre-registered gate) ---
    c2_correct = sum(1 for r in c2_lines if str(r["pred"]).strip() == str(r["gold"]).strip())
    c3_correct = sum(1 for r in c3_lines if str(r["pred"]).strip() == str(r["gold"]).strip())
    print(f"\n[descriptive] C2''' raw accuracy: {c2_correct}/100  "
          f"C3''' raw accuracy: {c3_correct}/100  "
          f"(no pre-registered gate this round — Step 12 reports descriptives only)",
          flush=True)


if __name__ == "__main__":
    main()
