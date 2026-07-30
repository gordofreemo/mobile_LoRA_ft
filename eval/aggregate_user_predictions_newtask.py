#!/usr/bin/env python3
"""
Consolidate per-user prediction files into two condition-level JSONL files
for the new User-LoRA tracks whose users hold MULTIPLE test records.

Handles both eval patterns, picked automatically from the pool JSON's
`eval_pattern` field:

  grouped_per_user (LaMP-2-news, R10/PT3) — 27 users holding 3-35 test
      records each. BOTH arms ran as per-user procs, so both sides are
      consolidated here, then compared with eval/paired_compare_per_user.py
      (per-user means, --metric accuracy).

  flat_record_level (LaMP-1, LaMP-2-movies, LaMP-5, LaMP-7) — every user
      holds exactly 1 test record. The baseline arm is user-invariant and
      already ran as ONE job via --user-records-from-file, landing in a
      single `..._topK<K>.predictions.jsonl`; only the per-user treatment arm
      needs consolidating. Matches how PT1/R8 did it for LaMP-3
      (eval/aggregate_user_predictions_lamp3_ptl.py), and the result is
      compared with the flat eval/paired_compare.py.

Either way the same gold byte-match leakage check runs across the two arms.

Generalizes eval/aggregate_user_predictions_lamp4{,_ptl}.py (R6 / PT2) across
task and track instead of hardcoding LaMP-4 and one base adapter.

Reads the task's pool JSON to know which users are in scope, locates each
user's two predictions files by eval_lamp.py's canonical stem naming, asserts
each has exactly the expected line count (from <task>_user_records.json), and
writes:

    results/<task>_test_<track>_C2.predictions.jsonl
    results/<task>_test_<track>_C3.predictions.jsonl

Each output line: {"id", "pred", "gold", "user_fingerprint"} — the
fingerprint is what paired_compare_per_user.py groups on.

Final leakage check: asserts gold byte-match between the two arms for every
record id, so a mismatched or stale file can't silently skew the comparison.

Usage (CPU, sub-second):
    python eval/aggregate_user_predictions_newtask.py --task LaMP_2_news --track r10
    python eval/aggregate_user_predictions_newtask.py --task LaMP_2_news --track pt3 --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"

ONE_LORA_FT_TAG = "a2_lamp_1ep_seed0_final"

# task -> (short tag, pool JSON, Per-Task-LoRA adapter tag)
TASKS = {
    "LaMP_2_news": ("lamp2news", "LaMP_2_news_top27_users.json",
                    "per_task_lamp2_news_1ep_seed0_final"),
    "LaMP_1": ("lamp1", "LaMP_1_top100_users.json",
               "per_task_lamp1_1ep_seed0_final"),
    "LaMP_7": ("lamp7", "LaMP_7_top100_users.json",
               "per_task_lamp7_1ep_seed0_final"),
    "LaMP_2_movies": ("lamp2movies", "LaMP_2_movies_top100_users.json",
                      "per_task_lamp2_movies_1ep_seed0_final"),
    "LaMP_5": ("lamp5", "LaMP_5_top100_users.json",
               "per_task_lamp5_1ep_seed0_final"),
}

TRACKS = {
    "r10": ("LaMP_2_news", True),   "pt3": ("LaMP_2_news", False),
    "r11": ("LaMP_1", True),        "pt4": ("LaMP_1", False),
    "r12": ("LaMP_7", True),        "pt5": ("LaMP_7", False),
    "r13": ("LaMP_2_movies", True), "pt6": ("LaMP_2_movies", False),
    "r14": ("LaMP_5", True),        "pt7": ("LaMP_5", False),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--track", required=True, choices=sorted(TRACKS))
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--bm25-k", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if TRACKS[args.track][0] != args.task:
        sys.exit(f"ERROR: track {args.track} belongs to "
                 f"{TRACKS[args.track][0]}, not {args.task}.")

    tag, pool_file, per_task_tag = TASKS[args.task]
    base_tag = ONE_LORA_FT_TAG if TRACKS[args.track][1] else per_task_tag

    pool_path = USER_STATS_DIR / pool_file
    records_path = USER_STATS_DIR / f"{args.task}_user_records.json"
    for p in (pool_path, records_path):
        if not p.exists():
            sys.exit(f"ERROR: missing {p}.")
    pool = json.loads(pool_path.read_text())
    flat = pool.get("eval_pattern") == "flat_record_level"
    fps = [u["user_fingerprint"] for u in pool["users"]]
    records = json.loads(records_path.read_text())

    profile_tag = f"bm25k{args.bm25_k}"

    def c2_path(fp):
        return RESULTS_DIR / (
            f"{args.task}_{args.split}_{base_tag}_{profile_tag}_seed0_"
            f"user{fp}.predictions.jsonl"
        )

    def c3_path(fp):
        user_tag = f"user_lora_{tag}_{fp}_{args.track}_seed0_final"
        return RESULTS_DIR / (
            f"{args.task}_{args.split}_{base_tag}_{user_tag}_{profile_tag}_"
            f"seed0_user{fp}.predictions.jsonl"
        )

    # Flat pattern: the baseline arm is one shared job, not K per-user files.
    c2_batch = RESULTS_DIR / (
        f"{args.task}_{args.split}_{base_tag}_{profile_tag}_seed0_"
        f"topK{pool['k']}.predictions.jsonl"
    )

    out_c2 = RESULTS_DIR / f"{args.task}_{args.split}_{args.track}_C2.predictions.jsonl"
    out_c3 = RESULTS_DIR / f"{args.task}_{args.split}_{args.track}_C3.predictions.jsonl"
    existing = [p for p in (out_c2, out_c3) if p.exists()]
    if existing and not args.overwrite:
        sys.exit(f"ERROR: refusing to overwrite {existing}. Pass --overwrite.")

    arms = [c3_path] if flat else [c2_path, c3_path]
    missing = [p for fp in fps for f in arms for p in (f(fp),) if not p.exists()]
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} of {len(arms)*len(fps)} per-user prediction "
            f"files are missing (e.g. {missing[0].name}). Every (user, arm) "
            f"cell must have completed before aggregating."
        )
    if flat and not c2_batch.exists():
        sys.exit(
            f"ERROR: missing the baseline batch file {c2_batch.name}. Run the "
            f"{args.track} baseline sub (single job, --user-records-from-file) "
            f"first."
        )

    rows_c2, rows_c3 = [], []
    n_count_mismatch = []
    for fp in fps:
        expected = len(records[fp][args.split])
        b = [json.loads(x) for x in c3_path(fp).read_text().splitlines() if x.strip()]
        a = ([] if flat else
             [json.loads(x) for x in c2_path(fp).read_text().splitlines() if x.strip()])
        if len(b) != expected or (not flat and len(a) != expected):
            n_count_mismatch.append((fp, expected, len(a), len(b)))
            continue
        for r in a:
            rows_c2.append({**{k: r[k] for k in ("id", "pred", "gold")},
                            "user_fingerprint": fp})
        for r in b:
            rows_c3.append({**{k: r[k] for k in ("id", "pred", "gold")},
                            "user_fingerprint": fp})

    if n_count_mismatch:
        sys.exit(
            f"ERROR: {len(n_count_mismatch)} users have the wrong record count "
            f"(fp, expected, n_c2, n_c3): {n_count_mismatch[:5]}. A count of 0 "
            f"usually means the eval job ran without LAMP_DIR=data/lamp_time "
            f"and matched nothing — check that before rerunning."
        )

    if flat:
        # Baseline already consolidated; carry the fingerprint over from the
        # C3 side by record id so both output files have the same schema.
        fp_by_id = {r["id"]: r["user_fingerprint"] for r in rows_c3}
        batch = [json.loads(x) for x in c2_batch.read_text().splitlines() if x.strip()]
        if len(batch) != len(fps):
            sys.exit(f"ERROR: {c2_batch.name} has {len(batch)} lines, "
                     f"expected {len(fps)} (one per pool user).")
        for r in batch:
            if r["id"] not in fp_by_id:
                sys.exit(f"ERROR: baseline batch has id {r['id']} with no "
                         f"matching per-user treatment record.")
            rows_c2.append({**{k: r[k] for k in ("id", "pred", "gold")},
                            "user_fingerprint": fp_by_id[r["id"]]})

    # --- leakage / alignment check: golds must byte-match across arms ------
    gold_c2 = {r["id"]: r["gold"] for r in rows_c2}
    gold_c3 = {r["id"]: r["gold"] for r in rows_c3}
    if set(gold_c2) != set(gold_c3):
        sys.exit("ERROR: record-id sets differ between the two arms.")
    bad = [rid for rid in gold_c2 if str(gold_c2[rid]) != str(gold_c3[rid])]
    if bad:
        sys.exit(f"ERROR: gold mismatch on {len(bad)} ids (e.g. {bad[:5]}).")

    out_c2.write_text("".join(json.dumps(r) + "\n" for r in rows_c2))
    out_c3.write_text("".join(json.dumps(r) + "\n" for r in rows_c3))
    print(f"[write] {len(rows_c2)} records across {len(fps)} users -> {out_c2}",
          flush=True)
    print(f"[write] {len(rows_c3)} records across {len(fps)} users -> {out_c3}",
          flush=True)
    print(f"[check] gold byte-match verified on all {len(gold_c2)} record ids",
          flush=True)


if __name__ == "__main__":
    main()
