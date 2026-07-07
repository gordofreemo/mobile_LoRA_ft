#!/usr/bin/env python3
"""
Round 6 Step 11 — per-user-mean paired comparison of the two consolidated
LaMP-4 predictions files from Step 10 (results/LaMP_4_test_round6_C{2,3}.predictions.jsonl).

Unlike Round 5's `paired_compare.py` (which pairs at the RECORD level — valid
for LaMP-3's R5 pool, where every user contributed exactly 1 test record, so
record-level == user-level), LaMP-4 users contribute 1-25 test records each
(248 total across the K=100 pool). Naively pairing at the record level would
over-weight users with many test records and violate the independence
assumption a paired-t across USERS requires. So this script:

  1. Scores ROUGE-1 per record (joined on `id`, gold byte-match re-verified).
  2. Groups per-record scores by `user_fingerprint` (present in each line,
     added by Step 10's aggregator).
  3. Computes each user's MEAN ROUGE-1 for C2 and C3 separately.
  4. Runs the paired-t / Wilcoxon / bootstrap-CI battery on the K=100
     per-user mean differences (not the 248 per-record differences).

Per the Round-6 plan's decision #9: NO pre-registered gate. This script
reports raw numbers only — no `passes_registered_gate` field, no PASS/FAIL
verdict. Per decision #10: this script is new (not a generalization of
paired_compare.py) to keep the R5 LaMP-3 paths/outputs untouched.

The rouge1_scorer / paired_t_test / wilcoxon_signed_rank / bootstrap_ci_mean
helpers are byte-duplicated from `eval/paired_compare.py` rather than
imported — matching this project's standalone-script convention (every
Condor-submitted script in data/, train/, eval/ avoids cross-file imports so
it isn't broken by a sandbox that transfers only the named executable; see
`train/build_user_dataset.py`'s docstring for the same rationale). If you
change scoring logic here, change it in paired_compare.py too.

Plan reference: experiments/2026-06-19-user-lora-round6-lamp4-multi-plan.md §Step 11.

Output:
  results/paired_compare_c2_a1lamp_bm25_vs_c3_a1lamp_userlora_bm25_round6_LaMP_4_test.json
  results/paired_compare_c2_a1lamp_bm25_vs_c3_a1lamp_userlora_bm25_round6_LaMP_4_test.pairs.jsonl
    (one row per USER, not per record — {"user_fingerprint", "n_test",
     "mean_score_a", "mean_score_b", "diff"})

Usage:
    python eval/paired_compare_per_user.py
    python eval/paired_compare_per_user.py --overwrite
"""

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/home/ange00008/projects/mobileFT_distill")
)
RESULTS_DIR = PROJECT_ROOT / "results"


def _git(*a):
    try:
        return (
            subprocess.check_output(["git", *a], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def collect_provenance() -> dict:
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


# --- duplicated from eval/paired_compare.py (see module docstring) ---------
def rouge1_scorer() -> Callable[[str, str], float]:
    # Match eval_lamp.py exactly: rouge_score package, use_stemmer=True, F1
    # on rouge1, scorer.score(target=gold, prediction=pred). Argument order
    # matters — gold first.
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return lambda gold, pred: rs.score(str(gold), str(pred))["rouge1"].fmeasure


def paired_t_test(diffs: list) -> tuple:
    from scipy import stats
    res = stats.ttest_rel([d + 1e-30 for d in diffs], [0.0] * len(diffs))
    return float(res.statistic), float(res.pvalue)


def wilcoxon_signed_rank(diffs: list) -> tuple:
    """Non-parametric robustness check. Returns (statistic, pvalue) or (None, None)
    if all diffs are zero (degenerate case scipy refuses to score)."""
    if all(d == 0 for d in diffs):
        return None, None
    from scipy import stats
    try:
        res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return None, None


def bootstrap_ci_mean(diffs: list, n_boot: int = 10_000, alpha: float = 0.05,
                      seed: int = 0) -> tuple:
    """Percentile bootstrap CI on the mean of `diffs`. Deterministic given seed."""
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return float(lo), float(hi)
# --- end duplicated block ---------------------------------------------------


def load_predictions(path: Path) -> dict:
    """Return {id: {pred, gold, user_fingerprint}} from a predictions JSONL."""
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            rid = str(r["id"])
            if rid in out:
                sys.exit(f"ERROR: duplicate id {rid} in {path}")
            out[rid] = {
                "pred": r["pred"],
                "gold": r["gold"],
                "user_fingerprint": r["user_fingerprint"],
            }
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pred-a", type=Path,
                        default=RESULTS_DIR / "LaMP_4_test_round6_C2.predictions.jsonl",
                        help="condition A (baseline) consolidated predictions.jsonl")
    parser.add_argument("--pred-b", type=Path,
                        default=RESULTS_DIR / "LaMP_4_test_round6_C3.predictions.jsonl",
                        help="condition B (treatment) consolidated predictions.jsonl")
    parser.add_argument("--label-a", default="c2_a1lamp_bm25")
    parser.add_argument("--label-b", default="c3_a1lamp_userlora_bm25")
    parser.add_argument("--task", default="LaMP_4")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for p in (args.pred_a, args.pred_b):
        if not p.exists():
            sys.exit(f"ERROR: {p} missing. Run Step 10 (aggregate_user_predictions_lamp4.py) first.")

    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] paired_compare_per_user task={args.task} split={args.split} "
        f"metric=rouge1 a={args.label_a} b={args.label_b} "
        f"commit={commit_short} dirty={provenance.get('git_dirty')} "
        f"condor={provenance.get('condor_cluster_id') or '-'}."
        f"{provenance.get('condor_proc_id') or '-'} "
        f"host={provenance.get('hostname')}",
        flush=True,
    )

    stem = f"paired_compare_{args.label_a}_vs_{args.label_b}_round6_{args.task}_{args.split}"
    out_path = RESULTS_DIR / f"{stem}.json"
    pairs_path = RESULTS_DIR / f"{stem}.pairs.jsonl"
    if not args.overwrite and (out_path.exists() or pairs_path.exists()):
        sys.exit(f"ERROR: refusing to overwrite {out_path} / {pairs_path} — pass --overwrite to force")

    # --- Load + join on id ---------------------------------------------------
    a = load_predictions(args.pred_a)
    b = load_predictions(args.pred_b)
    shared_ids = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    if only_a or only_b:
        print(
            f"[warn] {len(only_a)} ids only in A, {len(only_b)} ids only in B — "
            f"comparing on {len(shared_ids)} shared",
            flush=True,
        )
    if not shared_ids:
        sys.exit("ERROR: no shared ids between A and B")

    # --- Score per record, then group by user --------------------------------
    score_fn = rouge1_scorer()
    per_user_scores_a = defaultdict(list)
    per_user_scores_b = defaultdict(list)
    for rid in shared_ids:
        gold = a[rid]["gold"]
        gold_b = b[rid]["gold"]
        if str(gold) != str(gold_b):
            sys.exit(
                f"ERROR: gold mismatch at id={rid}: A has {gold!r}, B has {gold_b!r}. "
                "The two prediction files must be over the same gold set."
            )
        fp_a = a[rid]["user_fingerprint"]
        fp_b = b[rid]["user_fingerprint"]
        if fp_a != fp_b:
            sys.exit(
                f"ERROR: user_fingerprint mismatch at id={rid}: A has {fp_a!r}, "
                f"B has {fp_b!r}."
            )
        s_a = score_fn(gold, a[rid]["pred"])
        s_b = score_fn(gold, b[rid]["pred"])
        per_user_scores_a[fp_a].append(s_a)
        per_user_scores_b[fp_a].append(s_b)

    users = sorted(set(per_user_scores_a) | set(per_user_scores_b))
    if set(per_user_scores_a) != set(per_user_scores_b):
        sys.exit("ERROR: user sets differ between A and B scoring — should be impossible "
                 "given the id-level join above.")

    # --- Per-user means + paired diffs ---------------------------------------
    pairs = []
    diffs = []
    for fp in users:
        scores_a = per_user_scores_a[fp]
        scores_b = per_user_scores_b[fp]
        n_test = len(scores_a)
        mean_a_u = sum(scores_a) / n_test
        mean_b_u = sum(scores_b) / n_test
        d = mean_b_u - mean_a_u
        pairs.append({
            "user_fingerprint": fp,
            "n_test": n_test,
            "mean_score_a": mean_a_u,
            "mean_score_b": mean_b_u,
            "diff": d,
        })
        diffs.append(d)

    n = len(diffs)
    # Per-user mean-of-means (unweighted across users), per the plan's
    # decision #11 ("aggregated as per-user mean, not record-weighted").
    mean_a = sum(p["mean_score_a"] for p in pairs) / n
    mean_b = sum(p["mean_score_b"] for p in pairs) / n
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    std_diff = var_diff ** 0.5
    t_stat, p_val = paired_t_test(diffs)
    w_stat, w_pval = wilcoxon_signed_rank(diffs)
    ci_lo, ci_hi = bootstrap_ci_mean(diffs, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)

    wins_b = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    wins_a = sum(1 for d in diffs if d < 0)

    record = {
        "schema_version": 1,
        "task": args.task,
        "split": args.split,
        "metric": "rouge1",
        "metric_direction": "higher_is_better",
        "label_a": args.label_a,
        "label_b": args.label_b,
        "pred_a_path": str(args.pred_a),
        "pred_b_path": str(args.pred_b),
        "n": n,
        "n_records_total": len(shared_ids),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": t_stat,
        "t_pvalue": p_val,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_pval,
        "ci_alpha": args.alpha,
        "ci_lo_95": ci_lo,
        "ci_hi_95": ci_hi,
        "n_boot": args.n_boot,
        "boot_seed": args.seed,
        "wins_b_over_a": wins_b,
        "ties": ties,
        "wins_a_over_b": wins_a,
        **provenance,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    with pairs_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(
        f"[done] n={n} (users) n_records_total={len(shared_ids)} "
        f"mean_a={mean_a:.4f} mean_b={mean_b:.4f} "
        f"mean_diff={mean_diff:+.4f} 95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
        f"t_p={p_val:.4f} wilcoxon_p={w_pval if w_pval is None else f'{w_pval:.4f}'} "
        f"wins_b={wins_b} ties={ties} wins_a={wins_a}",
        flush=True,
    )
    print(f"[done] result -> {out_path}", flush=True)
    print(f"[done] pairs  -> {pairs_path}", flush=True)


if __name__ == "__main__":
    main()
