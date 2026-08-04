#!/usr/bin/env python3
"""
LL4-LL6 per-user paired comparison: Task-LoRA+BM25 baseline vs.
Task-LoRA+User-LoRA+BM25 stacked, over the K=100 pool for one task.

New script rather than a generalization of eval/paired_compare_per_user.py:
that script expects a single "consolidated predictions" file carrying a
`user_fingerprint` field per line, produced by a LaMP-specific
aggregate_user_predictions_*.py step that has no LongLaMP equivalent. Our
setup is also structurally simpler than the case that script was built for
-- every LL4-LL6 top-100 user has exactly 1 temporal-test record (confirmed
directly from data/longlamp_user_stats/<tag>_top100_users.json), so there is
no per-user multi-record averaging to do; each user contributes exactly one
(baseline_score, personalized_score) pair.

Reads:
  - the baseline batch predictions.jsonl (--user-records-from-file run,
    condor/eval_longlamp_user_lora_baseline.sub) -- keyed by the synthetic
    file-order `id`.
  - each of the 100 users' individual personalized-arm predictions.jsonl
    (condor/eval_longlamp_user_lora.sub), located by globbing on the
    `_user<safe_user_tag>` suffix eval_longlamp.py's own stem construction
    appends (see eval/eval_longlamp.py's `user_tag` logic) -- avoids
    hand-reconstructing the full stacked-adapter-tag filename.

The rouge1_scorer / paired_t_test / wilcoxon_signed_rank / bootstrap_ci_mean
helpers are byte-duplicated from eval/paired_compare_per_user.py (itself
duplicated from eval/paired_compare.py) -- matching this project's
standalone-script convention. Keep in sync if scoring logic changes there.

Output:
  results/paired_compare_longlamp_<tag>_baseline_vs_personalized_test.json
  results/paired_compare_longlamp_<tag>_baseline_vs_personalized_test.pairs.jsonl
    (one row per user: {user_id, baseline_rouge1, personalized_rouge1, diff})

Usage:
    python eval/paired_compare_longlamp_user.py --tag review
    python eval/paired_compare_longlamp_user.py --tag abstract --overwrite
"""

import argparse
import datetime
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/home/ange00008/projects/mobileFT_distill")
)
RESULTS_DIR = PROJECT_ROOT / "results"
USER_STATS_DIR = PROJECT_ROOT / "data" / "longlamp_user_stats"

TEMPORAL_TASK = {
    "review": "product_review_temporal",
    "abstract": "abstract_generation_temporal",
    "topic": "topic_writing_temporal",
}

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_user_tag(user_id: str) -> str:
    """Byte-identical to eval/eval_longlamp.py's / train/build_longlamp_user_dataset.py's
    copies -- keep all three in sync."""
    tag = _UNSAFE.sub("_", user_id).strip("_")
    return tag or "user"


# --- duplicated from eval/paired_compare_per_user.py (see module docstring) -
def rouge1_scorer():
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return lambda gold, pred: rs.score(str(gold), str(pred))["rouge1"].fmeasure


def paired_t_test(diffs: list) -> tuple:
    """Paired t-test of `diffs` against zero. Returns (statistic, pvalue), or
    (None, None) in the degenerate all-zero case.

    The all-zero guard is load-bearing, not defensive. Without it the `+ 1e-30`
    offset below turns a perfectly null result into a maximally significant one:
    ttest_rel([1e-30]*n, [0]*n) has zero variance, so t -> ~5.7e16 and p -> 0.0.
    A comparison where every single user tied would be reported as p<0.001 —
    the exact opposite of what happened. `wilcoxon_signed_rank` already returns
    (None, None) here; the t-test must agree.

    Backported 2026-08-04. The fix landed in eval/paired_compare.py and
    eval/paired_compare_per_user.py under commit a7caec4 but missed this third
    byte-duplicated copy, leaving the bug live on the LongLaMP track. No LL4-LL6
    comparison has hit it (all three tasks have non-zero diffs), but a future
    all-tie LongLaMP comparison would have published a spurious p<0.001.
    """
    if all(d == 0 for d in diffs):
        return None, None
    from scipy import stats
    res = stats.ttest_rel([d + 1e-30 for d in diffs], [0.0] * len(diffs))
    return float(res.statistic), float(res.pvalue)


def wilcoxon_signed_rank(diffs: list) -> tuple:
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


def load_predictions_by_id(path: Path) -> dict:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            rid = str(r["id"])
            if rid in out:
                sys.exit(f"ERROR: duplicate id {rid} in {path}")
            out[rid] = {"pred": r["pred"], "gold": r["gold"]}
    return out


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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tag", required=True, choices=list(TEMPORAL_TASK.keys()))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    temporal_task = TEMPORAL_TASK[args.tag]
    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(f"[run] paired_compare_longlamp_user tag={args.tag} commit={commit_short} "
          f"host={provenance.get('hostname')}", flush=True)

    stem = f"paired_compare_longlamp_{args.tag}_baseline_vs_personalized_test"
    out_path = RESULTS_DIR / f"{stem}.json"
    pairs_path = RESULTS_DIR / f"{stem}.pairs.jsonl"
    if not args.overwrite and (out_path.exists() or pairs_path.exists()):
        sys.exit(f"ERROR: refusing to overwrite {out_path} / {pairs_path} — pass --overwrite to force")

    top_users_path = USER_STATS_DIR / f"{args.tag}_top100_users.json"
    if not top_users_path.exists():
        sys.exit(f"ERROR: {top_users_path} missing.")
    top = json.loads(top_users_path.read_text())
    users = top["users"]
    if len(users) != 100:
        print(f"[warn] expected 100 users, found {len(users)}", flush=True)

    # This glob became ambiguous on 2026-08-04: the no-adapter base-model eval
    # (commit 729692f) writes a second `_topK100` file for the same task, so a
    # re-run of this script would have exited on "found 2" -- or, worse under a
    # laxer check, silently scored against the wrong baseline. The comparison is
    # defined against the TASK-LORA baseline; exclude the base-model arm.
    baseline_glob = [
        p for p in RESULTS_DIR.glob(
            f"LongLaMP_{temporal_task}_test_*_topK100.predictions.jsonl")
        if f"_test_base_" not in p.name
    ]
    if len(baseline_glob) != 1:
        sys.exit(f"ERROR: expected exactly 1 Task-LoRA baseline predictions file, "
                 f"found {len(baseline_glob)}: {[p.name for p in baseline_glob]}")
    baseline_preds = load_predictions_by_id(baseline_glob[0])
    print(f"[load] baseline: {baseline_glob[0].name} ({len(baseline_preds)} records)",
          flush=True)

    score_fn = rouge1_scorer()
    pairs = []
    diffs = []
    missing_users = []
    for u in users:
        user_id = u["user_id"]
        test_ids = u["test_record_ids"]
        if len(test_ids) != 1:
            sys.exit(f"ERROR: user {user_id!r} has {len(test_ids)} test records, "
                     f"expected exactly 1 (this script assumes the K=100 pool's "
                     f"1-record-per-user property confirmed for LL4-LL6).")
        rid = str(test_ids[0])

        utag = safe_user_tag(user_id)
        personalized_glob = list(RESULTS_DIR.glob(
            f"LongLaMP_{temporal_task}_test_*_user{utag}.predictions.jsonl"
        ))
        if len(personalized_glob) != 1:
            missing_users.append((user_id, len(personalized_glob)))
            continue
        personalized_preds = load_predictions_by_id(personalized_glob[0])

        if rid not in baseline_preds:
            sys.exit(f"ERROR: test record id {rid} (user {user_id!r}) not found "
                     f"in baseline predictions {baseline_glob[0].name}.")
        if rid not in personalized_preds:
            sys.exit(f"ERROR: test record id {rid} (user {user_id!r}) not found "
                     f"in personalized predictions {personalized_glob[0].name}.")

        gold_base = baseline_preds[rid]["gold"]
        gold_pers = personalized_preds[rid]["gold"]
        if str(gold_base) != str(gold_pers):
            sys.exit(f"ERROR: gold mismatch for user {user_id!r} id={rid}: "
                     f"baseline has {gold_base!r}, personalized has {gold_pers!r}.")

        s_base = score_fn(gold_base, baseline_preds[rid]["pred"])
        s_pers = score_fn(gold_pers, personalized_preds[rid]["pred"])
        d = s_pers - s_base
        pairs.append({
            "user_id": user_id,
            "test_record_id": rid,
            "baseline_rouge1": s_base,
            "personalized_rouge1": s_pers,
            "diff": d,
        })
        diffs.append(d)

    if missing_users:
        sys.exit(f"ERROR: {len(missing_users)} users had != 1 matching personalized "
                 f"predictions file (e.g. {missing_users[:5]}) -- expected all 100 "
                 f"from condor/eval_longlamp_user_lora.sub to be present.")

    n = len(diffs)
    print(f"[score] {n} users scored", flush=True)

    mean_base = sum(p["baseline_rouge1"] for p in pairs) / n
    mean_pers = sum(p["personalized_rouge1"] for p in pairs) / n
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    std_diff = var_diff ** 0.5
    t_stat, p_val = paired_t_test(diffs)
    w_stat, w_pval = wilcoxon_signed_rank(diffs)
    ci_lo, ci_hi = bootstrap_ci_mean(diffs, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)

    wins_pers = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    wins_base = sum(1 for d in diffs if d < 0)

    record = {
        "schema_version": 1,
        "tag": args.tag,
        "temporal_task": temporal_task,
        "split": "test",
        "metric": "rouge1",
        "metric_direction": "higher_is_better",
        "n_users": n,
        "mean_baseline": mean_base,
        "mean_personalized": mean_pers,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "paired_t_stat": t_stat,
        "paired_t_pvalue": p_val,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pval,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "n_boot": args.n_boot,
        "alpha": args.alpha,
        "wins_personalized": wins_pers,
        "ties": ties,
        "wins_baseline": wins_base,
        "baseline_predictions_file": str(baseline_glob[0]),
        "command": "python " + " ".join(sys.argv),
        **provenance,
    }
    out_path.write_text(json.dumps(record, indent=2))
    with pairs_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print("=" * 60, flush=True)
    print(f"tag={args.tag} n={n}", flush=True)
    print(f"mean_baseline={mean_base:.4f} mean_personalized={mean_pers:.4f} "
          f"mean_diff={mean_diff:+.4f}", flush=True)
    print(f"paired-t: stat={t_stat:.4f} p={p_val:.4f}", flush=True)
    print(f"wilcoxon: stat={w_stat} p={w_pval}", flush=True)
    print(f"bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]", flush=True)
    print(f"win/tie/loss (personalized/tie/baseline): {wins_pers}/{ties}/{wins_base}",
          flush=True)
    print(f"written -> {out_path}", flush=True)
    print(f"           {pairs_path}", flush=True)


if __name__ == "__main__":
    main()
