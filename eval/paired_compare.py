#!/usr/bin/env python3
"""Paired per-record comparison of two LaMP predictions files.

Joins two `*.predictions.jsonl` files on `id`, recomputes per-record metric
(ROUGE-1 for generation, accuracy or MAE for rating) against the shared gold,
then runs:
  - paired t-test on the per-record metric differences (B − A)
  - 95% bootstrap CI on the mean difference (10k resamples, percentile method)
  - Wilcoxon signed-rank as a non-parametric robustness check (supplementary)

Metrics:
  - rouge1 (higher-is-better): default for generation tasks (LaMP-4/-7).
  - accuracy (higher-is-better): exact-match for rating prediction (LaMP-3).
  - mae (lower-is-better): absolute distance for ordinal rating prediction
    (LaMP-3). For MAE the gate inverts to `mean_diff < 0 AND p < alpha`
    (improvement direction). Added 2026-06-18 for Round-5's per-user MAE
    gate (experiments/2026-06-18-user-lora-round5-lamp3-plan.md §Step 10).

The pre-registered Round-1 (LaMP-4 / rouge1) gate is `mean_diff > 0 AND
t_test_pvalue < 0.05` on test — see memory `project-user-lora-round1-design`.

Output (flat single-level JSON + sibling pairs JSONL, same shape as
`eval/eval_lamp.py`'s outputs so the summary tooling joins on the same fields):
  results/paired_compare_<label_a>_vs_<label_b>_<task>_<split>.json
  results/paired_compare_<label_a>_vs_<label_b>_<task>_<split>.pairs.jsonl

Usage:
    python eval/paired_compare.py \\
        --pred-a results/LaMP_4_test_..._useru00000011.predictions.jsonl \\
        --pred-b results/LaMP_4_test_..._useru00000011.predictions.jsonl \\
        --label-a c2_a1lamp_bm25 \\
        --label-b c3_a1lamp_userlora_bm25 \\
        --task LaMP_4 --split test
"""

import argparse
import datetime
import json
import os
import re
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/home/ange00008/projects/mobileFT_distill")
)
RESULTS_DIR = PROJECT_ROOT / "results"

METRICS = {"rouge1", "accuracy", "mae"}
LOWER_IS_BETTER = {"mae"}


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


def load_predictions(path: Path) -> dict:
    """Return {id: {pred, gold}} from a predictions JSONL."""
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            rid = str(r["id"])
            if rid in out:
                sys.exit(f"ERROR: duplicate id {rid} in {path}")
            out[rid] = {"pred": r["pred"], "gold": r["gold"]}
    return out


def rouge1_scorer() -> Callable[[str, str], float]:
    # Match eval_lamp.py exactly: rouge_score package, use_stemmer=True, F1
    # on rouge1, scorer.score(target=gold, prediction=pred). Argument order
    # matters — gold first.
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return lambda gold, pred: rs.score(str(gold), str(pred))["rouge1"].fmeasure


def accuracy_scorer() -> Callable[[str, str], float]:
    # LaMP-3 is rating prediction; per-record metric is 0/1 exact match on the
    # normalized string. Mirrors eval_lamp.score_rating's per-example logic.
    return lambda gold, pred: 1.0 if str(pred).strip() == str(gold).strip() else 0.0


def mae_scorer() -> Callable[[str, str], float]:
    # LaMP-3 absolute error per record. Mirrors eval_lamp.score_rating's MAE
    # logic (parse_rating + abs(int(pred) - int(gold))). Unparseable predictions
    # are penalized as the maximum possible distance (4 = |1-5|) so the paired
    # observations stay aligned — eval_lamp.py drops parse-fails from its
    # aggregate MAE, but a paired-t needs a per-example score for every record.
    # Round-5 Step-8 audit confirmed parse_fail_rate=0 across all 200 cells, so
    # this fallback is a safety net not load-bearing.
    def _score(gold, pred):
        m = re.search(r"[1-5]", str(pred))
        if m is None:
            return 4.0
        return float(abs(int(m.group(0)) - int(str(gold).strip())))
    return _score


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a", required=True, help="condition A predictions.jsonl (baseline)")
    parser.add_argument("--pred-b", required=True, help="condition B predictions.jsonl (treatment)")
    parser.add_argument("--label-a", required=True, help="short tag for A (e.g. c2_a1lamp_bm25)")
    parser.add_argument("--label-b", required=True, help="short tag for B (e.g. c3_a1lamp_userlora_bm25)")
    parser.add_argument("--task", required=True, help="task name for output stem (e.g. LaMP_4)")
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--metric", default="rouge1", choices=sorted(METRICS))
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pred_a_path = Path(args.pred_a)
    pred_b_path = Path(args.pred_b)
    for p in (pred_a_path, pred_b_path):
        if not p.exists():
            sys.exit(f"ERROR: {p} missing")

    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] paired_compare task={args.task} split={args.split} "
        f"metric={args.metric} a={args.label_a} b={args.label_b} "
        f"commit={commit_short} dirty={provenance.get('git_dirty')} "
        f"condor={provenance.get('condor_cluster_id') or '-'}."
        f"{provenance.get('condor_proc_id') or '-'} "
        f"host={provenance.get('hostname')}",
        flush=True,
    )

    stem = f"paired_compare_{args.label_a}_vs_{args.label_b}_{args.task}_{args.split}"
    out_path = RESULTS_DIR / f"{stem}.json"
    pairs_path = RESULTS_DIR / f"{stem}.pairs.jsonl"
    if not args.overwrite and (out_path.exists() or pairs_path.exists()):
        sys.exit(f"ERROR: refusing to overwrite {out_path} / {pairs_path} — pass --overwrite to force")

    # --- Load + join on id ---------------------------------------------------
    a = load_predictions(pred_a_path)
    b = load_predictions(pred_b_path)
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

    # --- Score per record ----------------------------------------------------
    if args.metric == "rouge1":
        score_fn = rouge1_scorer()
    elif args.metric == "mae":
        score_fn = mae_scorer()
    else:
        score_fn = accuracy_scorer()

    pairs = []
    diffs = []
    for rid in shared_ids:
        gold = a[rid]["gold"]
        gold_b = b[rid]["gold"]
        if str(gold) != str(gold_b):
            sys.exit(
                f"ERROR: gold mismatch at id={rid}: A has {gold!r}, B has {gold_b!r}. "
                "The two prediction files must be over the same gold set."
            )
        pred_a = a[rid]["pred"]
        pred_b = b[rid]["pred"]
        s_a = score_fn(gold, pred_a)
        s_b = score_fn(gold, pred_b)
        d = s_b - s_a
        pairs.append({
            "id": rid, "gold": gold,
            "pred_a": pred_a, "pred_b": pred_b,
            "score_a": s_a, "score_b": s_b, "diff": d,
        })
        diffs.append(d)

    n = len(diffs)
    mean_a = sum(p["score_a"] for p in pairs) / n
    mean_b = sum(p["score_b"] for p in pairs) / n
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    std_diff = var_diff ** 0.5
    t_stat, p_val = paired_t_test(diffs)
    w_stat, w_pval = wilcoxon_signed_rank(diffs)
    ci_lo, ci_hi = bootstrap_ci_mean(diffs, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)

    wins_b = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    wins_a = sum(1 for d in diffs if d < 0)

    # Pre-registered gate: improvement direction AND paired t-test p < args.alpha.
    # For higher-is-better metrics (rouge1, accuracy): improvement = mean_diff > 0.
    # For lower-is-better metrics (mae): improvement = mean_diff < 0.
    # Round-1 (LaMP-4 rouge1) used the > 0 form; Round-5 (LaMP-3 mae) uses < 0
    # per experiments/2026-06-18-user-lora-round5-lamp3-plan.md §"Design recap" row 7.
    if args.metric in LOWER_IS_BETTER:
        passes_gate = (mean_diff < 0) and (p_val is not None and p_val < args.alpha)
    else:
        passes_gate = (mean_diff > 0) and (p_val is not None and p_val < args.alpha)

    record = {
        "schema_version": 1,
        "task": args.task,
        "split": args.split,
        "metric": args.metric,
        "metric_direction": "lower_is_better" if args.metric in LOWER_IS_BETTER else "higher_is_better",
        "label_a": args.label_a,
        "label_b": args.label_b,
        "pred_a_path": str(pred_a_path),
        "pred_b_path": str(pred_b_path),
        "n": n,
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
        "passes_registered_gate": passes_gate,
        **provenance,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    with pairs_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(
        f"[done] n={n} mean_a={mean_a:.4f} mean_b={mean_b:.4f} "
        f"mean_diff={mean_diff:+.4f} 95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
        f"t_p={p_val:.4f} wilcoxon_p={w_pval if w_pval is None else f'{w_pval:.4f}'} "
        f"gate={'PASS' if passes_gate else 'FAIL'}",
        flush=True,
    )
    print(f"[done] result -> {out_path}", flush=True)
    print(f"[done] pairs  -> {pairs_path}", flush=True)


if __name__ == "__main__":
    main()
