#!/usr/bin/env python3
"""
Degeneration audit for a LongLaMP per-user paired comparison (LL4-LL6).

WHY THIS EXISTS
---------------
LL1/LL3 established that SmolLM3 + LoRA under plain greedy decoding falls into
verbatim repetition loops that run generation to the 1024-token cap. LL2 judged
Abstract Generation "largely fine" on the strength of its full-split mean, and
LL4-LL6 therefore treated Abstract as the one CLEAN test of per-user
personalization on long-form output.

That judgement was wrong at the per-record level. Re-analysis on 2026-08-04
found 30/100 Abstract BASELINE generations exceed 600 words against a ~144-word
median gold, and the personalized arm degenerates on only 21/100. That asymmetry
-- not personalization -- produces the whole of LL5's headline +0.0267 ROUGE-1:

    all users                    n=100  mean +0.0267  t +1.86
    baseline non-degenerate      n= 70  mean -0.0268  t -2.23
    BOTH arms non-degenerate     n= 63  mean +0.0013  t +0.20   <-- null
    baseline degenerate          n= 30  mean +0.1516  t +5.44
    corr(length change, diff) = -0.950

A mean ROUGE over a mixture of sane and runaway generations measures the
repetition lottery, not the model. Run this before reading ANY long-form paired
result, and report the both-arms-clean subset alongside the headline.

WHAT IT DOES
------------
Reuses eval/paired_compare_longlamp_user.py's file-location logic verbatim so
the audit reads exactly the predictions that comparison scored. For each user it
records baseline/personalized generation length against that record's own gold
length, flags degenerate generations, and recomputes the paired effect on the
subset where NEITHER arm degenerated.

THRESHOLD, CALIBRATED NOT GUESSED
---------------------------------
Degeneration is flagged per record as a FLAT word count, len(pred) >
--degenerate-words (default 600). That criterion was calibrated against the
no-adapter base-model arm (commit 729692f), which never runs away -- its longest
Abstract generation is 358 words:

    threshold        base flagged   Task-LoRA flagged
    2.0x gold             35                40
    3.0x gold             19                31
    4.0x gold              8                24
    flat >400w             0                30
    flat >600w             0                30      <-- default, mid-plateau

A gold-relative ratio is the wrong tool here: gold abstract length varies enough
that 2x flags 35/100 base generations that plainly did not degenerate. The flat
threshold separates perfectly and is stable anywhere in 400-600w. The ratio is
still computed and reported as a secondary figure (--length-ratio, default 4.0).

Output (flat single-level JSON + per-user sidecar, per project convention):
  results/longlamp_degeneration_audit_<tag>.json
  results/longlamp_degeneration_audit_<tag>.pairs.jsonl

Usage:
    condor_submit condor/longlamp_degeneration_audit.sub    # all 3 tasks
    python eval/longlamp_degeneration_audit.py --tag abstract
    python eval/longlamp_degeneration_audit.py --tag review --degenerate-words 500

NOTE: rouge_score/scipy are image-only, so this does not run on the login host.
The 2026-08-04 numbers quoted above were derived with a stubbed scorer and are
exact for lengths/counts but approximate for ROUGE -- the Condor run is what
confirms them.
"""

import argparse
import datetime
import json
import math
import os
import platform
import re
import socket
import statistics
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
    """Byte-identical to eval/paired_compare_longlamp_user.py's copy -- keep in sync."""
    tag = _UNSAFE.sub("_", user_id).strip("_")
    return tag or "user"


def decode_tag_selector(name: str, decode_tag: str, anchor: str) -> bool:
    """Byte-identical to eval/paired_compare_longlamp_user.py's copy -- keep in
    sync. Selects the predictions file carrying exactly `decode_tag` between
    the seed field and `anchor`; empty decode_tag selects plain greedy."""
    dt = f"_{re.escape(decode_tag)}" if decode_tag else ""
    return re.search(rf"_seed\d+{dt}_{re.escape(anchor)}\.", name) is not None


# --- duplicated from eval/paired_compare_longlamp_user.py (standalone-script
# --- convention; keep in sync if scoring logic changes there) ----------------
def rouge1_scorer():
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return lambda gold, pred: rs.score(str(gold), str(pred))["rouge1"].fmeasure


def paired_t_test(diffs: list) -> tuple:
    """See eval/paired_compare.py for why the all-zero guard is load-bearing."""
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
# --- end duplicated block ---------------------------------------------------


def pearson(xs: list, ys: list) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def load_predictions_by_id(path: Path) -> dict:
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[str(r["id"])] = {"pred": r["pred"], "gold": r["gold"]}
    return out


def collect_provenance() -> dict:
    def _git(*a):
        try:
            return subprocess.check_output(
                ["git", *a], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            ).decode().strip()
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


def summarize(diffs: list) -> dict:
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    t_stat, t_p = paired_t_test(diffs)
    w_stat, w_p = wilcoxon_signed_rank(diffs)
    return {
        "n": n,
        "mean_diff": mean,
        "std_diff": sd,
        "t_statistic": t_stat,
        "t_pvalue": t_p,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_p,
        "wins_personalized": sum(1 for d in diffs if d > 1e-9),
        "ties": sum(1 for d in diffs if abs(d) <= 1e-9),
        "wins_baseline": sum(1 for d in diffs if d < -1e-9),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tag", required=True, choices=list(TEMPORAL_TASK.keys()))
    parser.add_argument(
        "--decode-tag", default="",
        help="decoding-config filename tag to select, e.g. 'nrng3' for the "
             "--no-repeat-ngram-size 3 re-measurement (2026-08-10 round). "
             "Default '' selects the plain-greedy files and reproduces the "
             "2026-08-06 audit byte-for-byte. Appended to the output stem.",
    )
    parser.add_argument(
        "--degenerate-words", type=int, default=600,
        help="PRIMARY criterion: flag a generation as degenerate when it exceeds "
             "this many words. Default 600 is calibrated against the no-adapter "
             "base arm (0 false positives; stable 400-600) -- see module docstring.",
    )
    parser.add_argument(
        "--length-ratio", type=float, default=4.0,
        help="secondary, reported only: flag when word count exceeds this "
             "multiple of the record's own gold length (default 4.0)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    temporal_task = TEMPORAL_TASK[args.tag]
    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(f"[run] longlamp_degeneration_audit tag={args.tag} "
          f"ratio={args.length_ratio} commit={commit_short} "
          f"host={provenance.get('hostname')}", flush=True)

    stem = f"longlamp_degeneration_audit_{args.tag}"
    if args.decode_tag:
        stem += f"_{args.decode_tag}"
    out_path = RESULTS_DIR / f"{stem}.json"
    pairs_path = RESULTS_DIR / f"{stem}.pairs.jsonl"
    if not args.overwrite and (out_path.exists() or pairs_path.exists()):
        sys.exit(f"ERROR: refusing to overwrite {out_path} / {pairs_path} "
                 f"— pass --overwrite to force")

    top_users_path = USER_STATS_DIR / f"{args.tag}_top100_users.json"
    if not top_users_path.exists():
        sys.exit(f"ERROR: {top_users_path} missing.")
    users = json.loads(top_users_path.read_text())["users"]

    # The bare `_topK100` glob became ambiguous once the no-adapter base-model
    # eval landed (commit 729692f) -- it matches both the Task-LoRA baseline
    # this comparison is defined against and the base-model arm. Select the
    # Task-LoRA file explicitly rather than whichever the glob returns first.
    # The 2026-08-10 re-measurement adds a second axis of ambiguity (plain vs
    # `_nrng3` decoding), disambiguated by decode_tag_selector on --decode-tag.
    baseline_glob = [
        p for p in RESULTS_DIR.glob(
            f"LongLaMP_{temporal_task}_test_*_topK100.predictions.jsonl")
        if f"_test_base_" not in p.name
        and decode_tag_selector(p.name, args.decode_tag, "topK100")
    ]
    if len(baseline_glob) != 1:
        sys.exit(f"ERROR: expected exactly 1 Task-LoRA baseline predictions file, "
                 f"found {len(baseline_glob)}: {[p.name for p in baseline_glob]}")
    baseline_preds = load_predictions_by_id(baseline_glob[0])
    print(f"[load] baseline: {baseline_glob[0].name} "
          f"({len(baseline_preds)} records)", flush=True)

    score_fn = rouge1_scorer()
    rows = []
    for u in users:
        user_id = u["user_id"]
        rid = str(u["test_record_ids"][0])
        utag = safe_user_tag(user_id)
        pglob = [
            p for p in RESULTS_DIR.glob(
                f"LongLaMP_{temporal_task}_test_*_user{utag}.predictions.jsonl")
            if decode_tag_selector(p.name, args.decode_tag, f"user{utag}")
        ]
        if len(pglob) != 1 or rid not in baseline_preds:
            continue
        pers_preds = load_predictions_by_id(pglob[0])
        if rid not in pers_preds:
            continue

        gold = baseline_preds[rid]["gold"]
        b_pred, p_pred = baseline_preds[rid]["pred"], pers_preds[rid]["pred"]
        g_len = len(str(gold).split())
        b_len, p_len = len(str(b_pred).split()), len(str(p_pred).split())
        ratio_limit = args.length_ratio * g_len
        rows.append({
            "user_id": user_id,
            "test_record_id": rid,
            "gold_words": g_len,
            "baseline_words": b_len,
            "personalized_words": p_len,
            "baseline_rouge1": score_fn(gold, b_pred),
            "personalized_rouge1": score_fn(gold, p_pred),
            # primary (flat word count) -- drives the clean-subset recomputation
            "baseline_degenerate": b_len > args.degenerate_words,
            "personalized_degenerate": p_len > args.degenerate_words,
            # secondary (gold-relative), reported but not acted on
            "baseline_degenerate_ratio": bool(g_len and b_len > ratio_limit),
            "personalized_degenerate_ratio": bool(g_len and p_len > ratio_limit),
        })
    for r in rows:
        r["diff"] = r["personalized_rouge1"] - r["baseline_rouge1"]

    if not rows:
        sys.exit("ERROR: no users resolved — check the predictions globs.")

    clean = [r for r in rows
             if not r["baseline_degenerate"] and not r["personalized_degenerate"]]
    b_deg = [r for r in rows if r["baseline_degenerate"]]

    result = {
        # schema 2 (2026-08-10): added `decode_tag` (None = plain greedy);
        # existing schema-1 results on disk stay valid unchanged.
        "schema_version": 2,
        "tag": args.tag,
        "decode_tag": args.decode_tag or None,
        "temporal_task": temporal_task,
        "split": "test",
        "metric": "rouge1",
        "metric_direction": "higher_is_better",
        "degenerate_words": args.degenerate_words,
        "length_ratio": args.length_ratio,
        "n_users": len(rows),
        "gold_words_median": statistics.median(r["gold_words"] for r in rows),
        "baseline_words_median": statistics.median(r["baseline_words"] for r in rows),
        "personalized_words_median": statistics.median(
            r["personalized_words"] for r in rows),
        "n_baseline_degenerate": sum(r["baseline_degenerate"] for r in rows),
        "n_personalized_degenerate": sum(r["personalized_degenerate"] for r in rows),
        "n_baseline_degenerate_ratio": sum(r["baseline_degenerate_ratio"] for r in rows),
        "n_personalized_degenerate_ratio": sum(
            r["personalized_degenerate_ratio"] for r in rows),
        "corr_length_change_vs_diff": pearson(
            [r["personalized_words"] - r["baseline_words"] for r in rows],
            [r["diff"] for r in rows]),
        "corr_baseline_length_vs_diff": pearson(
            [r["baseline_words"] for r in rows], [r["diff"] for r in rows]),
        "baseline_predictions_file": str(baseline_glob[0]),
        "command": " ".join(sys.argv),
    }
    for label, subset in [("all", rows), ("clean", clean), ("baseline_degenerate", b_deg)]:
        for k, v in summarize([r["diff"] for r in subset]).items():
            result[f"{label}_{k}"] = v
    result.update(provenance)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    with pairs_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n  gold median {result['gold_words_median']:.0f}w | "
          f"baseline median {result['baseline_words_median']:.0f}w | "
          f"personalized median {result['personalized_words_median']:.0f}w")
    print(f"  degenerate (> {args.degenerate_words}w): "
          f"baseline {result['n_baseline_degenerate']}/{len(rows)}, "
          f"personalized {result['n_personalized_degenerate']}/{len(rows)}")
    print(f"  corr(length change, diff) = {result['corr_length_change_vs_diff']:+.3f}")
    for label in ("all", "clean", "baseline_degenerate"):
        n, m, t = (result[f"{label}_n"], result.get(f"{label}_mean_diff"),
                   result.get(f"{label}_t_statistic"))
        if n:
            # t is None in the all-tie case (paired_t_test's guard).
            t_str = f"{t:+.2f}" if t is not None else "n/a"
            print(f"  {label:20s} n={n:3d}  mean={m:+.4f}  t={t_str}")
    print(f"\n[write] {out_path}\n[write] {pairs_path}")


if __name__ == "__main__":
    main()
