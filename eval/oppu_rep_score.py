#!/usr/bin/env python3
"""Scoring + stats for the OPPU faithful-replication round. Login-host only.

Headline metrics come from THEIR code (third_party/OPPU/eval/evaluation.py,
the LaMP-official metrics, imported and called unmodified). On top, a
per-query score layer that mirrors their metric mapping exactly (validated:
its aggregate must reproduce their number or the script exits), which feeds
the two pre-registered stats layers:

  - PRIMARY ("their style"): query-level paired t + Wilcoxon on
    OPPU+RAG minus RAG, all queries pooled, uncorrected.
  - SECONDARY (honesty layer): per-user grouped — mean per-query score per
    user, paired across users; Bonferroni across 7 tasks is applied by the
    reader (raw p reported).

Usage:
  .venv-audit/bin/python eval/oppu_rep_score.py --task movie_tagging [--overwrite]
  .venv-audit/bin/python eval/oppu_rep_score.py --all [--overwrite]

Inputs (from run_task_lora.py / run_oppu.py):
  results/oppu_rep/<task>/task_k1_preds.json          RAG arm
  results/oppu_rep/<task>/oppu_k1_u*_preds.json       OPPU+RAG arm (shards)
Output: results/oppu_rep/score_<task>.json (flat) + printed summary.
"""

import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "OPPU" / "eval"))

RELEASE = PROJECT_ROOT / "data" / "oppu_release" / "data"
OUT = PROJECT_ROOT / "results" / "oppu_rep"

TASKS = {  # their task_name -> (LaMP task id, test-users file)
    "citation": ("LaMP_1", "user_top_100_history.json"),
    "movie_tagging": ("LaMP_2M", "user_top_100_history.json"),
    "news_categorize": ("LaMP_2N", "user_top_100_history.json"),
    "news_headline": ("LaMP_4", "user_top_100_history.json"),
    "product_rating": ("LaMP_3", "user_top_100_history.json"),
    "scholarly_title": ("LaMP_5", "user_top_100_history.json"),
    "tweet_paraphrase": ("LaMP_7", "user_more_100_history.json"),
}


def ensure_metric_cache():
    """Their evaluation.py sets HF_EVALUATE_OFFLINE=1 at import time; pre-cache
    every metric it needs BEFORE that import so offline loads succeed."""
    assert "evaluation" not in sys.modules, "must run before importing their module"
    os.environ.pop("HF_EVALUATE_OFFLINE", None)
    import evaluate as hf_evaluate
    for m in ("f1", "accuracy", "mse", "mae", "rouge"):
        hf_evaluate.load(m)


def provenance():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return "unknown"
    return {"git_commit": git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "hostname": socket.gethostname(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def load_arm_preds(task, arm, smoke=False):
    """Return {id: output}. For the oppu arm, merge non-smoke shards."""
    d = OUT / task
    if arm == "task":
        files = [d / ("task_k1_limit1t30_preds.json" if smoke else "task_k1_preds.json")]
    elif smoke:
        files = sorted(d.glob("oppu_k1_smoke_u*_preds.json"))
    else:
        files = sorted(p for p in d.glob("oppu_k1_u*_preds.json")
                       if "smoke" not in p.name)
    preds = {}
    for p in files:
        with open(p) as f:
            j = json.load(f)
        for rec in j["golds"]:
            if str(rec["id"]) in preds:
                print(f"FATAL: duplicate prediction id {rec['id']} in {p}")
                sys.exit(1)
            preds[str(rec["id"])] = rec["output"]
    if not preds:
        print(f"FATAL: no predictions for {task}/{arm}")
        sys.exit(1)
    return preds


def their_headline(task, lamp_id, preds, restrict_ids=None):
    """Run their LaMPEvaluation on a prediction dict, unmodified.

    restrict_ids (smoke only): their evaluator asserts gold ids == pred ids,
    so a partial-coverage smoke run needs the gold file cut down to match.
    """
    from evaluation import LaMPEvaluation
    label_file = next(iter((RELEASE / task).glob("*history_label.json")))
    with open(label_file) as f:
        lab = json.load(f)
    # RELEASE INCONSISTENCY (recorded in the audit): the shipped label files
    # carry task ids their own evaluator does not dispatch on — movie_tagging
    # says "LaMP_8" and news_categorize "LaMP_2", which would fall through to
    # the ROUGE metric instead of accuracy/F1. Normalize to the code's id.
    lab["task"] = lamp_id
    if restrict_ids is not None:
        lab["golds"] = [g for g in lab["golds"] if str(g["id"]) in restrict_ids]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(lab, f)
        label_file = f.name
    ev = LaMPEvaluation(single_gold_json_file_addr=str(label_file))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"task": lamp_id, "model": "x",
                   "golds": [{"id": i, "output": o} for i, o in preds.items()]}, f)
        tmp = f.name
    try:
        return ev.evaluate_task(tmp, lamp_id)
    finally:
        os.unlink(tmp)


def per_query_scores(lamp_id, preds, golds):
    """Mirror their metric mapping per query. Higher-is-better except mae."""
    from evaluation import create_metric_f1_accuracy  # noqa: F401 (parity import)
    if lamp_id in ("LaMP_1", "LaMP_2N", "LaMP_2M"):
        from evaluation import LaMPEvaluation
        labels = LaMPEvaluation.__dict__["_get_labels"](None, lamp_id)

        def mapping(x):
            try:
                return labels.index(str(x).strip())
            except Exception:
                return -1
        return {"metric": "accuracy",
                "scores": {i: {"accuracy": float(mapping(preds[i]) == mapping(golds[i])
                                                 and mapping(golds[i]) != -1)}
                           for i in preds}}
    if lamp_id == "LaMP_3":
        def mapping(x, y):
            try:
                return float(str(x).strip())
            except Exception:
                y = float(str(y).strip())
                return 1.0 if abs(1 - y) > abs(5 - y) else 5.0
        return {"metric": "mae",
                "scores": {i: {"mae": abs(mapping(preds[i], golds[i])
                                          - mapping(golds[i], golds[i]))}
                           for i in preds}}
    # their stack, per-example: same evaluate rouge metric, aggregation off.
    # (their headline uses the default bootstrap aggregator whose "mid" is
    # not exactly the plain mean — validation tolerance is loosened for rouge)
    import evaluate as hf_evaluate
    rouge = hf_evaluate.load("rouge")
    ids = sorted(preds)
    res = rouge.compute(predictions=[str(preds[i]).strip() for i in ids],
                        references=[[str(golds[i]).strip()] for i in ids],
                        use_aggregator=False)
    out = {i: {"rouge1": float(res["rouge1"][j]), "rougeL": float(res["rougeL"][j])}
           for j, i in enumerate(ids)}
    return {"metric": "rouge1", "scores": out}


def paired_stats(diffs):
    import statistics
    from scipy import stats as st
    n = len(diffs)
    mean = sum(diffs) / n
    if all(d == 0 for d in diffs):
        return {"n": n, "mean_diff": mean, "t_p": None, "wilcoxon_p": None,
                "note": "all diffs zero"}
    t = st.ttest_1samp(diffs, 0.0)
    try:
        w = st.wilcoxon(diffs)
        w_p = float(w.pvalue)
    except ValueError:
        w_p = None
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    ci = 1.96 * sd / (n ** 0.5) if n > 1 else 0.0
    return {"n": n, "mean_diff": mean, "ci95_lo": mean - ci, "ci95_hi": mean + ci,
            "t_p": float(t.pvalue), "wilcoxon_p": w_p,
            "wins": sum(1 for d in diffs if d > 0),
            "ties": sum(1 for d in diffs if d == 0),
            "losses": sum(1 for d in diffs if d < 0)}


def score_task(task, overwrite, smoke=False):
    lamp_id, test_fn = TASKS[task]
    out_path = OUT / (f"score_{task}_smoke.json" if smoke else f"score_{task}.json")
    if out_path.exists() and not overwrite:
        print(f"REFUSING to overwrite {out_path} (pass --overwrite)")
        sys.exit(1)

    with open(RELEASE / task / test_fn) as f:
        test_data = json.load(f)
    gold_by_id = {str(q["id"]): str(q["gold"]) for u in test_data for q in u["query"]}
    user_by_id = {str(q["id"]): str(u["user_id"]) for u in test_data for q in u["query"]}

    arms = {}
    for arm in ("task", "oppu"):
        preds = load_arm_preds(task, arm, smoke=smoke)
        if smoke:
            gold_by_id = {i: g for i, g in gold_by_id.items() if i in preds}
        missing = set(gold_by_id) - set(preds)
        if missing:
            print(f"FATAL: {task}/{arm} missing {len(missing)} predictions "
                  f"(e.g. {sorted(missing)[:5]}) — shards incomplete?")
            sys.exit(1)
        preds = {i: preds[i] for i in gold_by_id}
        headline = their_headline(task, lamp_id, preds,
                                  restrict_ids=set(gold_by_id) if smoke else None)
        pq = per_query_scores(lamp_id, preds, gold_by_id)
        # validation: our per-query aggregate must reproduce their number
        key0 = pq["metric"]
        ours = sum(v[key0 if key0 != "mae" else "mae"] for v in pq["scores"].values()) / len(pq["scores"])
        theirs_key = {"accuracy": "accuracy", "mae": "MAE", "rouge1": "rouge-1"}[key0]
        theirs = headline[theirs_key]
        tol = 5e-3 if key0 == "rouge1" else 1e-6   # rouge headline = bootstrap mid
        if abs(ours - theirs) > tol:
            print(f"FATAL: per-query layer diverges from their metric for "
                  f"{task}/{arm}: ours {ours:.6f} vs theirs {theirs:.6f}")
            sys.exit(1)
        arms[arm] = {"headline": headline, "pq": pq}

    key = arms["task"]["pq"]["metric"]
    sign = -1.0 if key == "mae" else 1.0     # mae: lower is better → flip diffs
    ids = sorted(gold_by_id)
    diffs = [sign * (arms["oppu"]["pq"]["scores"][i][key]
                     - arms["task"]["pq"]["scores"][i][key]) for i in ids]
    primary = paired_stats(diffs)

    by_user = {}
    for i, d in zip(ids, diffs):
        by_user.setdefault(user_by_id[i], []).append(d)
    user_means = [sum(v) / len(v) for v in by_user.values()]
    secondary = paired_stats(user_means)

    r = {"task": task, "lamp_id": lamp_id, "metric": key,
         "diff_direction": "positive = OPPU+RAG better (mae sign-flipped)",
         "n_queries": len(ids), "n_users": len(by_user)}
    for arm in ("task", "oppu"):
        for mk, mv in arms[arm]["headline"].items():
            r[f"{arm}_{mk.replace('-', '_')}"] = mv
    for k2, v in primary.items():
        r[f"query_level_{k2}"] = v
    for k2, v in secondary.items():
        r[f"user_grouped_{k2}"] = v
    r.update(provenance())

    r["smoke"] = smoke
    with open(out_path, "w") as f:
        json.dump(r, f, indent=2)
    pairs_path = OUT / (f"score_{task}_smoke.pairs.jsonl" if smoke else f"score_{task}.pairs.jsonl")
    with open(pairs_path, "w") as f:
        for i, d in zip(ids, diffs):
            f.write(json.dumps({"id": i, "user_id": user_by_id[i], "diff": d,
                                "task_score": arms["task"]["pq"]["scores"][i][key],
                                "oppu_score": arms["oppu"]["pq"]["scores"][i][key]}) + "\n")
    print(f"== {task} ({lamp_id}, {key}) n_q={len(ids)} n_u={len(by_user)}")
    print(f"   RAG arm:      {arms['task']['headline']}")
    print(f"   OPPU+RAG arm: {arms['oppu']['headline']}")
    print(f"   query-level:  mean_diff {primary['mean_diff']:+.4f} "
          f"t_p {primary.get('t_p')} w_p {primary.get('wilcoxon_p')} "
          f"W/T/L {primary.get('wins')}/{primary.get('ties')}/{primary.get('losses')}")
    print(f"   user-grouped: mean_diff {secondary['mean_diff']:+.4f} "
          f"t_p {secondary.get('t_p')} w_p {secondary.get('wilcoxon_p')} "
          f"W/T/L {secondary.get('wins')}/{secondary.get('ties')}/{secondary.get('losses')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="score the smoke artifacts (1 test user) — validates the pipeline")
    args = ap.parse_args()
    tasks = sorted(TASKS) if args.all else ([args.task] if args.task else None)
    if not tasks:
        ap.error("--task or --all required")
    p = provenance()
    print(f"[oppu_rep_score] tasks={tasks} smoke={args.smoke} commit={p['git_commit']} host={p['hostname']}")
    ensure_metric_cache()
    for t in tasks:
        score_task(t, args.overwrite, smoke=args.smoke)


if __name__ == "__main__":
    main()
