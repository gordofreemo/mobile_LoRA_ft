#!/usr/bin/env python3
"""Figure for the OPPU faithful-replication round: base model vs task adapter
vs personalized (task + per-user) under OPPU's protocol, per task.

Left panel: the five higher-is-better tasks (accuracy / ROUGE-1).
Right panel: LaMP-3 MAE (lower is better) on its own axis — never share an
axis between opposite-direction metrics. Citation is omitted from the figure
(both arms score exactly 0 due to the release's prompt defect; the writeup
table carries it).

Inputs: results/oppu_rep/score_<task>.json (task/oppu arms) and
results/oppu_rep/<task>/base_k1_preds.json (base arm, scored via their
evaluator). Output: results/oppu_rep/figures/oppu_rep_scores.{pdf,png}.

Two phases so the expensive part never runs on a login node (2 GB cgroup cap):
  --score-only   score the base arms -> base_headlines.json  (Condor CPU job)
  --plot-only    render the figure from base_headlines.json + score_*.json
                 (cheap; fine anywhere)
Default = both (in-job use).
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from oppu_rep_score import (TASKS, OUT, ensure_metric_cache, load_arm_preds,
                            their_headline)

# validated reference palette, first three categorical slots (light mode)
C_BASE, C_TASK, C_PERS = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"

UP_TASKS = [  # (their name, display name, metric key in score json, pretty metric)
    ("movie_tagging", "LaMP-2M\nmovie tags", "accuracy", "acc"),
    ("news_categorize", "LaMP-2N\nnews cat.", "accuracy", "acc"),
    ("news_headline", "LaMP-4\nheadlines", "rouge_1", "R-1"),
    ("scholarly_title", "LaMP-5\ntitles", "rouge_1", "R-1"),
    ("tweet_paraphrase", "LaMP-7\ntweets", "rouge_1", "R-1"),
]
THEIR_KEY = {"accuracy": "accuracy", "rouge_1": "rouge-1", "MAE": "MAE"}


def base_headlines():
    ensure_metric_cache()
    out = {}
    for task, (lamp_id, _) in TASKS.items():
        if task == "citation":
            continue
        preds = load_arm_preds(task, "base")
        out[task] = their_headline(task, lamp_id, preds)
    return out


def score_base_arms():
    base = base_headlines()
    with open(OUT / "base_headlines.json", "w") as f:
        json.dump(base, f, indent=2)
    for t, h in base.items():
        print(f"base {t}: {h}")
    return base


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = {}
    for task in TASKS:
        p = OUT / f"score_{task}.json"
        if p.exists():
            scores[task] = json.load(open(p))
    bh = OUT / "base_headlines.json"
    if not bh.exists():
        sys.exit("base_headlines.json missing — run --score-only in a Condor job first")
    base = json.load(open(bh))

    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [5.2, 1.15]})
    fig.patch.set_facecolor("white")

    w = 0.26
    xs = range(len(UP_TASKS))
    for ax_ in (ax, axm):
        ax_.set_facecolor("white")
        ax_.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
        for s in ("top", "right"):
            ax_.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax_.spines[s].set_color("#c8c7c0")
        ax_.tick_params(colors=MUTED, labelsize=8)

    def bars(ax_, x, vals, labelfmt="%.2f"):
        for dx, v, c in zip((-w, 0.0, w), vals, (C_BASE, C_TASK, C_PERS)):
            ax_.bar(x + dx, v, width=w - 0.03, color=c, zorder=3)
            ax_.text(x + dx, v + 0.012, labelfmt % v, ha="center", va="bottom",
                     fontsize=6.6, color=MUTED)

    for i, (task, disp, key, pretty) in enumerate(UP_TASKS):
        s = scores[task]
        vals = [base[task][THEIR_KEY[key]], s[f"task_{key}"], s[f"oppu_{key}"]]
        bars(ax, i, vals)
        ax.text(i, -0.145, f"({pretty})", ha="center", va="top", fontsize=7,
                color=MUTED, transform=ax.get_xaxis_transform())

    ax.set_xticks(list(xs))
    ax.set_xticklabels([d for _, d, _, _ in UP_TASKS], fontsize=8, color=INK)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("score  (higher is better)", fontsize=8.5, color=INK)

    s3 = scores["product_rating"]
    bars(axm, 0, [base["product_rating"]["MAE"], s3["task_MAE"], s3["oppu_MAE"]])
    axm.set_xticks([0])
    axm.set_xticklabels(["LaMP-3\nratings"], fontsize=8, color=INK)
    axm.text(0, -0.145, "(MAE)", ha="center", va="top", fontsize=7, color=MUTED,
             transform=axm.get_xaxis_transform())
    axm.set_ylim(0, max(base["product_rating"]["MAE"], s3["task_MAE"]) * 1.5)
    axm.set_xlim(-0.62, 0.62)
    axm.set_ylabel("MAE  (lower is better)", fontsize=8.5, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in (C_BASE, C_TASK, C_PERS)]
    ax.legend(handles,
              ["base model + retrieval",
               "+ task adapter",
               "+ task & per-user adapter"],
              loc="upper left", frameon=False, fontsize=8, ncol=1,
              handlelength=1.1, handleheight=1.0, borderaxespad=0.1)

    fig.tight_layout()
    outdir = OUT / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"oppu_rep_scores.{ext}", dpi=220,
                    bbox_inches="tight")
    print(f"wrote {outdir}/oppu_rep_scores.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-only", action="store_true",
                    help="score the base arms only (heavy — Condor job)")
    ap.add_argument("--plot-only", action="store_true",
                    help="render from existing base_headlines.json (cheap)")
    args = ap.parse_args()
    if not args.plot_only:
        score_base_arms()
    if not args.score_only:
        render()


if __name__ == "__main__":
    main()
