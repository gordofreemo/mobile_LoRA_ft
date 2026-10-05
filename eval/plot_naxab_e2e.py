#!/usr/bin/env python3
"""NAX A/B end-to-end figure: one complete User-LoRA training run per arm.

Two panels:
  (a) throughput (iterations/sec) vs elapsed wall-clock — the cost story,
      including how each arm decays as the device heats.
  (b) training loss vs step — the correctness story. Both arms consume an
      IDENTICAL batch sequence (LoRATrain.shuffleSeed), so the curves are
      directly comparable step-by-step and any divergence is attributable to
      the kernel numerics, not to data order.

Panel (b) is the load-bearing one. The ON arm computes backward through
Apple's neural-accelerator matrix units at TF32 precision where OFF uses fp32
on the general shader ALUs, so a small gap is expected; a sharp divergence
would mean the faster kernel is not training the same model.

Usage:
    python eval/plot_naxab_e2e.py results/ondevice/train_bench_metrics_naxab_e2e.jsonl \\
        --out results/ondevice/figures/naxab_e2e_2026-08-XX.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Baseline (stock dispatch) vs optimized (NAX). DIVERGING_COLORBLIND members.
COLOR = {"off": "#4575b4", "on": "#d73027"}
STYLE = {"off": "-", "on": "--"}  # never rely on colour alone
LABEL = {"off": "baseline (stock MLX)", "on": "NAX backward (patched)"}


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    tr = [r for r in rows if r.get("record_type") == "train" and r.get("nax_arm")]
    if not tr:
        raise SystemExit(f"no arm-tagged train records in {path}")
    df = pd.DataFrame(tr).dropna(subset=["elapsed_s", "iter_per_sec", "training_loss"])
    return df, rows


def roll(y, frac=0.06):
    """Rolling mean over a fraction of the series; both traces are noisy —
    throughput from thermal jitter, loss from per-example variance."""
    w = max(3, int(len(y) * frac))
    return pd.Series(y).rolling(w, center=True, min_periods=1).mean().values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default="results/ondevice/figures/naxab_e2e.pdf")
    args = ap.parse_args()

    df, _ = load(args.jsonl)
    arms = [a for a in ("off", "on") if a in set(df["nax_arm"])]

    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # --- Plot ---
    summary = {}
    for arm in arms:
        d = df[df["nax_arm"] == arm].sort_values("step")
        hrs = d["elapsed_s"].values / 3600.0
        ips, loss, step = d["iter_per_sec"].values, d["training_loss"].values, d["step"].values

        axes[0].plot(hrs, ips, color=COLOR[arm], alpha=0.22, linewidth=0.9, zorder=2)
        axes[0].plot(hrs, roll(ips), color=COLOR[arm], linestyle=STYLE[arm],
                     linewidth=2.4, zorder=4, label=LABEL[arm])

        axes[1].plot(step, loss, color=COLOR[arm], alpha=0.20, linewidth=0.9, zorder=2)
        axes[1].plot(step, roll(loss), color=COLOR[arm], linestyle=STYLE[arm],
                     linewidth=2.4, zorder=4, label=LABEL[arm])

        tail = max(1, len(loss) // 20)  # last 5% of reports
        summary[arm] = {
            "hours": hrs[-1],
            "iters": int(d["step"].max()),
            "final_loss": float(np.mean(loss[-tail:])),
            "mean_ips": float(np.mean(ips)),
            "end_xy": (hrs[-1], roll(ips)[-1]),
        }

    # --- Panel (a): throughput ---
    ax = axes[0]
    ax.set_xlabel("Elapsed wall-clock (hours)", fontsize=12, labelpad=8, color="dimgrey")
    ax.set_ylabel("Throughput (iterations / sec)", fontsize=12, labelpad=8, color="dimgrey")
    ax.set_title(r"$\bf{(a)}$" + "  Training throughput", loc="left", fontsize=12, pad=7,
                 color="dimgrey")
    # Total wall-clock above each curve's endpoint.
    for arm in arms:
        ax.annotate(f"{summary[arm]['hours']:.2f} h",
                    xy=summary[arm]["end_xy"], xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, weight="medium", color="dimgrey")

    # --- Panel (b): shade the gap between the two loss curves ---
    # Self-revealing: invisible when the arms agree, unmissable when they do
    # not. The question this figure exists to answer is whether the faster
    # kernel trains the same model, so the divergence gets its own ink rather
    # than leaving the reader to eyeball two overlapping lines.
    if len(arms) == 2:
        a = df[df["nax_arm"] == "off"].sort_values("step")
        b = df[df["nax_arm"] == "on"].sort_values("step")
        grid = np.union1d(a["step"].values, b["step"].values)
        la = np.interp(grid, a["step"].values, roll(a["training_loss"].values))
        lb = np.interp(grid, b["step"].values, roll(b["training_loss"].values))
        axes[1].fill_between(grid, la, lb, color="dimgrey", alpha=0.18, zorder=1,
                             label="divergence between arms")
        summary["max_gap"] = float(np.max(np.abs(la - lb)))

    # --- Panel (b): loss ---
    ax = axes[1]
    ax.set_xlabel("Training step", fontsize=12, labelpad=8, color="dimgrey")
    ax.set_ylabel("Training loss", fontsize=12, labelpad=8, color="dimgrey")
    ax.set_title(r"$\bf{(b)}$" + "  Training loss (identical batch order)",
                 loc="left", fontsize=12, pad=7, color="dimgrey")
    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)
        ax.legend(loc="lower left", fontsize=10, frameon=True, facecolor="white",
                  framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey")

    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    # --- Save ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")
    for arm in arms:
        s = summary[arm]
        print(f"  {arm:>3}: {s['iters']} iters in {s['hours']:.3f} h, "
              f"mean {s['mean_ips']:.4f} it/s, final loss {s['final_loss']:.4f}")
    if "max_gap" in summary:
        print(f"  max loss divergence between arms: {summary['max_gap']:.4f}")


if __name__ == "__main__":
    main()
