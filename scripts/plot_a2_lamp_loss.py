#!/usr/bin/env python3
"""
Plot the A2-lamp training loss curve from metrics.jsonl.

Usage:
    python scripts/plot_a2_lamp_loss.py
    python scripts/plot_a2_lamp_loss.py --metrics <path> --out-stem <path> --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Single-hue (blue) ramp, light -> dark: raw noisy loss gets the light tint,
# the smoothed trend gets the dark step. Both are the same underlying
# measurement at two levels of smoothing, not two different series, so one
# hue ramp is the correct encoding (see dataviz skill: sequential = one hue).
RAW_COLOR = "#86b6ef"    # ramp step 250
SMOOTH_COLOR = "#1c5cab"  # ramp step 550


def ema(values, alpha=0.05):
    out = []
    m = values[0]
    for v in values:
        m = alpha * v + (1 - alpha) * m
        out.append(m)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metrics",
        default=str(PROJECT_ROOT / "train/checkpoints/a2_lamp_1ep_seed0/metrics.jsonl"),
    )
    p.add_argument(
        "--out-stem",
        default=str(PROJECT_ROOT / "results/figures/a2_lamp_loss_2026-07-13"),
        help="writes <stem>.pdf and <stem>.png",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    pdf_path = Path(args.out_stem + ".pdf")
    png_path = Path(args.out_stem + ".png")
    if not args.overwrite and (pdf_path.exists() or png_path.exists()):
        print(f"ERROR: refusing to overwrite {pdf_path} / {png_path}. "
              f"Pass --overwrite.", file=sys.stderr)
        sys.exit(1)

    steps, losses = [], []
    with open(args.metrics) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "loss" not in rec:
                continue
            steps.append(rec["step"])
            losses.append(rec["loss"])

    if not steps:
        print("ERROR: no loss records found in metrics file.", file=sys.stderr)
        sys.exit(1)

    smoothed = ema(losses, alpha=0.05)

    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=150)
    ax.plot(steps, losses, color=RAW_COLOR, linewidth=1.0, alpha=0.85,
            label="per-step loss (logged every 10 steps)")
    ax.plot(steps, smoothed, color=SMOOTH_COLOR, linewidth=2.0,
            label="EMA-smoothed ($\\alpha$=0.05)")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Full LaMP Dataset Loss Curve (1 epoch, 2,250 steps)")

    # Recessive grid/axes: light horizontal gridlines only, thin spines.
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#888888")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors="#444444")

    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(f"n={len(steps)} points; first_loss={losses[0]:.4f} "
          f"final_loss={losses[-1]:.4f} min_loss={min(losses):.4f}")
    print(f"written -> {pdf_path}")
    print(f"           {png_path}")


if __name__ == "__main__":
    main()
