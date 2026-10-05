#!/usr/bin/env python3
"""Extrapolation: cumulative device-hours to train the top-100 users on-device.

Sorts the 100 users cheapest-first (by predicted wall = token_slope · tokens) and
plots cumulative device-hours vs users trained. Endpoint annotated in days.

  python eval/plot_cost_extrapolation.py \
      --tokens data/lamp_user_stats/LaMP_3_top100_token_counts.json \
      --out results/ondevice/figures/e2e_cost_extrapolation.pdf
"""
import argparse
import json
from pathlib import Path

from _e2e_plot_style import apply_rc, save, plt, TOKEN_SLOPE_S, token_slope, load_agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="data/lamp_user_stats/LaMP_3_top100_token_counts.json")
    ap.add_argument("--agg", default=None, help="aggregate JSON — use its measured token slope")
    ap.add_argument("--out", default="results/ondevice/figures/e2e_cost_extrapolation.pdf")
    args = ap.parse_args()
    apply_rc()

    slope = token_slope(load_agg(args.agg)) if args.agg else TOKEN_SLOPE_S
    d = json.loads(Path(args.tokens).read_text())
    hrs = sorted(slope * u["total_tokens_3ep"] / 3600.0 for u in d["users"])
    cum, s = [], 0.0
    for h in hrs:
        s += h
        cum.append(s)
    n = list(range(1, len(cum) + 1))
    total = cum[-1]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.fill_between(n, cum, color="#0072B2", alpha=0.12)
    ax.plot(n, cum, color="#0072B2", lw=1.8)
    ax.scatter([n[-1]], [total], s=30, color="#0072B2", zorder=3)
    ax.annotate(f"{total:.0f} h\n({total/24:.0f} d)", (n[-1], total),
                textcoords="offset points", xytext=(-8, 6), ha="right", fontsize=8)
    ax.set_xlabel("users trained")
    ax.set_ylabel("cumulative device-hours")
    ax.set_title("cost to train all 100 on-device")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, total * 1.08)
    save(fig, args.out)


if __name__ == "__main__":
    main()
