#!/usr/bin/env python3
"""Throughput throttle over full runs: iter/s vs elapsed, points colored by
thermal state. Each cold-start collapses to a throttled plateau within minutes.

Emits ONE SEPARATE plot per completed run (one per user's C0 run, plus any
--extra runs), all with IDENTICAL x and y axes (so they line up when placed
together), each with its own y-ticks/label. Output files get a per-panel
suffix, e.g. e2e_thermal_trajectory_S.pdf, e2e_thermal_trajectory_S-C2.pdf.

  python eval/plot_thermal_trajectory.py --agg <agg.json> \
      --out results/ondevice/figures/e2e_thermal_trajectory.pdf
  # single user only, no extras:
  python eval/plot_thermal_trajectory.py --agg <agg.json> --user u00008075 \
      --extra "" --out <...>.pdf
"""
import argparse

from _e2e_plot_style import (apply_rc, load_agg,
                             thermal_trajectory_panels, THERMAL_COLORS, save, plt)
from matplotlib.lines import Line2D


def one_plot(title, run, xmax, ymax, out):
    curve = [c for c in run["loss_curve"] if c.get("elapsed_s") is not None]
    mins = [c["elapsed_s"] / 60 for c in curve]
    ips = [c["iter_per_sec"] for c in curve]
    cols = [THERMAL_COLORS.get(c.get("thermal_state"), "#7f7f7f") for c in curve]

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(mins, ips, color="#bbbbbb", lw=0.7, zorder=1)
    ax.scatter(mins, ips, c=cols, s=8, zorder=2)
    ax.set_xlabel("elapsed (min)")
    ax.set_ylabel("iter/s")
    ax.set_title(title)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    present = [t for t in ("nominal", "fair", "serious", "critical")
               if any(c.get("thermal_state") == t for c in curve)]
    ax.legend([Line2D([0], [0], marker="o", ls="", color=THERMAL_COLORS[t], ms=6)
               for t in present], present, title="thermal", loc="upper right",
              fontsize=7, title_fontsize=7)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--user", default=None, help="single fingerprint (default: all C0 runs)")
    ap.add_argument("--extra", nargs="*", default=["u00008075:C2"],
                    metavar="FP:COND", help="extra non-C0 runs to add as their own panel "
                    "(default: S's C2/unplugged run); pass --extra with no values for none")
    ap.add_argument("--out", default="results/ondevice/figures/e2e_thermal_trajectory.pdf")
    args = ap.parse_args()
    apply_rc()

    agg = load_agg(args.agg)
    extra = [tuple(e.split(":", 1)) for e in args.extra if e]
    panels, xmax, ymax = thermal_trajectory_panels(agg, extra=extra)
    if args.user:
        panels = [(t, r) for t, r in panels if r["user_fingerprint"] == args.user]
    if not panels:
        raise SystemExit("no completed runs to plot")

    base, ext = args.out.rsplit(".", 1)
    for title, r in panels:
        one_plot(title, r, xmax, ymax, f"{base}_{title}.{ext}")


if __name__ == "__main__":
    main()
