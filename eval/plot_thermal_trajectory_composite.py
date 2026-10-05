#!/usr/bin/env python3
"""Single composite figure: one panel per completed E2E run (one per user's C0
run, plus --extra runs placed next to their own user's C0 panel), each on its
own independent axes (duration and throughput both vary too much across runs
for a shared scale to be legible), annotated with cold-start -> plateau iter/s
and the throttle percentage. One shared thermal-state legend.

  python eval/plot_thermal_trajectory_composite.py --agg <agg.json> \
      --out results/ondevice/figures/e2e_thermal_trajectory.pdf
"""
import argparse
import math

from _e2e_plot_style import (apply_rc, load_agg, thermal_trajectory_panels,
                             THERMAL_COLORS, save, plt)
from matplotlib.lines import Line2D

PLATEAU_FRAC = 0.5  # trailing fraction of a run treated as the throttled plateau


def plateau_stats(curve):
    ips = [c["iter_per_sec"] for c in curve]
    start = ips[0]
    n_plateau = max(1, int(len(ips) * PLATEAU_FRAC))
    floor = sum(ips[-n_plateau:]) / n_plateau
    throttle_pct = 100.0 * (1.0 - floor / start)
    mins = curve[-1]["elapsed_s"] / 60
    dur = f"{mins:.0f} min" if mins < 60 else f"{mins / 60:.1f} h"
    return start, floor, throttle_pct, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--extra", nargs="*", default=["u00008075:C2"],
                    metavar="FP:COND", help="extra non-C0 runs, placed next to their "
                    "user's C0 panel (default: XS's C2/unplugged run)")
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--out", default="results/ondevice/figures/e2e_thermal_trajectory.pdf")
    args = ap.parse_args()
    apply_rc()

    agg = load_agg(args.agg)
    extra = [tuple(e.split(":", 1)) for e in args.extra if e]
    panels, _, _ = thermal_trajectory_panels(agg, extra=extra)
    if not panels:
        raise SystemExit("no completed runs to plot")

    ncols = min(args.ncols, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.6 * nrows), squeeze=False)

    states_present = set()
    for i, (title, r) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        curve = [c for c in r["loss_curve"] if c.get("elapsed_s") is not None]
        mins = [c["elapsed_s"] / 60 for c in curve]
        ips = [c["iter_per_sec"] for c in curve]
        cols = [THERMAL_COLORS.get(c.get("thermal_state"), "#7f7f7f") for c in curve]
        states_present |= {c.get("thermal_state") for c in curve}

        ax.plot(mins, ips, color="#bbbbbb", lw=0.6, zorder=1)
        ax.scatter(mins, ips, c=cols, s=7, zorder=2)

        start, floor, throttle_pct, dur = plateau_stats(curve)
        ax.text(0.96, 0.94, f"{throttle_pct:.0f}% throttle\n"
                f"{start:.2f}$\\rightarrow${floor:.2f} iter/s\n{dur}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
                color="dimgrey", bbox=dict(boxstyle="round", fc="white",
                                           ec="lightgrey", alpha=0.85))
        ax.set_title(f"{title}  (n={r['profile_size']})", fontsize=8.5)
        ax.set_xlim(0, max(mins) * 1.05)
        ax.set_ylim(0, max(ips) * 1.18)
        ax.tick_params(labelcolor="dimgrey")
        if i % ncols == 0:
            ax.set_ylabel("iter/s")
        if i // ncols == nrows - 1 or i + ncols >= len(panels):
            ax.set_xlabel("elapsed (min)")

    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    present = [t for t in ("nominal", "fair", "serious", "critical") if t in states_present]
    fig.suptitle("Every profile throttles from a cold nominal burst to a "
                  "serious-state plateau within minutes, then holds it",
                  fontsize=9.5, y=1.02)
    fig.legend([Line2D([0], [0], marker="o", ls="", color=THERMAL_COLORS[t], ms=7)
               for t in present], present, title="thermal", loc="lower center",
               ncol=len(present), bbox_to_anchor=(0.5, -0.05), fontsize=8,
               title_fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save(fig, args.out)


if __name__ == "__main__":
    main()
