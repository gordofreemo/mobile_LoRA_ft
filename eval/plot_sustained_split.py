#!/usr/bin/env python3
"""Sustained-training figures, as two independent single-column plots.

  thermal_collapse.pdf   throughput / own cold reference vs elapsed, five
                         profiles overlaid. Normalising by each run's own cold
                         start removes the absolute iter/s spread (which tracks
                         profile token length, not thermal behaviour) so the
                         five curves become directly comparable: every profile
                         falls to roughly half within minutes and then holds.
  thermal_recovery.pdf   throughput / cold reference vs minutes since a burst
                         ended, points coloured by reported thermalState. The
                         point is the dissociation: throughput is back long
                         before ProcessInfo says "nominal".

Both figures characterise the patched (NAX-on) configuration of sec 5.4.

    .venv-mlx/bin/python eval/plot_sustained_split.py \
        --e2e results/ondevice_e2e_smollm3_a1lamp_nax-on_2026-08-13.json \
        --thermal results/ondevice_thermal_smollm3_4bit_nax-on_2026-08-14.json \
        --outdir ../overleafs/mobile_FT_paper/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Profile-size labels, ascending. Kept identical to _e2e_plot_style.USER_LABELS
# so a reader moving between figures sees the same names for the same users.
from _e2e_plot_style import (  # noqa: E402
    USER_LABELS, USER_COLORS, THERMAL_COLORS, apply_rc,
)
THERMAL_ORDER = ["nominal", "fair", "serious", "critical"]

LEG = dict(frameon=False, borderpad=0.4, handlelength=1.5, labelspacing=0.35)
PLATEAU_FRAC = 0.5  # trailing fraction treated as the throttled plateau


def roll(y, frac=0.04):
    """Centred rolling mean over the plateau, raw values over the leading edge.

    A centred window cannot be used at the start of these runs: throughput is
    collapsing steeply there, so averaging point 0 with the points after it
    drags the cold start well below its measured value (XXL rendered at 0.68
    instead of 1.0). Shortening the window does not fix it -- the distortion is
    driven by the gradient, not the window. The first w samples are therefore
    left raw, which costs nothing: the leading edge is a clean monotone drop,
    and the sampling noise this function exists to suppress is in the
    plateau."""
    w = max(3, int(len(y) * frac))
    sm = pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy(copy=True)
    sm[:w] = y[:w]
    return sm


def style(ax):
    """House style, matching _e2e_plot_style.apply_rc used by the other figures."""
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def collapse(e2e_path, out):
    agg = json.loads(Path(e2e_path).read_text())
    runs = [r for r in agg["runs"]
            if r.get("completed") and str(r["condition"]).startswith("C0")]
    runs.sort(key=lambda r: r["profile_size"])

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.axhline(1.0, color="#444444", lw=0.8, ls="--", zorder=2)

    for r in runs:
        curve = [c for c in r["loss_curve"] if c.get("elapsed_s") is not None]
        mins = np.array([c["elapsed_s"] / 60.0 for c in curve])
        ips = np.array([c["iter_per_sec"] for c in curve])
        cold = ips[0]
        rel = ips / cold
        n = max(1, int(len(rel) * PLATEAU_FRAC))
        floor = rel[-n:].mean()
        fp = r["user_fingerprint"]
        m = mins > 0                      # log x cannot show t=0
        ax.plot(mins[m], rel[m], color=USER_COLORS[fp], lw=0.6, alpha=0.22,
                zorder=3)
        ax.plot(mins[m], roll(rel)[m], color=USER_COLORS[fp], lw=1.4,
                alpha=0.95, zorder=4,
                label=f"{USER_LABELS[fp]} ($n$={r['profile_size']})")
        print(f"  {USER_LABELS[fp]:>3} n={r['profile_size']:>3}  "
              f"{cold:.3f}->{cold*floor:.3f} iter/s  "
              f"throttle {100*(1-floor):.0f}%  {mins[-1]/60:.1f} h")

    ax.text(0.015, 1.0 + 0.018, "cold start", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=7, color="#1A1A1A")
    # Linear, not log: the claim this figure carries is "half the throughput,
    # held for the rest of the run", and a log axis shrinks the plateau -- the
    # dominant feature -- to a tail. The decay timescale it would buy back is
    # not needed here; thermal_recovery.pdf carries the response-time story and
    # the duty-cycle verdict rests on the soak probes, not on this plot. The
    # collapse is consequently a near-vertical step at the origin, so its
    # timescale belongs in the prose as a number.
    ax.set_xlabel("elapsed wall-clock (min)")
    ax.set_xlim(-10, None)
    ax.set_ylabel("throughput / own cold start")
    ax.set_ylim(0.25, 1.52)   # headroom so the legend clears the cold-start line
    style(ax)
    ax.legend(loc="upper right", ncol=2, columnspacing=0.9, **LEG)
    sns.despine()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), bbox_inches="tight",
                    dpi=200 if ext == "png" else None)


def recovery(thermal_path, out, session=None):
    agg = json.loads(Path(thermal_path).read_text())
    sess = [s for s in agg["sessions"]
            if s.get("complete") and s.get("t95_s") is not None]
    if session:
        sess = [s for s in sess if s["bench_session_id"] == session]
    s = sess[0]
    curve = agg["curves"][s["bench_session_id"]]
    cold = s["cold_ref_seconds_per_iter"]
    hot = s["soak_plateau_seconds_per_iter"]
    t95 = s["t95_s"]  # reported below, no longer drawn

    xs = np.array([c["cooldown_elapsed_s"] / 60.0 for c in curve])
    ys = np.array([cold / c["seconds_per_iter"] for c in curve])

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot(xs, ys, "-", color="#bbbbbb", lw=0.7, zorder=1)
    for st in THERMAL_ORDER:
        m = [c["thermal_state"] == st for c in curve]
        if any(m):
            ax.scatter(xs[m], ys[m], s=11, color=THERMAL_COLORS[st],
                       edgecolors="none", zorder=3, label=st)

    ax.axhline(1.0, color="#444444", lw=0.8, ls="--", zorder=2)
    ax.axhline(cold / hot, color="#7f7f7f", lw=0.8, ls=":", zorder=2)
    ax.text(0.015, 1.0 + 0.012, "cold reference",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=7, color="#1A1A1A")
    ax.text(0.015, cold / hot - 0.015, "throttled steady state",
            transform=ax.get_yaxis_transform(), ha="left", va="top",
            fontsize=7, color="#1A1A1A")

    ax.set_xlabel("minutes since the training burst ended")
    ax.set_ylabel("throughput / cold reference")
    ax.set_ylim(0.35, 1.10)
    ax.set_xlim(-2.5, max(xs) + 2.5)
    style(ax)
    ax.legend(loc="center right", ncol=1, columnspacing=0.8,
              title="reported thermalState", title_fontsize=8, **LEG)
    sns.despine()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    print(f"  cold {cold:.3f} s/iter, plateau {hot:.3f} s/iter, "
          f"R={hot/cold:.3f}x, floor={cold/hot:.3f}, "
          f"t50={s['t50_s']:.0f}s t95={t95:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", required=True)
    ap.add_argument("--thermal", required=True)
    ap.add_argument("--session", default=None)
    ap.add_argument("--outdir", default="results/ondevice/figures")
    a = ap.parse_args()

    apply_rc()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("thermal_collapse:")
    collapse(a.e2e, out / "thermal_collapse.pdf")
    print("thermal_recovery:")
    recovery(a.thermal, out / "thermal_recovery.pdf", a.session)
    print(f"wrote {out}/thermal_collapse.pdf and {out}/thermal_recovery.pdf")


if __name__ == "__main__":
    main()
