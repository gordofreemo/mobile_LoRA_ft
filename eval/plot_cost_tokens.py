#!/usr/bin/env python3
"""Wall time against token count for five completed adapter runs.

One panel, not two. The claim is that token count predicts cost and entry
count does not, and both halves fit on a single axis: the points lie on the
token fit while their entry-count labels run out of order along it (550, 500
and 448 appear as tokens increase). A second panel plotting entry count would
spend a column restating what the label order already shows.

    .venv-mlx/bin/python eval/plot_cost_tokens.py \
        --agg results/ondevice_e2e_smollm3_a1lamp_nax-on_2026-08-13.json \
        --outdir ../overleafs/mobile_FT_paper/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

DOT = "#4575b4"
# (entry count) -> label offset in points, hand-placed: the three mid-size runs
# sit within 0.16 M tokens of each other and collide under any single rule.
OFFSET = {405: (6, -9), 550: (-4, 7), 500: (2, -11), 448: (4, 7), 987: (-6, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--outdir", default="results/ondevice/figures")
    a = ap.parse_args()

    d = json.loads(Path(a.agg).read_text())["derived"]
    pts = sorted(d["cost_points_c0"], key=lambda p: p["total_tokens_est"])
    fit = d["cost_fit_tokens"]

    x = np.array([p["total_tokens_est"] for p in pts]) / 1e6
    y = np.array([p["wall_time_s"] for p in pts]) / 3600.0
    n = [p["profile_size"] for p in pts]

    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    fig, ax = plt.subplots(figsize=(3.3, 2.3))

    xs = np.linspace(0.45, 2.55, 50)
    ax.plot(xs, (fit["slope_s_per_token"] * xs * 1e6 + fit["intercept_s"]) / 3600.0,
            color="#666666", lw=1.0, zorder=2)
    ax.scatter(x, y, s=26, color=DOT, zorder=3, edgecolors="none")
    for xi, yi, ni in zip(x, y, n):
        ax.annotate(str(ni), xy=(xi, yi), xytext=OFFSET[ni],
                    textcoords="offset points", fontsize=6.5, color="dimgrey")

    ax.set_xlabel("user history (million tokens)", fontsize=8, color="dimgrey")
    ax.set_ylabel("wall-clock time (h)", fontsize=8, color="dimgrey")
    ax.set_xlim(0.35, 2.65)
    ax.set_ylim(0.6, 7.4)
    ax.grid(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey",
                   labelsize=7)
    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"cost_tokens.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    for xi, yi, ni in zip(x, y, n):
        pred = (fit["slope_s_per_token"] * xi * 1e6 + fit["intercept_s"]) / 3600.0
        print(f"  {ni:>4} entries  {xi:.3f} M tok  {yi:.2f} h  "
              f"(fit {pred:.2f} h, {100*(pred-yi)/yi:+.0f}%)")
    print(f"wrote {out}/cost_tokens.pdf")


if __name__ == "__main__":
    main()
