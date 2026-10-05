#!/usr/bin/env python3
"""Battery cost of one adapter, on the repaired kernel only.

Single-arm by design. The paper's claim here is the absolute cost of an
adapter and the charge ceiling it runs into, neither of which needs a
stock-kernel comparison; the A/B belongs in the kernel section. Dropping the
second series also drops the weakest comparison in the paper -- the stock and
repaired energy runs are weeks apart, and the 100%-start fuel-gauge under-read
documented in the XS result biases that pair in the repair's favour.

    .venv-mlx/bin/python eval/plot_energy_cost.py \
        --outdir ../overleafs/mobile_FT_paper/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

BAR = "#4575b4"
POINTS = [  # (x label, result json)
    ("405", "results/ondevice_energy_naxon_XS_2026-08-13.json"),
    ("550", "results/ondevice_energy_naxon_L_2026-08-12.json"),
    ("987", "results/ondevice_energy_naxon_XXL_2026-08-16.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/ondevice/figures")
    a = ap.parse_args()
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for i, (label, path) in enumerate(POINTS):
        p = json.loads(Path(path).read_text())["point_nax_on"]
        pct = p["pct_full_battery_net"]
        died = not p["completed"]
        # hatch is the only annotation kept on the plot: it marks the one bar
        # that is not the cost of a finished adapter. Everything else -- exact
        # values, what "did not finish" means -- reads better in the caption
        # and the prose than printed over three bars.
        ax.bar(i, pct, width=0.55, color=BAR, zorder=3,
               hatch="//" if died else None,
               edgecolor="white" if died else "none", linewidth=0)
        print(f"  {label:>4} ex: {pct:5.1f}% of a charge, "
              f"{'died at %.1f%% of iters' % p['pct_iterations'] if died else 'complete'}")

    ax.axhline(100, color="#444444", lw=0.8, ls=":", zorder=4)
    ax.text(0.012, 101.5, "1 charge", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6.5, color="#444444")

    ax.set_xticks(range(len(POINTS)))
    ax.set_xticklabels([l for l, _ in POINTS])
    ax.set_xlabel("profile size (examples)", fontsize=8, color="dimgrey")
    ax.set_ylabel("% of one charge", fontsize=8, color="dimgrey")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(False)                  # clear the seaborn theme's x gridlines,
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)   # which otherwise cross the bars
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey",
                   labelsize=7)
    sns.despine(left=True, bottom=True)
    fig.tight_layout()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"energy_cost.{ext}", bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    print(f"wrote {out}/energy_cost.pdf")


if __name__ == "__main__":
    main()
