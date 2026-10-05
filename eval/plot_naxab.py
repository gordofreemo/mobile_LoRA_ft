#!/usr/bin/env python3
"""NAX A/B round figure: backward-path quantized matmul speedup.

Reads the aggregate produced by eval/naxab_aggregate.py and renders two panels:

  (a) absolute backward phase time per iteration, OFF vs ON
  (b) speedup (OFF/ON) for the backward phase and the fused whole iteration

Cool and hot passes are averaged per token count (the round found the effect
thermally invariant). Each cell pairs ON/OFF iterations seconds apart at the
same die temperature (arm alternated per iteration), so the ratio in (b) is a
paired within-cell comparison, not a cross-run one.

    .venv-mlx/bin/python eval/plot_naxab.py \
        --agg results/ondevice_naxab_smollm3_4bit_2026-08-06.json \
        --out results/ondevice/figures/naxab_speedup_2026-08-06.pdf
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

OFF_RED = "#d73027"
ON_BLUE = "#4575b4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="results/ondevice_naxab_smollm3_4bit_2026-08-06.json")
    ap.add_argument("--out", default="results/ondevice/figures/naxab_speedup_2026-08-06.pdf")
    args = ap.parse_args()

    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")
    pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)

    # --- Data ---
    agg = json.loads(Path(args.agg).read_text())
    # M reaching the quantized matmuls is tokens - 1 (LoRABatchIterator slices
    # inputs [:, :-1]); cool and hot passes are averaged per M.
    by_m = defaultdict(list)
    for c in agg["cells"]:
        by_m[c["target_tokens"] - 1].append(c)
    m_vals = sorted(by_m)

    def merged(field):
        return [statistics.fmean(c[field] for c in by_m[m]) for m in m_vals]

    backward_off = merged("backward_off")
    backward_on = merged("backward_on")
    backward_ratio = merged("backward_ratio_off_over_on")
    fused_speedup = merged("fused_speedup")

    # --- Plot ---
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=150)

    # (a) absolute backward time, OFF vs ON
    ax_a.plot(m_vals, backward_off, color=OFF_RED, marker="o", markersize=5,
              linewidth=2, zorder=3, label="NAX off (stock dispatch)")
    ax_a.plot(m_vals, backward_on, color=ON_BLUE, marker="o", markersize=5,
              linewidth=2, zorder=3, label="NAX on (patched)")
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(m_vals, [str(m) for m in m_vals])
    ax_a.set_xlabel("sequence length", fontsize=12,
                    labelpad=8, color="dimgrey")
    ax_a.set_ylabel("backward phase (s / iteration)", fontsize=12,
                    labelpad=8, color="dimgrey")
    ax_a.set_title(r"$\bf{(a)}$" + " Backward phase time, NAX off vs on",
                   fontsize=13, loc="left", pad=7, color="dimgrey")
    ax_a.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white",
                framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey")

    # (b) speedup ratios
    ax_b.plot(m_vals, backward_ratio, color=pal[5], marker="o", markersize=5,
              linewidth=2, zorder=3, label="backward phase")
    ax_b.plot(m_vals, fused_speedup, color=pal[2], marker="o", markersize=5,
              linewidth=2, zorder=3, label="whole iteration (fused)")
    ax_b.axhline(y=1.0, color="lightgrey", linewidth=0.8, zorder=1)
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(m_vals, [str(m) for m in m_vals])
    ax_b.set_ylim(0.95, 2.15)
    ax_b.set_xlabel("sequence length", fontsize=12,
                    labelpad=8, color="dimgrey")
    ax_b.set_ylabel("speedup (time off / time on)", fontsize=12,
                    labelpad=8, color="dimgrey")
    ax_b.set_title(r"$\bf{(b)}$" + " Speedup from enabling NAX on backward",
                   fontsize=13, loc="left", pad=7, color="dimgrey")
    ax_b.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white",
                framealpha=0.8, edgecolor="lightgrey", labelcolor="dimgrey")

    for ax in (ax_a, ax_b):
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
        ax.grid(False)
    sns.despine(left=True, bottom=True)
    for ax in (ax_a, ax_b):
        ax.patch.set_edgecolor("lightgrey")
        ax.patch.set_linewidth(0.8)

    fig.suptitle("NAX Quantized Matmul on Backwards Pass",
                 fontsize=14, color="dimgrey", y=1.02)
    fig.tight_layout()

    # --- Save ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
