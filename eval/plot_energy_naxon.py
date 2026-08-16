#!/usr/bin/env python3
"""Battery cost of one adapter, broken vs repaired kernel (h9 rerun).

Grouped bars: net energy as % of one full charge for the three unplugged
training runs (405-, 550-, 987-example profiles), broken kernel (h9,
2026-07-29) vs repaired (2026-08-12/13/16). The 987 bars are hatched: the run
dies at critical battery under BOTH kernels at the same ~90.5% net --- the
annotation carries the fraction of training completed at death, which is what
the repair moves.

    .venv-mlx/bin/python eval/plot_energy_naxon.py \
        --out results/ondevice/figures/energy_naxon_2026-08-16.pdf
"""
import argparse
import json

from _e2e_plot_style import apply_rc, save, plt

OFF_COLOR, ON_COLOR = "#0072B2", "#D55E00"

# (label, off net% of charge, off note, on-point JSON, on note)
POINTS = [
    ("405 ex.", 51.1, "complete", "results/ondevice_energy_naxon_XS_2026-08-13.json", "complete"),
    ("550 ex.", 86.9, "complete", "results/ondevice_energy_naxon_L_2026-08-12.json", "complete"),
    ("987 ex.", 90.5, "dies @ 49%", "results/ondevice_energy_naxon_XXL_2026-08-16.json", "dies @ 71%"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ondevice/figures/energy_naxon.pdf")
    args = ap.parse_args()
    apply_rc()

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    width = 0.36
    for i, (label, off_pct, off_note, on_path, on_note) in enumerate(POINTS):
        on = json.load(open(on_path))["point_nax_on"]
        on_pct = on["pct_full_battery_net"]
        died = "dies" in on_note
        ax.bar(i - width / 2 - 0.02, off_pct, width, color=OFF_COLOR,
               hatch="//" if "dies" in off_note else None,
               edgecolor="white", label="broken kernel" if i == 0 else None)
        ax.bar(i + width / 2 + 0.02, on_pct, width, color=ON_COLOR,
               hatch="//" if died else None,
               edgecolor="white", label="repaired kernel" if i == 0 else None)
        ax.text(i - width / 2 - 0.02, off_pct + 1.5, off_note,
                ha="right" if died else "center", fontsize=7, color="#3a3a3a")
        ax.text(i + width / 2 + 0.02, on_pct + 4.5 if died else on_pct + 1.5,
                on_note, ha="left" if died else "center", fontsize=7,
                color="#3a3a3a")
    ax.axhline(100, color="#999999", linewidth=0.8, linestyle=":")
    ax.text(2.35, 101.5, "one full charge", fontsize=7, ha="right", color="#666666")
    ax.set_xticks(range(len(POINTS)))
    ax.set_xticklabels([p[0] for p in POINTS])
    ax.set_ylabel("net energy, % of one full charge")
    ax.set_ylim(0, 112)
    ax.set_title("Battery cost of one adapter (unplugged); hatched = dies at\n"
                 "critical battery — the ~90.5% ceiling is kernel-independent")
    ax.legend(frameon=False, loc="upper left")
    save(fig, args.out)


if __name__ == "__main__":
    main()
