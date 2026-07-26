# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "pandas",
# ]
# ///
"""Tokens vs. seconds/iteration for on-device SmolLM3 LoRA training (h7).

Data: results/ondevice/train_bench_metrics_tokentime_2026-07-25_cold_v4.jsonl (400
raw iterations, 50-token grid, longer pre-run cooldown) + a matching per-cell-stats
aggregate results/ondevice_tokentime_smollm3_4bit_2026-07-25_cold_v4.json (per-cell
stats + precomputed OLS fit). This run shows three thermal states (nominal/fair/
serious), unlike the earlier fine50 run which only ever reached fair/serious.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    """Plot seconds/iteration vs. target token count for on-device LoRA training.

    Parameters
    ----------
    None (paths are hardcoded to this repo's on-device telemetry files).

    Saves
    -----
    ./figures/tokentime_skill.pdf
    ./figures/tokentime_skill.png
    """
    repo_root = Path(__file__).resolve().parent.parent
    jsonl_path = repo_root / "results/ondevice/train_bench_metrics_tokentime_2026-07-25_cold_v4.jsonl"
    agg_path = repo_root / "results/ondevice_tokentime_smollm3_4bit_2026-07-25_cold_v4.json"

    # --- Style Setup ---
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

    # --- Data ---
    raw = pd.read_json(jsonl_path, lines=True)
    raw = raw[["target_tokens", "seconds_per_iter", "thermal_state"]].dropna()

    agg = json.loads(agg_path.read_text())
    cells = agg["cells"]
    cell_x = np.array([c["target_tokens"] for c in cells])
    cell_mean = np.array([c["seconds_per_iter"]["mean"] for c in cells])
    cell_std = np.array([c["seconds_per_iter"]["std"] for c in cells])

    fit = agg["linear_fit"]
    intercept, slope = fit["intercept_s"], fit["slope_s_per_token"]

    # --- Palette ---
    # TRAFFIC_LIGHT (nominal/fair/serious thermal state) + neutral for mean/fit
    THERMAL_STYLE = {
        "nominal": ("#33a02c", "o"),  # green, circle
        "fair": ("#fdbf6f", "^"),     # amber, triangle
        "serious": ("#e31a1c", "s"),  # red, square
    }
    MEAN_COLOR = "#1f253f"     # cubehelix darkest
    FIT_COLOR = "dimgrey"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # individual iterations, colored + shaped by thermal state (double-encoded)
    for state in ["nominal", "fair", "serious"]:
        sub = raw[raw["thermal_state"] == state]
        if len(sub):
            color, marker = THERMAL_STYLE[state]
            ax.scatter(sub["target_tokens"], sub["seconds_per_iter"],
                       s=24, alpha=0.45, color=color, marker=marker, edgecolors="none",
                       label=f"Iteration ({state})", zorder=2)

    # per-cell mean +/- 1 std
    ax.errorbar(cell_x, cell_mean, yerr=cell_std, fmt="D", ms=8,
                color=MEAN_COLOR, ecolor=MEAN_COLOR, elinewidth=1.3,
                capsize=4, zorder=4, label="Cell mean ± 1σ")

    # OLS fit line
    x_line = np.linspace(0, cell_x.max() * 1.05, 100)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, "--", color=FIT_COLOR, linewidth=1.8, zorder=3,
            label="Linear fit")

    ax.set_xlim(0, cell_x.max() * 1.05)
    ax.set_ylim(bottom=0)  # seconds/iter can't go negative; clip the extrapolated fit

    ax.set_xlabel("Target sequence length (tokens)", fontsize=12, labelpad=8, color="dimgrey")
    ax.set_ylabel("Seconds per iteration", fontsize=12, labelpad=8, color="dimgrey")
    ax.set_title("Token Count vs. Iteration Count",
                 fontsize=14, loc="left", pad=7, color="dimgrey")

    ax.grid(alpha=0.4, linewidth=0.8, axis="both")
    sns.despine(left=True, bottom=True)
    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")

    ax.legend(frameon=True, facecolor="white", framealpha=0.8, edgecolor="lightgrey",
              labelcolor="dimgrey", loc="upper left", fontsize=10.5)

    # --- Save ---
    out_dir = repo_root / "figures"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "tokentime_skill.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(out_dir / "tokentime_skill.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
