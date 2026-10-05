#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///
"""Overlay plot for the 2026-07-03 capped-stress (bursty 128-tok) throttle
run: SmolLM3-3B-4bit vs Qwen3-8B-4bit decode speed over a 10-minute window of
back-to-back forced generations, each point's marker shaded by the reported
``ProcessInfo.thermalState``.

Reads the two h4 aggregate JSONs directly (``stress.decay``, which carries
real per-segment ``elapsed_s`` — no tok/s reconstruction needed).

Usage:
    uv run eval/plot_capped_stress_overlay.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# ProcessInfo.thermalState enum -> colour (Apple ordering: nominal<fair<serious<critical)
THERMAL_COLORS = {
    "nominal": "#2ca02c",
    "fair": "#fc8d59",
    "serious": "#d73027",
    "critical": "#7f0000",
}
THERMAL_ORDER = ["nominal", "fair", "serious", "critical"]

MODELS = [
    {
        "agg": "results/ondevice_stresscap_smollm3_4bit_2026-07-03.json",
        "label": "SmolLM3-3B-4bit",
        "color": "#4575b4",
        "marker": "o",
    },
    {
        "agg": "results/ondevice_stresscap_qwen3_8b_4bit_2026-07-03.json",
        "label": "Qwen3-8B-4bit",
        "color": "#762a83",
        "marker": "s",
    },
]


def load_decay(agg_path):
    agg = json.load(open(agg_path))
    decay = agg["stress"]["decay"]
    elapsed = np.array([d["elapsed_s"] for d in decay])
    tps = np.array([d["gen_tps"] for d in decay])
    therm = [d["thermal_state"] for d in decay]
    cum_tok = decay[-1]["cumulative_tokens"]
    peak_gb = agg["stress"]["peak_mem_bytes"]["max"] / 1e9
    return elapsed, tps, therm, cum_tok, peak_gb


def plateau_floor(tps, frac=0.5):
    n = max(1, int(len(tps) * frac))
    return tps[-n:].mean()


def state_changes(therm):
    """Indices where thermalState differs from the previous point (excludes index 0 —
    that's the starting state, not a change)."""
    return [i for i in range(1, len(therm)) if therm[i] != therm[i - 1]]


def main():
    """Render the SmolLM3-3B vs Qwen3-8B capped-stress throttle overlay.

    Saves
    -----
    results/ondevice/figures/capped_stress_overlay_2026-07-03.{pdf,png}
    """
    sns.set_theme(font_scale=1.0, style="whitegrid", font="DejaVu Sans")

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)

    # pre-load everything once so axis limits (needed to size the short
    # separator lines) are known before any drawing happens
    runs = [(spec, *load_decay(spec["agg"])) for spec in MODELS]
    xmax = max(elapsed[-1] for _, elapsed, *_ in runs)
    ymax = max(tps[0] for _, _, tps, *_ in runs) * 1.08
    ax.set_xlim(-5, xmax * 1.02)
    ax.set_ylim(0, ymax)
    sep_half_h = 0.045 * ymax  # "short" vertical separator half-height

    stats_y = {0: 0.135, 1: 0.02}  # axes-fraction y for the stat callouts
    states_seen = []

    for mi, (spec, elapsed, tps, therm, cum_tok, peak_gb) in enumerate(runs):
        start, floor = tps[0], plateau_floor(tps)
        throttle_pct = 100.0 * (1.0 - floor / start)

        # main curve
        ax.plot(elapsed, tps, "-", color=spec["color"], linewidth=2.3,
                alpha=0.9, zorder=3, solid_capstyle="round")

        # every point gets a small marker (identity via shape, not thinned)
        ax.plot(elapsed, tps, linestyle="none", marker=spec["marker"],
                markersize=3.2, markerfacecolor=spec["color"],
                markeredgecolor="none", alpha=0.85, zorder=4)

        # thermalState changes drawn as short dashed vertical separators,
        # colour = the state being entered, straddling the curve at that point
        for ti in state_changes(therm):
            y = tps[ti]
            ax.plot([elapsed[ti], elapsed[ti]], [y - sep_half_h, y + sep_half_h],
                    linestyle="--", linewidth=1.6, color=THERMAL_COLORS[therm[ti]],
                    alpha=0.95, zorder=5, dash_capstyle="butt")
            states_seen.append(therm[ti])

        # stat callout in the empty band under the settled plateau
        ax.text(0.355, stats_y[mi],
                f"{spec['label']}   {start:.1f}$\\rightarrow${floor:.1f} tok/s (−{throttle_pct:.0f}%)\n"
                f"{cum_tok:,} tok in {len(elapsed)} gens · peak {peak_gb:.1f} GB",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.8, color=spec["color"], fontweight="medium", linespacing=1.5)

    ax.set_xlabel("Elapsed time (s)", fontsize=12, labelpad=8, color="dimgrey")
    ax.set_ylabel("Decode speed (tok/s)", fontsize=12, labelpad=8, color="dimgrey")
    ax.set_title(
        "Continuous Inference Measurement - SmolLM3-3B vs Qwen3-8B",
        fontsize=14, loc="left", pad=14, color="dimgrey")

    ax.grid(alpha=0.35, linewidth=0.8)

    # legend: model identity (line+marker)
    model_handles = [
        Line2D([0], [0], color=m["color"], marker=m["marker"], markersize=7,
               markerfacecolor=m["color"], markeredgecolor="white", linewidth=2.3,
               label=m["label"])
        for m in MODELS
    ]
    leg1 = ax.legend(handles=model_handles, loc="upper right", fontsize=9.5,
                      frameon=True, facecolor="white", framealpha=0.85,
                      edgecolor="lightgrey", labelcolor="dimgrey",
                      title="Model", title_fontsize=9.5)
    ax.add_artist(leg1)

    # thermal-state legend — dashed line handles matching the separator style,
    # limited to states that actually got a separator drawn (no phantom entries)
    thermal_handles = [
        Line2D([0], [0], linestyle="--", linewidth=2.2, color=THERMAL_COLORS[s], label=f"→ {s}")
        for s in THERMAL_ORDER if s in states_seen
    ]
    ax.legend(handles=thermal_handles, loc="upper left", bbox_to_anchor=(0.40, 1.0),
              fontsize=9, frameon=True, facecolor="white", framealpha=0.85,
              edgecolor="lightgrey", labelcolor="dimgrey",
              title="thermalState change", title_fontsize=9)

    ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
    sns.despine(left=True, bottom=True)
    ax.patch.set_edgecolor("lightgrey")
    ax.patch.set_linewidth(0.8)

    fig.tight_layout()

    out_dir = Path("results/ondevice/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"capped_stress_overlay_2026-07-03.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
