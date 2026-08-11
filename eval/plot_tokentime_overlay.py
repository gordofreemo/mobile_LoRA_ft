#!/usr/bin/env python3
"""NAX off/on overlay of the h7 token-time cost model.

Seconds/iteration vs token count: measured cell means (markers) + linear fits
(lines) for up to four series — hot/cold regime x NAX off/on. Slope equations
annotated per series. The slope change IS the finding: the fix removes a
roughly constant fraction of backward, so the cost-per-token coefficient
drops while the story (linear in tokens, hot ~1.7x cold) stays.

    .venv-mlx/bin/python eval/plot_tokentime_overlay.py \
        --off-hot results/ondevice_tokentime_smollm3_4bit_2026-07-24.json \
        --off-cold results/ondevice_tokentime_smollm3_4bit_2026-07-25_cold_v4.json \
        --on-hot <rerun hot agg> --on-cold <rerun cold agg> \
        --out results/ondevice/figures/tokentime_overlay_<date>.pdf
"""
import argparse

from _e2e_plot_style import apply_rc, load_agg, save, plt

SERIES = {
    # (arm, regime) -> color, linestyle, marker
    ("off", "hot"): ("#0072B2", "-", "o"),
    ("on", "hot"): ("#D55E00", "-", "s"),
    ("off", "cold"): ("#0072B2", "--", "o"),
    ("on", "cold"): ("#D55E00", "--", "s"),
}


def plot_series(ax, agg, arm, regime, y_annot):
    color, ls, marker = SERIES[(arm, regime)]
    cells = [c for c in agg["cells"] if c["seconds_per_iter"]["mean"] is not None]
    xs = [c["target_tokens"] for c in cells]
    ys = [c["seconds_per_iter"]["mean"] for c in cells]
    ax.plot(xs, ys, marker, color=color, markersize=3.5,
            markerfacecolor="none" if regime == "cold" else color,
            linestyle="none", zorder=3)
    fit = agg.get("linear_fit")
    if fit:
        a, b = fit["intercept_s"], fit["slope_s_per_token"]
        lo, hi = min(xs), max(xs)
        ax.plot([lo, hi], [a + b * lo, a + b * hi], ls, color=color, linewidth=1.2,
                zorder=2)
        ax.text(0.99, y_annot,
                f"{regime} {arm}: {a:+.3f} + {1000 * b:.2f} ms/tok  "
                f"(r={fit['pearson_r']:.3f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-hot")
    ap.add_argument("--on-hot")
    ap.add_argument("--off-cold")
    ap.add_argument("--on-cold")
    ap.add_argument("--out", default="results/ondevice/figures/tokentime_overlay.pdf")
    args = ap.parse_args()
    apply_rc()

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    y = 0.98
    for arm_regime, path in (
        (("off", "hot"), args.off_hot), (("on", "hot"), args.on_hot),
        (("off", "cold"), args.off_cold), (("on", "cold"), args.on_cold),
    ):
        if path:
            plot_series(ax, load_agg(path), *arm_regime, y_annot=y)
            y -= 0.055

    ax.set_xlabel("target tokens")
    ax.set_ylabel("seconds per iteration")
    ax.set_title("Per-iteration cost vs tokens — NAX off (blue) vs on (orange);\n"
                 "solid = sustained-heat pass, dashed = cooled cells")
    save(fig, args.out)


if __name__ == "__main__":
    main()
