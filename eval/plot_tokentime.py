#!/usr/bin/env python3
"""Token-time cost model: seconds/iteration vs synthetic example token count.

Single panel. X=token count, Y=seconds/iteration, linear-linear. Measured
per-cell means shown as markers with light ±1 std whiskers; overlaid
least-squares linear fit with a shaded ±1 residual-std band and a direct text
annotation for the equation (no legend box competing with the title). Locked
design: experiments/2026-07-24-ondevice-tokentime-plan.md (pinned via
/grill_me 2026-07-24; grid refined to 50-token steps same day). Reuses the
shared on-device figure style (_e2e_plot_style.py) for visual consistency
with the E2E cost-law figure this feeds.

    .venv-mlx/bin/python eval/plot_tokentime.py \
        --agg results/ondevice_tokentime_smollm3_4bit_2026-07-24.json \
        --out results/ondevice/figures/tokentime_2026-07-24.pdf
"""
import argparse

from _e2e_plot_style import apply_rc, load_agg, save, plt

MEASURED_COLOR = "#0072B2"   # Okabe-Ito blue — matches every other on-device figure
MEASURED_ECOLOR = "#99C2E0"  # lighter tint of the same hue, for whiskers only
FIT_COLOR = "#3a3a3a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", default="results/ondevice/figures/tokentime.pdf")
    args = ap.parse_args()
    apply_rc()

    agg = load_agg(args.agg)
    cells = [c for c in agg["cells"] if c["seconds_per_iter"]["mean"] is not None]
    xs = [c["target_tokens"] for c in cells]
    ys = [c["seconds_per_iter"]["mean"] for c in cells]
    yerr = [c["seconds_per_iter"]["std"] or 0.0 for c in cells]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))

    fit = agg.get("linear_fit")
    if fit:
        a, b = fit["intercept_s"], fit["slope_s_per_token"]
        rstd = fit.get("residual_std_s") or 0.0
        lo, hi = min(xs), max(xs)
        pad = (hi - lo) * 0.04
        xfit = [lo - pad, hi + pad]
        yfit = [a + b * x for x in xfit]
        if rstd > 0:
            ax.fill_between(
                xfit, [y - rstd for y in yfit], [y + rstd for y in yfit],
                color=FIT_COLOR, alpha=0.10, linewidth=0, zorder=1,
                label="±1 residual σ",
            )
        ax.plot(xfit, yfit, color=FIT_COLOR, lw=1.6, zorder=2, solid_capstyle="round")

        # Direct annotation instead of a legend box — sits in the empty
        # lower-right region (high tokens/low cost is unpopulated since cost
        # rises with tokens).
        eq = (
            f"s/iter ≈ {a:.3f} + {b:.5f} × tokens\n"
            f"r = {fit['pearson_r']:.4f}   n = {fit['n_points']} cells"
        )
        ax.text(
            0.97, 0.06, eq, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#333",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#dddddd", linewidth=0.8),
        )

    ax.errorbar(
        xs, ys, yerr=yerr, fmt="o", color=MEASURED_COLOR, ecolor=MEASURED_ECOLOR,
        elinewidth=0.9, capsize=0, markersize=6, markeredgewidth=0.6,
        markeredgecolor="white", zorder=3,
    )

    ax.set_xlabel("tokens per example")
    ax.set_ylabel("seconds / iteration")
    ax.set_title("On-device LoRA training: per-iteration cost vs token count")
    ax.set_xlim(min(xs) - (max(xs) - min(xs)) * 0.04, max(xs) + (max(xs) - min(xs)) * 0.04)

    ax.text(
        0.03, 0.94, "measured: mean ± 1 std (n=20/cell)", transform=ax.transAxes,
        ha="left", va="top", fontsize=8, color=MEASURED_COLOR,
    )

    save(fig, args.out)


if __name__ == "__main__":
    main()
