#!/usr/bin/env python3
"""Thermal cooldown trajectory (h10): throughput recovery after a training burst.

Single panel. X = seconds since the training burst ended. Left Y = training
throughput as a fraction of the device's own cold reference, so 1.0 means
"fully recovered". Points coloured by the reported thermal state (the reserved
nominal/fair/serious status ramp, shared with the other on-device figures) —
which is the whole point of the figure: throughput is back to normal long
before the thermal state says so.

Reuses the shared on-device figure style (_e2e_plot_style.py).
Locked design: experiments/2026-07-28-ondevice-thermal-cooldown-h10-plan.md.

    .venv-mlx/bin/python eval/plot_thermal.py \
        --agg results/ondevice_thermal_smollm3_4bit_2026-07-28.json \
        --out results/ondevice/figures/thermal_cooldown_2026-07-28.pdf
"""
import argparse

from _e2e_plot_style import THERMAL_COLORS, apply_rc, load_agg, save, plt

THERMAL_ORDER = ["nominal", "fair", "serious", "critical"]
HOT_COLOR = "#7f7f7f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", default="results/ondevice/figures/thermal_cooldown.pdf")
    ap.add_argument(
        "--session", help="bench_session_id to plot (default: first complete run)"
    )
    args = ap.parse_args()
    apply_rc()

    agg = load_agg(args.agg)
    sessions = [s for s in agg["sessions"] if s.get("complete")]
    if args.session:
        sessions = [s for s in sessions if s["bench_session_id"] == args.session]
    if not sessions:
        raise SystemExit("no complete session in aggregate")
    s = sessions[0]
    curve = agg["curves"][s["bench_session_id"]]

    cold = s["cold_ref_seconds_per_iter"]
    hot = s["soak_plateau_seconds_per_iter"]

    fig, ax = plt.subplots(figsize=(5.4, 3.2))

    # Throughput relative to the session's own cold reference.
    xs = [c["cooldown_elapsed_s"] / 60.0 for c in curve]
    ys = [cold / c["seconds_per_iter"] for c in curve]

    ax.plot(xs, ys, "-", color="#999999", linewidth=0.9, zorder=1)
    for state in THERMAL_ORDER:
        px = [x for x, c in zip(xs, curve) if c["thermal_state"] == state]
        py = [y for y, c in zip(ys, curve) if c["thermal_state"] == state]
        if px:
            ax.scatter(px, py, s=22, color=THERMAL_COLORS[state],
                       label=f"reported: {state}", zorder=3, edgecolors="none")

    # Reference lines: fully recovered, and the throttled steady state.
    ax.axhline(1.0, color="#444444", linewidth=0.8, linestyle="--", zorder=2)
    ax.axhline(cold / hot, color=HOT_COLOR, linewidth=0.8, linestyle=":", zorder=2)
    ax.text(max(xs) * 0.995, 1.0 + 0.012, "cold reference", ha="right", va="bottom",
            fontsize=7.5, color="#444444")
    # Left-anchored: the legend owns the lower right.
    ax.text(0.5, cold / hot - 0.012, "throttled steady state (end of the burst)",
            ha="left", va="top", fontsize=7.5, color=HOT_COLOR)

    # 95%-recovery marker.
    t95 = s.get("t95_s")
    if t95:
        ax.axvline(t95 / 60.0, color="#444444", linewidth=0.7, linestyle="-", alpha=0.5)
        ax.annotate(
            f"95% recovered\n{t95/60:.1f} min",
            xy=(t95 / 60.0, 0.95), xytext=(t95 / 60.0 + 6, 0.72),
            fontsize=7.5, color="#222222",
            arrowprops=dict(arrowstyle="->", color="#666666", linewidth=0.7),
        )

    # The punchline: the reported thermal state only clears long after
    # throughput has been back to normal.
    first_nominal = next(
        (c for c in curve if c["thermal_state"] == "nominal"), None
    )
    if first_nominal and t95:
        xn = first_nominal["cooldown_elapsed_s"] / 60.0
        ax.annotate(
            f"reported state only returns to\n\"nominal\" here ({xn:.0f} min)",
            xy=(xn, first_nominal_y := cold / first_nominal["seconds_per_iter"]),
            xytext=(xn - 30, 0.55), fontsize=7.5, color="#222222",
            arrowprops=dict(arrowstyle="->", color="#666666", linewidth=0.7),
        )

    ax.set_xlabel("minutes since the training burst ended")
    ax.set_ylabel("throughput / cold reference")
    ax.set_ylim(0.35, 1.08)
    ax.set_xlim(-1.5, max(xs) + 1.5)
    ax.legend(loc="lower right", frameon=False, ncol=1)

    save(fig, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
