#!/usr/bin/env python3
"""h9 energy characterization, two panels.

(a) Energy per adapter as a share of one full battery charge, stacked as
    net (training) + idle-baseline correction = the gross drain measured.
    XXL never completed (died of battery exhaustion at 49.3% of its
    iterations) — drawn hatched in red.
(b) Energy per completed iteration: cost per unit of work rises with
    profile size, so the total scales faster than the iteration count.

  python eval/plot_energy.py --agg results/ondevice_e2e_smollm3_a1lamp_2026-07-29.json \
      --out results/ondevice/figures/energy_h9_2026-07-29.pdf
"""
import argparse

from _e2e_plot_style import USER_COLORS, apply_rc, label, load_agg, save, plt

GREY = "#dfdfdf"        # idle-baseline correction segment
DIED = "#a11212"        # incomplete-run accent
INK = "dimgrey"         # annotation ink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", default="results/ondevice/figures/energy_h9.pdf")
    args = ap.parse_args()
    apply_rc()

    agg = load_agg(args.agg)
    energy = agg["derived"]["energy_h9"]
    pts = sorted(energy["points"], key=lambda p: p["profile_size"])
    if not pts:
        raise SystemExit("no h9 energy points in this aggregate")
    full_j = energy["full_battery_joules"]

    gross = [p["pct_full_battery_gross"] for p in pts]
    net = [p["pct_full_battery_net"] for p in pts]
    j_per_iter = [p["net_j"] / p["iterations_reached"] for p in pts]
    x = list(range(len(pts)))

    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})

    # --- (a) share of one battery charge: net + idle baseline = gross drain ---
    ax.axhline(100, color=DIED, lw=0.9, ls="--", zorder=1)
    ax.text(-0.58, 97.5, "one full charge", fontsize=7.5, color=DIED,
            ha="left", va="top")

    w = 0.52
    for i, p in enumerate(pts):
        dead = not p["completed"]
        ax.bar(i, net[i], w, color=USER_COLORS.get(p["user"], "#444"),
               edgecolor=DIED if dead else "white",
               linewidth=1.1 if dead else 0.6, hatch="//" if dead else None,
               zorder=3)
        ax.bar(i, gross[i] - net[i], w, bottom=net[i], color=GREY,
               edgecolor=DIED if dead else "white",
               linewidth=1.1 if dead else 0.6, hatch="//" if dead else None,
               zorder=3)
        ax.text(i, net[i] / 2, f"{net[i]:.0f}", ha="center", va="center",
                fontsize=8.5, color="white", weight="semibold")
        top = f"{gross[i]:.0f}% drain"
        if dead:
            top += f"\ndied at {p['iterations_reached'] / p['iterations_total']:.0%}"
        # sit on the bar, unless that would clash with the full-charge line
        ax.text(i, gross[i] + 1.5 if gross[i] < 93 else 103, top,
                ha="center", va="bottom", fontsize=8,
                color=DIED if dead else INK)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{label(p['user'])}\n{p['profile_size']}" for p in pts])
    ax.set_xlim(-0.62, len(pts) - 0.38)
    ax.set_ylim(0, 125)
    ax.set_ylabel("% of a full battery charge")
    ax.set_xlabel("profile size (entries)")
    ax.set_title(f"(a)  energy per adapter (charge ≈ {full_j / 1000:.1f} kJ)",
                 loc="left")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#777777", label="net (training)"),
        plt.Rectangle((0, 0), 1, 1, color=GREY, label="idle baseline"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.70),
              frameon=False, labelcolor=INK, handlelength=1.1)

    # --- (b) cost per unit of work ---
    axr.vlines(x, 0, j_per_iter, color="#cccccc", lw=1.4, zorder=2)
    for i, p in enumerate(pts):
        axr.scatter(i, j_per_iter[i], s=64, zorder=3,
                    color=USER_COLORS.get(p["user"], "#444"))
        axr.text(i, j_per_iter[i] + 1.4, f"{j_per_iter[i]:.0f}", ha="center",
                 va="bottom", fontsize=8, color=INK)

    axr.set_xticks(x)
    axr.set_xticklabels([f"{label(p['user'])}\n{p['profile_size']}" for p in pts])
    axr.set_xlim(-0.6, len(pts) - 0.4)
    axr.set_ylim(0, max(j_per_iter) * 1.25)
    axr.set_ylabel("J/iter")
    axr.set_xlabel("profile size (entries)")
    axr.set_title("(b)  cost per iteration", loc="left")

    for a in (ax, axr):
        a.tick_params(axis="both", which="both", length=0, labelcolor=INK)

    fig.subplots_adjust(wspace=0.34)
    save(fig, args.out)


if __name__ == "__main__":
    main()
