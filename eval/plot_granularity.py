#!/usr/bin/env python3
"""GC-granularity sweep (h8): throughput + peak memory vs checkpoint group size K.

Dual-axis single panel, completed cells only (K=1,2,3,4,6 — the OOM'd K>=9
cells produced no data and aren't plotted; see the h8 writeup for that
boundary). X = K (checkpoint group size), plotted at EVENLY-SPACED categorical
positions (not the numeric values — 1,2,3,4,6 on a linear axis still isn't
uniform spacing). Left Y = iter/s (throughput), split by thermal state
(nominal/fair/serious — the reserved status ramp) since that decomposes the
per-K variance into what's actually driving it. Right Y = peak memory (MB).
Locked design: experiments/2026-07-25-ondevice-gc-granularity-plan.md (pinned
via /grill_me 2026-07-25). Reuses the shared on-device figure style
(_e2e_plot_style.py).

    .venv-mlx/bin/python eval/plot_granularity.py \
        --agg results/ondevice_granularity_smollm3_4bit_2026-07-26.json \
        --out results/ondevice/figures/granularity_2026-07-26.pdf
"""
import argparse
from collections import defaultdict

from _e2e_plot_style import THERMAL_COLORS, apply_rc, load_agg, save, plt

AXIS_COLOR = "#3A3A3A"    # neutral charcoal — no single series owns the left axis anymore
MEM_COLOR = "#000000"     # black — distinct from the green/orange/red thermal trio
THERMAL_ORDER = ["nominal", "fair", "serious", "critical"]  # cool -> hot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", default="results/ondevice/figures/granularity.pdf")
    args = ap.parse_args()
    apply_rc()

    agg = load_agg(args.agg)
    cells = sorted(agg["cells"], key=lambda c: c["checkpoint_granularity"])
    completed = [c for c in cells if c["completed"]]
    all_ks = [c["checkpoint_granularity"] for c in completed]
    # Evenly-spaced categorical x positions, one per completed K, in sweep
    # order — with the real K value carried only in the tick label.
    pos_of_k = {k: i for i, k in enumerate(all_ks)}

    xs = [pos_of_k[c["checkpoint_granularity"]] for c in completed]
    mem = [c["peak_mem_mb_mean"] for c in completed]

    # Per-(K, thermal_state) mean iter/s, from the raw per-window records —
    # this decomposes the overall per-K mean/SD into what's actually driving
    # it: thermal state, not checkpoint granularity per se.
    by_state = defaultdict(dict)  # thermal_state -> {x: (mean_ips, n)}
    for c, x in zip(completed, xs):
        vals = defaultdict(list)
        for w in c["windows"]:
            vals[w["thermal_state"]].append(w["iter_per_sec"])
        for state, v in vals.items():
            by_state[state][x] = (sum(v) / len(v), len(v))

    fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
    ax2 = ax1.twinx()

    for state in THERMAL_ORDER:
        if state not in by_state:
            continue
        pts = sorted(by_state[state].items())
        pxs = [p[0] for p in pts]
        pys = [p[1][0] for p in pts]
        ns = [p[1][1] for p in pts]
        color = THERMAL_COLORS[state]
        ax1.plot(
            pxs, pys, "o-", color=color, markersize=6, markeredgewidth=0.6,
            markeredgecolor="white", lw=1.6, zorder=3,
        )
        n_lo, n_hi = min(ns), max(ns)
        n_label = f"n={n_lo}" if n_lo == n_hi else f"n={n_lo}–{n_hi}"
        ax1.plot([], [], "o-", color=color, markersize=6, lw=1.6,
                  label=f"iter/s, {state} ({n_label})")
    ax2.plot(
        xs, mem, "s--", color=MEM_COLOR, markersize=6, markeredgewidth=0.6,
        markeredgecolor="white", lw=1.6, zorder=3, label="peak mem, MB (right axis)",
    )

    ax1.set_xlabel("Checkpoint Group Size")
    ax1.set_ylabel("iterations / second", color=AXIS_COLOR)
    ax2.set_ylabel("peak memory (MB)", color=MEM_COLOR)
    ax1.tick_params(axis="y", labelcolor=AXIS_COLOR)
    ax2.tick_params(axis="y", labelcolor=MEM_COLOR)
    ax2.grid(False)

    ax1.set_xlim(-0.5, len(all_ks) - 0.5)
    ax1.set_xticks(range(len(all_ks)))
    ax1.set_xticklabels([str(k) for k in all_ks])
    ax1.set_title("GC Granularity vs throughput/memory")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=2, frameon=True, facecolor="white", framealpha=0.8,
        edgecolor="lightgrey", labelcolor="dimgrey", fontsize=8,
    )

    save(fig, args.out)


if __name__ == "__main__":
    main()
