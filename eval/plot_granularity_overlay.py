#!/usr/bin/env python3
"""NAX off/on overlay of the h8 gradient-checkpointing granularity sweep.

Two panels: (left) throughput vs K for both arms — the axis the kernel fix
moves; (right) peak memory vs K for both arms — expected INVARIANT (the fix
changes dispatch, not allocation), shown as the built-in validity check.
Jetsammed cells (K>=9) marked at y=0 with an x, as in the original figure.

    .venv-mlx/bin/python eval/plot_granularity_overlay.py \
        --off results/ondevice_granularity_smollm3_4bit_2026-07-26.json \
        --on  <rerun agg> --out results/ondevice/figures/granularity_overlay_<date>.pdf
"""
import argparse

from _e2e_plot_style import apply_rc, load_agg, save, plt

COLORS = {"off": "#0072B2", "on": "#D55E00"}


def series(agg):
    ok, dead = [], []
    for c in agg["cells"]:
        k = c["checkpoint_granularity"]
        if c.get("completed") and not c.get("oom"):
            ips = c["iter_per_sec"]["mean"] if isinstance(c["iter_per_sec"], dict) \
                else c["iter_per_sec"]
            mem = c.get("peak_mem_mb_max") or c.get("peak_mem_mb_mean")
            ok.append((k, ips, mem))
        else:
            dead.append(k)
    ok.sort()
    return ok, sorted(dead)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--out", default="results/ondevice/figures/granularity_overlay.pdf")
    args = ap.parse_args()
    apply_rc()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for arm, path in (("off", args.off), ("on", args.on)):
        ok, dead = series(load_agg(path))
        ks = [k for k, _, _ in ok]
        ax1.plot(ks, [i for _, i, _ in ok], "o-", color=COLORS[arm], markersize=4,
                 label=f"NAX {arm}")
        ax2.plot(ks, [m for _, _, m in ok], "o-", color=COLORS[arm], markersize=4,
                 label=f"NAX {arm}")
        for k in dead:
            ax1.plot(k, 0, "x", color=COLORS[arm], markersize=6)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        allk = [1, 2, 3, 4, 6, 9, 12, 18, 36]
        ax.set_xticks(allk)
        ax.set_xticklabels([str(k) for k in allk])
        ax.minorticks_off()
        ax.set_xlabel("checkpoint group size K (blocks)")
    ax1.set_ylabel("iterations / s")
    ax1.set_title("throughput (x = jetsam)")
    ax1.legend(frameon=False)
    ax2.set_ylabel("peak memory (MB)")
    ax2.set_title("peak memory (validity: should overlap)")
    save(fig, args.out)


if __name__ == "__main__":
    main()
