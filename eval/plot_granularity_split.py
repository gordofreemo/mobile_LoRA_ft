#!/usr/bin/env python3
"""h8 granularity sweep as two independent single-column figures."""
import argparse
from _e2e_plot_style import apply_rc, load_agg, save, plt
COLOR, DEAD = "#0072B2", "#B00020"

def series(agg):
    ok, dead = [], []
    for c in agg["cells"]:
        k = c["checkpoint_granularity"]
        if c.get("completed") and not c.get("oom"):
            ips = c["iter_per_sec"]["mean"] if isinstance(c["iter_per_sec"], dict) else c["iter_per_sec"]
            ok.append((k, ips, c.get("peak_mem_mb_max") or c.get("peak_mem_mb_mean")))
        else:
            dead.append(k)
    ok.sort(); return ok, sorted(dead)

def axfmt(ax):
    ax.set_xscale("log")
    allk = [1, 2, 3, 4, 6, 9, 12, 18, 36]
    ax.set_xticks(allk); ax.set_xticklabels([str(k) for k in allk])
    ax.minorticks_off(); ax.set_xlabel("checkpoint group size $K$ (blocks)")

ap = argparse.ArgumentParser()
ap.add_argument("--agg", required=True); ap.add_argument("--outdir", required=True)
a = ap.parse_args(); apply_rc()
ok, dead = series(load_agg(a.agg))
ks = [k for k, _, _ in ok]

f1, ax = plt.subplots(figsize=(3.3, 2.5))
ax.plot(ks, [m for _, _, m in ok], "o-", color=COLOR, markersize=4)
axfmt(ax); ax.set_ylabel("peak memory (MB)"); ax.set_ylim(0, 6800)
ax.axhline(6000, color="0.4", ls="--", lw=0.8)
ax.text(1.05, 6120, "iOS limit, about 6 GB", fontsize=7, color="0.3", va="bottom")
ax.axvline(9, color=DEAD, ls="--", lw=0.8)
ax.text(9.8, 3000, "terminated\npast this point", fontsize=7, color=DEAD, ha="left", va="center")
save(f1, f"{a.outdir}/gc_memory.pdf")

f2, ax = plt.subplots(figsize=(3.3, 2.5))
ax.plot(ks, [i for _, i, _ in ok], "o-", color=COLOR, markersize=4)
for k in dead:
    ax.plot(k, 0, "x", color=DEAD, markersize=6)
ax.set_ylim(bottom=-0.004)
axfmt(ax); ax.set_ylabel("iterations / s")
ax.annotate("terminated", xy=(9, 0), xytext=(11, 0.022), color=DEAD, fontsize=7,
            ha="center", arrowprops=dict(arrowstyle="->", color=DEAD, lw=0.7))
save(f2, f"{a.outdir}/gc_throughput.pdf")
