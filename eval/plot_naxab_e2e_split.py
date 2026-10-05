#!/usr/bin/env python3
"""NAX A/B end-to-end, as two independent single-column figures.

Splits the former two-panel figure so each plot can be placed separately:
  naxab_throughput.pdf  iterations/sec vs elapsed wall-clock (the cost result)
  naxab_loss.pdf        training loss vs step (the correctness result)

Both arms consume an identical batch sequence (LoRATrain.shuffleSeed), so the
loss curves are comparable step-by-step and any divergence is attributable to
kernel numerics rather than data order.

    python eval/plot_naxab_e2e_split.py <jsonl> --outdir results/ondevice/figures
"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, seaborn as sns
from _e2e_plot_style import apply_rc

COLOR = {"off": "#0072B2", "on": "#D55E00"}   # Okabe-Ito, shared with the other figures
STYLE = {"off": "-", "on": "--"}
LABEL = {"off": "baseline (stock MLX)", "on": "NAX backward (patched)"}
LEG   = dict(loc="upper right", frameon=False, borderpad=0.4, handlelength=1.6)


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    tr = [r for r in rows if r.get("record_type") == "train" and r.get("nax_arm")]
    if not tr:
        raise SystemExit(f"no arm-tagged train records in {path}")
    return pd.DataFrame(tr).dropna(subset=["elapsed_s", "iter_per_sec", "training_loss"])


def roll(y, frac=0.06):
    """Centred rolling mean. min_periods = full window so the truncated
    windows at each end are dropped rather than averaged over fewer points,
    which otherwise amplifies whatever the last few samples happen to do."""
    w = max(3, int(len(y) * frac))
    return pd.Series(y).rolling(w, center=True, min_periods=w).mean().values


def style(ax):
    """House style, matching _e2e_plot_style.apply_rc used by the other figures."""
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--outdir", default="results/ondevice/figures")
    a = ap.parse_args()
    df = load(a.jsonl)
    arms = [x for x in ("off", "on") if x in set(df["nax_arm"])]
    apply_rc()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    # ---------------- throughput ----------------
    f1, ax = plt.subplots(figsize=(3.3, 2.5))
    hi = 0.0
    for arm in arms:
        d = df[df["nax_arm"] == arm].sort_values("step")
        hrs, ips = d["elapsed_s"].values / 3600.0, d["iter_per_sec"].values
        ax.plot(hrs, ips, color=COLOR[arm], alpha=0.20, lw=0.7, zorder=2)
        ax.plot(hrs, roll(ips), color=COLOR[arm], ls=STYLE[arm], lw=1.8,
                zorder=4, label=LABEL[arm])
        # label each curve's total wall-clock at its own endpoint, offset
        # away from the trace so it never sits on top of either arm
        yend = roll(ips)[~np.isnan(roll(ips))][-1]
        dy = 9
        ax.annotate(f"{hrs[-1]:.2f} h", xy=(hrs[-1], yend),
                    xytext=(2, dy), textcoords="offset points", ha="right",
                    va="center", fontsize=7, color=COLOR[arm])
        hi = max(hi, np.nanmax(ips))
    ax.set_ylim(top=hi * 1.34)          # headroom so the legend clears the data
    ax.set_xlabel("elapsed wall-clock (hours)")
    ax.set_ylabel("iterations / s")
    style(ax); ax.legend(**LEG)
    sns.despine(); f1.tight_layout()
    f1.savefig(out / "naxab_throughput.pdf", bbox_inches="tight")
    f1.savefig(out / "naxab_throughput.png", dpi=200, bbox_inches="tight")

    # ---------------- loss ----------------
    f2, ax = plt.subplots(figsize=(3.3, 2.5))
    for arm in arms:
        d = df[df["nax_arm"] == arm].sort_values("step")
        ax.plot(d["step"], d["training_loss"], color=COLOR[arm], alpha=0.18,
                lw=0.7, zorder=2)
        ax.plot(d["step"], roll(d["training_loss"].values), color=COLOR[arm],
                ls=STYLE[arm], lw=1.8, zorder=4, label=LABEL[arm])
    if len(arms) == 2:
        A = df[df["nax_arm"] == "off"].sort_values("step")
        B = df[df["nax_arm"] == "on"].sort_values("step")
        g = np.union1d(A["step"].values, B["step"].values)
        la = np.interp(g, A["step"].values, roll(A["training_loss"].values))
        lb = np.interp(g, B["step"].values, roll(B["training_loss"].values))
        ax.fill_between(g, la, lb, color="dimgrey", alpha=0.18, zorder=1,
                        label="divergence")
        print(f"  max loss divergence: {np.nanmax(np.abs(la-lb)):.4f}")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    style(ax); ax.legend(**LEG)
    sns.despine(); f2.tight_layout()
    f2.savefig(out / "naxab_loss.pdf", bbox_inches="tight")
    f2.savefig(out / "naxab_loss.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out}/naxab_throughput.pdf and {out}/naxab_loss.pdf")


if __name__ == "__main__":
    main()
