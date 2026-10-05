#!/usr/bin/env python3
"""On-device training loss vs epoch, one line per user. Shows the User-LoRAs
converge on-device (the fidelity signal). Single axis. Absolute loss is
full-sequence (not comparable to the cluster's assistant-only loss), so this is
convergence shape, not an absolute-value overlay.

  python eval/plot_loss_curves.py --agg <agg.json> \
      --out results/ondevice/figures/e2e_loss_curves.pdf
"""
import argparse

from _e2e_plot_style import (apply_rc, load_agg, real_completed, label,
                             USER_COLORS, save, plt)


def moving_avg(ys, w=7):
    """Centered moving average (window w), edge-shrinking so no NaN padding."""
    n = len(ys)
    out = []
    h = w // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        seg = ys[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", default="results/ondevice/figures/e2e_loss_curves.pdf")
    args = ap.parse_args()
    apply_rc()

    agg = load_agg(args.agg)
    runs = [r for r in real_completed(agg) if str(r["condition"]).startswith("C0")]
    runs.sort(key=lambda r: r["profile_size"])
    # one curve per user (first C0 run) so repeated runs don't double-draw
    seen, uniq = set(), []
    for r in runs:
        if r["user_fingerprint"] not in seen:
            seen.add(r["user_fingerprint"])
            uniq.append(r)
    runs = uniq

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for r in runs:
        fp = r["user_fingerprint"]
        col = USER_COLORS.get(fp, "#555")
        curve = [c for c in r["loss_curve"] if c.get("epoch") is not None]
        ep = [c["epoch"] for c in curve]
        loss = [c["loss"] for c in curve]
        ax.plot(ep, loss, color=col, lw=0.6, alpha=0.22, zorder=1)          # raw
        ax.plot(ep, moving_avg(loss), color=col, lw=1.6, label=label(fp), zorder=2)  # smoothed
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("on-device User-LoRA convergence")
    ax.legend(title="user", loc="upper right")
    save(fig, args.out)


if __name__ == "__main__":
    main()
