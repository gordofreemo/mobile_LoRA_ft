#!/usr/bin/env python3
"""Cost law: wall-time vs profile-size (weak) ‖ vs total-tokens (linear).

Two panels sharing the y-axis (wall hours). Shows the headline: on-device
training cost is token-bound, not profile-bound.

  python eval/plot_cost_law.py --agg results/ondevice_e2e_smollm3_a1lamp_2026-07-07.json \
      --out results/ondevice/figures/e2e_cost_law.pdf
"""
import argparse

from _e2e_plot_style import (apply_rc, load_agg, real_completed, label,
                             USER_COLORS, save, plt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", default="results/ondevice/figures/e2e_cost_law.pdf")
    args = ap.parse_args()
    apply_rc()

    agg = load_agg(args.agg)
    runs = [r for r in real_completed(agg)
            if str(r["condition"]).startswith("C0") and r.get("total_tokens_est")]
    # one point per user (mean if repeated)
    by_user = {}
    for r in runs:
        by_user.setdefault(r["user_fingerprint"], []).append(r)
    pts = []
    for fp, rs in by_user.items():
        pts.append((fp, rs[0]["profile_size"],
                    sum(x["total_tokens_est"] for x in rs) / len(rs),
                    sum(x["wall_time_s"] for x in rs) / len(rs) / 3600.0))

    dv = agg["derived"]
    fig, (axp, axt) = plt.subplots(1, 2, figsize=(6.2, 2.9), sharey=True)

    # dodge label offsets in x-sorted order so nearby points don't collide
    def dodge(i):
        return (5, 3) if i % 2 == 0 else (5, -11)

    pts_by_prof = sorted(pts, key=lambda p: p[1])
    for i, (fp, prof, _, hrs) in enumerate(pts_by_prof):
        axp.scatter(prof, hrs, s=42, color=USER_COLORS.get(fp, "#555"), zorder=3)
        axp.annotate(label(fp), (prof, hrs), textcoords="offset points",
                     xytext=dodge(i), fontsize=8)
    axp.set_xlabel("profile size (entries)")
    axp.set_ylabel("wall-time (h)")
    rp = (dv.get("cost_fit") or {}).get("pearson_r")
    axp.set_title(f"vs profile   r={rp:.2f}" if rp is not None else "vs profile")

    xs = [t / 1e6 for _, _, t, _ in pts]
    pts_by_tok = sorted(pts, key=lambda p: p[2])
    for i, (fp, _, tok, hrs) in enumerate(pts_by_tok):
        axt.scatter(tok / 1e6, hrs, s=42, color=USER_COLORS.get(fp, "#555"), zorder=3)
        axt.annotate(label(fp), (tok / 1e6, hrs), textcoords="offset points",
                     xytext=dodge(i), fontsize=8)
    tf = dv.get("cost_fit_tokens")
    if tf and len(xs) >= 2:
        lo, hi = min(xs), max(xs)
        a, b = tf["slope_s_per_token"], tf["intercept_s"]
        axt.plot([lo, hi], [(a * lo * 1e6 + b) / 3600, (a * hi * 1e6 + b) / 3600],
                 color="#444", lw=1.2, zorder=2)
        axt.set_title(f"vs tokens   r={tf['pearson_r']:.2f}")
    axt.set_xlabel("total tokens (M, 3 epochs)")

    save(fig, args.out)


if __name__ == "__main__":
    main()
