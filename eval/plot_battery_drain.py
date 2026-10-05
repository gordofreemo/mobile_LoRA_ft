#!/usr/bin/env python3
"""P5 battery drain: battery % vs elapsed for the unplugged (C2) run. The only
real energy measurement (per-process power isn't exposed; drain-on-battery is the
proxy). Slope -> energy per step -> energy per adapter.

  python eval/plot_battery_drain.py --agg <agg.json> \
      --out results/ondevice/figures/e2e_battery_drain.pdf
"""
import argparse

from _e2e_plot_style import apply_rc, load_agg, real_completed, label, save, plt


def linfit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    return a, my - a * mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", default="results/ondevice/figures/e2e_battery_drain.pdf")
    args = ap.parse_args()
    apply_rc()

    agg = load_agg(args.agg)
    runs = [r for r in real_completed(agg) if str(r["condition"]).startswith("C2")]
    if not runs:
        raise SystemExit("no completed C2 (unplugged) run to plot")
    run = runs[0]
    fp = run["user_fingerprint"]
    bc = [b for b in run["battery_curve"]
          if b.get("charging") is False and isinstance(b.get("level"), (int, float))
          and b["level"] >= 0 and b.get("elapsed_s") is not None]
    mins = [b["elapsed_s"] / 60 for b in bc]
    pct = [b["level"] * 100 for b in bc]
    drained = pct[0] - pct[-1]
    slope, intercept = linfit(mins, pct)  # %/min
    per_hr = -slope * 60

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(mins, pct, color="#009E73", lw=1.6, label=f"{label(fp)} (measured)")
    ax.plot([0, mins[-1]], [intercept, intercept + slope * mins[-1]],
            color="#444", lw=1.0, ls="--", label=f"fit: {per_hr:.1f}%/h")
    ax.set_xlabel("elapsed (min)")
    ax.set_ylabel("battery (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, mins[-1] * 1.02)
    ax.set_title(f"unplugged drain: {drained:.0f}% for one {label(fp)} adapter")
    ax.legend(loc="lower left")
    save(fig, args.out)


if __name__ == "__main__":
    main()
