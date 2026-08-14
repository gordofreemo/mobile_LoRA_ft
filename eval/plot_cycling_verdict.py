#!/usr/bin/env python3
"""The burst-cycling verdict across kernel and thermal conditions.

One panel: per-burst mean seconds/iteration (600 s burst / iterations
completed) for the three sustained-cycling sessions --- broken kernel
(2026-07-28), repaired kernel from a cool overnight chassis (2026-08-14
04:49), repaired kernel hot-started at midday against a same-hour continuous
reference (2026-08-14 11:10) --- each with its continuous-training reference
as a dashed line in the same color. Bursts above their line lose to
continuous; below it, win. The story: the broken kernel loses decisively,
the repaired kernel straddles its line depending on chassis/ambient state.

    .venv-mlx/bin/python eval/plot_cycling_verdict.py \
        --out results/ondevice/figures/cycling_verdict_2026-08-14.pdf
"""
import argparse
import json
from collections import Counter

from _e2e_plot_style import apply_rc, save, plt

# (label, jsonl with the cycle session, continuous ref s/iter, ref provenance)
SESSIONS = [
    ("broken kernel, evening",
     "results/ondevice/train_bench_metrics_thermal_2026-07-28.jsonl",
     9.979, "60-min-soak plateau, n=3 sessions", "#0072B2"),
    ("repaired, cool chassis (05h)",
     "results/ondevice/train_bench_metrics_thermal_nax-on_2026-08-14.jsonl",
     6.721, "60-min-soak plateau, runs A+B", "#D55E00"),
    ("repaired, hot start (11h)",
     "results/ondevice/train_bench_metrics_thermal_nax-on_2026-08-14b.jsonl",
     7.229, "same-hour 75-min continuous block", "#009E73"),
]


def burst_siter(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    starts = [r for r in rows if r["record_type"] == "cycle_run_start"]
    sid = starts[-1]["bench_session_id"]
    # cycle_burst rows carry probe_index = burst number (0-based);
    # cycle_burst_end carries cycle_index + the actual burst duration.
    iters = Counter(
        r["probe_index"] for r in rows
        if r.get("bench_session_id") == sid and r["record_type"] == "cycle_burst")
    ends = {r["cycle_index"]: r.get("burst_duration_s", 600.0)
            for r in rows
            if r.get("bench_session_id") == sid and r["record_type"] == "cycle_burst_end"}
    return [ends.get(i, 600.0) / iters[i] for i in sorted(iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ondevice/figures/cycling_verdict.pdf")
    args = ap.parse_args()
    apply_rc()

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    for label, path, ref, _prov, color in SESSIONS:
        ys = burst_siter(path)
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, "o-", color=color, markersize=4, label=label)
        ax.axhline(ref, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("burst index (10 min on / 2 min off)")
    ax.set_ylabel("seconds per iteration (per-burst mean)")
    ax.set_title("Burst-cycling vs continuous training (dashed = continuous reference)")
    ax.legend(frameon=False, loc="center right", fontsize=7.5)
    save(fig, args.out)


if __name__ == "__main__":
    main()
