#!/usr/bin/env python3
"""
Fast, focused check-in plot for the h6 background-training round: x = calendar
time since launch (hours), y = how long each real OS wake actually got to
compute before being cut off. Meant to be re-run after every `devicectl` pull
while the round is live — no aggregation state kept, just re-derives
everything from the JSONL each time.

Complements `eval/bg_progress.py` (which covers iteration progress, the loss
curve, and battery/thermal) rather than replacing it — this is the one
panel worth checking on its own, repeatedly, in isolation.

One real OS wake attempt = one `bench_session_id` (NOT `wake_number`, which
only increments on a completed checkpoint — most wakes so far have died
before ever checkpointing, so many wake attempts share the same
`wake_number`; `bench_session_id` is a fresh UUID per `runBGTrainWake()`
call and is the correct grouping key).

How the y-value (time slice) is derived, and what it means, per wake:
  - Reached `wake_end` (graceful) or `error`: the real measured
    `wake_elapsed_s` from that record. Exact.
  - Died silently but left at least one `wake_elapsed_s`-bearing marker
    (model_loaded / lora_apply_start / chunk_start / train / checkpoint /
    resume_start / resumed / training_setup_complete): the MAX
    `wake_elapsed_s` seen for that wake. This is a LOWER BOUND — the wake
    died sometime after that timestamp, exact death time isn't recoverable
    from the JSONL alone (the heartbeat file, `bg_heartbeat.json`, gets
    closer but is overwritten every ~1s so only the CURRENT/latest wake's
    value is ever capturable, not past ones retroactively).
  - Died before even `model_loaded` (no elapsed-bearing marker at all —
    happens rarely, e.g. one wake seen 2026-07-14 that died mid-model-load):
    plotted at a small floor value with a distinct hollow marker, since we
    have literally no duration signal from the JSONL for these — better to
    show the wake happened than silently drop it from the count.

Usage:
    python eval/bg_timeslice.py results/ondevice/train_bench_metrics_e2e_bg_2026-07-13.jsonl \
        --out results/ondevice/figures/bg_timeslice.png
"""
import argparse
import glob
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

UNKNOWN_FLOOR_S = 1.0  # plotted value for wakes with zero elapsed-bearing markers


def load_records(paths):
    recs = []
    # The device JSONL is cumulative across pulls — dedup by exact line
    # content (same convention as eval/e2e_aggregate.py / bg_progress.py).
    seen = set()
    for p in paths:
        for ln, line in enumerate(Path(p).read_text().splitlines(), 1):
            line = line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! skip {p}:{ln}: {e}", file=sys.stderr)
    return recs


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def per_wake_slices(records):
    sessions = {}
    for r in records:
        sid = r.get("bench_session_id")
        if sid is None:
            continue
        sessions.setdefault(sid, []).append(r)

    out = []
    for sid, recs in sessions.items():
        recs.sort(key=lambda r: r.get("timestamp_utc", ""))
        start_rec = next((r for r in recs if r["record_type"] == "wake_start"), recs[0])
        end_rec = next((r for r in recs if r["record_type"] in ("wake_end", "error")), None)
        elapsed_vals = [
            r["wake_elapsed_s"] for r in recs if isinstance(r.get("wake_elapsed_s"), (int, float))
        ]

        if end_rec is not None:
            category = "error" if end_rec["record_type"] == "error" else "graceful"
            time_slice_s = end_rec.get("wake_elapsed_s") or (max(elapsed_vals) if elapsed_vals else UNKNOWN_FLOOR_S)
        elif elapsed_vals:
            category = "lower_bound"
            time_slice_s = max(elapsed_vals)
        else:
            category = "unknown"
            time_slice_s = UNKNOWN_FLOOR_S

        out.append({
            "session_id": sid,
            "wake_start_utc": start_rec.get("timestamp_utc"),
            "time_slice_s": time_slice_s,
            "category": category,
            "termination_reason": (end_rec or {}).get("wake_termination_reason"),
        })
    out.sort(key=lambda w: w["wake_start_utc"] or "")
    return out


STYLE = {
    "graceful": dict(color="#2ca02c", marker="o", label="graceful (wake_end)"),
    "error": dict(color="#d62728", marker="o", label="errored"),
    "lower_bound": dict(color="#7f7f7f", marker="^", label="died silently (lower bound)"),
    "unknown": dict(color="#7f7f7f", marker="x", label=f"died before any marker (floor={UNKNOWN_FLOOR_S:.0f}s)"),
}


def plot(wakes, out, launch_override, title):
    if not wakes:
        print("No wake data to plot.", file=sys.stderr)
        sys.exit(1)

    launch = parse_ts(launch_override) if launch_override else parse_ts(wakes[0]["wake_start_utc"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    categories_present = set()
    xs_all, ys_all = [], []
    for w in wakes:
        t = parse_ts(w["wake_start_utc"])
        x = (t - launch).total_seconds() / 3600.0
        y = w["time_slice_s"]
        xs_all.append(x)
        ys_all.append(y)
        style = STYLE[w["category"]]
        categories_present.add(w["category"])
        if style["marker"] == "x":
            # "x" is an unfilled marker — matplotlib warns if given any
            # edgecolor kwarg at all, so omit it entirely rather than
            # passing a placeholder value.
            ax.scatter(x, y, c=style["color"], marker=style["marker"], s=55, zorder=3)
        else:
            ax.scatter(x, y, c=style["color"], marker=style["marker"], s=55, zorder=3,
                       edgecolors="white", linewidths=0.6)

    ax.plot(xs_all, ys_all, "-", color="0.8", lw=1, zorder=1)

    ax.set_yscale("log")
    ax.set_xlabel("Time since launch (hours)")
    ax.set_ylabel("Granted time slice (seconds, log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    handles = [
        Line2D([0], [0], marker=STYLE[c]["marker"], color="w", markerfacecolor=STYLE[c]["color"],
               markeredgecolor="0.3", markersize=9, label=STYLE[c]["label"])
        for c in ("graceful", "error", "lower_bound", "unknown") if c in categories_present
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    n = len(wakes)
    by_cat = {c: sum(1 for w in wakes if w["category"] == c) for c in STYLE}
    slices = sorted(w["time_slice_s"] for w in wakes)
    print(f"{n} wakes over {xs_all[-1]:.2f}h since launch "
          f"({launch.isoformat()})")
    print(f"  categories: " + ", ".join(f"{c}={n}" for c, n in by_cat.items() if n))
    print(f"  time_slice_s: min={slices[0]:.1f} median={slices[len(slices)//2]:.1f} "
          f"max={slices[-1]:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*",
                     help="bg JSONL file(s); default results/ondevice/train_bench_metrics_e2e_bg*.jsonl")
    ap.add_argument("--out", default="results/ondevice/figures/bg_timeslice.png")
    ap.add_argument("--launch-time",
                     help="ISO8601 UTC override for time-zero (default: first wake_start in the data)")
    ap.add_argument("--title", default="h6 background-training: OS-granted time slice per wake")
    ap.add_argument("--validation", action="store_true",
                     help="plot Xcode debug-forced validation records instead of the real run")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob("results/ondevice/train_bench_metrics_e2e_bg*.jsonl"))
    if not paths:
        print("No bg telemetry files matched.", file=sys.stderr)
        sys.exit(1)
    records = load_records(paths)
    records = [r for r in records if bool(r.get("validation")) == args.validation]
    if not records:
        print("No matching records.", file=sys.stderr)
        sys.exit(1)

    wakes = per_wake_slices(records)
    plot(wakes, args.out, args.launch_time, args.title)


if __name__ == "__main__":
    main()
