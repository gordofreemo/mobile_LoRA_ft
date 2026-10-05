#!/usr/bin/env python3
"""
Progress monitor for the h6 background-scheduled (BGProcessingTask) on-device
LoRA training round. Regenerated after every `devicectl` pull (daily + ad hoc,
per the design plan's Monitoring section — experiments/2026-07-13-ondevice-
bg-training-plan.md).

Reads `Documents/train_bench_metrics_e2e_bg.jsonl` (h6 schema, pulled locally
via `devicectl device copy from ... --domain-identifier mlx.LLMEvalJGW9U9Y36Y`)
plus the companion `Documents/bg_run_meta.json` (per-wake summaries + a rolling
run-level summary, written by the app itself at the end of every wake — see
`LLMEvaluator+BGTrain.swift`'s `appendBGRunMetaWake`). Produces one static
multi-panel PNG, opened in Preview — no new artifact-refresh workflow needed:

  1. iterations completed vs total (bar)
  2. loss curve so far, wake boundaries annotated. KNOWN, ACCEPTED DEVIATION:
     Adam's optimizer moments reset to zero every wake (see
     TrainBenchConstants.bgAppBuild's doc comment) — real small restart bumps
     at wake boundaries are expected here, not a bug; the boundaries are
     annotated rather than hidden so that's visible at a glance.
  3. wake timeline: gap since the previous wake, per wake
  4. battery level at wake end, coloured by thermal state, per wake

Mirrors eval/e2e_aggregate.py / eval/plot_thermal_stress.py conventions:
stdlib + matplotlib only (no pandas on the Mac MLX venv), flat JSON records
in. Unlike those two, aggregation and plotting are combined into one script
here since the whole point is "pull, then immediately regenerate one figure."

Usage:
    python eval/bg_progress.py \
        results/ondevice/train_bench_metrics_e2e_bg_2026-07-20.jsonl \
        --run-meta results/ondevice/bg_run_meta_2026-07-20.json \
        --out results/ondevice/figures/bg_progress_2026-07-20.png
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THERMAL_COLORS = {
    "nominal": "#2ca02c",
    "fair": "#ff7f0e",
    "serious": "#d62728",
    "critical": "#7f0000",
    "unknown": "#7f7f7f",
}


def load_records(paths):
    recs = []
    # The device JSONL is cumulative, so successive pulls overlap — dedup by
    # exact line content (same convention as eval/e2e_aggregate.py).
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


def load_run_meta(path):
    if not path or not Path(path).exists():
        return None
    text = Path(path).read_text().strip()
    if not text:
        # Present but empty: no wake has completed a checkpoint yet (the
        # per-chunk upsert only writes after the first successful chunk) —
        # treat the same as "no file" rather than crashing on empty JSON.
        return None
    return json.loads(text)


def plot_progress(records, run_meta, out, validation_only):
    records = [r for r in records if bool(r.get("validation")) == validation_only]
    if not records:
        print(f"No matching records (validation={validation_only}) to plot.", file=sys.stderr)
        sys.exit(1)

    train = [r for r in records if r["record_type"] == "train"]
    wake_ends = [r for r in records if r["record_type"] in ("wake_end", "error")]

    meta = records[0]
    iterations_total = meta.get("iterations_total")
    iterations_completed = max([t["step"] for t in train], default=0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_bar, ax_loss, ax_gap, ax_therm = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel 1: iterations completed vs total.
    frac = (iterations_completed / iterations_total) if iterations_total else 0.0
    ax_bar.barh(["progress"], [iterations_completed], color="#1f77b4")
    ax_bar.barh(
        ["progress"], [max((iterations_total or 0) - iterations_completed, 0)],
        left=iterations_completed, color="#dddddd")
    ax_bar.set_xlim(0, iterations_total or 1)
    ax_bar.set_title(f"Iterations: {iterations_completed}/{iterations_total} ({frac * 100:.0f}%)")
    ax_bar.set_xlabel("iteration")

    # Panel 2: loss curve so far, wake boundaries annotated (not hidden — see
    # module docstring on the accepted moment-reset-per-wake deviation).
    if train:
        steps = [t["step"] for t in train]
        losses = [t["training_loss"] for t in train]
        ax_loss.plot(steps, losses, "-o", ms=3, color="#1f77b4", lw=1)
        seen_wakes = set()
        for t in train:
            wn = t.get("wake_number")
            if wn not in seen_wakes:
                seen_wakes.add(wn)
                ax_loss.axvline(t["step"], color="0.7", ls="--", lw=0.8, zorder=0)
        ax_loss.set_xlabel("iteration")
        ax_loss.set_ylabel("training loss")
        ax_loss.set_title("Loss so far (dashed = wake boundary; moments reset per wake)")
        ax_loss.grid(True, alpha=0.3)
    else:
        ax_loss.text(0.5, 0.5, "no train records yet", ha="center", va="center",
                      transform=ax_loss.transAxes)

    # Panel 3: wake timeline — gap since the previous wake.
    wakes = (run_meta or {}).get("wakes", [])
    if wakes:
        nums = [w["wakeNumber"] for w in wakes]
        gaps_h = [((w.get("gapSincePreviousWakeS") or 0) / 3600.0) for w in wakes]
        ax_gap.bar(nums, gaps_h, color="#9467bd")
        ax_gap.set_xlabel("wake number")
        ax_gap.set_ylabel("gap since previous wake (hours)")
        ax_gap.set_title(f"Wake timeline (n={len(wakes)} wakes)")
        ax_gap.grid(True, alpha=0.3, axis="y")
    else:
        ax_gap.text(0.5, 0.5, "no wake summaries yet", ha="center", va="center",
                     transform=ax_gap.transAxes)

    # Panel 4: battery at wake end, coloured by thermal state, per wake.
    if wake_ends:
        xs = list(range(len(wake_ends)))
        batt = [w.get("battery_level") for w in wake_ends]
        therm = [w.get("thermal_state", "unknown") for w in wake_ends]
        colors = [THERMAL_COLORS.get(t, THERMAL_COLORS["unknown"]) for t in therm]
        heights = [b * 100 if isinstance(b, (int, float)) and b >= 0 else 0 for b in batt]
        ax_therm.bar(xs, heights, color=colors)
        ax_therm.set_xlabel("wake (chronological)")
        ax_therm.set_ylabel("battery level at wake end (%)")
        ax_therm.set_title("Battery + thermal per wake")
        ax_therm.set_ylim(0, 100)
    else:
        ax_therm.text(0.5, 0.5, "no wake_end records yet", ha="center", va="center",
                       transform=ax_therm.transAxes)

    summary = (run_meta or {}).get("summary") or {}
    fig.suptitle(
        f"h6 BG training progress — user={meta.get('user_fingerprint')}  "
        f"calendar_days={summary.get('totalCalendarDays', 0):.2f}  "
        f"device_compute_h={summary.get('totalDeviceComputeS', 0) / 3600.0:.2f}  "
        f"wakes={summary.get('totalWakes', len(wakes))}  "
        f"completed={summary.get('completed', False)}",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"iterations {iterations_completed}/{iterations_total}  "
          f"wakes={len(wakes)}  calendar_days={summary.get('totalCalendarDays')}  "
          f"cap_hit={summary.get('capHit')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*",
                     help="bg JSONL file(s); default results/ondevice/train_bench_metrics_e2e_bg*.jsonl")
    ap.add_argument("--run-meta",
                     help="bg_run_meta.json path; default latest results/ondevice/bg_run_meta*.json")
    ap.add_argument("--out", default="results/ondevice/figures/bg_progress.png")
    ap.add_argument("--validation", action="store_true",
                     help="plot the Xcode debug-forced validation records instead of the real run")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob("results/ondevice/train_bench_metrics_e2e_bg*.jsonl"))
    if not paths:
        print("No bg telemetry files matched.", file=sys.stderr)
        sys.exit(1)
    records = load_records(paths)
    if not records:
        print("No records loaded.", file=sys.stderr)
        sys.exit(1)

    run_meta_path = args.run_meta or next(
        iter(sorted(glob.glob("results/ondevice/bg_run_meta*.json"), reverse=True)), None)
    run_meta = load_run_meta(run_meta_path)

    plot_progress(records, run_meta, args.out, validation_only=args.validation)


if __name__ == "__main__":
    main()
