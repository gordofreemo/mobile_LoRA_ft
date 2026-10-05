#!/usr/bin/env python3
"""
Aggregate on-device LoRA-TRAINING benchmark telemetry
(Documents/train_bench_metrics.jsonl pulled off the phone) into per-(batch-size)
cost summaries.

Mirrors eval/bench_aggregate.py: stdlib only (no pandas on the Mac MLX venv),
flat JSON in, descriptive stats out. This is the analysis step of the locked
design in experiments/2026-06-29-ondevice-training-naive-plan.md.

Goal (decision 1): a COST BASELINE — raw time / memory / thermal cost of naive
on-device LoRA fine-tuning, per batch size, so later systems-optimized variants
have a comparison floor. Descriptive only — mean ± std (n), min/max. No tests.

Grouping key = (bench_session_id, batch_size). A run "cell" is one 200-step
training run at a fixed batch size; `train_bench_metrics.jsonl` APPENDS across
launches, so multiple sessions can coexist in one file.

Notes on two implementation-forced quirks (see harness header):
  * The first report window (step == steps_per_report) folds in the one
    forced iteration-0 validation forward pass — its iter/s and peak are
    slightly inflated. `--drop-first-window` excludes it from the throughput/
    memory stats (kept in the raw decay series).
  * Battery is sampled at cell boundaries on the main actor (UIDevice is
    @MainActor), so every record in a cell carries the same `battery_level`
    (cell start) and `battery_level_end` (cell end). Per-cell battery delta =
    battery_level_end - battery_level.

Usage:
    python eval/train_bench_aggregate.py results/ondevice/train_bench_metrics_*.jsonl \
        --out results/ondevice/train_bench_agg_smollm3_4bit_naive_2026-06-29.json
"""

import argparse
import glob
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# Per-cell throughput/memory metrics summarized across windows.
METRICS = ["iter_per_sec", "tok_per_sec", "peak_mem_bytes"]


def load_records(paths):
    records = []
    for p in paths:
        for ln, line in enumerate(Path(p).read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! skip {p}:{ln}: {e}", file=sys.stderr)
    return records


def describe(values):
    """mean / std / min / max / n for a list of numbers (std=0 when n<2)."""
    vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def _counts(it):
    out = {}
    for v in it:
        out[v] = out.get(v, 0) + 1
    return out


def summarize_cells(records, drop_first_window):
    """One summary per (session, batch_size, seq_cap) training cell."""
    train = [r for r in records if r.get("record_type") == "train"]
    groups = {}
    for r in train:
        key = (r.get("bench_session_id"), r.get("batch_size"), r.get("seq_cap"))
        groups.setdefault(key, []).append(r)

    cells = []
    for (session, bs, seq_cap), windows in sorted(
        groups.items(),
        key=lambda kv: (kv[0][2] if kv[0][2] is not None else 0,
                        kv[0][1] if kv[0][1] is not None else 0),
    ):
        windows.sort(key=lambda r: r.get("step") or 0)
        steps_per_report = next(
            (w.get("steps_per_report") for w in windows if w.get("steps_per_report")), None
        )
        stat_windows = windows
        if drop_first_window and steps_per_report is not None:
            stat_windows = [w for w in windows if w.get("step") != steps_per_report]
            if not stat_windows:  # degenerate: keep all rather than emit empty
                stat_windows = windows

        # battery delta (cell end - cell start); same on every record in a cell.
        bl0 = next((w.get("battery_level") for w in windows), None)
        bl1 = next((w.get("battery_level_end") for w in windows), None)
        battery_delta = (
            (bl1 - bl0)
            if isinstance(bl0, (int, float)) and isinstance(bl1, (int, float))
            and bl0 >= 0 and bl1 >= 0
            else None
        )

        elapsed = [w.get("elapsed_s") for w in windows if isinstance(w.get("elapsed_s"), (int, float))]
        out = {
            "bench_session_id": session,
            "batch_size": bs,
            "seq_cap": seq_cap,
            "n_windows": len(windows),
            "n_windows_in_stats": len(stat_windows),
            "dropped_first_window": drop_first_window,
            "iterations_total": next((w.get("iterations_total") for w in windows), None),
            "steps_per_report": steps_per_report,
            "num_train_examples": next((w.get("num_train_examples") for w in windows), None),
            "total_elapsed_s": max(elapsed) if elapsed else None,
            "thermal_states": _counts(w.get("thermal_state") for w in windows),
            "last_thermal_state": windows[-1].get("thermal_state") if windows else None,
            "battery_level_start": bl0,
            "battery_level_end": bl1,
            "battery_delta": battery_delta,
            "charging": next((w.get("charging") for w in windows), None),
            "low_power_mode_any": any(bool(w.get("low_power_mode")) for w in windows),
        }
        for m in METRICS:
            out[m] = describe([w.get(m) for w in stat_windows])
        # peak in MB for convenience.
        pk = out["peak_mem_bytes"]
        out["peak_mem_mb_mean"] = pk["mean"] / (1024 * 1024) if pk["mean"] is not None else None
        out["peak_mem_mb_max"] = pk["max"] / (1024 * 1024) if pk["max"] is not None else None
        # per-window decay series (full, undropped) for thermal/throughput plots.
        out["windows"] = [
            {
                "step": w.get("step"),
                "iter_per_sec": w.get("iter_per_sec"),
                "tok_per_sec": w.get("tok_per_sec"),
                "elapsed_s": w.get("elapsed_s"),
                "peak_mem_bytes": w.get("peak_mem_bytes"),
                "thermal_state": w.get("thermal_state"),
            }
            for w in windows
        ]
        cells.append(out)
    return cells


def summarize_feasibility(records):
    """Per (session, batch_size, seq_cap): did it produce train windows?
    A cap with a `cap_start` sentinel but zero `train` rows was SIGKILL'd
    (jetsam) — the uncatchable OOM the do/catch can't see. Returns the feasible
    caps, the OOM'd caps, and the feasible ceiling per (session, batch_size)."""
    starts, has_train = set(), set()
    for r in records:
        key = (r.get("bench_session_id"), r.get("batch_size"), r.get("seq_cap"))
        if r.get("record_type") == "cap_start":
            starts.add(key)
        elif r.get("record_type") == "train":
            has_train.add(key)
    all_keys = starts | has_train
    feasible = sorted(k for k in all_keys if k in has_train)
    oom = sorted(k for k in all_keys if k not in has_train)  # started, no train rows
    ceilings = {}
    for (s, bs, cap) in feasible:
        if cap is None:
            continue
        ck = (s, bs)
        ceilings[ck] = max(ceilings.get(ck, 0), cap)
    return {
        "feasible_cells": [
            {"bench_session_id": s, "batch_size": bs, "seq_cap": cap}
            for (s, bs, cap) in feasible
        ],
        "oom_cells": [
            {"bench_session_id": s, "batch_size": bs, "seq_cap": cap}
            for (s, bs, cap) in oom
        ],
        "feasible_ceiling_seq_cap": [
            {"bench_session_id": s, "batch_size": bs, "max_feasible_seq_cap": cap}
            for (s, bs), cap in sorted(ceilings.items())
        ],
    }


def summarize_ooms(records):
    ooms = [r for r in records if r.get("record_type") == "oom"]
    return [
        {
            "bench_session_id": r.get("bench_session_id"),
            "batch_size": r.get("batch_size"),
            "thermal_state": r.get("thermal_state"),
        }
        for r in ooms
    ]


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def print_report(agg):
    print(
        f"\nOn-device TRAINING benchmark aggregate — {agg['n_records']} records "
        f"from {len(agg['source_files'])} file(s)"
    )
    print(
        f"app_build(s): {', '.join(agg.get('app_builds') or [])}   "
        f"session(s): {len(agg.get('bench_session_ids') or [])}   "
        f"model: {agg.get('model')}"
    )
    lora = agg.get("lora") or {}
    print(
        f"LoRA: rank={lora.get('rank')} keys={lora.get('keys')} "
        f"layers={lora.get('num_layers')}"
    )

    print("\nPer-(seq_cap,batch_size) training cells:")
    hdr = (
        f"  {'cap':>5} {'bs':>3} {'win':>4} {'iter/s':>14} {'tok/s':>16} "
        f"{'peak_MB(mean/max)':>20} {'elapsed_s':>10} {'thermal(last)':>14} {'batt Δ':>8}"
    )
    print(hdr)
    for c in agg["cells"]:
        it = c["iter_per_sec"]
        tk = c["tok_per_sec"]
        print(
            f"  {str(c.get('seq_cap')):>5} {str(c['batch_size']):>3} {c['n_windows']:>4} "
            f"{fmt(it['mean']):>6}±{fmt(it['std']):<6} "
            f"{fmt(tk['mean'],1):>7}±{fmt(tk['std'],1):<7} "
            f"{fmt(c['peak_mem_mb_mean'],0):>8}/{fmt(c['peak_mem_mb_max'],0):<8} "
            f"{fmt(c['total_elapsed_s'],0):>10} "
            f"{str(c['last_thermal_state']):>14} "
            f"{fmt(c['battery_delta'],3):>8}"
        )

    feas = agg.get("feasibility") or {}
    if feas:
        ceil = feas.get("feasible_ceiling_seq_cap") or []
        print("\nFeasibility (cap_start sentinels vs train rows):")
        for c in ceil:
            print(
                f"  bs={c['batch_size']}: max feasible seq_cap = "
                f"{c['max_feasible_seq_cap']} tok"
            )
        oom = feas.get("oom_cells") or []
        if oom:
            print(
                "  OOM (jetsam, started but no train rows): "
                + ", ".join(f"cap={o['seq_cap']}@bs{o['batch_size']}" for o in oom)
            )
    if agg.get("ooms"):
        print(f"\nCaught-OOM records: {agg['ooms']}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "paths", nargs="*", help="JSONL telemetry file(s); default results/ondevice/train_bench_metrics_*.jsonl"
    )
    ap.add_argument("--out", help="write aggregated JSON here")
    ap.add_argument(
        "--drop-first-window",
        action="store_true",
        help="exclude the first report window (folds in the forced iter-0 "
        "validation forward pass) from throughput/memory stats",
    )
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob("results/ondevice/train_bench_metrics_*.jsonl"))
    if not paths:
        print("No telemetry files matched.", file=sys.stderr)
        sys.exit(1)

    records = load_records(paths)
    if not records:
        print("No records loaded.", file=sys.stderr)
        sys.exit(1)

    def first(field):
        return next((r.get(field) for r in records if r.get(field) is not None), None)

    agg = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(p) for p in paths],
        "n_records": len(records),
        "app_builds": sorted({r.get("app_build") for r in records if r.get("app_build")}),
        "bench_session_ids": sorted(
            {r.get("bench_session_id") for r in records if r.get("bench_session_id")}
        ),
        "device_model": first("device_model"),
        "os_version": first("os_version"),
        "model": first("model"),
        "lora": {
            "rank": first("lora_rank"),
            "keys": first("lora_keys"),
            "num_layers": first("num_lora_layers"),
        },
        "cells": summarize_cells(records, args.drop_first_window),
        "feasibility": summarize_feasibility(records),
        "ooms": summarize_ooms(records),
    }

    print_report(agg)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
