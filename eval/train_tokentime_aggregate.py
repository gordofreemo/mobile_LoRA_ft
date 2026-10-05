#!/usr/bin/env python3
"""
Aggregate the h7 token-time telemetry (Documents/train_bench_metrics_tokentime.jsonl
pulled off the phone) into a per-token-count cost summary + a linear cost-model fit.

Mirrors eval/train_bench_aggregate.py: stdlib only (no pandas on the Mac MLX
venv), flat JSON in, descriptive stats out. Locked design:
experiments/2026-07-24-ondevice-tokentime-plan.md (pinned via /grill_me
2026-07-24).

Grouping key = target_tokens (one cell per token count in the sweep grid).
Each cell already excludes iteration 0 (compile warm-up + the forced iter-0
validation pass) — the harness itself never writes that record, so no
--drop-first-window flag is needed here (unlike train_bench_aggregate.py).

Purpose: fit `seconds_per_iter ~= a + b * tokens` from the 6 per-cell means, so
the E2E (h5) cost-vs-profile-size story can predict wall time from a user's
example token-length distribution.

Usage:
    python eval/train_tokentime_aggregate.py results/ondevice/train_bench_metrics_tokentime_*.jsonl \
        --out results/ondevice_tokentime_smollm3_4bit_2026-07-24.json
"""

import argparse
import glob
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

METRICS = ["seconds_per_iter", "tok_per_sec", "peak_mem_bytes"]


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


def summarize_cells(records):
    """One summary per target_tokens cell (across all sessions in the input)."""
    train = [r for r in records if r.get("record_type") == "train"]
    groups = {}
    for r in train:
        groups.setdefault(r.get("target_tokens"), []).append(r)

    cells = []
    for tokens, rows in sorted(groups.items(), key=lambda kv: (kv[0] is None, kv[0])):
        rows.sort(key=lambda r: r.get("iteration") or 0)
        out = {
            "target_tokens": tokens,
            "n_iterations": len(rows),
            "thermal_states": _counts(r.get("thermal_state") for r in rows),
            "first_thermal_state": rows[0].get("thermal_state") if rows else None,
            "last_thermal_state": rows[-1].get("thermal_state") if rows else None,
            "low_power_mode_any": any(bool(r.get("low_power_mode")) for r in rows),
        }
        for m in METRICS:
            out[m] = describe([r.get(m) for r in rows])
        pk = out["peak_mem_bytes"]
        out["peak_mem_mb_mean"] = pk["mean"] / (1024 * 1024) if pk["mean"] is not None else None
        # raw per-iteration series, for a within-cell trend check (e.g. thermal
        # creep within a cell — expected under the locked no-cooldown design).
        out["iterations"] = [
            {
                "iteration": r.get("iteration"),
                "seconds_per_iter": r.get("seconds_per_iter"),
                "thermal_state": r.get("thermal_state"),
            }
            for r in rows
        ]
        cells.append(out)
    return cells


def linear_fit(points):
    """OLS y = a + b*x over (x, y) points. Returns None if <2 distinct x."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(points)
    if n < 2 or len(set(xs)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in points)
    syy = sum((y - my) ** 2 for y in ys)
    b = sxy / sxx
    a = my - b * mx
    r = sxy / (sxx * syy) ** 0.5 if syy > 0 else None
    residuals = [y - (a + b * x) for x, y in points]
    residual_std = statistics.stdev(residuals) if n > 2 else 0.0
    return {
        "intercept_s": a,
        "slope_s_per_token": b,
        "pearson_r": r,
        "r_squared": (r ** 2) if r is not None else None,
        "n_points": n,
        "residual_std_s": residual_std,
    }


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def print_report(agg):
    print(
        f"\nToken-time (h7) benchmark aggregate — {agg['n_records']} records "
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
        f"layers={lora.get('num_layers')}   GC={agg.get('gradient_checkpointing')}"
    )

    print("\nPer-token-count cells (iteration 0 already excluded by the harness):")
    hdr = f"  {'tokens':>7} {'n':>3} {'s/iter (mean±std)':>20} {'tok/s':>10} {'peak_MB':>9} {'thermal(first→last)':>22}"
    print(hdr)
    for c in agg["cells"]:
        sp = c["seconds_per_iter"]
        tk = c["tok_per_sec"]
        print(
            f"  {str(c['target_tokens']):>7} {c['n_iterations']:>3} "
            f"{fmt(sp['mean']):>9}±{fmt(sp['std']):<9} "
            f"{fmt(tk['mean'], 1):>10} "
            f"{fmt(c['peak_mem_mb_mean'], 0):>9} "
            f"{str(c['first_thermal_state']):>10}→{str(c['last_thermal_state']):<10}"
        )

    fit = agg.get("linear_fit")
    if fit:
        print(
            f"\nLinear cost model ({fit['n_points']} per-cell means): "
            f"seconds/iter ≈ {fit['intercept_s']:.4f} + {fit['slope_s_per_token']:.6f} × tokens "
            f"(r={fit['pearson_r']:.4f}, R²={fit['r_squared']:.4f}, "
            f"residual_std={fit['residual_std_s']:.4f}s)"
        )
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "paths", nargs="*",
        help="JSONL telemetry file(s); default results/ondevice/train_bench_metrics_tokentime_*.jsonl",
    )
    ap.add_argument("--out", help="write aggregated JSON here")
    args = ap.parse_args()

    paths = args.paths or sorted(
        glob.glob("results/ondevice/train_bench_metrics_tokentime_*.jsonl")
    )
    if not paths:
        print("No telemetry files matched.", file=sys.stderr)
        sys.exit(1)

    records = load_records(paths)
    if not records:
        print("No records loaded.", file=sys.stderr)
        sys.exit(1)

    def first(field):
        return next((r.get(field) for r in records if r.get(field) is not None), None)

    cells = summarize_cells(records)
    fit_points = [
        (c["target_tokens"], c["seconds_per_iter"]["mean"])
        for c in cells
        if c["seconds_per_iter"]["mean"] is not None
    ]

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
        "gradient_checkpointing": first("gradient_checkpointing"),
        "optimizer": first("optimizer"),
        "learning_rate": first("learning_rate"),
        "weight_decay": first("weight_decay"),
        "warmup_seconds": first("warmup_seconds"),
        "warmup_seq_cap": first("warmup_seq_cap"),
        "cells": cells,
        "linear_fit": linear_fit(fit_points),
    }

    print_report(agg)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
