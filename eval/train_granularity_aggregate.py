#!/usr/bin/env python3
"""
Aggregate the h8 GC-granularity-sweep telemetry
(Documents/train_bench_metrics_granularity.jsonl pulled off the phone) into a
per-K (checkpoint-group-size) cost summary.

Mirrors eval/train_bench_aggregate.py (h1-h4 cap sweep) and
eval/train_tokentime_aggregate.py (h7): stdlib only (no pandas on the Mac MLX
venv), flat JSON in, descriptive stats out. Locked design:
experiments/2026-07-25-ondevice-gc-granularity-plan.md (pinned via
/grill_me 2026-07-25).

Grouping key = checkpoint_granularity (K) ONLY, not (session, K) — the sweep
runs each K as exactly one process launch (see scripts/run_granularity_sweep.sh),
so under normal operation there is one session per K. If a K was re-launched
(e.g. to retry after an unrelated launch failure), records from every session
for that K are pooled into one cell — this is a deliberate simplification for
a "one row per K" table, not a session-aware merge.

OOM/failure detection mirrors h1-h4's `cap_start`/no-`train`-rows pattern: the
harness writes a `k_start` sentinel BEFORE training a cell, so a K that died
to an uncatchable jetsam (SIGKILL, no `train` records at all) is
distinguishable from one that trained cleanly. A `record_type == "error"`
row marks a CAUGHT (non-jetsam) training exception.

Usage:
    python eval/train_granularity_aggregate.py results/ondevice/train_bench_metrics_granularity_*.jsonl \\
        --out results/ondevice_granularity_smollm3_4bit_2026-07-2X.json
"""

import argparse
import glob
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

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


def summarize_cells(records):
    """One summary per K (checkpoint_granularity)."""
    by_k = {}
    for r in records:
        k = r.get("checkpoint_granularity")
        by_k.setdefault(k, []).append(r)

    cells = []
    for k, rows in sorted(by_k.items(), key=lambda kv: (kv[0] is None, kv[0])):
        train = [r for r in rows if r.get("record_type") == "train"]
        train.sort(key=lambda r: r.get("step") or 0)
        has_k_start = any(r.get("record_type") == "k_start" for r in rows)
        errors = [r for r in rows if r.get("record_type") == "error"]

        elapsed = [
            r.get("elapsed_s") for r in train if isinstance(r.get("elapsed_s"), (int, float))
        ]
        expected_windows = None
        iterations_total = next((r.get("iterations_total") for r in rows), None)
        steps_per_report = next((r.get("steps_per_report") for r in rows), None)
        if iterations_total and steps_per_report:
            expected_windows = iterations_total // steps_per_report

        # OOM (uncatchable jetsam): a k_start sentinel with zero train rows.
        # A caught (non-jetsam) exception shows up as an "error" row instead.
        oom = has_k_start and not train
        completed = (
            expected_windows is not None and len(train) >= expected_windows and not errors
        )

        out = {
            "checkpoint_granularity": k,
            "num_checkpoint_groups": next(
                (r.get("num_checkpoint_groups") for r in rows if r.get("num_checkpoint_groups") is not None),
                (36 // k if isinstance(k, int) and k else None),
            ),
            "n_windows": len(train),
            "n_expected_windows": expected_windows,
            "completed": completed,
            "oom": oom,
            "n_errors": len(errors),
            "error_messages": [e.get("error") for e in errors],
            "seq_cap": next((r.get("seq_cap") for r in rows), None),
            "iterations_total": iterations_total,
            "steps_per_report": steps_per_report,
            "wall_time_s": max(elapsed) if elapsed else None,
            "thermal_states": _counts(r.get("thermal_state") for r in train),
            "first_thermal_state": train[0].get("thermal_state") if train else None,
            "last_thermal_state": train[-1].get("thermal_state") if train else None,
            "low_power_mode_any": any(bool(r.get("low_power_mode")) for r in train),
            "bench_session_ids": sorted({r.get("bench_session_id") for r in rows if r.get("bench_session_id")}),
        }
        for m in METRICS:
            out[m] = describe([r.get(m) for r in train])
        pk = out["peak_mem_bytes"]
        out["peak_mem_mb_mean"] = pk["mean"] / (1024 * 1024) if pk["mean"] is not None else None
        out["peak_mem_mb_max"] = pk["max"] / (1024 * 1024) if pk["max"] is not None else None
        out["windows"] = [
            {
                "step": r.get("step"),
                "iter_per_sec": r.get("iter_per_sec"),
                "tok_per_sec": r.get("tok_per_sec"),
                "elapsed_s": r.get("elapsed_s"),
                "peak_mem_bytes": r.get("peak_mem_bytes"),
                "thermal_state": r.get("thermal_state"),
            }
            for r in train
        ]
        cells.append(out)
    return cells


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def print_report(agg):
    print(
        f"\nGC-granularity sweep (h8) aggregate — {agg['n_records']} records "
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
        f"layers={lora.get('num_layers')}   seq_cap={agg.get('seq_cap')}"
    )

    print("\nPer-K (checkpoint granularity) cells:")
    hdr = (
        f"  {'K':>3} {'groups':>6} {'win':>4} {'iter/s':>14} {'tok/s':>16} "
        f"{'peak_MB(mean/max)':>20} {'wall_s':>8} {'thermal(last)':>14} {'status':>10}"
    )
    print(hdr)
    for c in agg["cells"]:
        it = c["iter_per_sec"]
        tk = c["tok_per_sec"]
        status = "OOM" if c["oom"] else ("ERROR" if c["n_errors"] else ("OK" if c["completed"] else "PARTIAL"))
        print(
            f"  {str(c['checkpoint_granularity']):>3} {str(c['num_checkpoint_groups']):>6} "
            f"{c['n_windows']:>4} "
            f"{fmt(it['mean']):>6}±{fmt(it['std']):<6} "
            f"{fmt(tk['mean'],1):>7}±{fmt(tk['std'],1):<7} "
            f"{fmt(c['peak_mem_mb_mean'],0):>8}/{fmt(c['peak_mem_mb_max'],0):<8} "
            f"{fmt(c['wall_time_s'],0):>8} "
            f"{str(c['last_thermal_state']):>14} "
            f"{status:>10}"
        )

    oom_ks = [c["checkpoint_granularity"] for c in agg["cells"] if c["oom"]]
    if oom_ks:
        print(f"\nOOM'd (jetsam, k_start but no train rows): K = {oom_ks}")
    err_ks = [c["checkpoint_granularity"] for c in agg["cells"] if c["n_errors"]]
    if err_ks:
        print(f"Caught training errors: K = {err_ks}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "paths", nargs="*",
        help="JSONL telemetry file(s); default results/ondevice/train_bench_metrics_granularity_*.jsonl",
    )
    ap.add_argument("--out", help="write aggregated JSON here")
    args = ap.parse_args()

    paths = args.paths or sorted(
        glob.glob("results/ondevice/train_bench_metrics_granularity_*.jsonl")
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
        "seq_cap": first("seq_cap"),
        "lora": {
            "rank": first("lora_rank"),
            "keys": first("lora_keys"),
            "num_layers": first("num_lora_layers"),
        },
        "gradient_checkpointing": first("gradient_checkpointing"),
        "optimizer": first("optimizer"),
        "learning_rate": first("learning_rate"),
        "cells": summarize_cells(records),
    }

    print_report(agg)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
