#!/usr/bin/env python3
"""
Aggregate on-device benchmark telemetry (Documents/bench_metrics.jsonl pulled off
the phone) into per-cell descriptive summaries.

Mirrors eval/summary.py: stdlib only (no pandas on the Mac MLX venv), flat JSON in,
descriptive stats out. This is the analysis step of the locked design in
experiments/2026-06-21-ondevice-base-inference-plan.md.

Descriptive only — mean ± std (n), min/max. NO bootstrap CIs / paired tests; those
are reserved for the later base-vs-Task-LoRA comparison.

Three telemetry classes are kept separate (never averaged together):
  - steady-state grid : cell in {prefill, decode, realistic}, warmup==False, cold==False
                        Primary numbers are NOMINAL-thermal runs only; fair/serious/
                        critical runs are counted and reported, never silently mixed in.
  - cold              : cold==True — the "app launch → first answer" number
                        (model_load_ms + inflated ttft_ms).
  - stress            : cell=="stress" — one record per segment; emitted as a
                        tps-vs-cumulative_tokens decay series.

Usage:
    python eval/bench_aggregate.py results/ondevice/bench_metrics_*.jsonl \
        --out results/ondevice_base_smollm3_4bit_2026-06-21.json
"""

import argparse
import glob
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# Metrics summarized per steady-state cell.
METRICS = ["gen_tps", "prompt_tps", "ttft_ms", "peak_mem_bytes"]


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
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def cell_key(r):
    """Group steady-state runs by their commanded grid point."""
    return (r.get("cell"), r.get("target_prompt_tokens"), r.get("target_gen_tokens"))


def summarize_grid(records):
    """Per-cell summary. Primary = nominal-thermal runs; non-nominal counted aside."""
    groups = {}
    for r in records:
        if r.get("cell") not in ("prefill", "decode", "realistic"):
            continue
        if r.get("warmup") or r.get("cold"):
            continue
        groups.setdefault(cell_key(r), []).append(r)

    cells = []
    for key, runs in sorted(groups.items(), key=lambda kv: _sort_key(kv[0])):
        cell, tgt_p, tgt_g = key
        nominal = [r for r in runs if r.get("thermal_state") == "nominal"]
        non_nominal = [r for r in runs if r.get("thermal_state") != "nominal"]
        primary = nominal if nominal else runs  # fall back if no nominal run captured

        out = {
            "cell": cell,
            "target_prompt_tokens": tgt_p,
            "target_gen_tokens": tgt_g,
            "n_total": len(runs),
            "n_nominal": len(nominal),
            "n_non_nominal": len(non_nominal),
            "thermal_states": _counts(r.get("thermal_state") for r in runs),
            "primary_is_nominal": bool(nominal),
            "measured_prompt_tokens_mean": describe([r.get("prompt_tokens") for r in primary])["mean"],
            "measured_gen_tokens_mean": describe([r.get("gen_tokens") for r in primary])["mean"],
        }
        for m in METRICS:
            out[m] = describe([r.get(m) for r in primary])
        cells.append(out)
    return cells


def summarize_cold(records):
    cold = [r for r in records if r.get("cold")]
    if not cold:
        return None
    return {
        "n": len(cold),
        "model_load_ms": describe([r.get("model_load_ms") for r in cold]),
        "ttft_ms": describe([r.get("ttft_ms") for r in cold]),
        "gen_tps": describe([r.get("gen_tps") for r in cold]),
        "peak_mem_bytes": describe([r.get("peak_mem_bytes") for r in cold]),
        "thermal_states": _counts(r.get("thermal_state") for r in cold),
    }


def summarize_stress(records):
    seg = [r for r in records if r.get("cell") == "stress" and r.get("segment_idx") is not None]
    if not seg:
        return None
    seg.sort(key=lambda r: r.get("cumulative_tokens") or 0)
    decay = [
        {
            "segment_idx": r.get("segment_idx"),
            "cumulative_tokens": r.get("cumulative_tokens"),
            "gen_tps": r.get("gen_tps"),
            "thermal_state": r.get("thermal_state"),
        }
        for r in seg
    ]
    tps = [r.get("gen_tps") for r in seg if isinstance(r.get("gen_tps"), (int, float))]
    return {
        "n_segments": len(seg),
        "total_tokens": seg[-1].get("cumulative_tokens"),
        "first_segment_tps": tps[0] if tps else None,
        "last_segment_tps": tps[-1] if tps else None,
        "min_segment_tps": min(tps) if tps else None,
        "max_segment_tps": max(tps) if tps else None,
        "throttle_drop_pct": (
            100.0 * (tps[0] - min(tps)) / tps[0] if tps and tps[0] else None
        ),
        "thermal_states": _counts(r.get("thermal_state") for r in seg),
        "decay": decay,
    }


def _counts(it):
    out = {}
    for v in it:
        out[v] = out.get(v, 0) + 1
    return out


_CELL_ORDER = {"prefill": 0, "decode": 1, "realistic": 2}


def _sort_key(key):
    cell, tgt_p, tgt_g = key
    return (_CELL_ORDER.get(cell, 9), tgt_p or 0, tgt_g or 0)


def fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def print_report(agg):
    print(f"\nOn-device benchmark aggregate — {agg['n_records']} records "
          f"from {len(agg['source_files'])} file(s)")
    builds = agg.get("app_builds") or []
    sessions = agg.get("bench_session_ids") or []
    print(f"app_build(s): {', '.join(builds)}   session(s): {len(sessions)}")

    print("\nSteady-state grid (primary = nominal-thermal runs):")
    hdr = (f"  {'cell':<9} {'p_tok':>6} {'g_tok':>6} {'n':>4} {'nom':>4}  "
           f"{'gen_tps':>16}  {'prompt_tps':>16}  {'ttft_ms':>14}  {'peak_MB':>10}")
    print(hdr)
    for c in agg["grid_cells"]:
        gen = c["gen_tps"]; ptps = c["prompt_tps"]; ttft = c["ttft_ms"]; pk = c["peak_mem_bytes"]
        peak_mb = (pk["mean"] / (1024 * 1024)) if pk["mean"] is not None else None
        peak_mb_max = (pk["max"] / (1024 * 1024)) if pk["max"] is not None else None
        flag = "" if c["primary_is_nominal"] else " (no-nominal!)"
        print(
            f"  {c['cell']:<9} {str(c['target_prompt_tokens'] or '·'):>6} "
            f"{str(c['target_gen_tokens'] or '·'):>6} {c['n_total']:>4} {c['n_nominal']:>4}  "
            f"{fmt(gen['mean'],1):>7}±{fmt(gen['std'],1):<7} "
            f"{fmt(ptps['mean'],0):>7}±{fmt(ptps['std'],0):<7} "
            f"{fmt(ttft['mean'],0):>6}±{fmt(ttft['std'],0):<6} "
            f"{fmt(peak_mb,0):>4}/{fmt(peak_mb_max,0):<4}{flag}"
        )

    if agg.get("cold"):
        c = agg["cold"]
        print(f"\nCold start (n={c['n']}): "
              f"model_load_ms {fmt(c['model_load_ms']['mean'],0)}±{fmt(c['model_load_ms']['std'],0)}, "
              f"ttft_ms {fmt(c['ttft_ms']['mean'],0)}±{fmt(c['ttft_ms']['std'],0)}")

    if agg.get("stress"):
        s = agg["stress"]
        print(f"\nStress run: {s['n_segments']} segments, {s['total_tokens']} tokens, "
              f"tps {fmt(s['first_segment_tps'])}→{fmt(s['last_segment_tps'])} "
              f"(min {fmt(s['min_segment_tps'])}, drop {fmt(s['throttle_drop_pct'])}%), "
              f"thermal {s['thermal_states']}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="JSONL telemetry file(s); default results/ondevice/*.jsonl")
    ap.add_argument("--out", help="write aggregated JSON here")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob("results/ondevice/*.jsonl"))
    if not paths:
        print("No telemetry files matched.", file=sys.stderr)
        sys.exit(1)

    records = load_records(paths)
    if not records:
        print("No records loaded.", file=sys.stderr)
        sys.exit(1)

    agg = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(p) for p in paths],
        "n_records": len(records),
        "app_builds": sorted({r.get("app_build") for r in records if r.get("app_build")}),
        "bench_session_ids": sorted({r.get("bench_session_id") for r in records if r.get("bench_session_id")}),
        "device_model": next((r.get("device_model") for r in records if r.get("device_model")), None),
        "os_version": next((r.get("os_version") for r in records if r.get("os_version")), None),
        "model": next((r.get("model") for r in records if r.get("model")), None),
        "grid_cells": summarize_grid(records),
        "cold": summarize_cold(records),
        "stress": summarize_stress(records),
    }

    print_report(agg)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
