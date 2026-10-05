#!/usr/bin/env python3
"""
Aggregate the h11 per-op/per-phase telemetry (Documents/train_bench_metrics_perop.jsonl
pulled off the phone) into a per-(token count, pass) phase decomposition.

Stdlib only (no pandas on the Mac MLX venv), flat JSON in, descriptive stats out
— same shape as eval/train_tokentime_aggregate.py and eval/thermal_aggregate.py.
Locked design: experiments/2026-08-04-ondevice-perop-h11-plan.md (pinned via
/grill_me 2026-08-04).

NOTHING IS PRE-REGISTERED in this round (explicit user decision) — this script
reports, it does not adjudicate.

What it computes, per cell = (target_tokens, pass):
  * BARRIERED per-phase mean±sd and each phase's SHARE of the barriered total.
    Shares are the round's primary deliverable.
  * FUSED mean±sd of the whole iteration (stock LoRATrain.train, h7's exact
    measurement mode) — the validity control.
  * `sum_phases / fused` = the decomposition-overhead ratio. Barriers destroy
    cross-phase overlap, so this is >1 by construction; how much >1 is the
    honesty check on the shares (MELT flags the same caveat for vm_profiler).
    ABSOLUTE per-iteration cost predictions stay with h7's fused fits.
  * Loss agreement between the two modes. The harness restores the freshly
    initialised adapter before the barriered sub-block, so both modes train the
    same weights on the same data from the same start; a per-iteration loss
    mismatch would mean the barriered replica is NOT computing what the stock
    fused path computes (plan risk #1).
  * Peak memory per mode. If splitting `eval(lvalue)` from `eval(grad)` defeated
    gradient checkpointing, barriered peak memory would jump relative to fused
    (the other half of plan risk #1).

Iterations flagged `warmup: true` by the harness are excluded from all
statistics (log-raw-decide-later, h10 convention) but their count is reported.

Usage:
    python eval/perop_aggregate.py results/ondevice/train_bench_metrics_perop_*.jsonl \
        --out results/ondevice_perop_smollm3_4bit_2026-08-XX.json
"""

import argparse
import glob
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASES = ["data_prep", "graph_build", "forward", "backward", "optimizer", "readback"]
PHASE_FIELD = {p: f"phase_{p}_s" for p in PHASES}

# h7's published fits (experiments/2026-07-24-ondevice-tokentime-plan.md).
# CAVEAT, applied when reporting: h7 ran with the buggy shared loraLayers=28,
# h11 runs at the fixed 36. The LoRA layers are a SUFFIX of the 36 blocks, so at
# 28 the backward pass stops ~8 blocks early — h11 iterations are expected to be
# meaningfully SLOWER than h7's fits at the same token count. A ratio above 1 is
# the expected reading, not a discrepancy.
H7_FITS = {
    "hot": {"intercept": -1.957, "slope": 0.02154, "label": "h7 HOT (sustained)"},
    "cold": {"intercept": -0.678, "slope": 0.01158, "label": "h7 COLD (isolated)"},
}


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


def summarize_cold_ref(records):
    rows = [r for r in records if r.get("record_type") == "cold_ref"]
    if not rows:
        return None
    rows.sort(key=lambda r: (r.get("bench_session_id") or "", r.get("iter_index") or 0))
    kept = [r for r in rows if not r.get("warmup")]
    return {
        "target_tokens": rows[0].get("target_tokens"),
        "n_iterations": len(rows),
        "n_warmup_dropped": len(rows) - len(kept),
        "seconds_per_iter": describe([r.get("iter_seconds") for r in kept]),
        "all_iterations": [
            {
                "iter_index": r.get("iter_index"),
                "warmup": bool(r.get("warmup")),
                "iter_seconds": r.get("iter_seconds"),
                "thermal_state": r.get("thermal_state"),
            }
            for r in rows
        ],
    }


def loss_continuity(fused_rows, barriered_rows):
    """Does the barriered sub-block continue the fused one's loss curve?

    CORRECTED 2026-08-04, mid-run, against the first real cell. The harness
    snapshots `model.trainableParameters()` before the fused sub-block and
    restores it before the barriered one, intending both to start from
    identical weights — but that restore is a NO-OP. `Module.update`'s
    leaf-array case calls `p._updateInternal(newArray)`, which swaps the handle
    INSIDE the existing MLXArray object, so `trainableParameters()` returns
    ALIASES of the model's live arrays rather than copies; the "snapshot"
    tracks the weights through training. (Same aliasing the h6 v7 changelog in
    TrainBenchConstants.swift documents for the model-load path.) Observed
    directly: fused ran 0.5878 -> 0.4371 and barriered then started at 0.4239.

    So a paired |loss_fused - loss_barriered| comparison measures how far the
    loss moved over the fused sub-block, NOT replica fidelity. What the data
    CAN support is a continuity check across the mode boundary: the barriered
    block picks the trajectory up where the fused block left it, so a faithful
    replica continues the same per-step decay. A broken one (wrong loss,
    skipped optimizer step, defeated checkpointing) would show a step change or
    a different decay rate at the seam.

    Caveat that keeps this honest: a fresh AdamW is built for the barriered
    sub-block, so its Adam moments restart at the boundary. A small first-step
    perturbation is expected and is not evidence of a broken replica.

    NOTE: this does not affect any TIMING result. MLX's dense and quantized
    kernels are value-independent, so per-iteration cost does not depend on
    which weights are resident — both sub-blocks train identical shapes under
    an identical recipe.
    """
    fused = sorted(
        (r for r in fused_rows if isinstance(r.get("loss"), (int, float))),
        key=lambda r: r.get("iter_index") or 0,
    )
    barr = sorted(
        (r for r in barriered_rows if isinstance(r.get("loss"), (int, float))),
        key=lambda r: r.get("iter_index") or 0,
    )
    out = {
        "fused_first_loss": fused[0]["loss"] if fused else None,
        "fused_last_loss": fused[-1]["loss"] if fused else None,
        "barriered_first_loss": barr[0]["loss"] if barr else None,
        "barriered_last_loss": barr[-1]["loss"] if barr else None,
        "fused_last_step_delta": None,
        "boundary_step_delta": None,
        "barriered_first_step_delta": None,
        "continuity_residual": None,
    }
    if len(fused) >= 2:
        out["fused_last_step_delta"] = fused[-1]["loss"] - fused[-2]["loss"]
    if fused and barr:
        out["boundary_step_delta"] = barr[0]["loss"] - fused[-1]["loss"]
    if len(barr) >= 2:
        out["barriered_first_step_delta"] = barr[1]["loss"] - barr[0]["loss"]
    # How far the step ACROSS the seam departs from the step just before it.
    # ~0 ⇒ the barriered loop is doing the same work per iteration.
    if out["fused_last_step_delta"] is not None and out["boundary_step_delta"] is not None:
        out["continuity_residual"] = out["boundary_step_delta"] - out["fused_last_step_delta"]
    return out


def summarize_cells(records):
    """One summary per (target_tokens, pass), each holding both modes."""
    iters = [r for r in records if r.get("record_type") == "iter"]
    groups = {}
    for r in iters:
        groups.setdefault((r.get("pass"), r.get("target_tokens")), []).append(r)

    # Preserve the session's pass order (cool then hot) rather than sorting
    # alphabetically, which would put "hot" first.
    pass_order = []
    for r in iters:
        if r.get("pass") not in pass_order:
            pass_order.append(r.get("pass"))

    cells = []
    for (pass_name, tokens) in sorted(
        groups.keys(),
        key=lambda k: (pass_order.index(k[0]) if k[0] in pass_order else 99, k[1] or 0),
    ):
        rows = groups[(pass_name, tokens)]
        fused_all = [r for r in rows if r.get("mode") == "fused"]
        barr_all = [r for r in rows if r.get("mode") == "barriered"]
        fused = [r for r in fused_all if not r.get("warmup")]
        barr = [r for r in barr_all if not r.get("warmup")]

        phase_stats = {p: describe([r.get(PHASE_FIELD[p]) for r in barr]) for p in PHASES}
        barr_total = describe([r.get("iter_seconds") for r in barr])
        fused_total = describe([r.get("iter_seconds") for r in fused])

        # Shares are computed from the phase MEANS over the mean total, not as
        # a mean of per-iteration shares — the two differ slightly and the
        # former is what a stacked-bar figure should show.
        shares = None
        if barr_total["mean"]:
            shares = {
                p: (phase_stats[p]["mean"] / barr_total["mean"])
                if phase_stats[p]["mean"] is not None
                else None
                for p in PHASES
            }

        overhead = None
        if barr_total["mean"] is not None and fused_total["mean"]:
            overhead = barr_total["mean"] / fused_total["mean"]

        cell = {
            "pass": pass_name,
            "target_tokens": tokens,
            "n_fused": len(fused),
            "n_barriered": len(barr),
            "n_warmup_dropped": (len(fused_all) - len(fused)) + (len(barr_all) - len(barr)),
            "fused_seconds_per_iter": fused_total,
            "barriered_seconds_per_iter": barr_total,
            "phases": phase_stats,
            "phase_shares": shares,
            "decomposition_overhead_ratio": overhead,
            "loss_continuity": loss_continuity(fused_all, barr_all),
            "peak_mem_mb_fused": (describe([r.get("peak_mem_bytes") for r in fused])["mean"] or 0)
            / (1024 * 1024),
            "peak_mem_mb_barriered": (describe([r.get("peak_mem_bytes") for r in barr])["mean"] or 0)
            / (1024 * 1024),
            "thermal_states": _counts(r.get("thermal_state") for r in rows),
            "first_thermal_state": rows[0].get("thermal_state") if rows else None,
            "last_thermal_state": rows[-1].get("thermal_state") if rows else None,
            "low_power_mode_any": any(bool(r.get("low_power_mode")) for r in rows),
        }
        cell["peak_mem_ratio_barriered_over_fused"] = (
            cell["peak_mem_mb_barriered"] / cell["peak_mem_mb_fused"]
            if cell["peak_mem_mb_fused"]
            else None
        )
        cells.append(cell)
    return cells


def optimizer_flatness(cells):
    """Is the optimizer phase's ABSOLUTE cost independent of token count?

    AdamW touches only the LoRA parameters, whose count does not depend on
    sequence length, so a flat line here is the expected shape — reported as a
    max/min ratio per pass rather than asserted.
    """
    out = {}
    for pass_name in {c["pass"] for c in cells}:
        vals = [
            (c["target_tokens"], c["phases"]["optimizer"]["mean"])
            for c in cells
            if c["pass"] == pass_name and c["phases"]["optimizer"]["mean"] is not None
        ]
        if len(vals) < 2:
            continue
        ys = [v for _, v in vals]
        out[pass_name] = {
            "min_s": min(ys),
            "max_s": max(ys),
            "max_over_min": max(ys) / min(ys) if min(ys) > 0 else None,
            "by_tokens": {str(t): y for t, y in vals},
        }
    return out


def h7_cross_check(cells):
    """Fused per-iteration times vs h7's published fits (qualitative only).

    Not a gate. h7 measured at loraLayers=28 and h11 at the fixed 36, so
    observed/predicted above 1 is the expected reading — see H7_FITS.
    """
    rows = []
    for c in cells:
        obs = c["fused_seconds_per_iter"]["mean"]
        if obs is None or c["target_tokens"] is None:
            continue
        entry = {"pass": c["pass"], "target_tokens": c["target_tokens"], "observed_s": obs}
        for name, fit in H7_FITS.items():
            pred = fit["intercept"] + fit["slope"] * c["target_tokens"]
            entry[f"h7_{name}_predicted_s"] = pred
            entry[f"h7_{name}_observed_over_predicted"] = obs / pred if pred > 0 else None
        rows.append(entry)
    return rows


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def print_report(agg):
    print(
        f"\nPer-op/per-phase (h11) aggregate — {agg['n_records']} records "
        f"from {len(agg['source_files'])} file(s)"
    )
    print(
        f"app_build(s): {', '.join(agg.get('app_builds') or [])}   "
        f"session(s): {len(agg.get('bench_session_ids') or [])}   "
        f"model: {agg.get('model')}   device: {agg.get('device_model')}"
    )
    lora = agg.get("lora") or {}
    print(
        f"LoRA: rank={lora.get('rank')} keys={lora.get('keys')} "
        f"layers={lora.get('num_layers')}   GC={agg.get('gradient_checkpointing')}   "
        f"idle_minutes(honest input)={agg.get('idle_minutes')}"
    )
    run = agg.get("run") or {}
    if run.get("elapsed_s"):
        print(f"session elapsed: {run['elapsed_s'] / 60.0:.1f} min")

    cr = agg.get("cold_ref")
    if cr:
        print(
            f"\nCold reference ({cr['target_tokens']} tok, "
            f"{cr['seconds_per_iter']['n']} kept / {cr['n_warmup_dropped']} dropped): "
            f"{fmt(cr['seconds_per_iter']['mean'])} s/iter"
        )

    print("\nPer-cell phase decomposition (barriered; warmup iterations dropped):")
    hdr = (
        f"  {'pass':>5} {'tok':>5} {'n':>3} "
        + " ".join(f"{p[:9]:>9}" for p in PHASES)
        + f" {'Σphases':>8} {'fused':>8} {'Σ/fused':>8}"
    )
    print(hdr)
    for c in agg["cells"]:
        ph = " ".join(fmt(c["phases"][p]["mean"], 3).rjust(9) for p in PHASES)
        print(
            f"  {str(c['pass']):>5} {str(c['target_tokens']):>5} {c['n_barriered']:>3} {ph} "
            f"{fmt(c['barriered_seconds_per_iter']['mean']):>8} "
            f"{fmt(c['fused_seconds_per_iter']['mean']):>8} "
            f"{fmt(c['decomposition_overhead_ratio']):>8}"
        )

    print("\nPhase shares of the barriered iteration (%):")
    print(f"  {'pass':>5} {'tok':>5} " + " ".join(f"{p[:9]:>9}" for p in PHASES))
    for c in agg["cells"]:
        if not c["phase_shares"]:
            continue
        sh = " ".join(
            ("—".rjust(9) if c["phase_shares"][p] is None else f"{100 * c['phase_shares'][p]:9.1f}")
            for p in PHASES
        )
        print(f"  {str(c['pass']):>5} {str(c['target_tokens']):>5} {sh}")

    print("\nValidity checks (plan risk #1 — barriered replica vs stock fused path):")
    print(
        f"  {'pass':>5} {'tok':>5} {'fused Δ/step':>13} {'seam Δ':>9} {'residual':>9} "
        f"{'peak MB fused':>14} {'peak MB barr':>13} {'ratio':>7}"
    )
    for c in agg["cells"]:
        lc = c["loss_continuity"]
        print(
            f"  {str(c['pass']):>5} {str(c['target_tokens']):>5} "
            f"{fmt(lc['fused_last_step_delta'], 5):>13} {fmt(lc['boundary_step_delta'], 5):>9} "
            f"{fmt(lc['continuity_residual'], 5):>9} "
            f"{fmt(c['peak_mem_mb_fused'], 0):>14} {fmt(c['peak_mem_mb_barriered'], 0):>13} "
            f"{fmt(c['peak_mem_ratio_barriered_over_fused']):>7}"
        )
    print(
        "  (the barriered sub-block CONTINUES the fused one's weights — the intended\n"
        "   snapshot/restore is a no-op because Module.update aliases rather than copies,\n"
        "   see loss_continuity's docstring. So fidelity is read as continuity: residual ≈ 0\n"
        "   ⇒ the barriered loop does the same work per iteration as the stock path.\n"
        "   Peak-memory ratio ≈ 1 ⇒ splitting the evals did not defeat gradient checkpointing.\n"
        "   Neither affects the phase TIMINGS — MLX kernels are value-independent.)"
    )

    flat = agg.get("optimizer_flatness") or {}
    if flat:
        print("\nOptimizer-phase absolute flatness across the token grid:")
        for pass_name, f in flat.items():
            print(
                f"  {pass_name:>5}: {fmt(f['min_s'])}–{fmt(f['max_s'])} s "
                f"(max/min = {fmt(f['max_over_min'], 2)})"
            )

    xc = agg.get("h7_cross_check") or []
    if xc:
        print("\nFused times vs h7 fits (qualitative; h7 ran at 28 LoRA layers, h11 at 36):")
        print(f"  {'pass':>5} {'tok':>5} {'observed':>9} {'h7 HOT':>9} {'obs/HOT':>8} {'h7 COLD':>9} {'obs/COLD':>9}")
        for r in xc:
            print(
                f"  {str(r['pass']):>5} {r['target_tokens']:>5} {fmt(r['observed_s']):>9} "
                f"{fmt(r['h7_hot_predicted_s']):>9} {fmt(r['h7_hot_observed_over_predicted'], 2):>8} "
                f"{fmt(r['h7_cold_predicted_s']):>9} {fmt(r['h7_cold_observed_over_predicted'], 2):>9}"
            )

    cap = agg.get("capture_runs") or []
    for c in cap:
        print(
            f"\nTier-2 capture run: tokens={c.get('target_tokens')} "
            f"supported={c.get('capture_supported')} captured={c.get('captured')}"
        )
        if c.get("note"):
            print(f"  note: {c['note']}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="JSONL telemetry file(s); default results/ondevice/train_bench_metrics_perop*.jsonl",
    )
    ap.add_argument("--out", help="write aggregated JSON here")
    ap.add_argument(
        "--overwrite", action="store_true", help="allow --out to replace an existing file"
    )
    args = ap.parse_args()

    paths = args.paths or sorted(
        glob.glob("results/ondevice/train_bench_metrics_perop*.jsonl")
    )
    if not paths:
        print("No telemetry files matched.", file=sys.stderr)
        sys.exit(1)

    if args.out and Path(args.out).exists() and not args.overwrite:
        print(f"Refusing to overwrite {args.out} (pass --overwrite).", file=sys.stderr)
        sys.exit(1)

    records = load_records(paths)
    if not records:
        print("No records loaded.", file=sys.stderr)
        sys.exit(1)

    def first(field):
        return next((r.get(field) for r in records if r.get(field) is not None), None)

    cells = summarize_cells(records)
    run_start = next((r for r in records if r.get("record_type") == "run_start"), None)
    run_end = next((r for r in records if r.get("record_type") == "run_end"), None)

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
        "schema_version": first("bench_schema_version"),
        "git_commit": first("git_commit"),
        "git_dirty": first("git_dirty"),
        "idle_minutes": first("idle_minutes"),
        "lora": {
            "rank": first("lora_rank"),
            "keys": first("lora_keys"),
            "num_layers": first("num_lora_layers"),
        },
        "gradient_checkpointing": first("gradient_checkpointing"),
        "checkpoint_granularity": first("checkpoint_granularity"),
        "optimizer": first("optimizer"),
        "learning_rate": first("learning_rate"),
        "weight_decay": first("weight_decay"),
        "warmup_iterations": first("warmup_iterations"),
        "kept_iterations": first("kept_iterations"),
        "phases": PHASES,
        "run": {
            "start_utc": (run_start or {}).get("timestamp_utc"),
            "end_utc": (run_end or {}).get("timestamp_utc"),
            "elapsed_s": (run_end or {}).get("elapsed_s"),
            "battery_level_start": (run_start or {}).get("battery_level"),
            "battery_level_end": (run_end or {}).get("battery_level_end"),
        },
        "cold_ref": summarize_cold_ref(records),
        "cells": cells,
        "optimizer_flatness": optimizer_flatness(cells),
        "h7_cross_check": h7_cross_check(cells),
        "capture_runs": [r for r in records if r.get("record_type") == "capture_run"],
        "samples": [
            {
                "elapsed_s": r.get("elapsed_s"),
                "thermal_state": r.get("thermal_state"),
                "battery_level": r.get("battery_level"),
                "charging": r.get("charging"),
                "cpu_util_pct": r.get("cpu_util_pct"),
            }
            for r in records
            if r.get("record_type") == "sample"
        ],
    }

    print_report(agg)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(agg, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
