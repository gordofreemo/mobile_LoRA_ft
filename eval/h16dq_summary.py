#!/usr/bin/env python3
"""h16dq: summarise the three-arm (off / on / dequant) backward A/B.

Input : results/ondevice/train_bench_metrics_naxab_h16dq_<date>.jsonl
Output: results/ondevice/h16dq_summary_<date>.json and a printed table.

Analyses the LAST run_start session only (same-day reruns append). Warm-up iterations
are dropped. Per (pass, target_tokens, arm): median barriered backward-phase time,
median barriered step (sum of phases), median fused step, max allocator peak and max
phys_footprint over that arm's iterations, and n. Ratios dequant/on and dequant/off
are per cell, so each compares arms at the same sequence length and thermal state.
The Phase 0 gradient check (record_type dq_grad_check) is carried into the summary.
"""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out")
    ap.add_argument("--gradcheck", help="optional separate JSONL holding dq_grad_check records")
    a = ap.parse_args()
    rs = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    starts = [r for r in rs if r.get("record_type") == "run_start" and r.get("ab_arms")]
    if not starts:
        raise SystemExit("no three-arm run_start in file")
    sid = starts[-1]["bench_session_id"]
    v = [r for r in rs if r.get("bench_session_id") == sid]
    it = [r for r in v if r.get("record_type") == "iter" and not r.get("warmup")]

    cell = defaultdict(lambda: defaultdict(list))
    for r in it:
        cell[(r["pass"], r["target_tokens"])][(r["mode"], r["arm"])].append(r)

    def med(xs):
        return statistics.median(xs) if xs else None

    rows = []
    for (pas, tok) in sorted(cell, key=lambda k: (k[0] != "cool", k[1])):
        c = cell[(pas, tok)]
        row = {"pass": pas, "target_tokens": tok}
        for arm in ("off", "on", "dequant"):
            b = c.get(("barriered", arm), [])
            f = c.get(("fused", arm), [])
            allr = b + f
            row[arm] = {
                "n_barriered": len(b), "n_fused": len(f),
                "seq_len": b[0].get("seq_len") if b else None,
                "backward_s": med([x["phase_backward_s"] for x in b]),
                "forward_s": med([x["phase_forward_s"] for x in b]),
                "barriered_step_s": med([x["iter_seconds"] for x in b]),
                "fused_step_s": med([x["iter_seconds"] for x in f]),
                "peak_alloc_mb": max((x["peak_mem_bytes"] for x in allr), default=0) / 1e6 or None,
                "phys_footprint_max_mb": max((x.get("phys_footprint_bytes") or 0 for x in allr), default=0) / 1e6 or None,
                "phys_footprint_peak_mb": max((x.get("phys_footprint_peak_bytes") or 0 for x in allr), default=0) / 1e6 or None,
                "thermal": sorted({x.get("thermal_state") for x in allr}),
            }
        def ratio(k, num, den):
            n, d = row[num][k], row[den][k]
            return n / d if n and d else None
        row["dequant_over_on_backward"] = ratio("backward_s", "dequant", "on")
        row["dequant_over_off_backward"] = ratio("backward_s", "dequant", "off")
        row["dequant_over_on_fused"] = ratio("fused_step_s", "dequant", "on")
        row["dequant_over_off_fused"] = ratio("fused_step_s", "dequant", "off")
        row["dequant_minus_on_peak_mb"] = (
            (row["dequant"]["peak_alloc_mb"] or 0) - (row["on"]["peak_alloc_mb"] or 0)
            if row["dequant"]["peak_alloc_mb"] and row["on"]["peak_alloc_mb"] else None)
        rows.append(row)

    cells_started = [r for r in v if r.get("record_type") == "cell_start"]
    cells_ended = [r for r in v if r.get("record_type") == "cell_end"]
    run_end = [r for r in v if r.get("record_type") == "run_end"]
    samples = [r for r in v if r.get("record_type") == "sample"]

    grad = [r for r in rs if r.get("record_type") == "dq_grad_check"]
    if a.gradcheck:
        grad += [json.loads(l) for l in open(a.gradcheck) if l.strip()
                 and json.loads(l).get("record_type") == "dq_grad_check"]

    out = {
        "session": sid, "run_start": starts[-1].get("timestamp_utc"),
        "app_build": starts[-1].get("app_build"), "git_commit": starts[-1].get("git_commit"),
        "git_dirty": starts[-1].get("git_dirty"), "mlx_local_patches": starts[-1].get("mlx_local_patches"),
        "ab_arms": starts[-1].get("ab_arms"), "token_grid": starts[-1].get("token_grid"),
        "cells_started": len(cells_started), "cells_ended": len(cells_ended),
        "run_end_present": bool(run_end),
        "charging_any": any(s.get("charging") for s in samples),
        "battery_first_last": ([samples[0].get("battery_level"), samples[-1].get("battery_level")]
                               if samples else None),
        "grad_check": [{k: g.get(k) for k in ("target_tokens", "seq_len", "arm_a", "arm_b",
                        "max_rel_err", "global_rel_l2", "non_finite", "params_compared",
                        "params_zero_ref", "worst_param", "peak_mem_a", "peak_mem_b")} for g in grad],
        "cells": rows,
    }
    outp = Path(a.out) if a.out else Path(a.jsonl).with_name(
        Path(a.jsonl).name.replace("train_bench_metrics_naxab_h16dq", "h16dq_summary").replace(".jsonl", ".json"))
    outp.write_text(json.dumps(out, indent=2))

    print(f"session {sid[:8]}  build {out['app_build']}  commit {out['git_commit']} dirty={out['git_dirty']}")
    print(f"cells {out['cells_ended']}/{out['cells_started']} ended, run_end={out['run_end_present']}, charging_any={out['charging_any']}")
    hdr = ("pass", "tok", "M", "bwd off", "bwd on", "bwd dq", "dq/on", "dq/off",
           "fus off", "fus on", "fus dq", "dq/on f", "pk on MB", "pk dq MB", "dPk MB")
    print("%-4s %5s %5s | %7s %7s %7s %6s %6s | %7s %7s %7s %7s | %8s %8s %7s" % hdr)
    for r in rows:
        f = lambda x, p=3: ("%.*f" % (p, x)) if isinstance(x, (int, float)) else "  -"
        print("%-4s %5d %5s | %7s %7s %7s %6s %6s | %7s %7s %7s %7s | %8s %8s %7s" % (
            r["pass"], r["target_tokens"], r["on"]["seq_len"],
            f(r["off"]["backward_s"]), f(r["on"]["backward_s"]), f(r["dequant"]["backward_s"]),
            f(r["dequant_over_on_backward"], 2), f(r["dequant_over_off_backward"], 2),
            f(r["off"]["fused_step_s"]), f(r["on"]["fused_step_s"]), f(r["dequant"]["fused_step_s"]),
            f(r["dequant_over_on_fused"], 2),
            f(r["on"]["peak_alloc_mb"], 0), f(r["dequant"]["peak_alloc_mb"], 0),
            f(r["dequant_minus_on_peak_mb"], 0)))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
