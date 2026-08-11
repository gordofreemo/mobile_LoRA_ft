#!/usr/bin/env python3
"""Aggregate the h12 task-adapter device JSONL (train_bench_metrics_taskadapter.jsonl).

Usage:
  python eval/taskadapter_aggregate.py <device.jsonl> [--mac-ref <metrics.jsonl>]
      [--cluster-ref <metrics.jsonl>] [--json-out <path>]

With --mac-ref, runs the SMOKE parity checks (plan smoke criteria):
  1. device step-1 micro losses vs Mac reference micro losses (same 4-bit
     model, same pre-tokenized ids, same order) — catches mask off-by-one.
  2. 32 microbatches per opt_step (except a legitimate partial last window).
  3. peak memory present and sane.

For a full run, prints the systems summary: wall time, s/step trajectory,
thermal, memory, throughput, loss vs the cluster reference (if given).
"""

import argparse
import json
import sys
from collections import Counter


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("device_jsonl")
    ap.add_argument("--mac-ref", default=None)
    ap.add_argument("--cluster-ref", default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--session", default=None, help="restrict to one bench_session_id")
    args = ap.parse_args()

    records = load_jsonl(args.device_jsonl)
    if args.session:
        records = [r for r in records if r.get("bench_session_id") == args.session]
    sessions = sorted({r.get("bench_session_id") for r in records})
    if not args.session and len(sessions) > 1:
        # default to the latest session by first-appearance order
        last = records[-1]["bench_session_id"]
        records = [r for r in records if r.get("bench_session_id") == last]
        print(f"[note] {len(sessions)} sessions in file; using latest {last} "
              f"(pass --session to override)")

    by_type = Counter(r["record_type"] for r in records)
    steps = sorted((r for r in records if r["record_type"] == "opt_step"),
                   key=lambda r: r["step"])
    samples = [r for r in records if r["record_type"] == "sample"]
    run_start = next((r for r in records if r["record_type"] == "run_start"), None)
    run_end = next((r for r in records if r["record_type"] == "run_end"), None)
    error = next((r for r in records if r["record_type"] == "error"), None)

    print(f"records: {dict(by_type)}")
    if run_start:
        print(f"run_start: nax_arm={run_start.get('nax_arm')} model={run_start.get('model')} "
              f"n_examples={run_start.get('n_examples')} total_steps={run_start.get('total_steps')} "
              f"first_ids={run_start.get('first_example_ids')} "
              f"thermal={run_start.get('thermal_state')} build={run_start.get('app_build')} "
              f"commit={run_start.get('git_commit')}")
    if error:
        print(f"!! ERROR record: {error.get('error')} at elapsed={error.get('elapsed_s')}")

    summary = {}
    if steps:
        window_s = [r["window_s"] for r in steps]
        micro_s = [t for r in steps for t in r.get("micro_iter_s", [])]
        losses = [r["loss"] for r in steps]
        toks = sum(r["window_seq_tokens"] for r in steps)
        elapsed = steps[-1]["elapsed_s"] - (steps[0]["elapsed_s"] - steps[0]["window_s"])
        thermal_seq = [r["thermal_state"] for r in steps]
        peak_gb = max(r["peak_mem_bytes"] for r in steps) / 1e9
        n_micro_bad = [
            (r["step"], r["n_micro"]) for r in steps
            if r["n_micro"] != r["accum_window"] and r["step"] != r["schedule_total_steps"]
        ]
        summary = {
            "n_steps": len(steps),
            "loss_first": losses[0], "loss_last": losses[-1],
            "mean_window_s": sum(window_s) / len(window_s),
            "mean_micro_s": sum(micro_s) / len(micro_s) if micro_s else None,
            "total_seq_tokens": toks,
            "train_elapsed_s": elapsed,
            "tok_per_s": toks / elapsed if elapsed > 0 else None,
            "peak_mem_gb": peak_gb,
            "thermal_first": thermal_seq[0], "thermal_last": thermal_seq[-1],
            "thermal_counts": dict(Counter(thermal_seq)),
            "grad_norm_mean": sum(r["grad_norm_preclip"] for r in steps) / len(steps),
            "clip_frac": sum(1 for r in steps if r["clip_scale"] < 1.0) / len(steps),
        }
        print(f"\nsteps: {len(steps)}  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        print(f"window_s mean {summary['mean_window_s']:.2f} "
              f"(first {window_s[0]:.2f}, last {window_s[-1]:.2f}); "
              f"micro_iter_s mean {summary['mean_micro_s']:.3f}")
        print(f"tokens/s {summary['tok_per_s']:.1f}  peak_mem {peak_gb:.2f} GB  "
              f"thermal {thermal_seq[0]} -> {thermal_seq[-1]} {summary['thermal_counts']}")
        print(f"grad_norm mean {summary['grad_norm_mean']:.4f}  "
              f"clip fraction {summary['clip_frac']:.2%}")
        if n_micro_bad:
            print(f"!! unexpected partial windows: {n_micro_bad}")
        est_full_h = summary["mean_window_s"] * steps[0]["schedule_total_steps"] / 3600
        print(f"projected full-run training time at current rate: {est_full_h:.2f} h")

    if samples:
        cpu = [s["cpu_util_pct"] for s in samples if s.get("cpu_util_pct") is not None]
        lvl = [s.get("battery_level") for s in samples]
        print(f"\nsamples: {len(samples)} over {samples[-1]['elapsed_s'] / 60:.1f} min; "
              f"battery {lvl[0]:.2f} -> {lvl[-1]:.2f}; "
              f"cpu mean {sum(cpu) / len(cpu):.1f}%" if cpu else "no cpu samples")

    # --- smoke parity vs Mac reference ---------------------------------------
    if args.mac_ref and steps:
        mac = [r for r in load_jsonl(args.mac_ref) if r.get("record_type") == "opt_step"]
        mac = sorted(mac, key=lambda r: r["step"])
        print("\n--- smoke parity vs Mac reference ---")
        d1 = steps[0]
        m1 = mac[0]
        dm, mm = d1.get("micro_losses", []), m1.get("micro_losses", [])
        n = min(len(dm), len(mm))
        if n == 0:
            print("!! no micro losses to compare")
        else:
            diffs = [abs(a - b) for a, b in zip(dm[:n], mm[:n])]
            print(f"step-1 micro losses: n={n}  max|diff|={max(diffs):.5f}  "
                  f"mean|diff|={sum(diffs) / n:.5f}")
            print(f"  device[0:4]={[round(x, 4) for x in dm[:4]]}")
            print(f"  mac   [0:4]={[round(x, 4) for x in mm[:4]]}")
            summary["smoke_micro_max_diff"] = max(diffs)
        k = min(len(steps), len(mac))
        pairs = [(steps[i]["loss"], mac[i]["loss"]) for i in range(k)]
        wdiffs = [abs(a - b) for a, b in pairs]
        print(f"window losses over {k} steps: max|diff|={max(wdiffs):.5f}")
        gn = [(steps[i]["grad_norm_preclip"], mac[i]["grad_norm_preclip"]) for i in range(k)]
        print(f"grad norms step1: device {gn[0][0]:.4f} vs mac {gn[0][1]:.4f}")
        summary["smoke_window_max_diff"] = max(wdiffs)

    if args.cluster_ref and steps:
        cl = [r for r in load_jsonl(args.cluster_ref) if "loss" in r and "step" in r]
        print("\n--- vs cluster reference (10-step logging windows) ---")
        for r in cl:
            s = r["step"]
            dev_window = [d["loss"] for d in steps if s - 9 <= d["step"] <= s]
            if dev_window:
                dev_mean = sum(dev_window) / len(dev_window)
                print(f"  step {s:4d}: cluster {r['loss']:.4f}  device {dev_mean:.4f}  "
                      f"diff {dev_mean - r['loss']:+.4f}")

    if args.json_out and summary:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
