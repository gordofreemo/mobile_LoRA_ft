#!/usr/bin/env python3
"""Summarize the h14 e2e repeat runs against the original 2026-08-07 repaired-kernel run.

    python3 eval/h14_repeats_summary.py results/ondevice/train_bench_metrics_naxab_e2e_2026-08-07.jsonl \
        results/ondevice/train_bench_metrics_naxab_e2e_h14rep*_*.jsonl

Per session with nax_arm==on: wall time (run_end.elapsed_s), iterations, tokens,
s/token, mean loss over the last 6 reports, throttled floor (mean of the trailing 50%
of iter/s normalized to the first window), num_lora_layers, os_version, app_build.
Then mean +- sd across the 28-block sessions.
"""
import json, statistics, sys

sessions = {}
for path in sys.argv[1:]:
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("nax_arm") != "on" and "nax_arm" in r:
            continue
        s = sessions.setdefault(r["bench_session_id"], {"train": [], "meta": r, "end": None})
        if r["record_type"] == "train":
            s["train"].append(r)
        elif r["record_type"] == "run_end":
            s["end"] = r
        if r["record_type"] == "run_start":
            s["meta"] = r
rows = []
for sid, s in sessions.items():
    tr = sorted(s["train"], key=lambda r: r["step"])
    if not tr or s["end"] is None:
        print(f"{sid[:8]}: incomplete ({len(tr)} windows, run_end={'yes' if s['end'] else 'no'})")
        continue
    wall = s["end"]["elapsed_s"]
    iters = tr[-1]["step"]
    tok_per_step = statistics.mean(r["tok_per_sec"] / r["iter_per_sec"] for r in tr if r["iter_per_sec"] > 0)
    tokens = tok_per_step * iters
    ips = [r["iter_per_sec"] for r in tr]
    floor = statistics.mean(ips[len(ips)//2:]) / ips[0]
    loss_tail = statistics.mean(r["training_loss"] for r in tr[-6:])
    m = s["meta"]
    rows.append((m.get("app_build"), m.get("num_lora_layers"), m.get("os_version"), wall, iters, tokens, wall / tokens, floor, loss_tail, sid[:8]))
print(f"{'build':<40}{'blk':>4}{'wall s':>9}{'iters':>6}{'tokens':>10}{'s/tok':>9}{'floor':>7}{'loss6':>8}  os / session")
for b, blk, os_, wall, it, tok, spt, fl, ls, sid in sorted(rows, key=lambda x: x[0] or ""):
    print(f"{str(b):<40}{blk:>4}{wall:>9.0f}{it:>6}{tok:>10.0f}{spt:>9.5f}{fl:>7.3f}{ls:>8.4f}  {os_} {sid}")
w28 = [r[3] for r in rows if r[1] == 28]
if len(w28) > 1:
    print(f"\n28-block runs: n={len(w28)} wall mean {statistics.mean(w28):.0f} s, sd {statistics.stdev(w28):.0f} s "
          f"({100*statistics.stdev(w28)/statistics.mean(w28):.1f}%), range {min(w28):.0f}-{max(w28):.0f}")
