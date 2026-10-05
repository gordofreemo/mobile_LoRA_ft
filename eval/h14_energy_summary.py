#!/usr/bin/env python3
"""h9 energy arithmetic for an h14 unplugged e2e run (nax_arm on).
gross = drained fraction x FULL_J; net = gross - IDLE_W x duration. FULL_J uses the
rated 3988 mAh (Apple EU declaration) at a nominal 3.87 V = 55,553 J; the paper's
earlier 3998 mAh figure gives 55,700 J (also printed).
"""
import json, sys, statistics
IDLE_W = 0.2612
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for sid in {r["bench_session_id"] for r in recs}:
    rs = [r for r in recs if r["bench_session_id"] == sid]
    start = next((r for r in rs if r["record_type"] == "run_start"), None)
    end = next((r for r in rs if r["record_type"] in ("run_end", "error")), None)
    bat = sorted([r for r in rs if r["record_type"] == "battery"], key=lambda r: r["elapsed_s"])
    tr = sorted([r for r in rs if r["record_type"] == "train"], key=lambda r: r["step"])
    if not (start and bat and tr):
        print(sid[:8], "incomplete"); continue
    b0 = start.get("battery_level", bat[0]["battery_level"]); b1 = (end or bat[-1]).get("battery_level_end", bat[-1]["battery_level"])
    dur = (end or bat[-1])["elapsed_s"]
    iters = tr[-1]["step"]; tok = statistics.mean(r["tok_per_sec"] / r["iter_per_sec"] for r in tr if r["iter_per_sec"] > 0) * iters
    for label, full in (("3988mAh", 3988 * 3.87 * 3.6), ("3998mAh", 3998 * 3.87 * 3.6)):
        gross = (b0 - b1) * full; net = gross - IDLE_W * dur
        print(f"{sid[:8]} {label}: battery {b0:.2f}->{b1:.2f} dur {dur/3600:.2f} h iters {iters} tokens {tok:.0f} "
              f"gross {gross:.0f} J net {net:.0f} J = {100*net/full:.1f}% of charge, {net/iters:.1f} J/iter, {net/tok:.4f} J/tok, avg {net/dur:.2f} W")
    print(f"{sid[:8]} charging flags seen: {sorted(set(r.get('charging') for r in bat))}, os {start.get('os_version')}, blocks {start.get('num_lora_layers')}")
