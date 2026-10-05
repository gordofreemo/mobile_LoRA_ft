#!/usr/bin/env python3
"""Summarize the h14 cap sweeps (memory-consistency rerun + 8B feasibility).

    python3 eval/h14_capsweep_summary.py results/ondevice/train_bench_metrics_h14_nax-on_2026-09-11.jsonl

Per (model, gradient_checkpointing, num_lora_layers, seq_cap): windows kept (first
window dropped, as h4 did), mean/max peak memory in MiB, mean iter/s, last thermal
state; then the cap_start sentinels without train rows, i.e. the jetsam'd caps.
"""
import collections, json, sys

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
cells, starts = collections.defaultdict(list), collections.defaultdict(set)
for r in recs:
    key = (r["model"], r["gradient_checkpointing"], r["num_lora_layers"])
    if r["record_type"] == "train":
        cells[key + (r["seq_cap"],)].append(r)
    elif r["record_type"] in ("cap_start", "oom"):
        starts[key].add(r["seq_cap"])
print(f"{'model':<24}{'GC':<6}{'blk':<5}{'cap':>6}{'n':>4}{'peak MiB':>10}{'max':>8}{'iter/s':>8}  thermal  os")
for key in sorted(cells):
    rs = [r for r in cells[key] if r["step"] > 5] or cells[key]
    pk = [r["peak_mem_bytes"] / 2**20 for r in rs]
    ips = [r["iter_per_sec"] for r in rs]
    m, gc, blk, cap = key
    print(f"{m:<24}{str(gc):<6}{blk:<5}{cap:>6}{len(rs):>4}{sum(pk)/len(pk):>10.0f}{max(pk):>8.0f}{sum(ips)/len(ips):>8.3f}  {cells[key][-1]['thermal_state']:<8} {cells[key][-1]['os_version']}")
for key, caps in sorted(starts.items()):
    done = {k[3] for k in cells if k[:3] == key}
    dead = sorted(caps - done)
    if dead:
        print(f"jetsam: {key} at cap(s) {dead}")
