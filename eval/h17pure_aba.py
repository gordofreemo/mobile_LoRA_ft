#!/usr/bin/env python3
"""h17pure: pure single-arm sessions, dequant / on / dequant (ABA). In a pure session every
step follows a step of the same arm, so this is the per-step cost an app using that arm pays.
Averaging the two dequant sessions cancels linear drift across the three sessions."""
import json, statistics, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results/ondevice"
def load(tag):
    p = R / f"train_bench_metrics_naxab_{tag}_2026-09-23.jsonl"
    if not p.exists(): return None
    rs = [json.loads(l) for l in open(p) if l.strip()]
    st = [r for r in rs if r.get("record_type") == "run_start" and r.get("ab_arms")]
    if not st: return None
    sid = st[-1]["bench_session_id"]
    return [r for r in rs if r.get("bench_session_id") == sid and r.get("record_type") == "iter" and not r.get("warmup")]
med = lambda xs: statistics.median(xs) if xs else float("nan")
S = {t: load(t) for t in ("h17pure_dq1", "h17pure_on", "h17pure_dq2")}
def cell(rs, tok, mode, key):
    return med([r[key] for r in (rs or []) if r["target_tokens"] == tok and r["mode"] == mode])
print(f"{'tok':>5} | {'fused dq1':>9} {'fused on':>9} {'fused dq2':>9} {'dq/on':>6} | {'fwd dq':>7} {'fwd on':>7} {'bwd dq':>7} {'bwd on':>7} {'bwd dq/on':>9} | thermal")
for tok in (50, 100, 250, 500):
    d1, on, d2 = (cell(S[t], tok, "fused", "iter_seconds") for t in S)
    dq = statistics.mean([x for x in (d1, d2) if x == x])
    fdq = statistics.mean([cell(S[t], tok, "barriered", "phase_forward_s") for t in ("h17pure_dq1", "h17pure_dq2") if S[t]])
    bdq = statistics.mean([cell(S[t], tok, "barriered", "phase_backward_s") for t in ("h17pure_dq1", "h17pure_dq2") if S[t]])
    fon = cell(S["h17pure_on"], tok, "barriered", "phase_forward_s"); bon = cell(S["h17pure_on"], tok, "barriered", "phase_backward_s")
    th = {t: sorted({r["thermal_state"] for r in (S[t] or []) if r["target_tokens"] == tok}) for t in S}
    print(f"{tok:>5} | {d1:>9.3f} {on:>9.3f} {d2:>9.3f} {dq/on:>6.2f} | {fdq:>7.3f} {fon:>7.3f} {bdq:>7.3f} {bon:>7.3f} {bdq/bon:>9.2f} | {th}")
