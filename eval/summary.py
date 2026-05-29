#!/usr/bin/env python3
"""
Print a one-row-per-run table of every result JSON in results/.

Sorts by (task, retriever) so the non-personalized floor and the profile baseline
sit next to each other for each task — easy visual read of the floor→profile lift.

Usage:
    python eval/summary.py
    python eval/summary.py results/LaMP_3_*.json   # subset
"""

import glob
import json
import sys
from pathlib import Path

paths = sys.argv[1:] or sorted(glob.glob("results/*.json"))
if not paths:
    print("No result files matched.")
    sys.exit(0)

rows = [json.loads(Path(p).read_text()) for p in paths]
rows.sort(key=lambda r: (r["task"], r["retriever"], r["adapter_name"]))

# Header
hdr = (
    f"{'task':<7}  {'cond':<10}  {'adapter':<10}  {'n':>5}  "
    f"{'metric':<9}  {'mae':>6}  {'parse_fail':>10}  "
    f"{'prompt_tok':>10}  {'gen_tok':>7}  {'s/ex':>5}"
)
print(hdr)
print("-" * len(hdr))

def fmt(value, spec):
    """Format a scalar with `spec`, or '-' if it's None (task-specific metrics
    like MAE / parse_fail_rate are only populated for the rating task)."""
    return format(value, spec) if isinstance(value, (int, float)) else "-"


for r in rows:
    cond = "no-profile" if r["no_profile"] else f"bm25k{r['k']}"
    print(
        f"{r['task']:<7}  {cond:<10}  {r['adapter_name']:<10}  {r['n']:>5}  "
        f"{r['metric_name']}={r['metric_value']:.4f}  "
        f"{fmt(r.get('mae'), '.3f'):>6}  "
        f"{fmt(r.get('parse_fail_rate'), '.4f'):>10}  "
        f"{r['mean_prompt_tokens']:>10.1f}  {r['mean_generated_tokens']:>7.1f}  "
        f"{r['sec_per_example']:>5.2f}"
    )
