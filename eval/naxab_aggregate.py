#!/usr/bin/env python3
"""Aggregate the NAX A/B round: does routing backward's dX onto the neural
accelerator make on-device LoRA training faster?

Follow-on to h11. Reads `train_bench_metrics_naxab.jsonl`, whose cells alternate
`MLX_ENABLE_NAX_N` per iteration, so each cell yields ON and OFF measurements
seconds apart at the same die temperature — a paired comparison rather than a
cross-run one (h11's thermal swing at matched tokens is ~25%, the same order as
the effect being chased).

Two built-in validity controls, both reported before any speedup number:

  * `seq_len_aligned_64` — M reaching the quantized matmuls is (tokens - 1),
    because LoRABatchIterator slices inputs as [:, :-1]. The NAX non-transposed
    dispatch requires M % 64 == 0. If this is False the run measured the generic
    kernel in BOTH arms and the comparison is meaningless.
  * forward / optimizer — forward already used NAX before the patch and the
    optimizer contains no quantized matmul, so both should be unmoved. If either
    shifts materially, something other than the patch changed and the backward
    delta is not attributable.

Usage: python eval/naxab_aggregate.py <jsonl> [--json out.json]
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict

PHASES = ["data_prep", "graph_build", "forward", "backward", "optimizer", "readback"]


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def summarize(rows):
    """Group kept barriered iterations by (pass, tokens, arm)."""
    cells = defaultdict(lambda: defaultdict(list))
    align = defaultdict(set)
    for r in rows:
        if r.get("record_type") != "iter" or r.get("mode") != "barriered":
            continue
        if r.get("warmup"):
            continue
        arm = r.get("arm")
        if arm not in ("on", "off"):
            continue
        key = (r["pass"], r["target_tokens"])
        cells[key][arm].append(r)
        if r.get("seq_len") is not None:
            align[key].add((r["seq_len"], bool(r.get("seq_len_aligned_64"))))
    return cells, align


def fused_totals(rows):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("record_type") != "iter" or r.get("mode") != "fused":
            continue
        if r.get("warmup"):
            continue
        arm = r.get("arm")
        if arm in ("on", "off"):
            out[(r["pass"], r["target_tokens"])][arm].append(r["iter_seconds"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--json", dest="out")
    args = ap.parse_args()

    rows = load(args.jsonl)
    cells, align = summarize(rows)
    fused = fused_totals(rows)

    if not cells:
        print("no paired barriered iterations found", file=sys.stderr)
        return 1

    # --- sequence lengths actually exercised -----------------------------
    #
    # M alignment used to be a hard validity gate: before the partial-M-tile
    # handling was ported into qmm_n_nax_tgp_impl, the dispatch required
    # M % 64 == 0 and an unaligned cell would silently run the generic kernel
    # in BOTH arms, making the comparison meaningless. That is no longer true —
    # M is unconstrained — so this is now reported for information only, and
    # unaligned rows are the INTERESTING ones (they exercise the ported
    # partial-tile path). The real check that the dispatch changed is that the
    # two arms differ at all; see "dispatch actually changed" below.
    print("=" * 78)
    print("SEQUENCE LENGTHS EXERCISED (M = tokens-1; alignment informational only)")
    print("=" * 78)
    n_unaligned = 0
    for key in sorted(align):
        for seq_len, ok in sorted(align[key]):
            tag = "64-aligned" if ok else "unaligned (partial-tile path)"
            print(f"  pass={key[0]:<5} tokens={key[1]:<5} seq_len(M)={seq_len:<6} {tag}")
            if not ok:
                n_unaligned += 1
    bad = []  # M alignment is no longer a validity condition

    # --- the comparison --------------------------------------------------
    results = []
    print()
    print("=" * 78)
    print("PAIRED PHASE TIMES (barriered, warmup dropped), seconds/iteration")
    print("=" * 78)
    hdr = (f"{'pass':<5}{'tok':>6}{'n':>4} | {'backward off':>13}{'on':>10}{'ratio':>8}"
           f" | {'fwd off':>9}{'on':>9}{'Δ%':>7} | {'opt Δ%':>8}")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(cells):
        arms = cells[key]
        if not (arms.get("on") and arms.get("off")):
            continue
        rec = {"pass": key[0], "target_tokens": key[1],
               "n_on": len(arms["on"]), "n_off": len(arms["off"])}
        for ph in PHASES:
            for arm in ("on", "off"):
                rec[f"{ph}_{arm}"] = mean([r[f"phase_{ph}_s"] for r in arms[arm]])
        for arm in ("on", "off"):
            rec[f"total_{arm}"] = mean([r["iter_seconds"] for r in arms[arm]])
        b_off, b_on = rec["backward_off"], rec["backward_on"]
        f_off, f_on = rec["forward_off"], rec["forward_on"]
        o_off, o_on = rec["optimizer_off"], rec["optimizer_on"]
        rec["backward_ratio_off_over_on"] = b_off / b_on if b_on else float("nan")
        rec["backward_drop_pct"] = 100.0 * (b_off - b_on) / b_off if b_off else float("nan")
        rec["forward_delta_pct"] = 100.0 * (f_on - f_off) / f_off if f_off else float("nan")
        rec["optimizer_delta_pct"] = 100.0 * (o_on - o_off) / o_off if o_off else float("nan")
        rec["total_ratio_off_over_on"] = (
            rec["total_off"] / rec["total_on"] if rec["total_on"] else float("nan"))
        fk = fused.get(key, {})
        if fk.get("on") and fk.get("off"):
            rec["fused_off"] = mean(fk["off"])
            rec["fused_on"] = mean(fk["on"])
            rec["fused_speedup"] = rec["fused_off"] / rec["fused_on"]
        results.append(rec)
        print(f"{key[0]:<5}{key[1]:>6}{len(arms['on']):>4} | "
              f"{b_off:>13.3f}{b_on:>10.3f}{rec['backward_ratio_off_over_on']:>8.2f}x | "
              f"{f_off:>9.3f}{f_on:>9.3f}{rec['forward_delta_pct']:>+7.1f} | "
              f"{rec['optimizer_delta_pct']:>+8.1f}")

    # --- validity gate 2: the built-in controls --------------------------
    fwd = [abs(r["forward_delta_pct"]) for r in results]
    opt = [abs(r["optimizer_delta_pct"]) for r in results]
    print()
    print("=" * 78)
    print("VALIDITY: built-in controls (forward already used NAX; optimizer has "
          "no quantized matmul)")
    print("=" * 78)
    print(f"  |forward Δ|    max {max(fwd):5.1f}%  mean {mean(fwd):5.1f}%")
    print(f"  |optimizer Δ|  max {max(opt):5.1f}%  mean {mean(opt):5.1f}%")
    # Percentages alone are misleading for the optimizer: it is a ~0.06 s phase
    # (AdamW touches only the LoRA params), so a few ms of jitter reads as tens
    # of percent while being irrelevant next to a backward phase measured in
    # seconds. Print absolutes, and weigh the swing against the backward saving.
    print("\n  optimizer in absolute terms (AdamW touches only LoRA params):")
    worst = max(results, key=lambda r: abs(r["optimizer_delta_pct"]))
    for r in results:
        d = r["optimizer_on"] - r["optimizer_off"]
        print(f"    pass={r['pass']:<5} tok={r['target_tokens']:<5} "
              f"off={r['optimizer_off']:.4f}s on={r['optimizer_on']:.4f}s "
              f"Δ={d:+.4f}s ({r['optimizer_delta_pct']:+.1f}%)")
    b_save = worst["backward_off"] - worst["backward_on"]
    o_swing = abs(worst["optimizer_on"] - worst["optimizer_off"])
    print(f"\n  largest optimizer swing is {o_swing:.4f}s against a backward saving "
          f"of {b_save:.4f}s in the same cell ({o_swing / b_save * 100:.1f}% of it).")
    if max(fwd) > 5:
        print("  NOTE: forward moved >5% — attribute the backward delta with care.")
    else:
        print("  Forward control held: the backward delta is attributable to the patch.")

    # --- did the dispatch actually change? -------------------------------
    # This replaces the old M-alignment gate. If the guard had rejected these
    # shapes, both arms would run the identical generic kernel and every ratio
    # would sit at ~1.00. A ratio far from 1 IS the evidence that the ON arm
    # reached a different kernel.
    ratios = [r["backward_ratio_off_over_on"] for r in results]
    print()
    print("=" * 78)
    print("VALIDITY: dispatch actually changed")
    print("=" * 78)
    print(f"  {n_unaligned} of {len(align)} cells ran UNALIGNED M (partial-tile path).")
    print(f"  backward off/on ratio min {min(ratios):.2f}x — if the guard had rejected")
    print("  these shapes both arms would be the same kernel and this would be ~1.00x.")
    if min(ratios) < 1.05:
        print("  *** a cell is at ~1.00x — that cell did NOT take the fast path. ***")

    # --- headline --------------------------------------------------------
    print()
    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    drops = [r["backward_drop_pct"] for r in results]
    tots = [r["total_ratio_off_over_on"] for r in results]
    print(f"  backward time reduction : {min(drops):.1f}% .. {max(drops):.1f}% "
          f"(mean {mean(drops):.1f}%)")
    print(f"  barriered iteration      : {min(tots):.2f}x .. {max(tots):.2f}x "
          f"(mean {mean(tots):.2f}x)")
    fs = [r["fused_speedup"] for r in results if "fused_speedup" in r]
    if fs:
        print(f"  fused iteration (real)   : {min(fs):.2f}x .. {max(fs):.2f}x "
              f"(mean {mean(fs):.2f}x)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"cells": results,
                       "alignment_ok": not bad,
                       "control_forward_max_abs_pct": max(fwd),
                       "control_optimizer_max_abs_pct": max(opt)}, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
