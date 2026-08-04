#!/usr/bin/env python3
"""
Generation-length / degeneration rate for ANY LongLaMP predictions JSONL.

WHY THIS EXISTS
---------------
eval/longlamp_degeneration_audit.py answers "did the LL4-LL6 paired TEST
comparison sit on a repetition confound?" -- it is hardwired to a
baseline-vs-personalized pair drawn from the top-100 pool. That is the right
shape for auditing a finished round and the wrong shape for the decoding sweep,
which scores single unfiltered DEV runs one at a time and needs an answer before
deciding what to run next.

This script is the sweep's mechanical stopping rule. It reads any predictions
file the eval harness writes and reports the degeneration rate under the SAME
flat word-count criterion, so a sweep result and an audit result are directly
comparable.

Deliberately stdlib-only -- no rouge_score, no scipy -- so it runs on the login
host. A decoding sweep is an interactive loop; needing a Condor round-trip to
read each cell would make it unusable.

THRESHOLD -- CALIBRATED, DO NOT RE-DERIVE
-----------------------------------------
Degeneration is a FLAT word count, len(pred.split()) > --degenerate-words
(default 600), calibrated on 2026-08-04 against the no-adapter base-model arm
(commit 729692f), whose longest Abstract generation is 358 words:

    flat >400w   ->  0/100 base flagged, 30/100 Task-LoRA flagged
    flat >600w   ->  0/100 base flagged, 30/100 Task-LoRA flagged   <-- default
    2.0x gold    -> 35/100 base flagged  (gold length varies too much)

A gold-relative ratio is the wrong tool: at 2x it flags 35/100 base generations
that plainly did not run away. See the audit script's header and
experiments/2026-08-04-longlamp-degeneration-confound.md.

SUCCESS CRITERION for the decoding sweep, pre-registered and mechanical:
degeneration rate at >600 words near zero on EVERY arm. Select on --split dev,
never on test, and apply the winning config identically to every arm.

Usage:
    python eval/longlamp_length_check.py results/LongLaMP_*_dev_*_limit50.predictions.jsonl
    python eval/longlamp_length_check.py --json-out results/dev_sweep_summary.json results/*.jsonl
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def quantile(xs, q):
    """Nearest-rank quantile. Avoids a numpy dependency for three numbers."""
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def summarize_file(path: Path, degenerate_words: int, ratio: float) -> dict:
    preds, golds = [], []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                sys.exit(f"ERROR: malformed JSONL at {path}:{line_no}")
            preds.append(len(str(rec.get("pred", "")).split()))
            golds.append(len(str(rec.get("gold", "")).split()))

    if not preds:
        return {"file": path.name, "n": 0, "error": "empty"}

    n_deg = sum(1 for p in preds if p > degenerate_words)
    n_deg_ratio = sum(1 for p, g in zip(preds, golds) if g and p > ratio * g)

    return {
        "file": path.name,
        "n": len(preds),
        "degenerate_words": degenerate_words,
        "n_degenerate": n_deg,
        "pct_degenerate": round(100.0 * n_deg / len(preds), 1),
        "n_degenerate_ratio": n_deg_ratio,
        "length_ratio": ratio,
        "pred_words_median": statistics.median(preds),
        "pred_words_mean": round(statistics.fmean(preds), 1),
        "pred_words_p90": quantile(preds, 0.90),
        "pred_words_max": max(preds),
        "gold_words_median": statistics.median(golds),
        "gold_words_mean": round(statistics.fmean(golds), 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Degeneration rate for LongLaMP predictions JSONL files.")
    parser.add_argument("predictions", nargs="+",
                        help="one or more *.predictions.jsonl paths")
    parser.add_argument(
        "--degenerate-words", type=int, default=600,
        help="flag a generation as degenerate above this word count. "
        "Calibrated against the no-adapter arm -- see module docstring. "
        "Default 600.")
    parser.add_argument(
        "--length-ratio", type=float, default=4.0,
        help="secondary, reported but NOT the criterion: flag when a "
        "generation exceeds this multiple of its own gold. Default 4.0.")
    parser.add_argument(
        "--json-out", default=None,
        help="also write the rows to this path as a JSON list.")
    args = parser.parse_args()

    rows = []
    for p in args.predictions:
        path = Path(p)
        if not path.exists():
            print(f"WARN: missing, skipping: {path}", file=sys.stderr)
            continue
        rows.append(summarize_file(path, args.degenerate_words, args.length_ratio))

    if not rows:
        sys.exit("ERROR: no readable predictions files.")

    width = max(len(r["file"]) for r in rows)
    print(f"{'file':<{width}}  {'n':>4}  {'>' + str(args.degenerate_words) + 'w':>8}  "
          f"{'%':>6}  {'med':>5}  {'p90':>5}  {'max':>6}  {'gold_med':>8}")
    print("-" * (width + 52))
    for r in rows:
        if r.get("error"):
            print(f"{r['file']:<{width}}  {'--':>4}  {r['error']}")
            continue
        print(f"{r['file']:<{width}}  {r['n']:>4}  {r['n_degenerate']:>8}  "
              f"{r['pct_degenerate']:>6}  {r['pred_words_median']:>5}  "
              f"{r['pred_words_p90']:>5}  {r['pred_words_max']:>6}  "
              f"{r['gold_words_median']:>8}")

    worst = max((r for r in rows if not r.get("error")),
                key=lambda r: r["pct_degenerate"], default=None)
    if worst:
        print()
        print(f"[criterion] worst arm: {worst['file']} at "
              f"{worst['pct_degenerate']}% degenerate (>{args.degenerate_words}w). "
              f"Sweep target is near zero on EVERY arm.")

    if args.json_out:
        out = Path(args.json_out)
        if out.exists():
            sys.exit(f"ERROR: refusing to overwrite {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
        print(f"[write] {out}")


if __name__ == "__main__":
    main()
