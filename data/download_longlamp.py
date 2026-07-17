#!/usr/bin/env python3
"""
Download LongLaMP's Product Review Writing task (user/cold-start split) from
the public HF dataset `LongLaMP/LongLaMP`, config `product_review_user`.

LongLaMP (arXiv:2407.11016) is a second, long-form personalized-generation
benchmark, run on a separate track (LL1, LL2, ...) from this project's LaMP
work. Personalized Email Completion is excluded from the public HF release
(private Avocado corpus — the same reason LaMP-6 is excluded from
data/download_lamp.py); the release ships only
{abstract_generation, product_review, topic_writing} x {user, temporal}.

This script downloads only `product_review_user` (LL1's pinned scope — see
experiments/2026-07-17-longlamp-review-ll1-plan.md, decision #2). Confirmed
schema per split record (verified via the HF datasets-server API, not
guessed):
    reviewerId: string
    input: string        # already a fully-templated instruction
    output: string        # target review text
    profile: list[{description, overall, reviewText, summary}]

Output structure (mirrors data/lamp/'s per-task-dir shape, one JSON array per
split, no separate questions/outputs files since LongLaMP already bundles
input+output+profile per record):
    data/longlamp/product_review_user/
        train.json
        val.json
        test.json

Usage:
    python data/download_longlamp.py
    python data/download_longlamp.py --limit 20   # smoke test
"""

import argparse
import json
import os
from pathlib import Path

DATASET_NAME = "LongLaMP/LongLaMP"

# --split-type is reserved for future `temporal` support (LL1 is user-split
# only per the pinned plan); "user" is the only implemented value this round.
CONFIG_NAMES = {
    "user": "product_review_user",
    "temporal": "product_review_temporal",
}

# HF dataset split name -> local filename. LongLaMP calls its dev split
# "validation"; we write it as val.json to match data/lamp_time/'s "dev"-ish
# shorthand convention loosely (LaMP itself uses "dev", but LongLaMP's own HF
# split key is "validation" — keeping val.json avoids implying a false 1:1
# naming match with LaMP's dev/test).
SPLITS = {"train": "train.json", "validation": "val.json", "test": "test.json"}

DEFAULT_OUT_DIR = Path(__file__).parent / "longlamp"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-type", choices=["user"], default="user",
                   help="Which LongLaMP split variety to download. Only 'user' "
                        "(cold-start, disjoint users) is implemented this round; "
                        "'temporal' is reserved for future User-LoRA work.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override the output directory. Defaults to "
                        "data/longlamp/product_review_user/. The "
                        "LONGLAMP_OUT_DIR env var also overrides the default "
                        "but is itself overridden by this flag.")
    p.add_argument("--limit", type=int, default=0,
                   help="cap records per split (smoke testing)")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite existing split JSON files (default: refuse)")
    return p.parse_args()


def main():
    args = parse_args()
    config_name = CONFIG_NAMES[args.split_type]

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif "LONGLAMP_OUT_DIR" in os.environ:
        out_dir = Path(os.environ["LONGLAMP_OUT_DIR"])
    else:
        out_dir = DEFAULT_OUT_DIR / config_name

    print(f"[run] download_longlamp config={config_name} limit={args.limit} "
          f"out_dir={out_dir}", flush=True)

    existing = [out_dir / fname for fname in SPLITS.values() if (out_dir / fname).exists()]
    if existing and not args.overwrite:
        import sys
        print("ERROR: refusing to overwrite existing files:", file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        print("\nPass --overwrite to replace.", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    for hf_split, fname in SPLITS.items():
        print(f"  loading split={hf_split}...", flush=True)
        ds = load_dataset(DATASET_NAME, config_name, split=hf_split)
        records = []
        for i, rec in enumerate(ds):
            if args.limit > 0 and i >= args.limit:
                break
            records.append({
                "reviewerId": rec["reviewerId"],
                "input": rec["input"],
                "output": rec["output"],
                "profile": rec["profile"],
            })
        dest = out_dir / fname
        dest.write_text(json.dumps(records))
        print(f"  {hf_split}: {len(records)} records -> {dest}", flush=True)

    print("\nDone.")
    print("Spot-check one record's schema (reviewerId, input, output, profile) "
          "before trusting downstream builds.")


if __name__ == "__main__":
    main()
