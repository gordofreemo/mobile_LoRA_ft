#!/usr/bin/env python3
"""
Download a LongLaMP task's user/cold-start split from the public HF dataset
`LongLaMP/LongLaMP`.

LongLaMP (arXiv:2407.11016) is a second, long-form personalized-generation
benchmark, run on a separate track (LL1, LL2, LL3, ...) from this project's
LaMP work. Personalized Email Completion is excluded from the public HF
release (private Avocado corpus — the same reason LaMP-6 is excluded from
data/download_lamp.py); the release ships only
{abstract_generation, product_review, topic_writing} x {user, temporal}.

Three tasks supported (LL1/LL2/LL3's pinned scope): product_review_user,
abstract_generation_user, topic_writing_user. Schema confirmed per task via
the HF datasets-server API (not guessed) — the per-record user-id field
name differs by task:
    product_review_user:      reviewerId, input, output, profile[{description,overall,reviewText,summary}]
    abstract_generation_user: name,       input, output, profile[{abstract,id,title,year}]
    topic_writing_user:       author,     input, output, profile[{author,content,id,summary}]

Output structure (mirrors data/lamp/'s per-task-dir shape, one JSON array per
split, no separate questions/outputs files since LongLaMP already bundles
input+output+profile per record):
    data/longlamp/<task>/
        train.json
        val.json
        test.json

Usage:
    python data/download_longlamp.py --task product_review_user
    python data/download_longlamp.py --task abstract_generation_user
    python data/download_longlamp.py --task topic_writing_user
    python data/download_longlamp.py --task topic_writing_user --limit 20   # smoke test
"""

import argparse
import json
import os
from pathlib import Path

DATASET_NAME = "LongLaMP/LongLaMP"

# user-id field name per task -- the only per-task schema difference that
# matters for this script (everything else is input/output/profile alike).
TASKS = {
    "product_review_user": "reviewerId",
    "abstract_generation_user": "name",
    "topic_writing_user": "author",
}

# HF dataset split name -> local filename. Confirmed against a real error
# from the first LL1 download attempt (cluster 174653, 2026-07-17): the
# actual HF split key is "val" (not "validation", which was an unverified
# assumption — `ValueError: Unknown split "validation". Should be one of
# ['train', 'val', 'test'].`).
SPLITS = {"train": "train.json", "val": "val.json", "test": "test.json"}

DEFAULT_OUT_DIR = Path(__file__).parent / "longlamp"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(TASKS.keys()))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override the output directory. Defaults to "
                        "data/longlamp/<task>/. The LONGLAMP_OUT_DIR env var "
                        "also overrides the default but is itself overridden "
                        "by this flag.")
    p.add_argument("--limit", type=int, default=0,
                   help="cap records per split (smoke testing)")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite existing split JSON files (default: refuse)")
    return p.parse_args()


def main():
    args = parse_args()
    user_id_field = TASKS[args.task]

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif "LONGLAMP_OUT_DIR" in os.environ:
        out_dir = Path(os.environ["LONGLAMP_OUT_DIR"])
    else:
        out_dir = DEFAULT_OUT_DIR / args.task

    print(f"[run] download_longlamp task={args.task} limit={args.limit} "
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
        ds = load_dataset(DATASET_NAME, args.task, split=hf_split)
        records = []
        for i, rec in enumerate(ds):
            if args.limit > 0 and i >= args.limit:
                break
            # Keep the original per-task field name on disk (reviewerId /
            # name / author) rather than renaming to a generic key -- this
            # is a no-op for product_review_user, so LL1's already-downloaded
            # data and everything built from it (BM25 corpus, checkpoints,
            # eval results) stays untouched and doesn't need re-fetching.
            records.append({
                user_id_field: rec[user_id_field],
                "input": rec["input"],
                "output": rec["output"],
                "profile": rec["profile"],
            })
        dest = out_dir / fname
        dest.write_text(json.dumps(records))
        print(f"  {hf_split}: {len(records)} records -> {dest}", flush=True)

    print("\nDone.")
    print(f"Spot-check one record's schema ({user_id_field}, input, output, "
          "profile) before trusting downstream builds.")


if __name__ == "__main__":
    main()
