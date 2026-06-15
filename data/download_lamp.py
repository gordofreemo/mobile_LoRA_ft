#!/usr/bin/env python3
"""
Download LaMP benchmark splits (LaMP-3, LaMP-4, LaMP-7) from the official host.
Source: https://lamp-benchmark.github.io/download

LaMP publishes two split varieties for the same underlying interactions:

  user  — users are disjoint across train/dev/test. Used for the Task-LoRA
          (A1-lamp); the goal there is generalization to unseen users.
          Files at: ciir.cs.umass.edu/downloads/LaMP/{task}/{split}/...

  time  — same users appear in every split, but each user's interactions are
          partitioned chronologically (earlier in train, later in dev/test).
          This is the split needed for the per-user User-LoRA experiments
          (Q4, the active research question after the 2026-06-02 pivot).
          Files at: ciir.cs.umass.edu/downloads/LaMP/time/{task}/{split}/...

Each split has two files:
  {split}_questions.json  — the inputs (user profile + query)
  {split}_outputs.json    — the labels (ground truth answers)

Output structure (split-type-dependent so the two corpora never clobber
each other):
    data/lamp/         <-- --split-type user (default)
    data/lamp_time/    <-- --split-type time
        LaMP_3/
            train_questions.json
            train_outputs.json
            dev_questions.json
            dev_outputs.json
            test_questions.json     # labels may be withheld (leaderboard-only)
            test_outputs.json
        LaMP_4/
            ...
        LaMP_7/
            ...

Usage:
    python data/download_lamp.py                       # user-based split (default)
    python data/download_lamp.py --split-type time     # time-based split
"""

import argparse
import os
import urllib.error     # HTTPError / URLError live in this submodule
import urllib.request   # standard library HTTP client — no extra dependencies needed
from pathlib import Path

# Base URL where all LaMP files are hosted.
BASE_URL = "https://ciir.cs.umass.edu/downloads/LaMP"

# Tasks we download. LaMP-6 is excluded — its public release ships only
# Avocado email file-id placeholders (no text), so it can't be scored without
# the licensed Avocado corpus.
TASKS = ["LaMP_3", "LaMP_4", "LaMP_7"]

# Each task has three splits. "dev" is what the LaMP benchmark calls validation.
# Test outputs may be withheld by the authors (held for leaderboard evaluation).
SPLITS = ["train", "dev", "test"]

# Each split has two files: questions (inputs) and outputs (labels).
FILE_TYPES = ["questions", "outputs"]

# Default output directory per split type. Override either with --out-dir or
# the LAMP_OUT_DIR env var (env var wins over the default but not over the
# CLI flag). The two split varieties land in separate dirs by default so they
# can coexist without clobbering each other.
DEFAULT_OUT_DIRS = {
    "user": Path(__file__).parent / "lamp",
    "time": Path(__file__).parent / "lamp_time",
}


def download_file(url: str, dest: Path):
    """
    Download a single file from `url` and save it to `dest`.
    Skips if the file already exists — safe to re-run without re-downloading.
    """
    if dest.exists():
        print(f"  Already exists, skipping: {dest.name}")
        return

    print(f"  Downloading {url}")
    try:
        # urllib.request.urlretrieve fetches a URL and writes it directly to disk.
        # It's the simplest way to download a file without installing requests/httpx.
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved -> {dest}")
    except urllib.error.HTTPError as e:
        # HTTPError means the server responded but with an error code (404, 403, etc.)
        # Test outputs are often withheld by benchmark authors for leaderboard integrity,
        # so a 404 on test_outputs.json is expected and not a real problem.
        print(f"  HTTP {e.code} — skipping {dest.name} (may be withheld)")
    except urllib.error.URLError as e:
        # URLError means we couldn't reach the server at all (DNS failure, no network, etc.)
        print(f"  Network error: {e.reason}")
        raise


def task_base_url(split_type: str, task: str) -> str:
    """Return the base URL for one task, with the `/time/` segment inserted
    for the time-based split. The user-based split is the legacy default and
    has no extra segment."""
    if split_type == "time":
        return f"{BASE_URL}/time/{task}"
    return f"{BASE_URL}/{task}"


def download_task(task: str, split_type: str, out_dir: Path):
    """Download all splits and file types for one LaMP task.

    URL structure:
      user:  {BASE_URL}/{task}/{split}/{split}_{file_type}.json
      time:  {BASE_URL}/time/{task}/{split}/{split}_{file_type}.json
    """
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{task}")
    for split in SPLITS:
        for file_type in FILE_TYPES:
            filename = f"{split}_{file_type}.json"
            # Each split lives in its own subdirectory under the task directory.
            url = f"{task_base_url(split_type, task)}/{split}/{filename}"
            dest = task_dir / filename
            download_file(url, dest)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-type", choices=["user", "time"], default="user",
                   help="Which LaMP split variety to download. 'user' (default) "
                        "= disjoint users across train/dev/test (used for the "
                        "A1-lamp Task-LoRA). 'time' = same users, chronological "
                        "partition (used for per-user User-LoRA experiments).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override the output directory. Defaults to "
                        "data/lamp/ (user split) or data/lamp_time/ (time split). "
                        "The LAMP_OUT_DIR env var also overrides the default "
                        "but is itself overridden by this flag.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif "LAMP_OUT_DIR" in os.environ:
        out_dir = Path(os.environ["LAMP_OUT_DIR"])
    else:
        out_dir = DEFAULT_OUT_DIRS[args.split_type]

    print(f"Split type: {args.split_type}")
    print(f"Output dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        download_task(task, args.split_type, out_dir)

    print("\nDone.")
    if args.split_type == "user":
        print("Verify no user profile leakage between splits before training.")
    else:
        print("Time-based split: same users appear in every split by "
              "construction; the no-overlap rule applies *within* each user's "
              "interactions (LaMP enforces this chronologically).")
