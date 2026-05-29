#!/usr/bin/env python3
"""
Download LaMP benchmark splits (LaMP-3, LaMP-4, LaMP-7) from the official host.
Source: https://lamp-benchmark.github.io/download

Files are hosted at ciir.cs.umass.edu/downloads/LaMP/ and come as pairs:
  {split}_questions.json  — the inputs (user profile + query)
  {split}_outputs.json    — the labels (ground truth answers)

Output structure:
    data/lamp/
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
    python data/download_lamp.py
"""

import os
import urllib.request   # standard library HTTP client — no extra dependencies needed
from pathlib import Path

# Base URL where all LaMP files are hosted
BASE_URL = "https://ciir.cs.umass.edu/downloads/LaMP"

# Tasks we download. LaMP-6 is excluded — see note at top of file.
# Each entry maps our directory name to the split file prefixes available.
TASKS = ["LaMP_3", "LaMP_4", "LaMP_6", "LaMP_7"]

# Each task has three splits. "dev" is what the LaMP benchmark calls validation.
# Test outputs may be withheld by the authors (held for leaderboard evaluation).
SPLITS = ["train", "dev", "test"]

# Each split has two files: questions (inputs) and outputs (labels).
FILE_TYPES = ["questions", "outputs"]

# Allow overriding the output directory via environment variable.
# On the cluster, set LAMP_OUT_DIR to the absolute path in your home directory
# so files persist after the job ends. Defaults to data/lamp/ for local use.
OUT_DIR = Path(os.environ.get("LAMP_OUT_DIR", Path(__file__).parent / "lamp"))


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


def download_task(task: str):
    """Download all splits and file types for one LaMP task.

    URL structure: {BASE_URL}/{task}/{split}/{split}_{file_type}.json
    e.g. https://ciir.cs.umass.edu/downloads/LaMP/LaMP_3/train/train_questions.json

    We use the user-based split (the default at this URL) because our goal is
    generalization to new unseen users, not predicting future behavior of known users.
    """
    task_dir = OUT_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{task}")
    for split in SPLITS:
        for file_type in FILE_TYPES:
            filename = f"{split}_{file_type}.json"
            # Each split lives in its own subdirectory under the task directory.
            url = f"{BASE_URL}/{task}/{split}/{filename}"
            dest = task_dir / filename
            download_file(url, dest)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        download_task(task)

    print("\nDone.")
    print("Verify no user profile leakage between splits before training.")
