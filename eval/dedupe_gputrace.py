#!/usr/bin/env python3
"""Re-deduplicate a pulled `.gputrace` bundle by collapsing identical files to
symlinks — restoring the layout Metal originally wrote.

WHY (h11 Tier 2, 2026-08-04). Metal writes duplicate GPU buffers inside a
capture as symlinks (`MTLBuffer-26430-0 -> MTLBuffer-24997-0`). `devicectl`
cannot read symlinks at all — it aborts a transfer with openat(2) ELOOP — so
the harness converts them to hard links before the pull (see flattenSymlinks in
LLMEvaluator+TrainBenchmark.swift). Hard links keep the dedup ON DEVICE, but
the Mac-side copy necessarily expands each one into a distinct file: forward
went 1026 links -> 6.0 GB, backward 2526 -> 13 GB.

That expansion breaks REPLAY. Xcode uploads the bundle's resources back to the
phone to re-execute the capture, so an expanded bundle asks the replay guest to
allocate several GB where the original needed a fraction of that. Observed: the
1.9 GB optimizer trace replayed fine, the 6.0 GB forward trace killed the guest
app ("guest app crashed (512)", DYErrorDomain).

This script reverses the expansion locally. Files are grouped by size, hashed
within each size group, and every duplicate is replaced by a relative symlink to
the lowest-numbered member — Metal's own convention (links point at lower buffer
IDs). Byte-for-byte identical to what came off the device, just stored once.

Idempotent, and skips existing symlinks, so it is safe to re-run.

    python eval/dedupe_gputrace.py results/ondevice/captures/forward_*.gputrace
"""

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path


def file_digest(path, chunk=1 << 20):
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.digest()


def buffer_sort_key(path):
    """Metal points duplicates at the LOWEST buffer id, so keep that one."""
    digits = [int(t) for t in path.name.replace("-", " ").split() if t.isdigit()]
    return (digits or [0], path.name)


def assert_settled(bundle, seconds=3.0):
    """Refuse to run on a bundle that is still being written.

    Learned the hard way 2026-08-04: dedupe was run against a `devicectl` pull
    that was still in flight. Two partially-written files hash equal, so they
    get collapsed into one — and the damage is INVISIBLE afterwards (no broken
    links, no zero-byte files, plausible size). It reported 1.32 GB where the
    real deduped size was 1.99 GB, i.e. it silently destroyed content. A bundle
    handed to Xcode must be byte-exact, so this refuses rather than warns.
    """

    def stat():
        files = [p for p in bundle.iterdir() if p.is_file() and not p.is_symlink()]
        return len(files), sum(p.stat().st_size for p in files)

    first = stat()
    time.sleep(seconds)
    second = stat()
    if first != second:
        print(
            f"{bundle.name}: still changing ({first} -> {second}) — the transfer is "
            "probably still running. Refusing to dedupe.",
            file=sys.stderr,
        )
        sys.exit(1)


def dedupe(bundle, dry_run=False):
    entries = [p for p in bundle.iterdir() if p.is_file() and not p.is_symlink()]
    before = sum(p.stat().st_size for p in entries)

    by_size = defaultdict(list)
    for p in entries:
        by_size[p.stat().st_size].append(p)

    linked = 0
    reclaimed = 0
    for size, group in by_size.items():
        if len(group) < 2 or size == 0:
            continue
        by_hash = defaultdict(list)
        for p in group:
            by_hash[file_digest(p)].append(p)
        for dupes in by_hash.values():
            if len(dupes) < 2:
                continue
            dupes.sort(key=buffer_sort_key)
            keeper = dupes[0]
            for victim in dupes[1:]:
                if not dry_run:
                    victim.unlink()
                    victim.symlink_to(keeper.name)  # relative, same directory
                linked += 1
                reclaimed += size

    after = before - reclaimed
    print(
        f"{bundle.name}: {len(entries)} files, "
        f"{before / 1e9:.2f} GB -> {after / 1e9:.2f} GB "
        f"({linked} collapsed to symlinks, {reclaimed / 1e9:.2f} GB reclaimed)"
        + ("  [dry run]" if dry_run else "")
    )
    return linked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-settle-check", action="store_true",
        help="skip the in-progress-transfer guard (only if you know the pull finished)")
    args = ap.parse_args()

    for b in args.bundles:
        if not b.is_dir():
            print(f"not a bundle directory: {b}", file=sys.stderr)
            sys.exit(1)
        if not args.skip_settle_check:
            assert_settled(b)
        dedupe(b, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
