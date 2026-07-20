#!/usr/bin/env python3
"""
Per-user example-count analysis for LaMP time-based splits.

Motivation
----------
A1-lamp Task-LoRA was trained on data/lamp/LaMP_*/train_questions.json
(the user-based split's TRAIN partition). The next planned ML step is a
per-user User-LoRA, stacked on top of A1-lamp. For each candidate user we
need to know:

  1. Has A1-lamp already seen this user during Task-LoRA training? Users
     whose history overlaps with user-split TRAIN are "seen" and confound
     the personalization signal. UNSEEN users are the clean experimental pool.

  2. How many time-split records does this user have? Specifically how many
     in time-split TRAIN (candidate User-LoRA fitting data) and how many in
     time-split DEV (candidate evaluation data). Determines whether a single-
     user User-LoRA is feasible.

Method
------
LaMP has no `user_id`, so we identify users by profile-text overlap:

  - seen-text set = union of all profile texts in user-split TRAIN
  - record-to-user clustering: two time-split records belong to the same
    user iff their profile-text sets share >=1 text (MD5-fingerprinted).
    Connected components over that link relation = users.
  - per-user `seen_by_a1_lamp` flag = OR over all records in the component
    of (record's profile texts intersect seen-text set non-empty).

Caveat: a user with exactly K=2 time-split records and zero overlap of
profile entries between those two records would appear as two singleton
users. With K>=3 the chance of zero overlap is negligible (the profile
of record i excludes only interaction i; record j's profile excludes
only j; they share K-2 items). For LaMP users with very few interactions
this slightly overcounts users and undercounts n_train/n_dev per user.

Outputs
-------
  data/lamp_user_stats/<task>_users.csv
    columns: user_fingerprint, n_train, n_dev, n_test,
             seen_by_a1_lamp, example_record_id
  data/lamp_user_stats/<task>_user_records.json
    {user_fingerprint: {train: [record_id, ...], dev: [...], test: [...]}, ...}
    Downstream scripts (per-user dataset builders, per-user evaluators)
    read this to filter to a chosen user's record IDs without redoing
    the union-find clustering.
  + stdout summary + cutoff table per task.
"""

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
USER_SPLIT_DIR = ROOT / "data" / "lamp"
TIME_SPLIT_DIR = ROOT / "data" / "lamp_time"
OUT_DIR = ROOT / "data" / "lamp_user_stats"

TASKS = ["LaMP_3", "LaMP_4", "LaMP_7",
          "LaMP_1", "LaMP_2_movies", "LaMP_2_news", "LaMP_5"]
# NOTE (2026-07-20, R10-R13 viability prep): the `seen_by_a1_lamp` column
# name/docstring above is literally accurate only for LaMP_{3,4,7} — A1-lamp
# was trained on exactly those 3 tasks' user-split TRAIN partitions. For the
# 4 tasks added here, A1-lamp never touched them at all; the same
# computation (profile-text overlap with this task's user-split TRAIN) is
# instead measuring "seen by One-LoRA FT" (a2_lamp_1ep_seed0/final), the
# only Task-LoRA that ever trained on them. The column is kept as-is
# (downstream scripts key off this exact name) — just don't read
# `seen_by_a1_lamp=1` on a new-task row as "A1-lamp leakage," it isn't one.
TIME_SPLITS = ["train", "dev", "test"]


def fp(s: str) -> bytes:
    """8-byte truncated MD5 fingerprint of a text. Collision-free at our scale
    (~3.5M unique texts per task; 64-bit space gives <1e-6 birthday collision
    risk)."""
    return hashlib.md5(s.encode("utf-8")).digest()[:8]


def stream_array(path):
    """Yield each top-level dict from a JSON-array file without loading the
    entire array. The LaMP train_questions.json files for LaMP_3 reach 3.4 GB
    so naive json.load OOMs even on a 47 GB node."""
    dec = json.JSONDecoder()
    with open(path, "r") as f:
        buf = f.read(1024 * 1024)
        i = 0
        while i < len(buf) and buf[i] in " \t\n\r":
            i += 1
        if i >= len(buf) or buf[i] != "[":
            raise ValueError(f"{path}: expected top-level JSON array")
        buf = buf[i + 1:]
        while True:
            while buf and buf[0] in " \t\n\r,":
                buf = buf[1:]
            if not buf:
                buf = f.read(1024 * 1024)
                if not buf:
                    return
                continue
            if buf[0] == "]":
                return
            while True:
                try:
                    obj, end = dec.raw_decode(buf)
                    buf = buf[end:]
                    yield obj
                    break
                except json.JSONDecodeError:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        return
                    buf += chunk


def build_seen_set(task: str) -> set:
    """All profile-text hashes from user-split TRAIN for `task` — the corpus
    A1-lamp saw during Task-LoRA fine-tuning."""
    path = USER_SPLIT_DIR / task / "train_questions.json"
    print(f"  scanning user-split TRAIN ({path}) ...")
    seen = set()
    n_rec = 0
    for r in stream_array(str(path)):
        n_rec += 1
        for p in r.get("profile", []):
            t = p.get("text")
            if t:
                seen.add(fp(t))
    print(f"    -> {n_rec} records, {len(seen):,} unique seen texts")
    return seen


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def analyse_task(task: str, seen_texts: set) -> list:
    """Ingest time-split records, cluster into users, return per-user dicts."""
    records = []                    # list of {split, record_id, seen_solo}
    inv = defaultdict(list)         # text_hash -> [record_idx, ...]

    for split in TIME_SPLITS:
        path = TIME_SPLIT_DIR / task / f"{split}_questions.json"
        print(f"  scanning time-split {split} ...")
        n_before = len(records)
        for r in stream_array(str(path)):
            idx = len(records)
            phs = set()
            for p in r.get("profile", []):
                t = p.get("text")
                if t:
                    h = fp(t)
                    phs.add(h)
                    inv[h].append(idx)
            records.append({
                "split": split,
                "record_id": str(r["id"]),
                "seen_solo": bool(phs & seen_texts),
            })
        print(f"    -> {len(records) - n_before} records (cumulative {len(records)})")

    print(f"  union-find over {len(records)} records via "
          f"{len(inv):,} unique profile texts ...")
    uf = UnionFind(len(records))
    for idxs in inv.values():
        if len(idxs) < 2:
            continue
        a = idxs[0]
        for j in idxs[1:]:
            uf.union(a, j)
    inv.clear()                     # free the inverted index

    components = defaultdict(list)
    for idx in range(len(records)):
        components[uf.find(idx)].append(idx)

    users = []
    for root, idxs in components.items():
        seen = False
        ids_by_split = {"train": [], "dev": [], "test": []}
        for i in idxs:
            r = records[i]
            ids_by_split[r["split"]].append(r["record_id"])
            seen = seen or r["seen_solo"]
        # representative: prefer a train record (so the user has fitting data)
        rep = (ids_by_split["train"][0] if ids_by_split["train"]
               else ids_by_split["dev"][0] if ids_by_split["dev"]
               else ids_by_split["test"][0])
        users.append({
            "user_fingerprint": f"u{root:08d}",
            "n_train": len(ids_by_split["train"]),
            "n_dev": len(ids_by_split["dev"]),
            "n_test": len(ids_by_split["test"]),
            "seen_by_a1_lamp": int(seen),
            "example_record_id": rep,
            "ids_by_split": ids_by_split,
        })
    print(f"  -> {len(users):,} unique users discovered")
    return users


def write_csv(task: str, users: list) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{task}_users.csv"
    fields = ["user_fingerprint", "n_train", "n_dev", "n_test",
              "seen_by_a1_lamp", "example_record_id"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        # Sorting unseen-first, then by n_train desc, gives a useful manual-inspection order.
        for u in sorted(users, key=lambda x: (x["seen_by_a1_lamp"], -x["n_train"], -x["n_dev"])):
            w.writerow(u)
    print(f"  wrote {out} ({len(users)} rows)")
    return out


def write_records_json(task: str, users: list) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{task}_user_records.json"
    payload = {u["user_fingerprint"]: u["ids_by_split"] for u in users}
    with open(out, "w") as f:
        json.dump(payload, f)
    print(f"  wrote {out} ({len(payload)} users)")
    return out


def quantile(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(q * (len(xs) - 1))))
    return xs[k]


def summarise(task: str, users: list) -> None:
    seen_users = [u for u in users if u["seen_by_a1_lamp"]]
    unseen_users = [u for u in users if not u["seen_by_a1_lamp"]]
    print()
    print(f"==== {task} per-user distribution ====")
    print(f"  total users discovered:  {len(users):>7,}")
    print(f"    seen by A1-lamp:       {len(seen_users):>7,}  "
          f"({100*len(seen_users)/max(1,len(users)):5.1f}%)")
    print(f"    NOT seen by A1-lamp:   {len(unseen_users):>7,}  "
          f"({100*len(unseen_users)/max(1,len(users)):5.1f}%)")

    for label, pool in [("UNSEEN users", unseen_users), ("all users", users)]:
        if not pool:
            continue
        print(f"\n  ---- {label}: per-user record counts ----")
        for col in ("n_train", "n_dev", "n_test"):
            xs = [u[col] for u in pool]
            print(f"    {col}: mean {sum(xs)/len(xs):5.1f}  "
                  f"p25 {quantile(xs, .25):>4d}  med {quantile(xs, .5):>4d}  "
                  f"p75 {quantile(xs, .75):>4d}  p90 {quantile(xs, .9):>4d}  "
                  f"max {max(xs):>4d}")

        # Joint cutoff table — how many users meet (n_train >= N, n_dev >= M)
        n_cuts = [1, 4, 8, 16, 32, 64, 128, 256]
        m_cuts = [1, 4, 8, 16]
        print(f"\n  ---- {label}: # users with n_train >= N AND n_dev >= M ----")
        print("    n_train\\n_dev " + "".join(f"{m:>7d}" for m in m_cuts))
        for n in n_cuts:
            row = [sum(1 for u in pool if u["n_train"] >= n and u["n_dev"] >= m) for m in m_cuts]
            print(f"    >={n:6d}        " + "".join(f"{x:>7d}" for x in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS,
                        help="Subset of tasks to analyse. Defaults to all 3.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if the per-task CSV already exists.")
    args = parser.parse_args()

    for task in args.tasks:
        out_csv = OUT_DIR / f"{task}_users.csv"
        out_json = OUT_DIR / f"{task}_user_records.json"
        if out_csv.exists() and out_json.exists() and not args.overwrite:
            print(f"{out_csv} and {out_json} exist, skipping (use --overwrite to redo)")
            continue
        print(f"\n############ {task} ############")
        t0 = time.time()
        seen = build_seen_set(task)
        users = analyse_task(task, seen)
        del seen
        write_csv(task, users)
        write_records_json(task, users)
        summarise(task, users)
        print(f"  ({task} took {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
