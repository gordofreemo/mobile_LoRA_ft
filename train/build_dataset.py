#!/usr/bin/env python3
"""
Build LaMP training corpora for Task-LoRA training.

Streams the raw LaMP train splits, does BM25 top-k retrieval on each user's
profile, and writes compact JSONL files ready for HF Trainer to consume. The
retrieved-and-formatted system prompt replaces ~100 raw profile entries per
example, so the on-disk training corpus shrinks from ~3.7 GB total to ~150 MB.

Output files:
    data/lamp_train_LaMP_3_bm25k4.jsonl   per-task, in raw order
    data/lamp_train_LaMP_4_bm25k4.jsonl
    data/lamp_train_LaMP_7_bm25k4.jsonl
    data/lamp_train_mixed_bm25k4.jsonl    interleaved + deterministically shuffled
    data/lamp_train_mixed_bm25k4.meta.json  provenance sidecar

Each line is one training example:
    {"task": "LaMP_3", "id": "201",
     "system": "<profile context block>",
     "user":   "<LaMP input — instruction + new item>",
     "assistant": "<gold output>"}

No chat template applied yet — that happens in train/train.py at load time so
the corpus is tokenizer-agnostic.

CRITICAL: the BM25 + per-task formatting + role layout here MUST match
eval/eval_lamp.py exactly. Train and eval need to see the same prompt shape, or
adapter scores tank for harness reasons. The relevant blocks (TASKS,
SYSTEM_PREAMBLE, BM25, tokenize, retrieve_profile) are duplicated from
eval/eval_lamp.py rather than imported so this script can be transferred to a
Condor sandbox without import-path gymnastics; the duplication is intentional.
If you change retrieval here, change it in eval_lamp.py too (and vice versa).

LaMP-6 is intentionally absent: same Avocado licensing issue as the eval side.

Usage (run as a CPU Condor job — login node OOMs on the 2.9 GB LaMP-3 read):
    python train/build_dataset.py --k 4 --seed 0
    python train/build_dataset.py --tasks LaMP_3 --limit 100   # smoke
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

LAMP_DIR = os.environ.get(
    "LAMP_DIR",
    "/home/ange00008/projects/mobileFT_distill/data/lamp",
)
DATA_OUT_DIR = os.environ.get(
    "DATA_OUT_DIR",
    "/home/ange00008/projects/mobileFT_distill/data",
)
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/home/ange00008/projects/mobileFT_distill",
)

ENTRY_CHARS = 600
TITLE_CHARS = 200

# Hard ceiling on the streaming JSON parser's rolling buffer. Any single
# top-level record is well under 50 MB in our LaMP data; anything past this
# means raw_decode is in a non-converging retry loop (malformed object, or
# an env-specific decode failure) and we want a fast crash, not a silent
# multi-hour O(M^2) hang.
MAX_PARSER_BUF_BYTES = 64 * 1024 * 1024


def trim(text: str, n: int = ENTRY_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


# --- Per-task config (MUST stay in sync with eval/eval_lamp.py) --------------
TASKS = {
    "LaMP_3": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- Review: "{trim(it.get("text", ""))}" — the user rated it {it.get("score", "?")}/5',
    },
    "LaMP_4": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- Article: "{trim(it.get("text", ""))}" — the user\'s headline: "{trim(it.get("title", ""), TITLE_CHARS)}"',
    },
    "LaMP_7": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- "{trim(it.get("text", ""))}"',
    },
}

SYSTEM_PREAMBLE = (
    "The following are examples of this user's past activity. "
    "Use them to match this user's preferences and writing style.\n\n"
)


# --- BM25 (identical algorithm to eval/eval_lamp.py) -------------------------
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return _WORD.findall(str(text).lower())


class BM25:
    def __init__(self, docs_tokens: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.doc_freqs = [Counter(d) for d in docs_tokens]
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        df = Counter()
        for d in docs_tokens:
            for w in set(d):
                df[w] += 1
        self.idf = {
            w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()
        }

    def top_k(self, query_tokens: list, k: int) -> list:
        if self.N == 0:
            return []
        scored = []
        for i in range(self.N):
            freqs, dl = self.doc_freqs[i], self.doc_len[i]
            s = 0.0
            for w in query_tokens:
                tf = freqs.get(w)
                if not tf:
                    continue
                idf = self.idf.get(w, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (tf * (self.k1 + 1)) / denom
            scored.append((s, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scored[:k]]


def retrieve_profile(task: str, query: str, profile: list, k: int) -> list:
    if not profile or k <= 0:
        return []
    index_field = TASKS[task]["index_field"]
    docs = [tokenize(index_field(it)) for it in profile]
    bm25 = BM25(docs)
    idxs = bm25.top_k(tokenize(query), k)
    return [profile[i] for i in idxs]


# --- Streaming JSON parser (no ijson dep) ------------------------------------
def stream_json_array(path):
    """Yield each top-level object from a JSON-array file, without loading it
    all into memory.

    The LaMP train files are JSON arrays (`[ {...}, {...}, ... ]`) up to 2.9 GB,
    so `json.load` OOMs even on cluster nodes with modest memory. We use
    `json.JSONDecoder.raw_decode` to peel off one object at a time from a
    rolling buffer, refilling from disk when the decoder needs more bytes.
    Pure stdlib — no extra image deps.

    Subtlety: raw_decode does NOT skip leading whitespace and requires the
    first character to be a JSON value-starter (`{`, `[`, `"`, digit, ...).
    If a chunk-read boundary lands in the middle of the inter-record `, `
    separator, buf can end up starting with ' ' or ',' and raw_decode then
    fails forever at char 0. So we strip whitespace + optional separator
    comma at the top of EVERY iteration, including after a chunk refill.
    """
    decoder = json.JSONDecoder()
    # Pin encoding to UTF-8 so the parser sees the same characters
    # regardless of the host's locale-default encoding.
    with open(path, "r", encoding="utf-8") as f:
        # Skip leading whitespace + the '[' that opens the array.
        buf = ""
        while "[" not in buf:
            chunk = f.read(65536)
            if not chunk:
                return
            buf += chunk
        buf = buf[buf.index("[") + 1 :]

        while True:
            # Strip leading whitespace + optional separator comma. Must run
            # on every iteration, not just outer-loop entries, because chunk
            # reads can leave buf starting with whitespace/comma.
            buf = buf.lstrip()
            if buf.startswith(","):
                buf = buf[1:].lstrip()
            if buf.startswith("]"):
                return
            if not buf:
                chunk = f.read(65536)
                if not chunk:
                    return
                buf += chunk
                continue

            try:
                obj, idx = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = f.read(65536)
                if not chunk:
                    # Trailing whitespace or stray separator is fine; anything
                    # else means the file was truncated mid-object.
                    if buf.strip(" \t\r\n,]"):
                        raise
                    return
                buf += chunk
                if len(buf) > MAX_PARSER_BUF_BYTES:
                    raise RuntimeError(
                        f"stream_json_array: buf grew past "
                        f"{MAX_PARSER_BUF_BYTES:,} chars without a successful "
                        f"decode at file offset {f.tell():,}. Likely a malformed "
                        f"object. "
                        f"buf_head={buf[:200]!r} buf_tail={buf[-200:]!r}"
                    )
                continue

            yield obj
            buf = buf[idx:]


# --- Message building --------------------------------------------------------
def build_example_strings(task: str, question: dict, gold: str, k: int) -> dict:
    """Build the (system, user, assistant) triple for one LaMP example.

    Role layout matches eval/eval_lamp.py: system carries the retrieved profile
    context (or is empty when the profile is empty / k<=0), user carries the
    LaMP `input` field (instruction + the new item), assistant is the gold.
    """
    user_content = question["input"]
    profile = question.get("profile", [])
    retrieved = retrieve_profile(task, user_content, profile, k)
    if retrieved:
        lines = "\n".join(TASKS[task]["format"](it) for it in retrieved)
        system = SYSTEM_PREAMBLE + lines
    else:
        # No profile — train sees the same "no system message" prompt eval
        # uses for its --no-profile floor. Rare in LaMP train but possible.
        system = ""
    return {
        "task": task,
        "id": question["id"],
        "system": system,
        "user": user_content,
        "assistant": str(gold),
    }


# --- Provenance --------------------------------------------------------------
def collect_provenance() -> dict:
    import datetime
    import platform
    import socket
    import subprocess

    def _git(*a):
        try:
            return (
                subprocess.check_output(
                    ["git", *a], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            return None

    porcelain = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "hostname": socket.gethostname(),
        "condor_cluster_id": os.environ.get("CONDOR_CLUSTER_ID") or None,
        "condor_proc_id": os.environ.get("CONDOR_PROC_ID") or None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if porcelain is None else bool(porcelain),
        "python_version": platform.python_version(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default="LaMP_3,LaMP_4,LaMP_7",
        help="comma-separated LaMP tasks to process",
    )
    parser.add_argument(
        "--k", type=int, default=4, help="BM25 profile entries per example"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the mixed-file shuffle (deterministic)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap examples per task (smoke testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing JSONL / meta files (default: refuse)",
    )
    args = parser.parse_args()

    provenance = collect_provenance()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    suffix = f"bm25k{args.k}"
    per_task_paths = {
        t: Path(DATA_OUT_DIR) / f"lamp_train_{t}_{suffix}.jsonl" for t in tasks
    }
    mixed_path = Path(DATA_OUT_DIR) / f"lamp_train_mixed_{suffix}.jsonl"
    meta_path = Path(DATA_OUT_DIR) / f"lamp_train_mixed_{suffix}.meta.json"

    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] build_dataset tasks=[{','.join(tasks)}] k={args.k} seed={args.seed} "
        f"limit={args.limit} commit={commit_short} "
        f"condor={provenance.get('condor_cluster_id') or '-'}."
        f"{provenance.get('condor_proc_id') or '-'} "
        f"host={provenance.get('hostname')}",
        flush=True,
    )

    # Refuse-to-overwrite — preprocessing output is the input to every training
    # run, so silently clobbering it would invalidate every downstream
    # checkpoint and result.
    existing = [
        p
        for p in (list(per_task_paths.values()) + [mixed_path, meta_path])
        if p.exists()
    ]
    if existing and not args.overwrite:
        print("ERROR: refusing to overwrite existing files:", file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nPass --overwrite to replace, or change --k / --seed to write a new path.",
            file=sys.stderr,
        )
        sys.exit(1)

    Path(DATA_OUT_DIR).mkdir(parents=True, exist_ok=True)

    # --- Per-task pass: stream raw JSON → write compact JSONL ----------------
    per_task_counts = {}
    per_task_skipped = {}
    t0 = time.time()
    for task in tasks:
        q_path = Path(LAMP_DIR) / task / "train_questions.json"
        o_path = Path(LAMP_DIR) / task / "train_outputs.json"
        if not q_path.exists() or not o_path.exists():
            print(f"  SKIP {task}: missing data files", file=sys.stderr)
            per_task_counts[task] = 0
            per_task_skipped[task] = 0
            continue

        # Outputs are small (<5 MB) — load whole, index by id.
        golds = {g["id"]: g["output"] for g in json.load(o_path.open())["golds"]}

        n_written, n_skipped = 0, 0
        t_task = time.time()
        with per_task_paths[task].open("w") as out:
            for q in stream_json_array(q_path):
                qid = q.get("id")
                gold = golds.get(qid)
                if gold is None:
                    n_skipped += 1
                    continue
                ex = build_example_strings(task, q, gold, args.k)
                out.write(json.dumps(ex) + "\n")
                n_written += 1
                if args.limit > 0 and n_written >= args.limit:
                    break
                if n_written % 2000 == 0:
                    rate = n_written / max(time.time() - t_task, 1e-6)
                    print(
                        f"  {task} {n_written} ({rate:.0f}/s)",
                        flush=True,
                    )
        per_task_counts[task] = n_written
        per_task_skipped[task] = n_skipped
        elapsed_task = time.time() - t_task
        size_mb = per_task_paths[task].stat().st_size / 1e6
        print(
            f"{task}: {n_written} examples, {n_skipped} skipped "
            f"({elapsed_task:.0f}s, {size_mb:.1f} MB) -> {per_task_paths[task]}"
        )

    # --- Mix + shuffle pass --------------------------------------------------
    # We read all per-task JSONLs into memory (~150 MB total) for the shuffle.
    # That's well within the Condor job's RAM budget and avoids needing an
    # external sort. Deterministic via the fixed seed.
    rng = random.Random(args.seed)
    mixed_lines = []
    for task in tasks:
        if per_task_counts.get(task, 0) == 0:
            continue
        with per_task_paths[task].open() as f:
            mixed_lines.extend(f.readlines())
    rng.shuffle(mixed_lines)
    with mixed_path.open("w") as out:
        for line in mixed_lines:
            out.write(line)
    mixed_size_mb = mixed_path.stat().st_size / 1e6
    print(
        f"mixed: {len(mixed_lines)} examples ({mixed_size_mb:.1f} MB) -> {mixed_path}"
    )

    # --- Meta sidecar --------------------------------------------------------
    meta = {
        "schema_version": 1,
        "tasks": tasks,
        "k": args.k,
        "retriever": "bm25",
        "seed": args.seed,
        "limit": args.limit,
        "per_task_counts": per_task_counts,
        "per_task_skipped": per_task_skipped,
        "total_count": len(mixed_lines),
        "per_task_files": {t: str(per_task_paths[t]) for t in tasks},
        "mixed_file": str(mixed_path),
        "command": "python " + " ".join(sys.argv),
        **provenance,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"meta:  {meta_path}")
    print(f"total: {len(mixed_lines)} examples in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
