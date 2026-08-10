#!/usr/bin/env python3
"""Pre-tokenize a task corpus for on-device Task-LoRA training (h12).

Renders each {system, user, assistant} record with the HF chat template
exactly as train/train.py's build_example does (same tokenizer, same
assistant-tokens mask), and emits one JSON line per example:

    {"id": ..., "input_ids": [...], "loss_start": int, "loss_end": int}

where [loss_start, loss_end) is the assistant-supervised span in
input_ids. The device never tokenizes — this removes any swift-transformers
/ HF tokenizer-parity risk and sidesteps the documented-broken
prompt-prefix masking for SmolLM3 (see train.py's build_example docstring).

The output order is shuffled ONCE here with --shuffle-seed (default 0) and
consumed sequentially by both the device harness and the Mac control, so
the two arms see an identical data order and their loss curves overlay
directly. This matches HF Trainer's single-epoch seeded-shuffle semantics
in kind.

Every example's assistant mask is verified to be a single contiguous span;
a handful of examples are round-trip-decoded and cross-checked against
train.py's build_example labels.
"""

import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))
sys.path.insert(0, str(PROJECT_ROOT / "train"))


def git_info():
    def run(args):
        try:
            return subprocess.check_output(args, cwd=PROJECT_ROOT, text=True).strip()
        except Exception:
            return None
    commit = run(["git", "rev-parse", "--short", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    return commit, bool(dirty)


def mask_span(assistant_mask):
    """Return (start, end) of the single contiguous 1-span, or raise."""
    start = end = None
    for i, m in enumerate(assistant_mask):
        if m and start is None:
            start = i
        if m:
            if end is not None and i != end:
                raise ValueError("assistant mask is not contiguous")
            end = i + 1
    if start is None:
        raise ValueError("assistant mask is empty")
    return start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(PROJECT_ROOT / "data/lamp_train_LaMP_7_bm25k4.jsonl"))
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "data/models/SmolLM3-3B-mlx-4bit"),
                    help="Tokenizer dir — verified byte-identical to the cluster's data/models/SmolLM3-3B")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data/ondevice_task_data/lamp7_task.jsonl"))
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only N examples (adds _limitN suffix)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.limit > 0:
        out_path = out_path.with_name(out_path.stem + f"_limit{args.limit}" + out_path.suffix)
    meta_path = out_path.with_suffix(".meta.json")

    commit, dirty = git_info()
    print(f"[build_task_device_data] corpus={args.corpus} tokenizer={args.tokenizer} "
          f"out={out_path} max_length={args.max_length} shuffle_seed={args.shuffle_seed} "
          f"limit={args.limit} commit={commit} dirty={dirty} host={socket.gethostname()}")

    if out_path.exists() and not args.overwrite:
        print(f"REFUSING to overwrite {out_path} (pass --overwrite)", file=sys.stderr)
        sys.exit(1)

    from transformers import AutoTokenizer

    # Extract build_example + _flatten verbatim from train/train.py without
    # importing it (train.py's top-level imports need torch/datasets, absent
    # in the mlx venv). This keeps the cross-check against the literal
    # canonical source, not a copy that could drift.
    import ast
    train_src = (PROJECT_ROOT / "train/train.py").read_text()
    tree = ast.parse(train_src)
    wanted = {n for n in ("build_example", "_flatten")}
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "train/train.py", "exec"), ns)
    build_example = ns["build_example"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    records = [json.loads(l) for l in open(args.corpus)]
    n_corpus = len(records)
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(records)
    if args.limit > 0:
        records = records[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tok_total = 0
    max_len_seen = 0
    n_truncated = 0
    span_tok_total = 0
    lines = []
    for rec in records:
        messages = []
        if rec.get("system"):
            messages.append({"role": "system", "content": rec["system"]})
        messages.append({"role": "user", "content": rec["user"]})
        messages.append({"role": "assistant", "content": rec["assistant"]})
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True, enable_thinking=False,
        )
        input_ids = out["input_ids"]
        assistant_mask = out["assistant_masks"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids, assistant_mask = input_ids[0], assistant_mask[0]
        input_ids = list(input_ids)
        assistant_mask = list(assistant_mask)
        if len(input_ids) > args.max_length:
            n_truncated += 1
            input_ids = input_ids[: args.max_length]
            assistant_mask = assistant_mask[: args.max_length]
        start, end = mask_span(assistant_mask)
        n_tok_total += len(input_ids)
        span_tok_total += end - start
        max_len_seen = max(max_len_seen, len(input_ids))
        lines.append(json.dumps(
            {"id": rec["id"], "input_ids": input_ids, "loss_start": start, "loss_end": end},
            separators=(",", ":")))

    # --- verification against train.py's build_example -----------------------
    check_idx = [0, len(records) // 2, len(records) - 1]
    for i in check_idx:
        rec = records[i]
        ref = build_example(rec, tokenizer, args.max_length)
        mine = json.loads(lines[i])
        assert mine["input_ids"] == ref["input_ids"], f"input_ids mismatch on example {i}"
        ref_labels = ref["labels"]
        derived = [tok if mine["loss_start"] <= j < mine["loss_end"] else -100
                   for j, tok in enumerate(mine["input_ids"])]
        assert derived == ref_labels, f"loss span mismatch on example {i}"
        span_text = tokenizer.decode(mine["input_ids"][mine["loss_start"]:mine["loss_end"]])
        print(f"[verify] example {i} id={rec['id']} len={len(mine['input_ids'])} "
              f"span=[{mine['loss_start']},{mine['loss_end']}) OK; span decodes to: "
              f"{span_text[:120]!r}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    meta = {
        "corpus": str(args.corpus),
        "corpus_md5": hashlib.md5(open(args.corpus, "rb").read()).hexdigest(),
        "corpus_n": n_corpus,
        "tokenizer": str(args.tokenizer),
        "n_examples": len(records),
        "max_length": args.max_length,
        "n_truncated": n_truncated,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_note": "order is baked in; device and Mac control consume sequentially",
        "total_tokens": n_tok_total,
        "mean_tokens": round(n_tok_total / len(records), 1),
        "max_tokens": max_len_seen,
        "assistant_span_tokens_total": span_tok_total,
        "assistant_span_frac": round(span_tok_total / n_tok_total, 4),
        "git_commit": commit,
        "git_dirty": dirty,
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "transformers_version": __import__("transformers").__version__,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {len(records)} examples -> {out_path}")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("corpus_md5",)}, indent=2))


if __name__ == "__main__":
    main()
