#!/usr/bin/env python3
"""
One-off diagnostic (not part of the pipeline): tokenize every example in all
300 LL4-LL6 per-user training corpora with the real SmolLM3 tokenizer +
chat template (same method as train/train.py's build_example), report the
token-length distribution per task. Used to decide whether the OPPU config
templates' placeholder `max_seq_length=1024` needs correction before the
real per-user training runs, per plan decision #5's "smoke-test before
locking in" requirement (experiments/2026-07-27-longlamp-user-lora-ll4-ll6-plan.md).

Usage (needs the training Docker image for `transformers`):
    python train/check_longlamp_user_seqlen.py
"""

import glob
import json
from pathlib import Path

MODEL_DIR = "/home/ange00008/projects/mobileFT_distill/data/models/SmolLM3-3B"
DATA_DIR = Path("/home/ange00008/projects/mobileFT_distill/data")


def _flatten(x):
    """Duplicated from train/train.py's build_example -- Condor's sandbox
    only transfers this script itself, not sibling files, so this stays
    standalone (same rationale as every other duplicated-logic script in
    this project)."""
    return x[0] if (x and isinstance(x[0], list)) else x


def build_example(record, tokenizer):
    """Token length only -- duplicated (not imported) from
    train/train.py's build_example, minus the truncation/labels logic this
    diagnostic doesn't need. Keep in sync if that function's chat-template
    call ever changes."""
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages.append({"role": "user", "content": record["user"]})
    messages.append({"role": "assistant", "content": record["assistant"]})
    try:
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True, enable_thinking=False,
        )
    except TypeError:
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True,
        )
    return len(_flatten(out["input_ids"]))


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    for tag in ("review", "abstract", "topic"):
        lengths = []
        files = sorted(glob.glob(str(DATA_DIR / f"longlamp_user_train_{tag}_*_bm25k4.jsonl")))
        print(f"[{tag}] {len(files)} files", flush=True)
        for fp in files:
            with open(fp) as f:
                for line in f:
                    rec = json.loads(line)
                    lengths.append(build_example(rec, tokenizer))
        lengths.sort()
        n = len(lengths)
        if n == 0:
            print(f"[{tag}] NO EXAMPLES FOUND", flush=True)
            continue

        def pct(p):
            return lengths[min(n - 1, int(p * n))]

        print(
            f"[{tag}] n={n} min={lengths[0]} p50={pct(0.5)} p90={pct(0.9)} "
            f"p95={pct(0.95)} p99={pct(0.99)} max={lengths[-1]}",
            flush=True,
        )
        for cap in (512, 1024, 2048, 4096):
            over = sum(1 for x in lengths if x > cap)
            print(f"    at cap={cap}: {over}/{n} ({100*over/n:.2f}%) would be truncated",
                  flush=True)


if __name__ == "__main__":
    main()
