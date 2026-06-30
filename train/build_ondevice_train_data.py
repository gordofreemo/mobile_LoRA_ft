#!/usr/bin/env python3
"""
Prepare the bundled on-device LoRA-training corpus for the Phase-3 naive
training benchmark (see experiments/2026-06-29-ondevice-training-naive-plan.md).

What this produces
------------------
Two JSONL files, each line `{"text": "<chat-template-rendered string>"}`, the
exact format the MLX-Swift `LoRABatchIterator` / `loadLoRAData` expect on device:

    ios/mlx-swift-examples/Applications/LLMEval/Data/lora_train.jsonl  (first N_TRAIN)
    ios/mlx-swift-examples/Applications/LLMEval/Data/lora_valid.jsonl  (first N_VALID)

The `text` is one full (system?, user, assistant) conversation rendered through
SmolLM3's chat template with `enable_thinking=False` — byte-identical to the
render train/train.py uses (build_example), minus the assistant-token masking
(on-device naive training supervises the whole sequence; LoRATrain.loss masks
only padding). The device re-tokenises the string with its own tokenizer.

Deviation from the locked plan (and why)
---------------------------------------
The plan's §"Data prep script" had this Mac-side script do BM25 retrieval over
the user's `lamp_time` partition directly. Instead, BM25 retrieval is delegated
to the canonical cluster builder `train/build_user_dataset.py`
(`--task LaMP_3 --user <u> --bm25-k 4`), whose output JSONL
(`{system,user,assistant}` rows) is the INPUT here. Rationale:
  * That builder is the *exact* audited retrieval used for the R5 LaMP-3
    User-LoRAs — reusing it guarantees the on-device training distribution
    byte-matches the cluster recipe (the cardinal train/eval-consistency rule).
  * The `lamp_time/LaMP_3/train_questions.json` source file is 1.7 GB; running
    retrieval on the cluster and transferring the ~2 MB per-user result avoids
    moving it to the Mac.
So this script's job is narrowed to: pick/verify the median-profile user,
render the chat template, truncate, and write the bundle files.

Median-user selection
---------------------
The user with the median `profile_size` in
`data/lamp_user_stats/LaMP_3_top100_users.json` (the R5 top-100 list) — not too
small, not too large, representative of the per-user deployment workload. With
100 users the median index (50, 0-based, after ascending sort) is used. The
selected fingerprint is asserted against the input JSONL's filename so a stale
input file can't silently slip through.

Usage
-----
    .venv-mlx/bin/python train/build_ondevice_train_data.py \
        --user-jsonl data/lamp_user_train_LaMP_3_u00016746_bm25k4.jsonl
"""

import argparse
import datetime
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOP100_JSON = PROJECT_ROOT / "data" / "lamp_user_stats" / "LaMP_3_top100_users.json"
TOKENIZER_DIR = PROJECT_ROOT / "data" / "models" / "SmolLM3-3B-mlx-4bit"
BUNDLE_DATA_DIR = (
    PROJECT_ROOT / "ios" / "mlx-swift-examples" / "Applications" / "LLMEval" / "Data"
)

N_TRAIN = 50
N_VALID = 5


def median_user(top100_path: Path) -> dict:
    """Return the {user_fingerprint, profile_size, ...} dict at the median of
    the ascending profile_size ordering (index n//2, matching the plan)."""
    doc = json.loads(top100_path.read_text())
    users = doc["users"]
    ordered = sorted(users, key=lambda u: u["profile_size"])
    return ordered[len(ordered) // 2]


def render(messages: list, tokenizer) -> str:
    """Render one conversation to a string via SmolLM3's chat template,
    thinking off. Mirrors train/train.py build_example (tokenize=False here so
    the device re-tokenises). Falls back if enable_thinking is unsupported."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)


def to_messages(record: dict) -> list:
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages.append({"role": "user", "content": record["user"]})
    messages.append({"role": "assistant", "content": record["assistant"]})
    return messages


def git_provenance() -> dict:
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
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if porcelain is None else bool(porcelain),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "python_version": platform.python_version(),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--user-jsonl",
        required=True,
        help="per-user {system,user,assistant} JSONL from build_user_dataset.py "
        "(e.g. data/lamp_user_train_LaMP_3_u00016746_bm25k4.jsonl)",
    )
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-valid", type=int, default=N_VALID)
    ap.add_argument("--tokenizer", default=str(TOKENIZER_DIR))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.user_jsonl)
    if not in_path.exists():
        sys.exit(f"ERROR: input JSONL not found: {in_path}")

    # --- median-user provenance + cross-check against the input file ---------
    med = median_user(TOP100_JSON)
    med_fp = med["user_fingerprint"]
    m = re.search(r"_(u\d+)_", in_path.name)
    file_fp = m.group(1) if m else None
    if file_fp != med_fp:
        sys.exit(
            f"ERROR: input file user '{file_fp}' != median-profile user "
            f"'{med_fp}' (profile_size={med['profile_size']}). Rebuild the "
            f"per-user JSONL for the median user, or pass the matching file."
        )

    out_train = BUNDLE_DATA_DIR / "lora_train.jsonl"
    out_valid = BUNDLE_DATA_DIR / "lora_valid.jsonl"
    existing = [p for p in (out_train, out_valid) if p.exists()]
    if existing and not args.overwrite:
        print("ERROR: refusing to overwrite:", file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        print("Pass --overwrite to replace.", file=sys.stderr)
        sys.exit(1)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    n_take = args.n_train
    if len(rows) < n_take:
        sys.exit(f"ERROR: only {len(rows)} rows in input, need {n_take}.")

    BUNDLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    token_lengths = []
    with out_train.open("w") as f:
        for r in rows[:n_take]:
            text = render(to_messages(r), tok)
            token_lengths.append(len(tok.encode(text)))
            f.write(json.dumps({"text": text}) + "\n")

    with out_valid.open("w") as f:
        for r in rows[: args.n_valid]:
            text = render(to_messages(r), tok)
            f.write(json.dumps({"text": text}) + "\n")

    prov = git_provenance()
    n_over_2048 = sum(1 for n in token_lengths if n > 2048)
    print(
        f"[ondevice-train-data] user={med_fp} profile_size={med['profile_size']} "
        f"(median of top-100) commit={(prov['git_commit'] or 'unknown')[:8]} "
        f"dirty={prov['git_dirty']}"
    )
    print(f"  input          : {in_path} ({len(rows)} rows available)")
    print(f"  train          : {out_train} ({n_take} examples)")
    print(f"  valid (stub)   : {out_valid} ({args.n_valid} examples)")
    print(
        f"  token lengths  : min={min(token_lengths)} "
        f"median={sorted(token_lengths)[len(token_lengths)//2]} "
        f"max={max(token_lengths)}  (>2048: {n_over_2048})"
    )
    if n_over_2048:
        print(
            "  WARNING: some examples exceed 2048 tokens — LoRABatchIterator "
            "will warn and memory cost rises.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
