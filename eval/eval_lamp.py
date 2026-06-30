#!/usr/bin/env python3
"""
LaMP evaluation harness — baseline zero-shot and (later) LoRA adapters.

Runs SmolLM3-3B over a LaMP split, conditions on the user's profile via BM25
retrieval into the system prompt, generates deterministically, parses the
output, and reports quality metrics plus efficiency proxies.

Personalization mechanism is **retrieval** (the standard LaMP protocol), not
summarization — see notebooks/lamp_evaluation_approach.md for why.

Supported tasks: LaMP_3 (rating, Accuracy), LaMP_4 / LaMP_7 (generation, ROUGE-1).
LaMP_6 is intentionally unsupported: its public release ships only Avocado file-id
placeholders (no text), so it can't be scored without the licensed corpus.

Usage (run on the cluster, inside the Docker image):
    python eval/eval_lamp.py --task LaMP_3 --split dev --k 4 --seed 0
    python eval/eval_lamp.py --task LaMP_4 --split dev --no-profile        # floor
    python eval/eval_lamp.py --task LaMP_7 --split dev --limit 5           # smoke test
    python eval/eval_lamp.py --task LaMP_3 --split test --adapter /path/to/lora

This script must run inside the training Docker image — torch/transformers/peft
and rouge_score live there, not on the login node. BM25 is implemented in pure
Python so no extra dependency is required.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

MODEL_DIR = os.environ.get(
    "MODEL_OUT_DIR",
    "/home/ange00008/projects/mobileFT_distill/data/models/SmolLM3-3B",
)
LAMP_DIR = os.environ.get(
    "LAMP_DIR",
    "/home/ange00008/projects/mobileFT_distill/data/lamp",
)
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "/home/ange00008/projects/mobileFT_distill/results",
)
USER_STATS_DIR = os.environ.get(
    "USER_STATS_DIR",
    "/home/ange00008/projects/mobileFT_distill/data/lamp_user_stats",
)
# Absolute repo path (home-mounted) so git provenance works even when Condor runs a
# copy of this script from the scratch sandbox, where __file__ isn't in the repo.
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/home/ange00008/projects/mobileFT_distill",
)

# How many characters of any single profile entry / title we keep, so a few long
# reviews can't blow the 3B context budget.
ENTRY_CHARS = 600
TITLE_CHARS = 200


def trim(text: str, n: int = ENTRY_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


# --- Per-task configuration --------------------------------------------------
# index_field: the profile-entry text BM25 matches the query against.
# format:      how a retrieved entry is rendered as a context line in the system prompt.
# max_new_tokens / metric: generation budget and scoring method.
TASKS = {
    "LaMP_3": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- Review: "{trim(it.get("text", ""))}" — the user rated it {it.get("score", "?")}/5',
        "max_new_tokens": 8,
        "metric": "rating",
    },
    "LaMP_4": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- Article: "{trim(it.get("text", ""))}" — the user\'s headline: "{trim(it.get("title", ""), TITLE_CHARS)}"',
        "max_new_tokens": 64,
        "metric": "rouge1",
    },
    "LaMP_7": {
        "index_field": lambda it: it.get("text", ""),
        "format": lambda it: f'- "{trim(it.get("text", ""))}"',
        "max_new_tokens": 64,
        "metric": "rouge1",
    },
}

SYSTEM_PREAMBLE = (
    "The following are examples of this user's past activity. "
    "Use them to match this user's preferences and writing style.\n\n"
)


# --- BM25 (pure Python, no dependency) ---------------------------------------
# BM25 is the standard term-matching retriever used by the LaMP benchmark. Given a
# query and a set of documents (here: one user's profile entries), it ranks the
# documents by lexical relevance. We implement it ourselves in ~30 lines because
# `rank_bm25` is NOT in the Docker image's requirements, and adding it would mean
# rebuilding/republishing the image. A user's profile is tens-to-hundreds of short
# entries, so a naive pure-Python scan is plenty fast.
#
# The scoring intuition: a document scores high when it shares *rare* words with the
# query (idf), the shared words appear *often* in the document (tf), and the document
# isn't artificially long (length normalization). See the walkthrough notebook for the
# formula broken down.

# Lowercase and split on runs of letters/digits. Crude but deterministic and
# dependency-free — good enough for term matching over short profile texts.
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return _WORD.findall(str(text).lower())


class BM25:
    def __init__(self, docs_tokens: list, k1: float = 1.5, b: float = 0.75):
        # k1 controls how quickly extra occurrences of a word stop helping
        # (term-frequency saturation); b controls how much we penalize long
        # documents. 1.5 / 0.75 are the standard textbook defaults.
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        # Per-document word counts and lengths, precomputed once so scoring a query
        # is just dictionary lookups.
        self.doc_freqs = [Counter(d) for d in docs_tokens]
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        # Document frequency: how many documents each word appears in (count once
        # per doc via set()), used to compute idf below.
        df = Counter()
        for d in docs_tokens:
            for w in set(d):
                df[w] += 1
        # Inverse document frequency (Robertson-Sparck-Jones form). Rare words get a
        # high weight, ubiquitous words a low one. The "+1" inside log keeps the
        # value non-negative even for words present in every document.
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
                    continue  # word not in this document — contributes nothing
                idf = self.idf.get(w, 0.0)
                # The BM25 term score: idf * saturated, length-normalized tf.
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (tf * (self.k1 + 1)) / denom
            scored.append((s, i))
        # Highest score first; return the document *indices* so the caller can map
        # back to the original profile entries.
        scored.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in scored[:k]]


def retrieve_profile(task: str, query: str, profile: list, k: int) -> list:
    """Return the top-k profile entries most relevant to the query (BM25).

    We build a fresh BM25 index per question because each user has a different
    profile — there's no shared corpus to index once. `query` is the LaMP `input`
    string (instruction + the new item); we match it against each profile entry's
    text field (the past review / article / tweet).
    """
    if not profile or k <= 0:
        return []
    index_field = TASKS[task]["index_field"]
    docs = [tokenize(index_field(it)) for it in profile]
    bm25 = BM25(docs)
    idxs = bm25.top_k(tokenize(query), k)
    return [profile[i] for i in idxs]


# --- Prompt construction -----------------------------------------------------
def build_messages(task: str, question: dict, k: int, no_profile: bool) -> list:
    """Turn a LaMP question into chat-template messages.

    Role layout (must match how training formats LaMP examples):
        system = the retrieved profile context (the personalization signal)
        user   = the LaMP `input` (instruction + the new item)
    For the non-personalized floor (`--no-profile`) we omit the system message
    entirely, so the model sees only the task with no user information.
    """
    user_content = question["input"]
    if no_profile:
        return [{"role": "user", "content": user_content}]
    retrieved = retrieve_profile(task, user_content, question.get("profile", []), k)
    if not retrieved:
        # Some users have an empty profile — fall back to the no-profile prompt
        # rather than emitting an empty system message.
        return [{"role": "user", "content": user_content}]
    lines = "\n".join(TASKS[task]["format"](it) for it in retrieved)
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE + lines},
        {"role": "user", "content": user_content},
    ]


def render_prompt(tokenizer, messages: list) -> str:
    """Apply the chat template; ask the model not to emit reasoning if it can.

    SmolLM3's template emits a <think> block by default; for eval we want the answer
    only, so we pass enable_thinking=False. For Llama-3.1 (and most non-reasoning
    models) the template either ignores the kwarg or raises TypeError — both are
    handled below, so the call path is uniform across SmolLM3 / Llama-3.1. As a
    belt-and-braces measure, clean_output() also strips any <think> block that
    slips through.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_output(text: str) -> str:
    """Strip any reasoning block, then whitespace. Defensive — thinking should be off."""
    return _THINK.sub("", text).strip()


# --- Parsing + metrics -------------------------------------------------------
def parse_rating(text: str):
    """Pull the predicted star rating out of free-text output.

    LaMP-3 asks for "just answer with 1-5", but a model may still wrap it in prose.
    We take the *first* digit in 1-5 that appears. This is the common LaMP-3 parsing
    convention; it can misfire if the model says e.g. "not 1 star, I'd say 4" (it
    would grab the 1), so we also track how often parsing fails / looks suspect via
    parse_fail_rate — a high rate is itself a finding about the bare model.
    """
    m = re.search(r"[1-5]", text)
    return m.group(0) if m else None


def score_rating(preds: list, golds: list) -> dict:
    """Accuracy over all (parse failure = wrong); MAE over parseable predictions.

    LaMP-3 is ordinal (1-5), so besides exact-match Accuracy (the CLAUDE.md metric)
    we also report MAE — "off by one" is better than "off by three", which plain
    accuracy can't see. MAE is averaged only over predictions we could parse to a
    number; accuracy counts a parse failure as wrong (you produced no usable answer).
    """
    correct = 0
    abs_err = []
    parse_fail = 0
    for p, g in zip(preds, golds):
        r = parse_rating(p)
        if r is None:
            parse_fail += 1
            continue
        if r == str(g).strip():
            correct += 1
        abs_err.append(abs(int(r) - int(str(g).strip())))
    n = len(preds)
    return {
        "accuracy": correct / n if n else 0.0,
        "mae": (sum(abs_err) / len(abs_err)) if abs_err else None,
        "parse_fail_rate": parse_fail / n if n else 0.0,
        "n": n,
    }


def score_rouge1(preds: list, golds: list) -> dict:
    """Mean ROUGE-1 F1 — unigram (single-word) overlap between prediction and gold.

    ROUGE-1 F1 balances precision (did the prediction avoid junk words?) and recall
    (did it cover the reference's words?). It's word-order- and synonym-blind, but
    fast and the standard metric for these short LaMP generation tasks (4/6/7).
    rouge_scorer.score(target, prediction) — note the (gold, pred) argument order.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    f1 = [scorer.score(str(g), p)["rouge1"].fmeasure for p, g in zip(preds, golds)]
    return {"rouge1": (sum(f1) / len(f1)) if f1 else 0.0, "n": len(preds)}


# --- Provenance (so every result is reproducible) ----------------------------
def collect_provenance() -> dict:
    """Capture everything needed to reproduce / trust a result later.

    The git commit + dirty flag tell you exactly which version of the code ran;
    the library versions guard against silent metric drift across image rebuilds;
    the hostname records which execute node produced the numbers. CLAUDE.md makes
    reproducibility a hard constraint, so this rides along in every result file.
    """
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

    import torch
    import transformers

    try:
        import peft

        peft_v = peft.__version__
    except Exception:
        peft_v = None

    porcelain = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        # Condor IDs come from env vars set in the sub file's `environment` line.
        # They're what links a result JSON back to its runlog file
        # (runlogs/eval_lamp.<task>.<cluster>.<proc>.{out,err}). None when running
        # interactively (no Condor job around).
        "condor_cluster_id": os.environ.get("CONDOR_CLUSTER_ID") or None,
        "condor_proc_id": os.environ.get("CONDOR_PROC_ID") or None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if porcelain is None else bool(porcelain),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft_v,
    }


# --- Adapter-path tagging ----------------------------------------------------
def derive_adapter_tag(adapter_path: str) -> str:
    """Short identifier for filenames / records.

    `Path(args.adapter).name` alone yields uninformative tags like "final" or
    "checkpoint-500" because every training condition writes those same leaf
    names. When the leaf is generic, prepend the parent dir so results from
    e.g. `train/checkpoints/a1_lamp_seed0/final` get tagged `a1_lamp_seed0_final`.
    """
    if adapter_path.lower() == "none":
        return "base"
    p = Path(adapter_path.rstrip("/"))
    if p.name == "final" or p.name.startswith("checkpoint-"):
        return f"{p.parent.name}_{p.name}"
    return p.name


# --- Data loading ------------------------------------------------------------
def load_split(task: str, split: str):
    """Load a LaMP split. Questions are a list of {id, input, profile}; outputs are
    {task, golds:[{id, output}]}. We index golds by id (a string) so we can look up
    the right answer for each question regardless of ordering.
    """
    base = Path(LAMP_DIR) / task
    questions = json.loads((base / f"{split}_questions.json").read_text())
    outputs = json.loads((base / f"{split}_outputs.json").read_text())
    golds = {g["id"]: g["output"] for g in outputs["golds"]}
    return questions, golds


# --- Model loading -----------------------------------------------------------
def load_model(
    model_dir: str,
    adapter: str | None,
    base_adapter: str | None = None,
    device_map: str = "cuda",
):
    """Load a HuggingFace causal-LM in bf16, optionally with stacked LoRA adapters.

    Used for both SmolLM3-3B (cuda single-device) and Llama-3.1-{8B,70B}-Instruct
    (Llama-70B needs `device_map="auto"` for accelerate-dispatched multi-GPU
    sharding). The model is loaded with the user-supplied device_map verbatim:
    "cuda" / "cpu" → single-device placement; "auto" → accelerate dispatch.

    The two-adapter case (`base_adapter` set) is the User-LoRA stacking pattern:
    Task-LoRA is loaded first, merged into the frozen base, then a fresh
    per-user LoRA is loaded on top. This must match train/train.py's
    base-adapter plumbing exactly so train- and inference-time graphs are
    identical. The intermediate merge is required — PEFT can't stack two
    distinct LoraConfig adapters from disk without it. Note: merge_and_unload()
    is single-device only, so adapters + device_map="auto" is not supported
    (Llama runs are base-model-only, so this combination shouldn't arise).

    `adapter_params` counts only the *outer* (user) LoRA's parameters — the
    base adapter has been merged into the frozen weights and contributes
    nothing to the on-device weight cost at inference. When only the base is
    given (smoke test of the base-adapter path with `--adapter none`), we
    instead count the base's params so the field stays informative.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Resolve the device_map. "cuda" without a GPU silently downgrades to CPU
    # (matches the prior behavior so SmolLM3 CPU smoke runs still work).
    if device_map == "cuda" and not torch.cuda.is_available():
        effective_device_map = "cpu"
    else:
        effective_device_map = device_map
    print(f"Loading model with device_map={effective_device_map!r} (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=effective_device_map
    )

    from peft import PeftModel

    adapter_params = 0
    has_base = bool(base_adapter) and base_adapter.lower() != "none"
    has_user = bool(adapter) and adapter.lower() != "none"

    if has_base:
        print(f"Loading + merging base adapter from {base_adapter}...", flush=True)
        model = PeftModel.from_pretrained(model, base_adapter)
        base_params = sum(
            p.numel() for n, p in model.named_parameters() if "lora_" in n
        )
        model = model.merge_and_unload()
        # If no outer adapter, report the base's param count for context.
        if not has_user:
            adapter_params = base_params

    if has_user:
        print(f"Loading LoRA adapter from {adapter}...", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
        adapter_params = sum(
            p.numel() for n, p in model.named_parameters() if "lora_" in n
        )
        model = model.merge_and_unload()

    model.eval()
    # Device for input placement: returns the first parameter's device. Works
    # for single-device placement (cuda:0 / cpu) and for accelerate dispatch
    # (returns the embedding's device, typically cuda:0).
    device = next(model.parameters()).device
    return tokenizer, model, device, adapter_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--adapter", default="none", help="LoRA checkpoint path, or 'none'")
    parser.add_argument(
        "--base-adapter",
        default="none",
        help="optional pre-existing LoRA adapter to merge into the base before "
        "attaching --adapter on top (User-LoRA stacking pattern). 'none' disables.",
    )
    parser.add_argument(
        "--user-records",
        default=None,
        help="user fingerprint (e.g. u00000011); filter questions to those "
        "records via data/lamp_user_stats/<task>_user_records.json. Default: no filter.",
    )
    parser.add_argument(
        "--user-records-from-file",
        default=None,
        help="path to a top-K-users JSON (e.g. "
        "data/lamp_user_stats/LaMP_3_top100_users.json); filter questions to the "
        "example IDs in users[*].test_record_id. Used by the K=100 cross-model "
        "comparison so a single Llama eval job covers all K subset records at "
        "once. Mutually exclusive with --user-records.",
    )
    parser.add_argument("--k", type=int, default=4, help="BM25 profile entries to retrieve")
    parser.add_argument("--no-profile", action="store_true", help="non-personalized floor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="evaluate only first N (smoke test)")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=0, help="override per-task default")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing results JSON / predictions JSONL (default: refuse)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from an existing predictions JSONL: skip already-completed "
        "ids, append new generations, recompute summary over the union. Required "
        "for preempt-eligible Condor jobs that may be requeued mid-run.",
    )
    parser.add_argument(
        "--device-map",
        default="cuda",
        help="device_map passed to from_pretrained. 'cuda' (default) / 'cpu' for "
        "single-device placement; 'auto' for accelerate multi-GPU sharding (use "
        "for Llama-3.1-70B). 'cuda' silently falls back to 'cpu' if no GPU is "
        "available.",
    )
    args = parser.parse_args()

    if args.resume and args.overwrite:
        print("ERROR: --resume and --overwrite are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if args.user_records and args.user_records_from_file:
        print("ERROR: --user-records and --user-records-from-file are mutually exclusive.",
              file=sys.stderr)
        sys.exit(1)

    # Capture provenance once at startup so the banner and the result record agree
    # and `timestamp_utc` reflects start time. The banner makes the runlog
    # self-documenting: anyone reading runlogs/eval_lamp.<task>.<cluster>.<proc>.out
    # sees task, condition, seed, commit, and Condor IDs on the very first line.
    provenance = collect_provenance()
    cond_label = "noprofile" if args.no_profile else f"bm25k{args.k}"
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] task={args.task} split={args.split} cond={cond_label} "
        f"seed={args.seed} limit={args.limit} device_map={args.device_map} "
        f"resume={args.resume} commit={commit_short} "
        f"condor={provenance.get('condor_cluster_id') or '-'}."
        f"{provenance.get('condor_proc_id') or '-'} "
        f"host={provenance.get('hostname')}",
        flush=True,
    )

    # Compute the output paths up front so we can refuse to overwrite *before*
    # loading the 6 GB model and burning generation cycles. This is the safeguard
    # added after a smoke re-run silently clobbered a full baseline's results.
    # Default is refuse; `--overwrite` is the explicit opt-in.
    adapter_tag = derive_adapter_tag(args.adapter)
    base_adapter_tag = derive_adapter_tag(args.base_adapter)
    has_base = args.base_adapter.lower() != "none"
    has_user = args.adapter.lower() != "none"
    if has_base and has_user:
        stacked_tag = f"{base_adapter_tag}_{adapter_tag}"
    elif has_base:
        # Base-only mode (smoke-test the merge path): the base IS effectively
        # the lone adapter; reuse its tag so the stem describes what ran.
        stacked_tag = base_adapter_tag
    else:
        stacked_tag = adapter_tag
    profile_tag = "noprofile" if args.no_profile else f"bm25k{args.k}"
    # Model tag: empty for the default SmolLM3 dir (preserves all Phase 1
    # result filenames), set to the model dir leaf otherwise. Required as soon
    # as we eval comparator models (Llama-3.1-{8B,70B}-Instruct) — without it,
    # `--adapter none` runs across different bases all collide on the same
    # stem and a `--resume` job silently scores against the wrong model's
    # cached predictions.
    model_dir_name = Path(args.model_dir).name
    default_model_name = Path(MODEL_DIR).name
    model_tag = "" if model_dir_name == default_model_name else f"_{model_dir_name}"
    limit_tag = f"_limit{args.limit}" if args.limit > 0 else ""
    if args.user_records:
        user_tag = f"_user{args.user_records}"
    elif args.user_records_from_file:
        # The K=100 subset is a single shared evaluation set, not a per-user
        # eval, so we tag the result file by the source file's stem (e.g.
        # `LaMP_3_top100_users` → `_topK100`). Falls back to the bare stem if
        # the file doesn't include the K-count tag we expect.
        src_stem = Path(args.user_records_from_file).stem
        # Heuristic: look for `_top<N>_users` pattern and shorten to `_topK<N>`.
        m = re.search(r"_top(\d+)_users$", src_stem)
        user_tag = f"_topK{m.group(1)}" if m else f"_{src_stem}"
    else:
        user_tag = ""
    stem = f"{args.task}_{args.split}_{stacked_tag}{model_tag}_{profile_tag}_seed{args.seed}{user_tag}{limit_tag}"
    out_path = Path(RESULTS_DIR) / f"{stem}.json"
    pred_path = Path(RESULTS_DIR) / f"{stem}.predictions.jsonl"
    if not args.overwrite and not args.resume and (out_path.exists() or pred_path.exists()):
        print(
            "ERROR: refusing to overwrite existing results:",
            file=sys.stderr,
        )
        for p in (out_path, pred_path):
            mark = "exists" if p.exists() else "would-be-new"
            print(f"  {p}   [{mark}]", file=sys.stderr)
        print(
            "\nPass --overwrite to replace, --resume to continue an interrupted run,\n"
            "or vary --seed / --task / --adapter / etc. to write to a different path.\n"
            "(Smoke runs with --limit already get a _limitN suffix and won't collide\n"
            "with full runs.)",
            file=sys.stderr,
        )
        sys.exit(1)

    import random

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not Path(args.model_dir).exists():
        print(f"ERROR: model dir not found: {args.model_dir}", file=sys.stderr)
        print("Run condor/download_model.sub first.", file=sys.stderr)
        sys.exit(1)

    questions, golds = load_split(args.task, args.split)
    n_total_records = len(questions)

    # Optional per-user filter: keep only records belonging to one fingerprinted
    # user. Used by the User-LoRA experiments which evaluate on one user's 21/25
    # records, not the full 1500/1800 split. The fingerprint-to-record-id mapping
    # is precomputed by data/lamp_user_stats.py.
    n_filtered_records = None
    if args.user_records:
        records_json = Path(USER_STATS_DIR) / f"{args.task}_user_records.json"
        if not records_json.exists():
            print(f"ERROR: --user-records set but {records_json} missing — "
                  f"run data/lamp_user_stats.py first.", file=sys.stderr)
            sys.exit(1)
        user_map = json.loads(records_json.read_text())
        if args.user_records not in user_map:
            print(f"ERROR: user {args.user_records!r} not in {records_json}.",
                  file=sys.stderr)
            sys.exit(1)
        wanted_ids = set(user_map[args.user_records].get(args.split, []))
        if not wanted_ids:
            print(f"ERROR: user {args.user_records} has 0 records in split "
                  f"{args.split}.", file=sys.stderr)
            sys.exit(1)
        questions = [q for q in questions if str(q["id"]) in wanted_ids]
        n_filtered_records = len(questions)
        print(f"[filter] user={args.user_records} {args.split}: "
              f"{n_filtered_records}/{n_total_records} records kept",
              flush=True)

    # Multi-user subset filter: read the test_record_id values from a top-K-users
    # JSON. Each user in the file contributes one record to the eval set, so the
    # subset size equals the user count (K=100 for the Llama scale comparison).
    # This is the single-job equivalent of running 100 per-user `--user-records`
    # jobs — appropriate when the model has no per-user adaptation.
    if args.user_records_from_file:
        path = Path(args.user_records_from_file)
        if not path.exists():
            print(f"ERROR: --user-records-from-file path not found: {path}",
                  file=sys.stderr)
            sys.exit(1)
        data = json.loads(path.read_text())
        if "users" not in data or not data["users"]:
            print(f"ERROR: {path} has no 'users' list.", file=sys.stderr)
            sys.exit(1)
        wanted_ids = {str(u["test_record_id"]) for u in data["users"]}
        questions = [q for q in questions if str(q["id"]) in wanted_ids]
        n_filtered_records = len(questions)
        if n_filtered_records == 0:
            # Zero matches almost always means a split-source mismatch: the
            # top-K JSON was built against time-split test ids but eval is
            # loading the user-split (default LAMP_DIR), or vice-versa.
            # Hard-fail rather than produce a vacuous accuracy=0.0 result —
            # the K=100 jobs (2026-06-28) silently emitted four empty result
            # JSONs before this guard was added.
            print(f"ERROR: --user-records-from-file matched 0/{len(wanted_ids)} "
                  f"ids from {path.name} against {args.task}/{args.split}. "
                  f"Split-source mismatch: check the file's "
                  f"`inputs.test_questions` field and set LAMP_DIR to point at "
                  f"the same data directory.", file=sys.stderr)
            sys.exit(1)
        if n_filtered_records != len(wanted_ids):
            # Partial match — still suspicious but not always fatal (e.g.
            # legitimate union of users where some have been dropped).
            print(f"WARN: matched {n_filtered_records}/{len(wanted_ids)} ids from "
                  f"{path.name} against {args.task}/{args.split}; ensure --split "
                  f"matches the file's source split.", file=sys.stderr)
        print(f"[filter] {path.name} {args.split}: "
              f"{n_filtered_records}/{n_total_records} records kept "
              f"(from {len(data['users'])} users)", flush=True)

    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"{args.task}/{args.split}: {len(questions)} examples", flush=True)

    # --- Resume: load cached predictions before loading the model -----------
    # If --resume, read whatever predictions JSONL already exists into a dict
    # so we can skip those ids in the main loop. We do this *before* loading
    # the 6 GB model so a fully-completed run can exit fast without paying for
    # a forward pass it doesn't need.
    cached: dict = {}  # id -> {"pred": str, "gold": str}
    if args.resume and pred_path.exists():
        # Defense-in-depth: if a prior summary JSON exists, verify it was
        # produced with the same model_dir before reusing its predictions.
        # The stem already includes a model tag for non-default models, so a
        # mismatch here means something more fundamental is wrong (e.g. user
        # pointed --model-dir at the symlink target instead of the symlink).
        if out_path.exists():
            try:
                prior = json.loads(out_path.read_text())
                prior_model = prior.get("model_dir")
                if prior_model and prior_model != args.model_dir:
                    print(
                        f"ERROR: --resume refused: prior summary at {out_path}\n"
                        f"  was produced with model_dir={prior_model!r}\n"
                        f"  but this run uses    model_dir={args.model_dir!r}.\n"
                        f"  Either pass --overwrite, vary the output stem, or "
                        f"point --model-dir at the same path the prior run used.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            except (json.JSONDecodeError, OSError):
                pass  # malformed prior summary — fall through and let the
                      # JSONL pass handle it
        with pred_path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"ERROR: malformed JSONL at {pred_path}:{line_no}; "
                        "fix or delete the file before resuming.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                cached[str(rec["id"])] = {"pred": rec["pred"], "gold": rec["gold"]}
        print(f"[resume] loaded {len(cached)} cached predictions from {pred_path}",
              flush=True)

    tokenizer, model, device, adapter_params = load_model(
        args.model_dir, args.adapter, args.base_adapter, args.device_map
    )
    max_new = args.max_new_tokens or TASKS[args.task]["max_new_tokens"]

    # New (this-run-only) generations — used for efficiency proxies. Combined
    # with `cached` below to score the full union.
    new_ids, new_preds, new_golds = [], [], []
    prompt_tok, gen_tok = [], []
    t0 = time.time()

    # Predictions JSONL is written incrementally (one flushed line per example)
    # so a preempted job's progress survives requeue; --resume then picks up
    # where it left off. Mode choice:
    #   --overwrite  → truncate ("w")
    #   --resume     → append    ("a")  — preserves cached lines
    #   neither      → truncate ("w")  (refuse-check already passed)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    pred_mode = "a" if args.resume else "w"
    with pred_path.open(pred_mode) as pred_file:
        for i, q in enumerate(questions):
            if q["id"] not in golds:
                print(f"  WARN: id {q['id']} has no gold, skipping", file=sys.stderr)
                continue
            if str(q["id"]) in cached:
                continue  # already done in a prior session
            messages = build_messages(args.task, q, args.k, args.no_profile)
            prompt = render_prompt(tokenizer, messages)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,  # greedy — deterministic, reproducible eval
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][input_len:]
            # clean_up_tokenization_spaces=False: the default cleanup is designed for
            # WordPiece and is destructive for BPE (which SmolLM3 uses) — suppresses the
            # transformers warning and avoids corrupting punctuation spacing.
            text = clean_output(
                tokenizer.decode(
                    new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            )

            new_ids.append(q["id"])
            new_preds.append(text)
            new_golds.append(golds[q["id"]])
            prompt_tok.append(int(input_len))
            gen_tok.append(int(len(new_tokens)))

            # Flush so a SIGTERM (Condor preemption) loses at most the in-flight
            # example, never previously-completed work.
            pred_file.write(json.dumps({"id": q["id"], "pred": text, "gold": golds[q["id"]]}) + "\n")
            pred_file.flush()

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(questions)}", flush=True)

    elapsed = time.time() - t0

    # Score on the union of cached + new. The cached order is iteration order
    # of the dict (insertion order in CPython 3.7+), which matches the JSONL
    # line order — fine since both metrics are order-invariant.
    cached_preds_list = [v["pred"] for v in cached.values()]
    cached_golds_list = [v["gold"] for v in cached.values()]
    all_preds = cached_preds_list + new_preds
    all_golds = cached_golds_list + new_golds
    all_ids = list(cached.keys()) + [str(i) for i in new_ids]

    rating = TASKS[args.task]["metric"] == "rating"
    metrics = score_rating(all_preds, all_golds) if rating else score_rouge1(all_preds, all_golds)
    primary = "accuracy" if rating else "rouge1"

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    # FLAT, single-level record: every field is a scalar so a whole sweep of runs
    # loads straight into a dataframe with no unnesting —
    #   import pandas as pd, glob, json
    #   df = pd.DataFrame([json.load(open(f)) for f in glob.glob("results/*.json")])
    # `metric_name`/`metric_value` normalize the primary metric across tasks
    # (accuracy for LaMP-3, ROUGE-1 for 4/7) so they can be plotted on one axis;
    # the task-specific columns (accuracy/mae/rouge1) are kept too and are simply
    # null for tasks they don't apply to.
    record = {
        "schema_version": 1,
        # identity / config — enough to reproduce this exact run
        "task": args.task,
        "split": args.split,
        "adapter": args.adapter,
        "adapter_name": adapter_tag,
        "base_adapter": args.base_adapter,
        "base_adapter_name": base_adapter_tag,
        "stacked_adapter_name": stacked_tag,
        "user_fingerprint": args.user_records,
        "user_records_from_file": args.user_records_from_file,
        "n_total_records": n_total_records,
        "n_filtered_records": n_filtered_records,
        "no_profile": args.no_profile,
        "retriever": "none" if args.no_profile else "bm25",
        "k": 0 if args.no_profile else args.k,
        "seed": args.seed,
        "decoding": "greedy",
        "max_new_tokens": max_new,
        "limit": args.limit,
        "model_dir": args.model_dir,
        "command": "python " + " ".join(sys.argv),
        # metrics (primary normalized + task-specific)
        "metric_name": primary,
        "metric_value": metrics[primary],
        "accuracy": metrics.get("accuracy"),
        "mae": metrics.get("mae"),
        "parse_fail_rate": metrics.get("parse_fail_rate"),
        "rouge1": metrics.get("rouge1"),
        "n": metrics.get("n"),
        # efficiency proxies — computed over the NEW generations only (the
        # cached predictions have no token counts on disk). `n_new` is the
        # denominator; `n_resumed` is what we picked up from JSONL. In a
        # fresh (non-resume) run, n_resumed=0 and n_new=n.
        "n_resumed": len(cached),
        "n_new": len(new_preds),
        "mean_prompt_tokens": round(mean(prompt_tok), 1) if prompt_tok else None,
        "mean_generated_tokens": round(mean(gen_tok), 1) if gen_tok else None,
        "adapter_params": adapter_params,
        "device_map": args.device_map,
        "seconds": round(elapsed, 1),
        "sec_per_example": round(elapsed / max(len(new_preds), 1), 3) if new_preds else None,
        # provenance captured at run start — includes Condor IDs, git commit,
        # library versions, hostname, and start timestamp.
        **provenance,
    }

    # Predictions JSONL was already written incrementally inside the main loop;
    # only the summary JSON is written here. --resume rewrites this freely (a
    # resumed run by definition wants to refresh the summary).
    out_path.write_text(json.dumps(record, indent=2))

    print("=" * 60)
    for _id, p, g in list(zip(all_ids, all_preds, all_golds))[:3]:
        print(f"  [{_id}] pred={p!r}  gold={g!r}")
    print("=" * 60)
    print(f"metric:     {primary} = {metrics[primary]:.4f}  (n={metrics.get('n')})")
    print(f"all metrics:{metrics}")
    print(f"efficiency: prompt_tok={record['mean_prompt_tokens']} "
          f"gen_tok={record['mean_generated_tokens']} "
          f"{record['sec_per_example']}s/ex "
          f"(new={len(new_preds)} resumed={len(cached)})")
    print(f"written ->  {out_path}")
    print(f"            {pred_path}")


if __name__ == "__main__":
    main()
