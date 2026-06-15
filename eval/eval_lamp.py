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
    """Apply the chat template with thinking disabled (fall back if unsupported).

    SmolLM3 is a reasoning model whose template can emit a <think> block by default.
    For evaluation we want the answer only — reasoning eats the token budget and
    complicates parsing — so we pass enable_thinking=False. Older transformers /
    templates that don't accept the kwarg raise TypeError, so we fall back. (As a
    belt-and-braces measure, clean_output() also strips any <think> block that slips
    through; verify in the smoke-test output that thinking is actually off.)
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
def load_model(model_dir: str, adapter: str | None, base_adapter: str | None = None):
    """Load SmolLM3-3B in bf16, optionally with a stacked pair of LoRA adapters.

    The two-adapter case (`base_adapter` set) is the User-LoRA stacking pattern:
    Task-LoRA is loaded first, merged into the frozen base, then a fresh
    per-user LoRA is loaded on top. This must match train/train.py's
    base-adapter plumbing exactly so train- and inference-time graphs are
    identical. The intermediate merge is required — PEFT can't stack two
    distinct LoraConfig adapters from disk without it.

    `adapter_params` counts only the *outer* (user) LoRA's parameters — the
    base adapter has been merged into the frozen weights and contributes
    nothing to the on-device weight cost at inference. When only the base is
    given (smoke test of the base-adapter path with `--adapter none`), we
    instead count the base's params so the field stays informative.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model onto {device} (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device
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
    args = parser.parse_args()

    # Capture provenance once at startup so the banner and the result record agree
    # and `timestamp_utc` reflects start time. The banner makes the runlog
    # self-documenting: anyone reading runlogs/eval_lamp.<task>.<cluster>.<proc>.out
    # sees task, condition, seed, commit, and Condor IDs on the very first line.
    provenance = collect_provenance()
    cond_label = "noprofile" if args.no_profile else f"bm25k{args.k}"
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] task={args.task} split={args.split} cond={cond_label} "
        f"seed={args.seed} limit={args.limit} commit={commit_short} "
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
    limit_tag = f"_limit{args.limit}" if args.limit > 0 else ""
    user_tag = f"_user{args.user_records}" if args.user_records else ""
    stem = f"{args.task}_{args.split}_{stacked_tag}_{profile_tag}_seed{args.seed}{user_tag}{limit_tag}"
    out_path = Path(RESULTS_DIR) / f"{stem}.json"
    pred_path = Path(RESULTS_DIR) / f"{stem}.predictions.jsonl"
    if not args.overwrite and (out_path.exists() or pred_path.exists()):
        print(
            "ERROR: refusing to overwrite existing results:",
            file=sys.stderr,
        )
        for p in (out_path, pred_path):
            mark = "exists" if p.exists() else "would-be-new"
            print(f"  {p}   [{mark}]", file=sys.stderr)
        print(
            "\nPass --overwrite to replace, or vary --seed / --task / --adapter / etc.\n"
            "to write to a different path. (Smoke runs with --limit already get a\n"
            "_limitN suffix and won't collide with full runs.)",
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

    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"{args.task}/{args.split}: {len(questions)} examples", flush=True)

    tokenizer, model, device, adapter_params = load_model(
        args.model_dir, args.adapter, args.base_adapter
    )
    max_new = args.max_new_tokens or TASKS[args.task]["max_new_tokens"]

    ids, preds, gold_list = [], [], []
    prompt_tok, gen_tok = [], []
    t0 = time.time()

    for i, q in enumerate(questions):
        if q["id"] not in golds:
            print(f"  WARN: id {q['id']} has no gold, skipping", file=sys.stderr)
            continue
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

        ids.append(q["id"])
        preds.append(text)
        gold_list.append(golds[q["id"]])
        prompt_tok.append(int(input_len))
        gen_tok.append(int(len(new_tokens)))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(questions)}", flush=True)

    elapsed = time.time() - t0

    rating = TASKS[args.task]["metric"] == "rating"
    metrics = score_rating(preds, gold_list) if rating else score_rouge1(preds, gold_list)
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
        # efficiency proxies
        "mean_prompt_tokens": round(mean(prompt_tok), 1),
        "mean_generated_tokens": round(mean(gen_tok), 1),
        "adapter_params": adapter_params,
        "seconds": round(elapsed, 1),
        "sec_per_example": round(elapsed / max(len(preds), 1), 3),
        # provenance captured at run start — includes Condor IDs, git commit,
        # library versions, hostname, and start timestamp.
        **provenance,
    }

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))

    # Full per-example predictions go to a sibling JSONL — kept out of the summary
    # so the summary stays a clean one-row record, but available for error analysis
    # and re-scoring without re-running the model.
    with pred_path.open("w") as f:
        for _id, p, g in zip(ids, preds, gold_list):
            f.write(json.dumps({"id": _id, "pred": p, "gold": g}) + "\n")

    print("=" * 60)
    for _id, p, g in list(zip(ids, preds, gold_list))[:3]:
        print(f"  [{_id}] pred={p!r}  gold={g!r}")
    print("=" * 60)
    print(f"metric:     {primary} = {metrics[primary]:.4f}  (n={metrics.get('n')})")
    print(f"all metrics:{metrics}")
    print(f"efficiency: prompt_tok={record['mean_prompt_tokens']} "
          f"gen_tok={record['mean_generated_tokens']} "
          f"{record['sec_per_example']}s/ex")
    print(f"written ->  {out_path}")
    print(f"            {pred_path}")


if __name__ == "__main__":
    main()
