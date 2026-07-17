#!/usr/bin/env python3
"""
LongLaMP evaluation harness — baseline zero-shot and LongLaMP-LoRA adapters.

Runs SmolLM3-3B over a LongLaMP split, conditions on the reviewer's profile
via BM25 retrieval into the system prompt (PPEP-templated, paper-verbatim),
generates deterministically, and scores ROUGE-1 + ROUGE-L.

Mirrors eval/eval_lamp.py's CLI shape and conventions (provenance banner,
refuse-to-overwrite, --resume, --user-records-style filters omitted — no
per-user LongLaMP work exists yet), but is a separate script per the LL1 plan
(experiments/2026-07-17-longlamp-review-ll1-plan.md, decision #4): LaMP's
harness stays untouched.

Only task supported this round: product_review_user (LL1, Personalized
Review Writing, cold-start/user split). Unlike LaMP, LongLaMP's `input` field
is already a fully-templated instruction (verified via the HF
datasets-server API) — build_messages() does not construct a task
instruction, it only prepends the retrieved profile as a system message
ahead of `input` verbatim. Train/eval role-layout and BM25 MUST match
train/build_longlamp_dataset.py exactly (same cardinal rule as LaMP).

Usage (run inside the training Docker image):
    python eval/eval_longlamp.py --task product_review_user --split dev --k 4 --seed 0
    python eval/eval_longlamp.py --task product_review_user --split dev --no-profile   # floor
    python eval/eval_longlamp.py --task product_review_user --split dev --limit 5       # smoke
    python eval/eval_longlamp.py --task product_review_user --split test --adapter /path/to/lora
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
LONGLAMP_DIR = os.environ.get(
    "LONGLAMP_DIR",
    "/home/ange00008/projects/mobileFT_distill/data/longlamp",
)
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "/home/ange00008/projects/mobileFT_distill/results",
)
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/home/ange00008/projects/mobileFT_distill",
)

# Local split filenames (data/download_longlamp.py). LongLaMP's own HF split
# key is "validation"; we call it "dev" in the CLI to match eval_lamp.py's
# shape, and read it from val.json on disk.
SPLIT_FILES = {"dev": "val.json", "test": "test.json"}

# --- Per-task configuration ---------------------------------------------------
# index_field: profile-entry text BM25 matches the query against.
# format:      PPEP template (paper-verbatim, plan decision #8).
# max_new_tokens / metric: generation budget and scoring method.
TASKS = {
    "product_review_user": {
        "index_field": lambda it: " ".join([
            str(it.get("description", "")),
            str(it.get("summary", "")),
            str(it.get("reviewText", "")),
        ]),
        "format": lambda it: (
            f'{it.get("overall", "?")} is a rating for the product with '
            f'description {it.get("description", "")}. '
            f'{it.get("summary", "")} is summary for {it.get("reviewText", "")}'
        ),
        "max_new_tokens": 1024,
        "metric": "rouge",
    },
}

SYSTEM_PREAMBLE = (
    "The following are examples of this user's past activity. "
    "Use them to match this user's preferences and writing style.\n\n"
)


# --- BM25 (pure Python, no dependency; identical to LaMP's) ------------------
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


# --- Prompt construction -----------------------------------------------------
def build_messages(task: str, record: dict, k: int, no_profile: bool) -> list:
    """Turn a LongLaMP record into chat-template messages.

    Unlike LaMP, `record["input"]` is already the complete task instruction
    (verified against the real schema — see the plan doc), so the user turn
    is `input` verbatim; only the system message (retrieved profile) is
    built here.
    """
    user_content = record["input"]
    if no_profile:
        return [{"role": "user", "content": user_content}]
    retrieved = retrieve_profile(task, user_content, record.get("profile", []), k)
    if not retrieved:
        return [{"role": "user", "content": user_content}]
    lines = "\n".join(TASKS[task]["format"](it) for it in retrieved)
    return [
        {"role": "system", "content": SYSTEM_PREAMBLE + lines},
        {"role": "user", "content": user_content},
    ]


def render_prompt(tokenizer, messages: list) -> str:
    """Apply the chat template; ask the model not to emit reasoning if it can.
    Identical rationale to eval_lamp.py's render_prompt."""
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
    return _THINK.sub("", text).strip()


# --- Parsing + metrics -------------------------------------------------------
def score_rouge(preds: list, golds: list) -> dict:
    """Mean ROUGE-1 + ROUGE-L F1. Both come free from a single RougeScorer call
    (plan decision #5) — no new dependency, `rouge_score` is already in
    requirements.txt for LaMP-4/7's ROUGE-1.
    rouge_scorer.score(target, prediction) — note the (gold, pred) argument order.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1, rl = [], []
    for p, g in zip(preds, golds):
        s = scorer.score(str(g), p)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    n = len(preds)
    return {
        "rouge1": (sum(r1) / n) if n else 0.0,
        "rougeL": (sum(rl) / n) if n else 0.0,
        "n": n,
    }


# --- Provenance ----------------------------------------------------------------
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
        "condor_cluster_id": os.environ.get("CONDOR_CLUSTER_ID") or None,
        "condor_proc_id": os.environ.get("CONDOR_PROC_ID") or None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if porcelain is None else bool(porcelain),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft_v,
    }


# --- Adapter-path tagging ------------------------------------------------------
def derive_adapter_tag(adapter_path: str) -> str:
    if adapter_path.lower() == "none":
        return "base"
    p = Path(adapter_path.rstrip("/"))
    if p.name == "final" or p.name.startswith("checkpoint-"):
        return f"{p.parent.name}_{p.name}"
    return p.name


# --- Data loading ---------------------------------------------------------------
def load_split(task: str, split: str):
    """Load a LongLaMP split. Records are {reviewerId, input, output, profile}.
    Golds are keyed by a synthetic id (the record's index in file order) since
    LongLaMP doesn't ship a separate example id field."""
    fname = SPLIT_FILES[split]
    path = Path(LONGLAMP_DIR) / task / fname
    if not path.exists():
        print(f"ERROR: {path} not found — run data/download_longlamp.py first.",
              file=sys.stderr)
        sys.exit(1)
    records = json.loads(path.read_text())
    questions = [{"id": str(i), "input": r["input"], "profile": r.get("profile", [])}
                 for i, r in enumerate(records)]
    golds = {str(i): r["output"] for i, r in enumerate(records)}
    return questions, golds


# --- Model loading (identical to eval_lamp.py) --------------------------------
def load_model(
    model_dir: str,
    adapter: str | None,
    base_adapter: str | None = None,
    device_map: str = "cuda",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
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
    device = next(model.parameters()).device
    return tokenizer, model, device, adapter_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="product_review_user", choices=list(TASKS.keys()))
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--adapter", default="none", help="LoRA checkpoint path, or 'none'")
    parser.add_argument(
        "--base-adapter",
        default="none",
        help="optional pre-existing LoRA adapter to merge into the base before "
        "attaching --adapter on top. 'none' disables. (No LongLaMP User-LoRA "
        "exists yet — kept for CLI-shape parity with eval_lamp.py and future use.)",
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
        "ids, append new generations, recompute summary over the union.",
    )
    parser.add_argument(
        "--device-map",
        default="cuda",
        help="device_map passed to from_pretrained. 'cuda' (default) / 'cpu' for "
        "single-device placement; 'auto' for accelerate multi-GPU sharding.",
    )
    args = parser.parse_args()

    if args.resume and args.overwrite:
        print("ERROR: --resume and --overwrite are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

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

    adapter_tag = derive_adapter_tag(args.adapter)
    base_adapter_tag = derive_adapter_tag(args.base_adapter)
    has_base = args.base_adapter.lower() != "none"
    has_user = args.adapter.lower() != "none"
    if has_base and has_user:
        stacked_tag = f"{base_adapter_tag}_{adapter_tag}"
    elif has_base:
        stacked_tag = base_adapter_tag
    else:
        stacked_tag = adapter_tag
    profile_tag = "noprofile" if args.no_profile else f"bm25k{args.k}"
    limit_tag = f"_limit{args.limit}" if args.limit > 0 else ""
    stem = f"LongLaMP_{args.task}_{args.split}_{stacked_tag}_{profile_tag}_seed{args.seed}{limit_tag}"
    out_path = Path(RESULTS_DIR) / f"{stem}.json"
    pred_path = Path(RESULTS_DIR) / f"{stem}.predictions.jsonl"
    if not args.overwrite and not args.resume and (out_path.exists() or pred_path.exists()):
        print("ERROR: refusing to overwrite existing results:", file=sys.stderr)
        for p in (out_path, pred_path):
            mark = "exists" if p.exists() else "would-be-new"
            print(f"  {p}   [{mark}]", file=sys.stderr)
        print(
            "\nPass --overwrite to replace, --resume to continue an interrupted run,\n"
            "or vary --seed / --adapter / etc. to write to a different path.\n"
            "(Smoke runs with --limit already get a _limitN suffix and won't "
            "collide with full runs.)",
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

    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"{args.task}/{args.split}: {len(questions)} examples", flush=True)

    cached: dict = {}
    if args.resume and pred_path.exists():
        if out_path.exists():
            try:
                prior = json.loads(out_path.read_text())
                prior_model = prior.get("model_dir")
                if prior_model and prior_model != args.model_dir:
                    print(
                        f"ERROR: --resume refused: prior summary at {out_path}\n"
                        f"  was produced with model_dir={prior_model!r}\n"
                        f"  but this run uses    model_dir={args.model_dir!r}.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            except (json.JSONDecodeError, OSError):
                pass
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

    new_ids, new_preds, new_golds = [], [], []
    prompt_tok, gen_tok = [], []
    t0 = time.time()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    pred_mode = "a" if args.resume else "w"
    with pred_path.open(pred_mode) as pred_file:
        for i, q in enumerate(questions):
            if q["id"] not in golds:
                print(f"  WARN: id {q['id']} has no gold, skipping", file=sys.stderr)
                continue
            if str(q["id"]) in cached:
                continue
            messages = build_messages(args.task, q, args.k, args.no_profile)
            prompt = render_prompt(tokenizer, messages)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][input_len:]
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

            pred_file.write(json.dumps({"id": q["id"], "pred": text, "gold": golds[q["id"]]}) + "\n")
            pred_file.flush()

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(questions)}", flush=True)

    elapsed = time.time() - t0

    cached_preds_list = [v["pred"] for v in cached.values()]
    cached_golds_list = [v["gold"] for v in cached.values()]
    all_preds = cached_preds_list + new_preds
    all_golds = cached_golds_list + new_golds
    all_ids = list(cached.keys()) + [str(i) for i in new_ids]

    metrics = score_rouge(all_preds, all_golds)
    primary = "rouge1"

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    record = {
        "schema_version": 1,
        "task": args.task,
        "split": args.split,
        "adapter": args.adapter,
        "adapter_name": adapter_tag,
        "base_adapter": args.base_adapter,
        "base_adapter_name": base_adapter_tag,
        "stacked_adapter_name": stacked_tag,
        "n_total_records": n_total_records,
        "no_profile": args.no_profile,
        "retriever": "none" if args.no_profile else "bm25",
        "k": 0 if args.no_profile else args.k,
        "seed": args.seed,
        "decoding": "greedy",
        "max_new_tokens": max_new,
        "limit": args.limit,
        "model_dir": args.model_dir,
        "command": "python " + " ".join(sys.argv),
        "metric_name": primary,
        "metric_value": metrics[primary],
        "rouge1": metrics.get("rouge1"),
        "rougeL": metrics.get("rougeL"),
        "n": metrics.get("n"),
        "n_resumed": len(cached),
        "n_new": len(new_preds),
        "mean_prompt_tokens": round(mean(prompt_tok), 1) if prompt_tok else None,
        "mean_generated_tokens": round(mean(gen_tok), 1) if gen_tok else None,
        "adapter_params": adapter_params,
        "device_map": args.device_map,
        "seconds": round(elapsed, 1),
        "sec_per_example": round(elapsed / max(len(new_preds), 1), 3) if new_preds else None,
        **provenance,
    }

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
