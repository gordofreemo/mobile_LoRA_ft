#!/usr/bin/env python3
"""
BFCL evaluation harness — bare SmolLM3-3B and (later) LoRA adapters.

Why this exists: CLAUDE.md mandates a BFCL regression check before and after each
Task-LoRA training run (target ≥ 90, ref 92.3). This script produces that number.

Design choice: we use **our own** transformers-based generation and call BFCL's
official `ast_checker` as a library on the resulting outputs. This avoids
installing vllm/sglang and avoids upstreaming a SmolLM3 handler into bfcl-eval —
both heavy lifts that the catastrophic-forgetting-delta question doesn't need.

What's scored: the non-live AST categories (simple_python, simple_java,
simple_javascript, parallel, multiple, parallel_multiple, irrelevance). These are
the "92.3-style" categories — no API keys, no internet, no multi-turn.

Run on the cluster, inside the :ver4 Docker image (which has bfcl-eval installed).

Usage:
    python eval/eval_bfcl.py --seed 0
    python eval/eval_bfcl.py --categories simple_python --limit 5     # smoke
    python eval/eval_bfcl.py --adapter /path/to/lora                  # post-training
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter

# bfcl-eval's `eval_config.py` does `RESULT_PATH.mkdir(...)` at import time,
# rooted at $BFCL_PROJECT_ROOT (default: the package install dir, e.g.
# /opt/conda/lib/python3.11/site-packages/, which is read-only inside the Docker
# container). Point it at a writable scratch dir BEFORE any `import bfcl_eval`
# anywhere in the script — even transitively via load_bfcl_data or score_one.
_BFCL_ROOT = os.environ.setdefault(
    "BFCL_PROJECT_ROOT", "/tmp/bfcl_project_root"
)
Path(_BFCL_ROOT).mkdir(parents=True, exist_ok=True)

MODEL_DIR = os.environ.get(
    "MODEL_OUT_DIR",
    "/home/ange00008/projects/mobileFT_distill/data/models/SmolLM3-3B",
)
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "/home/ange00008/projects/mobileFT_distill/results",
)
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    "/home/ange00008/projects/mobileFT_distill",
)

# The non-live AST set — what the published BFCL "92.3-style" numbers come from.
DEFAULT_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "multiple",
    "parallel_multiple",
    "irrelevance",
]

# Per-category language. Only the simple_{java,javascript} categories use
# non-Python AST checking; everything else (parallel, multiple, etc.) is Python.
LANGUAGE_BY_CATEGORY = {
    "simple_python": "Python",
    "simple_java": "Java",
    "simple_javascript": "JavaScript",
    "parallel": "Python",
    "multiple": "Python",
    "parallel_multiple": "Python",
    "irrelevance": "Python",
}

# `ast_checker` takes a model_name string and uses it to look up per-model output
# conventions in MODEL_CONFIG_MAPPING. SmolLM3-3B isn't registered there, so we
# pass a neutral OSS-instruct placeholder that exists in the mapping. The spike
# confirmed this scores correctly on simple_python; document the choice loudly
# because the upstream config may evolve.
SCORER_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


# --- Adapter-path tagging ----------------------------------------------------
def derive_adapter_tag(adapter_path: str) -> str:
    """Short identifier for filenames / records.

    Mirrors eval_lamp.py.derive_adapter_tag — when the leaf is a generic name
    ("final" or "checkpoint-N"), prepend the parent dir so results from e.g.
    `train/checkpoints/a1_lamp_seed0/final` get tagged `a1_lamp_seed0_final`
    instead of the uninformative `final`.
    """
    if adapter_path.lower() == "none":
        return "base"
    p = Path(adapter_path.rstrip("/"))
    if p.name == "final" or p.name.startswith("checkpoint-"):
        return f"{p.parent.name}_{p.name}"
    return p.name


# --- BFCL data loading -------------------------------------------------------
def load_bfcl_data(category: str):
    """Load (questions, golds-by-id) for one BFCL category.

    The bfcl-eval package ships its test data alongside its code (under
    `bfcl_eval/data/`), as JSONL files: one BFCL_v4_<category>.json with test
    cases and a possible_answer/BFCL_v4_<category>.json with gold answers, both
    keyed by `id`.
    """
    import bfcl_eval

    pkg = Path(bfcl_eval.__file__).parent
    q_path = pkg / "data" / f"BFCL_v4_{category}.json"
    a_path = pkg / "data" / "possible_answer" / f"BFCL_v4_{category}.json"
    qs = [json.loads(line) for line in q_path.read_text().splitlines() if line.strip()]
    golds = {}
    for line in a_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        golds[rec["id"]] = rec
    return qs, golds


# --- Tool-call output parsing ------------------------------------------------
# Defensive parser for SmolLM3 / Qwen-style tool calls. The model is expected to
# emit each tool call as a JSON object inside <tool_call>...</tool_call> tags,
# possibly multiple per response (for parallel / multiple categories). We also
# accept bare JSON objects, JSON arrays, and code-fenced JSON as fallbacks.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", re.DOTALL)


def _normalize_call(obj) -> dict | None:
    """Turn an LLM tool-call JSON object into BFCL's expected dict shape.

    BFCL's `ast_checker` expects each call as `{"fn_name": {"arg": val, ...}}`.
    Most chat templates produce `{"name": "fn_name", "arguments": {...}}`.
    Some emit `{"name": "fn_name", "parameters": {...}}` or `arguments` as a
    JSON-encoded string. Handle the common variants; return None if shape is
    too off to recover.
    """
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function") or obj.get("tool_name")
    if not name:
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {name: args}


def parse_tool_calls(text: str) -> list:
    """Extract tool calls from raw model output as a list of {fn: {arg: val}}.

    Returns [] when no parseable tool call is found — which is the correct
    answer for the `irrelevance` category.
    """
    calls = []

    # Tagged tool calls first (SmolLM3 / Qwen native function-calling format).
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            for c in obj:
                norm = _normalize_call(c)
                if norm:
                    calls.append(norm)
        else:
            norm = _normalize_call(obj)
            if norm:
                calls.append(norm)
    if calls:
        return calls

    # Fenced JSON next.
    for m in _FENCED_JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            for c in obj:
                norm = _normalize_call(c)
                if norm:
                    calls.append(norm)
        else:
            norm = _normalize_call(obj)
            if norm:
                calls.append(norm)
    if calls:
        return calls

    # Bare JSON object at end of text (some models drop tags entirely).
    text_strip = text.strip()
    if text_strip.startswith("{") or text_strip.startswith("["):
        try:
            obj = json.loads(text_strip)
        except json.JSONDecodeError:
            return []
        if isinstance(obj, list):
            for c in obj:
                norm = _normalize_call(c)
                if norm:
                    calls.append(norm)
        else:
            norm = _normalize_call(obj)
            if norm:
                calls.append(norm)
    return calls


# --- Prompt construction -----------------------------------------------------
def build_messages(question_field) -> list:
    """BFCL `question` is a list of turn-lists. For non-multi-turn this is a
    single turn list with one user message. Flatten to the chat-template's
    expected `[{role, content}, ...]` shape.
    """
    if not question_field or not isinstance(question_field, list):
        return []
    first = question_field[0]
    if isinstance(first, list):
        return first
    return question_field


def render_prompt(tokenizer, messages: list, tools: list) -> str:
    """Apply chat template with tools= (native function-calling) and thinking off.

    SmolLM3-3B supports the OpenAI-style `tools=[{...}]` argument in
    `apply_chat_template`, which inlines the function schemas into the system
    prompt the same way it was trained on. We fall back through progressively
    weaker forms if the kwargs aren't accepted.
    """
    for kwargs in (
        {"tools": tools, "enable_thinking": False},
        {"tools": tools},
        {"enable_thinking": False},
        {},
    ):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except TypeError:
            continue
    raise RuntimeError("apply_chat_template did not accept any expected kwargs")


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_output(text: str) -> str:
    """Strip any leaked reasoning block, then whitespace."""
    return _THINK.sub("", text).strip()


# --- Provenance --------------------------------------------------------------
def collect_provenance() -> dict:
    """Same shape as eval_lamp.py — Condor IDs from env, git from subprocess."""
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

    try:
        import bfcl_eval

        bfcl_v = bfcl_eval.__version__ if hasattr(bfcl_eval, "__version__") else "?"
    except Exception:
        bfcl_v = None

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
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft_v,
        "bfcl_eval_version": bfcl_v,
    }


# --- Model loading -----------------------------------------------------------
def load_model(model_dir: str, adapter: str | None):
    """Load SmolLM3-3B (bf16, frozen). Same pattern as eval_lamp.py."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model onto {device} (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device
    )
    adapter_params = 0
    if adapter and adapter.lower() != "none":
        from peft import PeftModel

        print(f"Loading LoRA adapter from {adapter}...", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
        adapter_params = sum(
            p.numel() for n, p in model.named_parameters() if "lora_" in n
        )
        model = model.merge_and_unload()
    model.eval()
    return tokenizer, model, device, adapter_params


# --- Scoring -----------------------------------------------------------------
def score_one(category, q, gold, parsed_calls):
    """Score one example. Returns (is_valid, error_type, error_msgs).

    Uses bfcl-eval's ast_checker for everything except `irrelevance`, where the
    correct answer is "no tool call at all" — easy to score ourselves and the
    AST checker doesn't apply.
    """
    if category == "irrelevance":
        is_valid = len(parsed_calls) == 0
        return is_valid, None if is_valid else "spurious_call", []

    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    LANG = {
        "Python": Language.PYTHON,
        "Java": Language.JAVA,
        "JavaScript": Language.JAVASCRIPT,
    }
    try:
        result = ast_checker(
            func_description=q["function"],
            model_output=parsed_calls,
            possible_answer=gold["ground_truth"],
            language=LANG[LANGUAGE_BY_CATEGORY[category]],
            test_category=category,
            model_name=SCORER_MODEL_NAME,
        )
        return (
            bool(result.get("valid", False)),
            result.get("error_type"),
            result.get("error", []),
        )
    except Exception as e:
        return False, "scorer_exception", [str(e)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="comma-separated BFCL category list",
    )
    parser.add_argument("--adapter", default="none", help="LoRA checkpoint path, or 'none'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="evaluate only first N per category (smoke test)",
    )
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="generation budget per example",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing results files (default: refuse)",
    )
    args = parser.parse_args()

    provenance = collect_provenance()
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    adapter_tag = derive_adapter_tag(args.adapter)
    cat_tag = "ast" if set(categories) == set(DEFAULT_CATEGORIES) else "cats" + str(
        abs(hash(",".join(sorted(categories)))) % 10000
    )
    limit_tag = f"_limit{args.limit}" if args.limit > 0 else ""
    stem = f"bfcl_{cat_tag}_{adapter_tag}_seed{args.seed}{limit_tag}"
    out_path = Path(RESULTS_DIR) / f"{stem}.json"
    pred_path = Path(RESULTS_DIR) / f"{stem}.predictions.jsonl"

    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] eval=bfcl categories=[{','.join(categories)}] adapter={adapter_tag} "
        f"seed={args.seed} limit={args.limit} commit={commit_short} "
        f"condor={provenance.get('condor_cluster_id') or '-'}."
        f"{provenance.get('condor_proc_id') or '-'} "
        f"host={provenance.get('hostname')}",
        flush=True,
    )

    if not args.overwrite and (out_path.exists() or pred_path.exists()):
        print("ERROR: refusing to overwrite existing results:", file=sys.stderr)
        for p in (out_path, pred_path):
            mark = "exists" if p.exists() else "would-be-new"
            print(f"  {p}   [{mark}]", file=sys.stderr)
        print(
            "\nPass --overwrite to replace, or vary --seed / --adapter / etc.",
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
        sys.exit(1)

    tokenizer, model, device, adapter_params = load_model(args.model_dir, args.adapter)

    per_cat_correct = Counter()
    per_cat_total = Counter()
    error_type_counts = Counter()
    all_preds = []
    prompt_tok, gen_tok = [], []
    t0 = time.time()

    for category in categories:
        try:
            qs, golds = load_bfcl_data(category)
        except FileNotFoundError as e:
            print(f"  SKIP {category}: {e}", file=sys.stderr)
            continue
        if args.limit > 0:
            qs = qs[: args.limit]
        print(f"{category}: {len(qs)} cases", flush=True)

        for i, q in enumerate(qs):
            qid = q["id"]
            gold = golds.get(qid)
            if gold is None and category != "irrelevance":
                # irrelevance cases sometimes have no gold (since "no call" is the
                # correct answer); for others, missing gold is a data issue.
                print(f"  WARN: {qid} has no gold, skipping", file=sys.stderr)
                continue

            messages = build_messages(q.get("question", []))
            functions = q.get("function", [])
            prompt = render_prompt(tokenizer, messages, functions)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][input_len:]
            text = clean_output(
                tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
            parsed = parse_tool_calls(text)
            is_valid, error_type, error_msgs = score_one(category, q, gold, parsed)

            per_cat_total[category] += 1
            if is_valid:
                per_cat_correct[category] += 1
            else:
                error_type_counts[error_type or "unknown"] += 1

            prompt_tok.append(int(input_len))
            gen_tok.append(int(len(new_tokens)))

            all_preds.append(
                {
                    "id": qid,
                    "category": category,
                    "pred_text": text,
                    "pred_parsed": parsed,
                    "valid": is_valid,
                    "error_type": error_type,
                    "error": error_msgs[:2],  # cap to keep file size sane
                }
            )

            if (i + 1) % 50 == 0:
                print(f"  {category} {i + 1}/{len(qs)}", flush=True)

    elapsed = time.time() - t0

    # Weighted (per-example) accuracy across the included categories — same
    # convention BFCL uses for "overall AST".
    total_correct = sum(per_cat_correct.values())
    total_n = sum(per_cat_total.values())
    overall_acc = total_correct / total_n if total_n else 0.0

    def mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    per_category_accuracy = {
        c: (per_cat_correct[c] / per_cat_total[c]) if per_cat_total[c] else None
        for c in categories
    }

    record = {
        "schema_version": 1,
        "task": "BFCL_v4_ast",
        "categories": ",".join(categories),
        "adapter": args.adapter,
        "adapter_name": adapter_tag,
        "seed": args.seed,
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "scorer_model_name_placeholder": SCORER_MODEL_NAME,
        "model_dir": args.model_dir,
        "command": "python " + " ".join(sys.argv),
        "metric_name": "ast_accuracy",
        "metric_value": overall_acc,
        "accuracy": overall_acc,
        "n": total_n,
        "per_category_accuracy": per_category_accuracy,
        "per_category_n": {c: per_cat_total[c] for c in categories},
        "error_type_counts": dict(error_type_counts),
        "mean_prompt_tokens": round(mean(prompt_tok), 1),
        "mean_generated_tokens": round(mean(gen_tok), 1),
        "adapter_params": adapter_params,
        "seconds": round(elapsed, 1),
        "sec_per_example": round(elapsed / max(total_n, 1), 3),
        **provenance,
    }

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    with pred_path.open("w") as f:
        for p in all_preds:
            f.write(json.dumps(p) + "\n")

    print("=" * 60)
    print(f"BFCL AST accuracy: {overall_acc:.4f}   (n={total_n})")
    for c in categories:
        n = per_cat_total[c]
        if n:
            print(
                f"  {c:<22} acc={per_category_accuracy[c]:.4f}  "
                f"({per_cat_correct[c]}/{n})"
            )
        else:
            print(f"  {c:<22} (no cases)")
    if error_type_counts:
        print("  error types:", dict(error_type_counts))
    print("=" * 60)
    print(f"written ->  {out_path}")
    print(f"            {pred_path}")


if __name__ == "__main__":
    main()
