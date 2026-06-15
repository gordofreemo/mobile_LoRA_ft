#!/usr/bin/env python3
"""Train a Task-LoRA adapter on the LaMP training corpus.

Reads all hyperparameters from a JSON config (default:
train/config/a1_lamp.json) so a run is reproducible from a single CLI command
and the config file alone. Loads the bf16 SmolLM3-3B base from
data/models/SmolLM3-3B, wraps it with PEFT LoRA, applies the SmolLM3 chat
template per example (thinking disabled — matches eval/eval_lamp.py), masks
loss to the assistant span only, and runs HF Trainer.

Train/eval consistency (the cardinal rule per CLAUDE.md):
  - role layout: {system: BM25-retrieved profile, user: LaMP input,
    assistant: gold}, with empty/missing system meaning no system message
    at all (matches eval_lamp.build_messages exactly)
  - chat template: tokenizer.apply_chat_template(..., enable_thinking=False)
    so the model never sees the <think> block during training; eval also
    disables thinking, so the train and inference shapes line up

Usage:
    python train/train.py                                # full A1-lamp run
    python train/train.py --config train/config/a1_lamp.json
    python train/train.py --limit 64 --max_steps 4 --no-wandb   # smoke
    python train/train.py --resume                              # pick up latest ckpt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/home/ange00008/projects/mobileFT_distill")
)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _derive_adapter_tag(adapter_path: str) -> str:
    """Filename-safe short identifier for an adapter checkpoint dir. Mirrors
    eval/eval_lamp.py:derive_adapter_tag so train- and eval-time tags line up
    when a stacked adapter is recorded in provenance."""
    p = Path(adapter_path.rstrip("/"))
    if p.name == "final" or p.name.startswith("checkpoint-"):
        return f"{p.parent.name}_{p.name}"
    return p.name


# --- Chat template + loss mask ----------------------------------------------
def _flatten(x):
    """apply_chat_template returns flat lists for a single conversation but
    list-of-lists when input is batched — be defensive about both shapes."""
    return x[0] if (x and isinstance(x[0], list)) else x


def build_example(record, tokenizer, max_length: int):
    """Render one (system, user, assistant) record into (input_ids, labels).

    Loss mask = the `{% generation %}` span in the chat template, exposed
    by transformers' `return_assistant_tokens_mask=True`. SmolLM3's
    chat_template.jinja wraps the assistant turn in those markers, which
    is the canonical way to mark "supervise on these tokens, mask the
    rest." The earlier "diff two renders" approach doesn't work for
    SmolLM3 because the template auto-prepends a system metadata block
    (date / reasoning mode) and uses non-stripping `{% generation %}`
    tags, both of which break the prompt-is-prefix-of-full invariant.

    `enable_thinking=False` keeps the <think>...</think> block in the
    target so train and eval see the same shape (eval also disables
    thinking). Older transformers without the kwarg raise TypeError,
    hence the fallback.
    """
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages.append({"role": "user", "content": record["user"]})
    messages.append({"role": "assistant", "content": record["assistant"]})

    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            enable_thinking=False,
        )
    except TypeError:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )

    input_ids = list(_flatten(out["input_ids"]))
    attention_mask = list(_flatten(out["attention_mask"]))
    assistant_mask = list(_flatten(out["assistant_masks"]))

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        assistant_mask = assistant_mask[:max_length]

    labels = [tok if m else -100 for tok, m in zip(input_ids, assistant_mask)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# --- Metric streaming --------------------------------------------------------
class JsonlMetricCallback(TrainerCallback):
    """Append every HF Trainer log event to a JSONL file as it happens.

    Complements W&B (which may be unreachable or unconfigured) and gives
    post-hoc analysis a flat, greppable, pandas-friendly record. Survives
    crashes — partial events are on disk as soon as they fire.

    Each line is a JSON object with `step`, `epoch`, `wall_s`,
    `timestamp_utc`, plus whatever Trainer logged (loss, learning_rate,
    grad_norm, and the end-of-run train_runtime / train_samples_per_second /
    etc. summary).
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        import datetime

        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            "wall_s": round(time.time() - self._t0, 2),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            **logs,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def summarize_log_history(log_history: list) -> dict:
    """Reduce trainer.state.log_history to flat scalars for train_meta.json.

    `loss` events fire every `logging_steps`; `train_loss` is the end-of-run
    aggregate that fires once regardless. Short smoke runs (total_steps <
    logging_steps) only have the aggregate, so fall back to it.
    """
    losses = [e["loss"] for e in log_history if "loss" in e]
    lrs = [e["learning_rate"] for e in log_history if "learning_rate" in e]
    grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]
    final_summary = next(
        (e for e in reversed(log_history) if "train_runtime" in e), {}
    )
    aggregate_loss = final_summary.get("train_loss")
    return {
        "n_log_events": len(log_history),
        "n_loss_points": len(losses),
        "first_train_loss": losses[0] if losses else aggregate_loss,
        "final_train_loss": losses[-1] if losses else aggregate_loss,
        "min_train_loss": min(losses) if losses else aggregate_loss,
        "aggregate_train_loss": aggregate_loss,
        "final_learning_rate": lrs[-1] if lrs else None,
        "final_grad_norm": grad_norms[-1] if grad_norms else None,
        "train_runtime_s": final_summary.get("train_runtime"),
        "train_samples_per_second": final_summary.get("train_samples_per_second"),
        "train_steps_per_second": final_summary.get("train_steps_per_second"),
        "total_flos": final_summary.get("total_flos"),
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
        "torch_version": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "train/config/a1_lamp.json"),
        help="path to JSON hyperparameter config",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap training examples (smoke testing). Appends _limitN to output_dir.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="override num_train_epochs by global-step cap (smoke). "
        "Appends _stepsN to output_dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest checkpoint in output_dir (default: refuse "
        "to touch a populated output_dir)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="disable W&B regardless of config; useful for smoke tests",
    )
    args = parser.parse_args()

    # Resolve via _resolve so a relative `--config` (as passed from the
    # Condor sub file) is interpreted against PROJECT_ROOT, not the
    # sandbox CWD. Treats absolute paths unchanged.
    config_path = _resolve(args.config)
    cfg = json.loads(config_path.read_text())

    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] train condition={cfg['condition']} seed={cfg['seed']} "
        f"commit={commit_short} dirty={provenance.get('git_dirty')} "
        f"condor={provenance.get('condor_cluster_id') or '-'}."
        f"{provenance.get('condor_proc_id') or '-'} "
        f"host={provenance.get('hostname')} "
        f"limit={args.limit} max_steps={args.max_steps}",
        flush=True,
    )

    set_seed(cfg["seed"])

    # --- Smoke-run suffixes to avoid clobbering real outputs ---------------
    output_dir = _resolve(cfg["output_dir"])
    if args.limit > 0:
        output_dir = output_dir.parent / f"{output_dir.name}_limit{args.limit}"
    if args.max_steps > 0:
        output_dir = output_dir.parent / f"{output_dir.name}_steps{args.max_steps}"

    # --- Refuse-to-overwrite -----------------------------------------------
    meta_path = output_dir / "train_meta.json"
    if meta_path.exists() and not args.resume:
        sys.exit(
            f"ERROR: refusing to overwrite — {meta_path} exists.\n"
            f"  Pass --resume to continue from the latest checkpoint, or "
            f"delete {output_dir} to rerun from scratch."
        )

    # --- Tokenizer & model -------------------------------------------------
    model_path = _resolve(cfg["model_name_or_path"])
    print(f"[model] loading tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.chat_template is None:
        sys.exit("ERROR: tokenizer has no chat_template; cannot build SFT prompts.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] loading bf16 base from {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Optional: load a pre-existing LoRA adapter and merge it into the base
    # before attaching a fresh LoRA on top. This is the User-LoRA pattern —
    # the Task-LoRA's weights become part of the frozen backbone, then a
    # small per-user adapter is trained over it. After merge_and_unload the
    # model is a plain AutoModelForCausalLM again, so the rest of the path
    # (grad checkpointing + get_peft_model) is identical to the no-stack case.
    base_adapter_path = cfg.get("base_adapter")
    base_adapter_tag = None
    if base_adapter_path:
        base_adapter_path = str(_resolve(base_adapter_path))
        base_adapter_tag = _derive_adapter_tag(base_adapter_path)
        print(f"[model] merging base adapter {base_adapter_tag} from "
              f"{base_adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, base_adapter_path)
        model = model.merge_and_unload()

    # PEFT + gradient checkpointing requires inputs to require grads so the
    # backward pass can reach the LoRA params through the frozen base.
    if cfg["trainer"].get("gradient_checkpointing", False):
        model.enable_input_require_grads()

    lora_config = LoraConfig(**cfg["lora"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset -----------------------------------------------------------
    dataset_path = _resolve(cfg["dataset_path"])
    print(f"[data] loading {dataset_path}", flush=True)
    raw = []
    with open(dataset_path) as f:
        for line in f:
            raw.append(json.loads(line))
    if args.limit > 0:
        raw = raw[: args.limit]
    print(f"[data] {len(raw)} raw records", flush=True)

    max_seq_length = cfg["data"]["max_seq_length"]
    print(f"[data] tokenizing (max_seq_length={max_seq_length})", flush=True)
    t0 = time.time()
    processed = []
    dropped_truncated = 0
    for r in raw:
        ex = build_example(r, tokenizer, max_seq_length)
        if not any(label != -100 for label in ex["labels"]):
            # The prompt alone hit max_seq_length and truncation removed
            # the entire assistant span — this example has no supervision
            # signal, drop it.
            dropped_truncated += 1
            continue
        processed.append(ex)
    print(
        f"[data] tokenized in {time.time()-t0:.0f}s; "
        f"{len(processed)} usable, {dropped_truncated} dropped (truncated past assistant)",
        flush=True,
    )
    if not processed:
        sys.exit(
            "ERROR: 0 usable training examples after tokenization. Common causes:\n"
            "  - assistant_masks is all zero (chat template missing {% generation %} blocks)\n"
            "  - max_seq_length too small for typical examples\n"
            "Inspect the first record's labels manually to debug."
        )
    # Sanity stats on the first surviving example — a 0 here means the
    # supervision-mask plumbing is broken (would silently train on nothing).
    _first = processed[0]
    _n_sup = sum(1 for label in _first["labels"] if label != -100)
    print(
        f"[data] first example: {len(_first['input_ids'])} tokens, "
        f"{_n_sup} supervised (assistant span)",
        flush=True,
    )

    train_ds = Dataset.from_list(processed)

    # --- W&B setup ---------------------------------------------------------
    use_wandb = (not args.no_wandb) and cfg["trainer"].get("report_to") == "wandb"
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb"]["project"])

    trainer_kwargs = dict(cfg["trainer"])
    if args.no_wandb:
        trainer_kwargs["report_to"] = "none"
    if args.max_steps > 0:
        trainer_kwargs["max_steps"] = args.max_steps

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=cfg["seed"],
        run_name=cfg["wandb"]["run_name"] if use_wandb else None,
        **trainer_kwargs,
    )

    # Padding collator that also pads `labels` with -100 — standard pattern
    # for causal SFT done outside of trl.SFTTrainer.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    metrics_path = output_dir / "metrics.jsonl"
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[JsonlMetricCallback(metrics_path)],
    )

    print(f"[train] starting; output_dir={output_dir}", flush=True)
    print(f"[train] metrics streaming to {metrics_path}", flush=True)
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    # --- Save final adapter + run metadata ---------------------------------
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metric_summary = summarize_log_history(trainer.state.log_history)

    # Also write the full log_history for archival (it's also inside the
    # checkpoint trainer_state.json, but a top-level copy keeps post-hoc
    # analysis from having to dig through nested checkpoint dirs).
    log_hist_path = output_dir / "log_history.json"
    log_hist_path.write_text(json.dumps(trainer.state.log_history, indent=2))

    meta = {
        "schema_version": 1,
        "condition": cfg["condition"],
        "config_path": str(config_path),
        "config_snapshot": cfg,
        "command": "python " + " ".join(sys.argv),
        "limit": args.limit,
        "max_steps": args.max_steps,
        "final_adapter_dir": str(final_dir),
        "metrics_jsonl": str(metrics_path),
        "log_history_json": str(log_hist_path),
        "n_train_examples": len(processed),
        "n_dropped_truncated": dropped_truncated,
        "global_step": trainer.state.global_step,
        "base_adapter_path": base_adapter_path,
        "base_adapter_tag": base_adapter_tag,
        **metric_summary,
        **provenance,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(
        f"[done] adapter={final_dir} step={meta['global_step']} "
        f"final_loss={meta['final_train_loss']} "
        f"min_loss={meta['min_train_loss']} "
        f"runtime_s={meta['train_runtime_s']}",
        flush=True,
    )
    print(f"[done] meta -> {meta_path}", flush=True)
    print(f"[done] metrics -> {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
