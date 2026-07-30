#!/usr/bin/env python3
"""Train a per-user LoRA with plain causal-LM next-token prediction on the
user's raw history — OPPU's recipe for the history-MISALIGNED LaMP tasks.

Why this exists (and why it isn't a flag on train/train.py)
-----------------------------------------------------------
Every other User-LoRA round in this project trains on supervised
(input, gold) pairs: `train/train.py` renders each record through SmolLM3's
chat template and masks the loss to the assistant span. That needs the user's
history to carry a per-entry target. Two LaMP tasks don't have one:

  LaMP-1 (citation identification) — profile entries are the author's own
      papers ({title, abstract}); the task is a binary choice between two
      candidate references, so a lone paper is not a choice-pair.
  LaMP-7 (tweet paraphrase) — profile entries are raw past tweets with no
      paired original/paraphrase.

OPPU (arXiv:2402.04401 §3) classifies exactly these two the same way and, for
them, substitutes "right-shifted history x_u' for unsupervised next token
prediction" in place of the supervised objective. That one sentence is the
paper's entire specification: no equation, no chunking scheme, no loss-mask
discussion, no per-task results. Every concrete choice below was made in this
project, not read off the paper, and should be treated as a first attempt at
an underspecified recipe rather than a faithful reimplementation.

What "right-shifted history" collapses to here
-----------------------------------------------
Standard causal LM: for a history text x = (x_1 ... x_n), predict x_{t+1}
from x_{<=t}. The "right shift" is exactly what a causal LM's own label
alignment does (HF shifts labels internally), so this script does NOT shift
anything by hand — it sets `labels = input_ids` and lets the model's own
loss do the shift. Double-shifting would train the model to skip a token.

Concretely, versus train/train.py:
  - no chat template, no system/user/assistant roles (the corpus rows have a
    single `text` field, emitted by `build_user_dataset.py --framing unsupervised`)
  - no assistant-span mask — loss on EVERY token
  - no BM25 retrieval into a system slot at TRAIN time. Eval is untouched:
    `eval_lamp.py` still uses the ordinary supervised prompt shape with BM25
    retrieval, so this changes the User-LoRA's training objective only.
  - one training example per profile entry, matching the per-entry convention
    the supervised path already uses

Everything else — config schema, seed handling, `base_adapter` stacking,
refuse-to-overwrite, provenance, metrics streaming, `train_meta.json` — is
kept byte-compatible with `train/train.py` so the downstream tooling
(checkpoint layout, meta fields, sub files) works unchanged.

This is a separate script rather than a mode on `train/train.py` on purpose:
train.py is the single well-tested SFT path that every other round depends
on, and this project's convention is to duplicate rather than deep-branch a
shared script (same rationale `train/build_user_dataset.py` documents).

Plan reference: experiments/2026-07-27-user-lora-remaining-tasks-plan.md §Piece 3.

Usage:
    python train/train_unsupervised_clm.py --config train/config/user_lora_lamp7_oppu_<user>.json
    python train/train_unsupervised_clm.py --config <cfg> --limit 8 --max_steps 2 --no-wandb  # smoke
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
    eval/eval_lamp.py:derive_adapter_tag and train/train.py's copy."""
    p = Path(adapter_path.rstrip("/"))
    if p.name == "final" or p.name.startswith("checkpoint-"):
        return f"{p.parent.name}_{p.name}"
    return p.name


# --- Raw-text tokenization (no chat template, no loss mask) ------------------
def build_example(record, tokenizer, max_length: int):
    """Render one raw-history row into (input_ids, attention_mask, labels).

    `labels = input_ids` — supervision on every token. The causal shift is
    performed by the model's own loss (HF shifts logits/labels internally),
    so shifting here as well would teach the model to skip a token.

    An EOS token is appended (when the tokenizer didn't already add one) so
    each history entry is a terminated document rather than running into the
    next one; without it the model learns that entries never end.
    """
    text = record["text"]
    out = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = list(out["input_ids"])
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and len(input_ids) < max_length and (
        not input_ids or input_ids[-1] != eos_id
    ):
        input_ids.append(eos_id)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": list(input_ids),
    }


# --- Metric streaming (byte-identical to train/train.py) --------------------
class JsonlMetricCallback(TrainerCallback):
    """Append every HF Trainer log event to a JSONL file as it happens."""

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
    """Reduce trainer.state.log_history to flat scalars for train_meta.json."""
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
    parser.add_argument("--config", required=True,
                        help="path to JSON hyperparameter config")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap training examples (smoke). Appends _limitN to output_dir.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="override num_train_epochs by global-step cap (smoke). "
                             "Appends _stepsN to output_dir.")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the latest checkpoint in output_dir")
    parser.add_argument("--no-wandb", action="store_true",
                        help="disable W&B regardless of config")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    cfg = json.loads(config_path.read_text())

    provenance = collect_provenance()
    commit_short = (provenance.get("git_commit") or "unknown")[:8]
    print(
        f"[run] train_unsupervised_clm condition={cfg['condition']} "
        f"seed={cfg['seed']} commit={commit_short} "
        f"dirty={provenance.get('git_dirty')} "
        f"condor={provenance.get('condor_cluster_id') or '-'}."
        f"{provenance.get('condor_proc_id') or '-'} "
        f"host={provenance.get('hostname')} "
        f"limit={args.limit} max_steps={args.max_steps}",
        flush=True,
    )

    set_seed(cfg["seed"])

    output_dir = _resolve(cfg["output_dir"])
    if args.limit > 0:
        output_dir = output_dir.parent / f"{output_dir.name}_limit{args.limit}"
    if args.max_steps > 0:
        output_dir = output_dir.parent / f"{output_dir.name}_steps{args.max_steps}"

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
    # NOTE: unlike train/train.py we do NOT require a chat_template — this
    # path deliberately never applies one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] loading bf16 base from {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    base_adapter_path = cfg.get("base_adapter")
    base_adapter_tag = None
    if base_adapter_path:
        base_adapter_path = str(_resolve(base_adapter_path))
        base_adapter_tag = _derive_adapter_tag(base_adapter_path)
        print(f"[model] merging base adapter {base_adapter_tag} from "
              f"{base_adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, base_adapter_path)
        model = model.merge_and_unload()

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

    # Fail loudly on a supervised corpus handed to the unsupervised trainer —
    # otherwise every row would KeyError one at a time deep inside the loop.
    if raw and "text" not in raw[0]:
        sys.exit(
            f"ERROR: {dataset_path} rows have no `text` field (found keys: "
            f"{sorted(raw[0])}). This trainer consumes the output of "
            f"`build_user_dataset.py --framing unsupervised`; a supervised "
            f"{{system,user,assistant}} corpus belongs to train/train.py."
        )

    max_seq_length = cfg["data"]["max_seq_length"]
    print(f"[data] tokenizing raw history text (max_seq_length={max_seq_length}, "
          f"no chat template, loss on every token)", flush=True)
    t0 = time.time()
    processed = []
    dropped_empty = 0
    n_truncated = 0
    total_tokens = 0
    for r in raw:
        ex = build_example(r, tokenizer, max_seq_length)
        if len(ex["input_ids"]) < 2:
            # A 0- or 1-token entry carries no next-token signal at all.
            dropped_empty += 1
            continue
        if len(ex["input_ids"]) >= max_seq_length:
            n_truncated += 1
        total_tokens += len(ex["input_ids"])
        processed.append(ex)
    print(
        f"[data] tokenized in {time.time()-t0:.0f}s; {len(processed)} usable, "
        f"{dropped_empty} dropped (<2 tokens), {n_truncated} hit max_seq_length, "
        f"{total_tokens} supervised tokens total",
        flush=True,
    )
    if not processed:
        sys.exit(
            "ERROR: 0 usable training examples after tokenization — every row "
            "was empty or shorter than 2 tokens. Inspect the corpus JSONL."
        )
    _first = processed[0]
    print(
        f"[data] first example: {len(_first['input_ids'])} tokens, "
        f"{len(_first['input_ids'])} supervised (every token)",
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

    # Same collator as the SFT path: pads `labels` with -100 so padding never
    # contributes loss. Here every REAL token is supervised; only pad is masked.
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

    log_hist_path = output_dir / "log_history.json"
    log_hist_path.write_text(json.dumps(trainer.state.log_history, indent=2))

    meta = {
        "schema_version": 1,
        "condition": cfg["condition"],
        "training_objective": "unsupervised_clm_right_shifted_history",
        "config_path": str(config_path),
        "config_snapshot": cfg,
        "command": "python " + " ".join(sys.argv),
        "limit": args.limit,
        "max_steps": args.max_steps,
        "final_adapter_dir": str(final_dir),
        "metrics_jsonl": str(metrics_path),
        "log_history_json": str(log_hist_path),
        "n_train_examples": len(processed),
        "n_dropped_empty": dropped_empty,
        "n_truncated_at_max_seq_length": n_truncated,
        "n_supervised_tokens": total_tokens,
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
