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


# --- base_adapter_mode support ----------------------------------------------
# Byte-duplicated from train/train.py, per this script's standing convention of
# duplicating rather than deep-branching a shared path. If you change the guards
# here, change them there too.
#
# "merge"    (default, every round R5-PT7): merge the base adapter into the
#            frozen backbone, then attach a fresh zero-initialized adapter.
# "continue" (warm-start): load the base adapter TRAINABLE and keep training it.
BASE_ADAPTER_MODES = ("merge", "continue")


def _read_adapter_config(adapter_path: str) -> dict:
    p = Path(adapter_path) / "adapter_config.json"
    if not p.exists():
        sys.exit(f"ERROR: {p} does not exist — not a PEFT adapter directory.")
    return json.loads(p.read_text())


def _assert_adapter_shape_matches(cfg_lora: dict, disk_cfg: dict, adapter_path: str):
    """Guard 1. In "continue" mode the config's `lora` block is INERT — rank,
    alpha and target modules all come from the adapter on disk. Refuse on
    mismatch, or leaving OPPU's `r: 8` in a config while warm-starting an r=4
    adapter would silently train r=4 and look entirely successful."""
    checks = [("r", "r"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")]
    mismatches = []
    for cfg_key, disk_key in checks:
        if cfg_key in cfg_lora and cfg_lora[cfg_key] != disk_cfg.get(disk_key):
            mismatches.append(
                f"{cfg_key}: config={cfg_lora[cfg_key]!r} disk={disk_cfg.get(disk_key)!r}"
            )
    if "target_modules" in cfg_lora:
        want = set(cfg_lora["target_modules"])
        have = set(disk_cfg.get("target_modules") or [])
        if want != have:
            mismatches.append(
                f"target_modules: config={sorted(want)} disk={sorted(have)}"
            )
    if mismatches:
        sys.exit(
            "ERROR: base_adapter_mode='continue' but the config's `lora` block "
            f"disagrees with the adapter on disk ({adapter_path}):\n"
            + "".join(f"  - {m}\n" for m in mismatches)
            + "  In 'continue' mode the on-disk adapter wins and the config's\n"
            "  `lora` block is inert, so a mismatch means the config is lying\n"
            "  about what is being trained. Fix the config to match, or drop\n"
            "  the `lora` block entirely."
        )


def _lora_weight_hash(model) -> str:
    """Guard 3 helper. SHA-256 over every TRAINABLE tensor, in sorted name
    order — used to prove training actually moved the adapter.

    The failure this exists to catch: a saved `adapter_config.json` carries
    `"inference_mode": true`, so loading it without `is_trainable=True` yields
    a frozen adapter. Training then runs, reports a loss, saves, and exits 0 —
    having changed nothing. Across hundreds of runs that manufactures a clean,
    plausible, entirely fake null.
    """
    import hashlib

    h = hashlib.sha256()
    n_trainable = 0
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if not param.requires_grad:
            continue
        h.update(name.encode())
        h.update(param.detach().to(torch.float32).cpu().numpy().tobytes())
        n_trainable += 1
    if n_trainable == 0:
        sys.exit(
            "ERROR: the model has ZERO trainable parameters — training would "
            "do nothing and still exit 0. In 'continue' mode this means "
            "is_trainable=True did not take effect (the adapter_config.json "
            "carries inference_mode: true)."
        )
    return h.hexdigest()


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
    base_adapter_mode = cfg.get("base_adapter_mode", "merge")
    base_adapter_tag = None
    loaded_adapter_shape = None
    if base_adapter_mode not in BASE_ADAPTER_MODES:
        sys.exit(f"ERROR: base_adapter_mode must be one of {BASE_ADAPTER_MODES}, "
                 f"got {base_adapter_mode!r}.")
    if base_adapter_mode == "continue" and not base_adapter_path:
        sys.exit("ERROR: base_adapter_mode='continue' requires `base_adapter` — "
                 "there is nothing to continue training from.")

    use_gc = cfg["trainer"].get("gradient_checkpointing", False)

    if base_adapter_path:
        base_adapter_path = str(_resolve(base_adapter_path))
        base_adapter_tag = _derive_adapter_tag(base_adapter_path)

    if base_adapter_mode == "continue":
        # Warm start: the base adapter is loaded TRAINABLE and becomes the thing
        # being trained. No merge, no second adapter.
        #
        # NOTE for LaMP-1/LaMP-7 specifically: the Task-LoRA being continued here
        # was trained with the chat template and loss masked to assistant tokens,
        # while this script trains on raw untemplated text with loss on every
        # token. Under 'merge' that mismatch was structurally harmless — the task
        # adapter was frozen in the backbone and could not be damaged. Under
        # 'continue' the raw-text objective rewrites it directly. This is a
        # pre-registered risk, not an oversight; watch the parse-failure rate.
        disk_cfg = _read_adapter_config(base_adapter_path)
        _assert_adapter_shape_matches(cfg.get("lora") or {}, disk_cfg,
                                      base_adapter_path)
        if use_gc:
            model.enable_input_require_grads()
        print(f"[model] CONTINUING base adapter {base_adapter_tag} from "
              f"{base_adapter_path} (is_trainable=True)", flush=True)
        model = PeftModel.from_pretrained(model, base_adapter_path,
                                          is_trainable=True)
        loaded_adapter_shape = {
            "r": disk_cfg.get("r"),
            "lora_alpha": disk_cfg.get("lora_alpha"),
            "lora_dropout": disk_cfg.get("lora_dropout"),
            "target_modules": sorted(disk_cfg.get("target_modules") or []),
        }
        print(f"[model] loaded adapter shape (read off disk): "
              f"r={loaded_adapter_shape['r']} "
              f"alpha={loaded_adapter_shape['lora_alpha']} "
              f"target_modules={loaded_adapter_shape['target_modules']}",
              flush=True)
    else:
        if base_adapter_path:
            print(f"[model] merging base adapter {base_adapter_tag} from "
                  f"{base_adapter_path}", flush=True)
            model = PeftModel.from_pretrained(model, base_adapter_path)
            model = model.merge_and_unload()

        if use_gc:
            model.enable_input_require_grads()

        lora_config = LoraConfig(**cfg["lora"])
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Guard 3 (first half): fingerprint the trainable tensors before training.
    lora_hash_before = _lora_weight_hash(model)
    print(f"[model] trainable-weight hash before training: {lora_hash_before[:16]}",
          flush=True)

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

    # Guard 3 (second half): the adapter MUST have moved. An unchanged hash
    # means the run trained nothing — otherwise invisible, since loss is still
    # logged and the exit code is still 0.
    lora_hash_after = _lora_weight_hash(trainer.model)
    if lora_hash_after == lora_hash_before:
        sys.exit(
            "ERROR: trainable weights are byte-identical before and after "
            f"training (hash {lora_hash_before[:16]}). Nothing was learned.\n"
            f"  base_adapter_mode={base_adapter_mode}\n"
            "  In 'continue' mode the usual cause is the adapter loading "
            "frozen (adapter_config.json carries inference_mode: true, so "
            "is_trainable=True must be passed explicitly).\n"
            "  Refusing to save a checkpoint that would look like a clean null."
        )
    print(f"[model] trainable-weight hash after training:  {lora_hash_after[:16]} "
          f"(changed — OK)", flush=True)

    # --- Save final adapter + run metadata ---------------------------------
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metric_summary = summarize_log_history(trainer.state.log_history)

    log_hist_path = output_dir / "log_history.json"
    log_hist_path.write_text(json.dumps(trainer.state.log_history, indent=2))

    meta = {
        # v2 adds base_adapter_mode + loaded_adapter_* + lora_weight_hash_*.
        "schema_version": 2,
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
        # Guard 2: the mode that ran, and the shape actually read off disk in
        # 'continue' mode (None under 'merge', where cfg["lora"] governs).
        "base_adapter_mode": base_adapter_mode,
        "loaded_adapter_r": (loaded_adapter_shape or {}).get("r"),
        "loaded_adapter_lora_alpha": (loaded_adapter_shape or {}).get("lora_alpha"),
        "loaded_adapter_target_modules": ",".join(
            (loaded_adapter_shape or {}).get("target_modules") or []
        ) or None,
        # Guard 3 evidence.
        "lora_weight_hash_before": lora_hash_before,
        "lora_weight_hash_after": lora_hash_after,
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
