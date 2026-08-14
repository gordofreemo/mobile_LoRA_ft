#!/usr/bin/env python3
"""
Generate the Warm-Start User-LoRA training configs — 7 LaMP tasks x 2 arms.

  arm         base_adapter_mode   what the user trains
  ---------   -----------------   ----------------------------------------
  warm        "continue"          a PRIVATE COPY of that task's Per-Task-LoRA,
                                  loaded trainable and continued on the user's
                                  own data. Both LoRA matrices start non-zero.
  coldmatch   "merge"             a fresh ZERO-INITIALIZED adapter on top of the
                                  merged Per-Task-LoRA, at the SAME shape the
                                  warm arm inherits (r=4, alpha=8, 7 modules).

`coldmatch` is the control, and the reason it exists: without it a warm win
can't be told apart from "we gave it 4x the parameters across 5 more weight
types" (the existing PT1-PT7 rounds ran r=8 q+v only). Matching rank, modules,
LR and epochs leaves exactly one difference between the arms — whether the task
knowledge sits in trainable parameters or in the frozen backbone.

Base is the Per-Task-LoRA for every task; there is NO One-LoRA FT track this
round (that would double 1,254 trainings to 2,508, and warm-starting a 7-task
adapter on one task's user data mostly measures how fast it collapses to one
task).

Everything except the two arm-specific keys is held at the OPPU recipe every
User-LoRA round since R5 has used: LR=1e-5, weight_decay=1e-2, 3 epochs,
cosine + 3% warmup, per_device_batch=2, grad_accum=4. LR is deliberately NOT
varied (a cold@1e-4 arm was proposed and declined) so the numbers stay
comparable with every personalization result in the project.

Two things are read off disk rather than hardcoded, on purpose:
  - `max_seq_length` comes from each task's own T3 sizing JSON. Never copied
    between tasks: LaMP-3 needs 7168 and LaMP-7 needs 256.
  - the `lora` block (r / alpha / dropout / target_modules) is read from the
    Per-Task-LoRA's own adapter_config.json, so the generated configs cannot
    drift from the adapter they are meant to match. Both trainers refuse to run
    in "continue" mode if the two ever disagree.

Writes:
  train/config/warm_user_lora_<tag>_<arm>_template.json   (committed record)
  train/config/user_lora_<tag>_<user>_<arm>.json          (gitignored, 1,254)

Plan reference: experiments/2026-07-31-warm-start-user-lora-plan.md
  §Scaffolding items 2-3, plus Revision 8 (the 21-user save_total_limit=3 sample).

Usage (CPU-only, ~1s per task):
    python data/lamp_user_stats/warm_gen_configs.py --task LaMP_3
    python data/lamp_user_stats/warm_gen_configs.py --all --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"
CONFIG_DIR = PROJECT_ROOT / "train" / "config"

ARMS = ("warm", "coldmatch")

# task -> (tag, pool JSON, corpus suffix, Per-Task-LoRA ckpt, T3 sizing JSON,
#          trainer script)
#
# The T3 filenames are NOT uniform: LaMP-3 and LaMP-4 were sized during R5/R6
# and keep those rounds' names. Reading the wrong one would silently mis-size
# the run, so each is named explicitly rather than derived from the task.
TASKS = {
    "LaMP_1": (
        "lamp1", "LaMP_1_top100_users.json", "unsup",
        "train/checkpoints/per_task_lamp1_1ep_seed0/final",
        "LaMP_1_t3_sizing.json", "train/train_unsupervised_clm.py",
    ),
    "LaMP_2_movies": (
        "lamp2movies", "LaMP_2_movies_top100_users.json", "bm25k4",
        "train/checkpoints/per_task_lamp2_movies_1ep_seed0/final",
        "LaMP_2_movies_t3_sizing.json", "train/train.py",
    ),
    "LaMP_2_news": (
        "lamp2news", "LaMP_2_news_top27_users.json", "records_bm25k4",
        "train/checkpoints/per_task_lamp2_news_1ep_seed0/final",
        "LaMP_2_news_t3_sizing.json", "train/train.py",
    ),
    "LaMP_3": (
        "lamp3", "LaMP_3_top100_users.json", "bm25k4",
        "train/checkpoints/per_task_lamp3_1ep_seed0/final",
        "LaMP_3_round5_t3_sizing.json", "train/train.py",
    ),
    "LaMP_4": (
        "lamp4", "LaMP_4_top100_users.json", "bm25k4",
        "train/checkpoints/per_task_lamp4_1ep_seed0/final",
        "LaMP_4_round6_t3_sizing.json", "train/train.py",
    ),
    "LaMP_5": (
        "lamp5", "LaMP_5_top100_users.json", "bm25k4",
        "train/checkpoints/per_task_lamp5_1ep_seed0/final",
        "LaMP_5_t3_sizing.json", "train/train.py",
    ),
    "LaMP_7": (
        "lamp7", "LaMP_7_top100_users.json", "unsup",
        "train/checkpoints/per_task_lamp7_1ep_seed0/final",
        "LaMP_7_t3_sizing.json", "train/train_unsupervised_clm.py",
    ),
}

# Revision 8: this many users per task keep every epoch checkpoint
# (save_total_limit=3) instead of only the last. Under warm-start, extra epochs
# mean drift away from a converged adapter rather than growth from zero, so
# without an intermediate checkpoint "warm didn't help" and "warm helped at
# epoch 1 and drifted by epoch 3" are indistinguishable. 3 users x 7 tasks x
# 2 arms is ~6 GB and costs no GPU time.
N_CKPT_SAMPLE_PER_TASK = 3


def ckpt_sample(users: list) -> list:
    """Smallest / median / largest profile in the pool — deterministic, and
    spans the range where warm-vs-cold behaviour should differ most (the
    smallest profiles are where cold-start is closest to a no-op)."""
    ordered = sorted(users, key=lambda u: (u["profile_size"], u["user_fingerprint"]))
    if len(ordered) <= N_CKPT_SAMPLE_PER_TASK:
        return [u["user_fingerprint"] for u in ordered]
    picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    return [u["user_fingerprint"] for u in picks]


def template_doc(task, arm, tag, base_adapter, max_seq_length, trainer,
                 lora_block) -> str:
    if arm == "warm":
        what = (
            "The Per-Task-LoRA is loaded with is_trainable=True and CONTINUED "
            "on this user's data — it is the thing being trained, not a frozen "
            "backbone. Final weights are W + dW_user, started from dW_task. "
            "The `lora` block below is INERT in this mode (shape comes from the "
            "adapter on disk); it is recorded so the trainer's shape guard has "
            "something to check and so this file states what actually trains."
        )
    else:
        what = (
            "The Per-Task-LoRA is merged into the frozen backbone and a fresh "
            "ZERO-INITIALIZED adapter is trained on top — the same mechanism "
            "every round R5-PT7 used, but at the Per-Task-LoRA's own shape "
            "(r=4, alpha=8, 7 weight types) rather than OPPU's r=8 q+v. This is "
            "the CONTROL for the warm arm: same rank, same modules, same LR, "
            "same epochs, so the only remaining difference is whether task "
            "knowledge sits in trainable parameters or in the frozen backbone."
        )
    return (
        f"Warm-Start User-LoRA, {arm.upper()} arm — per-user adapter for {task}, "
        f"base Per-Task-LoRA({task}) ({base_adapter}). {what} "
        f"Trained by {trainer}. Recipe held at the OPPU hyperparameters used by "
        f"every User-LoRA round since R5 (LR=1e-5, weight_decay=1e-2, 3 epochs, "
        f"cosine + 3% warmup, per_device_batch=2, grad_accum=4); LR is "
        f"deliberately not varied. lora shape r={lora_block['r']} "
        f"alpha={lora_block['lora_alpha']} over {len(lora_block['target_modules'])} "
        f"weight types, read off the base adapter's own adapter_config.json. "
        f"max_seq_length={max_seq_length} comes from this task's own T3 sizing "
        f"pass, NOT copied from another task. Per-user instantiations live at "
        f"train/config/user_lora_{tag}_<user>_{arm}.json, generated by "
        f"data/lamp_user_stats/warm_gen_configs.py (gitignored)."
    )


def build_template(task, arm, tag, base_adapter, max_seq_length, trainer,
                   lora_block) -> dict:
    return {
        "_doc": template_doc(task, arm, tag, base_adapter, max_seq_length,
                             trainer, lora_block),
        "condition": "<<CONDITION>>",
        "seed": 0,
        "model_name_or_path": "data/models/SmolLM3-3B",
        "base_adapter": base_adapter,
        # The one key that differs between the two arms.
        "base_adapter_mode": "continue" if arm == "warm" else "merge",
        "dataset_path": "<<DATASET_PATH>>",
        "output_dir": "<<OUTPUT_DIR>>",
        "lora": lora_block,
        "data": {"max_seq_length": max_seq_length},
        "trainer": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "bf16": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "logging_steps": 1,
            "optim": "adamw_torch",
            "report_to": "none",
            "remove_unused_columns": False,
            "dataloader_num_workers": 2,
        },
        "wandb": {
            "_doc": "Disabled — report_to above is 'none'.",
            "project": "mobileFT_distill",
            "run_name": "<<CONDITION>>",
        },
    }


def gen_task(task: str, overwrite: bool) -> int:
    tag, pool_file, suffix, base_adapter, t3_file, trainer = TASKS[task]

    # --- lora shape, read off the base adapter itself ----------------------
    adapter_cfg_path = PROJECT_ROOT / base_adapter / "adapter_config.json"
    if not adapter_cfg_path.exists():
        sys.exit(f"ERROR: base adapter {base_adapter} has no adapter_config.json.")
    disk = json.loads(adapter_cfg_path.read_text())
    lora_block = {
        "r": disk["r"],
        "lora_alpha": disk["lora_alpha"],
        "lora_dropout": disk.get("lora_dropout", 0.05),
        # sorted so the generated configs are byte-stable across runs (PEFT
        # serializes target_modules from a set, so disk order is arbitrary)
        "target_modules": sorted(disk["target_modules"]),
        "bias": disk.get("bias", "none"),
        "task_type": disk.get("task_type", "CAUSAL_LM"),
    }
    if lora_block["r"] != 4 or len(lora_block["target_modules"]) != 7:
        sys.exit(
            f"ERROR: {base_adapter} is r={lora_block['r']} over "
            f"{len(lora_block['target_modules'])} weight types; every "
            f"Per-Task-LoRA in this round is expected to be r=4 over 7. "
            f"Check the checkpoint before generating configs."
        )

    # --- max_seq_length from this task's own T3 pass -----------------------
    t3_path = USER_STATS_DIR / t3_file
    if not t3_path.exists():
        sys.exit(f"ERROR: missing {t3_path} — max_seq_length must be measured "
                 f"on this task's corpora, never copied from another round.")
    max_seq_length = json.loads(t3_path.read_text()).get("max_seq_length_pinned")
    if not max_seq_length:
        sys.exit(f"ERROR: {t3_path} has max_seq_length_pinned=null.")

    # --- pool --------------------------------------------------------------
    pool_path = USER_STATS_DIR / pool_file
    if not pool_path.exists():
        sys.exit(f"ERROR: missing pool {pool_path}.")
    pool_users = json.loads(pool_path.read_text())["users"]
    fps = [u["user_fingerprint"] for u in pool_users]
    keep_all_epochs = set(ckpt_sample(pool_users))

    written = 0
    missing_corpora = []
    for arm in ARMS:
        template = build_template(task, arm, tag, base_adapter, max_seq_length,
                                  trainer, lora_block)

        template_path = CONFIG_DIR / f"warm_user_lora_{tag}_{arm}_template.json"
        if template_path.exists() and not overwrite:
            sys.exit(f"ERROR: refusing to overwrite {template_path}. Pass --overwrite.")
        template_path.write_text(json.dumps(template, indent=2) + "\n")

        out_paths = {fp: CONFIG_DIR / f"user_lora_{tag}_{fp}_{arm}.json" for fp in fps}
        existing = [p for p in out_paths.values() if p.exists()]
        if existing and not overwrite:
            sys.exit(f"ERROR: refusing to overwrite {len(existing)} existing "
                     f"configs (first: {existing[0]}). Pass --overwrite.")

        for fp in fps:
            cfg = json.loads(json.dumps(template))  # deep copy
            cfg["condition"] = f"user_lora_{tag}_{fp}_{arm}"
            cfg["dataset_path"] = f"data/lamp_user_train_{task}_{fp}_{suffix}.jsonl"
            cfg["output_dir"] = f"train/checkpoints/user_lora_{tag}_{fp}_{arm}_seed0"
            cfg["wandb"]["run_name"] = cfg["condition"]
            if fp in keep_all_epochs:
                cfg["trainer"]["save_total_limit"] = 3

            # Spot-asserts — the arm-defining keys and the pinned recipe.
            assert cfg["base_adapter_mode"] == (
                "continue" if arm == "warm" else "merge"), cfg["base_adapter_mode"]
            assert cfg["base_adapter"] == base_adapter, cfg["base_adapter"]
            assert cfg["lora"]["r"] == 4, cfg["lora"]
            assert len(cfg["lora"]["target_modules"]) == 7, cfg["lora"]
            assert cfg["trainer"]["learning_rate"] == 1e-5, cfg["trainer"]
            assert cfg["trainer"]["num_train_epochs"] == 3, cfg["trainer"]
            assert cfg["data"]["max_seq_length"] == max_seq_length

            if not (PROJECT_ROOT / cfg["dataset_path"]).exists():
                missing_corpora.append(fp)
            out_paths[fp].write_text(json.dumps(cfg, indent=2) + "\n")
            written += 1

        print(f"[gen] {task} {arm}: template -> {template_path.name}; "
              f"{len(fps)} per-user configs "
              f"(max_seq_length={max_seq_length}, r={lora_block['r']}, "
              f"{len(lora_block['target_modules'])} modules, "
              f"{len(keep_all_epochs)} keeping all 3 epoch ckpts)", flush=True)

    if missing_corpora:
        uniq = sorted(set(missing_corpora))
        print(f"[warn] {len(uniq)} users have no corpus on disk "
              f"(e.g. {uniq[:3]}) — configs written anyway, but training will "
              f"fail for them until the corpora exist.", flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--all", action="store_true", help="all 7 tasks")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.all:
        tasks = sorted(TASKS)
    elif args.task:
        tasks = [args.task]
    else:
        sys.exit("ERROR: pass --task <name> or --all.")

    total = sum(gen_task(t, args.overwrite) for t in tasks)
    print(f"[gen] done: {total} per-user configs across {len(tasks)} task(s) "
          f"x {len(ARMS)} arms", flush=True)


if __name__ == "__main__":
    main()
