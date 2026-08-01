#!/usr/bin/env python3
"""
Emit every Condor sub file for the Warm-Start User-LoRA round (7 tasks x 2 arms).

  arm         base_adapter_mode   eval invocation
  ---------   -----------------   ---------------------------------------------
  warm        "continue"          --adapter <user_warm> --base-adapter none
  coldmatch   "merge"             --adapter <user_cold> --base-adapter <per_task>

The eval asymmetry is NOT a mistake and is the single easiest thing to get wrong
here: under warm-start the user's adapter IS the task adapter (it was continued
from it), so stacking it on top of the Per-Task-LoRA would apply the task delta
twice. Under coldmatch the user's adapter is a separate zero-init delta and the
Per-Task-LoRA must be supplied as the base, exactly as in R5-PT7. The two arms
therefore land on different result stems (verified against eval_lamp.py's stem
construction, eval_lamp.py:671-707) and cannot collide.

No baseline sub is generated. Every task's baseline arm is already on disk and
is reused byte-identical:
  - 5 flat tasks: results/<task>_test_per_task_..._bm25k4_seed0_topK100.*
  - LaMP-2-news (27) / LaMP-4 (100): per-user files from PT3 / PT2

Infra guards baked into every generated GPU sub, per CLAUDE.md's standing notes
and the failures that motivated them:
  - `Machine != tyr1...` and `Machine != modi...` (CUDA-busy oversubscription
    and uncorrectable ECC; cost R8 83/100 jobs on its first submit)
  - `require_gpus = Capability >= 8.0 && Capability < 10.0` (ver4's PyTorch
    2.5.1+cu124 cannot target the sm_120 Blackwell nodes)
  - `LAMP_DIR=<abs>/data/lamp_time` on every eval sub — omitting it silently
    matched 0 of 2,500 records and still exited 0 in PT1's first smoke

Plan reference: experiments/2026-07-31-warm-start-user-lora-plan.md §Scaffolding 4.

Usage (CPU-only, instant):
    python condor/gen_warm_subs.py --all
    python condor/gen_warm_subs.py --task LaMP_3 --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = str(PROJECT_ROOT)
CONDOR_DIR = PROJECT_ROOT / "condor"
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"

IMAGE = "ghcr.io/gordofreemo/smollm3-train:ver4"
ARMS = ("warm", "coldmatch")

GPU_REQS = ('requirements          = UidDomain == "cs.uni-saarland.de" && '
            'Machine != "tyr1.hpc.uni-saarland.de" && '
            'Machine != "modi.hpc.uni-saarland.de"')
CPU_REQS = 'requirements          = UidDomain == "cs.uni-saarland.de"'
GPU_BLOCK = """gpus_minimum_memory     = 32000
gpus_minimum_capability = 8.0
require_gpus            = Capability >= 8.0 && Capability < 10.0
"""


class Task:
    def __init__(self, name, tag, k, pool, per_task_ckpt, trainer,
                 eval_pattern, metric, eval_mem):
        self.name = name
        self.tag = tag
        self.k = k
        self.pool = pool
        self.per_task_ckpt = per_task_ckpt
        self.per_task_tag = (per_task_ckpt.replace("train/checkpoints/", "")
                             .replace("/", "_"))
        self.trainer = trainer
        self.eval_pattern = eval_pattern   # "flat" | "grouped"
        self.metric = metric
        self.eval_mem = eval_mem


# eval_pattern is HARDCODED, not read from the pool JSON: LaMP_3's and LaMP_4's
# pool files predate the `eval_pattern` field and don't carry it, and those are
# exactly the two tasks whose shapes differ from each other (LaMP-3 flat at
# 1 test record/user, LaMP-4 grouped at 1-25). Verified against
# data/lamp_user_stats/<task>_user_records.json.
TASKS = {t.name: t for t in [
    Task("LaMP_1", "lamp1", 100, "LaMP_1_top100_users.json",
         "train/checkpoints/per_task_lamp1_1ep_seed0/final",
         "train/train_unsupervised_clm.py", "flat", "accuracy", "16G"),
    Task("LaMP_2_movies", "lamp2movies", 100, "LaMP_2_movies_top100_users.json",
         "train/checkpoints/per_task_lamp2_movies_1ep_seed0/final",
         "train/train.py", "flat", "accuracy", "16G"),
    Task("LaMP_2_news", "lamp2news", 27, "LaMP_2_news_top27_users.json",
         "train/checkpoints/per_task_lamp2_news_1ep_seed0/final",
         "train/train.py", "grouped", "accuracy", "16G"),
    Task("LaMP_3", "lamp3", 100, "LaMP_3_top100_users.json",
         "train/checkpoints/per_task_lamp3_1ep_seed0/final",
         "train/train.py", "flat", "mae", "24G"),
    Task("LaMP_4", "lamp4", 100, "LaMP_4_top100_users.json",
         "train/checkpoints/per_task_lamp4_1ep_seed0/final",
         "train/train.py", "grouped", "rouge1", "16G"),
    Task("LaMP_5", "lamp5", 100, "LaMP_5_top100_users.json",
         "train/checkpoints/per_task_lamp5_1ep_seed0/final",
         "train/train.py", "flat", "rouge1", "16G"),
    Task("LaMP_7", "lamp7", 100, "LaMP_7_top100_users.json",
         "train/checkpoints/per_task_lamp7_1ep_seed0/final",
         "train/train_unsupervised_clm.py", "flat", "rouge1", "16G"),
]}

# The three comparisons reported per task. `stack-r8` (PT1-PT7) rides along in
# the writeup table for continuity but needs no new job — it is already on disk.
COMPARISONS = [
    ("baseline", "coldmatch", "does a size-matched cold adapter help at all"),
    ("baseline", "warm", "does warm-start help vs no personalization"),
    ("coldmatch", "warm", "THE round's primary claim: warm vs size-matched cold"),
]


def header(title, body, submit_cmd):
    return (
        f"# {title}\n#\n"
        + "".join(f"# {line}\n" if line else "#\n" for line in body)
        + "#\n"
        + "# GENERATED by condor/gen_warm_subs.py — edit that script and\n"
        + "# regenerate rather than hand-editing this file, or the next\n"
        + "# regeneration will silently drop your change.\n"
        + "#\n"
        + "# Submit (after committing this file):\n"
        + f"#   {submit_cmd}\n\n"
    )


def logs(stem):
    return (
        f"output                = {ROOT}/runlogs/{stem}.$(ClusterId).$(ProcId).out\n"
        f"error                 = {ROOT}/runlogs/{stem}.$(ClusterId).$(ProcId).err\n"
        f"log                   = {ROOT}/runlogs/{stem}.$(ClusterId).log\n\n"
    )


def pool_users(task: Task):
    p = USER_STATS_DIR / task.pool
    if not p.exists():
        sys.exit(f"ERROR: missing pool {p}.")
    return json.loads(p.read_text())["users"]


def smallest_user(task: Task):
    """Smoke subject: smallest profile in the pool — fastest to train, and the
    most likely to expose an empty-corpus edge case."""
    return min(pool_users(task), key=lambda u: u["profile_size"])["user_fingerprint"]


def user_adapter(task: Task, fp: str, arm: str) -> str:
    return f"{ROOT}/train/checkpoints/user_lora_{task.tag}_{fp}_{arm}_seed0/final"


# --- training ----------------------------------------------------------------
def sub_train(task: Task, arm: str, smoke: bool):
    stem = f"train_warm_{task.tag}_{arm}" + ("_smoke" if smoke else "")
    fps = ([smallest_user(task)] if smoke
           else [u["user_fingerprint"] for u in pool_users(task)])
    cfgs = [f"{ROOT}/train/config/user_lora_{task.tag}_{fp}_{arm}.json" for fp in fps]

    if arm == "warm":
        what = [
            "WARM arm: base_adapter_mode='continue'. The Per-Task-LoRA is loaded",
            "with is_trainable=True and CONTINUED on this user's data — it is the",
            "thing being trained. Both LoRA matrices start non-zero, so both take",
            "real gradients from step one.",
            "",
            "is_trainable=True is load-bearing: the saved adapter_config.json",
            "carries \"inference_mode\": true, and an ordinary load returns a",
            "FROZEN adapter that would train nothing while still exiting 0.",
        ]
    else:
        what = [
            "COLDMATCH arm (the CONTROL): base_adapter_mode='merge'. The",
            "Per-Task-LoRA is merged into the frozen backbone and a fresh",
            "zero-initialized adapter is trained on top — the R5-PT7 mechanism,",
            "but at the Per-Task-LoRA's own shape (r=4, 7 weight types) instead",
            "of OPPU's r=8 q+v. Matching rank/modules/LR/epochs to the warm arm",
            "is what makes a warm win attributable to the starting point rather",
            "than to 4x the parameters.",
        ]

    return stem, (
        header(
            f"Warm-Start round — {task.name} {arm.upper()} arm"
            + (" — SMOKE (1 user)." if smoke else f" (K={len(fps)} parallel GPU procs)."),
            what + [
             "",
             f"Base: {task.per_task_ckpt}",
             f"Trainer: {task.trainer}",
             "",
             "Recipe pinned (do NOT edit): LR=1e-5, weight_decay=1e-2, 3 epochs,",
             "cosine + 3% warmup, per_device_batch=2, grad_accum=4. Configs come",
             "from data/lamp_user_stats/warm_gen_configs.py; max_seq_length is",
             "this task's own T3 value.",
             "",
             "Three guards run inside the trainer and ABORT on failure:",
             "  1. in 'continue' mode the config's lora block must match the",
             "     adapter on disk (else an r=8 config would silently train r=4)",
             "  2. train_meta.json records the shape actually read off disk",
             "  3. trainable weights are hashed before and after — an unchanged",
             "     hash fails the job rather than saving a fake clean null",
             "",
             "Acceptance per user: train_meta.json with",
             f"  base_adapter_mode={'continue' if arm == 'warm' else 'merge'},",
             "  loaded_adapter_r=4 (warm only), differing weight hashes,",
             "  non-NaN final loss.",
             "",
             "If a task OOMs, the documented fallback is per_device 1 /",
             "grad_accum 8 (coldmatch/warm carry ~4x the trainable params of the",
             "old r=8 q+v arm). LaMP-3 at max_seq_length 7168 is the likeliest.",
            ]
            + ([f"", f"SMOKE SUBJECT: {fps[0]} (smallest profile in the pool).",
                "Check the acceptance criteria before the full batch."]
               if smoke else
               ["", f"Run condor/train_warm_{task.tag}_{arm}_smoke.sub first.",
                "The smoke user recurs here and will hit one expected",
                "refuse-to-overwrite failure."]),
            f"condor_submit condor/{stem}.sub",
        )
        + f"universe              = docker\ndocker_image          = {IMAGE}\n"
        + f"executable            = {task.trainer}\n"
        + "arguments             = --config $(config_path)\n\n"
        + logs(stem)
        + "should_transfer_files = YES\n"
        + f'environment           = "PROJECT_ROOT={ROOT} '
          f'CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"\n\n'
        + GPU_BLOCK
        + "\nmax_job_retirement_time = 28800\n\n"
        + "stream_output           = true\nstream_error            = true\n"
          "when_to_transfer_output = ON_EXIT_OR_EVICT\n\n"
        + "request_GPUs          = 1\nrequest_CPUs          = 4\n"
          "request_memory        = 32G\n"
        + GPU_REQS + "\n+WantGPUHomeMounted   = true\n+WantScratchMounted   = true\n\n"
        + "queue config_path from (\n"
        + "".join(f"  {c}\n" for c in cfgs)
        + ")\n"
    )


# --- eval --------------------------------------------------------------------
def sub_eval(task: Task, arm: str, smoke: bool):
    stem = f"eval_warm_{task.tag}_{arm}" + ("_smoke" if smoke else "")
    fps = ([smallest_user(task)] if smoke
           else [u["user_fingerprint"] for u in pool_users(task)])
    # THE asymmetry: warm carries the task delta inside itself, coldmatch does not.
    base_arg = "none" if arm == "warm" else f"{ROOT}/{task.per_task_ckpt}"
    rows = [(user_adapter(task, fp, arm), base_arg, fp) for fp in fps]

    stem_note = (
        [f"Result stem: {task.name}_test_user_lora_{task.tag}_<user>_warm_seed0_final"
         "_bm25k4_seed0_user<user>"]
        if arm == "warm" else
        [f"Result stem: {task.name}_test_{task.per_task_tag}_user_lora_{task.tag}"
         "_<user>_coldmatch_seed0_final_bm25k4_seed0_user<user>"]
    )

    return stem, (
        header(
            f"Warm-Start round eval — {task.name} {arm.upper()} arm"
            + (" — SMOKE (1 user)." if smoke else f" ({len(rows)} procs)."),
            ([
             "--base-adapter is 'none' for this arm ON PURPOSE. The warm adapter",
             "WAS the Per-Task-LoRA and was continued from it, so supplying the",
             "Per-Task-LoRA as a base would apply the task delta twice."
             ] if arm == "warm" else [
             "--base-adapter is the Per-Task-LoRA, as in every round R5-PT7: this",
             "arm's user adapter is a separate zero-init delta and needs the task",
             "adapter underneath it."
             ]) + [
             "",
             "No baseline job here — the baseline arm is already on disk and is",
             "reused byte-identical ("
             + ("topK100 batch file" if task.eval_pattern == "flat"
                else f"{task.k} per-user files")
             + ").",
             "",
             ] + stem_note + [
             "",
             "LAMP_DIR is set to the TIME-based split below and is load-bearing:",
             "without it eval_lamp.py silently matches 0 records and still exits",
             "0 (the bug PT1's first smoke eval hit). After this runs, check the",
             "record COUNTS, not just exit codes.",
             "",
             "eval_lamp.py refuses to overwrite, so re-submits only fill gaps."]
            + ([f"", f"SMOKE SUBJECT: {fps[0]}."] if smoke else
               [f"", f"Run condor/{stem}_smoke.sub first, and only after the",
                f"training batch has produced all {task.k} adapters."]),
            f"condor_submit condor/{stem}.sub",
        )
        + f"universe              = docker\ndocker_image          = {IMAGE}\n"
        + "executable            = eval/eval_lamp.py\n\n"
        + "# Macros driven by the queue block. eval_lamp.py treats \"none\" as "
          "\"no adapter\".\n"
        + "adapter      = none\nbase_adapter = none\nuser_fp      = u00000000\n\n"
        + f"arguments             = --task {task.name} --split test --k 4 --seed 0 "
          f"--adapter $(adapter) --base-adapter $(base_adapter) "
          f"--user-records $(user_fp)\n\n"
        + logs(stem)
        + "should_transfer_files = YES\n"
        + f'environment           = "MODEL_OUT_DIR={ROOT}/data/models/SmolLM3-3B '
          f'LAMP_DIR={ROOT}/data/lamp_time '
          f'CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"\n\n'
        + "max_job_retirement_time = 600\n"
          "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
          "stream_output           = true\nstream_error            = true\n\n"
        + GPU_BLOCK
        + f"\nrequest_GPUs          = 1\nrequest_CPUs          = 2\n"
          f"request_memory        = {task.eval_mem}\n"
        + GPU_REQS + "\n+WantGPUHomeMounted   = true\n+WantScratchMounted   = true\n\n"
        + "queue adapter, base_adapter, user_fp from (\n"
        + "".join(f"  {a}, {b}, {u}\n" for a, b, u in rows)
        + ")\n"
    )


# --- paired comparison -------------------------------------------------------
def sub_paired(task: Task, arm_a: str, arm_b: str, why: str):
    stem = f"paired_warm_{task.tag}_{arm_a}_vs_{arm_b}"
    grouped = task.eval_pattern == "grouped"
    exe = "eval/paired_compare_per_user.py" if grouped else "eval/paired_compare.py"
    pa = f"{ROOT}/results/{task.name}_test_warmround_{arm_a}.predictions.jsonl"
    pb = f"{ROOT}/results/{task.name}_test_warmround_{arm_b}.predictions.jsonl"
    extra = f" --round-tag warm_{arm_a}_vs_{arm_b}" if grouped else ""
    return stem, (
        header(
            f"Warm-Start round paired comparison — {task.name}, "
            f"{arm_a} vs {arm_b}, metric={task.metric}.",
            [f"Purpose: {why}.",
             "",
             ] + ([
             "This task's users hold multiple test records, so the GROUPED",
             "per-user comparison applies: per-record scores are averaged within",
             "a user first, then the paired-t / Wilcoxon / bootstrap battery runs",
             f"over the {task.k} per-user means — not over records, which would",
             "over-weight heavy users and break independence."]
             if grouped else [
             "Every user holds exactly 1 test record, so record-level pairing IS",
             "user-level pairing and the flat paired_compare.py applies."])
            + ([
             "",
             "metric=mae is LOWER-is-better. Read wins_b_better / wins_a_better",
             "(schema v2), NOT wins_b_over_a / wins_a_over_b — the latter are raw",
             "sign counts kept for continuity with R5/R8/PT1's on-disk results",
             "and read BACKWARDS for MAE."] if task.metric == "mae" else [])
            + [
             "",
             "Descriptive reporting only — no pre-registered gate, matching every",
             "round since R6. NOTE: this round runs 21 such comparisons (7 tasks",
             "x 3) with no correction and no pooling, so a lone p<0.05 here means",
             "no more than R10's did.",
             "",
             "Run AFTER eval/aggregate_user_predictions_warm.py has built all",
             "three consolidated prediction files for this task."],
            f"condor_submit condor/{stem}.sub",
        )
        + f"universe              = docker\ndocker_image          = {IMAGE}\n"
        + f"executable            = {exe}\n\n"
        + f"arguments             = --pred-a {pa} --pred-b {pb} "
          f"--label-a {arm_a} --label-b {arm_b} "
          f"--task {task.name} --split test --metric {task.metric}{extra} --seed 0\n\n"
        + logs(stem)
        + "should_transfer_files = YES\n"
        + f'environment           = "PROJECT_ROOT={ROOT} '
          f'CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"\n\n'
        + "request_GPUs          = 0\nrequest_CPUs          = 1\n"
          "request_memory        = 2G\n\n"
        + "max_job_retirement_time = 600\n"
          "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
          "stream_output           = true\nstream_error            = true\n\n"
        + CPU_REQS + "\n+WantGPUHomeMounted   = true\n+WantScratchMounted   = true\n\n"
        + "queue 1\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.all:
        names = sorted(TASKS)
    elif args.task:
        names = [args.task]
    else:
        sys.exit("ERROR: pass --task <name> or --all.")

    out = {}
    for name in names:
        task = TASKS[name]
        for arm in ARMS:
            for smoke in (True, False):
                for gen in (sub_train, sub_eval):
                    stem, text = gen(task, arm, smoke)
                    out[stem] = text
        for arm_a, arm_b, why in COMPARISONS:
            stem, text = sub_paired(task, arm_a, arm_b, why)
            out[stem] = text

    existing = [s for s in out if (CONDOR_DIR / f"{s}.sub").exists()]
    if existing and not args.overwrite:
        sys.exit(f"ERROR: refusing to overwrite {len(existing)} existing subs "
                 f"(first: {existing[0]}.sub). Pass --overwrite.")

    for stem, text in sorted(out.items()):
        (CONDOR_DIR / f"{stem}.sub").write_text(text)
    print(f"[gen] wrote {len(out)} sub files to {CONDOR_DIR}", flush=True)
    for stem in sorted(out):
        print(f"        condor/{stem}.sub")


if __name__ == "__main__":
    main()
