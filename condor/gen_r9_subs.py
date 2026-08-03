#!/usr/bin/env python3
"""
Emit all five Condor sub files for R9 (LaMP-4 User-LoRA on One-LoRA FT).

R9 closes the one gap in the R-track: R8 covers LaMP-3 and R10-R14 cover the
five late tasks on the One-LoRA FT base, but LaMP-4 was pinned 2026-07-20
(experiments/2026-07-20-user-lora-round9-lamp4-onelora-plan.md) and never
built. It is the same mechanical base-adapter swap R8 was, applied to R6's
LaMP-4 pool.

Generated files (all overwritten in place — EDIT THIS SCRIPT, NOT THE .sub
FILES, same convention as condor/gen_newtask_subs.py and gen_warm_subs.py):

    condor/train_user_lora_lamp4_r9_smoke.sub   1 proc   (smoke user only)
    condor/train_user_lora_lamp4_r9.sub       100 procs  (K=100 training)
    condor/eval_lamp_user_lamp4_r9_smoke.sub    2 procs  (both arms, smoke user)
    condor/eval_lamp_user_lamp4_r9.sub        200 procs  (100 C2 + 100 C3)
    condor/paired_compare_lamp4_r9.sub          1 proc   (CPU-only stats)

Structure mirrors the PT2 subs (condor/*_lamp4_ptl.sub) exactly; the only
deltas are the base adapter (One-LoRA FT instead of Per-Task-LoRA(LaMP-4))
and the `_r9` suffix on every config/checkpoint/runlog path, so R6's `_oppu`
and PT2's `_ptl` artifacts are never touched.

Both arms carry the guards this project learned the hard way:
  - tyr1/modi excluded (R8's 83/100 failure batch: oversubscription + ECC)
  - require_gpus capability range 8.0-10.0 (Blackwell sm_120 has no ver4 kernels)
  - LAMP_DIR=data/lamp_time on every EVAL job (its omission silently made PT1's
    first smoke eval match 0/2500 records and still exit 0)

Usage (CPU-only, <1s):
    python condor/gen_r9_subs.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDOR_DIR = PROJECT_ROOT / "condor"
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"
TOP_USERS = USER_STATS_DIR / "LaMP_4_top100_users.json"

R = str(PROJECT_ROOT)
BASE_ADAPTER = f"{R}/train/checkpoints/a2_lamp_1ep_seed0/final"
IMAGE = "ghcr.io/gordofreemo/smollm3-train:ver4"

# Smallest profile in R6's top-100 (profile_size=17) — the fingerprint R6's,
# PT2's and this round's smokes all use.
SMOKE_FP = "u00000070"

GPU_GUARDS = """gpus_minimum_memory     = 32000
gpus_minimum_capability = 8.0
require_gpus            = Capability >= 8.0 && Capability < 10.0"""

REQUIREMENTS = ('requirements          = UidDomain == "cs.uni-saarland.de" && '
                'Machine != "tyr1.hpc.uni-saarland.de" && '
                'Machine != "modi.hpc.uni-saarland.de"')

MOUNTS = """+WantGPUHomeMounted   = true
+WantScratchMounted   = true"""


def cfg_path(fp: str) -> str:
    return f"{R}/train/config/user_lora_lamp4_oppu_{fp}_r9.json"


def adapter_path(fp: str) -> str:
    return f"{R}/train/checkpoints/user_lora_lamp4_{fp}_r9_seed0/final"


def logs(stem: str, per_proc_extra: str = "") -> str:
    tag = f"{stem}{per_proc_extra}"
    return (f"output                = {R}/runlogs/{tag}.$(ClusterId).$(ProcId).out\n"
            f"error                 = {R}/runlogs/{tag}.$(ClusterId).$(ProcId).err\n"
            f"log                   = {R}/runlogs/{stem}.$(ClusterId).log")


# --------------------------------------------------------------------------
# Training subs
# --------------------------------------------------------------------------

def train_smoke_sub() -> str:
    return f"""# R9 — single-user smoke before the 100-proc full submit.
#
# Trains ONE per-user LoRA ({SMOKE_FP}, smallest profile in R6's top-100,
# profile_size=17 — the same fingerprint R6's and PT2's smokes used) to
# validate the OPPU recipe stacked on One-LoRA FT before submitting
# condor/train_user_lora_lamp4_r9.sub for the full K=100.
#
# Delta over condor/train_user_lora_lamp4_ptl_smoke.sub: config points at the
# _r9 variant (base_adapter = a2_lamp_1ep_seed0/final, not
# per_task_lamp4_1ep_seed0/final); output lands at a fresh _r9_seed0 dir so
# R6's _oppu and PT2's _ptl adapters are untouched.
#
# Acceptance — check these, not merely exit 0:
#   - train/checkpoints/user_lora_lamp4_{SMOKE_FP}_r9_seed0/final/adapter_config.json
#     with r=8, target_modules=["q_proj", "v_proj"]
#   - train_meta.json has base_adapter_path ending in a2_lamp_1ep_seed0/final
#     (NOT a1_lamp.../per_task_lamp4...) and base_adapter_mode == "merge"
#   - non-NaN final loss
#   - no OOM
#
# Submit (after committing this file):
#   condor_submit condor/train_user_lora_lamp4_r9_smoke.sub

universe              = docker
docker_image          = {IMAGE}
executable            = train/train.py
arguments             = --config {cfg_path(SMOKE_FP)}

{logs("train_user_lora_lamp4_r9_smoke")}

should_transfer_files = YES
environment           = "PROJECT_ROOT={R} CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"

{GPU_GUARDS}

max_job_retirement_time = 28800

stream_output           = true
stream_error            = true
when_to_transfer_output = ON_EXIT_OR_EVICT

request_GPUs          = 1
request_CPUs          = 4
request_memory        = 32G
{REQUIREMENTS}
{MOUNTS}

queue 1
"""


def train_batch_sub(fps: list) -> str:
    rows = "\n".join(f"  {cfg_path(fp)}" for fp in fps)
    return f"""# R9 -- LaMP-4 multi-user training on One-LoRA FT (K=100 parallel GPU procs).
#
# Each proc reads a per-user OPPU config from train/config/ (generated by
# data/lamp_user_stats/round9_gen_configs.py from the R9 template) and trains
# a fresh r=8 q+v-only LoRA on top of One-LoRA FT
# (train/checkpoints/a2_lamp_1ep_seed0/final), stacked via train.py's
# base_adapter plumbing with base_adapter_mode="merge" -- the cold zero-init
# recipe every R/PT round uses, NOT the warm-start round's "continue" mode.
#
# Delta over condor/train_user_lora_lamp4_ptl.sub: base_adapter is One-LoRA FT
# instead of Per-Task-LoRA(LaMP-4); every config/output path carries an _r9
# suffix so R6's and PT2's 100 adapters are never touched. Reuses R6's exact
# K=100 pool and per-user JSONL corpora unchanged -- only the base Task-LoRA
# swaps, the same mechanical delta R8 applied to R5/LaMP-3.
#
# That pool reuse is sound for a One-LoRA FT base specifically because
# data/lamp_train_mixed7_bm25k4.meta.json records per_task_reused[LaMP_4]=true:
# R7 read LaMP-4's per-task training file read-only rather than rebuilding it,
# so the Task-LoRA saw the identical LaMP-4 examples under A1-lamp and
# One-LoRA FT and R6's leakage-eligibility computation still holds.
#
# OPPU recipe (pinned, do NOT edit, byte-identical to R6/PT2):
#   r=8, lora_alpha=16, target_modules=[q_proj, v_proj], lora_dropout=0.05
#   LR=1e-5, weight_decay=1e-2, 3 epochs, cosine + 3% warmup
#   per_device_batch=2, grad_accum=4 (R5/R6's working config)
#   max_seq_length=1024 (R6's T3 sizing, reused unchanged)
#   logging_steps=1 (R6's fix for low-step users, reused unchanged)
#
# Acceptance per user:
#   - train/checkpoints/user_lora_lamp4_<user>_r9_seed0/final/adapter_config.json
#     with r=8, target_modules=["q_proj", "v_proj"]
#   - train_meta.json with base_adapter_path ending in a2_lamp_1ep_seed0/final
#   - non-NaN final loss
#   - All 100 adapters present (or at least 95); surface specifics if 3-5 fail
#
# Run the smoke sub (condor/train_user_lora_lamp4_r9_smoke.sub) first.
#
# Plan: experiments/2026-07-20-user-lora-round9-lamp4-onelora-plan.md
#
# Submit (after committing this file, and after the smoke sub passes):
#   condor_submit condor/train_user_lora_lamp4_r9.sub

universe              = docker
docker_image          = {IMAGE}
executable            = train/train.py
arguments             = --config $(config_path)

{logs("train_user_lora_lamp4_r9")}

should_transfer_files = YES
environment           = "PROJECT_ROOT={R} CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"

{GPU_GUARDS}

max_job_retirement_time = 28800

stream_output           = true
stream_error            = true
when_to_transfer_output = ON_EXIT_OR_EVICT

request_GPUs          = 1
request_CPUs          = 4
request_memory        = 32G
{REQUIREMENTS}
{MOUNTS}

# 100 per-user config paths, in the canonical top-100 user ordering from
# data/lamp_user_stats/LaMP_4_top100_users.json.
queue config_path from (
{rows}
)
"""


# --------------------------------------------------------------------------
# Eval subs
# --------------------------------------------------------------------------

EVAL_ENV = (f'environment           = "MODEL_OUT_DIR={R}/data/models/SmolLM3-3B '
            f'LAMP_DIR={R}/data/lamp_time '
            f'CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"')


def eval_smoke_sub() -> str:
    return f"""# R9 — smoke eval of both arms (C2-R9 baseline, C3-R9 stacked), restricted
# to the one smoke user ({SMOKE_FP}) trained by
# condor/train_user_lora_lamp4_r9_smoke.sub. Confirms both eval paths work
# before submitting the full K=100 batch (condor/eval_lamp_user_lamp4_r9.sub).
#
# baseline row: One-LoRA FT alone (--adapter set, --base-adapter none),
# restricted via --user-records to this one user.
# stacked row: One-LoRA FT + User-LoRA-R9 stacked (--base-adapter + --adapter),
# same as a single row of the full C3-R9 per-user batch.
#
# No --limit: LaMP-4's --user-records filter already restricts to this one
# user's small test set (<=25 records), and --limit would additionally truncate
# the underlying full split BEFORE that filter runs.
#
# Acceptance — check the record count, not merely exit 0. PT1's first smoke
# eval matched 0/2500 records and still exited 0 because LAMP_DIR was missing;
# it is set below, and both rows must report a NON-ZERO matched record count
# equal to this user's n_test.
#
# Submit (after committing this file):
#   condor_submit condor/eval_lamp_user_lamp4_r9_smoke.sub

universe              = docker
docker_image          = {IMAGE}
executable            = eval/eval_lamp.py

{logs("eval_lamp_user_lamp4_r9_smoke", ".$(arm)")}

should_transfer_files = YES
{EVAL_ENV}

request_GPUs          = 1
request_CPUs          = 2
request_memory        = 16G
{REQUIREMENTS}
require_gpus          = Capability >= 8.0 && Capability < 10.0
{MOUNTS}

arguments = --task LaMP_4 --split test --k 4 --seed 0 --adapter {BASE_ADAPTER} --base-adapter none --user-records {SMOKE_FP}
arm       = baseline
queue 1

arguments = --task LaMP_4 --split test --k 4 --seed 0 --adapter {adapter_path(SMOKE_FP)} --base-adapter {BASE_ADAPTER} --user-records {SMOKE_FP}
arm       = stacked
queue 1
"""


def eval_batch_sub(fps: list) -> str:
    c2 = "\n".join(f"  {BASE_ADAPTER}, none, {fp}" for fp in fps)
    c3 = "\n".join(f"  {adapter_path(fp)}, {BASE_ADAPTER}, {fp}" for fp in fps)
    return f"""# LaMP eval -- R9, 100 users x {{C2-R9, C3-R9}} on the LaMP-4 time-split.
#
# 200 parallel GPU procs (mirrors condor/eval_lamp_user_lamp4_ptl.sub's
# structure exactly, since LaMP-4 users have 1-25 test records each -- unlike
# LaMP-3, R9 cannot use R8's single-job --user-records-from-file shortcut,
# which only pulls one test_record_id per user):
#
#   C2-R9 (One-LoRA FT + BM25, no User-LoRA):
#       adapter      = a2_lamp_1ep_seed0/final
#       base_adapter = none
#       Reproduces the per-user baseline on each user's LaMP_4 test records.
#       Model is user-invariant; run per-user anyway (same as R6's/PT2's C2)
#       so results land in per-user files with the same n_test line count as C3.
#       NOTE: unlike PT2, these files do NOT already exist on disk -- only the
#       full-split LaMP_4_test_a2_lamp_1ep_seed0_final_bm25k4_seed0.json does,
#       which is the wrong pool. All 100 C2 rows really do need to run.
#
#   C3-R9 (One-LoRA FT + User-LoRA-R9 + BM25, stacked):
#       adapter      = user_lora_lamp4_<user>_r9_seed0/final  (condor/train_user_lora_lamp4_r9.sub output)
#       base_adapter = a2_lamp_1ep_seed0/final
#
# Both conditions use --user-records <fp> to pin eval to that user's LaMP_4
# test records, per data/lamp_user_stats/LaMP_4_user_records.json.
#
# eval_lamp.py refuses to overwrite by default, so re-submits after partial
# sweeps only re-run the missing combinations.
#
# tyr1/modi excluded -- known oversubscription/ECC pattern hit by R8/PT1.
# LAMP_DIR set -- its omission silently made PT1's first smoke match 0 records.
#
# Run the smoke sub (condor/eval_lamp_user_lamp4_r9_smoke.sub) first, and only
# after the K=100 training batch (condor/train_user_lora_lamp4_r9.sub) has
# produced all 100 adapters.
#
# Submit (after committing this file):
#   condor_submit condor/eval_lamp_user_lamp4_r9.sub

universe              = docker
docker_image          = {IMAGE}
executable            = eval/eval_lamp.py

# Macros driven by the queue block. eval_lamp.py treats "none" as "no adapter".
adapter      = none
base_adapter = none
user_fp      = u00000000

arguments    = --task LaMP_4 --split test --k 4 --seed 0 --adapter $(adapter) --base-adapter $(base_adapter) --user-records $(user_fp)

{logs("eval_lamp_user_lamp4_r9")}

should_transfer_files = YES
{EVAL_ENV}

max_job_retirement_time = 600
when_to_transfer_output = ON_EXIT_OR_EVICT
stream_output           = true
stream_error            = true

{GPU_GUARDS}

request_GPUs          = 1
request_CPUs          = 2
request_memory        = 16G
{REQUIREMENTS}
{MOUNTS}

# 200 (adapter, base_adapter, user_fp) rows: 100 C2-R9 (One-LoRA FT alone)
# then 100 C3-R9 (User-LoRA-R9 stacked). Ordered to match the canonical
# top-100 user ordering from data/lamp_user_stats/LaMP_4_top100_users.json.
queue adapter, base_adapter, user_fp from (
{c2}
{c3}
)
"""


def paired_compare_sub() -> str:
    return f"""# Paired comparison -- R9 (LaMP-4 User-LoRA on One-LoRA FT).
#
# CPU-only job. Runs eval/paired_compare_per_user.py UNMODIFIED on the
# consolidated C2-R9 vs C3-R9 prediction files:
#   pred-a: results/LaMP_4_test_r9_C2.predictions.jsonl
#   pred-b: results/LaMP_4_test_r9_C3.predictions.jsonl
#
# Same script R6/PT2 used (eval/paired_compare_per_user.py, not the flat
# eval/paired_compare.py R8 used) -- LaMP-4 users have 1-25 test records each,
# so this scores ROUGE-1 per record then groups by user, pairing at the
# per-user-mean level (K=100 paired observations).
#
# ROUGE-1 is higher-is-better, so paired_compare_per_user.py's
# wins_b_over_a/wins_a_over_b read the natural way here -- the MAE sign gotcha
# that R5/R8/PT1 each had to hand-correct does not apply to this round.
#
# NO pre-registered gate this round (same convention as R6/R8/PT1/PT2/R10-R14)
# -- p-values/CI/wins-ties-losses are reported side-by-side with R6's archived
# mean dR-1 +0.007 (ns, p=0.20) and PT2's -0.0129, not used for pass/fail.
#
# Read the result against the other two LaMP-4 rounds, and note that R9 is the
# 22nd uncorrected test in this program's User-LoRA sequence -- a lone nominal
# p<0.05 carries no more weight than R10's did.
#
# Must run via condor_submit, not directly -- scipy isn't on the login node.
#
# Submit (after committing this file, and after the aggregator has run):
#   condor_submit condor/paired_compare_lamp4_r9.sub

universe              = docker
docker_image          = {IMAGE}
executable            = eval/paired_compare_per_user.py

arguments = --pred-a {R}/results/LaMP_4_test_r9_C2.predictions.jsonl --pred-b {R}/results/LaMP_4_test_r9_C3.predictions.jsonl --label-a c2_r9_bm25 --label-b c3_r9_userlora_bm25 --task LaMP_4 --split test --seed 0

{logs("paired_compare_lamp4_r9")}

should_transfer_files = YES
environment           = "PROJECT_ROOT={R} CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"

request_GPUs          = 0
request_CPUs          = 1
request_memory        = 2G

max_job_retirement_time = 600
when_to_transfer_output = ON_EXIT_OR_EVICT
stream_output           = true
stream_error            = true

requirements          = UidDomain == "cs.uni-saarland.de"
{MOUNTS}

queue 1
"""


def main():
    if not TOP_USERS.exists():
        sys.exit(f"ERROR: missing {TOP_USERS}.")
    fps = [u["user_fingerprint"] for u in json.loads(TOP_USERS.read_text())["users"]]
    if len(fps) != len(set(fps)):
        sys.exit("ERROR: duplicate fingerprints in the top-100 pool.")
    if SMOKE_FP not in fps:
        sys.exit(f"ERROR: smoke user {SMOKE_FP} is not in the pool.")

    # Every per-user config must already exist — the subs reference them by
    # path, and Condor would happily queue jobs pointing at nothing.
    missing = [fp for fp in fps if not Path(cfg_path(fp)).exists()]
    if missing:
        sys.exit(f"ERROR: {len(missing)} per-user configs missing (first: "
                 f"{cfg_path(missing[0])}). Run "
                 f"data/lamp_user_stats/round9_gen_configs.py first.")

    outputs = {
        "train_user_lora_lamp4_r9_smoke.sub": train_smoke_sub(),
        "train_user_lora_lamp4_r9.sub": train_batch_sub(fps),
        "eval_lamp_user_lamp4_r9_smoke.sub": eval_smoke_sub(),
        "eval_lamp_user_lamp4_r9.sub": eval_batch_sub(fps),
        "paired_compare_lamp4_r9.sub": paired_compare_sub(),
    }
    for name, text in outputs.items():
        (CONDOR_DIR / name).write_text(text)
        print(f"[gen] {name}", flush=True)
    print(f"[gen] wrote {len(outputs)} subs for {len(fps)} users to {CONDOR_DIR}",
          flush=True)


if __name__ == "__main__":
    main()
