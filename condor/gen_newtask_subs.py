#!/usr/bin/env python3
"""
Emit every Condor sub file for the ten new User-LoRA tracks.

  task           R-track (base = One-LoRA FT)   PT-track (base = Per-Task-LoRA)
  LaMP_2_news    R10                            PT3
  LaMP_1         R11                            PT4
  LaMP_7         R12                            PT5
  LaMP_2_movies  R13                            PT6
  LaMP_5         R14                            PT7

Why a generator instead of ~70 hand-written files: the earlier rounds each
had ONE track, so hand-writing `condor/train_user_lora_lamp4_ptl.sub` with
its 100-line inline `queue` block was reasonable. Ten tracks across five
tasks with two different eval patterns is not — hand-copying would guarantee
drift in exactly the fields that must not drift (the LAMP_DIR override, the
tyr1/modi exclusions, the Blackwell capability range). Every generated file
carries a header saying it was generated and how to regenerate it; the files
themselves are still committed, still human-readable, and still the artifact
`condor_submit` consumes.

The per-user fingerprints are embedded INLINE in each queue block (read from
the pool JSON at generation time, not at submit time) so each sub file stays
a self-contained, reviewable pre-registration artifact — the same rationale
condor/build_user_dataset_lamp4_multi.sub documents.

Infra guards baked into every generated GPU sub from the first submit, per
CLAUDE.md's standing notes:
  - `Machine != tyr1...` and `Machine != modi...` (the CUDA-busy
    oversubscription and uncorrectable-ECC hosts that cost R8 83/100 jobs)
  - `require_gpus = Capability >= 8.0 && Capability < 10.0` (ver4's
    PyTorch 2.5.1+cu124 cannot target the sm_120 Blackwell nodes)
  - `LAMP_DIR=<abs>/data/lamp_time` on every eval sub — omitting it silently
    matched 0 records and "succeeded" with n=0 in PT1's first smoke run

Usage (CPU-only, instant):
    python condor/gen_newtask_subs.py --all
    python condor/gen_newtask_subs.py --task LaMP_7 --overwrite
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
ONE_LORA_FT = "train/checkpoints/a2_lamp_1ep_seed0/final"
ONE_LORA_FT_TAG = "a2_lamp_1ep_seed0_final"

GPU_REQS = ('requirements          = UidDomain == "cs.uni-saarland.de" && '
            'Machine != "tyr1.hpc.uni-saarland.de" && '
            'Machine != "modi.hpc.uni-saarland.de"')
CPU_REQS = 'requirements          = UidDomain == "cs.uni-saarland.de"'
GPU_BLOCK = """gpus_minimum_memory     = 32000
gpus_minimum_capability = 8.0
require_gpus            = Capability >= 8.0 && Capability < 10.0
"""


class Task:
    def __init__(self, name, tag, k, pool, build_args, suffix, trainer,
                 eval_pattern, metric, per_task_ckpt, build_mem):
        self.name = name
        self.tag = tag
        self.k = k
        self.pool = pool
        self.build_args = build_args
        self.suffix = suffix
        self.trainer = trainer
        self.eval_pattern = eval_pattern   # "flat" | "grouped"
        self.metric = metric               # paired-compare metric
        self.per_task_ckpt = per_task_ckpt
        self.per_task_tag = per_task_ckpt.replace("train/checkpoints/", "").replace("/", "_")
        self.build_mem = build_mem


TASKS = {t.name: t for t in [
    Task("LaMP_2_news", "lamp2news", 27, "LaMP_2_news_top27_users.json",
         "--framing records --bm25-k 4", "records_bm25k4", "train/train.py",
         "grouped", "accuracy",
         "train/checkpoints/per_task_lamp2_news_1ep_seed0/final", "24G"),
    Task("LaMP_1", "lamp1", 100, "LaMP_1_top100_users.json",
         "--framing unsupervised", "unsup", "train/train_unsupervised_clm.py",
         "flat", "accuracy",
         "train/checkpoints/per_task_lamp1_1ep_seed0/final", "8G"),
    Task("LaMP_7", "lamp7", 100, "LaMP_7_top100_users.json",
         "--framing unsupervised", "unsup", "train/train_unsupervised_clm.py",
         "flat", "rouge1",
         "train/checkpoints/per_task_lamp7_1ep_seed0/final", "8G"),
    Task("LaMP_2_movies", "lamp2movies", 100, "LaMP_2_movies_top100_users.json",
         "--bm25-k 4", "bm25k4", "train/train.py",
         "flat", "accuracy",
         "train/checkpoints/per_task_lamp2_movies_1ep_seed0/final", "8G"),
    Task("LaMP_5", "lamp5", 100, "LaMP_5_top100_users.json",
         "--bm25-k 4", "bm25k4", "train/train.py",
         "flat", "rouge1",
         "train/checkpoints/per_task_lamp5_1ep_seed0/final", "16G"),
]}

# track -> (task name, uses One-LoRA FT?)
TRACKS = {
    "r10": ("LaMP_2_news", True),   "pt3": ("LaMP_2_news", False),
    "r11": ("LaMP_1", True),        "pt4": ("LaMP_1", False),
    "r12": ("LaMP_7", True),        "pt5": ("LaMP_7", False),
    "r13": ("LaMP_2_movies", True), "pt6": ("LaMP_2_movies", False),
    "r14": ("LaMP_5", True),        "pt7": ("LaMP_5", False),
}


def header(title, body, submit_cmd):
    return (
        f"# {title}\n#\n"
        + "".join(f"# {line}\n" if line else "#\n" for line in body)
        + "#\n"
        + "# GENERATED by condor/gen_newtask_subs.py — edit that script and\n"
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
        sys.exit(f"ERROR: missing pool {p}. Run "
                 f"data/select_top_users_newtasks.py --task {task.name} first.")
    doc = json.loads(p.read_text())
    return [u["user_fingerprint"] for u in doc["users"]], doc


def smallest_user(task: Task):
    """Smoke subject: the smallest profile in the pool — fastest to train and
    the most likely to expose an empty-corpus edge case."""
    _, doc = pool_users(task)
    return min(doc["users"], key=lambda u: u["profile_size"])["user_fingerprint"]


# --- generators --------------------------------------------------------------
def sub_pool(task: Task):
    stem = f"select_top_users_{task.tag}"
    return stem, (
        header(
            f"Pool selection for {task.name} (K={task.k}).",
            [f"Streams {task.name}/test_questions.json, ranks eligible users by",
             "profile size, writes data/lamp_user_stats/" + task.pool + ".",
             "",
             "CPU-only. Streaming read + a small in-memory eligibility set."],
            f"condor_submit condor/{stem}.sub",
        )
        + f"universe              = docker\ndocker_image          = {IMAGE}\n"
        + "executable            = data/select_top_users_newtasks.py\n"
        + f"arguments             = --task {task.name}\n\n"
        + logs(stem)
        + "should_transfer_files = YES\n"
        + f'environment           = "PROJECT_ROOT={ROOT} '
          f'LAMP_DIR={ROOT}/data/lamp_time '
          f'CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"\n\n'
        + "request_CPUs          = 2\nrequest_memory        = 8G\n"
          "request_disk          = 2G\n"
        + CPU_REQS + "\n+WantGPUHomeMounted   = true\n+WantScratchMounted   = true\n\n"
        + "queue 1\n"
    )


def sub_t3(task: Task):
    stem = f"t3_sizing_{task.tag}"
    framing = ("unsupervised" if task.suffix == "unsup"
               else "records" if task.suffix.startswith("records") else "profile")
    return stem, (
        header(
            f"T3 sizing for {task.name} — pin max_seq_length from the real corpora.",
            [f"Tokenizes all {task.k} per-user {task.suffix} corpora and pins",
             "max_seq_length = round_up_to_256(min(global_max, 8192)).",
             "",
             f"Framing is {framing}, so tokenization "
             + ("skips the chat template (matching train_unsupervised_clm.py)."
                if framing == "unsupervised" else
                "applies the SmolLM3 chat template (matching train.py)."),
             "",
             "STOPS with a non-zero exit if the global max exceeds the 8192",
             "positional ceiling — do not auto-truncate, decide jointly.",
             "",
             "Run AFTER the per-user corpus build, BEFORE generating configs."],
            f"condor_submit condor/{stem}.sub",
        )
        + f"universe              = docker\ndocker_image          = {IMAGE}\n"
        + "executable            = data/lamp_user_stats/newtask_t3_sizing.py\n"
        + f"arguments             = --task {task.name} --framing {framing}\n\n"
        + logs(stem)
        + "should_transfer_files = YES\n"
        + f'environment           = "PROJECT_ROOT={ROOT} '
          f'CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"\n\n'
        + "request_CPUs          = 2\nrequest_memory        = 16G\n"
          "request_disk          = 4G\n"
        + CPU_REQS + "\n+WantGPUHomeMounted   = true\n+WantScratchMounted   = true\n\n"
        + "queue 1\n"
    )


def sub_build(task: Task, smoke: bool):
    fps, _ = pool_users(task)
    stem = f"build_user_dataset_{task.tag}" + ("_smoke" if smoke else "")
    if smoke:
        fps = [smallest_user(task)]
    snapshot_note = (
        ["This task's users appear in exactly ONE split — none holds both a",
         "train record and a test record — so build_user_dataset.py sources the",
         "profile snapshot from the user's own TEST record (documented fallback,",
         "not leakage: a record's profile is history prior to that record)."]
        if task.name in ("LaMP_1", "LaMP_2_movies", "LaMP_5") else
        ["Users hold real train-split records, so the snapshot comes from the",
         "train split as usual."]
    )
    return stem, (
        header(
            f"Per-user {task.name} training corpus build"
            + (" — SMOKE (1 user)." if smoke else f" ({len(fps)} parallel CPU procs)."),
            [f"Emits data/lamp_user_train_{task.name}_<user>_{task.suffix}.jsonl",
             "(+ .meta.json) for each user in the pinned pool",
             f"data/lamp_user_stats/{task.pool}.",
             "",
             f"Framing: {task.build_args}",
             ""] + snapshot_note + [
             "",
             "build_user_dataset.py refuses to overwrite by default, so a",
             "re-submit after a partial sweep only fills the gaps. The smoke",
             "user's file will collide on the full submit — that ONE",
             "refuse-to-overwrite failure is expected, not a real failure."]
            + ([f"", f"SMOKE SUBJECT: {fps[0]} (smallest profile in the pool)."]
               if smoke else []),
            f"condor_submit condor/{stem}.sub",
        )
        + f"universe              = docker\ndocker_image          = {IMAGE}\n"
        + "executable            = train/build_user_dataset.py\n"
        + f"arguments             = --task {task.name} --user $(user) {task.build_args}\n\n"
        + logs(stem)
        + "should_transfer_files = YES\n"
        + f'environment           = "PROJECT_ROOT={ROOT} '
          f'LAMP_DIR={ROOT}/data/lamp_time '
          f'CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"\n\n'
        + f"request_CPUs          = 2\nrequest_memory        = {task.build_mem}\n"
          "request_disk          = 4G\n"
        + CPU_REQS + "\n+WantGPUHomeMounted   = true\n+WantScratchMounted   = true\n\n"
        + "queue user from (\n"
        + "".join(f"  {fp}\n" for fp in fps)
        + ")\n"
    )


def sub_train(task: Task, track: str, smoke: bool):
    _, use_one_lora = TRACKS[track]
    base = ONE_LORA_FT if use_one_lora else task.per_task_ckpt
    base_name = "One-LoRA FT" if use_one_lora else f"Per-Task-LoRA({task.name})"
    fps, _ = pool_users(task)
    stem = f"train_user_lora_{task.tag}_{track}" + ("_smoke" if smoke else "")
    if smoke:
        fps = [smallest_user(task)]
    cfgs = [f"{ROOT}/train/config/user_lora_{task.tag}_oppu_{fp}_{track}.json"
            for fp in fps]
    objective = ("OPPU's UNSUPERVISED right-shifted-history objective "
                 "(train/train_unsupervised_clm.py — raw history text, no chat "
                 "template, loss on every token)"
                 if task.suffix == "unsup" else
                 "the standard supervised SFT objective (train/train.py)")
    return stem, (
        header(
            f"{track.upper()} — {task.name} User-LoRA on {base_name}"
            + (" — SMOKE (1 user)." if smoke else f" (K={len(fps)} parallel GPU procs)."),
            [f"Trains a fresh r=8 q+v LoRA per user on top of {base},",
             "stacked via base_adapter (merged into the backbone, NOT a second",
             "live adapter) — unchanged from R5/R6's stack-vs-merge decision.",
             "",
             f"Objective: {objective}.",
             "",
             "OPPU recipe (pinned, do NOT edit): r=8, alpha=16, q_proj+v_proj,",
             "dropout=0.05, LR=1e-5, weight_decay=1e-2, 3 epochs, cosine + 3%",
             "warmup, per_device_batch=2, grad_accum=4, logging_steps=1.",
             "max_seq_length comes from this task's own T3 sizing pass.",
             "",
             "Configs are generated by",
             f"data/lamp_user_stats/newtask_gen_configs.py --track {track}.",
             "",
             "Acceptance per user: final/adapter_config.json with r=8 and",
             "target_modules=[q_proj, v_proj]; train_meta.json whose",
             f"base_adapter_path ends in {base.split('/')[-2]}/final;",
             "non-NaN final loss."]
            + ([f"", f"SMOKE SUBJECT: {fps[0]} (smallest profile in the pool).",
                "Run this and check the acceptance criteria before the full batch."]
               if smoke else
               ["", f"Run condor/train_user_lora_{task.tag}_{track}_smoke.sub first.",
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


def _eval_common(stem, args_line, mem="16G"):
    return (
        f"universe              = docker\ndocker_image          = {IMAGE}\n"
        + "executable            = eval/eval_lamp.py\n\n"
        + args_line
        + logs(stem)
        + "should_transfer_files = YES\n"
        # LAMP_DIR is load-bearing: --user-records filtering is built against
        # the TIME-based split's record ids. Without it eval_lamp.py defaults
        # to the user-based split, matches 0 records, and exits 0 with n=0.
        + f'environment           = "MODEL_OUT_DIR={ROOT}/data/models/SmolLM3-3B '
          f'LAMP_DIR={ROOT}/data/lamp_time '
          f'CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)"\n\n'
        + "max_job_retirement_time = 600\n"
          "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
          "stream_output           = true\nstream_error            = true\n\n"
        + GPU_BLOCK
        + f"\nrequest_GPUs          = 1\nrequest_CPUs          = 2\n"
          f"request_memory        = {mem}\n"
        + GPU_REQS + "\n+WantGPUHomeMounted   = true\n+WantScratchMounted   = true\n\n"
    )


def sub_eval_baseline(task: Task, track: str):
    """Flat pattern only: the baseline arm is user-invariant, so ONE job with
    --user-records-from-file covers all K subset records at once."""
    _, use_one_lora = TRACKS[track]
    base = ONE_LORA_FT if use_one_lora else task.per_task_ckpt
    base_name = "One-LoRA FT" if use_one_lora else f"Per-Task-LoRA({task.name})"
    stem = f"eval_lamp_user_{task.tag}_{track}_baseline"
    pool_path = f"{ROOT}/data/lamp_user_stats/{task.pool}"
    args_line = (
        f"arguments             = --task {task.name} --split test --k 4 --seed 0 "
        f"--adapter {ROOT}/{base} --user-records-from-file {pool_path}\n\n"
    )
    return stem, (
        header(
            f"{track.upper()} baseline arm (C2) — {base_name} alone, K={task.k} subset.",
            [f"Every {task.name} pool user holds exactly ONE test record, so the",
             "whole baseline is one job: --user-records-from-file pulls each",
             "user's single test_record_id out of the pool JSON.",
             "",
             "No User-LoRA here — this is the arm the per-user adapters are",
             "compared AGAINST.",
             "",
             f"Output: results/{task.name}_test_<basetag>_bm25k4_seed0_topK{task.k}"
             ".{json,predictions.jsonl}"],
            f"condor_submit condor/{stem}.sub",
        )
        + _eval_common(stem, args_line)
        + "queue 1\n"
    )


def sub_eval(task: Task, track: str, smoke: bool):
    _, use_one_lora = TRACKS[track]
    base = ONE_LORA_FT if use_one_lora else task.per_task_ckpt
    base_name = "One-LoRA FT" if use_one_lora else f"Per-Task-LoRA({task.name})"
    fps, _ = pool_users(task)
    stem = f"eval_lamp_user_{task.tag}_{track}" + ("_smoke" if smoke else "")
    if smoke:
        fps = [smallest_user(task)]

    def user_adapter(fp):
        return f"{ROOT}/train/checkpoints/user_lora_{task.tag}_{fp}_{track}_seed0/final"

    rows = []
    if task.eval_pattern == "grouped" or smoke:
        # Both arms per user. (Smoke always runs both so the pair is
        # comparable on one user before committing to the full sweep.)
        for fp in fps:
            rows.append((f"{ROOT}/{base}", "none", fp))            # C2
        for fp in fps:
            rows.append((user_adapter(fp), f"{ROOT}/{base}", fp))  # C3
        arm_note = ["Both arms run per-user (C2 rows first, then C3):"]
    else:
        for fp in fps:
            rows.append((user_adapter(fp), f"{ROOT}/{base}", fp))
        arm_note = [
            "Treatment arm (C3) only — the baseline arm is user-invariant and",
            f"runs as ONE job in condor/eval_lamp_user_{task.tag}_{track}_baseline.sub."]

    args_line = (
        f"# Macros driven by the queue block. eval_lamp.py treats \"none\" as "
        f"\"no adapter\".\n"
        f"adapter      = none\nbase_adapter = none\nuser_fp      = u00000000\n\n"
        f"arguments             = --task {task.name} --split test --k 4 --seed 0 "
        f"--adapter $(adapter) --base-adapter $(base_adapter) "
        f"--user-records $(user_fp)\n\n"
    )
    return stem, (
        header(
            f"{track.upper()} eval — {task.name}, base {base_name}"
            + (" — SMOKE (1 user, both arms)." if smoke else f" ({len(rows)} procs)."),
            arm_note + [
             "",
             "  C2 = base Task-LoRA + BM25, no personalization",
             "  C3 = base Task-LoRA + per-user User-LoRA + BM25 (stacked)",
             "",
             "--user-records <fp> pins eval to that user's test records via",
             f"data/lamp_user_stats/{task.name}_user_records.json.",
             "",
             "LAMP_DIR is set to the TIME-based split below and is load-bearing:",
             "without it eval_lamp.py silently matches 0 records and still",
             "exits 0 (the bug PT1's first smoke eval hit). After this runs,",
             "check the record COUNTS, not just exit codes.",
             "",
             "eval_lamp.py refuses to overwrite, so re-submits only fill gaps."]
            + ([f"", f"SMOKE SUBJECT: {fps[0]}."] if smoke else
               [f"", f"Run condor/{stem}_smoke.sub first, and only after the",
                f"training batch has produced all {task.k} adapters."]),
            f"condor_submit condor/{stem}.sub",
        )
        + _eval_common(stem, args_line)
        + "queue adapter, base_adapter, user_fp from (\n"
        + "".join(f"  {a}, {b}, {u}\n" for a, b, u in rows)
        + ")\n"
    )


def sub_paired(task: Task, track: str):
    stem = f"paired_compare_{task.tag}_{track}"
    grouped = task.eval_pattern == "grouped"
    exe = ("eval/paired_compare_per_user.py" if grouped else "eval/paired_compare.py")
    c2 = f"{ROOT}/results/{task.name}_test_{track}_C2.predictions.jsonl"
    c3 = f"{ROOT}/results/{task.name}_test_{track}_C3.predictions.jsonl"
    extra = f" --round-tag {track}" if grouped else ""
    return stem, (
        header(
            f"{track.upper()} paired comparison — {task.name}, metric={task.metric}.",
            ([f"Users hold 3-35 test records each, so this uses the GROUPED",
              "per-user comparison: per-record scores are averaged within a",
              "user first, then the paired-t / Wilcoxon / bootstrap battery",
              f"runs over the {task.k} per-user means — not over records, which",
              "would over-weight heavy users and break independence."]
             if grouped else
             ["Every user holds exactly 1 test record, so record-level pairing",
              "IS user-level pairing and the flat paired_compare.py applies."])
            + ["",
               "Descriptive reporting only — no pre-registered gate, matching",
               "the convention every round since R6.",
               "",
               "Run AFTER eval/aggregate_user_predictions_newtask.py has built",
               "both consolidated prediction files."],
            f"condor_submit condor/{stem}.sub",
        )
        + f"universe              = docker\ndocker_image          = {IMAGE}\n"
        + f"executable            = {exe}\n\n"
        + f"arguments             = --pred-a {c2} --pred-b {c3} "
          f"--label-a c2_{track}_bm25 --label-b c3_{track}_userlora_bm25 "
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
        task_names = sorted(TASKS)
    elif args.task:
        task_names = [args.task]
    else:
        sys.exit("ERROR: pass --task <name> or --all.")

    out = {}
    for name in task_names:
        task = TASKS[name]
        for gen in (sub_pool, sub_t3):
            stem, text = gen(task)
            out[stem] = text
        for smoke in (True, False):
            stem, text = sub_build(task, smoke)
            out[stem] = text
        for track, (tname, _) in TRACKS.items():
            if tname != name:
                continue
            for smoke in (True, False):
                stem, text = sub_train(task, track, smoke)
                out[stem] = text
                stem, text = sub_eval(task, track, smoke)
                out[stem] = text
            if task.eval_pattern == "flat":
                stem, text = sub_eval_baseline(task, track)
                out[stem] = text
            stem, text = sub_paired(task, track)
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
