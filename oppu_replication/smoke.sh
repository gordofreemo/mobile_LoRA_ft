#!/bin/bash
# OPPU-replication smoke: both stages for one task, one test user.
# Stage 1 trains the task-LoRA on 30 train users and evaluates the RAG arm on
# test user 0; stage 2 trains the per-user LoRA for test user 0 on top of the
# stage-1 checkpoint and evaluates the OPPU+RAG arm. The python scripts carry
# the real assertions (lora_B nonzero -> exit 2, prediction counts -> exit 3);
# set -e propagates them to the Condor exit code.
set -euo pipefail
TASK="$1"
cd "${PROJECT_ROOT:?PROJECT_ROOT must be set}"
python3 oppu_replication/run_task_lora.py --task_name "$TASK" --k 1 \
    --limit 1 --limit-train 30 --overwrite
python3 oppu_replication/run_oppu.py --task_name "$TASK" --k 1 \
    --task_lora "train/checkpoints/oppu_rep/$TASK/task_lora_k1_limit1t30" \
    --user-start 0 --user-end 1 --tag _smoke --overwrite
echo "SMOKE OK: $TASK"
