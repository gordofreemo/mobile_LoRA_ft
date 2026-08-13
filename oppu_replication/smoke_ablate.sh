#!/bin/bash
# Recipe-ablation + seed-re-decode smoke (movie tagging, 1 test user).
# Exercises all three new code paths before the batches:
#   1. run_oppu --user-recipe r5      (P19: R5 bundle trains + evals 1 user)
#   2. run_oppu --eval-only --seed 1  (P20: loads user000's saved hot adapter)
#   3. run_task_lora --eval-only --seed 1 --limit 1  (P20: task-arm re-decode)
set -euo pipefail
cd "${PROJECT_ROOT:?PROJECT_ROOT must be set}"
TASK_LORA=train/checkpoints/oppu_rep/movie_tagging/task_lora_k1
python3 oppu_replication/run_oppu.py --task_name movie_tagging --k 1 \
    --task_lora "$TASK_LORA" --user-start 0 --user-end 1 \
    --user-recipe r5 --tag _r5smoke --overwrite
python3 oppu_replication/run_oppu.py --task_name movie_tagging --k 1 \
    --task_lora "$TASK_LORA" --user-start 0 --user-end 1 \
    --eval-only --seed 1 --tag _seed1smoke --overwrite
python3 oppu_replication/run_task_lora.py --task_name movie_tagging --k 1 \
    --eval-only --seed 1 --tag _seed1smoke --limit 1 --overwrite
echo "SMOKE OK: ablate + re-decode paths"
