#!/bin/bash
# LaMP-1 repaired-prompt smoke (OUR repair, not their release — see PATCHES.md
# P18): both stages, 1 test user, using oppu_replication/prompt_fixed.json,
# which adds the two candidate-reference slots + gold slot the released
# citation templates are missing. Separate --out-root/--ckpt-root so nothing
# can collide with the faithful round's artifacts.
set -euo pipefail
cd "${PROJECT_ROOT:?PROJECT_ROOT must be set}"
FIX_ARGS=(--prompt-file oppu_replication/prompt_fixed.json
          --out-root results/oppu_rep_fixed
          --ckpt-root train/checkpoints/oppu_rep_fixed)
python3 oppu_replication/run_task_lora.py --task_name citation --k 1 \
    "${FIX_ARGS[@]}" --limit 1 --limit-train 30 --overwrite
python3 oppu_replication/run_oppu.py --task_name citation --k 1 \
    "${FIX_ARGS[@]}" \
    --task_lora train/checkpoints/oppu_rep_fixed/citation/task_lora_k1_limit1t30 \
    --user-start 0 --user-end 1 --tag _smoke --overwrite
echo "--- first RAG-arm record (eyeball: pred should be [1]/[2], not a title) ---"
head -n 1 results/oppu_rep_fixed/citation/task_k1_limit1t30_preds.jsonl
echo "SMOKE OK: citation (repaired prompts)"
