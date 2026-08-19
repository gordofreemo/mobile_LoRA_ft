#!/bin/bash
# Catch-up sweep for the Mac-control arm.
#
# The device queue runs whether or not a user's Mac adapter exists yet; the eval
# mode simply skips arms with no adapter. So if Mac-control training is paused
# (e.g. to keep the laptop usable during the day), those users end up with
# rag/cluster/device but no mac. This re-runs ONLY the mac arm for them: it
# stages the adapter, evaluates that one arm, and re-pulls.
#
# Cost is one extra model load (~1 min) per user, which is far cheaper than
# stalling the device queue -- the phone is the ~50 h critical path, the Mac
# needs ~6 h.
set -uo pipefail
ROOT="$HOME/Documents/Research/mobile_LoRA_ft"
PRE="$ROOT/results/ondevice/h13_preds"
MACCK="$ROOT/train/checkpoints_mlx/h13_mac_control"

N=0
for D in "$PRE"/*/; do
  U=$(basename "$D")
  [ -s "$D/rag.jsonl" ] || continue                    # user not evaluated yet
  [ -s "$D/mac.jsonl" ] && continue                    # already has the mac arm
  [ -f "$MACCK/$U/adapters.safetensors" ] || { echo "[catchup] $U: no mac adapter yet"; continue; }
  echo "[catchup] === $U ==="
  SKIP_TRAIN=1 ARMS=mac "$ROOT/scripts/h13/run_h13_user.sh" "$U" || echo "[catchup] $U FAILED"
  N=$((N + 1))
done
echo "[catchup] ran $N users"
