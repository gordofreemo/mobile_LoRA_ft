#!/bin/bash
# Pull every on-device-trained h13 adapter off the phone in one sweep.
#
# The per-user pull inside run_h13_user.sh silently produced nothing: it never
# created the destination directory, and `devicectl copy from` fails when the
# destination is missing (the `|| true` there swallowed it). Nothing was lost --
# the adapters live on the phone under h13_adapters/<user>/device/ -- so this
# collects them after the fact. It is also the right shape for the job: one
# sweep instead of 100 per-user calls.
#
# Fix to apply to run_h13_user.sh when the queue is next idle: add
#   mkdir -p "$ROOT/results/ondevice/h13_device_adapters/$USER_ID"
# immediately before that copy.
set -uo pipefail
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
DEV=00008150-000674C60A3B401C
APP=mlx.LLMEvalJGW9U9Y36Y
ROOT="$HOME/Documents/Research/mobile_LoRA_ft"
DEST="$ROOT/results/ondevice/h13_device_adapters"

OK=0; MISS=0
for D in "$ROOT"/results/ondevice/h13_preds/*/; do
  U=$(basename "$D")
  OUT="$DEST/$U"
  [ -s "$OUT/adapters.safetensors" ] && { OK=$((OK+1)); continue; }
  mkdir -p "$OUT"
  if xcrun devicectl device copy from --timeout 600 --device "$DEV" \
       --domain-type appDataContainer --domain-identifier "$APP" \
       --source "Documents/h13_adapters/$U/device" --destination "$OUT/" \
       >/dev/null 2>&1 && [ -s "$OUT/adapters.safetensors" ]; then
    OK=$((OK+1)); echo "[pull] $U ok"
  else
    MISS=$((MISS+1)); echo "[pull] $U MISSING"
    rmdir "$OUT" 2>/dev/null
  fi
done
echo "[pull] $OK adapters present, $MISS missing -> $DEST"
