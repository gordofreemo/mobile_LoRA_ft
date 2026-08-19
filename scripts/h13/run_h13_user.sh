#!/bin/bash
# h13 per-user device runner: train -> cooldown -> eval ALL arms in one launch -> pull.
#
# TRAP THIS SCRIPT ENCODES (cost: ~1 h on 2026-08-19):
#   `devicectl device copy to` with a SINGLE --source RENAMES that source to
#   --destination. `--source cfg.json --destination Documents/h13_model/`
#   replaced the h13_model DIRECTORY with a file; a later single-file push
#   replaced Documents ITSELF with an 18-byte file and every read returned
#   CoreDeviceError 7000. Recovery: push a PARENT DIRECTORY as
#   `--source <stage> --destination Documents`. Never push a lone file.
#
# The eval runs every arm inside ONE process launch: the model load is
# ~30-60 s against ~0.4 s of generation per query, so per-arm launches would
# spend more wall-clock loading than measuring.
set -euo pipefail

export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
DEV=00008150-000674C60A3B401C
APP=mlx.LLMEvalJGW9U9Y36Y
ROOT="$HOME/Documents/Research/mobile_LoRA_ft"
PULL="$ROOT/results/ondevice/h13_preds"
COOLDOWN=${COOLDOWN:-300}          # h10: t95 ~ 346 s after a training burst
ARMS=${ARMS:-all}
SKIP_TRAIN=${SKIP_TRAIN:-0}
USER_ID=$1
LOG=/tmp/h13_stderr_$USER_ID.log

log() { echo "[$(date +%H:%M:%S)] $*"; }

pull_log() {
  rm -f "$LOG"
  xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$APP" --source Documents/h13_stderr.log \
    --destination "$LOG" >/dev/null 2>&1 || true
}

count_of() {
  pull_log
  local n
  n=$( { grep -c "$1" "$LOG" 2>/dev/null || true; } | head -1 )
  echo "${n:-0}"
}

wait_for() {  # wait_for <marker> <target-count> <timeout-s>
  local marker=$1 want=$2 limit=${3:-36000} waited=0
  while [ "$waited" -lt "$limit" ]; do
    sleep 20; waited=$((waited + 20))
    pull_log
    local have
    have=$( { grep -c "$marker" "$LOG" 2>/dev/null || true; } | head -1 )
    [ "${have:-0}" -ge "$want" ] && return 0
  done
  log "TIMEOUT waiting for: $marker"; return 1
}

if [ "$SKIP_TRAIN" != 1 ]; then
  # Baseline counted from a FRESHLY PULLED log — a stale local copy deadlocks
  # the wait (it already contains the marker from an earlier run).
  BEFORE=$(count_of "h13 train complete user=$USER_ID")
  log "train $USER_ID (baseline $BEFORE)"
  xcrun devicectl device process launch --device "$DEV" --terminate-existing "$APP" \
    --benchmark-h13-train --user "$USER_ID" --nax-arm on --condition C0 >/dev/null 2>&1
  wait_for "h13 train complete user=$USER_ID" $((BEFORE + 1))
  log "train done; cooldown ${COOLDOWN}s"
  sleep "$COOLDOWN"
fi

# Push this user's Mac-control adapter if it has been trained since the bulk
# stage. Single-source copies RENAME, so the destination carries the final
# component ("/mac") explicitly — see the trap note above.
MACDIR="$ROOT/train/checkpoints_mlx/h13_mac_control/$USER_ID"
if [ -f "$MACDIR/adapters.safetensors" ]; then
  TMPM=$(mktemp -d); mkdir -p "$TMPM/mac"
  cp "$MACDIR/adapters.safetensors" "$TMPM/mac/"
  xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$APP" --source "$TMPM/mac" \
    --destination "Documents/h13_adapters/$USER_ID/mac" >/dev/null 2>&1 \
    && log "staged mac adapter for $USER_ID" || log "WARN: mac adapter push failed"
  rm -rf "$TMPM"
else
  log "no mac adapter yet for $USER_ID (arm will be skipped)"
fi

BEFORE=$(count_of "h13 eval ALL DONE user=$USER_ID")
log "eval $USER_ID arms=$ARMS (baseline $BEFORE)"
xcrun devicectl device process launch --device "$DEV" --terminate-existing "$APP" \
  --benchmark-h13-eval --user "$USER_ID" --arm "$ARMS" --nax-arm on >/dev/null 2>&1
wait_for "h13 eval ALL DONE user=$USER_ID" $((BEFORE + 1))

mkdir -p "$PULL/$USER_ID" "$ROOT/results/ondevice/h13_telemetry"
TMP=$(mktemp -d)
xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$APP" --source "Documents/h13_preds/$USER_ID" \
  --destination "$TMP/" >/dev/null 2>&1 || true
cp "$TMP"/*.jsonl "$PULL/$USER_ID/" 2>/dev/null || true
rm -rf "$TMP"
for F in train_bench_metrics_h13_nax-on.jsonl eval_bench_metrics_h13_nax-on.jsonl; do
  xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$APP" --source "Documents/$F" \
    --destination "$ROOT/results/ondevice/h13_telemetry/$F" >/dev/null 2>&1 || true
done
# The device adapter is a deliverable in its own right (device-vs-Mac weights).
xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$APP" --source "Documents/h13_adapters/$USER_ID/device" \
  --destination "$ROOT/results/ondevice/h13_device_adapters/$USER_ID/" >/dev/null 2>&1 || true
log "user $USER_ID COMPLETE: $(ls "$PULL/$USER_ID" 2>/dev/null | tr '\n' ' ')"
