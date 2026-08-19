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
set -uo pipefail

export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
DEV=00008150-000674C60A3B401C
DCTL_TIMEOUT=${DCTL_TIMEOUT:-180}
APP=mlx.LLMEvalJGW9U9Y36Y
ROOT="$HOME/Documents/Research/mobile_LoRA_ft"
PULL="$ROOT/results/ondevice/h13_preds"
COOLDOWN=${COOLDOWN:-300}          # h10: t95 ~ 346 s after a training burst
ARMS=${ARMS:-all}
SKIP_TRAIN=${SKIP_TRAIN:-0}
USER_ID=$1
LOG=/tmp/h13_stderr_$USER_ID.log

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_device() {
  local tries=0
  until xcrun devicectl list devices --timeout 60 2>/dev/null \
        | grep -E "available \(paired\)|connected" | grep -qv "unavailable"; do
    tries=$((tries + 1))
    [ "$tries" -eq 1 ] && log "device unavailable, waiting (unlock it if it is asleep)"
    [ "$tries" -gt 240 ] && { log "device still unavailable after 2 h"; return 1; }
    sleep 30
  done
  return 0
}

# Launch with retries. A phone that auto-locked, dropped its tunnel, or was busy
# returns non-zero here, and that must never be fatal to the campaign.
launch() {
  local tries=0
  while [ "$tries" -lt 40 ]; do
    wait_for_device || return 1
    if xcrun devicectl device process launch --timeout "$DCTL_TIMEOUT" --device "$DEV" \
         --terminate-existing "$APP" "$@" >/dev/null 2>&1; then
      return 0
    fi
    tries=$((tries + 1))
    # A locked phone still reports "available (paired)" but refuses the launch
    # (FBSOpenApplicationErrorDomain error 7), so ride it out rather than
    # skipping the user. Set Auto-Lock to Never to avoid this entirely.
    log "launch failed (attempt $tries), retrying in 90 s"
    sleep 90
  done
  log "launch failed 40x (1 h), giving up on this step; a later queue pass retries"
  return 1
}

pull_log() {
  rm -f "$LOG"
  xcrun devicectl device copy from --timeout "$DCTL_TIMEOUT" --device "$DEV" \
    --domain-type appDataContainer \
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
  launch --benchmark-h13-train --user "$USER_ID" --nax-arm on --condition C0 || exit 1
  wait_for "h13 train complete user=$USER_ID" $((BEFORE + 1)) || exit 1
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
  xcrun devicectl device copy to --timeout "$DCTL_TIMEOUT" --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$APP" --source "$TMPM/mac" \
    --destination "Documents/h13_adapters/$USER_ID/mac" >/dev/null 2>&1 \
    && log "staged mac adapter for $USER_ID" || log "WARN: mac adapter push failed"
  rm -rf "$TMPM"
else
  log "no mac adapter yet for $USER_ID (arm will be skipped)"
fi

BEFORE=$(count_of "h13 eval ALL DONE user=$USER_ID")
log "eval $USER_ID arms=$ARMS (baseline $BEFORE)"
launch --benchmark-h13-eval --user "$USER_ID" --arm "$ARMS" --nax-arm on || exit 1
wait_for "h13 eval ALL DONE user=$USER_ID" $((BEFORE + 1)) || exit 1

mkdir -p "$PULL/$USER_ID" "$ROOT/results/ondevice/h13_telemetry"
TMP=$(mktemp -d)
xcrun devicectl device copy from --timeout 600 --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$APP" --source "Documents/h13_preds/$USER_ID" \
  --destination "$TMP/" >/dev/null 2>&1 || true
cp "$TMP"/*.jsonl "$PULL/$USER_ID/" 2>/dev/null || true
rm -rf "$TMP"
for F in train_bench_metrics_h13_nax-on.jsonl eval_bench_metrics_h13_nax-on.jsonl; do
  xcrun devicectl device copy from --timeout 600 --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$APP" --source "Documents/$F" \
    --destination "$ROOT/results/ondevice/h13_telemetry/$F" >/dev/null 2>&1 || true
done
# The device adapter is a deliverable in its own right (device-vs-Mac weights).
xcrun devicectl device copy from --timeout 600 --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$APP" --source "Documents/h13_adapters/$USER_ID/device" \
  --destination "$ROOT/results/ondevice/h13_device_adapters/$USER_ID/" >/dev/null 2>&1 || true
log "user $USER_ID COMPLETE: $(ls "$PULL/$USER_ID" 2>/dev/null | tr '\n' ' ')"
