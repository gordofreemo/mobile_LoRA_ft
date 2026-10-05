#!/usr/bin/env bash
# h15 campaign A (2026-09-17): all five cost-law / thermal-collapse points in ONE
# session, plugged, repaired kernel, 36 LoRA blocks — the configuration Table 1
# DECLARES, measured directly instead of scaled.
#
# Why 36 and not the published 28: Table 5 and Table 6 are 28-block runs scaled to 36
# by a factor of 1.21 that rests on ONE 36-block run (6872 s, h14rep3) on ONE profile
# (405). That factor is assumed profile-independent and was never tested. Measuring all
# five profiles at 36 removes the assumption, retires the Table 1 caption erratum and
# §10's "understate the 36-block configuration by 21%", and turns 1.21x into a
# five-profile measurement. The published 28-block points are kept, not replaced.
#
# Secondary: the published 448/500/550/987 points come from the 2026-08-12..15 NAX-ON
# campaign (37 C heat-wave week) while the 405 mean comes from 2026-09-11/12, so the
# cost-law fit mixes two ambients. Running all five back to back fixes that too.
#
# Takes a list of profiles; defaults to all five, shortest first:
#   bash scripts/run_h15_a_cost.sh 405 448        # overnight block
#   bash scripts/run_h15_a_cost.sh 500 550 987    # resume later
# Splitting matters because the phone leaves the desk during the workday: a run must
# never be in flight when the device is unplugged and pocketed.
# 60-minute idle BEFORE each run, so every run starts from a nominal chassis
# (95% throughput recovery takes 344 s, so 3600 s is ample).
#
# Phone: PLUGGED, unlocked, Auto-Lock Never, same surface/orientation throughout.
# Run detached:
#   nohup caffeinate -ims scripts/run_h15_a_cost.sh > /tmp/h15_a.log 2>&1 &
# Expect ~28 h (22.7 h of runs + 5 h of idles).
set -uo pipefail
DEV=00008150-000674C60A3B401C
BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
# Resolve the repo root by git, not by directory depth: these scripts are also run
# from snapshot copies under scripts/.h15run/, where dirname/.. lands in scripts/ and
# silently wrote results to mobile_LoRA_ft/scripts/results/ (caught 2026-09-18).
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "$REPO/results" ] || { echo "FATAL: $REPO/results not found — refusing to run"; exit 3; }
OUTDIR="$REPO/results/ondevice"; mkdir -p "$OUTDIR"
IDLE_S=${IDLE_S:-3600}
LAYERS=${LAYERS:-36}
# SKIP_FIRST_IDLE=1 drops the lead idle when the phone is already cool (e.g. resuming
# after a commute). The idle BETWEEN runs is always kept — it is the protocol.
SKIP_FIRST_IDLE=${SKIP_FIRST_IDLE:-0}
# DEADLINE="HH:MM" (today, 24h) skips any run whose projected end would fall past it,
# instead of letting it be cut off mid-run when the device leaves the desk. Same idea
# as the latest-start gate in scripts/nax_rerun_night3_plugged.sh.
DEADLINE=${DEADLINE:-}
DEADLINE_EPOCH=""
if [ -n "$DEADLINE" ]; then
  DEADLINE_EPOCH=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%F) ${DEADLINE}:00" +%s 2>/dev/null || true)
  # a deadline already past today means tomorrow
  if [ -n "$DEADLINE_EPOCH" ] && [ "$DEADLINE_EPOCH" -le "$(date +%s)" ]; then
    DEADLINE_EPOCH=$(( DEADLINE_EPOCH + 86400 ))
  fi
fi
FIRST_RUN=1

log() { echo "$(date '+%F %T') $*"; }

kill_resident() {
  local pid
  pid=$(xcrun devicectl device info processes --device "$DEV" 2>/dev/null \
        | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1)
  [[ -n "${pid:-}" ]] && { log "[kill] resident pid=$pid"; \
    xcrun devicectl device process signal --device "$DEV" --signal SIGKILL --pid "$pid" >/dev/null 2>&1 || true; sleep 3; }
  return 0
}

pull() {
  xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$BID" --source "Documents/$1" --destination "$2" >/dev/null 2>&1 || true
}

wait_for_exit() {  # $1 = timeout_s
  local t=0 limit=$1
  while (( t < limit )); do
    sleep 60; t=$((t+60))
    xcrun devicectl device info processes --device "$DEV" 2>/dev/null \
      | grep -qi "LLMEval.app/LLMEval" || { log "process exited after ${t}s"; return 0; }
  done
  log "[timeout] run exceeded ${limit}s — killing"; kill_resident; return 1
}


# A resident app silently no-ops new CLI args (devicectl just foregrounds it), so the
# SIGKILL must happen immediately before the launch, not before an hour of idle.
# After launching, confirm the run actually started by checking that its metrics file
# appears on the device; retry once if it did not. Caught 2026-09-18 when h15a448
# launched into a live process and trained nothing.
launch_verified() {
  local tag=$1 user=$2 timeout_s=$3 attempt
  for attempt in 1 2; do
    kill_resident; sleep 5
    log "=== launch $tag user=$user layers=$LAYERS C0 nax=on (attempt $attempt) ==="
    xcrun devicectl device process launch --device "$DEV" --terminate-existing "$BID" \
      --benchmark-train-e2e --user "$user" --condition C0 --nax-arm on \
      --lora-layers "$LAYERS" --run-tag "$tag" 2>&1 | grep -iE "Launched|error" || true
    # Probe by PULLING the one file, not by listing the container: the full listing takes
    # 90-180s on this device and truncates, which on 2026-09-19 reported a healthy run as
    # "no metrics file" and killed it to retry. A targeted copy answers in seconds.
    local w=0
    while (( w < 180 )); do
      sleep 20; w=$((w+20))
      if timeout 60 xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
           --domain-identifier "$BID" --source "Documents/train_bench_metrics_naxab_e2e_${tag}.jsonl" \
           --destination "/tmp/h15_startcheck_${tag}.jsonl" >/dev/null 2>&1; then
        log "=== $tag confirmed started (metrics file present after ${w}s) ==="; return 0
      fi
    done
    log "=== $tag produced no metrics file in 180s — app likely ignored args, retrying ==="
  done
  return 1
}

# run_point <tag> <user> <timeout_s> <expected_s>
run_point() {
  local tag=$1 user=$2 timeout_s=$3 expected_s=$4
  local lead=$IDLE_S
  if [ "$FIRST_RUN" = "1" ] && [ "$SKIP_FIRST_IDLE" = "1" ]; then
    lead=0; log "=== lead idle skipped for $tag (SKIP_FIRST_IDLE=1) ==="
  fi
  FIRST_RUN=0
  if [ -n "$DEADLINE_EPOCH" ]; then
    local projected=$(( $(date +%s) + lead + expected_s ))
    if [ "$projected" -gt "$DEADLINE_EPOCH" ]; then
      log "=== SKIP $tag: projected end $(date -r "$projected" '+%F %H:%M') is past deadline $(date -r "$DEADLINE_EPOCH" '+%F %H:%M') ==="
      return 0
    fi
    log "=== $tag projected end $(date -r "$projected" '+%H:%M'), deadline $(date -r "$DEADLINE_EPOCH" '+%H:%M') — proceeding ==="
  fi
  log "=== idle ${lead}s before $tag ($user) ==="
  kill_resident; [ "$lead" -gt 0 ] && sleep "$lead"
  launch_verified "$tag" "$user" "$timeout_s" || { log "=== $tag NEVER STARTED after 2 attempts — skipping ==="; return 1; }
  wait_for_exit "$timeout_s"
  pull "train_bench_metrics_naxab_e2e_${tag}.jsonl" \
       "$OUTDIR/train_bench_metrics_naxab_e2e_${tag}_$(date +%F).jsonl"
  log "=== $tag done -> $OUTDIR/train_bench_metrics_naxab_e2e_${tag}_$(date +%F).jsonl ==="
}

log "==== h15 campaign A start (plugged, ${LAYERS} blocks, repaired kernel) ===="
# profile -> user, timeout_s, expected_s (36 blocks)
dispatch() {
  case "$1" in
    405) run_point h15a405 u00008075 13200  6850 ;;   # 1215 it
    448) run_point h15a448 u00005228 26400 16100 ;;   # 1344 it
    500) run_point h15a500 u00011077 26400 14400 ;;   # 1500 it
    550) run_point h15a550 u00005020 26400 16100 ;;   # 1650 it
    987) run_point h15a987 u00012502 43200 28400 ;;   # 2961 it
    *) log "unknown profile '$1' — skipping" ;;
  esac
}

PROFILES=("$@")
[ ${#PROFILES[@]} -eq 0 ] && PROFILES=(405 448 500 550 987)
log "profiles this invocation: ${PROFILES[*]}"
for pr in "${PROFILES[@]}"; do dispatch "$pr"; done
log "==== h15 campaign A complete ===="
