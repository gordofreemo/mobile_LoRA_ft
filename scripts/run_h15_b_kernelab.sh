#!/usr/bin/env bash
# h15 campaign B (2026-09-17): second end-to-end kernel A/B pair over the 405 profile.
#
# Why: the published 1.93x (Figure 4 / Section 6) is ONE run per arm, at 28 blocks.
# The interleaved 1.65-2.13x measurement is thermally matched, but the end-to-end
# factor is not — the paper itself attributes the 1.55 -> 1.93 gap to the stock arm
# spending 1.4 more hours throttled.
#
# Run at 36 blocks, matching campaigns A and C, so the whole paper reports one
# configuration. The ratio is NOT block-count-independent: the repair accelerates the
# backward pass, and at 36 blocks the backward pass covers all 36 layers instead of
# stopping at the first adapted one, so there is more of exactly the work the repair
# speeds up. Expect a ratio at or above the published 1.93x.
#
# Secondary result: the published 28-block 1.93x and this 36-block figure together
# measure how the repair's payoff scales with adapter depth, which is the lever
# Section 5.2 identifies as the one that moves backward-pass compute.
#
# --nax-arm also sets shuffleSeed, so both arms consume the identical seeded batch
# order (shuffle_seed 20260806, as in the 2026-08-07 A/B). Repaired arm first: it is
# the shorter run, so an interruption costs the cheaper half.
#
# Phone: PLUGGED, unlocked, Auto-Lock Never.
# Run detached:
#   nohup caffeinate -ims scripts/run_h15_b_kernelab.sh > /tmp/h15_b.log 2>&1 &
# Expect ~7.6 h (1.8 h repaired + 3.5-4.0 h stock + 2 h of idles).
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
USER_FP=u00008075
IDLE_S=${IDLE_S:-3600}
LAYERS=${LAYERS:-36}

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

wait_for_exit() {
  local t=0 limit=$1
  while (( t < limit )); do
    sleep 60; t=$((t+60))
    xcrun devicectl device info processes --device "$DEV" 2>/dev/null \
      | grep -qi "LLMEval.app/LLMEval" || { log "process exited after ${t}s"; return 0; }
  done
  log "[timeout] run exceeded ${limit}s — killing"; kill_resident; return 1
}

# run_arm <tag> <on|off> <timeout_s>
run_arm() {
  local tag=$1 arm=$2 timeout_s=$3
  log "=== idle ${IDLE_S}s before $tag (nax=$arm) ==="
  kill_resident; sleep "$IDLE_S"
  # SIGKILL immediately before launch: a resident app silently no-ops new CLI args.
  local attempt started=0
  for attempt in 1 2; do
    kill_resident; sleep 5
    log "=== launch $tag user=$USER_FP layers=$LAYERS C0 nax=$arm (attempt $attempt) ==="
    xcrun devicectl device process launch --device "$DEV" --terminate-existing "$BID" \
      --benchmark-train-e2e --user "$USER_FP" --condition C0 --nax-arm "$arm" \
      --lora-layers "$LAYERS" --run-tag "$tag" 2>&1 | grep -iE "Launched|error" || true
    # Targeted pull, not a container listing (see run_h15_a_cost.sh, 2026-09-19).
    local w=0
    while (( w < 180 )); do
      sleep 20; w=$((w+20))
      timeout 60 xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
        --domain-identifier "$BID" --source "Documents/train_bench_metrics_naxab_e2e_${tag}.jsonl" \
        --destination "/tmp/h15_startcheck_${tag}.jsonl" >/dev/null 2>&1 \
        && { log "=== $tag confirmed started after ${w}s ==="; started=1; break; }
    done
    [ "$started" = "1" ] && break
    log "=== $tag no metrics file in 180s — retrying ==="
  done
  [ "$started" = "1" ] || { log "=== $tag NEVER STARTED — skipping ==="; return 1; }
  wait_for_exit "$timeout_s"
  pull "train_bench_metrics_naxab_e2e_${tag}.jsonl" \
       "$OUTDIR/train_bench_metrics_naxab_e2e_${tag}_$(date +%F).jsonl"
  log "=== $tag done -> $OUTDIR/train_bench_metrics_naxab_e2e_${tag}_$(date +%F).jsonl ==="
}

log "==== h15 campaign B start (plugged, ${LAYERS} blocks, 405 profile) ===="
run_arm h15bon  on  13200   # expect ~6550 s at 36 blocks
run_arm h15boff off 28800   # expect ~13000-14500 s at 36 blocks
log "==== h15 campaign B complete ===="
log "Aggregate with: python3 eval/naxab_aggregate.py (or eval/plot_naxab_e2e.py)"
