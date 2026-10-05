#!/usr/bin/env bash
# h15 campaign E (2026-09-22): does ANY application-chosen schedule raise work done
# per wall-clock hour under sustained load, by more than run-to-run variation?
#
# The existing evidence is four single runs on three different days, and the answer
# FLIPS SIGN with chassis state (0.76x off-kernel, 1.10x cool chassis, 0.92x hot).
# Chassis history and ambient are the confounds, so every arm is measured in the same
# hour as its control, several times, in BOTH orders.
#
#   soak 45 min (discard) then: C P1 C P2 C B C P2 C P1 C B C
#
#   C  continuous            --selflimit-delay 0, 20 min
#   P1 pacing 1 s per step   --selflimit-delay 1, 20 min
#   P2 pacing 3 s per step   --selflimit-delay 3, 20 min
#   B  bursts                two launches of 10 min, 120 s killed+idle between
#
# Metric: steps per wall-clock MINUTE, pauses included, over minutes 5-20 of each
# block (the first five absorb the transition from the previous block). Each scheduled
# block is compared to the mean of the two C blocks bracketing it: three paired ratios
# for P1 and P2, two for B. The spread of the C blocks themselves is the noise floor.
#
# A synthetic 500-token step throughout, so every block does identical work.
# PLUGGED, screen on, Auto-Lock Never.
#
# NOTE: --run-tag does NOT reach the self-limit writer; every block appends to
# Documents/train_bench_metrics_selflimit_nax-on.jsonl. Blocks are separated by
# bench_session_id and arms by delay_s, and this script writes a MANIFEST mapping
# block index -> arm -> wall-clock window, because B and C share delay_s=0 and are
# otherwise distinguishable only by structure.
#
# Run: nohup caffeinate -ims bash scripts/run_h15_e_schedule.sh > /tmp/h15_e.log 2>&1 &
set -uo pipefail
DEV=00008150-000674C60A3B401C
BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "$REPO/results" ] || { echo "FATAL: $REPO/results not found"; exit 3; }
OUTDIR="$REPO/results/ondevice"; mkdir -p "$OUTDIR"
STAMP=$(date +%Y-%m-%d)
DEVFILE="Documents/train_bench_metrics_selflimit_nax-on.jsonl"
MANIFEST="$OUTDIR/h15e_schedule_manifest_${STAMP}.tsv"
SOAK_MIN=${SOAK_MIN:-45}
BLOCK_MIN=${BLOCK_MIN:-20}
BURST_MIN=${BURST_MIN:-10}
BURST_REST=${BURST_REST:-120}
SEQ=${SEQ:-"C P1 C P2 C B C P2 C P1 C B C"}

log(){ echo "$(date '+%F %T') $*"; }
kill_resident() {
  local pid
  pid=$(xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1)
  [ -n "${pid:-}" ] && { xcrun devicectl device process signal --device "$DEV" --signal SIGKILL --pid "$pid" >/dev/null 2>&1; sleep 3; }
  return 0
}
launch_block() {  # launch_block <delay_s> <minutes> <tag>
  xcrun devicectl device process launch --device "$DEV" --terminate-existing --console "$BID" \
    --benchmark-thermal-selflimit --selflimit-delay "$1" --selflimit-minutes "$2" \
    --nax-arm on --run-tag "$3" 2>&1 | tail -1
}
pull(){ xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$BID" --source "$DEVFILE" --destination "$1" >/dev/null 2>&1 || true; }

printf "block\tarm\tdelay_s\tstart_utc\tend_utc\n" > "$MANIFEST"
log "==== h15 campaign E: schedule arms, sequence [$SEQ] ===="
log "manifest -> $MANIFEST"

log "=== soak ${SOAK_MIN} min (discarded) ==="
kill_resident
launch_block 0 "$SOAK_MIN" "sched_soak"
log "=== soak done; blocks run back to back, no idle beyond the model reload ==="

i=0
for arm in $SEQ; do
  i=$((i+1))
  s=$(date -u +%FT%TZ)
  case "$arm" in
    C)  log "=== block $i: C  (continuous, delay 0, ${BLOCK_MIN}m) ==="
        kill_resident; launch_block 0 "$BLOCK_MIN" "sched_C_$i"; d=0 ;;
    P1) log "=== block $i: P1 (pace 1s, ${BLOCK_MIN}m) ==="
        kill_resident; launch_block 1 "$BLOCK_MIN" "sched_P1_$i"; d=1 ;;
    P2) log "=== block $i: P2 (pace 3s, ${BLOCK_MIN}m) ==="
        kill_resident; launch_block 3 "$BLOCK_MIN" "sched_P2_$i"; d=3 ;;
    B)  log "=== block $i: B  (burst ${BURST_MIN}m, rest ${BURST_REST}s, burst ${BURST_MIN}m) ==="
        kill_resident; launch_block 0 "$BURST_MIN" "sched_B_${i}a"
        log "    burst 1 done; killed idle ${BURST_REST}s"
        kill_resident; sleep "$BURST_REST"
        launch_block 0 "$BURST_MIN" "sched_B_${i}b"; d=0 ;;
    *)  log "unknown arm $arm — skipping"; continue ;;
  esac
  e=$(date -u +%FT%TZ)
  printf "%s\t%s\t%s\t%s\t%s\n" "$i" "$arm" "$d" "$s" "$e" >> "$MANIFEST"
  log "=== block $i ($arm) done ==="
done

pull "$OUTDIR/train_bench_metrics_selflimit_nax-on_h15e_${STAMP}.jsonl"
log "==== campaign E complete -> $OUTDIR/train_bench_metrics_selflimit_nax-on_h15e_${STAMP}.jsonl ===="
log "==== manifest -> $MANIFEST ===="
