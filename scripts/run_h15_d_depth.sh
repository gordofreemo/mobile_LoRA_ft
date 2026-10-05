#!/usr/bin/env bash
# h15 campaign D (2026-09-21): adapter-DEPTH sweep.
#
# Tests the mechanism claim behind Section 5.2/6.2 — "adapter depth determines backward
# compute, rank does not" — which currently rests on two points measured a week apart.
#
# Prediction: step time is linear in the number of adapted blocks N,
#     t(N) = t_fwd + N * t_blk
# intercept = the full forward pass (every block runs forward regardless of adaptation),
# slope = one block's backward pass plus its recomputation. From 5.1 the intercept should
# be ~20% of the 36-block step. Peak memory should also fall with N, since checkpointed
# activations of blocks before the first adapted one need not be retained.
#
# Thermal control is by SOAK THEN INTERLEAVE, not by avoidance: consecutive per-op arms
# in this project drift by up to 1.4x even at the plateau, so the three-pass
# forward/reverse/forward order matters more than the soak length. Drift that is
# monotonic in time cancels in the per-depth mean.
#
#   45-min continuous soak -> pass 1 forward -> pass 2 reversed -> pass 3 forward
#
# Phone must be PLUGGED (this is a throughput/memory experiment, not an energy one).
# Run: nohup caffeinate -ims bash scripts/run_h15_d_depth.sh > /tmp/h15_d.log 2>&1 &
set -uo pipefail
DEV=00008150-000674C60A3B401C
BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "$REPO/results" ] || { echo "FATAL: $REPO/results not found"; exit 3; }
OUTDIR="$REPO/results/ondevice"; mkdir -p "$OUTDIR"
DEVFILE="Documents/train_bench_metrics_h14_nax-on.jsonl"
DEPTHS=${DEPTHS:-"36 30 24 18 12 6"}
CAPS=${CAPS:-512,1024}
ITERS=${ITERS:-30}
SOAK_MIN=${SOAK_MIN:-45}

log(){ echo "$(date '+%F %T') $*"; }
kill_resident() {
  local pid
  pid=$(xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1)
  [ -n "${pid:-}" ] && { xcrun devicectl device process signal --device "$DEV" --signal SIGKILL --pid "$pid" >/dev/null 2>&1; sleep 3; }
  return 0
}
pull() { xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$BID" --source "$DEVFILE" --destination "$1" >/dev/null 2>&1 || true; }

cell() {  # cell <pass> <depth>
  local pass=$1 n=$2
  log "=== pass $pass depth $n (caps $CAPS, $ITERS iters) ==="
  kill_resident
  # --console blocks until the process exits. Never kill this monitor: devicectl
  # --console propagates SIGTERM to the app.
  xcrun devicectl device process launch --device "$DEV" --terminate-existing --console "$BID" \
    --benchmark-train --nax-arm on --gc on --lora-layers "$n" \
    --iterations "$ITERS" --caps "$CAPS" 2>&1 | grep -viE "^\s*$" | tail -2
  log "=== pass $pass depth $n done ==="
}

log "==== h15 campaign D: depth sweep, depths [$DEPTHS], caps $CAPS ===="
log "=== soak: ${SOAK_MIN} min continuous, to reach the throttled plateau ==="
kill_resident
xcrun devicectl device process launch --device "$DEV" --terminate-existing --console "$BID" \
  --benchmark-thermal-selflimit --selflimit-delay 0 --selflimit-minutes "$SOAK_MIN" --nax-arm on 2>&1 | tail -2
log "=== soak complete, starting cells back to back (no idle: stay at the plateau) ==="

FWD=($DEPTHS)
REV=(); for ((i=${#FWD[@]}-1;i>=0;i--)); do REV+=("${FWD[i]}"); done
for n in "${FWD[@]}"; do cell 1 "$n"; done
for n in "${REV[@]}"; do cell 2 "$n"; done
for n in "${FWD[@]}"; do cell 3 "$n"; done

STAMP=$(date +%Y-%m-%d)
pull "$OUTDIR/train_bench_metrics_h14_nax-on_depth_${STAMP}.jsonl"
log "==== campaign D complete -> $OUTDIR/train_bench_metrics_h14_nax-on_depth_${STAMP}.jsonl ===="
