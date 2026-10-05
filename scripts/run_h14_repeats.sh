#!/usr/bin/env bash
# h14 repeats (2026-09-11): variance for the 405-example (u00008075) end-to-end run.
# Three sustained-protocol C0 runs with the repaired kernel, each preceded by a
# 60-minute idle so every run starts from a nominal chassis:
#   rep1  28 LoRA blocks (matches the existing cost-law point set)
#   rep2  28 LoRA blocks
#   rep3  36 LoRA blocks (Table 1's declared configuration; measures the 28-vs-36 delta)
# Same seeded batch order as the 2026-08-07 A/B (--nax-arm on sets shuffleSeed), so
# run-to-run spread isolates the device, not the data order.
# Records go to Documents/train_bench_metrics_naxab_e2e_h14rep<N>.jsonl (--run-tag).
# Phone: plugged, unlocked, Auto-Lock Never. Run detached:
#   nohup caffeinate -ims scripts/run_h14_repeats.sh > /tmp/h14_repeats.log 2>&1 &
set -uo pipefail
DEV=00008150-000674C60A3B401C
BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO/results/ondevice"; mkdir -p "$OUTDIR"
USER_FP=u00008075
IDLE_S=${IDLE_S:-3600}
TIMEOUT_S=${TIMEOUT_S:-14400}   # 4 h per run safety ceiling (expected ~1.5 h)

kill_resident() {
  local pid
  pid=$(xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1)
  [[ -n "${pid:-}" ]] && { echo "[kill] resident pid=$pid"; xcrun devicectl device process signal --device "$DEV" --signal SIGKILL --pid "$pid" >/dev/null 2>&1 || true; sleep 3; }
}
pull() { xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$BID" --source "Documents/$1" --destination "$2" >/dev/null 2>&1 || true; }
wait_for_exit() {  # poll until no LLMEval process or timeout
  local t=0
  while (( t < TIMEOUT_S )); do
    sleep 60; t=$((t+60))
    xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -qi "LLMEval.app/LLMEval" || return 0
  done
  echo "[timeout] run exceeded ${TIMEOUT_S}s"; kill_resident; return 1
}

run_rep() {
  local tag=$1 layers=$2
  echo "=== [$(date '+%F %T')] idle ${IDLE_S}s before $tag ==="
  kill_resident; sleep "$IDLE_S"
  echo "=== [$(date '+%F %T')] launch $tag (lora-layers=$layers) ==="
  # Detached launch (no --console: devicectl --console propagates SIGTERM to the app).
  xcrun devicectl device process launch --device "$DEV" --terminate-existing "$BID" \
    --benchmark-train-e2e --user "$USER_FP" --condition C0 --nax-arm on \
    --lora-layers "$layers" --run-tag "$tag" 2>&1 | grep -iE "Launched|error" || true
  wait_for_exit
  pull "train_bench_metrics_naxab_e2e_${tag}.jsonl" "$OUTDIR/train_bench_metrics_naxab_e2e_${tag}_$(date +%F).jsonl"
  echo "=== [$(date '+%F %T')] $tag done -> $OUTDIR/train_bench_metrics_naxab_e2e_${tag}_$(date +%F).jsonl ==="
}

run_rep h14rep1 28
run_rep h14rep2 28
run_rep h14rep3 36
echo "=== [$(date '+%F %T')] ALL DONE ==="
