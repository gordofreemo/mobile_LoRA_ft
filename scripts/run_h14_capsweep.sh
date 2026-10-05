#!/usr/bin/env bash
# h14 (2026-09-11): memory-consistency cap sweep for the paper.
#   arm A  SmolLM3 (a1lamp fused 4-bit), 36 LoRA blocks, GC ON,  NAX on
#   arm B  SmolLM3 (a1lamp fused 4-bit), 36 LoRA blocks, GC OFF, NAX on  (expects jetsam at 512)
#   arm C  Qwen3-8B-4bit,                36 LoRA blocks, GC ON,  NAX on  (feasibility of "8B trains?")
#   arm D  Qwen3-8B-4bit,                36 LoRA blocks, GC OFF, NAX on
# One process launch per arm; a jetsam ends the arm, records for the smaller
# caps persist on device (cap_start sentinel marks the dead cap). Pull after each.
#
# Usage: scripts/run_h14_capsweep.sh A B        # run these arms in order
set -uo pipefail
DEV=00008150-000674C60A3B401C
BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO/results/ondevice"
STAMP=$(date +%Y-%m-%d)
DEVFILE="Documents/train_bench_metrics_h14_nax-on.jsonl"
ITERS=${ITERS:-40}
CAPS=${CAPS:-32,64,128,256,512,1024}
mkdir -p "$OUTDIR"

kill_resident() {
  local pid
  pid=$(xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1)
  if [[ -n "${pid:-}" ]]; then
    echo "[kill] resident LLMEval pid=$pid"
    xcrun devicectl device process signal --device "$DEV" --signal SIGKILL --pid "$pid" >/dev/null 2>&1 || true
    sleep 3
  fi
}
pull() { xcrun devicectl device copy from --device "$DEV" \
  --domain-type appDataContainer --domain-identifier "$BID" \
  --source "$DEVFILE" --destination "$1" >/dev/null 2>&1 || true; }

run_arm() {
  local label=$1; shift
  echo "=== [$(date '+%H:%M:%S')] arm $label: $* ==="
  kill_resident
  # --console blocks until exit (clean or jetsam). Do NOT kill this monitor:
  # devicectl --console propagates SIGTERM to the app.
  xcrun devicectl device process launch --device "$DEV" --terminate-existing --console "$BID" \
    --benchmark-train --nax-arm on --lora-layers 36 --iterations "$ITERS" --caps "$CAPS" "$@"
  echo "=== [$(date '+%H:%M:%S')] arm $label exited rc=$? ==="
  pull "$OUTDIR/train_bench_metrics_h14_nax-on_${STAMP}.jsonl"
  echo "[pull] -> $OUTDIR/train_bench_metrics_h14_nax-on_${STAMP}.jsonl"
}

for ARM in "$@"; do
  case "$ARM" in
    A) run_arm A --gc on ;;
    B) run_arm B --gc off ;;
    C) run_arm C --gc on  --model mlx-community/Qwen3-8B-4bit ;;
    D) run_arm D --gc off --model mlx-community/Qwen3-8B-4bit ;;
    *) echo "unknown arm $ARM" ;;
  esac
done
