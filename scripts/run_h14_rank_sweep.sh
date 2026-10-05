#!/usr/bin/env bash
# h14 rank/placement cost ablation (reviewer ask): does a bigger adapter cost anything on-device?
# Configs (all 36 blocks, GC on, repaired kernel, caps 512 and 1024, 20 iterations per cell):
#   A  r8  q,v            (paper configuration)
#   B  r32 q,v
#   C  r8  q,k,v,o,gate,up,down
#   D  r32 q,k,v,o,gate,up,down
#   E  r64 q,k,v,o,gate,up,down
# Two passes in opposite order (A..E, E..A) so every config is measured both early and late in
# the thermal history; compare within-pass and averaged.
set -uo pipefail
DEV=00008150-000674C60A3B401C; BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; OUTDIR="$REPO/results/ondevice"; STAMP=$(date +%F)
DEVFILE="Documents/train_bench_metrics_h14_nax-on.jsonl"
kill_resident() { local pid; pid=$(xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1); [[ -n "${pid:-}" ]] && { xcrun devicectl device process signal --device "$DEV" --signal SIGKILL --pid "$pid" >/dev/null 2>&1 || true; sleep 3; }; }
run() { local label=$1 rank=$2 keys=$3
  echo "=== [$(date '+%H:%M:%S')] $label rank=$rank keys=$keys ==="; kill_resident
  xcrun devicectl device process launch --device "$DEV" --terminate-existing --console "$BID" \
    --benchmark-train --nax-arm on --lora-layers 36 --gc on --iterations 20 --caps 512,1024 --lora-rank "$rank" --lora-keys "$keys" 2>&1 | grep -E "complete windows|first window|error" | sed 's/^/    /'
  echo "=== [$(date '+%H:%M:%S')] $label exited ==="; }
pass() { for cfg in "$@"; do case $cfg in
  A) run A 8 q,v;; B) run B 32 q,v;; C) run C 8 q,k,v,o,gate,up,down;; D) run D 32 q,k,v,o,gate,up,down;; E) run E 64 q,k,v,o,gate,up,down;; esac; done; }
pass A B C D E
pass E D C B A
xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer --domain-identifier "$BID" \
  --source "$DEVFILE" --destination "$OUTDIR/train_bench_metrics_h14_nax-on_${STAMP}_rank.jsonl" >/dev/null 2>&1 && echo "[pull] -> $OUTDIR/train_bench_metrics_h14_nax-on_${STAMP}_rank.jsonl"
echo "RANK SWEEP DONE"
