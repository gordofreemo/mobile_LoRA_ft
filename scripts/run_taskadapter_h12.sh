#!/bin/bash
# h12 full overnight run launcher — Per-Task-LoRA (LaMP-7) on-device, NAX ON.
#
# Encodes the h11 orchestration lessons:
#  1. NEVER launch with --console (SIGTERM propagates to the app when the
#     console session dies) — launch detached, poll the device JSONL.
#  2. A resident app SILENTLY ABSORBS new launch args — SIGKILL any existing
#     LLMEval PID before EVERY launch.
#  3. Run this script under nohup if driven by an agent/session that may be
#     killed: nohup bash scripts/run_taskadapter_h12.sh > /tmp/h12_launch.log 2>&1 &
#
# Preconditions (operator): device plugged, airplane mode ON (model already
# cached — the run needs no network), min brightness, flat hard surface,
# device unlocked at launch time.

set -euo pipefail

DEVICE=00008150-000674C60A3B401C
BUNDLE=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

CONDITION="${1:-c0_plugged}"
IDLE_MINUTES="${2:-}"

# Kill any resident instance (lesson 2). No capture mode in h12, so SIGKILL is safe.
PID=$(xcrun devicectl device info processes --device "$DEVICE" 2>/dev/null \
    | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1 || true)
if [ -n "$PID" ]; then
    echo "killing resident LLMEval pid $PID"
    xcrun devicectl device process signal --signal SIGKILL --pid "$PID" --device "$DEVICE" || true
    sleep 3
fi

ARGS=(--benchmark-train-taskadapter --nax-arm on --condition "$CONDITION")
if [ -n "$IDLE_MINUTES" ]; then
    ARGS+=(--idle-minutes "$IDLE_MINUTES")
fi

echo "launching full h12 run: ${ARGS[*]}"
xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" "${ARGS[@]}"
echo "launched detached at $(date). Poll with:"
echo "  xcrun devicectl device copy from --device $DEVICE --domain-type appDataContainer \\"
echo "    --domain-identifier $BUNDLE --source Documents/train_bench_metrics_taskadapter.jsonl \\"
echo "    --destination /tmp/devpull/taskadapter_$(date +%Y%m%d).jsonl"
