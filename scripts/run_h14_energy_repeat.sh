#!/usr/bin/env bash
# h14 energy repeat: unplugged (C2) 405-profile run on the repaired kernel, 28 blocks,
# same seeded batch order as the campaign's XS point. Start with the phone UNPLUGGED at
# <=85% charge (fuel-gauge plateau rule), unlocked, Auto-Lock Never, airplane mode as in h9.
#   nohup caffeinate -ims scripts/run_h14_energy_repeat.sh > /tmp/h14_energy.log 2>&1 &
# Wireless devicectl may be down while unplugged; the run is self-contained on the phone.
# Afterwards (USB reconnected): python3 eval/h14_energy_summary.py results/ondevice/train_bench_metrics_naxab_e2e_h14energy_<date>.jsonl
set -uo pipefail
DEV=00008150-000674C60A3B401C; BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; OUTDIR="$REPO/results/ondevice"
TAG=h14energy
pid=$(xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1)
[[ -n "${pid:-}" ]] && xcrun devicectl device process signal --device "$DEV" --signal SIGKILL --pid "$pid" >/dev/null 2>&1; sleep 3
echo "=== [$(date '+%F %T')] launch $TAG (C2, 28 blocks) ==="
xcrun devicectl device process launch --device "$DEV" --terminate-existing "$BID" \
  --benchmark-train-e2e --user u00008075 --condition C2 --nax-arm on --lora-layers 28 --run-tag "$TAG" 2>&1 | grep -iE "Launched|error"
for i in $(seq 1 300); do sleep 60; xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -qi "LLMEval.app/LLMEval" || break; done
echo "=== [$(date '+%F %T')] app exited; pulling ==="
xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer --domain-identifier "$BID" \
  --source "Documents/train_bench_metrics_naxab_e2e_${TAG}.jsonl" --destination "$OUTDIR/train_bench_metrics_naxab_e2e_${TAG}_$(date +%F).jsonl" && echo "pulled"
