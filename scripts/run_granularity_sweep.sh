#!/usr/bin/env bash
# Driver for the h8 GC-granularity sweep: launches ONE process per K value in
# TrainBenchConstants.granularityKValues, ascending, each via `devicectl
# device process launch --console` (blocks on that process's exit — clean or
# jetsam) before moving to the next K. A jetsam/OOM at a given K is valid
# data, not a failure — this driver does NOT retry, it just continues to the
# next K regardless of exit code. See
# experiments/2026-07-25-ondevice-gc-granularity-plan.md.
#
# Each cell's own cooldown-to-nominal (uncapped, by design — "impossible for
# the phone not to cool down") + 10min buffer happens INSIDE the app before
# training starts, so a single K's `--console` wait can legitimately be many
# minutes to over an hour. The device must stay unlocked/awake for the full
# duration of each launch — devicectl launches fail with
# FBSOpenApplicationErrorDomain error 7 while the device is locked (a
# recurring operational gotcha in prior on-device rounds: h6, E2E).
#
# Usage:
#   scripts/run_granularity_sweep.sh              # full ascending sweep
#   scripts/run_granularity_sweep.sh 2 3 4         # only these K, in order
#                                                   # (e.g. resume a partial
#                                                   # sweep or re-run one K)
set -uo pipefail  # not -e: a jetsam'd launch's non-zero exit must not abort the sweep

DEV=00008150-000674C60A3B401C
BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO/results/ondevice"
OUTFILE="$OUTDIR/train_bench_metrics_granularity_$(date +%Y-%m-%d).jsonl"
DEVFILE="Documents/train_bench_metrics_granularity.jsonl"

if [[ $# -gt 0 ]]; then
  K_VALUES=("$@")
else
  K_VALUES=(1 2 3 4 6 9 12 18 36)
fi

mkdir -p "$OUTDIR"

pull() { xcrun devicectl device copy from --device "$DEV" \
  --domain-type appDataContainer --domain-identifier "$BID" \
  --source "$DEVFILE" --destination "$1" >/dev/null 2>&1 || true; }

for K in "${K_VALUES[@]}"; do
  echo "=== [$(date '+%H:%M:%S')] K=$K: launching (on-device cooldown-to-nominal + 10min buffer runs first) ==="
  xcrun devicectl device process launch --device "$DEV" --terminate-existing --console "$BID" \
    --benchmark-train-granularity --granularity-k "$K"
  RC=$?
  echo "=== [$(date '+%H:%M:%S')] K=$K: process exited rc=$RC (non-zero may mean a clean harness exit path, a jetsam, or a device-lock launch failure — check the pulled JSONL below for this K's k_start/train/error records to tell them apart) ==="
  pull "$OUTFILE"
done

echo "[pull] full device JSONL -> $OUTFILE"
python3 - "$OUTFILE" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by_k = {}
for r in recs:
    k = r.get("checkpoint_granularity")
    by_k.setdefault(k, {"k_start": 0, "train": 0, "error": 0})
    rt = r.get("record_type")
    if rt in by_k[k]:
        by_k[k][rt] += 1
print("  summary by K (k_start / train windows / error):")
for k in sorted(by_k):
    c = by_k[k]
    status = "OK" if c["train"] >= 19 else ("OOM/ERROR" if c["error"] else "INCOMPLETE")
    print(f"    K={k:>2}: k_start={c['k_start']} train={c['train']} error={c['error']}  [{status}]")
PY
