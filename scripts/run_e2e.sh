#!/usr/bin/env bash
# Driver for one Phase-3 E2E on-device per-user training run.
#
# Launches the h5 e2e harness DETACHED (so the run survives this terminal),
# polls the device JSONL printing live progress (step / loss / iter-s / thermal),
# and on completion pulls the FULL cumulative device JSONL down to
# results/ondevice/ (the device file accumulates every run, so the pulled file
# is always the complete record set).
#
# The 14-run matrix is staged PHYSICALLY (plug/unplug, Low Power Mode, 3D game,
# cooldown) — this script drives one run at a time; you set up the condition.
#
# Usage:
#   scripts/run_e2e.sh --user u00008075 --condition C0            # full 3xn_user
#   scripts/run_e2e.sh --user u00005020 --condition C1 --max-iters 40   # smoke
#
# Users (fingerprint -> profile size): S u00008075/405, u00005228/448,
#   u00011077/500, M u00005020/550, u00013218/653, L u00012502/987.
# NOTE: runs with --max-iters < steps_per_report (10) emit NO train windows
# (the trainer reports loss only every 10 iters) — use >=10 for a visible smoke.
set -uo pipefail  # not -e: non-matching greps are normal in the poll loop

DEV=00008150-000674C60A3B401C
BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO/results/ondevice"
OUTFILE="$OUTDIR/train_bench_metrics_e2e_smollm3_a1lamp_$(date +%Y-%m-%d).jsonl"
DEVFILE="Documents/train_bench_metrics_e2e.jsonl"
POLL_SECONDS=30
MAX_WAIT_MIN=${MAX_WAIT_MIN:-900}  # safety ceiling 15h (covers L's ~12h overnight); override via env

USER_FP=""; CONDITION=""; MAX_ITERS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)      USER_FP="$2"; shift 2;;
    --condition) CONDITION="$2"; shift 2;;
    --max-iters) MAX_ITERS="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -z "$USER_FP" || -z "$CONDITION" ]] && { echo "usage: $0 --user <fp> --condition <label> [--max-iters N]" >&2; exit 2; }

mkdir -p "$OUTDIR"
TMP=$(mktemp)
trap 'rm -f "$TMP" "$TMP".seen' EXIT

pull() { xcrun devicectl device copy from --device "$DEV" \
  --domain-type appDataContainer --domain-identifier "$BID" \
  --source "$DEVFILE" --destination "$1" >/dev/null 2>&1 || true; }

# Baseline: capture the set of session_ids ALREADY on the device, so we detect
# OUR new session by id (robust vs the old global run_end count, which
# false-positived on a prior run's run_end if the baseline pull hiccuped).
# Retry the baseline pull until the file is non-empty (or 5 tries) so a tunnel
# hiccup can't yield an empty baseline.
SEEN="$TMP.seen"
for _ in 1 2 3 4 5; do pull "$TMP"; [[ -s "$TMP" ]] && break; sleep 3; done
python3 -c "import json,sys
sids=set()
for ln in open(sys.argv[1]):
    ln=ln.strip()
    if not ln: continue
    try: sids.add(json.loads(ln).get('bench_session_id',''))
    except Exception: pass
print('\n'.join(s for s in sids if s))" "$TMP" > "$SEEN" 2>/dev/null || true
echo "[baseline] $(wc -l < "$SEEN" | tr -d ' ') prior session(s) on device"

# Pre-run cool check: warn if the last run ended hot.
LAST_THERMAL=$(grep '"record_type":"train"' "$TMP" 2>/dev/null | tail -1 | sed -n 's/.*"thermal_state":"\([a-z]*\)".*/\1/p' || true)
if [[ -n "${LAST_THERMAL:-}" && "$LAST_THERMAL" != "nominal" ]]; then
  echo "⚠️  last run ended thermal=$LAST_THERMAL — consider cooling to nominal before a cost/condition run."
fi

echo "[run] user=$USER_FP condition=$CONDITION ${MAX_ITERS:+max-iters=$MAX_ITERS}"
ARGS=(--benchmark-train-e2e --user "$USER_FP" --condition "$CONDITION")
[[ -n "$MAX_ITERS" ]] && ARGS+=(--max-iters "$MAX_ITERS")
xcrun devicectl device process launch --device "$DEV" --terminate-existing "$BID" "${ARGS[@]}" \
  2>&1 | grep -iE "Launched|error" || true

echo "[poll] every ${POLL_SECONDS}s (first record may lag if the fused model is re-downloading ~1.6 GB)"
START=$(date +%s); LAST_STATUS=""
STATUS_PY="$(dirname "${BASH_SOURCE[0]}")/_e2e_status.py"
while true; do
  sleep "$POLL_SECONDS"
  pull "$TMP"
  ELAPSED_MIN=$(( ($(date +%s) - START) / 60 ))
  STATUS=$(python3 "$STATUS_PY" "$TMP" "$USER_FP" "$SEEN"); RC=$?
  [[ "$STATUS" != "$LAST_STATUS" ]] && { echo "  ${ELAPSED_MIN}m: $STATUS"; LAST_STATUS="$STATUS"; }
  if (( RC == 0 )); then echo "[done] run finished after ${ELAPSED_MIN}m"; break; fi
  if (( ELAPSED_MIN >= MAX_WAIT_MIN )); then echo "[timeout] hit ${MAX_WAIT_MIN}m ceiling — pulling what exists" >&2; break; fi
done

pull "$OUTFILE"
echo "[pull] full device JSONL -> $OUTFILE"
# One-line summary of the run we just did (latest session).
python3 - "$OUTFILE" "$USER_FP" <<'PY'
import json, sys
recs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
mine=[r for r in recs if r.get("user_fingerprint")==sys.argv[2]]
if not mine: print("  (no records for this user yet)"); sys.exit()
sid=mine[-1]["bench_session_id"]; s=[r for r in recs if r.get("bench_session_id")==sid]
tr=[r for r in s if r["record_type"]=="train"]; end=[r for r in s if r["record_type"] in ("run_end","error")]
prof=next((r.get("profile_size") for r in s), "?")
if tr:
    fl=tr[-1]["training_loss"]; th=tr[-1]["thermal_state"]; peak=max(r["peak_mem_bytes"] for r in tr)//1048576
    print(f"  summary: profile={prof} windows={len(tr)} final_loss={fl:.4f} end_thermal={th} peak={peak}MB", end="")
if end:
    e=end[0]; print(f" completed={e['record_type']=='run_end'} adapter_saved={e.get('adapter_saved')} wall={e.get('elapsed_s',0):.0f}s")
else:
    print(" (no run_end — jetsam/timeout?)")
PY
