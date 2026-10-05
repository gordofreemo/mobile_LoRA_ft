#!/usr/bin/env bash
# h15 campaign F (2026-09-22): the pre-submission rerun list, one item per invocation.
#
#   bash scripts/run_h15_f_rerun.sh r1    # R1 stock-kernel 405, unplugged, START AT 85%
#   bash scripts/run_h15_f_rerun.sh r4    # R4 550 energy,       unplugged, START AT 100%
#   bash scripts/run_h15_f_rerun.sh r5    # R5 idle baseline,    unplugged, any charge
#
# Same launch pattern as campaign C: detached devicectl launch, an app-side probe for
# the charging state (devicectl cannot see it from the Mac), sparse 10-min polling so
# the radio traffic stays inside the idle baseline's envelope, and a retry-hardened
# pull that never mistakes a wireless-tunnel drop for a dead run.
#
# R1 exists because Figure 1 still shows the July stock arm at 28 blocks (1.93x, ~3 W)
# while the body now reports 1.58x and 4 W at 36. Both traces must start at 85% so the
# fuel-gauge plateau cannot bias one arm: h15c405 started at 85%, so this one does too.
#
# R5 writes to Documents/train_bench_metrics_e2e.jsonl, NOT a tagged file: the app's
# idle-baseline path calls emitE2E() with no filename, so --run-tag and --nax-arm do
# not reach it. Records are separated by bench_session_id and condition.
#
# NOTE on R5's duration. The gauge quantizes to 5%, about 2.8 kJ. At the h9 idle rate
# (0.26 W) a 60-minute window spends ~0.9 kJ and will read a flat 0% change, so 60 min
# cannot measure idle power at all. The h9 canonical baseline ran 10,693 s to resolve a
# single 5% step; this uses the same 3 h.
set -u
DEV=00008150-000674C60A3B401C
BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "$REPO/results" ] || { echo "FATAL: $REPO/results not found — refusing to run"; exit 3; }
OUTDIR="$REPO/results/ondevice"; mkdir -p "$OUTDIR"
LAYERS=${LAYERS:-36}

ITEM="${1:-}"
case "$ITEM" in
  r1) USER_FP=u00008075; ARM=off; TAG="h15f405_stock_unplugged"; TIMEOUT_S=18000; KIND=train ;;
  r4) USER_FP=u00005020; ARM=on;  TAG="h15f550_full";            TIMEOUT_S=26400; KIND=train ;;
  r5) USER_FP=u00008075; ARM=on;  TAG="h15f_idle";               TIMEOUT_S=12600; KIND=idle
      IDLE_S=${IDLE_S:-10800} ;;
  *) echo "usage: $0 {r1|r4|r5}"; exit 2 ;;
esac

if [ "$KIND" = idle ]; then
  REMOTE="Documents/train_bench_metrics_e2e.jsonl"
else
  REMOTE="Documents/train_bench_metrics_naxab_e2e_${TAG}.jsonl"
fi
PULL="$OUTDIR/train_bench_metrics_naxab_e2e_${TAG}_$(date +%F).jsonl"
[ "$KIND" = idle ] && PULL="$OUTDIR/train_bench_metrics_e2e_h15f_idle_$(date +%F).jsonl"

log() { echo "$(date '+%F %T') $*"; }

# Only kill_resident uses this now, and it is wrapped in a timeout: a hung
# CoreDevice call must never block the caller (2026-09-22).
resident_pid() {
  timeout 90 xcrun devicectl device info processes --device "$DEV" 2>/dev/null \
    | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1
}

kill_resident() {
  local pid; pid=$(resident_pid || true)
  if [ -n "${pid:-}" ]; then
    xcrun devicectl device process signal --signal SIGKILL --pid "$pid" --device "$DEV" >/dev/null 2>&1 || true
    sleep 3
  fi
  return 0
}

pull() {
  local i
  for i in 1 2 3 4; do
    xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
      --domain-identifier "$BID" --source "$REMOTE" \
      --destination "$PULL" >/dev/null 2>&1 && return 0
    sleep 15
  done
  return 1
}

launch() {
  if [ "$KIND" = idle ]; then
    xcrun devicectl device process launch --device "$DEV" --terminate-existing "$BID" \
      --benchmark-idle-baseline --user "$USER_FP" --condition C2 \
      --baseline-duration-seconds "$IDLE_S" >/dev/null 2>&1
  else
    xcrun devicectl device process launch --device "$DEV" --terminate-existing "$BID" \
      --benchmark-train-e2e --user "$USER_FP" --condition C2 --nax-arm "$ARM" \
      --lora-layers "$LAYERS" --run-tag "$TAG" >/dev/null 2>&1
  fi
}

log "==== h15 campaign F: item=$ITEM user=$USER_FP arm=$ARM layers=$LAYERS tag=$TAG ===="
log "remote=$REMOTE  local=$PULL"

# Probe-launch loop: only the app can see the charging state. Retries for an hour so
# the run can be armed BEFORE the phone is unplugged and start itself when it is.
STARTED=0
for attempt in $(seq 1 18); do
  kill_resident
  if ! launch; then
    log "launch failed (attempt $attempt) — wireless tunnel down? retry in 5 min"; sleep 300; continue
  fi
  sleep 150
  if ! pull; then
    if xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -qi "LLMEval.app/LLMEval"; then
      log "pull failed but app IS RUNNING — treating as telemetry flake, entering monitor mode"
      STARTED=1; break
    fi
    log "pull failed and app is gone (attempt $attempt) — retry in 5 min"; kill_resident; sleep 300; continue
  fi
  if [ "$KIND" = idle ]; then
    STATE=$(python3 - "$PULL" <<'PY'
import json,sys
rs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
b=[r for r in rs if r.get("record_type")=="idle_baseline_start"]
print(f"{b[-1].get('charging')} {b[-1].get('battery_level')}" if b else "None None")
PY
)
  else
    STATE=$(python3 "$REPO/scripts/_night2_run_start.py" "$USER_FP" "$PULL")
  fi
  log "probe $attempt: (charging battery ...) -> $STATE"
  case "$STATE" in
    False*|false*|0*) log "device UNPLUGGED — run is live, monitoring"; STARTED=1; break ;;
    *) log "device still PLUGGED (or no record) — killing, retry in 5 min"; kill_resident; sleep 300 ;;
  esac
done

if [ "$STARTED" -eq 0 ]; then log "never saw an unplugged run_start — aborting"; exit 1; fi

# Liveness is judged from the pulled file GROWING, never from a process listing.
# `devicectl device info processes` hung for 15 minutes on 2026-09-22 and blocked
# this loop; killing it made resident_pid return empty, which the old code read as
# "process gone" and used to end monitoring on a live run. The process listing is
# also indistinguishable between a dead app and a dropped wireless tunnel, and the
# tunnel drops routinely. A run is declared over only when run_end appears in the
# file, or when the step count stops advancing across several polls.
RUN_START=$(date +%s)
LAST_STEP=-1
STALL=0
FAILS=0
while true; do
  sleep 600
  ELAPSED=$(($(date +%s) - RUN_START))
  if [ "$ELAPSED" -gt "$TIMEOUT_S" ]; then log "hard stop after ${ELAPSED}s — killing"; kill_resident; break; fi
  if pull; then
    FAILS=0
    STATE=$(python3 - "$PULL" <<'PY'
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
tr = [r for r in rs if r.get("record_type") == "train"]
en = [r for r in rs if r.get("record_type") in ("run_end", "idle_baseline_end")]
b  = [r for r in rs if r.get("record_type") in ("battery", "idle_baseline")]
step = tr[-1].get("step", 0) if tr else 0
lvl  = (b[-1].get("battery_level") or 0) * 100 if b else -1
print(f'{step} {lvl:.0f} {"ENDED" if en else "live"}')
PY
)
    set -- $STATE; STEP=$1; LVL=$2; FIN=$3
    log "alive: ${ELAPSED}s, step ${STEP}, battery ${LVL}%, $(wc -l < "$PULL" | tr -d ' ') records"
    if [ "$FIN" = "ENDED" ]; then log "run_end present — run complete at step ${STEP}"; break; fi
    if [ "$STEP" = "$LAST_STEP" ]; then
      STALL=$((STALL + 1))
      log "no progress for ${STALL} poll(s) (step ${STEP})"
      if [ "$STALL" -ge 3 ]; then log "stalled ~30 min at step ${STEP} — treating as ended"; break; fi
    else
      STALL=0
    fi
    LAST_STEP=$STEP
  else
    FAILS=$((FAILS + 1))
    log "pull failed (${FAILS}) — wireless drop or device down; NOT concluding the run died"
    if [ "$FAILS" -ge 6 ]; then log "unreachable for ~1 h — giving up on telemetry; pull by hand once the tunnel returns"; break; fi
  fi
done

sleep 10
if pull; then log "final pull: $(wc -l < "$PULL" | tr -d ' ') records -> $PULL"
else log "final pull FAILED — likely dead battery; pull over USB after recharging"; fi
log "==== h15 F item=$ITEM done — PLUG THE PHONE BACK IN ===="
