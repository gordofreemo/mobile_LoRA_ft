#!/usr/bin/env bash
# h15 campaign C (2026-09-17): unplugged energy, ONE profile per invocation.
#
#   bash scripts/run_h15_c_energy.sh 405   # start at <=85% charge
#   bash scripts/run_h15_c_energy.sh 550   # start at <=85% charge  <-- the Tier-1 fix
#   bash scripts/run_h15_c_energy.sh 987   # start at 100% (needs >1 charge by design)
#
# Why: Table 6's 550 row was "launched plugged at 100%, unplugged ~2 min in" and sits
# in the fuel-gauge plateau, where the gauge under-reads drain. The project already
# VOIDED a 405 energy run for exactly this: from a 100% start it read 14.77 kJ / 2.25 W
# against 17.67 kJ / 3.06 W from a 65% start — 16% low on energy, 26% low on power.
# The 550 row is the sole source of "two thirds of a charge" and feeds the J/token
# range behind "1.3-1.9 M tokens per charge".
#
# The 987 run CANNOT be fixed this way: it needs more than one charge to finish, so a
# <=85% start is impossible. Keep reporting its 50.4 kJ strictly as a lower bound.
#
# 36 blocks, matching campaign A, so Table 5 and Table 6 report one configuration —
# the one Table 1 declares. The energy ceiling (~50.4 kJ) is a property of the battery,
# not the block count, so the 987 run dies at about the same wall clock as at 28 blocks
# but after fewer iterations: that directly measures how much history one charge buys
# at the declared configuration, instead of deriving it by scaling.
#
# OPERATOR STEPS, per invocation:
#   1. Settings > Battery > Charging: set the limit to 85% (or 100% for the 987 run),
#      and let the phone settle there. This is what makes a <=85% start exact and
#      avoids a ~9 h idle discharge from 100%.
#   2. UNPLUG the phone. Leave it unlocked, Auto-Lock Never, on the usual surface,
#      at the usual brightness, Wi-Fi ON (wireless devicectl is the only telemetry
#      path while unplugged).
#   3. Start this script. It verifies the unplugged state from the app itself —
#      devicectl exposes no charging state to the Mac — and retries if it sees
#      charging=true.
#   4. When it reports done, plug the phone back in.
#
# Run detached:
#   nohup caffeinate -ims bash scripts/run_h15_c_energy.sh 550 > /tmp/h15_c_550.log 2>&1 &
set -u
DEV=00008150-000674C60A3B401C
BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
# Resolve the repo root by git, not by directory depth: these scripts are also run
# from snapshot copies under scripts/.h15run/, where dirname/.. lands in scripts/ and
# silently wrote results to mobile_LoRA_ft/scripts/results/ (caught 2026-09-18).
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"
[ -z "$REPO" ] && REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "$REPO/results" ] || { echo "FATAL: $REPO/results not found — refusing to run"; exit 3; }
OUTDIR="$REPO/results/ondevice"; mkdir -p "$OUTDIR"
LAYERS=${LAYERS:-36}

PROFILE="${1:-}"
case "$PROFILE" in
  405) USER_FP=u00008075; TIMEOUT_S=13200 ;;
  550) USER_FP=u00005020; TIMEOUT_S=26400 ;;
  987) USER_FP=u00012502; TIMEOUT_S=28800 ;;
  *) echo "usage: $0 {405|550|987}"; exit 2 ;;
esac
TAG="h15c${PROFILE}"
PULL="$OUTDIR/train_bench_metrics_naxab_e2e_${TAG}_$(date +%F).jsonl"

log() { echo "$(date '+%F %T') $*"; }

resident_pid() {
  xcrun devicectl device info processes --device "$DEV" 2>/dev/null \
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

# Retry: the wireless tunnel drops transient pulls. On 2026-09-20 a single failed pull
# made the probe kill a healthy run that was already at step 30 with charging=False.
# A pull failure is only meaningful after several tries.
pull() {
  local i
  for i in 1 2 3 4; do
    xcrun devicectl device copy from --device "$DEV" --domain-type appDataContainer \
      --domain-identifier "$BID" --source "Documents/train_bench_metrics_naxab_e2e_${TAG}.jsonl" \
      --destination "$PULL" >/dev/null 2>&1 && return 0
    sleep 15
  done
  return 1
}

log "==== h15 campaign C: profile=$PROFILE user=$USER_FP layers=$LAYERS tag=$TAG ===="

# Probe-launch loop: the app is the only thing that can see charging state.
STARTED=0
for attempt in 1 2 3 4 5 6; do
  kill_resident
  if ! xcrun devicectl device process launch --device "$DEV" --terminate-existing "$BID" \
      --benchmark-train-e2e --user "$USER_FP" --condition C2 --nax-arm on \
      --lora-layers "$LAYERS" --run-tag "$TAG" >/dev/null 2>&1; then
    log "launch failed (attempt $attempt) — wireless tunnel down? retry in 10 min"; sleep 600; continue
  fi
  sleep 150
  if ! pull; then
    # Do not kill on a pull failure if the device is still reachable and the app is
    # alive: the run is probably fine and only telemetry failed.
    if xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -qi "LLMEval.app/LLMEval"; then
      log "pull failed but app IS RUNNING — treating as telemetry flake, entering monitor mode"
      STARTED=1; break
    fi
    log "pull failed and app is gone (attempt $attempt) — retry in 5 min"; kill_resident; sleep 300; continue
  fi
  STATE=$(python3 "$REPO/scripts/_night2_run_start.py" "$USER_FP" "$PULL")
  log "probe $attempt: run_start(charging battery session) -> $STATE"
  case "$STATE" in
    False*|false*|0*) log "device UNPLUGGED — run is live, monitoring"; STARTED=1; break ;;
    *) log "device still PLUGGED (or no record) — killing, retry in 10 min"; kill_resident; sleep 600 ;;
  esac
done

if [ "$STARTED" -eq 0 ]; then log "never saw an unplugged run_start — aborting"; exit 1; fi

# Monitor sparsely (10 min) to stay inside the idle baseline's radio-activity envelope.
# A critical-battery force-shutdown (the expected 987 outcome) appears as process-gone
# with no run_end; the JSONL tail is the forensic record.
RUN_START=$(date +%s)
while true; do
  sleep 600
  ELAPSED=$(($(date +%s) - RUN_START))
  PID=$(resident_pid || true)
  if [ -z "${PID:-}" ]; then log "process gone after ${ELAPSED}s"; break; fi
  if [ "$ELAPSED" -gt "$TIMEOUT_S" ]; then log "hard stop after ${ELAPSED}s — killing"; kill_resident; break; fi
  if pull; then log "alive: ${ELAPSED}s, $(wc -l < "$PULL" | tr -d ' ') records"
  else log "alive (pull failed — device asleep or wireless drop, expected late in battery)"; fi
done

sleep 10
if pull; then log "final pull: $(wc -l < "$PULL" | tr -d ' ') records -> $PULL"
else log "final pull FAILED — likely dead battery; pull over USB after recharging"; fi
log "==== h15 C profile=$PROFILE done — PLUG THE PHONE BACK IN ===="
log "Summarize with: python3 eval/h14_energy_summary.py $PULL"
