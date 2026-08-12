#!/bin/bash
# NAX-ON rerun campaign, night 2: h9 energy point, UNPLUGGED (condition C2).
# Default subject: L = u00005020 (550 examples, 1650 iters). Night 3 runs
# XXL = u00012502 (the one-charge marquee) the same way:
#   bash scripts/nax_rerun_night2.sh u00012502
#
# Operator preconditions (before bed):
#   - phone charged to ~100%, then UNPLUGGED and left on the usual surface
#   - airplane mode OFF / Wi-Fi ON (wireless devicectl is the only telemetry
#     path while unplugged; h9 protocol)
#   - Auto-Lock still Never
# The script CANNOT read charging state from the Mac (devicectl exposes none),
# so it verifies via the app itself: launch, read run_start.charging from the
# pulled JSONL, and kill + retry if still plugged (partial sessions are
# tagged and harmless).
#
# Energy accounting: idle baseline reused from h9 (idle power is
# kernel-independent); pulls kept sparse (10 min) to stay within the paired
# baseline's radio-activity envelope.
#
# Run: nohup caffeinate -ims bash scripts/nax_rerun_night2.sh [user] \
#        >> /tmp/nax_rerun_night2.log 2>&1 &

set -u

DEVICE=00008150-000674C60A3B401C
BUNDLE=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

SUBJECT="${1:-u00005020}"
JSONL=train_bench_metrics_naxab_e2e.jsonl
PULLDIR=/tmp/devpull/nax_rerun
START_HOUR=0
START_MIN=45
MAX_RUN_S=25200   # 7 h hard stop
mkdir -p "$PULLDIR"

log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

resident_pid() {
    xcrun devicectl device info processes --device "$DEVICE" 2>/dev/null \
        | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1
}

kill_resident() {
    local pid
    pid=$(resident_pid || true)
    if [ -n "${pid:-}" ]; then
        xcrun devicectl device process signal --signal SIGKILL --pid "$pid" \
            --device "$DEVICE" >/dev/null 2>&1 || true
        sleep 3
    fi
}

pull() {
    xcrun devicectl device copy from --device "$DEVICE" \
        --domain-type appDataContainer --domain-identifier "$BUNDLE" \
        --source "Documents/$JSONL" --destination "$PULLDIR/night2_e2e.jsonl" \
        >/dev/null 2>&1
}

# newest run_start for this subject: print "charging battery_level session"
# (separate .py helper — no heredocs in master scripts, see the h12 lesson)
latest_run_start() {
    python3 "$(dirname "$0")/_night2_run_start.py" "$SUBJECT" "$PULLDIR/night2_e2e.jsonl"
}

# Wait until the start time (tonight, or tomorrow if already past it).
NOW=$(date +%s)
TARGET=$(date -j -v${START_HOUR}H -v${START_MIN}M -v0S +%s)
if [ "$TARGET" -le "$NOW" ]; then TARGET=$(date -j -v+1d -v${START_HOUR}H -v${START_MIN}M -v0S +%s); fi
log "night 2 subject=$SUBJECT — sleeping until $(date -r "$TARGET" '+%m-%d %H:%M')"
sleep $((TARGET - NOW))

# Probe-launch loop: verify the device is actually unplugged.
STARTED=0
for attempt in 1 2 3 4 5 6 7 8; do
    kill_resident
    if ! xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" \
        --benchmark-train-e2e --user "$SUBJECT" --condition C2 --nax-arm on \
        >/dev/null 2>&1; then
        log "launch failed (attempt $attempt) — wireless tunnel down? retrying in 15 min"
        sleep 900
        continue
    fi
    sleep 150
    if ! pull; then
        log "pull failed after launch (attempt $attempt) — retrying in 15 min"
        kill_resident
        sleep 900
        continue
    fi
    STATE=$(latest_run_start)
    log "probe attempt $attempt: run_start -> $STATE"
    case "$STATE" in
        False*|false*|0*)
            log "device UNPLUGGED — run is live, entering monitor mode"
            STARTED=1
            break
            ;;
        *)
            log "device still PLUGGED (or no record) — killing, retry in 15 min"
            kill_resident
            sleep 900
            ;;
    esac
done

if [ "$STARTED" -eq 0 ]; then
    log "never saw an unplugged run_start — giving up for tonight"
    exit 1
fi

# Monitor sparsely (10 min) until process exit or hard stop. An OS
# critical-battery force-shutdown (the off-round XXL outcome) shows up as
# process-gone with no run_end — the JSONL tail is the forensic record.
RUN_START=$(date +%s)
while true; do
    sleep 600
    ELAPSED=$(($(date +%s) - RUN_START))
    PID=$(resident_pid || true)
    if [ -z "${PID:-}" ]; then
        log "process gone after ${ELAPSED}s"
        break
    fi
    if [ "$ELAPSED" -gt "$MAX_RUN_S" ]; then
        log "hard stop after ${ELAPSED}s — killing"
        kill_resident
        break
    fi
    if pull; then
        N=$(wc -l < "$PULLDIR/night2_e2e.jsonl" | tr -d ' ')
        log "alive: ${ELAPSED}s elapsed, $N records"
    else
        log "alive (pull failed — device asleep/wireless drop is expected late in battery)"
    fi
done

sleep 10
if pull; then
    log "final pull: $(wc -l < "$PULLDIR/night2_e2e.jsonl" | tr -d ' ') records"
else
    log "final pull FAILED — device likely dead-battery or tunnel dropped; pull via USB in the morning"
fi
log "night 2 done"
