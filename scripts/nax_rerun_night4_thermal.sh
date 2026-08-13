#!/bin/bash
# NAX-ON rerun campaign, night 4: remaining h10 thermal arms, plugged.
#   1. 30-min pre-idle, then Run B  (60-min soak, 240s probes — self-heating control)
#   2. 20-min gap, then Run C       (10-min soak, 120s probes — short-burst arm)
#   3. 20-min gap, then cycling arm (10-min bursts / 120s rests x 6)
# All --nax-arm on; JSONL routes to train_bench_metrics_thermal_nax-on.jsonl.
# Run: nohup caffeinate -ims bash scripts/nax_rerun_night4_thermal.sh >> /tmp/nax_rerun_night4.log 2>&1 &

set -u
DEVICE=00008150-000674C60A3B401C
BUNDLE=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
PULLDIR=/tmp/devpull/nax_rerun
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

# run_phase <label> <timeout_s> <launch args...>
run_phase() {
    local label=$1 timeout_s=$2
    shift 2
    kill_resident
    local attempt=0 launched=0
    while [ $attempt -lt 3 ]; do
        if xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" \
            "$@" >/dev/null 2>&1; then
            launched=1; break
        fi
        attempt=$((attempt + 1))
        log "launch FAILED ($label attempt $attempt) — retry in 60s"
        sleep 60
    done
    if [ $launched -eq 0 ]; then log "phase SKIPPED ($label)"; return; fi
    log "launched $label: $*"
    local start elapsed poll=0
    start=$(date +%s)
    sleep 60
    while true; do
        local pid
        pid=$(resident_pid || true)
        elapsed=$(($(date +%s) - start))
        if [ -z "${pid:-}" ]; then log "$label exited after ${elapsed}s"; break; fi
        if [ "$elapsed" -gt "$timeout_s" ]; then
            log "$label TIMEOUT after ${elapsed}s — killing"; kill_resident; break
        fi
        poll=$((poll + 1))
        if [ $((poll % 6)) -eq 0 ]; then
            xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" \
                >/dev/null 2>&1 || true
        fi
        sleep 60
    done
    sleep 5
    if xcrun devicectl device copy from --device "$DEVICE" \
        --domain-type appDataContainer --domain-identifier "$BUNDLE" \
        --source Documents/train_bench_metrics_thermal_nax-on.jsonl \
        --destination "$PULLDIR/thermal_naxon_night4.jsonl" >/dev/null 2>&1; then
        log "pulled thermal_nax-on ($(wc -l < "$PULLDIR/thermal_naxon_night4.jsonl" | tr -d ' ') records)"
    else
        log "pull FAILED after $label"
    fi
}

log "==== night-4 thermal arms start ===="
log "pre-idle 1800s before Run B"
sleep 1800
run_phase runB 12000 --benchmark-thermal-cooldown --soak-minutes 60 \
    --probe-interval-s 240 --nax-arm on
log "gap 1200s before Run C"
sleep 1200
run_phase runC 9000 --benchmark-thermal-cooldown --soak-minutes 10 \
    --probe-interval-s 120 --nax-arm on
log "gap 1200s before cycling arm"
sleep 1200
run_phase cycle 7200 --benchmark-thermal-cycle --burst-minutes 10 \
    --rest-seconds 120 --cycles 6 --nax-arm on
log "==== night-4 thermal arms complete ===="
