#!/bin/bash
# NAX-ON rerun campaign: cycling-verdict confirmation run (daytime, plugged).
#   1. Continuous reference: self-limit mode, delay 0 = plain continuous
#      training, 75 min — same-hour plateau, no 90-min observation overhead.
#   2. 15-min gap.
#   3. Cycling arm 10on/2off x6, starting on the HEATED chassis — the
#      condition most biased AGAINST cycling. If sustained cycling still
#      reads >=1.0x the same-hour continuous plateau, the flip is confirmed
#      conservatively.
# Run: nohup caffeinate -ims bash scripts/nax_rerun_cycling_confirm.sh >> /tmp/nax_cycling_confirm.log 2>&1 &

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

# run_phase <label> <timeout_s> <ondevice_jsonl> <launch args...>
run_phase() {
    local label=$1 timeout_s=$2 jsonl=$3
    shift 3
    kill_resident
    if ! xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" \
        "$@" >/dev/null 2>&1; then
        log "launch FAILED ($label)"; return 1
    fi
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
    xcrun devicectl device copy from --device "$DEVICE" \
        --domain-type appDataContainer --domain-identifier "$BUNDLE" \
        --source "Documents/$jsonl" --destination "$PULLDIR/confirm_$label.jsonl" \
        >/dev/null 2>&1 && log "pulled $jsonl"
}

log "==== cycling confirmation start ===="
run_phase continuous 6600 train_bench_metrics_selflimit_nax-on.jsonl \
    --benchmark-thermal-selflimit --selflimit-delay 0 --selflimit-minutes 75 \
    --nax-arm on
log "gap 900s"
sleep 900
run_phase cycle 7200 train_bench_metrics_thermal_nax-on.jsonl \
    --benchmark-thermal-cycle --burst-minutes 10 --rest-seconds 120 --cycles 6 \
    --nax-arm on
log "==== cycling confirmation complete ===="
