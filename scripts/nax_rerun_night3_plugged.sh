#!/bin/bash
# NAX-ON rerun campaign, night 3 (plugged, user call: no charge-wait tonight).
# Remaining e2e C0 cost-law points, longest first — the uninterrupted overnight
# window is the only slot that fits the 987-user run (~7-9h at the measured
# long-example pace). 500/550 follow only if they can still START early enough
# to finish by ~09:00.
# Run: nohup caffeinate -ims bash scripts/nax_rerun_night3_plugged.sh >> /tmp/nax_rerun_night3b.log 2>&1 &

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

# run_user <user> <timeout_s> <latest_start_epoch>
run_user() {
    local user=$1 timeout_s=$2 latest_start=$3
    if [ "$(date +%s)" -ge "$latest_start" ]; then
        log "past latest-start for $user — skipping (runs tomorrow)"
        return
    fi
    kill_resident
    if ! xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" \
        --benchmark-train-e2e --user "$user" --condition C0 --nax-arm on \
        >/dev/null 2>&1; then
        log "launch FAILED for $user — skipping"
        return
    fi
    log "launched C0 nax-on: $user"
    local start elapsed poll=0
    start=$(date +%s)
    sleep 60
    while true; do
        local pid
        pid=$(resident_pid || true)
        elapsed=$(($(date +%s) - start))
        if [ -z "${pid:-}" ]; then
            log "process exited after ${elapsed}s: $user"
            break
        fi
        if [ "$elapsed" -gt "$timeout_s" ]; then
            log "TIMEOUT after ${elapsed}s — killing $user"
            kill_resident
            break
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
        --source Documents/train_bench_metrics_naxab_e2e.jsonl \
        --destination "$PULLDIR/night3_e2e.jsonl" >/dev/null 2>&1; then
        log "pulled cumulative e2e jsonl ($(wc -l < "$PULLDIR/night3_e2e.jsonl" | tr -d ' ') records)"
    else
        log "pull FAILED after $user"
    fi
    log "gap 600s"
    sleep 600
}

log "==== night-3 plugged C0 sequence start ===="
# 987: 2961 iters; at 0.09-0.13 iter/s -> 6.3-9.1h. Start immediately.
run_user u00012502 34200 "$(($(date +%s) + 60))"
# 500: 1500 iters (~3.2-4.2h). Latest start 05:00.
run_user u00011077 18000 "$(date -j -v+1d -v5H -v0M -v0S +%s)"
# 550: 1650 iters (~3.5-4.6h). Latest start 05:00 (only reached if 500 skipped/fast).
run_user u00005020 18000 "$(date -j -v+1d -v5H -v0M -v0S +%s)"
log "==== night-3 plugged sequence complete ===="
