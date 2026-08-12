#!/bin/bash
# NAX-ON rerun campaign, day-2 afternoon: plugged e2e C0 runs for the cost-law
# family (448 -> 500 -> 550). Same launch discipline as night 1.
# Run: nohup caffeinate -ims bash scripts/nax_rerun_day2_c0.sh >> /tmp/nax_rerun_day2.log 2>&1 &

set -u
DEVICE=00008150-000674C60A3B401C
BUNDLE=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
PULLDIR=/tmp/devpull/nax_rerun
CUTOFF=$(date -j -v23H -v30M -v0S +%s)   # leave the phone free before the 00:45 XXL run
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

run_user() {
    local user=$1 timeout_s=$2
    if [ "$(date +%s)" -ge "$CUTOFF" ]; then
        log "cutoff reached — skipping $user"
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
        --destination "$PULLDIR/day2_e2e.jsonl" >/dev/null 2>&1; then
        log "pulled cumulative e2e jsonl ($(wc -l < "$PULLDIR/day2_e2e.jsonl" | tr -d ' ') records)"
    else
        log "pull FAILED after $user"
    fi
    log "gap 600s"
    sleep 600
}

log "==== day-2 C0 sequence start; cutoff $(date -r "$CUTOFF" '+%H:%M') ===="
run_user u00005228 10800   # 448 examples, 1344 iters
run_user u00011077 12600   # 500 examples, 1500 iters
run_user u00005020 12600   # 550 examples, 1650 iters
log "==== day-2 C0 sequence complete ===="
