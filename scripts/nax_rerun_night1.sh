#!/bin/bash
# NAX-ON rerun campaign, night 1 (2026-08-11 → 08-12). Plugged, autonomous.
# Plan: experiments/2026-08-11-nax-on-rerun-campaign-plan.md
#
# Sequence (verbatim NAX-off protocols, only --nax-arm on added):
#   1. 60-min idle, then h11 per-op rerun      (--benchmark-train-perop)
#   2. pinned-arm discrepancy check            (--benchmark-nax-ab --pin-arms)
#   3. h7 token-time HOT                       (--benchmark-train-tokentime)
#   4. h7 token-time COLD (self-gating)        (--benchmark-train-tokentime-cold)
#   5. 30-min idle, then h10 thermal Run A     (--benchmark-thermal-cooldown 60/120)
#   6. h8 granularity sweep, K in 1..36        (--benchmark-train-granularity)
#
# Operational rules encoded (all learned the hard way, see plan §Ops):
#   - launch DETACHED, never --console (SIGTERM propagates on console death)
#   - SIGKILL any resident LLMEval before EVERY launch (residents absorb args)
#   - keep-alive foreground ping every ~6 min ONLY while the process is alive
#     (the 08-10/11 stochastic killer was interaction-sensitive; a ping on a
#     DEAD process would start an idle instance and break exit detection)
#   - per-phase hard timeout; on timeout kill and move on
#   - hard cutoff 08:45 (phone unplugs at 09:00) — no phase starts after it
#   - run under: caffeinate -ims nohup bash scripts/nax_rerun_night1.sh ...
#
# No heredocs anywhere (a heredoc escaping bug silently killed a previous
# night's master script). bash -n this file before launching.

set -u

DEVICE=00008150-000674C60A3B401C
BUNDLE=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

REPO=/Users/andrewgeyko/Documents/Research/mobile_LoRA_ft
PULLDIR=/tmp/devpull/nax_rerun
LOG=/tmp/nax_rerun_night1.log
CUTOFF=$(date -j -v+1d -v8H -v45M -v0S +%s)
mkdir -p "$PULLDIR" "$REPO/results/ondevice"

log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

resident_pid() {
    xcrun devicectl device info processes --device "$DEVICE" 2>/dev/null \
        | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1
}

kill_resident() {
    local pid
    pid=$(resident_pid || true)
    if [ -n "${pid:-}" ]; then
        log "killing resident LLMEval pid $pid"
        xcrun devicectl device process signal --signal SIGKILL --pid "$pid" \
            --device "$DEVICE" >/dev/null 2>&1 || true
        sleep 3
    fi
}

pull_file() {
    # $1 = on-device filename, $2 = local destination
    xcrun devicectl device copy from --device "$DEVICE" \
        --domain-type appDataContainer --domain-identifier "$BUNDLE" \
        --source "Documents/$1" --destination "$2" >/dev/null 2>&1
}

# launch_and_wait <timeout_s> <ondevice_jsonl> <pull_tag> <launch args...>
# Launch detached, poll every 60s until the process exits or timeout.
# Keep-alive foreground ping every 6th poll, only while the process is alive.
# After exit: pull the JSONL to $PULLDIR/<pull_tag>.jsonl and log its tail.
launch_and_wait() {
    local timeout_s=$1 jsonl=$2 tag=$3
    shift 3
    kill_resident
    local attempt=0 launched=0
    while [ $attempt -lt 3 ]; do
        if xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" \
            "$@" >/dev/null 2>&1; then
            launched=1
            break
        fi
        attempt=$((attempt + 1))
        log "launch FAILED (attempt $attempt) for: $* — retrying in 60s"
        sleep 60
    done
    if [ $launched -eq 0 ]; then
        log "phase SKIPPED (launch failed 3x): $*"
        return 1
    fi
    log "launched: $*"
    local start elapsed poll=0
    start=$(date +%s)
    sleep 30
    while true; do
        local pid
        pid=$(resident_pid || true)
        elapsed=$(($(date +%s) - start))
        if [ -z "${pid:-}" ]; then
            log "process exited after ${elapsed}s: $*"
            break
        fi
        if [ $elapsed -gt "$timeout_s" ]; then
            log "TIMEOUT after ${elapsed}s — killing: $*"
            kill_resident
            break
        fi
        poll=$((poll + 1))
        if [ $((poll % 6)) -eq 0 ]; then
            # keep-alive: foreground the RESIDENT app (args absorbed = no-op
            # relaunch, which is exactly what we want here)
            xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" \
                >/dev/null 2>&1 || true
            log "keepalive ping (elapsed ${elapsed}s)"
        fi
        sleep 60
    done
    sleep 5
    if pull_file "$jsonl" "$PULLDIR/$tag.jsonl"; then
        local n
        n=$(wc -l < "$PULLDIR/$tag.jsonl" | tr -d ' ')
        log "pulled $jsonl -> $tag.jsonl ($n records)"
    else
        log "pull FAILED for $jsonl"
    fi
    return 0
}

check_cutoff() {
    if [ "$(date +%s)" -ge "$CUTOFF" ]; then
        log "cutoff reached — skipping remaining phases"
        return 1
    fi
    return 0
}

log "==== night 1 start; cutoff $(date -r "$CUTOFF" '+%m-%d %H:%M') ===="

# ---- Phase 1: 60-min idle, then h11 per-op rerun -------------------------
log "phase 1: idling 3600s before h11 (protocol pre-idle)"
sleep 3600
check_cutoff && launch_and_wait 5400 \
    train_bench_metrics_perop_nax-on.jsonl perop_naxon \
    --benchmark-train-perop --nax-arm on --idle-minutes 60

# ---- Phase 2: pinned-arm discrepancy check -------------------------------
log "phase 2 in 900s (gap)"
sleep 900
check_cutoff && launch_and_wait 10800 \
    train_bench_metrics_naxab_pinned.jsonl naxab_pinned \
    --benchmark-nax-ab --pin-arms --idle-minutes 15

# ---- Phase 3: h7 token-time HOT ------------------------------------------
log "phase 3 in 600s (gap)"
sleep 600
check_cutoff && launch_and_wait 10800 \
    train_bench_metrics_tokentime_nax-on.jsonl tokentime_hot_naxon \
    --benchmark-train-tokentime --nax-arm on

# ---- Phase 4: h7 token-time COLD (self-gating) ---------------------------
check_cutoff && launch_and_wait 16200 \
    train_bench_metrics_tokentime_cold_nax-on.jsonl tokentime_cold_naxon \
    --benchmark-train-tokentime-cold --nax-arm on

# ---- Phase 5: 30-min idle, then h10 thermal Run A ------------------------
log "phase 5: idling 1800s before h10 Run A"
sleep 1800
check_cutoff && launch_and_wait 12000 \
    train_bench_metrics_thermal_nax-on.jsonl thermal_runA_naxon \
    --benchmark-thermal-cooldown --soak-minutes 60 --probe-interval-s 120 \
    --nax-arm on

# ---- Phase 6: h8 granularity sweep ---------------------------------------
# Real cells first (K=1..6), jetsam-expected confirmations last (K>=9 die on
# the first chunk, ~2-3 min each). One launch per K, exactly like the
# original sweep.
for K in 1 2 3 4 6 9 12 18 36; do
    check_cutoff || break
    launch_and_wait 2700 \
        train_bench_metrics_granularity_nax-on.jsonl "granularity_naxon_k$K" \
        --benchmark-train-granularity --granularity-k "$K" --nax-arm on
done

log "==== night 1 sequence complete ===="
