#!/bin/bash
# h12 v3 watchdog: keep the full task-adapter run alive to completion.
#
# Every POLL_S seconds: pull the device JSONL and classify the newest
# pt_lamp7_h12 session:
#   RUNNING — records fresh: also re-foreground the app (idle-mechanism guard).
#   DONE    — run_end/error record present: pull artifacts, notify, exit.
#   DEAD    — records stale >STALE_S or process gone: SIGKILL any resident,
#             relaunch with the standard args. v3's on-device checkpoint/resume
#             (every 5 steps, exact optimizer-state continuity) turns each
#             relaunch into a continuation, so even a systematic ~15-min
#             session killer only costs the tail since the last checkpoint.
#
# Run fully detached:
#   nohup bash scripts/h12_watchdog.sh > /tmp/h12_watchdog.log 2>&1 &

set -u
DEVICE=00008150-000674C60A3B401C
BUNDLE=mlx.LLMEvalJGW9U9Y36Y
POLL_S=120
STALE_S=300
MAX_RELAUNCH=80
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

RELAUNCHES=0
PULL=/tmp/devpull/taskadapter_watchdog.jsonl

log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

relaunch() {
    PID=$(xcrun devicectl device info processes --device "$DEVICE" 2>/dev/null \
        | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1 || true)
    if [ -n "$PID" ]; then
        xcrun devicectl device process signal --signal SIGKILL --pid "$PID" --device "$DEVICE" >/dev/null 2>&1 || true
        sleep 3
    fi
    if xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" \
        --benchmark-train-taskadapter --nax-arm on --condition c0_plugged >/dev/null 2>&1; then
        RELAUNCHES=$((RELAUNCHES + 1))
        log "relaunched (#$RELAUNCHES)"
    else
        log "relaunch FAILED (device locked?) — will retry next cycle"
    fi
}

log "watchdog start"
while true; do
    sleep "$POLL_S"
    if ! xcrun devicectl device copy from --device "$DEVICE" \
        --domain-type appDataContainer --domain-identifier "$BUNDLE" \
        --source Documents/train_bench_metrics_taskadapter.jsonl \
        --destination "$PULL" >/dev/null 2>&1; then
        log "pull failed — retrying next cycle"
        continue
    fi
    STATUS=$(python3 - <<'EOF'
import json, datetime
recs = [json.loads(l) for l in open('/tmp/devpull/taskadapter_watchdog.jsonl')]
starts = [r for r in recs if r['record_type'] == 'run_start' and r.get('run_name') == 'pt_lamp7_h12']
sess = starts[-1]['bench_session_id']
mine = [r for r in recs if r.get('bench_session_id') == sess]
steps = [r for r in mine if r['record_type'] == 'opt_step']
ends = [r for r in mine if r['record_type'] in ('run_end', 'error')]
now = datetime.datetime.now(datetime.UTC)
last = datetime.datetime.strptime(mine[-1]['timestamp_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.UTC)
stale = (now - last).total_seconds()
n = steps[-1]['step'] if steps else 0
if ends and ends[-1]['record_type'] == 'run_end':
    print(f"DONE step={n}")
elif ends:
    print(f"ERROR step={n} err={ends[-1].get('error')}")
elif stale > 300:
    print(f"DEAD step={n} stale_s={int(stale)}")
else:
    print(f"RUNNING step={n} stale_s={int(stale)}")
EOF
)
    log "$STATUS"
    case "$STATUS" in
        DONE*)
            log "run complete — pulling artifacts"
            mkdir -p /tmp/devpull/final
            xcrun devicectl device copy from --device "$DEVICE" \
                --domain-type appDataContainer --domain-identifier "$BUNDLE" \
                --source Documents/task_adapters/pt_lamp7_h12 \
                --destination /tmp/devpull/final/pt_lamp7_h12 >/dev/null 2>&1 || true
            cp "$PULL" /tmp/devpull/final/train_bench_metrics_taskadapter_final.jsonl
            log "artifacts in /tmp/devpull/final; exiting"
            exit 0
            ;;
        ERROR*)
            log "on-device error record — relaunching (resume covers it)"
            relaunch
            ;;
        DEAD*)
            relaunch
            ;;
        RUNNING*)
            # Idle-mechanism guard: foreground the resident app.
            xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" >/dev/null 2>&1 || true
            ;;
    esac
    if [ "$RELAUNCHES" -ge "$MAX_RELAUNCH" ]; then
        log "relaunch cap reached — giving up"
        exit 1
    fi
done
