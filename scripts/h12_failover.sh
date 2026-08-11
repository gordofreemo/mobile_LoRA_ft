#!/bin/bash
# h12 failover: babysit the live v2 (no-checkpoint) full run; if it dies,
# install the already-built v3 app (checkpoint/resume + stderr capture) and
# hand the night over to h12_watchdog.sh. If it completes, pull artifacts.
#
# Run fully detached:
#   nohup bash scripts/h12_failover.sh > /tmp/h12_failover.log 2>&1 &

set -u
DEVICE=00008150-000674C60A3B401C
BUNDLE=mlx.LLMEvalJGW9U9Y36Y
APP=/Users/andrewgeyko/Documents/Research/mobile_LoRA_ft/ios/mlx-swift-examples/build/Build/Products/Debug-iphoneos/LLMEval.app
WATCHDOG=/Users/andrewgeyko/Documents/Research/mobile_LoRA_ft/scripts/h12_watchdog.sh
POLL_S=120
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

PULL=/tmp/devpull/taskadapter_failover.jsonl
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

log "failover monitor start (riding v2 session, v3 armed)"
while true; do
    sleep "$POLL_S"
    if ! xcrun devicectl device copy from --device "$DEVICE" \
        --domain-type appDataContainer --domain-identifier "$BUNDLE" \
        --source Documents/train_bench_metrics_taskadapter.jsonl \
        --destination "$PULL" >/dev/null 2>&1; then
        log "pull failed — retry next cycle"
        continue
    fi
    STATUS=$(python3 - <<'EOF'
import json, datetime
recs = [json.loads(l) for l in open('/tmp/devpull/taskadapter_failover.jsonl')]
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
    print(f"ERROR step={n}")
elif stale > 300:
    print(f"DEAD step={n} stale_s={int(stale)}")
else:
    print(f"RUNNING step={n} stale_s={int(stale)}")
EOF
)
    log "$STATUS"
    case "$STATUS" in
        DONE*)
            log "v2 run COMPLETED — pulling artifacts"
            mkdir -p /tmp/devpull/final
            xcrun devicectl device copy from --device "$DEVICE" \
                --domain-type appDataContainer --domain-identifier "$BUNDLE" \
                --source Documents/task_adapters/pt_lamp7_h12 \
                --destination /tmp/devpull/final/pt_lamp7_h12 >/dev/null 2>&1 || true
            cp "$PULL" /tmp/devpull/final/train_bench_metrics_taskadapter_final.jsonl
            log "artifacts staged in /tmp/devpull/final"
            exit 0
            ;;
        DEAD*|ERROR*)
            log "v2 run died — failing over to v3"
            PID=$(xcrun devicectl device info processes --device "$DEVICE" 2>/dev/null \
                | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1 || true)
            [ -n "$PID" ] && xcrun devicectl device process signal --signal SIGKILL \
                --pid "$PID" --device "$DEVICE" >/dev/null 2>&1
            sleep 3
            if xcrun devicectl device install app --device "$DEVICE" "$APP" >/dev/null 2>&1; then
                log "v3 installed"
            else
                log "v3 install FAILED — retrying once"
                sleep 10
                xcrun devicectl device install app --device "$DEVICE" "$APP" >/dev/null 2>&1 || log "install failed again"
            fi
            xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" \
                --benchmark-train-taskadapter --nax-arm on --condition c0_plugged >/dev/null 2>&1 \
                && log "v3 run launched (fresh, will self-checkpoint)"
            log "handing over to watchdog"
            exec bash "$WATCHDOG"
            ;;
    esac
done
