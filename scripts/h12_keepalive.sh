#!/bin/bash
# h12 keep-alive: re-foreground LLMEval every KEEPALIVE_S seconds.
#
# Diagnostic + workaround for the 2026-08-10/11 failure pattern: unattended
# sessions die 13-19 min in (always during step 11's wall-clock window), while
# the one session with user interaction in that window survived. If an iOS
# inactivity mechanism is the killer, periodic foregrounding may reset it.
# Launching over a RESIDENT app only foregrounds it (args are absorbed —
# normally a trap, here the tool). If the app has died, the no-arg launch
# starts a benign idle instance; the poller notices the JSONL is stale and
# this script stops so the operator (or the agent) can decide.
#
# Run fully detached:  nohup bash scripts/h12_keepalive.sh > /tmp/h12_keepalive.log 2>&1 &

set -u
DEVICE=00008150-000674C60A3B401C
BUNDLE=mlx.LLMEvalJGW9U9Y36Y
KEEPALIVE_S=360
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

while true; do
    sleep "$KEEPALIVE_S"
    TS=$(date "+%H:%M:%S")
    # Foreground the (resident) app. Failure here usually means locked device.
    if xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE" >/dev/null 2>&1; then
        echo "$TS keepalive: foregrounded"
    else
        echo "$TS keepalive: LAUNCH FAILED (locked?)"
    fi
    # Health probe: pull the JSONL and check freshness.
    if xcrun devicectl device copy from --device "$DEVICE" \
        --domain-type appDataContainer --domain-identifier "$BUNDLE" \
        --source Documents/train_bench_metrics_taskadapter.jsonl \
        --destination /tmp/devpull/taskadapter_keepalive.jsonl >/dev/null 2>&1; then
        STATUS=$(python3 - <<'EOF'
import json, datetime
recs = [json.loads(l) for l in open('/tmp/devpull/taskadapter_keepalive.jsonl')]
v2 = [r for r in recs if r.get('loss_impl') == 'sliced_lm_head']
steps = [r for r in v2 if r['record_type'] == 'opt_step']
ends = [r for r in v2 if r['record_type'] in ('run_end', 'error')]
now = datetime.datetime.now(datetime.UTC)
last = datetime.datetime.strptime(v2[-1]['timestamp_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.UTC)
stale = (now - last).total_seconds()
n = steps[-1]['step'] if steps else 0
if ends:
    print(f"DONE {ends[-1]['record_type']} step={n}")
elif stale > 300:
    print(f"STALE step={n} stale_s={int(stale)}")
else:
    print(f"OK step={n} stale_s={int(stale)}")
EOF
)
        echo "$TS health: $STATUS"
        case "$STATUS" in
            DONE*|STALE*) echo "$TS stopping keep-alive"; exit 0 ;;
        esac
    else
        echo "$TS health: pull failed"
    fi
done
