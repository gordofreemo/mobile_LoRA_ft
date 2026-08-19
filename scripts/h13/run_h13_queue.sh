#!/bin/bash
# Walk the frozen h13 queue. Any prefix is a complete, reportable result, so
# this is restartable: users whose predictions are already pulled are skipped.
# (macOS ships bash 3.2 -- no `mapfile`, hence the while-read loop.)
set -uo pipefail
ROOT="$HOME/Documents/Research/mobile_LoRA_ft"
START=${START:-0}
END=${END:-100}

QFILE=$(mktemp)
python3 -c "
import json
q=json.load(open('$ROOT/data/oppu_movie/h13_queue.json'))['queue']
print('\n'.join(e['user_id'] for e in q[$START:$END]))" > "$QFILE"
echo "[queue] $(wc -l < "$QFILE" | tr -d ' ') users, ranks $START..$END"

while IFS= read -r U; do
  [ -z "$U" ] && continue
  if [ -s "$ROOT/results/ondevice/h13_preds/$U/device.jsonl" ] \
     && [ -s "$ROOT/results/ondevice/h13_preds/$U/rag.jsonl" ] \
     && [ -s "$ROOT/results/ondevice/h13_preds/$U/mac.jsonl" ]; then
    echo "[queue] $U already complete, skipping"; continue
  fi
  echo "[queue] === $U ($(date +%H:%M:%S)) ==="
  "$ROOT/scripts/h13/run_h13_user.sh" "$U" || echo "[queue] $U FAILED, continuing"
done < "$QFILE"
rm -f "$QFILE"
echo "[queue] done $(date +%H:%M:%S)"
