#!/usr/bin/env bash
# h15 campaign A resume after an app reinstall (2026-09-19).
# A reinstall can wipe the app container, which holds the five staged LaMP-3 corpora.
# Without them the app launches and trains nothing, exactly as h15a448 did on 09-18.
# This verifies the app launches, restores any missing corpus, then runs 500/550/987.
#   bash scripts/run_h15_a_resume.sh
set -uo pipefail
DEV=00008150-000674C60A3B401C; BID=mlx.LLMEvalJGW9U9Y36Y
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
D="$REPO/data/ondevice_user_data"
log(){ echo "$(date '+%F %T') $*"; }

log "=== 1. can the app launch? ==="
out=$(xcrun devicectl device process launch --device "$DEV" --terminate-existing "$BID" 2>&1)
if grep -qiE "invalid code signature|not been explicitly trusted|RequestDenied" <<<"$out"; then
  log "FATAL: app still will not launch — signing/profile not fixed yet"; echo "$out" | grep -iE "error|Security" | head -3; exit 2
fi
log "app launches OK"
pid=$(xcrun devicectl device info processes --device "$DEV" 2>/dev/null | grep -i "LLMEval.app/LLMEval" | awk '{print $1}' | head -1)
[ -n "${pid:-}" ] && xcrun devicectl device process signal --device "$DEV" --signal SIGKILL --pid "$pid" >/dev/null 2>&1
sleep 3

log "=== 2. are the corpora present? ==="
listing=$(xcrun devicectl device info files --device "$DEV" --domain-type appDataContainer \
  --domain-identifier "$BID" --username mobile 2>/dev/null)
missing=()
for u in u00011077 u00005020 u00012502; do
  grep -q "lamp3_${u}.jsonl" <<<"$listing" && log "  present: $u" || { log "  MISSING: $u"; missing+=("--source" "$D/lamp3_${u}.jsonl"); }
done
if [ ${#missing[@]} -gt 0 ]; then
  log "restoring $(( ${#missing[@]} / 2 )) corpora (multi-source: a lone --source RENAMES and has destroyed Documents before)"
  xcrun devicectl device copy to --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$BID" "${missing[@]}" --destination "Documents/user_data" >/dev/null 2>&1
  listing=$(xcrun devicectl device info files --device "$DEV" --domain-type appDataContainer \
    --domain-identifier "$BID" --username mobile 2>/dev/null)
  for u in u00011077 u00005020 u00012502; do
    grep -q "lamp3_${u}.jsonl" <<<"$listing" && log "  restored: $u" || { log "FATAL: could not restore $u"; exit 3; }
  done
fi

log "=== 3. launching 500 550 987 (36 blocks, no lead idle: phone is cold) ==="
cp "$REPO/scripts/run_h15_a_cost.sh" "$REPO/scripts/.h15run/a_resume.sh"
SKIP_FIRST_IDLE=1 exec bash "$REPO/scripts/.h15run/a_resume.sh" 500 550 987
