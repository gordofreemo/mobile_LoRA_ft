#!/usr/bin/env python3
"""Poll-status for ONE E2E run driven by run_e2e.sh.

Identifies OUR run = the newest session that has a run_start for --user whose
bench_session_id is NOT in the pre-launch baseline (SEEN) file. Reports that
session's latest progress and whether it has terminated. This replaces the
earlier global run_end COUNT, which false-positived on a prior run's run_end
when the baseline pull hiccuped.

Args:  <jsonl_path> <user_fp> <seen_sids_path>
Stdout: one progress line (step/loss/iter-s/thermal, or a status phrase).
Exit:  0 = our run ended (run_end/error)   [driver breaks]
       1 = our run running                  [driver keeps polling]
       2 = our run not started yet          [driver keeps polling]
"""
import json
import sys

path, user, seen_path = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    seen = {ln.strip() for ln in open(seen_path) if ln.strip()}
except FileNotFoundError:
    seen = set()

recs = []
try:
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            pass  # tolerate a half-written trailing line mid-pull
except FileNotFoundError:
    pass

sess = {}
for r in recs:
    sess.setdefault(r.get("bench_session_id"), []).append(r)

# our sessions: NEW (unseen) session with a run_start for our user, in file order
ours = [
    (sid, rs) for sid, rs in sess.items()
    if sid not in seen
    and any(r["record_type"] == "run_start" and r.get("user_fingerprint") == user for r in rs)
]
if not ours:
    print("started, waiting for first record (model loading)")
    sys.exit(2)

sid, rs = ours[-1]  # newest matching session
tr = [r for r in rs if r["record_type"] == "train"]
end = [r for r in rs if r["record_type"] in ("run_end", "error")]

if tr:
    t = tr[-1]
    print(f"step={t['step']} loss={t['training_loss']:.4f} "
          f"iter/s={t['iter_per_sec']:.3f} thermal={t['thermal_state']}")
else:
    print("started, first window not reached yet (<10 steps)")

if end:
    e = end[0]
    if e["record_type"] == "error":
        print("[ERROR / jetsam]")
    sys.exit(0)
sys.exit(1)
