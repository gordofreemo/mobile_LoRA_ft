#!/usr/bin/env python3
"""Print 'charging battery_level session' for the newest run_start of a user.

Helper for scripts/nax_rerun_night2.sh (kept as a separate file so the shell
script needs no heredoc — a heredoc escaping bug killed a previous night's
master script).
"""
import json
import sys

user, path = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path) if l.strip()]
starts = [r for r in rows if r.get("record_type") == "run_start"
          and r.get("user_fingerprint") == user]
if not starts:
    print("none")
else:
    r = starts[-1]
    print(r.get("charging"), r.get("battery_level"), r.get("bench_session_id"))
