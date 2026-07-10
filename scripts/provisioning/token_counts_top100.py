#!/usr/bin/env python3
"""Compute per-user total training tokens for all top-100 LaMP-3 users, so the
token-based cost law (wall ≈ s/token · total_tokens, r=1.000 on-device) can be
extrapolated to the full 100 and used to predict remaining run times.

For each user: render their profile-entry BM25(k=4) examples exactly as the
on-device data (SmolLM3 chat template, /no_think), tokenize, cap each at 1024
(the on-device seq cap), and sum. total_tokens = 3 epochs × Σ min(len_i, 1024).

Single streaming pass over the 3.4 GB train_questions.json for all 100 snapshots.
Output: data/lamp_user_stats/LaMP_3_top100_token_counts.json
"""
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.build_user_dataset import emit_bm25_records, stream_json_array, TIME_SPLIT_DIR  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TASK, BM25_K, CAP, EPOCHS = "LaMP_3", 4, 1024, 3
TOP100 = "data/lamp_user_stats/LaMP_3_top100_users.json"
TOK_SRC = "data/models/SmolLM3-3B-a1lamp-merged"
OUT = "data/lamp_user_stats/LaMP_3_top100_token_counts.json"

users = json.loads(Path(TOP100).read_text())["users"]
want = {u["user_fingerprint"]: u["profile_size"] for u in users}
print(f"[users] {len(want)} top-100 users", flush=True)

# resolve each user's snapshot (largest-profile) train record in one pass
from train.build_user_dataset import USER_RECORDS_DIR  # noqa: E402
recs_meta = json.loads((USER_RECORDS_DIR / f"{TASK}_user_records.json").read_text())
train_ids_by_user = {u: set(recs_meta[u]["train"]) for u in want}
id_to_user = {str(i): u for u, ids in train_ids_by_user.items() for i in ids}
n_needed = {u: len(train_ids_by_user[u]) for u in want}

best = {u: (None, -1) for u in want}
seen = {u: 0 for u in want}
q = TIME_SPLIT_DIR / TASK / "train_questions.json"
print(f"[scan] streaming {q} ...", flush=True)
for r in stream_json_array(str(q)):
    u = id_to_user.get(str(r.get("id")))
    if u is None:
        continue
    seen[u] += 1
    ps = len(r.get("profile", []))
    if ps > best[u][1]:
        best[u] = (r, ps)
    if all(seen[x] >= n_needed[x] for x in want):
        print("[scan] all snapshots found — early exit", flush=True)
        break

tok = AutoTokenizer.from_pretrained(TOK_SRC)


def render(rec):
    msgs = []
    if rec.get("system"):
        msgs.append({"role": "system", "content": rec["system"]})
    msgs.append({"role": "user", "content": rec["user"]})
    msgs.append({"role": "assistant", "content": rec["assistant"]})
    try:
        return tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False)


out = []
for i, (u, ps) in enumerate(sorted(want.items(), key=lambda x: x[1]), 1):
    snap, _ = best[u]
    buf = io.StringIO()
    emit_bm25_records(snap.get("profile", []), TASK, BM25_K, buf)
    lens = []
    for line in buf.getvalue().splitlines():
        if not line.strip():
            continue
        t = render(json.loads(line))
        lens.append(min(len(tok(t)["input_ids"]), CAP))
    sum_capped = sum(lens)
    mean_len = sum_capped / len(lens) if lens else 0
    total_tokens = EPOCHS * sum_capped
    out.append({
        "user_fingerprint": u, "profile_size": ps, "n_examples": len(lens),
        "mean_capped_seq_len": round(mean_len, 1),
        "sum_capped_tokens": sum_capped, "total_tokens_3ep": total_tokens,
    })
    print(f"  [{i}/100] {u} prof={ps} n={len(lens)} mean_len={mean_len:.0f} "
          f"total_tokens={total_tokens:,}", flush=True)

Path(OUT).write_text(json.dumps({"cap": CAP, "epochs": EPOCHS, "users": out}, indent=2))
grand = sum(o["total_tokens_3ep"] for o in out)
print(f"[done] {len(out)} users, grand total_tokens={grand:,} -> {OUT}", flush=True)
