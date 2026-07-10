#!/usr/bin/env python3
"""Build + render on-device per-user LaMP_3 training JSONL for the 6 E2E sample users.

Single streaming pass over the 3.4 GB train_questions.json to collect each user's
snapshot (largest-profile) train record, then per user:
  1. emit R5's exact profile-entry BM25(k=4, strictly-prior) flat records
     (reuses train.build_user_dataset.emit_bm25_records — byte-faithful to R5),
  2. render each {system,user,assistant} through SmolLM3's chat template
     (enable_thinking=False, tokenize=False) into {"text": ...} for the MLX
     on-device loader (MLXLLM.loadLoRAData expects {"text": ...}).

Output: data/ondevice_user_data/lamp3_<user>.jsonl  (device side-load files)
"""
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train.build_user_dataset import (  # noqa: E402
    emit_bm25_records,
    stream_json_array,
    TIME_SPLIT_DIR,
    USER_RECORDS_DIR,
)
from transformers import AutoTokenizer  # noqa: E402

TASK = "LaMP_3"
BM25_K = 4
USERS = ["u00008075", "u00005228", "u00011077", "u00005020", "u00013218", "u00012502"]
OUT_DIR = Path("data/ondevice_user_data")
TOK_SRC = "data/models/SmolLM3-3B-a1lamp-merged"  # has SmolLM3 chat_template

records_json = USER_RECORDS_DIR / f"{TASK}_user_records.json"
users_meta = json.loads(records_json.read_text())
# user -> set(train_ids)
train_ids_by_user = {u: set(users_meta[u]["train"]) for u in USERS}
# id -> user (train records are disjoint across users within a task)
id_to_user = {}
for u, ids in train_ids_by_user.items():
    for i in ids:
        id_to_user[str(i)] = u

n_needed = {u: len(train_ids_by_user[u]) for u in USERS}
print("[users] needed train records:", n_needed, flush=True)

# --- single pass: per-user largest-profile snapshot ---
best = {u: (None, -1) for u in USERS}  # user -> (record, profile_size)
seen = {u: 0 for u in USERS}
q_path = TIME_SPLIT_DIR / TASK / "train_questions.json"
print(f"[scan] streaming {q_path} ...", flush=True)
n = 0
for r in stream_json_array(str(q_path)):
    u = id_to_user.get(str(r.get("id")))
    if u is None:
        continue
    seen[u] += 1
    psize = len(r.get("profile", []))
    if psize > best[u][1]:
        best[u] = (r, psize)
    n += 1
    if n % 200 == 0:
        print(f"  matched {n} records so far (per-user seen: {seen})", flush=True)
    if all(seen[x] >= n_needed[x] for x in USERS):
        print(f"  all users' train records found after {n} matches — early exit", flush=True)
        break
print(f"[scan] done. per-user seen={seen}", flush=True)
for u in USERS:
    assert seen[u] == n_needed[u], f"{u}: saw {seen[u]} != needed {n_needed[u]}"

# --- tokenizer for rendering ---
print(f"[tok] loading {TOK_SRC} ...", flush=True)
tok = AutoTokenizer.from_pretrained(TOK_SRC)


def render(rec: dict) -> str:
    messages = []
    if rec.get("system"):
        messages.append({"role": "system", "content": rec["system"]})
    messages.append({"role": "user", "content": rec["user"]})
    messages.append({"role": "assistant", "content": rec["assistant"]})
    try:
        return tok.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False)


OUT_DIR.mkdir(parents=True, exist_ok=True)
summary = []
for u in USERS:
    snap, psize = best[u]
    profile = snap.get("profile", [])
    # emit flat records to an in-memory buffer via emit_bm25_records
    buf = io.StringIO()
    stats = emit_bm25_records(profile, TASK, BM25_K, buf)
    flat_lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    out_path = OUT_DIR / f"lamp3_{u}.jsonl"
    n_tok_max = 0
    with out_path.open("w") as f:
        for line in flat_lines:
            rec = json.loads(line)
            text = render(rec)
            n_tok = len(tok(text)["input_ids"])
            n_tok_max = max(n_tok_max, n_tok)
            f.write(json.dumps({"text": text}) + "\n")
    summary.append(
        dict(user=u, snapshot_id=str(snap["id"]), profile_size=psize,
             n_examples=len(flat_lines), stats=stats, max_tok=n_tok_max,
             out=str(out_path))
    )
    print(f"[write] {u}: profile={psize} examples={len(flat_lines)} "
          f"max_tok={n_tok_max} -> {out_path}", flush=True)

Path("data/ondevice_user_data/_build_summary.json").write_text(json.dumps(summary, indent=2))
print("[done] summary -> data/ondevice_user_data/_build_summary.json", flush=True)
