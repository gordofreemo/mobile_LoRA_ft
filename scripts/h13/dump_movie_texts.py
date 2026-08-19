#!/usr/bin/env python3
"""Dump the exact per-user training and eval texts the OPPU movie-tagging arm consumed.

Runs on conduit. Reuses their utils.py + the vendored rank_bm25 + their prompt.json,
and reproduces the text-construction blocks of oppu_replication/run_oppu.py line for
line (no model, no torch). Output feeds the h13 device/Mac data builder so the
on-device arm trains on byte-identical sequences.
"""
import json, os, sys
from pathlib import Path

ROOT = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(ROOT / "oppu_replication"))
sys.path.insert(0, str(ROOT / "third_party" / "OPPU"))
from rank_bm25 import BM25Okapi
from utils import get_first_k_tokens, extract_movie

TASK = "movie_tagging"
K = 1
DATA = ROOT / "data/oppu_release/data" / TASK
PROMPTS = json.load(open(ROOT / "third_party/OPPU/prompt/prompt.json"))[TASK]
test_data = json.load(open(DATA / "user_top_100_history.json"))
gold = {str(r["id"]): r["output"]
        for r in json.load(open(DATA / "user_top_100_history_label.json"))["golds"]}

out = []
for i, user in enumerate(test_data):
    # ---- training text (run_oppu.py profile loop) ----
    train_data = []
    for idx, q in enumerate(user["profile"]):
        for key in list(q):
            q[key] = get_first_k_tokens(str(q[key]), 768)
        prompt = PROMPTS["OPPU_input"].format(**q)
        full_prompt = PROMPTS["OPPU_full"].format(**q)
        if K > 0 and idx != 0:
            visible = user["profile"][:idx]
            for p in visible:
                for key in list(p):
                    p[key] = get_first_k_tokens(str(p[key]), 768)
            history_list = [PROMPTS["retrieval_history"].format(**p) for p in visible]
            bm25 = BM25Okapi([doc.split(" ") for doc in history_list])
            tq = PROMPTS["retrieval_query"].format(**q).split(" ")
            hist = "".join(bm25.get_top_n(tq, history_list, n=K))
            prompt = hist + "\n" + prompt
            full_prompt = hist + "\n" + full_prompt
        train_data.append({"prompt": prompt, "full_prompt": full_prompt})

    # ---- eval text (run_oppu.py test-inference block) ----
    visible = user["profile"]
    for p in visible:
        for key in list(p):
            p[key] = get_first_k_tokens(str(p[key]), 368)
    history_list = [PROMPTS["retrieval_history"].format(**p) for p in visible]
    bm25 = BM25Okapi([doc.split(" ") for doc in history_list])

    queries = []
    for q in user["query"]:
        article = extract_movie(q["input"])
        tp = PROMPTS["prompt"].format(article)
        tq = PROMPTS["retrieval_query_wokey"].format(article).split(" ")
        tp = "".join(bm25.get_top_n(tq, history_list, n=K)) + "\n" + tp
        queries.append({"id": str(q["id"]), "prompt": tp, "gold": gold.get(str(q["id"]))})

    out.append({"user_index": i, "user_id": str(user["user_id"]),
                "train": train_data, "queries": queries})
    print(f"user {i:03d} {user['user_id']}: {len(train_data)} train, {len(queries)} queries",
          flush=True)

dest = Path(sys.argv[1])
dest.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(dest, "w"))
print("wrote", dest, dest.stat().st_size, "bytes")
