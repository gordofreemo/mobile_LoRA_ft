#!/usr/bin/env python3
"""Freeze the h13 anytime-queue order: descending predicted paired-queries-per-device-hour.

Cost model (device, NAX-ON, from the campaign's own measurements):
  train_s = TRAIN_SLOPE * epochs * train_tokens + TRAIN_INTERCEPT      (h5/E2E C0 cost law)
  eval_s  = n_arms * (LOAD_S + n_q * (prompt_tok/PREFILL + DECODE_TOK/DECODE))
Parameters are recorded in the output so the frozen order stays auditable.
Ordering only; absolute predictions are refined by the Phase 0 smoke but the ORDER IS FROZEN.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 3
TRAIN_SLOPE, TRAIN_INTERCEPT = 0.0101, 222.0    # wall_s ~ 0.0101*tokens + 222 (r=0.984)
N_ARMS = 4                                       # rag, cluster, mac, device
LOAD_S = 60.0
PREFILL_TOK_S = 620.0                            # h2 inference, 3B 4-bit
DECODE_TOK_S = 22.0                              # throttled sustained plateau (h2/h4)
DECODE_TOK = 200.0                               # their max_new_tokens

stats = json.load(open(ROOT / "data/oppu_movie/h13_user_stats.json"))
for r in stats:
    r["train_s"] = TRAIN_SLOPE * EPOCHS * r["tok_train"] + TRAIN_INTERCEPT
    r["eval_s"] = N_ARMS * (LOAD_S + r["n_q"] * (r["med_q"] / PREFILL_TOK_S
                                                 + DECODE_TOK / DECODE_TOK_S))
    r["cost_s"] = r["train_s"] + r["eval_s"]
    r["q_per_hour"] = r["n_q"] / (r["cost_s"] / 3600.0)

order = sorted(stats, key=lambda r: (-r["q_per_hour"], -r["n_q"], r["i"]))
out = {"frozen": "2026-08-19", "task": "movie_tagging", "n_users": len(order),
       "params": dict(epochs=EPOCHS, train_slope=TRAIN_SLOPE, train_intercept=TRAIN_INTERCEPT,
                      n_arms=N_ARMS, load_s=LOAD_S, prefill_tok_s=PREFILL_TOK_S,
                      decode_tok_s=DECODE_TOK_S, decode_tok=DECODE_TOK),
       "queue": [{"rank": n, "user_index": r["i"], "user_id": r["uid"], "n_train": r["n_train"],
                  "n_q": r["n_q"], "tok_train": r["tok_train"],
                  "pred_train_s": round(r["train_s"], 1), "pred_eval_s": round(r["eval_s"], 1),
                  "pred_q_per_hour": round(r["q_per_hour"], 2)}
                 for n, r in enumerate(order)]}
dest = ROOT / "data/oppu_movie/h13_queue.json"
json.dump(out, open(dest, "w"), indent=1)

cum_q = cum_s = 0
print(f"{'rank':>4} {'idx':>4} {'user':>9} {'ntr':>4} {'nq':>4} {'train_m':>8} {'eval_m':>7} {'q/h':>7} {'cumq':>6} {'cumh':>6}")
for e in out["queue"]:
    cum_q += e["n_q"]; cum_s += e["pred_train_s"] + e["pred_eval_s"]
    if e["rank"] < 12 or e["rank"] > 96 or e["rank"] in (24, 49, 74):
        print(f"{e['rank']:>4} {e['user_index']:>4} {e['user_id']:>9} {e['n_train']:>4} "
              f"{e['n_q']:>4} {e['pred_train_s']/60:>8.1f} {e['pred_eval_s']/60:>7.1f} "
              f"{e['pred_q_per_hour']:>7.1f} {cum_q:>6} {cum_s/3600:>6.1f}")
print(f"\nALL 100: {cum_q} paired queries, {cum_s/3600:.1f} predicted device-hours")
