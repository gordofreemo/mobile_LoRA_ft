#!/usr/bin/env python3
"""Eval-design power analysis by subsampling the OPPU-round pairs files.

Question: would this project's own eval shape (1 query per user, K~100 users)
have detected the effects the OPPU protocol measured on the same adapters?
For each arm's pairs.jsonl, repeatedly subsample s queries per user
(s = 1, 2, 3, 5, 10, all), run the paired t-test on the subsample, and record
the fraction of resamples reaching p<0.05 — the detection rate ("power") of
that eval shape against this exact, already-measured effect.

Pure analysis over existing per-query diffs — no model, no GPU. Run in a
Condor CPU job (login nodes are capped at 2 GB).

Output: results/oppu_rep/power_subsample.json (+ printed table).
"""

import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
OUT = PROJECT_ROOT / "results" / "oppu_rep"

ARMS = [  # (label, pairs file)
    ("movie_hot", OUT / "score_movie_tagging.pairs.jsonl"),
    ("movie_cold_r5", OUT / "score_movie_tagging_r5.pairs.jsonl"),
    ("headline", OUT / "score_news_headline.pairs.jsonl"),
    ("news_cat_null", OUT / "score_news_categorize.pairs.jsonl"),
]
S_LEVELS = [1, 2, 3, 5, 10, None]   # queries per user; None = all
N_RESAMPLES = 10_000
SEED = 0


def main():
    from scipy import stats as st
    rng = random.Random(SEED)
    out_path = OUT / "power_subsample.json"
    if out_path.exists() and "--overwrite" not in sys.argv:
        sys.exit(f"REFUSING to overwrite {out_path} (pass --overwrite)")

    results = {"n_resamples": N_RESAMPLES, "seed": SEED, "arms": {}}
    for label, path in ARMS:
        by_user = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                by_user.setdefault(r["user_id"], []).append(r["diff"])
        users = sorted(by_user)
        full = [d for u in users for d in by_user[u]]
        full_mean = sum(full) / len(full)
        rows = {}
        for s in S_LEVELS:
            if s is None:
                t = st.ttest_1samp(full, 0.0)
                rows["all"] = {"queries_per_user": "all", "n_mean": len(full),
                               "detect_rate": float(t.pvalue < 0.05),
                               "mean_effect": full_mean}
                continue
            hits, n_tot, eff_tot = 0, 0, 0.0
            for _ in range(N_RESAMPLES):
                sample = []
                for u in users:
                    qs = by_user[u]
                    take = qs if len(qs) <= s else rng.sample(qs, s)
                    sample.extend(take)
                if all(d == sample[0] for d in sample):
                    p = 1.0    # zero-variance subsample can't be tested
                else:
                    p = float(st.ttest_1samp(sample, 0.0).pvalue)
                hits += (p < 0.05)
                n_tot += len(sample)
                eff_tot += sum(sample) / len(sample)
            rows[str(s)] = {"queries_per_user": s,
                            "n_mean": n_tot / N_RESAMPLES,
                            "detect_rate": hits / N_RESAMPLES,
                            "mean_effect": eff_tot / N_RESAMPLES}
        results["arms"][label] = {"n_users": len(users), "n_queries": len(full),
                                  "full_mean_effect": full_mean, "levels": rows}
        print(f"== {label}: {len(users)} users, {len(full)} queries, "
              f"full effect {full_mean:+.4f}")
        for key, row in rows.items():
            print(f"   s={key:>3}  n~{row['n_mean']:6.0f}  "
                  f"detect@p<.05 {row['detect_rate']:6.1%}  "
                  f"subsample effect {row['mean_effect']:+.4f}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
