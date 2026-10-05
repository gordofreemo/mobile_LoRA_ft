"""Adapter depth at 4-bit (experiments/2026-09-24-depth-quality-4bit-plan.md).

Movie tagging, 100 users, 3,302 queries. Rows: 36 blocks (_r5q4t), the last 24 (_r5q4td24)
and the last 18 (_r5q4td18), all trained and evaluated on the 4-bit task model and scored
against the shared 4-bit Task arm (_q4). Same layout as the 16-bit depth table in the plan,
plus the paired difference against 36 blocks and same%. Reads results/oppu_rep_q4b_2026-09-24/.

Delta = query-level mean improvement over Task, Delta_u = mean of per-user improvements,
p_u = user-grouped t-test. retains = Delta (Delta_u) as a share of the 36-block value.
same% = share of queries whose prediction string equals the 36-block adapter's.
"""
import glob
import json
import os
from collections import defaultdict

from scipy import stats

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                 'results', 'oppu_rep_q4b_2026-09-24')
ARMS = [(36, '_r5q4t'), (24, '_r5q4td24'), (18, '_r5q4td18')]


def load_pairs(tag):
    return {r["id"]: r for r in map(json.loads, open(f'{R}/scores/score_movie_tagging{tag}.pairs.jsonl'))}


def preds(tag):
    out = {}
    for f in sorted(glob.glob(f'{R}/movie_preds/oppu_k1{tag}_u*_preds.json')):
        for x in json.load(open(f))["golds"]:
            out[x["id"]] = x["output"]
    return out


def ttest(x):
    return stats.ttest_1samp(x, 0).pvalue if any(v != 0 for v in x) else 1.0


def grouped(x, g):
    per = defaultdict(list)
    for v, u in zip(x, g):
        per[u].append(v)
    m = [sum(v) / len(v) for v in per.values()]
    return sum(x) / len(x), ttest(x), sum(m) / len(m), ttest(m)


P = {tag: load_pairs(tag) for _, tag in ARMS}
S = {tag: json.load(open(f'{R}/scores/score_movie_tagging{tag}.json')) for _, tag in ARMS}
ids = sorted(P['_r5q4t'])
for _, tag in ARMS:
    assert P[tag].keys() == set(ids), tag
    # every arm is scored against the same 4-bit Task predictions
    assert all(P[tag][i]['task_score'] == P['_r5q4t'][i]['task_score'] for i in ids), tag
g = [P['_r5q4t'][i]['user_id'] for i in ids]
full = grouped([P['_r5q4t'][i]['diff'] for i in ids], g)
ref = preds('_r5q4t')
print("4-bit Task arm (_q4): %.4f   n = %d queries, %d users" % (
    S['_r5q4t']['task_accuracy'], len(ids), S['_r5q4t']['n_users']))
print("%-6s %-10s | %6s %8s %8s %8s | %15s | %8s %8s %8s %8s | %6s" % (
    "depth", "tag", "acc", "D", "D_u", "p_u", "retains D / D_u",
    "vs 36", "p", "d_u", "p_u", "same%"))
for depth, tag in ARMS:
    own = grouped([P[tag][i]['diff'] for i in ids], g)
    if depth == 36:
        print("%-6d %-10s | %6.4f %+8.4f %+8.4f %8.3g | %15s | %8s %8s %8s %8s | %6s" % (
            depth, tag, S[tag]['oppu_accuracy'], own[0], own[2], own[3], "-", "-", "-", "-", "-", "-"))
        continue
    d = grouped([P[tag][i]['oppu_score'] - P['_r5q4t'][i]['oppu_score'] for i in ids], g)
    pr = preds(tag)
    assert pr.keys() == ref.keys() == set(ids), tag
    same = 100.0 * sum(pr[k] == ref[k] for k in ids) / len(ids)
    print("%-6d %-10s | %6.4f %+8.4f %+8.4f %8.3g | %6.0f%% / %5.0f%% | %+8.4f %8.3g %+8.4f %8.3g | %5.1f%%" % (
        depth, tag, S[tag]['oppu_accuracy'], own[0], own[2], own[3],
        100 * own[0] / full[0], 100 * own[2] / full[2], d[0], d[1], d[2], d[3], same))
print("\nvs 36 = accuracy of this depth minus the 36-block adapter, paired over the same queries.")
print("Task scores asserted identical per query across all three arms.")
