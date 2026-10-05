"""Shuffled-history control at 4-bit (experiments/2026-09-24-shuffled-control-4bit-plan.md).

Same layout as results/oppu_rep_r2_shuffled_control_2026-09-23.txt (the 16-bit control):
Task, own-history (_r5q4t) and shuffled-history (_r5q4tshuf) arms, each with D, D_u, p_u,
then the paired own - shuffled difference and same%. All three arms run on the 4-bit task
model; the Task arm is the shared _q4 baseline. Reads results/oppu_rep_q4b_2026-09-24/.

D = query-level mean improvement over Task, D_u = mean of per-user improvements,
p_u = user-grouped t-test. MAE is oriented so + = better (the scorer flips it).
same% = share of queries where the two adapters give the identical prediction string.
"""
import glob
import json
import os
import sys
from collections import defaultdict

from scipy import stats

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                 'results', 'oppu_rep_q4b_2026-09-24')
PREDS = {'movie_tagging': 'movie_preds', 'news_categorize': 'news_categorize_preds'}
TASKS = [t for t in PREDS if os.path.exists(f'{R}/scores/score_{t}_r5q4tshuf.json')]


def load_pairs(f):
    return {r["id"]: r for r in map(json.loads, open(f))}


def preds(task, tag):
    out = {}
    for f in sorted(glob.glob(f'{R}/{PREDS[task]}/oppu_k1{tag}_u*_preds.json')):
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


def metric(d):
    for k in ("accuracy", "rouge_1", "MAE"):
        if "task_" + k in d:
            return k
    raise KeyError(f"no known metric in {d.get('task')}")


if not TASKS:
    sys.exit("no _r5q4tshuf score files yet")
print("%-17s %-8s %4s | %7s %7s %8s %8s %8s | %7s %8s %8s %8s | %8s %8s %8s %8s | %6s" % (
    "task", "metric", "n_u", "Task", "+User", "D", "D_u", "p_u", "+Shuf", "D", "D_u", "p_u",
    "own-shuf", "p", "d_u", "p_u", "same%"))
for t in TASKS:
    a = json.load(open(f'{R}/scores/score_{t}_r5q4t.json'))
    s = json.load(open(f'{R}/scores/score_{t}_r5q4tshuf.json'))
    m = metric(a)
    pa = load_pairs(f'{R}/scores/score_{t}_r5q4t.pairs.jsonl')
    ps = load_pairs(f'{R}/scores/score_{t}_r5q4tshuf.pairs.jsonl')
    assert pa.keys() == ps.keys(), t
    # both arms are scored against the same 4-bit Task predictions
    assert all(pa[i]['task_score'] == ps[i]['task_score'] for i in pa), t
    ids = sorted(pa)
    g = [pa[i]['user_id'] for i in ids]
    own = grouped([pa[i]['diff'] for i in ids], g)
    shuf = grouped([ps[i]['diff'] for i in ids], g)
    d = grouped([pa[i]['diff'] - ps[i]['diff'] for i in ids], g)
    U, S = preds(t, '_r5q4t'), preds(t, '_r5q4tshuf')
    assert U.keys() == S.keys() == set(ids), t
    same = 100.0 * sum(U[k] == S[k] for k in ids) / len(ids)
    drift = abs(a["task_" + m] - s["task_" + m])
    print("%-17s %-8s %4d | %7.4f %7.4f %+8.4f %+8.4f %8.3g | %7.4f %+8.4f %+8.4f %8.3g | %+8.4f %8.3g %+8.4f %8.3g | %5.1f%%" % (
        t, m, a["n_users"], a["task_" + m], a["oppu_" + m], own[0], own[2], own[3],
        s["oppu_" + m], shuf[0], shuf[2], shuf[3], d[0], d[1], d[2], d[3], same)
        + ("   (Task baseline drift %.5f between scoring runs)" % drift if drift > 1e-9 else ""))
    if own[0]:
        print("%-17s shuffled D / own D = %.2f   (plan: <= ~1/3 personal, > ~1/2 partly quantization recovery)"
              % ("", shuf[0] / own[0]))
print("\nown-shuf = own-history minus shuffled-history adapter, paired over the same queries.")
print("Task scores asserted identical per query between the two arms.")
