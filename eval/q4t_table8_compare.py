"""Table 8 at 4-bit, three arms over the same queries: the 4-bit Task arm (_q4), the
bf16-trained per-user adapters evaluated on it (_r5q4), and per-user adapters trained
on the 4-bit model itself (_r5q4t). Reads results/oppu_rep_q4b_2026-09-24/scores/.

All differences are oriented so + = better (the scorer flips MAE). p is a paired
t-test over queries, p_u over per-user means.
"""
import json
import os
from collections import defaultdict

from scipy import stats

Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                 'results', 'oppu_rep_q4b_2026-09-24', 'scores')
T = ['news_categorize', 'news_headline', 'movie_tagging', 'citation',
     'product_rating', 'tweet_paraphrase', 'scholarly_title']


def load(f):
    return {r["id"]: r for r in map(json.loads, open(f))}


def ttest(x):
    return stats.ttest_1samp(x, 0).pvalue if any(v != 0 for v in x) else 1.0


def st(x, g):
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


print("%-17s %-8s | %6s | %-27s | %-27s | %-27s" % (
    "", "", "4-bit", "+User, bf16-trained", "+User, 4-bit-trained", "4-bit-trained - bf16-trained"))
print("%-17s %-8s | %6s | %6s %6s %6s %6s | %6s %6s %6s %6s | %6s %6s %6s %6s" % (
    "task", "metric", "Task", "score", "D", "D_u", "p_u", "score", "D", "D_u", "p_u", "d", "p", "d_u", "p_u"))
for t in T:
    sa = json.load(open(f'{Q}/score_{t}_r5q4.json'))
    sb = json.load(open(f'{Q}/score_{t}_r5q4t.json'))
    m = metric(sa)
    a = load(f'{Q}/score_{t}_r5q4.pairs.jsonl')
    b = load(f'{Q}/score_{t}_r5q4t.pairs.jsonl')
    assert a.keys() == b.keys(), t
    # both arms are scored against the same 4-bit Task predictions
    assert all(a[i]['task_score'] == b[i]['task_score'] for i in a), t
    sign = -1 if t == 'product_rating' else 1
    ids = sorted(a)
    g = [a[i]['user_id'] for i in ids]
    ga = st([a[i]['diff'] for i in ids], g)
    gb = st([b[i]['diff'] for i in ids], g)
    dd = st([sign * (b[i]['oppu_score'] - a[i]['oppu_score']) for i in ids], g)
    print("%-17s %-8s | %6.3f | %6.3f %+6.3f %+6.3f %6.2g | %6.3f %+6.3f %+6.3f %6.2g | %+6.3f %6.2g %+6.3f %6.2g" % (
        t, m, sa['task_' + m],
        sa['oppu_' + m], ga[0], ga[2], ga[3],
        sb['oppu_' + m], gb[0], gb[2], gb[3],
        dd[0], dd[1], dd[2], dd[3]))
