import json, os
from collections import defaultdict
from scipy import stats
Q=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results/oppu_rep_q4b_2026-09-24','scores')
T=['news_categorize','news_headline','movie_tagging','citation','product_rating','tweet_paraphrase','scholarly_title']
def load(f): return {r["id"]:r for r in map(json.loads, open(f)) if r}
def st(x, g):
    per=defaultdict(list)
    for v,u in zip(x,g): per[u].append(v)
    m=[sum(v)/len(v) for v in per.values()]
    p = stats.ttest_1samp(x,0).pvalue if any(x) else 1.0
    pu = stats.ttest_1samp(m,0).pvalue if any(m) else 1.0
    return sum(x)/len(x), p, sum(m)/len(m), pu
print('4-bit minus bf16, oriented + = 4-bit better (MAE sign-flipped). query-level D (p) | user-grouped D_u (p_u)')
for t in T:
    a=load(f'{Q}/score_{t}_r5.pairs.jsonl'); b=load(f'{Q}/score_{t}_r5q4.pairs.jsonl')
    assert a.keys()==b.keys()
    s=-1 if t=='product_rating' else 1
    ids=sorted(a); g=[a[i]["user_id"] for i in ids]
    flips_t=sum(a[i]["task_score"]!=b[i]["task_score"] for i in ids)/len(ids)
    out=[]
    for lab,f in (('Task',lambda r:r["task_score"]),('+User',lambda r:r["oppu_score"])):
        x=[s*(f(b[i])-f(a[i])) for i in ids]; out.append((lab,)+st(x,g))
    x=[b[i]["diff"]-a[i]["diff"] for i in ids]; out.append(('gain',)+st(x,g))
    print(f'{t:17s} n={len(ids):5d} ' + '  '.join(f'{l}: {d:+.3f} ({p:.2g}) | {du:+.3f} ({pu:.2g})' for l,d,p,du,pu in out) + f'  task-score changed on {flips_t:.1%}')
