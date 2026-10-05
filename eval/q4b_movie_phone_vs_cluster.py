import json, glob, os, sys, math
from collections import defaultdict
from scipy import stats
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
sys.path.insert(0, R+'/eval')
from h13_score import score, LABELS
Q=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results/oppu_rep_q4b_2026-09-24')
texts={u["user_id"]:u for u in json.load(open(R+'/data/oppu_movie/h13_movie_texts.json'))}
gold={q["id"]:q["gold"] for u in texts.values() for q in u["queries"]}
uid={q["id"]:u for u,v in texts.items() for q in v["queries"]}
def lc(f): return {g["id"]:g["output"] for g in json.load(open(f))["golds"]}
def shards(p):
    d={}
    for f in sorted(glob.glob(p)): d.update(lc(f))
    return d
arms={'C16T':lc(Q+'/movie_preds/task_k1_preds.json'),'C4T':lc(Q+'/movie_preds/task_k1_q4_preds.json'),
      'C16U':shards(Q+'/movie_preds/oppu_k1_r5_u*_preds.json'),'C4U':shards(Q+'/movie_preds/oppu_k1_r5q4_u*_preds.json')}
_t=shards(Q+'/movie_preds/oppu_k1_r5q4t_u*_preds.json')   # per-user adapters trained on the 4-bit model
if _t: arms['C4Ut']=_t
for name,fn in (('PT','rag'),('PU','cluster'),('PD','device')):
    d={}
    for u in texts:
        p=f'{R}/results/ondevice/h13_preds/{u}/{fn}.jsonl'
        for l in open(p):
            if l.strip(): r=json.loads(l); d[r["id"]]=r["output"]
    arms[name]=d
ids=sorted(gold)
print('queries', len(ids), 'users', len(texts), {k:len(v) for k,v in arms.items()}, 'missing', {k:len(set(ids)-set(v)) for k,v in arms.items()})
def lab(x):
    s=str(x).strip(); return LABELS.index(s) if s in LABELS else -1
acc={k:sum(score(v[i],gold[i]) for i in ids)/len(ids) for k,v in arms.items()}
print('accuracy', {k:round(a,4) for k,a in acc.items()})
def agree(a,b): return sum(lab(arms[a][i])==lab(arms[b][i]) for i in ids)/len(ids)
for a,b in [x for x in (('C4T','PT'),('C4U','PU'),('C4U','PD'),('C16T','C4T'),('C16T','PT'),('C16U','C4U'),('C16U','PU'),('PU','PD'),('C4Ut','PD'),('C4Ut','C4U')) if x[0] in arms and x[1] in arms]:
    print(f'agree {a:4s} vs {b:4s}: {agree(a,b):.4f}  (differ on {1-agree(a,b):.2%})')
def grouped(t,o):
    per=defaultdict(list)
    for i in ids: per[uid[i]].append(score(arms[o][i],gold[i])-score(arms[t][i],gold[i]))
    m=[sum(v)/len(v) for v in per.values()]
    ql=[score(arms[o][i],gold[i])-score(arms[t][i],gold[i]) for i in ids]
    return sum(ql)/len(ql), stats.ttest_1samp(ql,0).pvalue, sum(m)/len(m), stats.ttest_1samp(m,0).pvalue
for t,o in [x for x in (('C16T','C16U'),('C4T','C4U'),('PT','PU'),('PT','PD'),('C4T','PT'),('C4U','PU'),('C4T','C4Ut'),('C4Ut','PD'),('C4U','C4Ut')) if x[0] in arms and x[1] in arms]:
    d,p,du,pu=grouped(t,o); print(f'{o:4s} - {t:4s}: D {d:+.4f} (p {p:.3g})  D_u {du:+.4f} (p_u {pu:.3g})')
