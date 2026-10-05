import json,glob,statistics,collections,sys,os
files=sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results/ondevice/train_bench_metrics_perop-gc-*h14b*.jsonl')))
print(files)
rows=[]
for f in files:
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        if r.get('record_type') not in ('iter','perop_iter') and 'phase_backward_s' not in r: continue
        rows.append(r)
print("iter records",len(rows), collections.Counter(r.get('record_type') for r in rows))
# group by (gc, session, tokens)
g=collections.defaultdict(list)
for r in rows:
    if r.get('warmup'): continue
    if 'phase_backward_s' not in r: continue
    g[(r['gradient_checkpointing'], (r.get('nax_arm') or 'na')+':'+r['bench_session_id'][:8], r['target_tokens'])].append(r)
sess=sorted({(k[0],k[1]) for k in g}, key=lambda x:(x[1]))
# order sessions by first timestamp
first={}
for r in rows: first.setdefault((r.get('nax_arm') or 'na')+':'+r['bench_session_id'][:8], r['timestamp_utc'])
sess=sorted({(k[0],k[1]) for k in g}, key=lambda x: first[x[1]])
print("sessions (gc, id, first ts):",[(a,b,first[b]) for a,b in sess])
def med(v): return statistics.median(v)
print(f"{'tok':>5} {'gc':>5} {'sess':>16} {'n':>3} {'iter':>7} {'fwd':>7} {'bwd':>7} {'bwd/fwd':>7} {'peakMB':>7}")
tab={}
for (gc,sid,tok),rs in sorted(g.items(), key=lambda kv:(kv[0][2], first[kv[0][1]])):
    it=med([r['iter_seconds'] for r in rs]); fw=med([r['phase_forward_s'] for r in rs]); bw=med([r['phase_backward_s'] for r in rs]); pk=max(r['peak_mem_bytes'] for r in rs)/2**20
    tab[(gc,sid,tok)]=(fw,bw,it)
    print(f"{tok:>5} {str(gc):>5} {sid:>16} {len(rs):>3} {it:7.3f} {fw:7.3f} {bw:7.3f} {bw/fw:7.2f} {pk:7.0f}")
print("\nRecompute share = (bwd_on - bwd_off)/bwd_on, forward-units = (bwd_on-bwd_off)/fwd_on")
ons=[s for gc,s in sess if gc]; offs=[s for gc,s in sess if not gc]
for tok in sorted({k[2] for k in tab}):
    for on in ons:
        for off in offs:
            if (True,on,tok) in tab and (False,off,tok) in tab:
                fw,bw,_=tab[(True,on,tok)]; fwo,bwo,_=tab[(False,off,tok)]
                print(f"tok {tok}: on {on} vs off {off}: recompute {bw-bwo:.3f}s = {100*(bw-bwo)/bw:.1f}% of bwd_on, {(bw-bwo)/fw:.2f} fwd units; bwd_off/fwd_off={bwo/fwo:.2f}; fwd on/off {fw:.3f}/{fwo:.3f}")

print("\nForward-normalized (thermal-robust): ratio_on - ratio_off = recompute in forward units; share = that / ratio_on")
for tok in sorted({k[2] for k in tab}):
    for on in ons:
        for off in offs:
            if on.split(':')[0]!=off.split(':')[0]: continue
            if (True,on,tok) in tab and (False,off,tok) in tab:
                fw,bw,_=tab[(True,on,tok)]; fwo,bwo,_=tab[(False,off,tok)]
                ron=bw/fw; roff=bwo/fwo
                print(f"[{on.split(':')[0]}] tok {tok}: bwd/fwd on={ron:.2f} off={roff:.2f} -> recompute {ron-roff:.2f} fwd units = {100*(ron-roff)/ron:.0f}% of backward (on {on[-8:]}, off {off[-8:]})")
