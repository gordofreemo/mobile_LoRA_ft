import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _e2e_plot_style import apply_rc
import argparse

# Figure 1's two traces are selected on the command line. They used to be
# hardcoded to the July 28-block pair (stock 2026-07-08 vs repaired h14energy
# 2026-09-12), which is the 1.93x / ~3 W pair the body no longer reports. The
# 36-block pair is R1 (stock, unplugged, 85% start) against h15c405 (repaired,
# unplugged, 85% start) -- both traces must start at the same gauge level or
# the fuel-gauge plateau biases one arm's energy.
_ap = argparse.ArgumentParser()
_ap.add_argument("--stock-file"); _ap.add_argument("--stock-start")
_ap.add_argument("--repaired-file"); _ap.add_argument("--repaired-start")
_ap.add_argument("--xmax", type=float, default=None)
_A, _ = _ap.parse_known_args()

R=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ondevice") + os.sep
def load(f, start_ts):
    sid=None; train=[]; batt=[]
    for l in open(R+f):
        try: r=json.loads(l)
        except: continue
        if r.get("record_type")=="run_start" and r.get("timestamp_utc")==start_ts: sid=r["bench_session_id"]
        if sid and r.get("bench_session_id")==sid:
            if r["record_type"]=="train": train.append((r["elapsed_s"]/3600, r["iter_per_sec"], r["step"]))
            if r["record_type"]=="battery": batt.append((r["elapsed_s"]/3600, 100*r["battery_level"]))
            if r["record_type"]=="run_end": end=r
    return np.array(train), np.array(batt), end
_SD = ("train_bench_metrics_e2e_smollm3_a1lamp_2026-07-08.jsonl", "2026-07-07T19:15:44Z")
_RD = ("train_bench_metrics_naxab_e2e_h14energy_2026-09-12.jsonl", "2026-09-12T09:12:23Z")
stock_t, stock_b, stock_end = load(_A.stock_file or _SD[0], _A.stock_start or _SD[1])
rep_t, rep_b, rep_end = load(_A.repaired_file or _RD[0], _A.repaired_start or _RD[1])
for n,t,b in [("unrepaired",stock_t,stock_b),("repaired",rep_t,rep_b)]:
    print(n, "train pts",len(t),"hours",t[-1,0].round(2),"steps",int(t[-1,2]),"battery %.0f->%.0f"%(b[0,1],b[-1,1]), "cold iter/s %.3f"%t[:3,1].mean(), "final iter/s %.3f"%t[-10:,1].mean())
def smooth(y,k=5):
    """Centred rolling mean. np.convolve(mode="same") zero-pads past the ends,
    which dragged the last samples toward zero and put a false collapse at the
    tail of both curves. min_periods = full window drops the truncated windows
    instead, matching roll() in the other figure scripts."""
    import pandas as pd
    k=min(k,len(y))
    if k<2: return y
    return pd.Series(y).rolling(k,center=True,min_periods=k).mean().values
apply_rc()   # shared house style: weights, colours, grid, spines
fig,ax=plt.subplots(figsize=(3.3,2.5),dpi=200)
c_s,c_r="#0072B2","#D55E00"
for t,b,c,lab in [(stock_t,stock_b,c_s,"unrepaired runtime"),(rep_t,rep_b,c_r,"repaired runtime")]:
    ax.plot(t[:,0],t[:,1],color=c,alpha=0.25,lw=0.6)
    ax.plot(t[:,0],smooth(t[:,1]),color=c,lw=1.6,label=lab)
ax.set_xlabel("wall-clock time (hours)"); ax.set_ylabel("throughput (steps/s)")
ax.set_ylim(0,None)
ax.set_xlim(0, _A.xmax or max(stock_t[-1,0], rep_t[-1,0]) * 1.03)
ax2=ax.twinx()
for b,c in [(stock_b,c_s),(rep_b,c_r)]:
    ax2.step(b[:,0],b[:,1],where="post",color=c,lw=1.0,ls="--")
ax2.set_ylabel("battery level (%, dashed)"); ax2.set_ylim(0,100); ax2.spines["top"].set_visible(False)
ax.text(rep_t[-1,0]+0.06,0.185,"done at %.1f h,\n%d%% of charge"%(rep_t[-1,0],rep_b[0,1]-rep_b[-1,1]),color=c_r,fontsize=7,va="top")
# Stock label goes in the empty band below its own throughput trace: at the old
# position (end - 0.92 h, 0.085) it sat on the stock battery staircase once the
# 36-block run ran to 2.5 h.
ax.text(0.55,0.075,"done at %.1f h,\n%d%% of charge"%(stock_t[-1,0],stock_b[0,1]-stock_b[-1,1]),color=c_s,fontsize=7,va="top")
ax.legend(loc="upper right",frameon=False,fontsize=7)
fig.tight_layout(); OUT = os.environ.get("HOOK_OUT", "results/ondevice/figures/hook_run")
fig.savefig(OUT+".pdf", bbox_inches="tight"); fig.savefig(OUT+".png", dpi=200, bbox_inches="tight")
print("wrote", OUT+".pdf")
print("saved")
