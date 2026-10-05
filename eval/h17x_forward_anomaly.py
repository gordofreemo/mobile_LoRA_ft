#!/usr/bin/env python3
"""h17x Part F: is the stock arm's slower forward real, or an artifact of the barriered
decomposition? Reads explain_fwd records from train_bench_metrics_naxab_h17x_<date>.jsonl.

F1 = forward only (no backward graph built).
F2 = build value_and_grad, then eval(loss) [forward], then eval(grads) [backward]; this is
     exactly what the per-op harness does.
F4 = build the graph under the iteration's arm, flip MLX_ENABLE_NAX_N to the OTHER value,
     then eval(loss), eval(grads).

Reading: if off-on differs in F2 but not F1, the "forward" gap is produced by evaluating a
forward that has a backward graph attached; if F4's forward follows the eval-time env rather
than the graph-time arm, the forward evaluation is dispatching non-transposed kernels.
"""
import json, statistics, sys
from collections import defaultdict
p = sys.argv[1]
rs = [json.loads(l) for l in open(p) if l.strip()]
f = [r for r in rs if r.get("record_type") == "explain_fwd"]
d = defaultdict(list)
for r in f: d[(r["target_tokens"], r["arm"])].append(r)
med = lambda xs: statistics.median(xs) if xs else float("nan")
print(f"{'tok':>5} {'arm':4} {'n':>2} | {'F1 fwd-only':>11} | {'F2 graph':>8} {'F2 fwd':>7} {'F2 bwd':>7} | {'F4 fwd':>7} {'F4 bwd':>7} (eval-time NAX_N)")
for k in sorted(d):
    v = d[k]
    print(f"{k[0]:>5} {k[1]:4} {len(v):>2} | {med([x['f1_forward_only_s'] for x in v]):>11.3f} | "
          f"{med([x['f2_graph_s'] for x in v]):>8.3f} {med([x['f2_forward_s'] for x in v]):>7.3f} {med([x['f2_backward_s'] for x in v]):>7.3f} | "
          f"{med([x['f4_forward_s'] for x in v]):>7.3f} {med([x['f4_backward_s'] for x in v]):>7.3f} ({v[0]['f4_eval_nax_n']})")
print()
for t in sorted({k[0] for k in d}):
    off, on = d[(t, "off")], d[(t, "on")]
    g = lambda arr, key: med([x[key] for x in arr])
    print(f"{t:>5} tok: F1 off-on {g(off,'f1_forward_only_s')-g(on,'f1_forward_only_s'):+.3f} s | "
          f"F2 fwd off-on {g(off,'f2_forward_s')-g(on,'f2_forward_s'):+.3f} s | "
          f"F2 bwd off-on {g(off,'f2_backward_s')-g(on,'f2_backward_s'):+.3f} s | "
          f"F4 fwd (graph=off, eval NAX_N=1) - (graph=on, eval NAX_N=0) {g(off,'f4_forward_s')-g(on,'f4_forward_s'):+.3f} s | "
          f"F4 bwd {g(off,'f4_backward_s')-g(on,'f4_backward_s'):+.3f} s")
print("thermal:", sorted({r['thermal_state'] for r in f}))
