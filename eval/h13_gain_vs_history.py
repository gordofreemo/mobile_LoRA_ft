#!/usr/bin/env python3
"""R3: per-user personalization gain against history length (h13, analysis only).

Joins the h13 on-device eval plane (results/ondevice/h13_preds/<user>/{rag,cluster,
device}.jsonl) with each user's training history size and measured on-device training
cost (data/oppu_movie/h13_queue.json for records/tokens, the h13 training telemetry
for wall-clock seconds), and asks whether the device-minus-rag gain grows with the
number of history records.

No new runs: every input already exists. Reuses eval/h13_score.py's scorer verbatim.
"""
import argparse, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from h13_score import score, paired_stats, LABELS  # noqa: E402

ARMS = ["rag", "cluster", "device"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(ROOT / "results/ondevice/h13_preds"))
    ap.add_argument("--texts", default=str(ROOT / "data/oppu_movie/h13_movie_texts.json"))
    ap.add_argument("--queue", default=str(ROOT / "data/oppu_movie/h13_queue.json"))
    ap.add_argument("--telemetry",
                    default=str(ROOT / "results/ondevice/h13_telemetry/train_bench_metrics_h13_nax-on.jsonl"))
    ap.add_argument("--out-json", default=str(ROOT / "results/ondevice/h13_gain_vs_history.json"))
    ap.add_argument("--out-tsv", default=str(ROOT / "results/ondevice/h13_gain_vs_history.tsv"))
    ap.add_argument("--out-fig", default=str(ROOT / "figures/h13_gain_vs_history.pdf"))
    args = ap.parse_args()

    texts = {u["user_id"]: u for u in json.load(open(args.texts))}
    gold = {q["id"]: q["gold"] for u in texts.values() for q in u["queries"]}
    qentries = {e["user_id"]: e for e in json.load(open(args.queue))["queue"]}
    queue_order = [e["user_id"] for e in json.load(open(args.queue))["queue"]]

    # measured on-device training seconds: last run_end per user (reruns append)
    train_s, train_steps = {}, {}
    for line in open(args.telemetry):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("record_type") != "run_end" or not r.get("user_id"):
            continue
        if r.get("error"):
            continue
        train_s[r["user_id"]] = r.get("elapsed_s")
        train_steps[r["user_id"]] = r.get("total_steps")

    rows, qs = [], {a: {} for a in ARMS}
    q_user = {}
    for uid in queue_order:
        d = Path(args.preds) / uid
        got = {}
        for a in ARMS:
            f = d / f"{a}.jsonl"
            if f.exists():
                got[a] = {json.loads(l)["id"]: json.loads(l) for l in open(f) if l.strip()}
        want = {q["id"] for q in texts[uid]["queries"]}
        if not all(a in got and set(got[a]) >= want for a in ARMS):
            continue
        acc = {}
        for a in ARMS:
            s = [score(got[a][q["id"]]["output"], gold[q["id"]]) for q in texts[uid]["queries"]]
            acc[a] = sum(s) / len(s)
            for q in texts[uid]["queries"]:
                qs[a][q["id"]] = score(got[a][q["id"]]["output"], gold[q["id"]])
                q_user[q["id"]] = uid
        e = qentries[uid]
        rows.append({
            "user_id": uid, "rank": e["rank"],
            "n_train": e["n_train"], "n_q": e["n_q"], "tok_train": e["tok_train"],
            "train_s": train_s.get(uid), "train_steps": train_steps.get(uid),
            "rag": acc["rag"], "cluster": acc["cluster"], "device": acc["device"],
            "device_rag": acc["device"] - acc["rag"],
            "device_cluster": acc["device"] - acc["cluster"],
        })

    rows.sort(key=lambda r: r["n_train"])
    n = len(rows)
    print(f"users with all three arms: {n}   queries: {len(q_user)}")

    # ---- correlation of the per-user gain with history length ----
    def corr(xs, ys):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
        if sx == 0 or sy == 0:
            return float("nan")
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

    from scipy import stats as st
    out = {"n_users": n, "n_queries": len(q_user), "rows": rows, "correlations": {}, "bins": {}}
    x = [r["n_train"] for r in rows]
    logx = [__import__("math").log10(v) for v in x]
    for target in ("device_rag", "device_cluster"):
        y = [r[target] for r in rows]
        pr = st.pearsonr(x, y)
        prl = st.pearsonr(logx, y)
        sr = st.spearmanr(x, y)
        sl = st.linregress(x, y)
        out["correlations"][target] = {
            "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "pearson_r_log10n": float(prl.statistic), "pearson_p_log10n": float(prl.pvalue),
            "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue),
            "slope_per_record": float(sl.slope), "slope_p": float(sl.pvalue),
            "intercept": float(sl.intercept),
            "mean": sum(y) / n, "sd": statistics.stdev(y),
        }

    # ---- quartile bins by history length ----
    edges = [0, 25, 50, 100, 10 ** 9]
    labels = ["<25", "25-49", "50-99", ">=100"]
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        grp = [r for r in rows if lo <= r["n_train"] < hi]
        if not grp:
            continue
        out["bins"][lab] = {
            "n_users": len(grp),
            "n_queries": sum(r["n_q"] for r in grp),
            "median_n_train": statistics.median(r["n_train"] for r in grp),
            "mean_device_rag": sum(r["device_rag"] for r in grp) / len(grp),
            "sd_device_rag": statistics.stdev(r["device_rag"] for r in grp) if len(grp) > 1 else 0.0,
            "mean_device_cluster": sum(r["device_cluster"] for r in grp) / len(grp),
            "mean_rag": sum(r["rag"] for r in grp) / len(grp),
            "mean_device": sum(r["device"] for r in grp) / len(grp),
            "mean_train_s": (sum(r["train_s"] for r in grp if r["train_s"]) /
                             max(1, sum(1 for r in grp if r["train_s"]))),
        }

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    with open(args.out_tsv, "w") as fh:
        cols = ["user_id", "rank", "n_train", "n_q", "tok_train", "train_s", "train_steps",
                "rag", "cluster", "device", "device_rag", "device_cluster"]
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join("" if r[c] is None else
                               (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))
                               for c in cols) + "\n")

    print(f"\n{'bin':>8} {'users':>6} {'queries':>8} {'med n':>6} {'rag':>7} {'device':>7} "
          f"{'dev-rag':>9} {'sd':>7} {'dev-clu':>9} {'train s':>9}")
    for lab, b in out["bins"].items():
        print(f"{lab:>8} {b['n_users']:>6} {b['n_queries']:>8} {b['median_n_train']:>6.0f} "
              f"{b['mean_rag']:>7.3f} {b['mean_device']:>7.3f} {b['mean_device_rag']:>+9.4f} "
              f"{b['sd_device_rag']:>7.4f} {b['mean_device_cluster']:>+9.4f} {b['mean_train_s']:>9.0f}")
    for t, c in out["correlations"].items():
        print(f"\n{t}: mean {c['mean']:+.4f} sd {c['sd']:.4f}")
        print(f"  pearson r(n_train)  = {c['pearson_r']:+.3f}  p={c['pearson_p']:.3f}")
        print(f"  pearson r(log10 n)  = {c['pearson_r_log10n']:+.3f}  p={c['pearson_p_log10n']:.3f}")
        print(f"  spearman rho        = {c['spearman_rho']:+.3f}  p={c['spearman_p']:.3f}")
        print(f"  OLS slope           = {c['slope_per_record']:+.3e} /record  p={c['slope_p']:.3f}")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        ax.axhline(0, color="0.6", lw=0.8)
        ax.scatter(x, [r["device_rag"] for r in rows], s=[max(6, r["n_q"] / 3) for r in rows],
                   alpha=0.45, color="#2b6cb0", edgecolors="none", label="user")
        bx, by, be = [], [], []
        for lab, b in out["bins"].items():
            bx.append(b["median_n_train"]); by.append(b["mean_device_rag"])
            be.append(b["sd_device_rag"] / max(1, b["n_users"]) ** 0.5)
        ax.errorbar(bx, by, yerr=be, color="#c05621", marker="s", ms=4, lw=1.4,
                    capsize=3, label="binned mean $\\pm$ s.e.")
        ax.set_xscale("log")
        ax.set_xlabel("history records per user")
        ax.set_ylabel("accuracy, device $-$ RAG")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_fig)
        print(f"\nwrote {args.out_fig}")
    except Exception as exc:
        print(f"figure skipped: {exc}")
    print(f"wrote {args.out_json}\nwrote {args.out_tsv}")


if __name__ == "__main__":
    main()
