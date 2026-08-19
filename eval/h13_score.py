#!/usr/bin/env python3
"""h13 reporting kit — score the four on-device arms and emit the full table.

Input : results/ondevice/h13_preds/<user_id>/<arm>.jsonl  (pulled from the phone)
        arms: rag | cluster | mac | device
Output: results/ondevice/h13_scores.json  + a printed table

Scoring replicates third_party/OPPU/eval/evaluation.py for LaMP_2M exactly
(label-list index match after strip; anything not in the list is wrong). This
local scorer was validated against the cluster's own prediction files: it
reproduces the published rag=0.4933 / oppu_r5=0.5697 / delta=+0.0763 to four
decimals. The authoritative paper numbers still come from running their
unchanged evaluator via eval/oppu_rep_score.py on conduit; this is the fast
in-loop layer.

Everything is reported per prefix of the frozen queue, and nothing is pooled
across arms beyond the paired contrasts listed in the plan.
"""
import argparse, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["rag", "cluster", "mac", "device"]
CONTRASTS = [("cluster", "rag"), ("device", "rag"), ("mac", "rag"),
             ("device", "cluster"), ("device", "mac")]
LABELS = ["sci-fi", "based on a book", "comedy", "action", "twist ending", "dystopia",
          "dark comedy", "classic", "psychology", "fantasy", "romance",
          "thought-provoking", "social commentary", "violence", "true story"]


def score(pred, gold):
    def m(x):
        try:
            return LABELS.index(str(x).strip())
        except ValueError:
            return -1
    return float(m(pred) == m(gold) and m(gold) != -1)


def paired_stats(diffs):
    """Mirrors eval/oppu_rep_score.py's paired_stats (incl. its all-zero guard,
    which exists because the +1e-30 offset used to return p=0.0 for a perfectly
    null result -- see the cluster-infra traps)."""
    from scipy import stats as st
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    mean = sum(diffs) / n
    if all(d == 0 for d in diffs):
        return {"n": n, "mean_diff": 0.0, "t_p": None, "wilcoxon_p": None,
                "wins": 0, "ties": n, "losses": 0, "note": "all diffs zero"}
    t = st.ttest_1samp(diffs, 0.0)
    try:
        w_p = float(st.wilcoxon(diffs).pvalue)
    except ValueError:
        w_p = None
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    ci = 1.96 * sd / (n ** 0.5) if n > 1 else 0.0
    return {"n": n, "mean_diff": mean, "ci95_lo": mean - ci, "ci95_hi": mean + ci,
            "t_p": float(t.pvalue), "wilcoxon_p": w_p,
            "wins": sum(1 for d in diffs if d > 0),
            "ties": sum(1 for d in diffs if d == 0),
            "losses": sum(1 for d in diffs if d < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(ROOT / "results/ondevice/h13_preds"))
    ap.add_argument("--texts", default=str(ROOT / "data/oppu_movie/h13_movie_texts.json"))
    ap.add_argument("--queue", default=str(ROOT / "data/oppu_movie/h13_queue.json"))
    ap.add_argument("--out", default=str(ROOT / "results/ondevice/h13_scores.json"))
    ap.add_argument("--require-arms", default="rag,cluster,device",
                    help="a user counts as complete only with all of these")
    args = ap.parse_args()

    texts = {u["user_id"]: u for u in json.load(open(args.texts))}
    gold = {q["id"]: q["gold"] for u in texts.values() for q in u["queries"]}
    queue = [e["user_id"] for e in json.load(open(args.queue))["queue"]]
    need = args.require_arms.split(",")
    preds_root = Path(args.preds)

    # ---- load, keeping only users complete in every required arm ----
    per_user, incomplete = {}, []
    for uid in queue:
        d = preds_root / uid
        got = {}
        for arm in ARMS:
            f = d / f"{arm}.jsonl"
            if f.exists():
                got[arm] = {json.loads(l)["id"]: json.loads(l) for l in open(f) if l.strip()}
        want_ids = {q["id"] for q in texts[uid]["queries"]}
        ok = [a for a in need if a in got and set(got[a]) >= want_ids]
        if len(ok) == len(need):
            per_user[uid] = {a: v for a, v in got.items() if set(v) >= want_ids}
        elif got:
            incomplete.append((uid, sorted(got), len(want_ids)))

    if not per_user:
        raise SystemExit(f"no complete users under {preds_root} (arms required: {need})")
    users = [u for u in queue if u in per_user]
    present = [a for a in ARMS if all(a in per_user[u] for u in users)]
    print(f"complete users: {len(users)}/{len(queue)}   arms present in all: {present}")
    if incomplete:
        print(f"partial users (excluded): {len(incomplete)} e.g. {incomplete[:3]}")

    # ---- per-query scores ----
    q_scores = {a: {} for a in present}
    q_user = {}
    for uid in users:
        for q in texts[uid]["queries"]:
            q_user[q["id"]] = uid
            for a in present:
                q_scores[a][q["id"]] = score(per_user[uid][a][q["id"]]["output"], gold[q["id"]])
    qids = sorted(q_user)

    out = {"n_users": len(users), "n_queries": len(qids), "users": users,
           "arms_present": present, "headline": {}, "contrasts": {},
           "change_rates": {}, "invalid_rates": {}}

    for a in present:
        out["headline"][a] = sum(q_scores[a][i] for i in qids) / len(qids)
        inv = sum(1 for uid in users for q in texts[uid]["queries"]
                  if str(per_user[uid][a][q["id"]]["output"]).strip() not in LABELS)
        out["invalid_rates"][a] = inv / len(qids)

    for hi, lo in CONTRASTS:
        if hi not in present or lo not in present:
            continue
        qd = [q_scores[hi][i] - q_scores[lo][i] for i in qids]
        gd = []
        for uid in users:
            ids = [q["id"] for q in texts[uid]["queries"]]
            gd.append(sum(q_scores[hi][i] - q_scores[lo][i] for i in ids) / len(ids))
        out["contrasts"][f"{hi}-{lo}"] = {"query_level": paired_stats(qd),
                                          "grouped_per_user": paired_stats(gd)}
        ch = sum(1 for uid in users for q in texts[uid]["queries"]
                 if per_user[uid][hi][q["id"]]["output"].strip()
                 != per_user[uid][lo][q["id"]]["output"].strip())
        out["change_rates"][f"{hi}-{lo}"] = {"n_changed": ch, "rate": ch / len(qids)}

    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n=== h13 on-device eval plane: {len(users)} users, {len(qids)} queries ===")
    print(f"{'arm':10} {'accuracy':>10} {'invalid':>9}")
    for a in present:
        print(f"{a:10} {out['headline'][a]:>10.4f} {out['invalid_rates'][a]:>9.3f}")
    print(f"\n{'contrast':18} {'query mean':>11} {'t_p':>9} {'W/T/L':>14} "
          f"{'grouped':>9} {'g t_p':>9} {'changed':>8}")
    for k, v in out["contrasts"].items():
        q, g = v["query_level"], v["grouped_per_user"]
        wtl = f"{q.get('wins',0)}/{q.get('ties',0)}/{q.get('losses',0)}"
        tp = f"{q['t_p']:.3g}" if q.get("t_p") is not None else "n/a"
        gtp = f"{g['t_p']:.3g}" if g.get("t_p") is not None else "n/a"
        print(f"{k:18} {q.get('mean_diff',0):>+11.4f} {tp:>9} {wtl:>14} "
              f"{g.get('mean_diff',0):>+9.4f} {gtp:>9} "
              f"{out['change_rates'][k]['rate']:>8.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
