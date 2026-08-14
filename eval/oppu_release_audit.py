#!/usr/bin/env python3
"""Step-0 audit of OPPU's released processed data (arXiv:2402.04401).

Login-host only (no GPU, no torch). Answers, per task:
  1. Structure: n test users, queries/user, profile entries/user, total #Q
     (vs the paper's Table 2), label-file consistency.
  2. Provenance: where their query ids and profile-entry ids land in LaMP's
     own time-split partitions (via our local data/lamp_time/*_outputs.json
     id sets + profile ids from the pool-independent question files are NOT
     needed — outputs files carry the question ids per partition).
  3. Temporal semantics: is the per-user profile sorted by date, and do
     queries postdate the profile (only checkable via partition membership)?
  4. Leakage A (within-user): does a test query's article text appear in the
     same user's PEFT-training profile? Exact normalized match + max token
     Jaccard (near-dup >= 0.60, the LL5-retraction method).
  5. Leakage B (task-corpus): do test users / test queries / test profile
     entries appear inside user_others.json (the task-LoRA training corpus)?
     Streamed with ijson (files up to 2.8 GB — never json.load'd).

Usage:
  python eval/oppu_release_audit.py --task product_rating [--overwrite]
  python eval/oppu_release_audit.py --all [--overwrite]

Results: results/oppu_release_audit_<task>.json (flat scalars).
"""

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
RELEASE_DIR = PROJECT_ROOT / "data" / "oppu_release" / "data"
LAMP_TIME_DIR = PROJECT_ROOT / "data" / "lamp_time"
RESULTS_DIR = PROJECT_ROOT / "results"

# their task_name -> (our lamp_time dir, paper Table 2 #Q, query-article marker,
#                     profile content field, profile output field or None)
TASKS = {
    "citation":        ("LaMP_1",        123,  None,                                                          "title",       "citation"),
    "movie_tagging":   ("LaMP_2_movies", 3302, "] description: ",                                             "description", "tag"),
    "news_categorize": ("LaMP_2_news",   6033, "] article: ",                                                 "text",        "category"),
    "product_rating":  ("LaMP_3",        112,  "without further explanation. review: ",                      "text",        "score"),
    "news_headline":   ("LaMP_4",        6275, "Generate a headline for the following article: ",             "text",        "title"),
    "scholarly_title": ("LaMP_5",        107,  "Generate a title for the following abstract of a paper: ",    "abstract",    "title"),
    "tweet_paraphrase": ("LaMP_7",       109,  "Paraphrase the following tweet without any explanation before or after it: ", "text", None),
}

CITATION_RE = re.compile(r'written the paper with the title "([^"]*)"')
WS_RE = re.compile(r"\s+")


def provenance():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return "unknown"
    return {
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "condor_cluster_id": os.environ.get("CONDOR_CLUSTER_ID"),
        "condor_proc_id": os.environ.get("CONDOR_PROC_ID"),
    }


def norm(s):
    return WS_RE.sub(" ", s.strip().lower())


def content_hash(s):
    return hashlib.md5(norm(s).encode()).hexdigest()


def extract_article(task, query_input):
    if task == "citation":
        m = CITATION_RE.search(query_input)
        return m.group(1) if m else query_input
    marker = TASKS[task][2]
    pos = query_input.find(marker)
    return query_input[pos + len(marker):] if pos >= 0 else query_input


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def partition_id_sets(lamp_dir):
    """Question-id sets per partition from the small *_outputs.json files."""
    sets = {}
    for part in ("train", "dev", "test"):
        p = LAMP_TIME_DIR / lamp_dir / f"{part}_outputs.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        sets[part] = {str(g["id"]) for g in d["golds"]}
    return sets


def date_key(d):
    return str(d) if d is not None else ""


def audit_task(task, skip_others=False):
    lamp_dir, paper_q, _, content_field, output_field = TASKS[task]
    tdir = RELEASE_DIR / task
    test_fn = tdir / ("user_more_100_history.json" if task == "tweet_paraphrase"
                      else "user_top_100_history.json")
    label_fn = next(iter(tdir.glob("*history_label.json")))

    with open(test_fn) as f:
        test = json.load(f)
    with open(label_fn) as f:
        label = json.load(f)
    label_ids = {str(g["id"]) for g in label["golds"]}

    r = {"task": task, "lamp_task": lamp_dir, "paper_table2_q": paper_q,
         "n_test_users": len(test)}

    # -- structure --------------------------------------------------------
    q_counts = [len(u["query"]) for u in test]
    p_counts = [len(u["profile"]) for u in test]
    r["total_queries"] = sum(q_counts)
    r["queries_per_user_min"] = min(q_counts)
    r["queries_per_user_max"] = max(q_counts)
    r["profile_per_user_min"] = min(p_counts)
    r["profile_per_user_max"] = max(p_counts)
    r["profile_entries_total"] = sum(p_counts)
    all_q_ids = [str(q["id"]) for u in test for q in u["query"]]
    r["query_ids_unique"] = len(set(all_q_ids))
    r["label_ids_total"] = len(label_ids)
    r["query_ids_missing_from_label"] = len(set(all_q_ids) - label_ids)
    r["duplicate_test_user_ids"] = len(test) - len({u["user_id"] for u in test})

    # -- gold consistency: query.gold vs label file -----------------------
    gold_by_id = {str(g["id"]): g["output"] for g in label["golds"]}
    mismatch = sum(1 for u in test for q in u["query"]
                   if gold_by_id.get(str(q["id"])) not in (None, q.get("gold")))
    r["gold_mismatch_vs_label"] = mismatch

    # -- partition provenance --------------------------------------------
    parts = partition_id_sets(lamp_dir)
    for part in ("train", "dev", "test"):
        ids = parts.get(part, set())
        r[f"queries_in_lamp_{part}"] = sum(1 for qid in set(all_q_ids) if qid in ids)
    r["queries_in_no_partition"] = (r["query_ids_unique"]
                                    - sum(r[f"queries_in_lamp_{p}"] for p in ("train", "dev", "test")))
    # profile-entry ids vs partitions (sampled cap to keep it cheap)
    prof_ids = {str(p["id"]) for u in test for p in u["profile"] if "id" in p}
    for part in ("train", "dev", "test"):
        ids = parts.get(part, set())
        r[f"profile_ids_in_lamp_{part}"] = sum(1 for pid in prof_ids if pid in ids)
    # queries that are ALSO a profile id anywhere (same id used twice)
    r["query_ids_also_profile_ids_same_release"] = len(set(all_q_ids) & prof_ids)

    # -- temporal ---------------------------------------------------------
    sorted_users = 0
    dated_users = 0
    for u in test:
        dates = [date_key(p.get("date")) for p in u["profile"] if p.get("date") is not None]
        if len(dates) >= 2:
            dated_users += 1
            if all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1)):
                sorted_users += 1
    r["users_with_dated_profiles"] = dated_users
    r["users_profile_date_sorted"] = sorted_users

    # -- leakage A: query article vs same user's profile ------------------
    exact_dup = 0
    near_dup = 0          # jaccard >= 0.60, excluding exacts
    max_j_sum = 0.0
    gold_in_profile_out = 0   # generation tasks: gold output equals a profile output
    per_q = 0
    worst = []
    for u in test:
        prof_norm = [norm(str(p.get(content_field, ""))) for p in u["profile"]]
        prof_sets = [set(s.split()) for s in prof_norm]
        prof_hash = {content_hash(str(p.get(content_field, ""))) for p in u["profile"]}
        out_hash = ({content_hash(str(p.get(output_field, ""))) for p in u["profile"]}
                    if output_field else set())
        for q in u["query"]:
            per_q += 1
            art = extract_article(task, q["input"])
            h = content_hash(art)
            toks = set(norm(art).split())
            mj = max((jaccard(toks, ps) for ps in prof_sets), default=0.0)
            max_j_sum += mj
            if h in prof_hash:
                exact_dup += 1
            elif mj >= 0.60:
                near_dup += 1
            if mj >= 0.60:
                worst.append((round(mj, 3), str(u["user_id"]), str(q["id"])))
            if output_field and len(norm(str(q.get("gold", "")))) > 20 \
                    and content_hash(str(q.get("gold", ""))) in out_hash:
                gold_in_profile_out += 1
    r["leakA_queries_checked"] = per_q
    r["leakA_exact_dup_in_own_profile"] = exact_dup
    r["leakA_near_dup_j060"] = near_dup
    r["leakA_mean_max_jaccard"] = round(max_j_sum / max(per_q, 1), 4)
    r["leakA_gold_output_in_profile_outputs"] = gold_in_profile_out
    worst.sort(reverse=True)
    r["leakA_worst_pairs"] = [f"{j}:{uid}:{qid}" for j, uid, qid in worst[:20]]

    # -- leakage B: test users/queries/content inside user_others ---------
    if not skip_others:
        import ijson  # only needed for the big files
        test_uid = {str(u["user_id"]) for u in test}
        test_qid = set(all_q_ids)
        test_prof_hash = {content_hash(str(p.get(content_field, "")))
                          for u in test for p in u["profile"]}
        n_train_users = 0
        n_train_queries = 0
        uid_overlap = 0
        qid_overlap = 0
        prof_hits = 0
        prof_total = 0
        suspicious_users = 0   # >50% of a train user's profile hashes hit test set
        with open(tdir / "user_others.json", "rb") as f:
            for u in ijson.items(f, "item"):
                n_train_users += 1
                n_train_queries += len(u.get("query", []))
                if str(u.get("user_id")) in test_uid:
                    uid_overlap += 1
                qid_overlap += sum(1 for q in u.get("query", [])
                                   if str(q.get("id")) in test_qid)
                hits = 0
                prof = u.get("profile", [])
                for p in prof:
                    prof_total += 1
                    if content_hash(str(p.get(content_field, ""))) in test_prof_hash:
                        hits += 1
                prof_hits += hits
                if prof and hits / len(prof) > 0.5:
                    suspicious_users += 1
        r["others_n_users"] = n_train_users
        r["others_total_queries"] = n_train_queries
        r["leakB_test_user_ids_in_others"] = uid_overlap
        r["leakB_test_query_ids_in_others_queries"] = qid_overlap
        r["leakB_profile_hash_hits"] = prof_hits
        r["leakB_others_profile_entries_total"] = prof_total
        r["leakB_suspicious_train_users_gt50pct"] = suspicious_users

    r.update(provenance())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-others", action="store_true",
                    help="skip the (slow) user_others.json streaming pass")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tasks = sorted(TASKS) if args.all else ([args.task] if args.task else None)
    if not tasks:
        ap.error("--task or --all required")

    print(f"[oppu_release_audit] tasks={tasks} skip_others={args.skip_others} "
          f"host={socket.gethostname()} {provenance()['git_commit']}")

    for t in tasks:
        out = RESULTS_DIR / f"oppu_release_audit_{t}.json"
        if out.exists() and not args.overwrite:
            print(f"REFUSING to overwrite {out} (pass --overwrite)")
            sys.exit(1)

    for t in tasks:
        print(f"=== {t} ...", flush=True)
        r = audit_task(t, skip_others=args.skip_others)
        out = RESULTS_DIR / f"oppu_release_audit_{t}.json"
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        keys = [k for k in r if k.startswith(("n_", "total", "queries", "profile_",
                                              "leakA", "leakB", "others", "users_",
                                              "gold_", "label_", "duplicate", "query_ids"))]
        for k in keys:
            print(f"  {k}: {r[k]}")
    print("done")


if __name__ == "__main__":
    main()
