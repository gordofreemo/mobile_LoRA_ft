#!/usr/bin/env python3
"""
Consolidate the Warm-Start round's per-user prediction files into three
condition-level JSONL files per task, and emit the round's telemetry.

Three arms per task:
  baseline    Per-Task-LoRA alone, no personalization. NOT re-run this round —
              reused byte-identical from disk. Two different shapes:
                flat tasks    one batch file, ..._bm25k4_seed0_topK100.*
                grouped tasks K per-user files, ..._bm25k4_seed0_user<fp>.*
  coldmatch   fresh zero-init r=4/7-module adapter on the MERGED Per-Task-LoRA
  warm        the Per-Task-LoRA itself, continued on the user's data

The two treatment arms land on DIFFERENT result stems because they are
evaluated differently — warm runs with `--base-adapter none` (its adapter
already carries the task delta), coldmatch runs stacked. Both stems are built
here the same way eval_lamp.py builds them (eval_lamp.py:671-707).

Eval shape is HARDCODED per task below, deliberately NOT read from the pool
JSON: LaMP_3's and LaMP_4's pool files predate the `eval_pattern` field, and
those are precisely the two tasks whose shapes differ from each other (LaMP-3
is flat at 1 test record/user; LaMP-4 is grouped at 1-25). Verified against
data/lamp_user_stats/<task>_user_records.json.

Telemetry (the round's manipulation check — see plan §Free telemetry and
Revision 5). Per user per arm, written to a sibling JSONL plus a per-task
summary JSON:

  n_records            records scored for that user
  n_parse_failures     raw generations the task's own parser could not read.
                       The drift canary that caught R7's format-fidelity
                       collapse (0.7%->21.7%) and LaMP-5's title capture. For
                       the free-generation tasks (rouge1) there is no parser,
                       so this counts EMPTY generations instead — recorded as
                       `parse_failure_kind` so the two are never conflated.
  n_changed_raw        predictions whose RAW text differs from the baseline
                       arm's. This is what R10-R14's "7.9% of predictions
                       changed" headline measured, so it is the continuity
                       number this round is trying to beat.
  n_changed_scored     predictions whose per-record SCORE differs from the
                       baseline's. Differs from the raw count on closed-vocab
                       tasks, where the model can rewrite its prose wrapper
                       while the parser extracts the identical label — a raw
                       change that is invisible to the metric.

Without n_changed_*, "the recipe moved the output a lot and it didn't help" and
"the recipe still barely moves the output" are indistinguishable — both look
like a null, which is exactly the ambiguity R10-R14 ran into.

Plan reference: experiments/2026-07-31-warm-start-user-lora-plan.md §Scaffolding 5.

Usage (CPU, sub-second):
    python eval/aggregate_user_predictions_warm.py --task LaMP_3
    python eval/aggregate_user_predictions_warm.py --all --overwrite
"""

import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"

ARMS = ("baseline", "coldmatch", "warm")

# task -> (tag, pool JSON, Per-Task-LoRA adapter tag, eval_pattern, metric)
TASKS = {
    "LaMP_1": ("lamp1", "LaMP_1_top100_users.json",
               "per_task_lamp1_1ep_seed0_final", "flat", "accuracy"),
    "LaMP_2_movies": ("lamp2movies", "LaMP_2_movies_top100_users.json",
                      "per_task_lamp2_movies_1ep_seed0_final", "flat", "accuracy"),
    "LaMP_2_news": ("lamp2news", "LaMP_2_news_top27_users.json",
                    "per_task_lamp2_news_1ep_seed0_final", "grouped", "accuracy"),
    "LaMP_3": ("lamp3", "LaMP_3_top100_users.json",
               "per_task_lamp3_1ep_seed0_final", "flat", "mae"),
    "LaMP_4": ("lamp4", "LaMP_4_top100_users.json",
               "per_task_lamp4_1ep_seed0_final", "grouped", "rouge1"),
    "LaMP_5": ("lamp5", "LaMP_5_top100_users.json",
               "per_task_lamp5_1ep_seed0_final", "flat", "rouge1"),
    "LaMP_7": ("lamp7", "LaMP_7_top100_users.json",
               "per_task_lamp7_1ep_seed0_final", "flat", "rouge1"),
}


# --- parsers / scorers, mirroring eval_lamp.py -------------------------------
# Duplicated rather than imported, per this project's standalone-script
# convention (a Condor sandbox transfers only the named executable). If you
# change these, change eval/paired_compare*.py too.
LAMP2_MOVIES_LABELS = [
    "action", "based on a book", "classic", "comedy", "dark comedy",
    "dystopia", "fantasy", "psychology", "romance", "sci-fi",
    "social commentary", "thought-provoking", "true story", "twist ending",
    "violence",
]
LAMP2_NEWS_LABELS = [
    "business", "crime", "culture & arts", "education", "entertainment",
    "food & drink", "healthy living", "parents", "politics", "religion",
    "science & technology", "sports", "style & beauty", "travel", "women",
]


def parse_bracket_choice(text, _labels):
    m = re.search(r"\[?\s*([12])\s*\]?", str(text))
    return f"[{m.group(1)}]" if m else None


def parse_closed_vocab_label(text, labels):
    def norm(s):
        return " ".join(str(s).lower().split())
    norm_map = {norm(u): u for u in labels}
    t = norm(text)
    if t in norm_map:
        return norm_map[t]
    matches = [u for u_norm, u in norm_map.items() if u_norm in t]
    return max(matches, key=len) if matches else None


def parse_rating(text):
    m = re.search(r"[1-5]", str(text))
    return m.group(0) if m else None


CLASSIFICATION_PARSERS = {
    "LaMP_1": (["[1]", "[2]"], parse_bracket_choice),
    "LaMP_2_movies": (LAMP2_MOVIES_LABELS, parse_closed_vocab_label),
    "LaMP_2_news": (LAMP2_NEWS_LABELS, parse_closed_vocab_label),
}


def make_scorer(task, metric):
    """(gold, pred) -> (score, parsed_ok). Matches how eval_lamp.py scores."""
    if metric == "rouge1":
        from rouge_score import rouge_scorer
        rs = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

        def _rouge(gold, pred):
            # No parser for free generation; an EMPTY generation is the
            # degenerate case worth counting.
            ok = bool(str(pred).strip())
            return rs.score(str(gold), str(pred))["rouge1"].fmeasure, ok
        return _rouge, "empty_generation"

    if metric == "mae":
        def _mae(gold, pred):
            p = parse_rating(pred)
            if p is None:
                # eval_lamp drops parse-fails from its aggregate, but a paired
                # test needs a score for every record; max distance is the
                # documented penalty (same convention as paired_compare.py).
                return 4.0, False
            return float(abs(int(p) - int(str(gold).strip()))), True
        return _mae, "unparseable_rating"

    entry = CLASSIFICATION_PARSERS.get(task)
    if entry is None:
        def _exact(gold, pred):
            return (1.0 if str(pred).strip() == str(gold).strip() else 0.0), True
        return _exact, "none"
    labels, parse_fn = entry

    def _cls(gold, pred):
        parsed = parse_fn(pred, labels)
        return (1.0 if parsed == str(gold).strip() else 0.0), parsed is not None
    return _cls, "unparseable_label"


def collect_provenance():
    def _git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=PROJECT_ROOT,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    porcelain = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "hostname": socket.gethostname(),
        "condor_cluster_id": os.environ.get("CONDOR_CLUSTER_ID") or None,
        "condor_proc_id": os.environ.get("CONDOR_PROC_ID") or None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if porcelain is None else bool(porcelain),
    }


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def run_task(task, split, bm25_k, overwrite):
    tag, pool_file, per_task_tag, eval_pattern, metric = TASKS[task]
    flat = eval_pattern == "flat"
    profile_tag = f"bm25k{bm25_k}"

    pool_path = USER_STATS_DIR / pool_file
    records_path = USER_STATS_DIR / f"{task}_user_records.json"
    for p in (pool_path, records_path):
        if not p.exists():
            sys.exit(f"ERROR: missing {p}.")
    pool = json.loads(pool_path.read_text())
    fps = [u["user_fingerprint"] for u in pool["users"]]
    records = json.loads(records_path.read_text())

    def path_for(arm, fp):
        if arm == "baseline":
            return RESULTS_DIR / (f"{task}_{split}_{per_task_tag}_{profile_tag}_"
                                  f"seed0_user{fp}.predictions.jsonl")
        user_tag = f"user_lora_{tag}_{fp}_{arm}_seed0_final"
        if arm == "warm":
            # evaluated with --base-adapter none, so the stem carries the user
            # adapter's tag ALONE
            stacked = user_tag
        else:
            stacked = f"{per_task_tag}_{user_tag}"
        return RESULTS_DIR / (f"{task}_{split}_{stacked}_{profile_tag}_"
                              f"seed0_user{fp}.predictions.jsonl")

    baseline_batch = RESULTS_DIR / (f"{task}_{split}_{per_task_tag}_{profile_tag}_"
                                    f"seed0_topK{pool['k']}.predictions.jsonl")

    outs = {a: RESULTS_DIR / f"{task}_{split}_warmround_{a}.predictions.jsonl"
            for a in ARMS}
    tele_jsonl = RESULTS_DIR / f"{task}_{split}_warmround_telemetry.jsonl"
    tele_json = RESULTS_DIR / f"{task}_{split}_warmround_telemetry.json"
    existing = [p for p in list(outs.values()) + [tele_jsonl, tele_json] if p.exists()]
    if existing and not overwrite:
        sys.exit(f"ERROR: refusing to overwrite {len(existing)} existing files "
                 f"(first: {existing[0].name}). Pass --overwrite.")

    # --- presence check before doing any work -------------------------------
    need = [(a, fp) for a in ("coldmatch", "warm") for fp in fps]
    if not flat:
        need += [("baseline", fp) for fp in fps]
    missing = [(a, fp) for a, fp in need if not path_for(a, fp).exists()]
    if missing:
        sys.exit(f"ERROR: {len(missing)} of {len(need)} per-user prediction files "
                 f"are missing (e.g. {path_for(*missing[0]).name}). Every "
                 f"(user, arm) cell must have completed before aggregating.")
    if flat and not baseline_batch.exists():
        sys.exit(f"ERROR: missing the reused baseline batch file "
                 f"{baseline_batch.name}.")

    score_fn, pf_kind = make_scorer(task, metric)

    # --- baseline first: everything else is measured against it -------------
    base_rows = {}   # fp -> [row]
    if flat:
        batch = read_jsonl(baseline_batch)
        if len(batch) != len(fps):
            sys.exit(f"ERROR: {baseline_batch.name} has {len(batch)} lines, "
                     f"expected {len(fps)} (one per pool user).")
        # the flat baseline is one shared file; map records back to users via
        # each user's pinned test_record_id
        id_to_fp = {str(u["test_record_id"]): u["user_fingerprint"]
                    for u in pool["users"]}
        for r in batch:
            fp = id_to_fp.get(str(r["id"]))
            if fp is None:
                sys.exit(f"ERROR: baseline batch has id {r['id']} matching no "
                         f"pool user's test_record_id.")
            base_rows.setdefault(fp, []).append(r)
    else:
        for fp in fps:
            base_rows[fp] = read_jsonl(path_for("baseline", fp))

    rows_out = {a: [] for a in ARMS}
    telemetry = []
    count_mismatch = []

    for fp in fps:
        expected = len(records[fp][split])
        base = {str(r["id"]): r for r in base_rows[fp]}
        if len(base) != expected:
            count_mismatch.append((fp, "baseline", expected, len(base)))
            continue

        per_arm = {"baseline": base}
        ok = True
        for arm in ("coldmatch", "warm"):
            rows = read_jsonl(path_for(arm, fp))
            if len(rows) != expected:
                count_mismatch.append((fp, arm, expected, len(rows)))
                ok = False
                break
            per_arm[arm] = {str(r["id"]): r for r in rows}
        if not ok:
            continue

        for arm in ARMS:
            n_pf = n_raw = n_scored = 0
            for rid, r in per_arm[arm].items():
                gold = r["gold"]
                b = base.get(rid)
                if b is None:
                    sys.exit(f"ERROR: {task} {arm} user {fp}: record id {rid} "
                             f"is absent from the baseline arm.")
                if str(b["gold"]) != str(gold):
                    sys.exit(f"ERROR: gold mismatch at id={rid} between "
                             f"baseline and {arm} — the arms are not over the "
                             f"same gold set.")
                s, parsed_ok = score_fn(gold, r["pred"])
                if not parsed_ok:
                    n_pf += 1
                if arm != "baseline":
                    if str(r["pred"]) != str(b["pred"]):
                        n_raw += 1
                    s_base, _ = score_fn(gold, b["pred"])
                    if s != s_base:
                        n_scored += 1
                rows_out[arm].append({
                    "id": rid, "pred": r["pred"], "gold": gold,
                    "user_fingerprint": fp,
                })
            telemetry.append({
                "task": task, "user_fingerprint": fp, "arm": arm,
                "n_records": expected,
                "n_parse_failures": n_pf,
                "parse_failure_kind": pf_kind,
                "n_changed_raw": None if arm == "baseline" else n_raw,
                "n_changed_scored": None if arm == "baseline" else n_scored,
            })

    if count_mismatch:
        sys.exit(
            f"ERROR: {len(count_mismatch)} (user, arm) cells have the wrong "
            f"record count (fp, arm, expected, got): {count_mismatch[:5]}. A "
            f"count of 0 usually means the eval job ran without "
            f"LAMP_DIR=data/lamp_time and matched nothing."
        )

    for arm in ARMS:
        outs[arm].write_text("".join(json.dumps(r) + "\n" for r in rows_out[arm]))
    tele_jsonl.write_text("".join(json.dumps(t) + "\n" for t in telemetry))

    def total(arm, field):
        return sum(t[field] or 0 for t in telemetry if t["arm"] == arm)

    n_rec = total("baseline", "n_records")
    summary = {
        "schema_version": 1,
        "task": task, "split": split, "metric": metric,
        "eval_pattern": eval_pattern,
        "k_users": len(fps),
        "n_records_total": n_rec,
        **{f"{a}_n_parse_failures": total(a, "n_parse_failures") for a in ARMS},
        **{f"{a}_n_changed_raw": total(a, "n_changed_raw")
           for a in ("coldmatch", "warm")},
        **{f"{a}_n_changed_scored": total(a, "n_changed_scored")
           for a in ("coldmatch", "warm")},
        **{f"{a}_changed_raw_rate": (total(a, "n_changed_raw") / n_rec) if n_rec else None
           for a in ("coldmatch", "warm")},
        "parse_failure_kind": pf_kind,
        "baseline_source": ("reused topK batch file" if flat
                            else "reused per-user files"),
        **collect_provenance(),
    }
    tele_json.write_text(json.dumps(summary, indent=2))

    print(f"[{task}] {len(fps)} users, {n_rec} records/arm", flush=True)
    for arm in ARMS:
        extra = ""
        if arm != "baseline":
            extra = (f" changed_raw={total(arm,'n_changed_raw')} "
                     f"({total(arm,'n_changed_raw')/n_rec:.1%}) "
                     f"changed_scored={total(arm,'n_changed_scored')}")
        print(f"    {arm:<10} parse_failures={total(arm,'n_parse_failures')}"
              f" ({pf_kind}){extra}", flush=True)
    print(f"    -> {outs['warm'].name} (+2 arms), {tele_json.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--bm25-k", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.all:
        names = sorted(TASKS)
    elif args.task:
        names = [args.task]
    else:
        sys.exit("ERROR: pass --task <name> or --all.")
    for name in names:
        run_task(name, args.split, args.bm25_k, args.overwrite)


if __name__ == "__main__":
    main()
