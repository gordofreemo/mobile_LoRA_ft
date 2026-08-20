#!/usr/bin/env python3
"""Package h13 on-device predictions into THEIR prediction format so the
unchanged OPPU evaluator can score them.

`eval/h13_score.py` is the fast in-loop layer and was validated against the
cluster's own files (it reproduces rag 0.4933 / oppu_r5 0.5697 / +0.0763 exactly).
The paper's numbers still come from their evaluator, which is what this feeds:

  results/oppu_rep/movie_tagging/task_k1<rag-tag>_preds.json    <- an h13 arm as "task"
  results/oppu_rep/movie_tagging/oppu_k1<adapter-tag>_u000-100_preds.json  <- as "oppu"

Then, on conduit:
  eval/oppu_rep_score.py --task movie_tagging --restrict-to-preds \
      --task-tag <rag-tag> --oppu-tag <adapter-tag> --out-tag <out-tag>

Any pair of h13 arms can be contrasted this way (device vs rag, device vs
cluster, ...), because their scorer only knows "task arm" and "oppu arm".
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAMP_ID = "LaMP_2M"


def collect(preds_root: Path, arm: str):
    out = {}
    for d in sorted(preds_root.iterdir()):
        f = d / f"{arm}.jsonl"
        if not (d.is_dir() and f.exists()):
            continue
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                out[str(r["id"])] = r["output"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(ROOT / "results/ondevice/h13_preds"))
    ap.add_argument("--out", default=str(ROOT / "results/oppu_rep_h13/movie_tagging"),
                    help="staging dir; rsync to conduit's results root")
    ap.add_argument("--baseline-arm", default="rag")
    ap.add_argument("--adapter-arm", default="device")
    ap.add_argument("--baseline-tag", default=None, help="default _h13<baseline-arm>")
    ap.add_argument("--adapter-tag", default=None, help="default _h13<adapter-arm>")
    args = ap.parse_args()

    btag = args.baseline_tag or f"_h13{args.baseline_arm}"
    atag = args.adapter_tag or f"_h13{args.adapter_arm}"
    preds_root = Path(args.preds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base = collect(preds_root, args.baseline_arm)
    adap = collect(preds_root, args.adapter_arm)
    common = sorted(set(base) & set(adap))
    if not common:
        raise SystemExit(f"no ids shared by arms {args.baseline_arm}/{args.adapter_arm}")

    model = "h13-ondevice-SmolLM3-3B-oppu-movie-4bit"
    for tag, preds, stem in ((btag, base, f"task_k1{btag}_preds.json"),
                             (atag, adap, f"oppu_k1{atag}_u000-100_preds.json")):
        payload = {"task": LAMP_ID,
                   "golds": [{"id": i, "output": preds[i]} for i in common],
                   "model": model}
        (out / stem).write_text(json.dumps(payload, indent=4))
        print(f"wrote {out / stem}  ({len(common)} predictions)")

    n_users = sum(1 for d in preds_root.iterdir()
                  if d.is_dir() and (d / f"{args.adapter_arm}.jsonl").exists())
    print(f"\n{n_users} users, {len(common)} paired queries")
    print("on conduit:\n  .venv/bin/python eval/oppu_rep_score.py --task movie_tagging "
          f"--restrict-to-preds --task-tag {btag} --oppu-tag {atag} "
          f"--out-tag _h13_{args.adapter_arm}_vs_{args.baseline_arm} "
          "--results-root results/oppu_rep_h13")


if __name__ == "__main__":
    main()
