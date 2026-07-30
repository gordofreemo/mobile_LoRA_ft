#!/usr/bin/env python3
"""
Generate per-user OPPU training configs for the LL4-LL6 User-LoRA round, and
print the Condor `queue ... from (...)` fragments needed to fill in
condor/{build_longlamp_user_dataset,train_longlamp_user_lora,eval_longlamp_user_lora}.sub.

Mirrors data/lamp_user_stats/round5_gen_configs.py (and round6/round8/pt2's
siblings), generalized across LL4/LL5/LL6's three tags (review/abstract/topic)
since they share one template shape (train/config/longlamp_user_lora_<tag>_oppu_template.json).

Reads data/longlamp_user_stats/<tag>_top100_users.json (written by
data/longlamp_user_stats.py) and, per user:
  <<CONDITION>>     -> longlamp_user_lora_<tag>_<user_tag>_oppu
  <<DATASET_PATH>>  -> data/longlamp_user_train_<tag>_<user_tag>_bm25k4.jsonl
  <<OUTPUT_DIR>>    -> train/checkpoints/longlamp_user_lora_<tag>_<user_tag>_seed0

`user_tag` is the filesystem-safe form of the raw user id (see
train/build_longlamp_user_dataset.py's safe_user_tag -- duplicated here,
byte-identical, so filenames match exactly what that script produces).

Per experiments/2026-07-27-longlamp-user-lora-ll4-ll6-plan.md, Execution
Step 4/5 -- this is the "generate configs" half; the per-user JSONL corpora
themselves come from train/build_longlamp_user_dataset.py (Step 4).

Usage (CPU-only, <1s):
    python data/longlamp_user_stats/gen_configs.py --tag review
    python data/longlamp_user_stats/gen_configs.py --tag review --overwrite
    python data/longlamp_user_stats/gen_configs.py --tag review --print-queue build > /tmp/build_queue.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
USER_STATS_DIR = PROJECT_ROOT / "data" / "longlamp_user_stats"
CONFIG_DIR = PROJECT_ROOT / "train" / "config"

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_user_tag(user_id: str) -> str:
    """Byte-identical to train/build_longlamp_user_dataset.py's safe_user_tag
    and eval/eval_longlamp.py's copy -- keep all three in sync."""
    tag = _UNSAFE.sub("_", user_id).strip("_")
    return tag or "user"


def per_user_substitutions(tag: str, user_tag: str) -> dict:
    return {
        "<<CONDITION>>": f"longlamp_user_lora_{tag}_{user_tag}_oppu",
        "<<DATASET_PATH>>": f"data/longlamp_user_train_{tag}_{user_tag}_bm25k4.jsonl",
        "<<OUTPUT_DIR>>": f"train/checkpoints/longlamp_user_lora_{tag}_{user_tag}_seed0",
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tag", required=True, choices=["review", "abstract", "topic"])
    parser.add_argument("--top-users", type=Path, default=None,
                        help="default: data/longlamp_user_stats/<tag>_top100_users.json")
    parser.add_argument("--template", type=Path, default=None,
                        help="default: train/config/longlamp_user_lora_<tag>_oppu_template.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing per-user configs (default: refuse)")
    parser.add_argument(
        "--print-queue", choices=["build", "train", "eval"], default=None,
        help="instead of writing configs, print the Condor queue-block lines "
        "for the given sub file (build_longlamp_user_dataset.sub / "
        "train_longlamp_user_lora.sub / eval_longlamp_user_lora.sub) to stdout.",
    )
    args = parser.parse_args()

    top_users_path = args.top_users or USER_STATS_DIR / f"{args.tag}_top100_users.json"
    if not top_users_path.exists():
        sys.exit(f"ERROR: missing {top_users_path}. Run data/longlamp_user_stats.py first.")
    top = json.loads(top_users_path.read_text())
    users = top["users"]

    # tag -> (_temporal task name, Task-LoRA checkpoint dir name) -- needed to
    # fill in the eval queue's `task` / `base_adapter` macros.
    TEMPORAL_TASK = {
        "review": "product_review_temporal",
        "abstract": "abstract_generation_temporal",
        "topic": "topic_writing_temporal",
    }
    TASK_LORA_CKPT = {
        "review": "longlamp_lora_review_1ep_seed0",
        "abstract": "longlamp_lora_abstract_1ep_seed0",
        "topic": "longlamp_lora_topic_1ep_seed0",
    }

    if args.print_queue:
        # IMPORTANT: Condor's `queue vars from (...)` splits each line on
        # BOTH commas AND whitespace by default (confirmed empirically
        # 2026-07-29/30 -- neither single- nor double-quoting a field
        # preserves embedded spaces the way you'd expect from shell
        # conventions; the only thing that reliably works is putting the
        # space-containing field LAST in the variable list, where Condor's
        # parser lets the final variable absorb the rest of the line
        # verbatim). Abstract's user_ids are author names with spaces (e.g.
        # "Lei Zhang"), so `user_id` must be the LAST field in every queue
        # row below, and any field derived from it (adapter_ckpt, via
        # safe_user_tag) must come BEFORE it, not after.
        for u in users:
            uid = u["user_id"]
            utag = safe_user_tag(uid)
            if args.print_queue == "build":
                print(f"  {args.tag}, {uid}")
            elif args.print_queue == "train":
                print(f"  train/config/longlamp_user_lora_{args.tag}_{utag}_oppu.json")
            elif args.print_queue == "eval":
                adapter_ckpt = f"longlamp_user_lora_{args.tag}_{utag}_seed0"
                print(f"  {TEMPORAL_TASK[args.tag]}, {TASK_LORA_CKPT[args.tag]}, "
                      f"{adapter_ckpt}, {uid}")
        return

    template_path = args.template or CONFIG_DIR / f"longlamp_user_lora_{args.tag}_oppu_template.json"
    if not template_path.exists():
        sys.exit(f"ERROR: missing template {template_path}.")
    template_text = template_path.read_text()
    for ph in ("<<CONDITION>>", "<<DATASET_PATH>>", "<<OUTPUT_DIR>>"):
        if ph not in template_text:
            sys.exit(f"ERROR: template missing placeholder {ph}.")

    print(f"[gen] generating {len(users)} per-user configs from {template_path}", flush=True)

    out_paths = {}
    for u in users:
        utag = safe_user_tag(u["user_id"])
        out_paths[utag] = CONFIG_DIR / f"longlamp_user_lora_{args.tag}_{utag}_oppu.json"

    existing = [p for p in out_paths.values() if p.exists()]
    if existing and not args.overwrite:
        sys.exit(f"ERROR: refusing to overwrite {len(existing)} existing configs "
                 f"(first: {existing[0]}). Pass --overwrite.")

    written = 0
    for u in users:
        utag = safe_user_tag(u["user_id"])
        cfg_text = template_text
        for k, v in per_user_substitutions(args.tag, utag).items():
            cfg_text = cfg_text.replace(k, v)
        try:
            cfg = json.loads(cfg_text)
        except json.JSONDecodeError as e:
            sys.exit(f"ERROR: substituted config for {utag} not valid JSON: {e}")
        assert cfg["condition"] == f"longlamp_user_lora_{args.tag}_{utag}_oppu", cfg["condition"]
        assert cfg["dataset_path"].endswith(f"{utag}_bm25k4.jsonl"), cfg["dataset_path"]
        assert cfg["output_dir"].endswith(f"{utag}_seed0"), cfg["output_dir"]
        assert cfg["lora"]["r"] == 8, cfg["lora"]
        assert cfg["lora"]["target_modules"] == ["q_proj", "v_proj"], cfg["lora"]
        assert cfg["trainer"]["learning_rate"] == 1e-5, cfg["trainer"]
        assert cfg["trainer"]["weight_decay"] == 0.01, cfg["trainer"]
        out_paths[utag].write_text(cfg_text)
        written += 1
    print(f"[gen] wrote {written} configs to {CONFIG_DIR}/longlamp_user_lora_{args.tag}_<user_tag>_oppu.json",
          flush=True)


if __name__ == "__main__":
    main()
