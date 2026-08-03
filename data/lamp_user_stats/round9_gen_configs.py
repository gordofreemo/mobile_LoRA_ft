#!/usr/bin/env python3
"""
Generate 100 per-user OPPU training configs from the Round-9 template.

Reads `train/config/user_lora_lamp4_oppu_r9_template.json` and substitutes
per-user placeholders for each fingerprint in `LaMP_4_top100_users.json`
(R6's exact K=100 pool, reused unchanged):

  <<CONDITION>>     -> user_lora_lamp4_<user>_r9_oppu
  <<DATASET_PATH>>  -> data/lamp_user_train_LaMP_4_<user>_bm25k4.jsonl  (R6's existing file, not rebuilt)
  <<OUTPUT_DIR>>    -> train/checkpoints/user_lora_lamp4_<user>_r9_seed0

Output: `train/config/user_lora_lamp4_oppu_<user>_r9.json` (100 files,
gitignored — deterministically derivable from the template + the top-100
JSON).

Delta over data/lamp_user_stats/round6_gen_configs.py: template swapped to
the R9 variant (base_adapter -> One-LoRA FT, train/checkpoints/a2_lamp_1ep_seed0/final)
and all generated condition/dataset_path/output_dir/filenames get an `_r9`
suffix so R6's 100 configs/adapters/results and PT2's `_ptl` ones are never
touched. dataset_path still points at R6's existing per-user JSONL — no
rebuild this round.

This is the LaMP-4 counterpart of data/lamp_user_stats/round8_gen_configs.py
(R8, LaMP-3 on the same base), and the base-adapter mirror of
data/lamp_user_stats/pt2_gen_configs.py (PT2, same task on Per-Task-LoRA).

Usage (CPU-only, <1s):
    python data/lamp_user_stats/round9_gen_configs.py
    python data/lamp_user_stats/round9_gen_configs.py --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
USER_STATS_DIR = PROJECT_ROOT / "data" / "lamp_user_stats"
CONFIG_DIR = PROJECT_ROOT / "train" / "config"
DEFAULT_TEMPLATE = CONFIG_DIR / "user_lora_lamp4_oppu_r9_template.json"
DEFAULT_TOP_USERS = USER_STATS_DIR / "LaMP_4_top100_users.json"


def per_user_substitutions(fp: str) -> dict:
    return {
        "<<CONDITION>>": f"user_lora_lamp4_{fp}_r9_oppu",
        "<<DATASET_PATH>>": f"data/lamp_user_train_LaMP_4_{fp}_bm25k4.jsonl",
        "<<OUTPUT_DIR>>": f"train/checkpoints/user_lora_lamp4_{fp}_r9_seed0",
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--top-users", type=Path, default=DEFAULT_TOP_USERS)
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing per-user configs (default: refuse)")
    args = parser.parse_args()

    if not args.template.exists():
        sys.exit(f"ERROR: missing template {args.template}.")
    if not args.top_users.exists():
        sys.exit(f"ERROR: missing top-users JSON {args.top_users}.")

    template_text = args.template.read_text()
    # Sanity: all three placeholders are in the template.
    for ph in ("<<CONDITION>>", "<<DATASET_PATH>>", "<<OUTPUT_DIR>>"):
        if ph not in template_text:
            sys.exit(f"ERROR: template missing placeholder {ph}.")
    if "a2_lamp_1ep_seed0/final" not in template_text:
        sys.exit("ERROR: template's base_adapter is not the One-LoRA FT checkpoint "
                 "(a2_lamp_1ep_seed0/final) — refusing to generate R9 configs "
                 "against a stale/wrong template.")

    top = json.loads(args.top_users.read_text())
    fps = [u["user_fingerprint"] for u in top["users"]]
    print(f"[gen] generating {len(fps)} per-user R9 configs from {args.template}",
          flush=True)

    # Refuse-to-overwrite check up front: collect existing files.
    out_paths = {
        fp: CONFIG_DIR / f"user_lora_lamp4_oppu_{fp}_r9.json"
        for fp in fps
    }
    existing = [p for p in out_paths.values() if p.exists()]
    if existing and not args.overwrite:
        print(f"ERROR: refusing to overwrite {len(existing)} existing configs "
              f"(first: {existing[0]}). Pass --overwrite.", file=sys.stderr)
        sys.exit(1)

    written = 0
    for fp in fps:
        cfg_text = template_text
        for k, v in per_user_substitutions(fp).items():
            cfg_text = cfg_text.replace(k, v)
        # Validate JSON parses (catches bad placeholder substitution).
        try:
            cfg = json.loads(cfg_text)
        except json.JSONDecodeError as e:
            sys.exit(f"ERROR: substituted config for {fp} not valid JSON: {e}")
        # Spot-asserts to catch silent bugs early.
        assert cfg["condition"] == f"user_lora_lamp4_{fp}_r9_oppu", cfg["condition"]
        assert cfg["dataset_path"].endswith(f"{fp}_bm25k4.jsonl"), cfg["dataset_path"]
        assert cfg["output_dir"].endswith(f"{fp}_r9_seed0"), cfg["output_dir"]
        assert cfg["base_adapter"].endswith("a2_lamp_1ep_seed0/final"), cfg["base_adapter"]
        # R9 is a COLD zero-init User-LoRA on a merged base, like every R/PT
        # round — not the warm-start round's `continue` mode. train.py already
        # defaults to "merge"; assert it so a stray edit can't silently turn
        # this into a warm-start run whose r=8 q+v block would then be
        # shape-checked against One-LoRA FT's r=4 7-module adapter.
        assert cfg.get("base_adapter_mode", "merge") == "merge", cfg.get("base_adapter_mode")
        assert cfg["lora"]["r"] == 8, cfg["lora"]
        assert cfg["lora"]["target_modules"] == ["q_proj", "v_proj"], cfg["lora"]
        assert cfg["data"]["max_seq_length"] == 1024, cfg["data"]
        assert cfg["trainer"]["learning_rate"] == 1e-5, cfg["trainer"]
        assert cfg["trainer"]["weight_decay"] == 0.01, cfg["trainer"]
        assert cfg["trainer"]["per_device_train_batch_size"] == 2, cfg["trainer"]
        assert cfg["trainer"]["gradient_accumulation_steps"] == 4, cfg["trainer"]
        # Confirm the dataset file this config points at actually exists —
        # R9 reuses R6's per-user JSONLs unchanged and never rebuilds them.
        dataset_path = PROJECT_ROOT / cfg["dataset_path"]
        if not dataset_path.exists():
            sys.exit(f"ERROR: {fp}'s dataset_path {dataset_path} does not exist "
                     f"(R9 reuses R6's per-user files read-only; nothing should "
                     f"be missing).")
        out_paths[fp].write_text(cfg_text)
        written += 1
    print(f"[gen] wrote {written} configs to {CONFIG_DIR}/user_lora_lamp4_oppu_<user>_r9.json",
          flush=True)


if __name__ == "__main__":
    main()
