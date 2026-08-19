#!/usr/bin/env python3
"""Fuse the OPPU movie-tagging task-LoRA into SmolLM3-3B and save the merged model.

The OPPU protocol trains each User-LoRA on top of a MERGED task adapter
(run_oppu.py: PeftModel.from_pretrained(base, task_lora) -> merge_and_unload()).
The h13 device/Mac base must therefore be a 4-bit MLX quantisation of THAT merge,
not a1lamp and not the bare base.

Replicates their load path exactly (bf16 -> prepare_model_for_kbit_training ->
attach -> merge) and checks the result against the merged-base hash their cluster
run recorded, so the device base is provably the same starting point.
"""
import hashlib
from pathlib import Path

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "HuggingFaceTB/SmolLM3-3B"
ADAPTER = "train/checkpoints/oppu_rep_movie/task_lora_k1"
OUT = Path("data/models/SmolLM3-3B-oppu-movie-merged")
EXPECTED_HASH = "20cc0133e8d1ebf83c80b802e1edc240"   # results/oppu_rep/movie_tagging/oppu_k1_r5_u000-010_meta.json


def base_weights_hash(model, limit_tensors=4):
    """oppu_replication/wrapper_common.py, verbatim."""
    h, seen = hashlib.md5(), 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            continue
        if "q_proj" in name or "v_proj" in name:
            h.update(param.detach().float().cpu().numpy().tobytes()[:65536])
            seen += 1
            if seen >= limit_tensors:
                break
    return h.hexdigest()


print(f"[fuse] loading base {BASE} (bf16, cpu) ...", flush=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=True)
base.config.use_cache = False
print("[fuse] prepare_model_for_kbit_training (their path; casts to fp32) ...", flush=True)
base = prepare_model_for_kbit_training(base)

print(f"[fuse] attaching {ADAPTER} ...", flush=True)
merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()

got = base_weights_hash(merged)
print(f"[fuse] merged-base hash: {got}  expected: {EXPECTED_HASH}", flush=True)
if got != EXPECTED_HASH:
    print("[fuse] WARNING: hash mismatch vs the cluster run — investigate before trusting parity")
else:
    print("[fuse] hash MATCHES the cluster movie run's merged base", flush=True)

merged = merged.to(torch.bfloat16)
OUT.mkdir(parents=True, exist_ok=True)
print(f"[fuse] saving -> {OUT} ...", flush=True)
merged.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)

# transformers >=5 writes rope settings ONLY into a nested "rope_parameters"
# block. Both mlx-lm (python) and mlx-swift read the TOP-LEVEL `rope_theta` and
# silently fall back to 10000 when it is absent -- a 500x error for SmolLM3
# (5e6), which leaves the model coherent but measurably worse. Restore the flat
# keys so every downstream MLX conversion inherits them.
import json
cfg_path = OUT / "config.json"
cfg = json.load(open(cfg_path))
rp = cfg.get("rope_parameters") or {}
if cfg.get("rope_theta") is None and rp.get("rope_theta") is not None:
    cfg["rope_theta"] = rp["rope_theta"]
    cfg["rope_scaling"] = None if rp.get("rope_type", "default") == "default" else rp
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"[fuse] restored top-level rope_theta={cfg['rope_theta']}", flush=True)
print("[fuse] DONE", flush=True)
