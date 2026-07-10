#!/usr/bin/env python3
"""Fuse the A1-lamp Task-LoRA into SmolLM3-3B (bf16) and save the merged model.

Faithful to R5's stacking: R5 loaded base + A1-lamp (via base_adapter, frozen)
then trained a fresh q+v User-LoRA on top. Pre-merging A1-lamp into the base
weights and later training a fresh q+v LoRA on-device is mathematically the
same starting point. Output feeds `mlx_lm convert -q 4`.
"""
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "HuggingFaceTB/SmolLM3-3B"
ADAPTER = "train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000"
OUT = Path("data/models/SmolLM3-3B-a1lamp-merged")

print(f"[fuse] loading base {BASE} (bf16, cpu) ...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)
tok = AutoTokenizer.from_pretrained(BASE)

print(f"[fuse] attaching adapter {ADAPTER} ...", flush=True)
peft_model = PeftModel.from_pretrained(base, ADAPTER)

print("[fuse] merge_and_unload ...", flush=True)
merged = peft_model.merge_and_unload()

OUT.mkdir(parents=True, exist_ok=True)
print(f"[fuse] saving merged model -> {OUT} ...", flush=True)
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("[fuse] DONE", flush=True)
