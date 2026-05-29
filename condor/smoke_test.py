#!/usr/bin/env python3
"""
Smoke test — verifies the Docker environment is correctly set up on the cluster.
Imports all key packages, prints versions, and confirms GPU visibility.
No training happens here.
"""

import sys

print("=" * 50)
print("Python version:", sys.version)
print("=" * 50)

# --- PyTorch + CUDA ----------------------------------------------------------
import torch
print("torch version:        ", torch.__version__)
print("CUDA available:       ", torch.cuda.is_available())
print("CUDA version:         ", torch.version.cuda)
print("GPU count:            ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:             ", torch.cuda.get_device_name(0))
    print("GPU memory (GB):      ", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

print("=" * 50)

# --- HuggingFace stack -------------------------------------------------------
import transformers
print("transformers version: ", transformers.__version__)

import peft
print("peft version:         ", peft.__version__)

import datasets
print("datasets version:     ", datasets.__version__)

import accelerate
print("accelerate version:   ", accelerate.__version__)

print("=" * 50)

# --- Experiment tracking -----------------------------------------------------
import wandb
print("wandb version:        ", wandb.__version__)

print("=" * 50)

# --- Evaluation --------------------------------------------------------------
import rouge_score
print("rouge_score:           installed ok")

print("=" * 50)
print("All imports successful. Environment is ready.")

# --- Model load check --------------------------------------------------------
# Verify that the SmolLM3-3B weights were downloaded correctly and can be
# loaded into GPU memory. This is the most important check before training:
# if this fails, every training job will fail in exactly the same way.
#
# We do a real end-to-end load (tokenizer + full model weights) rather than
# just checking that the files exist on disk. A file can be present but
# corrupted; only loading it actually proves it's usable.

import os
from pathlib import Path

MODEL_DIR = os.environ.get(
    "MODEL_OUT_DIR",
    "/home/ange00008/projects/mobileFT_distill/data/models/SmolLM3-3B",
)

print(f"Model directory: {MODEL_DIR}")

# Quick file-existence pre-check so we get a clear error message rather than
# a cryptic HuggingFace exception if the download job hasn't run yet.
model_path = Path(MODEL_DIR)
if not model_path.exists():
    print(f"FAIL: model directory not found — run condor/download_model.sub first", file=sys.stderr)
    sys.exit(1)

shards = list(model_path.glob("*.safetensors"))
if not shards:
    print("FAIL: no .safetensors weight shards in model directory", file=sys.stderr)
    sys.exit(1)

print(f"Found {len(shards)} weight shard(s) on disk — attempting load...")

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer — this reads tokenizer.json and tokenizer_config.json.
# It's fast (no GPU needed) and proves the non-weight files are intact.
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("Tokenizer loaded OK")
print(f"  Vocab size:      {tokenizer.vocab_size}")
print(f"  Model max length: {tokenizer.model_max_length}")

# Load the full model in bf16 onto the GPU that Condor allocated.
# bf16 is what we'll use during training, so this confirms the GPU can handle
# it. Loading takes ~30s and uses ~6 GB of VRAM.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,  # same precision we use in training
    device_map=device,
)
print(f"Model loaded OK  (device: {device})")
param_count = sum(p.numel() for p in model.parameters())
print(f"  Parameter count: {param_count / 1e9:.2f}B")
if torch.cuda.is_available():
    vram_used_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM in use:     {vram_used_gb:.1f} GB")

# Run one forward pass with a tiny input to confirm the model actually
# executes. We don't check the output — any non-crashing run means the
# weights are consistent with the architecture in config.json.
test_input = tokenizer("Hello, world!", return_tensors="pt").to(device)
with torch.no_grad():
    # We only need logits; past_key_values and hidden states aren't required.
    output = model(**test_input)

print(f"  Forward pass OK  (output shape: {output.logits.shape})")
print("=" * 50)
print("Model check passed. Ready for training.")
