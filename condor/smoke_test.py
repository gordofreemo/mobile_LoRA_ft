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
