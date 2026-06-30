#!/usr/bin/env python3
"""
Download a Llama-3.1-Instruct model from HuggingFace Hub for the
Llama-family scale comparison (cluster-side, parallel to Phase 3).

Single-model executable. The Condor submit file (condor/download_llama.sub)
queues this twice — once for each comparator — by varying MODEL_DIR_NAME:
    Llama-3.1-8B-Instruct   (~16 GB safetensors)
    Llama-3.1-70B-Instruct  (~141 GB safetensors)

Files land in /scratch/chair_brinkmann/ange00008/models/<dir>. The repo's
data/models/<dir> is a symlink into that scratch path, so downstream
from_pretrained() calls don't need to know about the scratch indirection.

HuggingFace auth: huggingface_hub auto-discovers the token at
$HOME/.cache/huggingface/token, which the Condor jobs see via
+WantGPUHomeMounted = true. No env-var plumbing, no token in any
submit file or runlog.

Group ownership: the parent /scratch/chair_brinkmann/ange00008/models/
has setgid + group=chair_brinkmann, so files created within inherit the
quota group automatically — no chgrp pass needed.
"""

import os
import sys
from pathlib import Path

# Which Llama variant to pull, e.g. "Llama-3.1-8B-Instruct". Set by the
# submit file's queue table. Fail loud if unset rather than guessing.
MODEL_DIR_NAME = os.environ.get("MODEL_DIR_NAME")
if not MODEL_DIR_NAME:
    print("ERROR: MODEL_DIR_NAME env var not set (expected from submit file)",
          file=sys.stderr)
    sys.exit(1)

MODEL_ID = f"meta-llama/{MODEL_DIR_NAME}"
OUT_DIR = os.environ.get(
    "MODEL_OUT_DIR",
    f"/scratch/chair_brinkmann/ange00008/models/{MODEL_DIR_NAME}",
)

print(f"Downloading {MODEL_ID}")
print(f"Destination: {OUT_DIR}")
print("=" * 60)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not found — wrong Docker image?", file=sys.stderr)
    sys.exit(1)

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ignore_patterns drops formats we don't load:
#   *.msgpack / flax_* / tf_* / rust_*  — non-PyTorch weight formats
#   original/*                          — Meta's pickled .pth consolidated
#                                         weights, duplicate of the safetensors
#                                         shards (~16 GB for 8B, ~140 GB for 70B)
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=OUT_DIR,
    ignore_patterns=[
        "*.msgpack", "flax_model*", "tf_model*", "rust_model*",
        "original/*",
    ],
)

print("=" * 60)

# --- Sanity checks -----------------------------------------------------------
config = Path(OUT_DIR) / "config.json"
shards = list(Path(OUT_DIR).glob("*.safetensors"))

if not config.exists():
    print("FAIL: config.json missing", file=sys.stderr)
    sys.exit(1)

if not shards:
    print("FAIL: no .safetensors weight shards found", file=sys.stderr)
    sys.exit(1)

total_gb = sum(s.stat().st_size for s in shards) / 1e9
print(f"config.json:   present")
print(f"weight shards: {len(shards)} file(s), {total_gb:.1f} GB total")
print(f"Download complete. {MODEL_ID} is ready for eval jobs.")
