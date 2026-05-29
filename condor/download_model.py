#!/usr/bin/env python3
"""
Download SmolLM3-3B weights from HuggingFace Hub to a local directory.

Runs as a Condor job so the weights land on the cluster's persistent home
mount and are available to all subsequent training jobs without re-downloading.
No GPU needed — this is pure network + disk I/O.
"""

import os
import sys
from pathlib import Path

# The HuggingFace repo ID — this is the "organisation/model-name" slug that
# uniquely identifies the model on huggingface.co. snapshot_download uses this
# to find the right set of files to pull.
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

# Where to put the downloaded files on disk. We read from an environment
# variable first so the Condor submit file can override it without touching
# this script. If the variable isn't set, we fall back to a sensible default
# inside the project's data/ directory (which is gitignored — weights are too
# large to commit).
OUT_DIR = os.environ.get(
    "MODEL_OUT_DIR",
    "/home/ange00008/projects/mobileFT_distill/data/models/SmolLM3-3B",
)

print(f"Downloading {MODEL_ID}")
print(f"Destination: {OUT_DIR}")
print("=" * 60)

# Guard against being run in the wrong Docker image (one that doesn't have the
# HuggingFace stack installed). Failing loudly here is better than a confusing
# AttributeError later.
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not found — wrong Docker image?", file=sys.stderr)
    sys.exit(1)

# Create the output directory if it doesn't exist yet. parents=True means it
# will also create any missing intermediate directories (e.g. data/models/).
# exist_ok=True means it won't error if the directory already exists — useful
# if the job is re-run after a partial download.
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# snapshot_download fetches the entire model repository — every file HuggingFace
# hosts for this model — and writes it into local_dir. It downloads shards in
# parallel and verifies SHA256 checksums, so a partial or corrupted download
# will be detected automatically.
#
# ignore_patterns strips out weight formats we don't need:
#   *.msgpack   — JAX/Flax serialization format (we use PyTorch)
#   flax_model* — Flax checkpoint files
#   tf_model*   — TensorFlow SavedModel files
#   rust_model* — candle (Rust ML) format
# Skipping these saves ~2 GB and avoids downloading files we'll never load.
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=OUT_DIR,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)

print("=" * 60)

# --- Sanity checks -----------------------------------------------------------
# Verify the two things training absolutely needs:
#   1. config.json  — describes the model architecture (number of layers,
#                     attention heads, hidden size, etc.). Without this,
#                     from_pretrained() can't reconstruct the model object.
#   2. *.safetensors — the actual weight files. SafeTensors is a format
#                     designed to load fast and safely (no pickle, no arbitrary
#                     code execution). The model is split across multiple shards
#                     because a single 6 GB file would be awkward to handle.
config = Path(OUT_DIR) / "config.json"
shards = list(Path(OUT_DIR).glob("*.safetensors"))

if not config.exists():
    print("FAIL: config.json missing", file=sys.stderr)
    sys.exit(1)

if not shards:
    print("FAIL: no .safetensors weight shards found", file=sys.stderr)
    sys.exit(1)

# Sum the sizes of all shards and convert bytes → gigabytes. This lets us
# quickly confirm we got roughly the expected ~6 GB rather than a truncated
# download that passed the checksum check on cached metadata.
total_gb = sum(s.stat().st_size for s in shards) / 1e9
print(f"config.json:   present")
print(f"weight shards: {len(shards)} file(s), {total_gb:.1f} GB total")
print("Download complete. Model is ready for training jobs.")
