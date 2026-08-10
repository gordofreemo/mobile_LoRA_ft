#!/usr/bin/env python3
"""Convert an MLX LoRA adapter (device h12 harness or train_task_mlx.py Mac
control) to PEFT format so the unchanged cluster eval_lamp.py / eval_bfcl.py
can evaluate it over the bf16 base.

Input dir:  adapters.safetensors (+ adapter_meta.json sidecar)
Output dir: adapter_model.safetensors + adapter_config.json

Key/shape mapping (verified from both implementations' source, then asserted
against the actual tensor shapes here rather than assumed):

  MLX  model.layers.N.<mod>.lora_a  [in, r]   (y += scale * (x @ a) @ b)
  MLX  model.layers.N.<mod>.lora_b  [r, out]
  PEFT base_model.model.model.layers.N.<mod>.lora_A.weight  [r, in]
  PEFT base_model.model.model.layers.N.<mod>.lora_B.weight  [out, r]
  (PEFT: y += (alpha/r) * B @ A @ x  ->  A = a.T, B = b.T, alpha = scale*r)

Shapes disambiguate the orientation: with r=4, lora_a.shape[1] == r and
lora_b.shape[0] == r for every projection; both are asserted per tensor.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from safetensors import safe_open

PEFT_CONFIG_TEMPLATE = {
    "alora_invocation_tokens": None,
    "alpha_pattern": {},
    "arrow_config": None,
    "auto_mapping": None,
    "base_model_name_or_path": "/home/ange00008/projects/mobileFT_distill/data/models/SmolLM3-3B",
    "bias": "none",
    "corda_config": None,
    "ensure_weight_tying": False,
    "eva_config": None,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": 8,
    "lora_bias": False,
    "lora_dropout": 0.05,
    "lora_ga_config": None,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "peft_version": "0.19.1",
    "qalora_group_size": 16,
    "r": 4,
    "rank_pattern": {},
    "revision": None,
    "target_modules": ["o_proj", "k_proj", "down_proj", "gate_proj", "v_proj", "q_proj", "up_proj"],
    "target_parameters": None,
    "task_type": "CAUSAL_LM",
    "trainable_token_indices": None,
    "use_bdlora": None,
    "use_dora": False,
    "use_qalora": False,
    "use_rslora": False,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir", help="dir with adapters.safetensors (+ adapter_meta.json)")
    ap.add_argument("out_dir", help="PEFT output dir")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.adapter_dir)
    out_dir = Path(args.out_dir)
    src = in_dir / "adapters.safetensors"
    if not src.exists():
        print(f"missing {src}", file=sys.stderr)
        sys.exit(1)
    if (out_dir / "adapter_model.safetensors").exists() and not args.overwrite:
        print(f"REFUSING to overwrite {out_dir} (pass --overwrite)", file=sys.stderr)
        sys.exit(1)

    meta = {}
    meta_path = in_dir / "adapter_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    rank = int(meta.get("lora_rank", 4))
    scale = float(meta.get("lora_scale", 2.0))
    alpha = scale * rank

    tensors = {}
    with safe_open(src, framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    out = {}
    modules = set()
    for key, val in tensors.items():
        if key.endswith(".lora_a"):
            base, suffix = key[: -len(".lora_a")], "lora_A"
            assert val.shape[1] == rank, f"{key}: expected [*, {rank}], got {val.shape}"
        elif key.endswith(".lora_b"):
            base, suffix = key[: -len(".lora_b")], "lora_B"
            assert val.shape[0] == rank, f"{key}: expected [{rank}, *], got {val.shape}"
        else:
            print(f"unexpected non-LoRA key in adapter: {key}", file=sys.stderr)
            sys.exit(1)
        modules.add(base)
        peft_key = f"base_model.model.{base}.{suffix}.weight"
        out[peft_key] = np.ascontiguousarray(val.T.astype(np.float32))

    # Every module must have both halves.
    for m in sorted(modules):
        for s in ("lora_A", "lora_B"):
            assert f"base_model.model.{m}.{s}.weight" in out, f"{m} missing {s}"

    config = dict(PEFT_CONFIG_TEMPLATE)
    config["r"] = rank
    config["lora_alpha"] = alpha
    # Dropout is inference-inactive; record the truth (MLX arms train without it).
    config["lora_dropout"] = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if meta:
        (out_dir / "source_adapter_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    n_params = sum(v.size for v in out.values())
    print(f"[convert] {len(modules)} modules, {len(out)} tensors, {n_params} params "
          f"-> {out_dir} (r={rank}, alpha={alpha})")
    ex = sorted(modules)[0]
    a = out[f"base_model.model.{ex}.lora_A.weight"]
    b = out[f"base_model.model.{ex}.lora_B.weight"]
    print(f"[convert] e.g. {ex}: A{list(a.shape)} B{list(b.shape)}")


if __name__ == "__main__":
    main()
