#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter (cluster OPPU user adapter) to MLX format so the
on-device h13 eval harness can load it unfused over the 4-bit base.

Exact inverse of eval/convert_mlx_adapter_to_peft.py:
  PEFT base_model.model.<mod>.lora_A.weight  [r, in]   -> MLX <mod>.lora_a  [in, r]
  PEFT base_model.model.<mod>.lora_B.weight  [out, r]  -> MLX <mod>.lora_b  [r, out]
The runtime scale (alpha/r) is set by the harness, not baked into the weights,
so both sides must agree: MLX LoRALinear computes y + scale*(x@a)@b and PEFT
computes y + (alpha/r)*B@A@x — identical when scale == alpha/r.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def convert(in_dir: Path, out_dir: Path, overwrite=False, dtype=np.float16):
    src = in_dir / "adapter_model.safetensors"
    if not src.exists():
        sys.exit(f"missing {src}")
    dst = out_dir / "adapters.safetensors"
    if dst.exists() and not overwrite:
        sys.exit(f"REFUSING to overwrite {dst} (pass --overwrite)")

    cfg = json.loads((in_dir / "adapter_config.json").read_text())
    rank, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scale = alpha / rank

    out, modules = {}, set()
    with safe_open(src, framework="numpy") as f:
        for key in f.keys():
            val = f.get_tensor(key)
            if not key.startswith("base_model.model."):
                sys.exit(f"unexpected key prefix: {key}")
            body = key[len("base_model.model."):]
            if body.endswith(".lora_A.weight"):
                base, suffix = body[: -len(".lora_A.weight")], "lora_a"
                assert val.shape[0] == rank, f"{key}: expected [{rank}, *], got {val.shape}"
            elif body.endswith(".lora_B.weight"):
                base, suffix = body[: -len(".lora_B.weight")], "lora_b"
                assert val.shape[1] == rank, f"{key}: expected [*, {rank}], got {val.shape}"
            else:
                sys.exit(f"unexpected non-LoRA key: {key}")
            modules.add(base)
            out[f"{base}.{suffix}"] = np.ascontiguousarray(val.T.astype(dtype))

    for m in sorted(modules):
        for s in ("lora_a", "lora_b"):
            assert f"{m}.{s}" in out, f"{m} missing {s}"

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst))
    meta = {"source": str(in_dir), "lora_rank": rank, "lora_alpha": alpha,
            "lora_scale": scale, "n_modules": len(modules),
            "n_params": int(sum(v.size for v in out.values())),
            "dtype": np.dtype(dtype).name}
    (out_dir / "adapter_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta, sorted(modules)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    meta, modules = convert(Path(args.adapter_dir), Path(args.out_dir), args.overwrite)
    print(f"[convert] {meta['n_modules']} modules, {meta['n_params']} params, "
          f"r={meta['lora_rank']} alpha={meta['lora_alpha']} scale={meta['lora_scale']} "
          f"-> {args.out_dir}")
    print(f"[convert] e.g. {modules[0]}")


if __name__ == "__main__":
    main()
