#!/usr/bin/env python3
"""Audit an inference-first runtime for the quantized-backward asymmetry.

Reproduces every claim in experiments/2026-08-19-cross-runtime-quant-backward-audit.md
by re-running the greps against local checkouts, so the paper's cross-runtime table is
regenerable rather than transcribed.

Five questions, asked of each runtime:
  Q1 does a quantized matmul exist, and what orientation does it assume?
  Q2 does a non-transposed (backward-orientation) quantized path exist?
  Q3 is it reachable from a backward pass?
  Q4 is it covered by a test?
  Q5 what does training actually dispatch to?

Usage:
  python eval/runtime_quant_backward_audit.py --roots /path/to/checkouts \
      [--runtime llama.cpp] [--out results/runtime_quant_backward_audit.json]
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# probe := (runtime, question, label, path_glob, regex, expect)
# expect: "present" -> finding is that the pattern EXISTS; "absent" -> that it does NOT.
PROBES = [
    # ---------------- llama.cpp / ggml ----------------
    ("llama.cpp", "Q1", "mul_mat asserts a single orientation",
     "ggml/src/ggml.c", r"GGML_ASSERT\(!ggml_is_transposed\(a\)\)", "present"),
    ("llama.cpp", "Q2", "backward dX routed through OUT_PROD, not mul_mat",
     "ggml/src/ggml.c", r"ggml_out_prod\(ctx,\s*//.*\n\s*src0,", "present"),
    ("llama.cpp", "Q2", "Metal backend has no out_prod at all",
     "ggml/src/ggml-metal/**", r"out_prod", "absent"),
    ("llama.cpp", "Q2", "CUDA out_prod is F32-only",
     "ggml/src/ggml-cuda/ggml-cuda.cu",
     r"GGML_OP_OUT_PROD:\s*\n\s*return op->type == GGML_TYPE_F32", "present"),
    ("llama.cpp", "Q2", "Vulkan out_prod is F32-only",
     "ggml/src/ggml-vulkan/ggml-vulkan.cpp", r"pipeline_out_prod_f32", "present"),
    ("llama.cpp", "Q2", "CPU out_prod accepts quantized src0",
     "ggml/src/ggml-cpu/ggml-cpu.cpp", r"ggml_is_quantized\(src0->type\)", "present"),
    ("llama.cpp", "Q5", "CPU quantized out_prod dequantizes then scalar-AXPYs",
     "ggml/src/ggml-cpu/ops.cpp",
     r"dequantize_row_q\(s0, wdata, ne0\);\s*\n\s*ggml_vec_mad_f32", "present"),
    ("llama.cpp", "Q4", "quantized types are in the out_prod test sweep",
     "tests/test-backend-ops.cpp", r"new test_out_prod\(type_a, type_b", "present"),
    ("llama.cpp", "Q4", "unsupported ops are skipped, not failed",
     "tests/test-backend-ops.cpp", r'printf\("not supported', "present"),
    ("llama.cpp", "Q3", "finetune forces F32 KV cache for lack of OUT_PROD support",
     "examples/training/finetune.cpp", r"lack of f16 support for OUT_PROD", "present"),
    ("llama.cpp", "Q3", "training documented as FP32-models-only",
     "examples/training/README.md", r"for FP32 models", "present"),
    ("llama.cpp", "Q5", "AdamW aborts on any non-F32 parameter",
     "ggml/src/ggml-cpu/ops.cpp",
     r"void ggml_compute_forward_opt_step_adamw\([\s\S]{0,500}?GGML_ABORT\(\"fatal error\"\)", "present"),

    # ---------------- llama.cpp: Android GPU / NPU backends ----------------
    ("llama.cpp", "Q2", "[android] OpenCL/Adreno has ZERO training ops",
     "ggml/src/ggml-opencl/**",
     r"GGML_OP_(OUT_PROD|OPT_STEP_ADAMW|CROSS_ENTROPY_LOSS_BACK|SOFT_MAX_BACK|RMS_NORM_BACK)", "absent"),
    ("llama.cpp", "Q1", "[android] ...while OpenCL/Adreno has quantized matmul kernels for inference",
     "ggml/src/ggml-opencl/**", r"(mul_mat|gemm|gemv)_[a-z0-9_]*q[0-9]", "present"),
    ("llama.cpp", "Q2", "[android] Hexagon NPU has ZERO training ops",
     "ggml/src/ggml-hexagon/**",
     r"GGML_OP_(OUT_PROD|OPT_STEP_ADAMW|CROSS_ENTROPY_LOSS_BACK|SOFT_MAX_BACK|RMS_NORM_BACK)", "absent"),
    ("llama.cpp", "Q1", "[android] ...while Hexagon NPU does support MUL_MAT for inference",
     "ggml/src/ggml-hexagon/**", r"GGML_OP_MUL_MAT", "present"),
    ("llama.cpp", "Q2", "[android] WebGPU has ZERO training ops",
     "ggml/src/ggml-webgpu/**",
     r"GGML_OP_(OUT_PROD|OPT_STEP_ADAMW|CROSS_ENTROPY_LOSS_BACK)", "absent"),
    ("llama.cpp", "Q2", "[android] Vulkan DOES have OUT_PROD (better than Metal)",
     "ggml/src/ggml-vulkan/ggml-vulkan.cpp", r"GGML_OP_OUT_PROD", "present"),

    # ---------------- ExecuTorch: the Android-GPU 4-bit training stack (2026-07-21) ----------------
    ("executorch", "Q2", "[android] Vulkan HAS a fused 4-bit input-grad kernel",
     "backends/vulkan/runtime/graph/ops/glsl/q4gsw_backward.glsl",
     r"d_x\[M, K\] = d_out\[M, N\] @ dequant\(W\)\[N, K\]", "present"),
    ("executorch", "Q2", "[android] ...and it reads the SAME packed 4-bit weights as the forward",
     "backends/vulkan/runtime/graph/ops/glsl/q4gsw_backward.glsl",
     r"const int code = int\(\(uint\(w_int\) >> \(8 \* kl \+ nib_hi\)\) & 0xFu\)", "present"),
    ("executorch", "Q3", "[android] built explicitly for LoRA over a frozen 4-bit base",
     "backends/vulkan/custom_ops_lib.py",
     r"for on-device LoRA training through a frozen\n#\s*4-bit base", "present"),
    ("executorch", "Q3", "[android] registered as a delegatable Vulkan op",
     "backends/vulkan/op_registry.py",
     r"@update_features\(exir_ops\.edge\.et_vk\.linear_q4gsw_backward\.default\)", "present"),
    ("executorch", "Q5", "[android] but the training extension knows nothing about it",
     "extension/training/**", r"et_vk|q4gsw", "absent"),
    ("executorch", "Q5", "[android] and no pass rewrites a model into it (no e2e flow)",
     "backends/vulkan/_passes/**", r"q4gsw_backward|adamw_step|q4gsw_requant", "absent"),
    ("executorch", "Q2", "[apple] Apple backend has NO training kernels",
     "backends/apple/**", r"q4gsw_backward|linear_dW|adamw_step|q4gsw_requant", "absent"),
    ("executorch", "Q2", "[android] Qualcomm NPU backend has NO training kernels",
     "backends/qualcomm/**", r"q4gsw_backward|linear_dW|adamw_step|q4gsw_requant", "absent"),
    ("executorch", "Q2", "[android] XNNPACK (default Android CPU) has NO training kernels",
     "backends/xnnpack/**", r"q4gsw_backward|linear_dW|adamw_step|q4gsw_requant", "absent"),
    ("executorch", "Q4", "[android] the 4-bit backward IS covered by a real op test",
     "backends/vulkan/test/op_tests/quantized_linear_backward_test.cpp",
     r"linear_q4gsw_backward_reference_impl", "present"),

    # ---------------- MNN: Android GPU ----------------
    ("MNN", "Q2", "[android] OpenCL backend registers ZERO gradient ops",
     "source/backend/opencl/**", r"OpType_[A-Za-z]*Grad", "absent"),
    ("MNN", "Q2", "[android] Vulkan backend registers ZERO gradient ops",
     "source/backend/vulkan/**", r"OpType_[A-Za-z]*Grad", "absent"),
    ("MNN", "Q1", "[android] OpenCL fast tile path is itself orientation-gated",
     "source/backend/opencl/execution/buffer/MatMulBufExecution.cpp",
     r"canUseLargeTile = canUseTile && mTransposeA && !mTransposeB", "present"),

    # ---------------- LiteRT: Android delegates ----------------
    ("litert", "Q2", "[android] GPU delegate has no gradient op",
     "tflite/delegates/gpu/common/**", r"OperationType::[A-Z_]*GRAD", "absent"),
    ("litert", "Q2", "[android] NNAPI delegate has no gradient op",
     "tflite/delegates/nnapi/**", r"_grad|backward_", "absent"),
    ("litert", "Q2", "[android] Hexagon delegate has no gradient op",
     "tflite/delegates/hexagon/**", r"_grad|backward_", "absent"),

    # ---------------- ExecuTorch ----------------
    ("executorch", "Q1", "quantized mixed_linear assumes weight [N,K] w/ per-out-channel scales",
     "kernels/quantized/cpu/op_mixed_linear.cpp",
     r"tensors_have_same_size_at_dims\(in, 1, weight, 1\)", "present"),
    ("executorch", "Q2", "no backward/grad anywhere in the quantized op set",
     "kernels/quantized/quantized.yaml", r"backward|grad", "absent"),
    ("executorch", "Q3", "backward graph is captured AOT, before quantization",
     "extension/training/README.md", r"capture the backward graph ahead of time", "present"),
    ("executorch", "Q3", "reference LoRA training export applies no quantization",
     "examples/models/phi-3-mini-lora/export_model.py", r"quantize_|quantizer|int4|int8", "absent"),
    ("executorch", "Q5", "QAT is defined as float compute over fake-quantized values",
     "docs/source/concepts.md",
     r"all computations are still done with floating point numbers", "present"),

    # ---------------- MNN ----------------
    ("MNN", "Q1", "float MatMul carries transposeA/transposeB flags",
     "express/MathOp.cpp", r"VARP _MatMul\(VARP a, VARP b, bool tranposeA, bool tranposeB\)", "present"),
    ("MNN", "Q2", "MatMulGrad implements the full transpose-flag algebra",
     "tools/train/source/grad/MatMulGrad.cpp", r"if \(transA && transB\)", "present"),
    ("MNN", "Q2", "no Int8/quantized op has a registered gradient",
     "tools/train/source/grad/**", r"Int8|Quant", "absent"),
    ("MNN", "Q1", "the op with transpose flags is float-only",
     "source/backend/cpu/CPUMatMul.cpp", r"inputs\[0\]->host<float>\(\)", "present"),
    ("MNN", "Q5", "QAT rounds weights but calls the FLOAT conv",
     "tools/train/source/nn/NN.cpp",
     r"clamp\(_Round\(mWeight \* _Reciprocal\(weightScale\)\), mWeightClampValue\) \* weightScale", "present"),
    ("MNN", "Q3", "LLM LoRA support is load-for-inference, not training",
     "transformers/llm/engine/src/llm.cpp", r"Llm::create_lora", "present"),

    # ---------------- LiteRT / TFLite ----------------
    ("litert", "Q2", "entire gradient op set is one broadcast shape helper",
     "tflite/kernels/gradient/gradient_ops.cc", r"BroadcastGradientArgs", "present"),
    ("litert", "Q2", "no gemm/fully-connected backward kernel exists",
     "tflite/kernels/gradient/**", r"matmul|fully_connected|gemm", "absent"),
    ("litert", "Q1", "quantized FC scales are pinned to the output-channel axis",
     "tflite/kernels/fully_connected.cc", r"quantized_dimension, 0", "present"),

    # ---------------- ONNX Runtime ----------------
    ("onnxruntime", "Q3", "ships a real on-device training API",
     "orttraining/orttraining/training_api/module.h", r"struct Module \{", "present"),
    ("onnxruntime", "Q2", "HAS a 4-bit quantized matmul gradient (MatMulBnb4)",
     "orttraining/orttraining/core/graph/gradient_builder.cc",
     r"IMPLEMENT_GRADIENT_BUILDER\(GetMatmulBnb4Gradient\)", "present"),
    ("onnxruntime", "Q2", "...computed by FLIPPING the transpose flag, exactly MLX's pattern",
     "orttraining/orttraining/core/graph/gradient_builder.cc",
     r"transB_value = \(transB_value \+ 1\) % 2;\s*//\s*revert the transpose", "present"),
    ("onnxruntime", "Q3", "...and no gradient wrt the quantized weights",
     "orttraining/orttraining/core/graph/gradient_builder.cc",
     r"Gradient propagation to B is not supported yet", "present"),
    ("onnxruntime", "Q1", "inference orientation is the default (transB=1)",
     "onnxruntime/contrib_ops/cpu/quantization/matmul_bnb4.cc",
     r'GetAttrOrDefault\("transB", static_cast<int64_t>\(1\)\)', "present"),
    ("onnxruntime", "Q5", "but the kernel dequantizes the whole weight then calls dense SGEMM",
     "onnxruntime/contrib_ops/cpu/quantization/matmul_bnb4.cc",
     r"DequantizeBlockwiseBnb4<float>\([\s\S]{0,2500}?MlasGemmBatch", "present"),
    ("onnxruntime", "Q5", "...explicitly a placeholder",
     "onnxruntime/contrib_ops/cpu/quantization/matmul_bnb4.cc",
     r"// TODO: implement with native kernel", "present"),
    ("onnxruntime", "Q2", "the OPTIMIZED int4 LLM op (MatMulNBits) has NO gradient",
     "orttraining/orttraining/core/graph/gradient_builder.cc", r"MatMulNBits", "absent"),
    ("onnxruntime", "Q3", "training API has no mobile EP wiring (no NNAPI/CoreML/QNN/XNNPACK)",
     "orttraining/orttraining/training_api/module.cc", r"nnapi|coreml|qnn|xnnpack", "absent"),

    # ---------------- ncnn ----------------
    ("ncnn", "Q1", "rich int8 quantized inference (quantize/dequantize/requantize layers)",
     "src/layer/requantize.cpp", r"Requantize", "present"),
    ("ncnn", "Q2", "ZERO gradient/backward layers among ~110 layers",
     "src/layer/**", r"backward|_grad\b|Gradient", "absent"),

    # ---------------- MLC-LLM / TVM ----------------
    ("mlc-llm", "Q3", "never imports TVM's training module (inference/serving only)",
     "python/mlc_llm/**", r"relax\.training|relax import training", "absent"),
    ("tvm", "Q3", "TVM DOES have a relax training module",
     "python/tvm/relax/training/trainer.py", r"class Trainer", "present"),
    ("tvm", "Q2", "...but zero of its 49 registered gradients are quantized",
     "python/tvm/relax/**", r'register_gradient\("[a-z_.]*(quant|int4|int8)', "absent"),

    # ---------------- Core ML / coremltools ----------------
    ("coremltools", "Q3", "Core ML on-device training covers only TWO layer types",
     "coremltools/models/neural_network/builder.py",
     r'_SUPPORTED_UPDATABLE_LAYERS = \["innerProduct", "convolution"\]', "present"),
    ("coremltools", "Q5", "quantizing an updatable model is explicitly REFUSED (even FP16)",
     "coremltools/models/neural_network/quantization_utils.py",
     r'updatable models cannot get quantized to FP16', "present"),
    ("coremltools", "Q3", "updatable is legacy neuralnetwork-format only, absent from modern MIL",
     "coremltools/converters/mil/**", r"isUpdatable|make_updatable", "absent"),
    ("coremltools", "Q1", "...while modern 4-bit/palettization lives in the MIL path",
     "coremltools/optimize/coreml/_quantization_passes.py", r"palettize|affine_quantize", "present"),

    # ---------------- bitsandbytes (reference QLoRA) ----------------
    ("bnb", "Q1", "forward uses a fused 4-bit gemm kernel",
     "bitsandbytes/autograd/_functions.py", r"torch\.ops\.bitsandbytes\.gemm_4bit", "present"),
    ("bnb", "Q5", "backward dequantizes the weight and calls a dense matmul",
     "bitsandbytes/autograd/_functions.py",
     r"torch\.matmul\(grad_output, F\.dequantize_4bit\(B, ctx\.state\)", "present"),

    # ---------------- MLX (ours) ----------------
    ("mlx-upstream", "Q3", "the ONLY caller of transpose=false is the vjp",
     "mlx/primitives.cpp",
     r"QuantizedMatmul::vjp[\s\S]{0,1200}?!transpose_", "present"),
    ("mlx-upstream", "Q2", "a non-transposed NAX quantized kernel exists",
     "mlx/backend/metal/quantized.cpp", r"qmm_n_nax", "present"),
    ("mlx-upstream", "Q3", "but dispatch still requires transpose==true",
     "mlx/backend/metal/quantized.cpp", r"if \(has_nax_kernel && transpose &&", "present"),
    ("mlx-upstream", "Q4", "non-transposed qmm now has a dedicated test (PR #4051)",
     "python/tests/test_quantized.py", r"def test_qmm_non_transposed", "present"),
    ("mlx-upstream", "Q4", "...but it is skipped in CI",
     "python/tests/test_quantized.py",
     r'skipIf\("CI" in os\.environ[\s\S]{0,80}?def test_qmm_non_transposed', "present"),
]


def iter_files(root: Path, glob: str):
    if glob.endswith("/**"):
        base = root / glob[:-3]
        if base.is_dir():
            for p in base.rglob("*"):
                if p.is_file():
                    yield p
    else:
        p = root / glob
        if p.is_file():
            yield p


def run_probe(root: Path, glob: str, regex: str):
    pat = re.compile(regex, re.IGNORECASE | re.MULTILINE)
    hits = []
    scanned = 0
    for p in iter_files(root, glob):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        scanned += 1
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            hits.append({"file": str(p.relative_to(root)), "line": line})
            if len(hits) >= 5:
                break
        if len(hits) >= 5:
            break
    return hits, scanned


def git_meta(root: Path):
    try:
        out = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%H|%ad",
                              "--date=short"], capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            h, d = out.stdout.strip().split("|")
            return h, d
    except Exception:
        pass
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True,
                    help="directory holding the runtime checkouts (one subdir per runtime)")
    ap.add_argument("--runtime", action="append", default=None,
                    help="restrict to these runtimes (repeatable)")
    ap.add_argument("--out", default=None, help="write flat JSON result here")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    roots = Path(args.roots).expanduser().resolve()
    wanted = set(args.runtime) if args.runtime else None

    print(f"# runtime_quant_backward_audit | roots={roots} | "
          f"utc={datetime.now(timezone.utc).isoformat()} | host={os.uname().nodename}")

    results, missing = [], []
    n_ok = n_bad = 0
    for runtime in sorted({p[0] for p in PROBES}):
        if wanted and runtime not in wanted:
            continue
        root = roots / runtime
        if not root.is_dir():
            missing.append(runtime)
            print(f"\n## {runtime}: SKIPPED (no checkout at {root})")
            continue
        sha, date = git_meta(root)
        print(f"\n## {runtime} @ {sha[:9] if sha else '?'} ({date})")
        for rt, q, label, glob, regex, expect in PROBES:
            if rt != runtime:
                continue
            hits, scanned = run_probe(root, glob, regex)
            found = len(hits) > 0
            ok = (found and expect == "present") or (not found and expect == "absent")
            n_ok, n_bad = (n_ok + 1, n_bad) if ok else (n_ok, n_bad + 1)
            where = f"{hits[0]['file']}:{hits[0]['line']}" if hits else "-"
            print(f"  [{'OK ' if ok else 'DIFF'}] {q} {label}")
            print(f"         expect={expect:<7} found={str(found):<5} at={where} (files scanned={scanned})")
            results.append({
                "runtime": runtime, "commit": sha, "commit_date": date,
                "question": q, "label": label, "glob": glob, "regex": regex,
                "expect": expect, "found": found, "reproduced": ok,
                "first_hit_file": hits[0]["file"] if hits else None,
                "first_hit_line": hits[0]["line"] if hits else None,
                "n_hits": len(hits), "n_files_scanned": scanned,
            })

    print(f"\n# reproduced {n_ok}/{n_ok + n_bad} probes"
          + (f" | missing checkouts: {', '.join(missing)}" if missing else ""))

    if args.out:
        outp = Path(args.out)
        if outp.exists() and not args.overwrite:
            print(f"refusing to overwrite {outp} (use --overwrite)", file=sys.stderr)
            return 1
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(results, indent=2))
        print(f"wrote {outp} ({len(results)} probes)")
    return 0 if n_bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
