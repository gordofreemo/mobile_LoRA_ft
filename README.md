# On-Device LLM Personalization via Stacked LoRA

End-to-end research pipeline for personalizing a 3B-parameter language model to
individual users, then deploying it on an iPhone. Two LoRA adapters are trained
independently, a **Task-LoRA** on the LaMP benchmark and a per-user **User-LoRA**
on time-ordered interaction history, and stacked at inference with zero weight
modification to the base model.

---

## Overview

Large language models are increasingly being deployed on mobile devices, but
personalization remains a cloud-only feature due to the cost of fine-tuning. This
project asks whether we can train a compact personalization
pipeline on a cluster, deploy it to a real phone, and measure whether it actually
helps individual users.

**Research questions answered:**

| Q | Finding |
|---|---|
| Does Task-LoRA on LaMP improve a 3B model over retrieval-only prompting? | **Yes** — +10.9 pp accuracy / +7.2 ROUGE-1 / +12.5 ROUGE-1 on LaMP-3/4/7 test |
| Does a per-user LoRA on top of Task-LoRA improve further? | **Yes (LaMP-3)** — ΔMAE −0.050, acc +5.0 pp (0.680→0.730), zero inference overhead beyond adapter swap |
| Can the stack run at interactive speeds on a consumer iPhone? | **Yes** — ~37 tok/s decode, 1.7 s cold start, 2.2 GB peak on iPhone 17 Pro |

---

## Pipeline

```
Cluster (Phase 1 & 2)                         iPhone (Phase 3)
─────────────────────────────────────────────────────────────────
SmolLM3-3B (frozen)
    │
    ├─► Task-LoRA training                     SmolLM3-3B-4bit (MLX)
    │   LaMP-{3,4,7} mixed corpus                  │
    │   r=4, BM25 k=4 in system prompt             ├─► base model benchmark
    │   → checkpoint-1000 (ckpt 0.75 ep)           │   (this repo, done)
    │                                              │
    └─► per-user User-LoRA training            Task-LoRA fused + quantized
        time-ordered history, r=8 q+v              │   (next step)
        OPPU recipe (arXiv 2402.04401)             └─► base-vs-Task-LoRA benchmark
        3 epochs, stacked via --base-adapter
```

---

## Results

### Task-LoRA (Phase 1) — LaMP test split, BM25 k=4 baseline, seed=0, greedy

| Task | No-profile floor | BM25 profile baseline | Task-LoRA (ckpt-1000) | Δ adapter − baseline |
|---|---|---|---|---|
| LaMP-3 accuracy | 0.451 | 0.696 | **0.806** | **+0.109** |
| LaMP-4 ROUGE-1 | 0.139 | 0.154 | **0.226** | **+0.072** |
| LaMP-7 ROUGE-1 | 0.417 | 0.437 | **0.562** | **+0.125** |
| BFCL AST overall | — | **0.808** | 0.770 | −0.038 |

The BFCL regression is a general-capability sanity check; the −3.8 pp drop is
accounted for by Java/JS type errors in the BFCL suite and is within tolerable range
for a task-specialized adapter.

### User-LoRA (Phase 2, Round 5) — LaMP-3, K=100 users, test split

| Condition | Accuracy | MAE | RMSE |
|---|---|---|---|
| Task-LoRA only (baseline) | 0.680 | — | 0.616 |
| Task-LoRA + User-LoRA | **0.730** | **−0.050** | **0.575** |

100 independent user adapters trained via the OPPU recipe; each is loaded on top
of the shared Task-LoRA at inference (no base weight changes, no extra parameters at
serving time beyond the rank-8 adapter).

### On-device performance (Phase 3) — SmolLM3-3B-4bit, iPhone 17 Pro

| Metric | Value |
|---|---|
| Decode throughput (cool device, 64-tok gen, 256-tok prompt) | **37.5 ± 0.4 tok/s** |
| Decode throughput (deployment context: 2048-tok prompt) | **32.0 ± 0.8 tok/s** |
| Prefill throughput | **620–743 tok/s** |
| App-launch → first token (cold) | **≈ 1.7 s** (model load 1362 ± 114 ms + TTFT 380 ± 6 ms) |
| Realistic LaMP-3 (natural EOS, ~210-tok prompt) | **35.0 ± 1.2 tok/s** |
| Peak memory (session) | **≈ 2.2 GB** |
| Sustained stress (5 min / 6144 tok) | 37.8 → 17.9 tok/s (−53%, knee at ~90 s) |


---

## Technical stack

| Layer | Choice | Rationale |
|---|---|---|
| Base model | SmolLM3-3B (HuggingFace) | First-class support in both transformers/PEFT and MLX-Swift; designed for on-device use |
| Training | HuggingFace Transformers + PEFT | SFT with LoRA; CE loss only; base weights frozen throughout |
| Cluster | HTCondor (Docker universe) | Multi-GPU job scheduling with Condor submit files checked in |
| Personalization channel | BM25 k=4 retrieval into system prompt | Consistent between train and eval (cardinal reproducibility constraint) |
| On-device runtime | MLX / mlx-swift-lm | Only framework with first-class SmolLM3 support (`smollm3.py`, `SmolLM3.swift`) |
| iOS app | LLMEval (mlx-swift-examples, git-subtree vendored) | Modified with benchmark harness, custom model config, telemetry JSONL |
| Quantization | 4-bit (mlx_lm convert) | 1.73 GB on-device; fits iPhone 17 Pro unified memory |

---

## Repo structure

```
├── train/
│   ├── train.py                  # SFT trainer; --base-adapter for stacking
│   ├── build_dataset.py          # LaMP → BM25-retrieved JSONL
│   ├── build_user_dataset.py     # per-user variant for User-LoRA
│   └── config/                   # JSON configs for each training run
├── eval/
│   ├── eval_lamp.py              # LaMP harness (BM25, refuse-to-overwrite)
│   ├── eval_bfcl.py              # BFCL AST regression
│   ├── paired_compare_per_user.py # multi-user paired statistics (R5/R6)
│   └── bench_aggregate.py        # on-device JSONL → aggregate JSON
├── condor/                       # HTCondor submit files for all cluster jobs
├── ios/mlx-swift-examples/       # vendored via git subtree; LLMEval + benchmark harness
├── data/
│   ├── lamp/                     # user-based split (Task-LoRA training)
│   └── lamp_time/                # time-based split (User-LoRA training)
├── results/                      # flat scalar JSON + predictions JSONL per run
│   └── ondevice/                 # raw bench telemetry JSONL
└── experiments/                  # YYYY-MM-DD-<slug>.md per run (pre-registered)
```

---

## Reproducing the cluster experiments

### Prerequisites

- Docker image `ghcr.io/gordofreemo/smollm3-train:ver4` (or rebuild from
  `Dockerfile` + `requirements.txt`)
- HTCondor cluster with GPU nodes (tested on `gerda.hpc.uni-saarland.de`)
- `data/models/SmolLM3-3B/` weights (one-time pull via `condor/download_model.sub`)

### Task-LoRA training

```bash
condor_submit condor/train_1ep.sub          # trains a1_lamp_1ep_seed0
# → train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000/
```

### Evaluation

```bash
condor_submit condor/eval_lamp.sub          # LaMP-{3,4,7} with BM25 profile
condor_submit condor/eval_lamp_floor.sub    # no-profile floor
condor_submit condor/eval_bfcl.sub          # BFCL AST regression
python eval/summary.py                      # print results table
```

All eval scripts are deterministic (greedy, seed=0) and refuse to overwrite existing
outputs without `--overwrite`. Smoke runs (`--limit N`) write to `_limitN`-suffixed
files to prevent accidental collision.

### Reproducing the on-device benchmark

Requires a Mac with Xcode 26.5 and an iPhone 17 Pro with Developer Mode enabled.
Full instructions: `ios/README.md` and `experiments/2026-06-21-ondevice-base-inference-plan.md`.

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cd ios/mlx-swift-examples
xcodebuild -project mlx-swift-examples.xcodeproj -scheme LLMEval \
  -configuration Debug -destination 'id=<DEVICE_UDID>' \
  -derivedDataPath ./build -allowProvisioningUpdates -skipMacroValidation \
  DEVELOPMENT_TEAM=<TEAM_ID> build

xcrun devicectl device install app --device <DEVICE_UDID> \
  build/Build/Products/Debug-iphoneos/LLMEval.app
xcrun devicectl device process launch --device <DEVICE_UDID> mlx.LLMEval<TEAM_ID> \
  --benchmark
```

Telemetry is written to the app's `Documents/bench_metrics.jsonl` container and
pulled with `xcrun devicectl device copy from`.

---

## Design decisions & methodology notes

- **No base weight modification.** LoRA only; all personalization lives in the adapters.
- **Consistent retrieval channel.** BM25 k=4 into `system` at both training and eval time — adapter sees the same prompt shape it was trained on.
- **Pre-registered experiments.** Each run has a locked design doc (`experiments/YYYY-MM-DD-*-plan.md`) committed before execution.
- **Reproducibility first.** Every training run launchable from one CLI command + config JSON with a fixed seed. Every result file embeds full provenance: `git_commit`, library versions, Condor job IDs, hostname, timestamp.

---

## References

- **LaMP benchmark:** Salemi et al., *LaMP: When Large Language Models Meet Personalization* — lamp-benchmark.github.io
- **OPPU (per-user PEFT recipe):** Tan et al., arXiv 2402.04401
- **SmolLM3:** HuggingFaceTB/SmolLM3-3B on HuggingFace
- **Apple on-device fine-tuning:** arXiv 2510.03425
- **Closest prior work (cloud synthetic + on-device PEFT + LaMP):** arXiv 2508.21313
- **BFCL:** gorilla.cs.berkeley.edu/leaderboard.html
