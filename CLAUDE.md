# Research Project — On-Device LLM Personalization via PEFT

## Project direction update (2026-06-21) — Phase 3 OPEN: on-device deployment

**New active phase. The "no on-device/mobile code" hard constraint is lifted
for this phase** (it remains the historical framing for Phases 1–2). Goal:
actually deploy SmolLM3 on the testbed iPhone and measure real training +
inference metrics. First milestone — **base SmolLM3-3B running inference
on-device — is DONE (2026-06-21).** What follows is the runbook so a future
session can rebuild, deploy, and read metrics off the phone without
re-deriving the environment.

### Runtime decision (verified against official sources only)

- **Runtime = MLX** (`mlx-swift` on device, `mlx-lm` on the Mac).
- Apple's "standard" on-device LLM framework is **Foundation Models**; the
  thing announced as "Common AI" / **Core AI** is a *provider inside it*
  (WWDC26 session 339 lists providers: System Language Model, Private Cloud
  Compute, Core AI = ship-your-own-local-model, MLX = HF mlx-community models,
  + Anthropic/Google packages).
- Apple's **adapter-training toolkit is Apple-model-only** (rank-32 LoRA bound
  to a specific OS's system model) — it **cannot** adapt SmolLM3, so it is not
  a path for our Task-LoRA/User-LoRA stack.
- **SmolLM3 is first-class in MLX**: `mlx_lm/models/smollm3.py` (Python) and
  `Libraries/MLXLLM/Models/SmolLM3.swift` (Swift, in `ml-explore/mlx-swift-lm`)
  both exist (confirmed by reading source, not blogs).
- MLX is also the only credible **on-device training** route (the later
  training-metrics goal): `mlx_lm.lora`, the `LoRATrainingExample` app, and the
  Apple paper **arXiv:2510.03425** (Song & Tang, memory-efficient backprop for
  on-device fine-tuning) all use MLX. llama.cpp/ExecuTorch rejected.
- **Model delivery = download from HF on-device** (project-lead decision). The
  app pulls **`mlx-community/SmolLM3-3B-4bit`** (public MLX 4-bit conversion of
  `HuggingFaceTB/SmolLM3-3B`; identical to a local conversion for the *base*
  model) into its sandbox on first generation. We publish our own HF repo only
  once we fuse the Task-LoRA.

### Host / device / signing facts

- Mac: Apple **M3, 16 GB**. **Full Xcode 26.5** at `/Applications/Xcode.app`,
  but active dev dir is CommandLineTools — **prefix every Xcode/devicectl
  command** with `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- Signing: **Apple Development: andrew.geyko@icloud.com**, team **`JGW9U9Y36Y`**
  (free personal team; bundle IDs auto-disambiguated via
  `DISAMBIGUATOR=${DEVELOPMENT_TEAM}` in `Configuration/Build.xcconfig`).
- Testbed: **iPhone 17 Pro** (`iPhone18,1`), iOS **26.5.1**, Developer Mode on.
  UDID **`00008150-000674C60A3B401C`**. List: `xcrun devicectl list devices`.

### Local MLX toolchain (Mac-side convert / sanity-check)

- venv **`.venv-mlx/`** (Python **3.11** — 3.14 too new for MLX wheels),
  `mlx-lm` (mlx 0.31.2). Gitignored.
- Convert + quantize (done; output gitignored under `data/models/*`):
  `.venv-mlx/bin/python -m mlx_lm convert --hf-path HuggingFaceTB/SmolLM3-3B --mlx-path data/models/SmolLM3-3B-mlx-4bit -q --q-bits 4`
- Sanity generate: `.venv-mlx/bin/python -m mlx_lm generate --model data/models/SmolLM3-3B-mlx-4bit --prompt "..." --max-tokens 60`
  (SmolLM3 has **thinking mode on by default** — emits `<think>…</think>`).

### iOS app: location, edits, build, deploy

- **`ios/mlx-swift-examples/`** is **vendored into this repo via `git subtree`**
  (from `ml-explore/mlx-swift-examples`, base `378f244`; was a gitignored clone until
  2026-06-21 — see `ios/README.md`). Only `build/` + Xcode user state are gitignored.
  LLM libs come from the SPM package `ml-explore/mlx-swift-lm`, not vendored.
- **Edited file:** `ios/mlx-swift-examples/Applications/LLMEval/ViewModels/LLMEvaluator.swift`
  - `modelConfiguration = ModelConfiguration(id: "mlx-community/SmolLM3-3B-4bit", …)`.
  - Added `appendBenchRecord(...)` → appends one flat-JSON line per generation
    to `Documents/bench_metrics.jsonl` (model, prompt_tokens, gen_tokens,
    ttft_ms, gen_tps, prompt_tps, gen_time_s, peak_mem_bytes, max_tokens,
    thinking, truncated, device_model, os_version, timestamp_utc) + a
    `hardwareModelIdentifier()` helper.
- **Build (device, signed)** — `-skipMacroValidation` is **required** (else
  fails on `MLXHuggingFaceMacros … must be enabled`):
  ```
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
  cd ios/mlx-swift-examples
  xcodebuild -project mlx-swift-examples.xcodeproj -scheme LLMEval \
    -configuration Debug -destination 'id=00008150-000674C60A3B401C' \
    -derivedDataPath ./build -allowProvisioningUpdates -skipMacroValidation \
    DEVELOPMENT_TEAM=JGW9U9Y36Y build
  ```
  Output app: `build/Build/Products/Debug-iphoneos/LLMEval.app`,
  bundle id **`mlx.LLMEvalJGW9U9Y36Y`**.
- **Install + launch:**
  ```
  xcrun devicectl device install app --device 00008150-000674C60A3B401C build/Build/Products/Debug-iphoneos/LLMEval.app
  xcrun devicectl device process launch --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y
  ```
  First generation downloads ~1.73 GB from HF (Wi-Fi, one-time; the app data
  container — and the model — **persists across reinstalls** of the same
  bundle id). The app UI requires a **human tap** to start generation.

### Reading metrics off the device (the "hook")

No live device console works here (macOS `log stream` has no `--device`,
`log collect --device` needs **root**, `idevicesyslog` not installed). Working
path = pull the JSONL from the app container (no sudo); accumulates one line
per run:
```
xcrun devicectl device copy from --device 00008150-000674C60A3B401C \
  --domain-type appDataContainer --domain-identifier mlx.LLMEvalJGW9U9Y36Y \
  --source Documents/bench_metrics.jsonl --destination /tmp/devpull/bench_metrics.jsonl
```

### Base-inference characterization benchmark — DONE 2026-06-21

**Design locked** (`experiments/2026-06-21-ondevice-base-inference-plan.md`),
**implemented, run, and written up** (`experiments/2026-06-21-ondevice-base-inference.md`).
The reusable Mac-driven one-command rig now exists and is reused verbatim for the
base-vs-Task-LoRA comparison next.

**Headline (steady-state, cool device, n=5/cell):** decode **~37 tok/s** at
deployment context sizes (prefill curve gen64: 38.8 tok/s @64-tok prompt →
32.0 @2048), **prefill 620–740 tok/s**, **cold app-launch→first-answer ≈ 1.7 s**
(model load 1362±114 ms + cold TTFT 380±6 ms), **realistic LaMP-3 (natural EOS)
35.0±1.2 tok/s**, **peak ≈ 2.2 GB** (at 2048-tok contexts). The prior n=1 anecdote
(37.5 tok/s, 1.82 GB) sits inside this distribution.

**Two findings that shape the next pass:**
1. **Sustained decode throttles −53%** (37.8 → 17.9 tok/s over 5 min / 6144 tokens,
   knee at ~90 s) and **`ProcessInfo.thermalState` stayed `nominal` throughout** —
   the coarse enum is a useless throttle proxy; trust per-segment tok/s.
2. **Clean steady-state long-decode curves are unobtainable while plugged** (device
   never returns to nominal between heavy cells, so the decode-length grid cells for
   gen≥1024 are thermally confounded). The pre-registered **unplugged-over-Wi-Fi
   follow-up** is the way to get a clean decode curve — deferred, not blocking.

**Harness (verified vs `mlx-swift-lm` source):** EOS suppression for forced length =
drive `TokenIterator` directly and ignore the stop-token set to `maxTokens` (the EOS
check is in MLXLMCommon's loop wrapper, not `TokenIterator.next()`); `model_load_ms`
brackets the `ModelContainer` load. New launch args: `--benchmark` (cold + prefill +
decode), `--benchmark-tail` (realistic + 5-min stress, separate launch so the long
cells don't overrun one `--console` session), `--benchmark-cold` (load+1 gen+exit, ×3
for cold variance). `app_build` baked in (`smollm3-ondevice-bench-h2`).

**Deliverables:**
- Swift harness: `ios/mlx-swift-examples/Applications/LLMEval/Benchmark/{BenchmarkSupport,LLMEvaluator+Benchmark}.swift`
  (+ edits to `LLMEvaluator.swift`, `ContentView.swift`, `project.pbxproj`). The whole
  `ios/mlx-swift-examples/` tree is now **vendored into this repo via `git subtree`**
  (upstream base `378f244`) — our harness lives as ordinary tracked files with full
  history; only `build/` + Xcode user state stay gitignored. See `ios/README.md`.
  **Closes the version-control loose end below.**
- Aggregator: `eval/bench_aggregate.py` (stdlib-only, descriptive; nominal=primary,
  cold + stress-decay segregated).
- Raw telemetry: `results/ondevice/bench_metrics_smollm3-4bit-base_2026-06-21.jsonl`
  (89 records). Aggregate: `results/ondevice_base_smollm3_4bit_2026-06-21.json`.

**`peak_mem_bytes` caveat for reuse:** MLX `peakMemory` is a process high-water mark
(monotonic within a session), so per-cell peaks are confounded by execution order —
the meaningful number is the session peak (~2.2 GB). Reset-per-run is the h3
improvement to make for the base-vs-LoRA comparison.

### Loose ends / next steps (Phase 3)

- **Version control — CLOSED 2026-06-21 (git subtree):** `ios/mlx-swift-examples/`
  is **vendored via `git subtree`** at upstream base `378f244`; our LLMEval benchmark
  harness is committed as normal tracked files (full history, one-clone reproducible).
  `.gitignore` keeps only `ios/mlx-swift-examples/build/` + Xcode user state ignored.
  `.venv-mlx/` remains gitignored (rebuildable from the documented `mlx-lm` install).
  **Workflow** (see `ios/README.md`): edit harness files → ordinary `git commit` (no
  patch dance). Bump upstream:
  `git subtree pull --prefix=ios/mlx-swift-examples https://github.com/ml-explore/mlx-swift-examples <tag> --squash`.
  Bump `BenchConstants.appBuild` whenever harness logic changes (provenance in each record).
- **Task-LoRA on-device:** fuse `a1_lamp_1ep_seed0/checkpoint-1000` into
  SmolLM3, convert to MLX, publish an HF repo, swap the `modelConfiguration`
  id; measure base vs. Task-LoRA. (Adapter must be pulled from the cluster
  first — it is **not** on this Mac.) The benchmark rig is built to be reused
  verbatim for this comparison.
- **Characterize:** sweep prompt/generation length for a tok/s-vs-context
  curve; watch thermal throttling on sustained runs. **← this is exactly the
  active next task, now fully designed in the plan file.**
- **On-device training metrics:** via `LoRATrainingExample` / `mlx_lm.lora`
  (the reason MLX was chosen over llama.cpp).

## Project direction update (2026-06-19) — Phase 2 complete

**Phase 2 / User-LoRA is COMPLETE. Q4 is ANSWERED YES.** Round 5 (LaMP-3 multi-user, K=100, OPPU recipe) confirms that an additional per-user LoRA on time-ordered user history meaningfully personalizes beyond the Task-LoRA + BM25 baseline. Headline result: mean ΔMAE −0.050 (at the pre-registered MDE), accuracy 0.680 → 0.730 (+5 net flips), C3 head-to-head wins 7 vs C2 wins 2, ties 91, RMSE 0.616 → 0.575, all with zero overhead at inference (MPT, MGT, latency identical between C2 and C3). Effect size lies in OPPU's published range (−0.071 on LaMP-3). Full numbers + provenance in `experiments/2026-06-18-user-lora-lamp3-round5-multi.md`; the gate output JSON at `results/paired_compare_c2_a1lamp_bm25_vs_c3_a1lamp_userlora_bm25_round5_LaMP_3_test.json`.

**No further User-LoRA rounds planned.** R6+ candidate axes (K=200, recipe ablation, dev/test asymmetry diagnostic) are deferred indefinitely.

**Next milestone: TBD.** A fresh session should treat the User-LoRA work as closed and ask the project lead for the next research thrust before proposing follow-up experiments.

Frozen artifacts that survive Phase 2:
- A1-lamp Task-LoRA: `train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000/`
- 100 per-user User-LoRA adapters: `train/checkpoints/user_lora_lamp3_<fp>_seed0/final/` (one per user in `data/lamp_user_stats/LaMP_3_top100_users.json`)
- 4 single-user LaMP-4 User-LoRAs from R1 / R2-B / R4 (kept on disk for re-analysis)
- All 200 R5 eval predictions and the consolidated `results/LaMP_3_test_round5_C{2,3}.predictions.jsonl`

The 2026-06-02 update below is the prior direction; sections under it that describe how Phase 2 was scoped and tracked remain useful for understanding what was done, but the "active work" framing is superseded by this paragraph.

---

## Project direction update (2026-06-02)

The project's ML scope has been **narrowed**. The original plan (cluster-side synthetic data generation feeding A1-full and A2 ablations, with on-device User-LoRA deferred to a future phase) is no longer active. **Now:**

- **Synthetic preference-conditional data generation is DROPPED.** No teacher-model offline generation. Ablation conditions A1-full and A2 are dropped, along with research questions Q2 and Q3.
- **A1-lamp is DONE and FROZEN.** Canonical Task-LoRA: `train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000/` (the 1-epoch-sweep deliverable; see "Phase 1 implementation status" below).
- **Active ML work is now per-user User-LoRA on the time-based LaMP split**, sitting on top of the frozen A1-lamp ckpt-1000. This is what the original spec called "Phase 2 / on-device User-LoRA", pulled forward and run on the cluster (simulated per-user adaptation, not on-device hardware).
- Sections below describing synthetic data, A1-full, A2, and domain-specific corpora are kept for historical record but **marked DEFERRED**. They reflect the original plan, not the current direction.

Hard constraints from the original spec all still hold (no KD, no base-weight modification, no on-device code, reproducibility-first). Profile-leakage rule is now reinterpreted for time-based splits (no overlap *within* a user between their train-period and dev-period interactions — enforced by LaMP's split by construction).

---

## What this project is

Fine-tuning **SmolLM3-3B** for personalized instruction following using a two-stage LoRA pipeline, both trained on the cluster:

1. **Task-LoRA (DONE).** Trained on LaMP-{3,4,7}, BM25-retrieved profile pinned in the system prompt. The model learns "use a user-profile cue in the system slot to personalize". Canonical adapter: A1-lamp ckpt-1000.
2. **User-LoRA (DONE — Phase 2 closed 2026-06-19).** A separate small LoRA per user, fine-tuned on that user's early-period interactions (from LaMP's **time-based** split, not the user-based split used for the Task-LoRA training). Evaluated on the user's later-period interactions. Sits on top of the frozen Task-LoRA at inference. Validated the on-device personalization story on the cluster without requiring on-device hardware — see the 2026-06-19 direction update at the top for the Round-5 headline result.

---

## Research questions

| Q | Status |
|---|---|
| **Q1** — Does fine-tuning on LaMP improve over zero-shot at all for a 3B model? | **ANSWERED: YES** — see `experiments/2026-05-31-a1-lamp.md`, `experiments/2026-06-02-a1-lamp-1ep-pareto.md`, and `experiments/2026-06-13-lamp-test-split-correction.md`. A1-lamp ckpt-1000 gives +0.11 / +0.07 / +0.13 over the BM25 baseline on LaMP-3/4/7 (test; dev results within ±0.01). |
| **Q2** — Does adding synthetic preference-conditional data on top of LaMP improve further? | **DROPPED** with the 2026-06-02 pivot. |
| **Q3** — Does a general Task-LoRA generalize as well as domain-specific ones? | **DROPPED** with the 2026-06-02 pivot. |
| **Q4** *(new)* — Does an additional per-user LoRA on time-ordered user history meaningfully personalize beyond the Task-LoRA alone? | **ANSWERED YES** (2026-06-19, closing the User-LoRA phase). Round 5 (LaMP-3 multi-user, K=100, OPPU recipe) confirms the hypothesis: mean ΔMAE −0.050 (at MDE), accuracy 0.680 → 0.730, RMSE 0.616 → 0.575, all at zero inference-time overhead (MPT/MGT/latency identical). See `experiments/2026-06-18-user-lora-lamp3-round5-multi.md`. Historical record of the prior LaMP-4 single-user rounds (which did not produce a positive signal at n=25) follows: Rounds 1, 2-B, 3-α, and 4 all failed the pre-registered gate on test. Round 1 (bare train, BM25 eval): Δ=+0.003, p=0.93 (`experiments/2026-06-15-user-lora-lamp4-u00000011-round1.md`). Round 2-B (BM25 train, BM25 eval): Δ=−0.004, p=0.89 (`experiments/2026-06-16-user-lora-lamp4-u00000011-round2-B.md`) — empirically downweighted the train/eval shape-mismatch hypothesis. Round 3-α (eval-only no-profile redundancy test): Δ=+0.010, p=0.73 (`experiments/2026-06-16-user-lora-lamp4-u00000011-round3-alpha.md`) — refuted the BM25/User-LoRA redundancy hypothesis (BM25 contributed ~0.003 at inference, so wasn't masking anything). Round 4 (record-level framing — 241 train-period (input,gold) record pairs instead of 1100 profile entries, BM25-train + BM25-eval, single-axis change vs R2-B), executed 2026-06-17: Δ=−0.018, p=0.43 — widest negative point estimate of the four rounds; records framing inert vs R2-B (test Δ=−0.014 p=0.68; dev Δ=−0.003 p=0.90). dev/test split asymmetry now a robust 4-round pattern (dev Δ +0.030/+0.043/+0.047/+0.040 vs test Δ +0.003/−0.004/+0.010/−0.018). Matrix→R5 = **dev/test asymmetry diagnostic on u00000011** (the matrix's pre-committed axis); writeup: `experiments/2026-06-17-user-lora-lamp4-u00000011-round4.md`. **R5 axis explicitly overridden 2026-06-18** in `experiments/2026-06-18-user-lora-round5-lamp3-plan.md`: R5 is now LaMP-3 multi-user (K=100, OPPU recipe) on the grounds that OPPU literature comparison shows our LaMP-4 +0.010 lift is in their published range — the nulls are power-limited at n=25, not methodological. Asymmetry diagnostic parked at R6+ candidate set (returns to front if R5 nulls). |

---

## Ablation conditions

| Condition | Training data | Status |
|---|---|---|
| **Baseline** | None (zero-shot) | Done |
| **A1-lamp** | LaMP-{3,4,7} training splits (user-based), profile in system at train + inference | **Done, canonical adapter = `train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000/`** |
| ~~**A1-full**~~ | ~~LaMP + synthetic preference-conditional data~~ | **DROPPED 2026-06-02** |
| ~~**A2**~~ | ~~Domain-specific corpora + domain synthetic data~~ | **DROPPED 2026-06-02** |
| **U** *(closed 2026-06-19)* | Per-user fine-tune on LaMP time-split early-period interactions, stacked on top of A1-lamp ckpt-1000 | **DONE — Q4 confirmed by Round 5 (LaMP-3 multi-user, K=100, OPPU recipe).** Mean ΔMAE −0.050 (at MDE), accuracy 0.680 → 0.730, RMSE 0.616 → 0.575, zero inference overhead. 100 per-user adapters at `train/checkpoints/user_lora_lamp3_<fp>_seed0/final/`. Writeup `experiments/2026-06-18-user-lora-lamp3-round5-multi.md`. Historical record of the prior LaMP-4 single-user rounds (kept for re-analysis): Round 1 (bare train, BM25 eval): Δ=+0.003, p=0.93 (adapter `train/checkpoints/user_lora_lamp4_u00000011_seed0/final/`). Round 2-B (BM25 train, BM25 eval): Δ=−0.004, p=0.89 (adapter `train/checkpoints/user_lora_lamp4_u00000011_bm25k4_seed0/final/`). Round 3-α (eval-only, --no-profile system slot, reused both adapters): Δ=+0.010, p=0.73 — refuted the BM25/User-LoRA redundancy hypothesis. Round 4 (record-level framing — 241 (input,gold) record pairs vs 1100 profile entries — BM25-train + BM25-eval, single-axis change vs R2-B), executed 2026-06-17: Δ=−0.018, p=0.43; R2B-C3' → C3-R4 framing-axis test Δ=−0.014, p=0.68 (records framing inert vs R2-B). Adapter `train/checkpoints/user_lora_lamp4_u00000011_records_bm25k4_seed0/final/`. R5 axis was pre-committed by R4's matrix Row 2 to a dev/test asymmetry diagnostic on u00000011, then explicitly overridden 2026-06-18 to LaMP-3 multi-user (K=100 users, OPPU recipe r=8 q+v only LR=1e-5 L2=1e-2). The asymmetry diagnostic and other R6+ candidate axes (K=200, recipe ablation) are shelved as of 2026-06-19. |

---

## Model

- **Base:** `HuggingFaceTB/SmolLM3-3B`
- **Weights:** bf16, frozen throughout all training
- **Framework:** HuggingFace Transformers + PEFT

---

## Training method — SFT only, no KD

**Do not use knowledge distillation.** CE loss only:

```
Loss = CrossEntropy(Task-LoRA outputs, ground truth)
```

The teacher model (Qwen3-30B or GPT-4o) is used **only for offline synthetic data generation**, never loaded alongside the student during training.

**Do not** co-load teacher and student. **Do not** compute KL divergence loss. **Do not** modify base model weights.

---

## Task-LoRA config

```python
from peft import LoraConfig

task_lora = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Rank choice (r=4, alpha=8):** the original spec used r=64 / alpha=128. Dropped to
r=4 because the whole project's value proposition is on-device efficiency — adapter
parameter count scales linearly in r, and r=4 is 16× smaller than r=64 in both disk
and inference overhead. alpha is dropped proportionally (preserving the standard
alpha/r = 2 scaling); see [arXiv:2402.04401] OPPU and the LoRA literature for
typical r=4–16 mobile configurations.

- **A1-lamp:** one training run, LaMP splits only (no synthetic data)
- **A1-full:** one training run, LaMP splits + synthetic data mixed
- **A2:** three independent training runs (one per domain), same config each

---

## Training setup

```
Optimizer:    AdamW, lr=3e-4, cosine decay, warmup 3%
Batch:        per-device bs=4, gradient accumulation=8 (effective bs=32)
Epochs:       2–3
Checkpoint:   every 500 steps
Logging:      metrics.jsonl streamed per logging_steps + train_meta.json
              summary scalars at end. W&B is wired up in train.py but
              defaults off (no account); flip `report_to` in the config
              to re-enable.
```

---

## Datasets

### LaMP training splits (primary)
Download from the LaMP benchmark. Use subtasks: **LaMP-3, LaMP-4, LaMP-6, LaMP-7**.
- LaMP-3: personalized product rating prediction
- LaMP-4: personalized news headline generation
- LaMP-6: personalized email subject generation
- LaMP-7: personalized tweet paraphrasing

### LaMP time-based splits (for User-LoRA, Phase 2)
The LaMP benchmark publishes both a *user-based* split (the corpus already at `data/lamp/`, used for the A1-lamp Task-LoRA — users are disjoint across train / dev / test) and a *time-based* split (same users in every split, but each user's interactions are partitioned chronologically — earlier ones in train, later in dev/test). The time-based split is the one needed for per-user User-LoRA experiments.

**Downloaded 2026-06-10** to `data/lamp_time/LaMP_{3,4,7}/` via the updated `data/download_lamp.py --split-type time` (LaMP_6 dropped from the active task list per the Avocado-placeholder issue). All 18 files are non-empty and JSON-parseable; time-split test_outputs are present (not withheld as on user-split test); profile entries carry a `date` field for the chronological partition.

**Per-user data volume varies sharply by task** (analysis 2026-06-12, see `experiments/2026-06-12-lamp-time-split-per-user-counts.md` + `data/lamp_user_stats.py`): LaMP_4 averages ~7.5 time-split records per user (max 946; 17 unseen-by-A1-lamp users have ≥1 train + ≥4 dev); LaMP_3 and LaMP_7 are essentially one-record-per-user in the time-split. With the records-as-training-examples framing only LaMP_4 supports a single-user User-LoRA. With the profile-entries-as-(input,gold)-pairs reframing LaMP_3 becomes the richest (~175 examples/user from review→rating pairs); LaMP_4 stays viable from either source; LaMP_7 has no profile-level target (raw past tweets), so it stays stuck at 1-2 examples/user regardless.

### Synthetic preference-conditional data — DEFERRED (2026-06-02 pivot)
*Kept in this spec for historical record. The original plan generated this data offline with a teacher model (Qwen3-30B or GPT-4o) using contrastive pairs across the tool-calling / email-drafting / recommendations domains, mixed 60/40 LaMP/synthetic for A1-full and A2. The pivot drops both ablation conditions and the data pipeline.*

### Domain-specific corpora (for A2) — DEFERRED (2026-06-02 pivot)

*Kept in this spec for historical record. The intended sources were:*

| Domain | Sources |
|---|---|
| Tool-calling | `Salesforce/xlam-function-calling-60k`, `glaiveai/glaive-function-calling-v2` (filtered) |
| Email / text drafting | LaMP-6 training split, Enron email corpus (public) |
| Recommendations | LaMP-3 training split, Yelp open dataset (public) |

---

## Evaluation

Run on LaMP-3, LaMP-4, LaMP-6, LaMP-7 test splits.

| Task | Metric |
|---|---|
| LaMP-3 (rating prediction) | Accuracy |
| LaMP-4 (headline generation) | ROUGE-1 |
| LaMP-6 (email subject) | ROUGE-1 |
| LaMP-7 (tweet paraphrase) | ROUGE-1 |

Also run **BFCL regression** before and after each Task-LoRA training run. Target: score stays ≥ 90 (baseline is 92.3). Sanity check only — improvement on BFCL is not a goal.

**Key comparison chain (post-pivot, 2026-06-02):**
1. Baseline → A1-lamp: does fine-tuning help at all? **Answered: yes** (see A1-lamp ckpt-1000 results in `experiments/2026-06-02-a1-lamp-1ep-pareto.md`; headline numbers re-issued on the test split in `experiments/2026-06-13-lamp-test-split-correction.md`).
2. A1-lamp (Task-LoRA only) → A1-lamp + User-LoRA: does an additional per-user LoRA on the user's own time-ordered history meaningfully personalize beyond the Task-LoRA's profile-in-system mechanism?

The original comparison chain (Baseline → A1-lamp → A1-full → A2) is obsolete.

---

## Repo structure (current — reflects what's actually built)

```
/
├── CLAUDE.md, Dockerfile, requirements.txt, pyrightconfig.json
├── condor/
│   ├── build_dataset.sub        # one-shot CPU job: preprocess LaMP train → JSONL
│   ├── download_model.{py,sub}  # one-time HF Hub pull of SmolLM3-3B
│   ├── interactive.sub          # GPU shell for smoke tests
│   ├── eval_lamp.sub            # LaMP profile-baseline + adapter eval (3 parallel jobs)
│   ├── eval_lamp_floor.sub      # LaMP non-personalized floor (3 parallel jobs)
│   ├── eval_bfcl.sub            # BFCL AST regression (1 GPU job, all categories)
│   ├── train.sub                # Task-LoRA training, 2-epoch config (1 GPU job, reads train/config/a1_lamp.json) — superseded by train_1ep.sub
│   ├── train_1ep.sub            # Task-LoRA training, 1-epoch config (canonical going forward)
│   ├── chat.py                  # interactive REPL with the model + optional adapter
│   └── smoke_test.py            # Docker-image env check
├── data/
│   ├── download_lamp.py         # LaMP download; `--split-type {user,time}` flag (default user)
│   ├── lamp/                    # user-based split — LaMP-{3,4,7}/{train,dev,test}_{questions,outputs}.json
│   ├── lamp_time/               # time-based split — same task layout, downloaded 2026-06-10
│   ├── lamp_user_stats.py       # per-user record-count analysis for time-split (User-LoRA scoping)
│   ├── lamp_user_stats/         # output CSVs: <task>_users.csv per task
│   ├── models/SmolLM3-3B/       # downloaded weights (~6 GB)
│   ├── lamp_train_{LaMP_3,LaMP_4,LaMP_7,mixed}_bm25k4.jsonl  # built by build_dataset.py
│   ├── lamp_train_mixed_bm25k4.meta.json                     # provenance sidecar
│   ├── synthetic/               # DEFERRED 2026-06-02
│   └── tool/                    # DEFERRED 2026-06-02
├── train/
│   ├── build_dataset.py         # streams raw LaMP train → BM25-retrieved JSONL
│   ├── train.py                 # SFT trainer — reads config JSON, applies SmolLM3 chat template (thinking off), loss-masked to assistant, streams metrics.jsonl
│   ├── config/
│   │   ├── a1_lamp.json         # 2-epoch A1-lamp config (superseded; kept for provenance)
│   │   └── a1_lamp_1ep.json     # 1-epoch A1-lamp config (canonical going forward — produces checkpoint-1000 deliverable)
│   └── checkpoints/             # training output (gitignored)
│       ├── a1_lamp_seed0/       # 2-epoch run — superseded
│       └── a1_lamp_1ep_seed0/   # 1-epoch sweep; canonical adapter is checkpoint-1000/
├── eval/
│   ├── eval_lamp.py             # LaMP harness (BM25 k=4 retrieval, refuse-to-overwrite)
│   ├── eval_bfcl.py             # BFCL harness using bfcl-eval's ast_checker as a library
│   └── summary.py               # flatten results/*.json → one-row-per-run table
├── results/                     # flat scalar JSON records + predictions JSONL
├── runlogs/                     # Condor stdout/stderr (gitignored)
├── experiments/                 # YYYY-MM-DD-<slug>.md per run
└── notebooks/                   # personal analysis (gitignored)
```

---

## Phase 1 implementation status (as of session refresh)

**Completed:**
- Zero-shot baselines on LaMP-3 / LaMP-4 / LaMP-7 dev — both the profile baseline
  (BM25 k=4 in system) and the non-personalized floor. Documented in
  `experiments/2026-05-29-baseline-lamp.md`.
- BFCL bare-model AST regression (catastrophic-forgetting reference).
  Documented in `notebooks/bfcl_baseline_results.md`.
- Training-corpus preprocessing (`train/build_dataset.py`) — produced the
  full LaMP-3/4/7 BM25-k=4 corpus: 42,964 examples in
  `data/lamp_train_mixed_bm25k4.jsonl` (20,000 + 12,527 + 10,437),
  87.9 MB, via Condor job 163355. Provenance in `lamp_train_mixed_bm25k4.meta.json`.
- **A1-lamp training (2-epoch) — completed and SUPERSEDED.** Trained 2026-06-01
  (Condor job 163392.0, 5h 08m on A100). Evaluated 2026-06-02 — strong LaMP
  gains but a 10.87 pp BFCL regression (overall AST 0.6991). Subsequent
  follow-up showed both `final` (step 2682) and `checkpoint-2000` are
  Pareto-dominated by the 1-epoch sweep below. Adapter still on disk for
  provenance: `train/checkpoints/a1_lamp_seed0/final/`.
- **A1-lamp 1-epoch Pareto sweep — completed and CANONICAL.** Retrained
  2026-06-02 with `train/config/a1_lamp_1ep.json` (num_train_epochs=1,
  save_steps=200, save_total_limit=null), evaluated all 7 retained
  checkpoints on LaMP + BFCL. Documented in
  `experiments/2026-06-02-a1-lamp-1ep-pareto.md`.

**Canonical A1-lamp adapter (use this for all downstream work):**
`train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000/` (step 1000, epoch 0.75).
LaMP-3 acc 0.8056, LaMP-4 rouge1 0.2259, LaMP-7 rouge1 0.5619 (test split), BFCL AST 0.7696
— LaMP within noise of any other 1-epoch checkpoint; BFCL is the second-best
of any adapter (only the further-undertrained `checkpoint-400` is higher, at
the cost of LaMP-7). `checkpoint-400` is the alternative if maximum BFCL
retention is the dominant criterion.

**Result files for the canonical adapter:**
- `results/LaMP_{3,4,7}_test_a1_lamp_1ep_seed0_checkpoint-1000_bm25k4_seed0.{json,predictions.jsonl}` (canonical)
- `results/LaMP_{3,4,7}_dev_a1_lamp_1ep_seed0_checkpoint-1000_bm25k4_seed0.{json,predictions.jsonl}` (dev — used for model selection during the 1-epoch Pareto sweep)
- `results/bfcl_ast_a1_lamp_1ep_seed0_checkpoint-1000_seed0.{json,predictions.jsonl}`

**Baselines + canonical adapter for comparison** (LaMP test, seed 0, greedy, BM25 k=4 — dev numbers within ±0.01, see `experiments/2026-06-13-lamp-test-split-correction.md`):

| Task | No-profile floor | Profile baseline | A1-lamp (ckpt-1000) | $\Delta$ adapter − baseline |
|---|---|---|---|---|
| LaMP-3 | acc 0.4508 | acc 0.6964 | **0.8056** | **+0.109** |
| LaMP-4 | rouge1 0.1393 | rouge1 0.1537 | **0.2259** | **+0.072** |
| LaMP-7 | rouge1 0.4170 | rouge1 0.4372 | **0.5619** | **+0.125** |
| **BFCL AST overall** | — | **0.8078** (Py-only 0.8870) | **0.7696** | **−0.038** |

BFCL is split-independent (its own gold set, not LaMP's dev/test partition) — the
2026-06-02 numbers carry over unchanged.

Baseline result files: `results/LaMP_{3,4,7}_test_base_{bm25k4,noprofile}_seed0.{json,predictions.jsonl}` and `results/bfcl_ast_base_seed0.{json,predictions.jsonl}`. The corresponding `_dev_*` baseline files also still exist on disk.

**Next milestone:** TBD (awaiting next direction from the project lead).
Phase 2 / User-LoRA closed 2026-06-19 — see the top-of-file direction
update for the rationale and the headline R5 numbers. Fresh sessions
should ask before proposing follow-up User-LoRA experiments; the
relevant R6+ candidate axes (K=200, recipe ablation, dev/test asymmetry
diagnostic on u00000011) are shelved.

The A1-lamp ckpt-1000 remains the frozen Task-LoRA foundation. On-disk
adapters preserved for any future re-analysis:
- 100 LaMP-3 multi-user User-LoRAs from R5: `train/checkpoints/user_lora_lamp3_<fp>_seed0/final/`
- 4 single-user LaMP-4 User-LoRAs from R1 / R2-B / R4 (R3-α reused R1's adapter)

### Round 1 — DONE 2026-06-15, null result

Pre-registered single-user existence proof on u00000011 (LaMP_4). Full
writeup: `experiments/2026-06-15-user-lora-lamp4-u00000011-round1.md`.
Design: `experiments/2026-06-14-user-lora-round1-plan.md` + memory
`project-user-lora-round1-design`.

**Headline:** the pre-registered gate (mean paired ROUGE-1 Δ > 0 AND paired
t-test p < 0.05 on test, MDE ≈ +0.05) **fails on both splits**:

| | n | mean Δ (C3 − C2) | t p-value | Wilcoxon p | 95% bootstrap CI | gate |
|---|---|---|---|---|---|---|
| **test (primary)** | 25 | +0.0032 | 0.925 | 0.812 | [−0.058, +0.072] | **FAIL** |
| dev (transparency) | 21 | +0.0295 | 0.150 | 0.139 | [−0.007, +0.068] | FAIL |

C1 = base + BM25, C2 = A1-lamp ckpt-1000 + BM25, C3 = A1-lamp + User-LoRA + BM25.
Training loss collapse at epoch boundaries (1.93 → 1.10 → 0.64) is consistent
with memorization of the 1100 profile entries that doesn't transfer to later
headlines. Result was reported honestly per pre-registration discipline; no
post-hoc loosening.

Scaffolding shipped as a side-effect (kept on disk): per-user data builder
(`train/build_user_dataset.py`), per-user dataset
(`data/lamp_user_train_LaMP_4_u00000011_bare.jsonl`), train.py
`base_adapter` plumbing, eval_lamp.py `--base-adapter` + `--user-records`
flags, `eval/paired_compare.py`, and the three new Condor sub files
(`condor/build_user_dataset.sub`, `condor/train_user_lora.sub`,
`condor/eval_lamp_user.sub`, `condor/paired_compare.sub`). Two latent
scaffolding bugs caught + fixed during eval (no result contamination):
`LAMP_DIR` defaulted to `data/lamp/` (user-based) instead of
`data/lamp_time/`; `--adapter` / `--base-adapter` paths weren't resolved
against `PROJECT_ROOT`, so they had to be passed absolute in the Condor
queue block.

### Round 2 — variant B done 2026-06-16, null result

Pre-registered retrain with BM25 in the system slot at train time, matching
eval-time shape. Full writeup:
`experiments/2026-06-16-user-lora-lamp4-u00000011-round2-B.md`. Plan:
`experiments/2026-06-16-user-lora-round2-B-plan.md`.

**Headline:** the pre-registered gate (C3' vs C2 paired-t on test ROUGE-1)
**fails on test** but shows a narrow miss on dev:

| | n | mean Δ (C3' − C2) | t p-value | Wilcoxon p | 95% bootstrap CI | gate |
|---|---|---|---|---|---|---|
| **test (primary)** | 25 | −0.0036 | 0.89 | — | — | **FAIL** |
| dev (transparency) | 21 | +0.043 | 0.06 | — | — | FAIL (near miss) |

C3' = A1-lamp ckpt-1000 + Round-2-B User-LoRA (BM25-trained) + BM25 eval. The
**secondary R1→R2-B comparison on test** (Δ=−0.007, p=0.80) empirically
downweights the train/eval shape-mismatch hypothesis — matching the prompt
shapes did not move the needle, leaving the redundancy hypothesis (Round 3 α)
as the most-supported next axis.

Adapter retained on disk:
`train/checkpoints/user_lora_lamp4_u00000011_bm25k4_seed0/final/`.

### Round 3 — variant α done 2026-06-16, null result

Plan: `experiments/2026-06-16-user-lora-round3-alpha-plan.md` (9 design axes
pre-registered via /grill_me; memory `project-user-lora-round3-alpha-design`).
Writeup: `experiments/2026-06-16-user-lora-lamp4-u00000011-round3-alpha.md`.

**Headline:** the pre-registered gate (α-bare vs C2-α paired-t on test
ROUGE-1) **fails on test** for the third round running on essentially the
same point estimate as Round 1:

| | n | mean Δ (α-bare − C2-α) | t p-value | Wilcoxon p | 95% bootstrap CI | gate |
|---|---|---|---|---|---|---|
| **test (primary)** | 25 | +0.0098 | 0.732 | 0.449 | [−0.046, +0.061] | **FAIL** |
| dev (transparency) | 21 | +0.0475 | 0.039 | 0.057 | [+0.007, +0.090] | FAIL (gate is test-only) |

C1-α=base+no-profile, C2-α=A1-lamp+no-profile, α-bare=A1-lamp+R1-User-LoRA+no-profile,
α-B=A1-lamp+R2B-User-LoRA+no-profile. The **redundancy hypothesis as posed**
is partly refuted by C2's own no-BM25 evaluation: C2-α (0.1875 test) ≈
C2-with-BM25 (0.1906 test) — BM25 at inference contributes ~0.003 ROUGE-1
for this user, so cannot have been masking the User-LoRA. The Task-LoRA
absorbed the personalization signal at training time. The User-LoRA does
add a small consistent lift on top (+0.010 test, +0.047 dev) below the +0.05
MDE the experiment was powered to detect. The dev/test asymmetry observed
in R1 and R2-B is now a robust three-round pattern (dev Δ +0.030 / +0.043 /
+0.047 vs test Δ +0.003 / −0.004 / +0.010) — deferred to R5 as a diagnostic
target if R4 also nulls.

### Round 4 — DONE 2026-06-17, null result

Pre-registered single-axis follow-up to R2-B per
`experiments/2026-06-17-user-lora-round4-plan.md` (7 design axes pre-reg via
/grill_me; memory `project-user-lora-round4-design`). Full writeup:
`experiments/2026-06-17-user-lora-lamp4-u00000011-round4.md`.

**Single-axis change vs R2-B:** training-data framing — 241 LaMP_4
train-period (input, gold) record pairs instead of the 1100 profile-entry
(text, title) pairs. BM25-train + BM25-eval otherwise identical to R2-B.
A1-lamp ckpt-1000 base, $r=4$, $\alpha=8$, 3 epochs, seed=0.

**Headline:** the pre-registered gate (C3-R4 vs C2 paired-t on test ROUGE-1,
mean Δ > 0 AND p < 0.05, MDE ≈ +0.05) **fails on test** for the fourth
round running, on the widest negative point estimate of the four:

| | n | mean Δ (C3-R4 − C2) | t p-value | Wilcoxon p | 95% bootstrap CI | gate |
|---|---|---|---|---|---|---|
| **test (primary)** | 25 | −0.0178 | 0.429 | 0.506 | [−0.0598, +0.0237] | **FAIL** |
| dev (transparency) | 21 | +0.0396 | 0.041 | 0.033 | [+0.0037, +0.0736] | FAIL (gate is test-only) |

C1 = base + BM25, C2 = A1-lamp + BM25, C3-R4 = A1-lamp + R4 User-LoRA
(records, BM25-trained) + BM25 eval. Per-record wins (test): C3-R4 8 /
ties 3 / C2 14. Cell ROUGE-1 means on test: C1 0.144, C2 0.191, R1-C3
0.194, R2B-C3' 0.187, **C3-R4 0.173** (lowest of the User-LoRA-stacked
cells).

The **cleanest single-axis descriptive R2B-C3' → C3-R4** (both BM25-trained,
only the training-data framing differs) finds the framing change essentially
inert on dev (Δ=−0.003, p=0.90) and a slight regression on test (Δ=−0.014,
p=0.68). Records framing changed the User-LoRA's behavior basically not at
all. The plan's structural argument was correct about the training
dynamics (R4 loss trajectory 1.71 → 1.11 → 0.79 vs R1's 1.93 → 1.10 → 0.64
— smaller memorization crater consistent with ~5× fewer training examples)
but the smaller crater did not, on this user, translate into measurably
better test transfer.

The dev/test asymmetry pattern is now four rounds wide: dev Δ
+0.030/+0.043/+0.047/+0.040 vs test Δ +0.003/−0.004/+0.010/−0.018 across
R1/R2-B/R3-α/R4. Adapter retained on disk:
`train/checkpoints/user_lora_lamp4_u00000011_records_bm25k4_seed0/final/`.

**R5 axis selection (matrix Row 2 fired, mechanical):** gate fails on test
AND dev Δ > +0.02 AND dev paired-t p < 0.10 — all three numeric thresholds
met. **R5 = dev/test asymmetry diagnostic on u00000011.** Per-record
date / length / topic analysis of why this user's test records systematically
underperform their dev records under personalization. The C1 → C3-R4
descriptive shows the same asymmetry at the whole-stack level (dev Δ +0.069
p=0.008 vs test Δ +0.029 p=0.37), so the pattern is a property of the data,
not the adapter — Row 2 was reserved exactly for triggering a data-side
investigation. Higher rank, epoch-reduction Pareto, and multi-user
replication remain parked at R5 rows 3 / 4 / 1 behind the asymmetry
diagnostic.

### Round 5 — DONE 2026-06-18, Q4 confirmed, phase closed 2026-06-19

Multi-user LaMP-3 (K=100) replication of OPPU's per-user PEFT recipe
(r=8 on q+v only, LR=1e-5, AdamW + L2 weight_decay=1e-2, 3 epochs).
Stacked on top of A1-lamp ckpt-1000 via `train.py`'s `base_adapter`
plumbing. Plan: `experiments/2026-06-18-user-lora-round5-lamp3-plan.md`.
Writeup: `experiments/2026-06-18-user-lora-lamp3-round5-multi.md`.

R5 was an explicit override of R4's matrix-pre-committed dev/test
asymmetry diagnostic, on the grounds that the OPPU literature comparison
showed our LaMP-4 +0.010 lift was in OPPU's published range and the
LaMP-4 nulls were power-limited at n=25 rather than methodologically
broken.

**Headline (test, n=100 paired):**
- Mean ΔMAE −0.050 (at the pre-registered MDE)
- Accuracy 0.680 → 0.730 (+5 net flips wrong→right)
- RMSE 0.616 → 0.575
- C3 head-to-head wins 7 vs C2 wins 2, ties 91
- Zero inference overhead (MPT/MGT/latency identical between C2 and C3)
- Effect size in OPPU's published range (their LaMP-3 lift −0.071)

**Phase closed 2026-06-19.** Project decision: R5 results confirm Q4.
No further User-LoRA rounds planned. R6+ candidate axes (K=200, recipe
ablation, dev/test asymmetry diagnostic on u00000011) are shelved.
A1-full and A2 ablations from the original plan remain dropped.

---

## Standard script patterns (converged on across all eval/train/data-prep scripts)

- **Provenance banner at startup** — first stdout line prints task / split /
  condition / seed / commit short SHA / Condor cluster.proc IDs / host. Makes
  runlogs self-documenting.
- **Provenance dict in every result record** — `git_commit`, `git_dirty`,
  `condor_cluster_id`, `condor_proc_id`, `hostname`, `timestamp_utc`, library
  versions (torch / transformers / peft / bfcl-eval).
- **Flat single-level JSON result records** — every field is a scalar, so
  `pd.DataFrame([json.load(open(p)) for p in glob.glob("results/*.json")])`
  produces a usable sweep table with zero unnesting.
- **Per-example predictions in a sibling JSONL** — one `{id, pred, gold}` per
  line (BFCL adds `category`, `pred_text`, `pred_parsed`, `valid`, `error_type`).
- **Refuse-to-overwrite by default** — every output-producing script checks
  existing files up front and `sys.exit(1)` unless `--overwrite` is passed.
  Smoke runs (`--limit > 0`) get an `_limitN` filename suffix so they can't
  collide with full-run outputs even if `--overwrite` was used.
- **Condor IDs forwarded via the submit file's `environment` line**:
  `CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)` — the script reads
  them via `os.environ.get`. This is what links a result record back to its
  runlog file.

---

## Eval methodology choices (frozen — don't relitigate)

- **LaMP personalization channel = BM25 top-k retrieval** (k=4) of the user's
  profile into the system slot. Not summarization. The summarization detour
  was tried, then reverted — see `notebooks/lamp_evaluation_approach.md` for
  the reasoning. The same BM25, same k, same per-task formatting, and same
  role layout are used at training time (build_dataset.py) and eval time
  (eval_lamp.py); train/eval consistency is the cardinal rule.
- **BFCL eval uses Path C** — install bfcl-eval in the image, generate via our
  own transformers stack, call the official `ast_checker` as a library on the
  outputs. Avoids the vllm/sglang requirement of `bfcl generate` and avoids
  upstreaming a SmolLM3 handler. Caveats: SmolLM3 isn't in BFCL's
  `MODEL_CONFIG_MAPPING`, so we pass `model_name="meta-llama/Llama-3.1-8B-Instruct"`
  as a neutral placeholder (recorded in every result as
  `scorer_model_name_placeholder`); the `BFCL_PROJECT_ROOT` env var must be
  set before any bfcl_eval import (eval_bfcl.py sets it to `/tmp/bfcl_project_root`).
- **LaMP-6 is unsupported** — its public release ships only Avocado email
  file-id placeholders (no text), so it can't be scored without the licensed
  Avocado corpus. Treat the LaMP task list as effectively LaMP-{3,4,7}.
- **BFCL `irrelevance` category is currently skipped** (the data file
  `possible_answer/BFCL_v4_irrelevance.json` doesn't ship — correct answer is
  "no call", no gold needed). `eval_bfcl.py` could be extended in ~10 lines to
  load questions only and score "correct iff `pred_parsed == []`".

---

## Docker image

Current tag: **`ghcr.io/gordofreemo/smollm3-train:ver4`**. Layers (in order):

1. Base: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (Python 3.11, torch 2.5.1+cu124)
2. `apt-get install git` — added ver3 — for `git_commit` provenance capture inside the container
3. `pip install -r requirements.txt` — transformers, peft, datasets, accelerate, wandb, rouge_score, bfcl-eval, soundfile

**When you change `requirements.txt` or the Dockerfile**, bump the tag and update
**all seven** sub files: `condor/{eval_lamp,eval_lamp_floor,eval_bfcl,build_dataset,train,interactive,download_model}.sub`.

---

## Open design decisions

1. ~~**Train-time prompt regime.**~~ **Resolved 2026-05-31:** system-always
   regime picked. The BM25 profile sits in `system` for every training
   example, so the Task-LoRA expects that input shape at inference and the
   deployed model pays the +118 to +482 prompt-token tax per query. The
   open hypothesis is that Phase-2 on-device User-LoRA training can absorb
   the profile into adapter weights and let us drop it from `system` at
   inference. If A1-lamp doesn't beat the profile baseline at all, the
   fallback is to rebuild the corpus in mixed regime (some examples with
   profile, some without) and retry — `build_dataset.py` would need a
   small flag for that.
2. **BFCL `irrelevance` not yet scored** — small fix (above).
3. **BFCL Java/JS errors not investigated** — 80 `type_error:{java,js}`
   account for most of the 80.78 vs 92.3 gap. Could be a model limitation or
   a parser/normalizer issue. Worth a 2-min spot-check before the
   post-training comparison.

---

## Hard constraints

- **Never modify base model weights.** LoRA only; base is always frozen.
- **No KD loss.** CE only.
- **No co-loading teacher and student.** Teacher is offline data generation only.
- **No on-device / mobile code** in this phase. ExecuTorch, Core ML, and iOS-specific code are out of scope.
- **Reproducibility first.** Every training run must be launchable from one CLI command with a fixed seed. Log the full command in the experiment log entry.
- **No profile leakage between splits.** Validate this explicitly before training.

---

## Experiment log format

Every run gets a file at `experiments/YYYY-MM-DD-<slug>.md`:

```markdown
## Hypothesis
## Setup (command, config, seed)
## Result (loss curve, eval numbers)
## Conclusion
```

---

## Key references (do not hallucinate URLs)

- SmolLM3-3B: `HuggingFaceTB/SmolLM3-3B` on HuggingFace
- LaMP benchmark: lamp-benchmark.github.io
- PTBench (synthetic data methodology): arXiv 2505.04072
- BFCL: gorilla.cs.berkeley.edu/leaderboard.html
- xlam-function-calling-60k: `Salesforce/xlam-function-calling-60k` on HuggingFace
- CDCDA-PLM (closest prior work — cloud synthetic data + on-device PEFT + LaMP): arXiv 2508.21313
