# Research Project — On-Device LLM Personalization via PEFT

Fine-tuning **SmolLM3-3B** for personalized instruction following via a two-stage
LoRA pipeline: a **Task-LoRA** (LaMP-{3,4,7}, BM25 profile in `system`) plus a
per-user **User-LoRA** on time-ordered history, stacked at inference. Phases 1 & 2
ran on the cluster; Phase 3 deploys to a real iPhone.

## Active work (as of 2026-06-22)

- **Phase 3 — on-device deployment (PRIMARY).** Base SmolLM3-3B inference on iPhone DONE 2026-06-21; characterization benchmark DONE. Next on-device milestone: fuse A1-lamp ckpt-1000, convert to MLX, swap the `modelConfiguration` id, measure base vs Task-LoRA with the same rig.
- **Round 6 (LaMP-4 multi-user) reopened from Phase 2.** Status PRE-EXECUTION as of 2026-06-19. Fresh sessions should check `experiments/2026-06-19-user-lora-round6-lamp4-multi-plan.md` (design) and `ls results/paired_compare_*round6*.json` (whether the gate output landed) before proposing anything new in this thread.

**The "no on-device/mobile code" hard constraint is LIFTED for Phase 3** (it
remains the historical framing for Phases 1–2).

---

## Phase 3 runbook — on-device deployment

### Runtime decision (verified against official sources only)

- **Runtime = MLX** (`mlx-swift` on device, `mlx-lm` on the Mac). llama.cpp / ExecuTorch rejected.
- Apple Foundation Models is the standard framework; "Core AI" is its ship-your-own-local-model provider. Apple's adapter-training toolkit is Apple-model-only (rank-32 LoRA bound to the OS system model) — cannot adapt SmolLM3.
- **SmolLM3 is first-class in MLX**: `mlx_lm/models/smollm3.py` (Python) and `Libraries/MLXLLM/Models/SmolLM3.swift` (Swift, in `ml-explore/mlx-swift-lm`). Confirmed by reading source.
- MLX is also the credible **on-device training** route (`mlx_lm.lora`, the `LoRATrainingExample` app, Apple paper arXiv:2510.03425).
- **Model delivery = HF download on-device.** The app pulls `mlx-community/SmolLM3-3B-4bit` into its sandbox on first generation. We publish our own HF repo only once we fuse Task-LoRA.

### Host / device / signing facts

- Mac: Apple **M3, 16 GB**. Full Xcode 26.5 at `/Applications/Xcode.app`, but active dev dir is CommandLineTools — **prefix every Xcode/devicectl command** with `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- Signing: **Apple Development: andrew.geyko@icloud.com**, team **`JGW9U9Y36Y`** (free personal team; bundle IDs auto-disambiguated via `DISAMBIGUATOR=${DEVELOPMENT_TEAM}` in `Configuration/Build.xcconfig`).
- Testbed: **iPhone 17 Pro** (`iPhone18,1`), iOS **26.5.1**, Developer Mode on. UDID **`00008150-000674C60A3B401C`**. List: `xcrun devicectl list devices`.

### Local MLX toolchain (Mac-side)

- venv **`.venv-mlx/`** (Python **3.11** — 3.14 too new for MLX wheels), `mlx-lm` (mlx 0.31.2). Gitignored.
- Convert + quantize: `.venv-mlx/bin/python -m mlx_lm convert --hf-path HuggingFaceTB/SmolLM3-3B --mlx-path data/models/SmolLM3-3B-mlx-4bit -q --q-bits 4`
- Sanity generate: `.venv-mlx/bin/python -m mlx_lm generate --model data/models/SmolLM3-3B-mlx-4bit --prompt "..." --max-tokens 60` (SmolLM3 has **thinking mode on by default** — emits `<think>…</think>`).

### iOS app — vendored, edited, built, deployed

- **`ios/mlx-swift-examples/`** is vendored into this repo via `git subtree` (upstream `ml-explore/mlx-swift-examples` base `378f244`). Edit harness files → ordinary `git commit`. Bump upstream: `git subtree pull --prefix=ios/mlx-swift-examples https://github.com/ml-explore/mlx-swift-examples <tag> --squash`. Only `build/` + Xcode user state gitignored. See `ios/README.md`.
- LLM libs come from the SPM package `ml-explore/mlx-swift-lm`, not vendored.
- **Edited:** `ios/mlx-swift-examples/Applications/LLMEval/ViewModels/LLMEvaluator.swift` (`modelConfiguration` → `mlx-community/SmolLM3-3B-4bit`; `appendBenchRecord(...)` appends one flat-JSON line per generation to `Documents/bench_metrics.jsonl`; `hardwareModelIdentifier()` helper).
- Benchmark harness: `ios/mlx-swift-examples/Applications/LLMEval/Benchmark/{BenchmarkSupport,LLMEvaluator+Benchmark}.swift` (+ edits to `LLMEvaluator.swift`, `ContentView.swift`, `project.pbxproj`). Bump `BenchConstants.appBuild` whenever harness logic changes.

**Build (device, signed)** — `-skipMacroValidation` is **required** (else fails on `MLXHuggingFaceMacros … must be enabled`):
```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cd ios/mlx-swift-examples
xcodebuild -project mlx-swift-examples.xcodeproj -scheme LLMEval \
  -configuration Debug -destination 'id=00008150-000674C60A3B401C' \
  -derivedDataPath ./build -allowProvisioningUpdates -skipMacroValidation \
  DEVELOPMENT_TEAM=JGW9U9Y36Y build
```
Output: `build/Build/Products/Debug-iphoneos/LLMEval.app`, bundle id **`mlx.LLMEvalJGW9U9Y36Y`**.

**Install + launch:**
```
xcrun devicectl device install app --device 00008150-000674C60A3B401C build/Build/Products/Debug-iphoneos/LLMEval.app
xcrun devicectl device process launch --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y
```
First generation downloads ~1.73 GB from HF (Wi-Fi, one-time; model persists across reinstalls of the same bundle id). The app UI requires a human tap to start generation.

### Reading metrics off the device

No live console (macOS `log stream` has no `--device`; `log collect --device` needs root; `idevicesyslog` not installed). Pull the JSONL from the app container (no sudo); accumulates one line per run:
```
xcrun devicectl device copy from --device 00008150-000674C60A3B401C \
  --domain-type appDataContainer --domain-identifier mlx.LLMEvalJGW9U9Y36Y \
  --source Documents/bench_metrics.jsonl --destination /tmp/devpull/bench_metrics.jsonl
```

### Base-inference benchmark — DONE 2026-06-21

Design: `experiments/2026-06-21-ondevice-base-inference-plan.md`. Writeup: `experiments/2026-06-21-ondevice-base-inference.md`. Rig is reused verbatim for base-vs-Task-LoRA.

**Headline (steady-state, cool device, n=5/cell):** decode ~37 tok/s at deployment context sizes (gen64: 38.8 @64-tok prompt → 32.0 @2048), prefill 620–740 tok/s, cold app-launch→first-answer ≈ 1.7 s (model load 1362±114 ms + cold TTFT 380±6 ms), realistic LaMP-3 (natural EOS) 35.0±1.2 tok/s, peak ≈ 2.2 GB (at 2048-tok contexts).

**Two findings that shape the next pass:**
1. **Sustained decode throttles −53%** (37.8 → 17.9 tok/s over 5 min / 6144 tokens, knee ~90 s) and `ProcessInfo.thermalState` stayed `nominal` throughout — coarse enum is useless as throttle proxy; trust per-segment tok/s.
2. **Clean steady-state long-decode curves unobtainable while plugged.** Pre-registered unplugged-over-Wi-Fi follow-up is the way to get a clean decode curve — deferred, not blocking.

**Harness verified vs `mlx-swift-lm` source:** EOS suppression for forced length = drive `TokenIterator` directly and ignore the stop-token set to `maxTokens` (EOS check is in MLXLMCommon's loop wrapper, not `TokenIterator.next()`); `model_load_ms` brackets the `ModelContainer` load. Launch args: `--benchmark` (cold + prefill + decode), `--benchmark-tail` (realistic + 5-min stress, separate launch), `--benchmark-cold` (load+1 gen+exit, ×3 for cold variance). `app_build` baked in (`smollm3-ondevice-bench-h2`).

**Deliverables:** aggregator `eval/bench_aggregate.py` (stdlib-only); raw telemetry `results/ondevice/bench_metrics_smollm3-4bit-base_2026-06-21.jsonl` (89 records); aggregate `results/ondevice_base_smollm3_4bit_2026-06-21.json`.

**`peak_mem_bytes` caveat:** MLX `peakMemory` is a process high-water mark (monotonic within a session), so per-cell peaks are confounded by execution order — meaningful number is the session peak (~2.2 GB). Reset-per-run is the h3 improvement for the base-vs-LoRA comparison.

### Phase 3 next steps

- **Task-LoRA on-device:** fuse `a1_lamp_1ep_seed0/checkpoint-1000` into SmolLM3, convert to MLX, publish an HF repo, swap `modelConfiguration` id; measure base vs Task-LoRA with the same rig. (Adapter must be pulled from the cluster first — not on this Mac.)
- **Unplugged decode-curve follow-up** (pre-registered, deferred).
- **On-device training metrics** via `LoRATrainingExample` / `mlx_lm.lora`.

---

## Phases 1 & 2 — frozen state

### Research questions

| Q | Status |
|---|---|
| **Q1** — Does fine-tuning on LaMP help a 3B model at all? | **YES** — A1-lamp ckpt-1000 gives +0.11 / +0.07 / +0.13 on LaMP-3/4/7 test over BM25 baseline. |
| Q2 — Synthetic preference-conditional data on top of LaMP? | DROPPED with 2026-06-02 pivot. |
| Q3 — General Task-LoRA vs domain-specific? | DROPPED with 2026-06-02 pivot. |
| **Q4** — Per-user LoRA on time-ordered user history beyond Task-LoRA alone? | **YES (LaMP-3)** confirmed by R5 (2026-06-19): ΔMAE −0.050, acc 0.680→0.730, RMSE 0.616→0.575, zero inference overhead. R6 LaMP-4 cross-task replication PRE-EXECUTION. |

### Canonical artifacts

- **A1-lamp Task-LoRA:** `train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000/` (1-epoch sweep, step 1000, epoch 0.75, frozen 2026-06-02).
- **100 LaMP-3 User-LoRAs (R5):** `train/checkpoints/user_lora_lamp3_<fp>_seed0/final/` (one per user in `data/lamp_user_stats/LaMP_3_top100_users.json`).
- **4 single-user LaMP-4 User-LoRAs (R1/R2-B/R4):** retained for re-analysis.
- **(R6 will add)** 100 LaMP-4 multi-user User-LoRAs at `train/checkpoints/user_lora_lamp4_<fp>_oppu_seed0/final/`.

### Phase 1 headline numbers (LaMP test, seed=0, greedy, BM25 k=4 — dev numbers within ±0.01; see `experiments/2026-06-13-lamp-test-split-correction.md`)

| Task | No-profile floor | Profile baseline | A1-lamp (ckpt-1000) | Δ adapter − baseline |
|---|---|---|---|---|
| LaMP-3 (acc) | 0.4508 | 0.6964 | **0.8056** | **+0.109** |
| LaMP-4 (rouge1) | 0.1393 | 0.1537 | **0.2259** | **+0.072** |
| LaMP-7 (rouge1) | 0.4170 | 0.4372 | **0.5619** | **+0.125** |
| BFCL AST overall | — | **0.8078** (Py-only 0.8870) | **0.7696** | **−0.038** |

Result files: `results/LaMP_{3,4,7}_test_a1_lamp_1ep_seed0_checkpoint-1000_bm25k4_seed0.{json,predictions.jsonl}` (test, canonical), plus `_dev_*` variants. BFCL: `results/bfcl_ast_a1_lamp_1ep_seed0_checkpoint-1000_seed0.{json,predictions.jsonl}`. Baselines: `results/LaMP_{3,4,7}_test_base_{bm25k4,noprofile}_seed0.*` and `results/bfcl_ast_base_seed0.*`.

Earlier `a1_lamp_seed0/` (2-epoch run) is Pareto-dominated but on disk for provenance. Full Pareto sweep narrative: `experiments/2026-06-02-a1-lamp-1ep-pareto.md`. `checkpoint-400` is the alternative if maximum BFCL retention is the dominant criterion.

### Phase 2 history (one line per round)

Single-user u00000011 LaMP-4 rounds **R1–R4 all failed pre-registered gates on test** (dev/test asymmetry across all four: dev Δ +0.030/+0.043/+0.047/+0.040 vs test Δ +0.003/−0.004/+0.010/−0.018). **R5 LaMP-3 K=100 OPPU recipe** (r=8 q+v only, LR=1e-5, L2=1e-2, 3 epochs, stacked on A1-lamp ckpt-1000) confirmed Q4 at MDE. **Phase 2 closed 2026-06-19**, reopened same day as R6 cross-task descriptive replication. Full per-round detail in `experiments/2026-06-{15,16,17,18,19}-*.md` and memory `project_user_lora_lamp4_single_user_retrospective.md`, `project_user_lora_round5_lamp3_design.md`, `project_user_lora_round6_lamp4_design.md`.

**R6 carryovers from R5 (settled, not relitigated):** OPPU recipe verbatim (r=8, q+v only, alpha=16, dropout=0.05, AdamW, LR=1e-5, L2=1e-2, cosine + 3% warmup, 3 epochs, save_strategy=epoch, save_total_limit=1); per_device=2 / grad_accum=4 (R5's final working config — skip the OOM iteration); base = SmolLM3-3B + A1-lamp ckpt-1000 stacked via `--base-adapter`; eval = BM25 k=4 + greedy + seed=0 + `enable_thinking=False` + max_new_tokens=64; smoke = one user (smallest profile_size).

---

## Model & training

- **Base:** `HuggingFaceTB/SmolLM3-3B`, bf16, frozen.
- **Framework:** HuggingFace Transformers + PEFT.
- **Loss:** CE only. **No KD, no teacher co-loading, no base-weight modification.** Teacher is offline data generation only (and that whole branch is dropped — see hard constraints).

**Task-LoRA config:**
```python
LoraConfig(
    r=4, lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
```
r=4 (vs original spec r=64) because the value prop is on-device efficiency — adapter params scale linearly in r; alpha drops proportionally (alpha/r=2). See OPPU arXiv:2402.04401 for typical r=4–16 mobile configs.

**Training setup:** AdamW lr=3e-4, cosine + 3% warmup, per-device bs=4 × grad_accum=8 (effective 32), 2–3 epochs, checkpoint every 500 steps. Metrics streamed to `metrics.jsonl` + `train_meta.json` summary. W&B wired up in `train.py` but defaults off (flip `report_to` in config to re-enable).

---

## Datasets

- **LaMP user-based split** at `data/lamp/LaMP_{3,4,7}/` — used for A1-lamp Task-LoRA training (users disjoint across train/dev/test).
- **LaMP time-based split** at `data/lamp_time/LaMP_{3,4,7}/` — same users in every split, partitioned chronologically. Used for User-LoRA. Downloaded 2026-06-10 via `data/download_lamp.py --split-type time`. Test_outputs present (not withheld); profile entries carry `date` for the partition.
- **Per-user volume varies sharply by task** (`experiments/2026-06-12-lamp-time-split-per-user-counts.md`): LaMP_4 ~7.5 records/user avg with records framing; LaMP_3 richest with profile-entry reframing (~175 review→rating pairs/user); LaMP_7 stuck at 1–2 examples/user regardless. Tasks: LaMP-3 (rating prediction), LaMP-4 (headline gen), LaMP-7 (tweet paraphrase). **LaMP-6 unsupported** — only Avocado email file-id placeholders ship; needs licensed corpus.
- **Built corpora:** `data/lamp_train_{LaMP_3,LaMP_4,LaMP_7,mixed}_bm25k4.jsonl` (42,964 examples in `mixed`; 20,000 + 12,527 + 10,437; 87.9 MB), provenance in `lamp_train_mixed_bm25k4.meta.json`.
- **Synthetic preference-conditional data and domain-specific A2 corpora are DROPPED** (2026-06-02 pivot — Q2/Q3 dropped).

---

## Evaluation

| Task | Metric |
|---|---|
| LaMP-3 (rating prediction) | Accuracy |
| LaMP-4 (headline gen) | ROUGE-1 |
| LaMP-7 (tweet paraphrase) | ROUGE-1 |

Plus **BFCL AST regression** before/after each Task-LoRA training run (target ≥90; baseline 92.3; sanity check only).

**Comparison chain (post-pivot):** Baseline → A1-lamp (Q1, answered) → A1-lamp + User-LoRA (Q4, answered for LaMP-3, in-progress for LaMP-4).

---

## Repo structure

```
/
├── CLAUDE.md, Dockerfile, requirements.txt, pyrightconfig.json
├── condor/                       # Condor submit files + helper scripts
│   ├── build_dataset.sub         # CPU: preprocess LaMP train → JSONL
│   ├── download_model.{py,sub}   # one-time HF Hub pull of SmolLM3-3B
│   ├── interactive.sub           # GPU shell for smoke tests
│   ├── eval_lamp.sub             # LaMP profile-baseline + adapter eval (×3 parallel)
│   ├── eval_lamp_floor.sub       # LaMP non-personalized floor (×3 parallel)
│   ├── eval_bfcl.sub             # BFCL AST regression (1 GPU, all categories)
│   ├── train.sub                 # superseded (2-epoch A1-lamp)
│   ├── train_1ep.sub             # canonical (1-epoch A1-lamp)
│   ├── chat.py                   # REPL with model + optional adapter
│   └── smoke_test.py             # Docker-image env check
├── data/
│   ├── download_lamp.py          # `--split-type {user,time}` (default user)
│   ├── lamp/                     # user-based split — A1-lamp training
│   ├── lamp_time/                # time-based split — User-LoRA
│   ├── lamp_user_stats.py        # per-user record-count analysis
│   ├── lamp_user_stats/          # per-task user CSVs + R5/R6 top-K JSONs
│   ├── models/SmolLM3-3B/        # downloaded weights (~6 GB)
│   ├── lamp_train_*_bm25k4.jsonl # built by build_dataset.py
│   └── lamp_train_mixed_bm25k4.meta.json
├── train/
│   ├── build_dataset.py          # raw LaMP train → BM25-retrieved JSONL
│   ├── build_user_dataset.py     # per-user variant (User-LoRA)
│   ├── train.py                  # SFT trainer — config-driven, SmolLM3 chat template (thinking off), loss-masked to assistant, supports --base-adapter
│   ├── config/
│   │   ├── a1_lamp.json          # superseded (2-epoch)
│   │   ├── a1_lamp_1ep.json      # canonical (1-epoch, → checkpoint-1000)
│   │   └── user_lora_*.json      # R5/R6 OPPU templates
│   └── checkpoints/              # training output (gitignored)
├── eval/
│   ├── eval_lamp.py              # LaMP harness (BM25 k=4, refuse-to-overwrite, --base-adapter, --user-records)
│   ├── eval_bfcl.py              # BFCL via bfcl-eval's ast_checker as a library
│   ├── paired_compare.py         # single-user paired stats (User-LoRA R1-R4)
│   ├── paired_compare_per_user.py # multi-user grouped paired stats (R5/R6)
│   ├── bench_aggregate.py        # on-device bench JSONL → aggregate JSON
│   └── summary.py                # flatten results/*.json → table
├── ios/mlx-swift-examples/       # git-subtree vendored (upstream 378f244); LLMEval edited for SmolLM3 + benchmark harness
├── results/                      # flat scalar JSON + per-example predictions JSONL; ondevice/ subdir for bench telemetry
├── runlogs/                      # Condor stdout/stderr (gitignored)
├── experiments/                  # YYYY-MM-DD-<slug>.md per run
└── notebooks/                    # personal analysis (gitignored)
```

---

## Standard script patterns (converged across all eval/train/data-prep scripts)

- **Provenance banner at startup** — first stdout line prints task / split / condition / seed / commit short SHA / Condor cluster.proc IDs / host.
- **Provenance dict in every result record** — `git_commit`, `git_dirty`, `condor_cluster_id`, `condor_proc_id`, `hostname`, `timestamp_utc`, library versions.
- **Flat single-level JSON result records** — every field a scalar, so `pd.DataFrame([json.load(open(p)) for p in glob("results/*.json")])` works with zero unnesting.
- **Per-example predictions in a sibling JSONL** — one `{id, pred, gold}` per line (BFCL adds `category`, `pred_text`, `pred_parsed`, `valid`, `error_type`).
- **Refuse-to-overwrite by default** — every output-producing script checks existing files and `sys.exit(1)` unless `--overwrite` is passed. Smoke runs (`--limit > 0`) get an `_limitN` filename suffix so they can't collide with full-run outputs even if `--overwrite` was used.
- **Condor IDs forwarded via submit file's `environment`**: `CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)` — script reads via `os.environ.get`. Links result records back to runlog files.

---

## Eval methodology choices (frozen — don't relitigate)

- **LaMP personalization channel = BM25 top-k retrieval** (k=4) of the user's profile into the `system` slot. Not summarization (tried, reverted — see `notebooks/lamp_evaluation_approach.md`). Same BM25, same k, same per-task formatting, same role layout at training time (`build_dataset.py`) and eval time (`eval_lamp.py`). Train/eval consistency is the cardinal rule.
- **System-always prompt regime** (resolved 2026-05-31) — BM25 profile sits in `system` for every training example, so the Task-LoRA expects that shape at inference. Open hypothesis is on-device User-LoRA could absorb the profile into adapter weights and drop the `system` prompt tax (+118 to +482 tokens/query).
- **BFCL eval uses Path C** — install bfcl-eval in the image, generate via our own transformers stack, call `ast_checker` as a library on outputs. SmolLM3 isn't in BFCL's `MODEL_CONFIG_MAPPING`, so we pass `model_name="meta-llama/Llama-3.1-8B-Instruct"` as a neutral placeholder (recorded as `scorer_model_name_placeholder`); `BFCL_PROJECT_ROOT` must be set before any `bfcl_eval` import (`eval_bfcl.py` sets it to `/tmp/bfcl_project_root`).
- **BFCL `irrelevance` skipped** (data file `possible_answer/BFCL_v4_irrelevance.json` doesn't ship — correct answer is "no call"). Could be extended in ~10 lines to score `correct iff pred_parsed == []`.
- **BFCL Java/JS errors not investigated** — 80 `type_error:{java,js}` account for most of the 80.78 vs 92.3 gap. Worth a 2-min spot-check before post-training comparison.

---

## Docker image

Current tag: **`ghcr.io/gordofreemo/smollm3-train:ver4`**.
1. Base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (Python 3.11, torch 2.5.1+cu124)
2. `apt-get install git` (added ver3) — for `git_commit` provenance inside the container
3. `pip install -r requirements.txt` — transformers, peft, datasets, accelerate, wandb, rouge_score, bfcl-eval, soundfile

**When you change `requirements.txt` or the Dockerfile**, bump the tag and update **all seven** sub files: `condor/{eval_lamp,eval_lamp_floor,eval_bfcl,build_dataset,train,interactive,download_model}.sub`.

---

## Hard constraints

- **Never modify base model weights.** LoRA only; base frozen.
- **No KD loss.** CE only.
- **No co-loading teacher and student.** (Teacher branch dropped entirely with the 2026-06-02 pivot.)
- **Reproducibility first.** Every training run launchable from one CLI command with fixed seed. Log full command in the experiment file.
- **No profile leakage between splits.** Validate explicitly. For time-based splits this means no overlap within a user between train-period and dev-period interactions — enforced by LaMP's split by construction.
- **No on-device / mobile code** — **LIFTED for Phase 3** only. Phases 1 & 2 retain it as historical framing.

---

## Experiment log format

Every run gets `experiments/YYYY-MM-DD-<slug>.md`:

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
- OPPU (per-user PEFT recipe used in R5/R6): arXiv 2402.04401
- BFCL: gorilla.cs.berkeley.edu/leaderboard.html
- CDCDA-PLM (closest prior work — cloud synthetic + on-device PEFT + LaMP): arXiv 2508.21313
- Apple on-device fine-tuning (memory-efficient backprop): arXiv 2510.03425
