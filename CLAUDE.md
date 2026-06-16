# Research Project — On-Device LLM Personalization via PEFT

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
2. **User-LoRA (ACTIVE WORK).** A separate small LoRA per user, fine-tuned on that user's early-period interactions (from LaMP's **time-based** split, not the user-based split used for the Task-LoRA training). Evaluated on the user's later-period interactions. Sits on top of the frozen Task-LoRA at inference. This validates the on-device personalization story without requiring on-device hardware.

---

## Research questions

| Q | Status |
|---|---|
| **Q1** — Does fine-tuning on LaMP improve over zero-shot at all for a 3B model? | **ANSWERED: YES** — see `experiments/2026-05-31-a1-lamp.md`, `experiments/2026-06-02-a1-lamp-1ep-pareto.md`, and `experiments/2026-06-13-lamp-test-split-correction.md`. A1-lamp ckpt-1000 gives +0.11 / +0.07 / +0.13 over the BM25 baseline on LaMP-3/4/7 (test; dev results within ±0.01). |
| **Q2** — Does adding synthetic preference-conditional data on top of LaMP improve further? | **DROPPED** with the 2026-06-02 pivot. |
| **Q3** — Does a general Task-LoRA generalize as well as domain-specific ones? | **DROPPED** with the 2026-06-02 pivot. |
| **Q4** *(new)* — Does an additional per-user LoRA on time-ordered user history meaningfully personalize beyond the Task-LoRA alone? | **OPEN** — Rounds 1 and 2-B both failed the pre-registered gate on test. Round 1 (bare train, BM25 eval): Δ=+0.003, p=0.93 (`experiments/2026-06-15-user-lora-lamp4-u00000011-round1.md`). Round 2-B (BM25 train, BM25 eval): Δ=−0.004, p=0.89 (`experiments/2026-06-16-user-lora-lamp4-u00000011-round2-B.md`) — empirically downweighted the train/eval shape-mismatch hypothesis. Round 3 α (eval-only redundancy test, no retrain) pinned 2026-06-16, pre-execution: `experiments/2026-06-16-user-lora-round3-alpha-plan.md`. |

---

## Ablation conditions

| Condition | Training data | Status |
|---|---|---|
| **Baseline** | None (zero-shot) | Done |
| **A1-lamp** | LaMP-{3,4,7} training splits (user-based), profile in system at train + inference | **Done, canonical adapter = `train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000/`** |
| ~~**A1-full**~~ | ~~LaMP + synthetic preference-conditional data~~ | **DROPPED 2026-06-02** |
| ~~**A2**~~ | ~~Domain-specific corpora + domain synthetic data~~ | **DROPPED 2026-06-02** |
| **U** *(new, active)* | Per-user fine-tune on LaMP time-split early-period interactions, stacked on top of A1-lamp ckpt-1000 | **Rounds 1 + 2-B done — both null on test.** Round 1 (bare train, BM25 eval): Δ=+0.003, p=0.93 (adapter `train/checkpoints/user_lora_lamp4_u00000011_seed0/final/`). Round 2-B (BM25 train, BM25 eval): Δ=−0.004, p=0.89 (adapter `train/checkpoints/user_lora_lamp4_u00000011_bm25k4_seed0/final/`). **Round 3 α pinned 2026-06-16, pre-execution**: eval-only redundancy test (no retrain), reuses both User-LoRA adapters; plan at `experiments/2026-06-16-user-lora-round3-alpha-plan.md`. |

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

### LaMP time-based splits (for User-LoRA, ACTIVE)
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

**Next milestone:** User-LoRA experiments on the LaMP time-based split (the
2026-06-02 pivot — see "Project direction update" at the top of this file).
The active research question is Q4: does an additional per-user LoRA on time-
ordered user history meaningfully personalize beyond the Task-LoRA alone?
The A1-lamp ckpt-1000 above is the frozen Task-LoRA foundation; per-user
LoRAs sit on top.

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

### Round 3 — variant α pinned 2026-06-16, pre-execution

Plan: `experiments/2026-06-16-user-lora-round3-alpha-plan.md` (9 design axes
pre-registered via /grill_me; see also memory
`project-user-lora-round3-alpha-design`).

**Hypothesis:** the User-LoRA's contribution is redundant with the
BM25-retrieved profile context at inference. Removing BM25 from the eval
system slot should hurt C2 substantially but hurt User-LoRA conditions less
(or even invert the gap).

**Design:** eval-only on u00000011 / LaMP_4 — no retraining. Reuses Round-1
and Round-2-B User-LoRA adapters as-is plus A1-lamp ckpt-1000.

- **Primary gate (adjudicated):** α-bare vs C2-α paired-t on test ROUGE-1, mean Δ > 0 AND p < 0.05, MDE ≈ +0.05 (identical to Rounds 1+2).
- **Cells:** α-bare, α-B, C2-α, C1-α (all four; 4 cells × {dev, test} = 8 new eval files).
- **Descriptives (pre-reg, not gates):** α-bare vs R1-C2-with-BM25; α-B vs C2-α; α-B vs α-bare; α-bare vs C1-α; C2-α vs C1-α.
- **Honest-null disposition:** matrix-driven Round-4 axis selection (record-level framing / epoch reduction / higher rank, with multi-user replication parked until a positive within-user signal appears).

### Other Round-1-named axes (still parked)

The Round-1 design named several fallbacks beyond B and α. After R2-B's null
and R3-α being the next round, these remain open for future rounds:

- **Fewer epochs** with Pareto sweep — bounds the memorization regime visible in the Round-1 loss curve.
- **Higher rank** (r=8 or 16) — more usable capacity per user.
- **Record-level framing** on LaMP_4 — 241-example per-user train records instead of 1100-example profile-entry framing.
- **Different user(s)** — u00000011 is a power-user pick; a median user makes the bar higher.

The R3-α plan's §"What success / failure would mean" matrix maps observed α
patterns onto these axes, so the Round-4 choice is determined by R3-α's
result rather than picked post-hoc.

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
