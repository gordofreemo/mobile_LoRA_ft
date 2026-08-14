# Research Project — On-Device LLM Training (SmolLM3-3B + LoRA, iPhone 17 Pro)

Two-stage LoRA pipeline (Task-LoRA on LaMP + per-user User-LoRA, stacked at inference).
Phases 1 & 2 ran on the cluster; **Phase 3 is the live work: deploying and characterizing
training on a real iPhone.**

> Compacted 2026-08-14. The previous 152 KB narrative version is recoverable verbatim:
> `git show c230c48:CLAUDE.md`. Per-round detail lives in `experiments/*.md` and in the
> memory files listed in `MEMORY.md`; this file keeps the durable facts, numbers, commands
> and traps. When a round's memory file and this file disagree, **memory is newer**.

---

## Paper story (pivoted 2026-08-10, pinned via `/grill_me`)

The paper is a **systems characterization of on-device LLM training**, in three acts:

1. **Characterization** (h4–h11): memory wall, thermal wall, energy ceiling, scheduling null,
   per-op breakdown.
2. **Kernel fix**: MLX's non-transposed NAX quantized matmul — 1.93x end-to-end on a real
   adapter. Upstream **PR #4051 filed 2026-08-07**.
3. **Demonstration (h12)**: train the Per-Task-LoRA (LaMP-7) *entirely on-device* with NAX ON
   and show benchmark parity with cluster training.

**Per-user personalization is DROPPED from this track** (R5 at-MDE, R6 null, R8 exact
cancellation) — demoted to motivation at most. Phases 1 & 2 stay frozen below as history.

**Deadlines:** ODI (NeurIPS workshop) **2026-08-29 AoE**, 5 pages non-archival — existing draft
`workshop_odi2026/` needs restructure + NeurIPS-2026 template swap. **HotMobile 2026-10-09.**

**Landscape (survey 2026-08-10, memory `project_systems_pivot_landscape_2026-08.md`):** no
"MELT for training" exists; no ≥1B real-task adapter has been phone-trained with
benchmark-verified quality → h12 is a genuine first. Claim-calibration facts (NAX issues
#3362/#3435, PEFT-compute prior art, bitsandbytes backward) are in that memory.

---

## Current state (2026-08-14)

**LIVE: NAX-ON rerun campaign** — recreate every NAX-off on-device figure with the fixed kernel,
verbatim protocols, plus off/on overlays. Plan `experiments/2026-08-11-nax-on-rerun-campaign-plan.md`;
**live state is memory `project_nax_rerun_campaign.md` (authoritative, updated per night)**.

- Done: h11 per-op, pinned-arm discrepancy check, h7 hot+cold, h10 Runs A/B/C + cycling, h8 sweep,
  h9 XS + L energy points, full e2e C0 cost-law figure family.
- Open: **cycling-verdict confirmation run at matched conditions** (sequencer committed c230c48),
  h9 XXL unplugged point, battery-drain S C2, e2e figure refresh as points land.
- Dropped from the campaign for good (user call 2026-08-12): **Low-Power-Mode (C1) and
  game-contention (C4) arms** — never run NAX-off either, feed no figure, do not re-propose.
- **h12 deprioritized behind the campaign** (user call 2026-08-11). Harness + data builder + Mac
  control + MLX→PEFT converter built and committed (`6cc1488`); cluster reference already exists
  (327 steps, LaMP-7 R-1 0.5597). Spec: `experiments/2026-08-10-ondevice-task-adapter-lamp7-h12-plan.md`
  (self-contained, Decisions table settled), state: memory `project_task_adapter_h12_execution.md`.

**Device is now on iOS 26.6 (23G71)**; every NAX-off round ran on 26.5.2 — carry this caveat on
every off/on comparison.

**Deferred / open, not abandoned:**
- **Base-vs-Task-LoRA on-device inference** — the fused 4-bit model it needs already exists
  (`ageyko/SmolLM3-3B-a1lamp-4bit`); swap `modelConfiguration` and reuse the h2/h3 inference rig.
- **Unplugged decode-curve run** (pre-registered in the h1 plan): clean steady-state decode curves
  are unobtainable while plugged, for both 3B and 8B.
- **h5 E2E backlog** (7 of 14 NAX-off runs remain) — largely superseded by the NAX-ON campaign's C0
  point set; C1/C4 are dropped for good.
- **R9 / PT-track** cluster rounds are designed and pinned but not run (and are off the paper's
  critical path after the pivot).
- **MLX upstream:** audit the rest of the quantized `transpose=false` training path (see NAX section).

---

## Phase 3 — findings ledger

One block per round. Numbers here are the quotable ones; raw telemetry paths are given so every
figure is reproducible.

### Inference (h1–h3, closed)

- **SmolLM3-3B-4bit base inference, 2026-06-21** (`experiments/2026-06-21-ondevice-base-inference.md`):
  decode ~37 tok/s (38.8 @64-tok prompt → 32.0 @2048), prefill 620–740 tok/s, cold
  launch→answer ≈1.7 s (load 1362±114 ms + TTFT 380±6 ms), realistic LaMP-3 35.0±1.2 tok/s,
  peak ≈2.2 GB. **Sustained decode throttles −53%** (37.8→17.9 tok/s over 5 min, knee ~90 s)
  while `thermalState` stayed `nominal` the whole time — the enum is useless as a throttle proxy
  at 3B. Telemetry `results/ondevice/bench_metrics_smollm3-4bit-base_2026-06-21.jsonl`.
- **Qwen3-8B-4bit, 2026-06-22** (`experiments/2026-06-22-ondevice-qwen3-8b-inference.md`):
  feasible, no OOM under `increased-memory-limit`. Peak 4.74→5.43 GB, decode ~15.5 tok/s,
  prefill ~240 tok/s, cold ≈2.8 s — all tracking the ~2.7x param ratio. Heat-soaks to `serious`
  during the prefill sweep; sustained decode collapses to ~5 tok/s. **At 8B the thermalState enum
  DOES report the throttle** (opposite of 3B). Harness h3 added `GPU.resetPeakMemory()` per cell
  (before that, `peak_mem_bytes` was a session high-water mark confounded by execution order —
  only the session peak was meaningful) plus `git_commit`/`git_dirty` per record.
- **Capped/bursty stress, 2026-07-03** (`experiments/2026-07-03-ondevice-capped-stress.md`):
  repeated 128-tok generations for 10 min from a cooled device. **3B 38.5→21.0 tok/s (−46%,
  knee ~83 s), 8B 16.0→9.5 tok/s (−40%)**, flat peak 2.11/5.01 GB. Per-query gaps buy a little
  headroom but do not avoid the throttle: the budget a user feels is the plateau, ~half the
  cold-decode rate. 3B stays interactive throttled; 8B marginal (~13.5 s per 128-tok answer).
  Figures `results/ondevice/figures/capped_stress_*_2026-07-03.*`.

### h4 — gradient checkpointing (closed 2026-06-30)

`experiments/2026-06-30-ondevice-training-gc.md`. Naive LoRA FT jetsams on the first backward at
deployment lengths; feasible only to a **256-tok ceiling** (0.41 iter/s, 4.1 GB). Per-block GC
lifts it **256 → 1024 tok (4x)**, zero OOM across the sweep; savings grow with length (−17% @32
→ −41% @256); **GC@1024 (4014 MB) fits in less peak than naive@256 (4128 MB)**; recompute costs
0.78–0.85x iter/s. The bound became thermal, not memory. Implementation: per-block checkpoint via
public MLX `CustomFunction`+`vjp` with LoRA params threaded as explicit differentiable inputs
(NOT the raw `mlx_checkpoint` C binding — `Cmlx` is not a public product). No fork of
`LoraTrain.swift`; a flag on the model drives the stock trainer.

### h5 — E2E per-user training (partially executed, superseded by the campaign)

Plan `experiments/2026-07-03-ondevice-e2e-training-plan.md`, memory `project_e2e_ondevice_training_plan.md`.
Train real top-100 LaMP-3 User-LoRAs to completion (3 epochs, faithful R5 recipe, adapter saved)
and measure cost vs profile size. Recipe: 4-bit SmolLM3 **+ fused A1-lamp Task-LoRA**,
`AdamW(1e-5)` (default wd 0.01 == R5 L2), r=8 q+v α16, batch 1, GC on, `iterations = 3 × n_user`,
cap 1024. Forced deviations: 4-bit not bf16, no dropout, batch 1 not effective-8, fixed LR.
**Sample users:** S/XS=`u00008075`/405, M=`u00005020`/550, L=`u00012502`/987, plus
448=`u00005228`, 500=`u00011077`, 653=`u00013218`. (h9 relabels these XS=405, L=550, XXL=987.)
Conditions C0 ideal / C1 Low-Power-Mode / C2 unplugged / C4 game contention. 7 of 14 planned runs
completed NAX-off; **the NAX-ON campaign has since produced the complete C0 cost-law point set**
(405/448/500/550/987) — token cost law `wall_s ≈ 0.0101·tokens + 222` (r=0.984) vs off `0.0183`,
**1.81x cheaper per token**. Aggregate `results/ondevice_e2e_smollm3_a1lamp_nax-on_2026-08-13.json`,
figures `{cost_law,cost_extrapolation,loss_curves,thermal_trajectory_composite,battery_drain}_nax-on_2026-08-13.*`.

### h6 — background-scheduled training (closed 2026-07-16, negative result)

Memory `project_bg_ondevice_training_plan.md` (full 100 KB blow-by-blow; schema v1→v7 changelog).
**Mystery solved: plain `BGProcessingTask` grants ~2.3 s of real GPU access and then explicitly
revokes it**, regardless of the granted wall-clock window. Control app `ios/BGProbe/`
(`com.geyko.bgprobe`, trivial, no model) got 240 s+ at the *exact same* wake timestamps where
LLMEval got ~9.5 s — so it is not a platform ceiling, it is this app's footprint (model load
peaked ~3.46 GB for a ~1.6 GB 4-bit file) plus GPU revocation. The proper fix
(`BGContinuedProcessingTaskRequest` + GPU entitlement) is **confirmed unsupported on iPhone 17 Pro
/ iOS 26.5.2** via a `supportedResources` check. **True-background GPU training is unachievable on
this hardware; foreground/screen-on is the only working path.** Checkpoint/resume works (LoRA
weights + iteration counter every 10 iterations); AdamW moments reset each wake because
`MLXOptimizers.AdamW`'s m/v are `internal` with no accessor.
API facts worth keeping if this is ever revisited: the property is `requiresExternalPower` (not
`…Connected`); SwiftUI's `.backgroundTask` modifier has no `.processing` case, so registration must
use `BGTaskScheduler.register(forTaskWithIdentifier:using:launchHandler:)` from `App.init()`; and
`runBGTrainWake()` re-arms the next request as its first action, so wiping on-disk state does not
stop the chain — cancel it (`--bg-train-cancel`).

### h7 — per-iteration token-time cost model (closed 2026-07-25)

Plan+results `experiments/2026-07-24-ondevice-tokentime-plan.md`, memory `project_tokentime_plan.md`.
**Cost is `f(tokens, thermal_history)`, not `f(tokens)`.** Grid 50…1000 tok, same h5 recipe.
- HOT (sustained, no cooldown): `s/iter ≈ -1.957 + 0.02154·tokens` (r=0.992) ← **use this for E2E
  predictions**.
- COLD (cooldown-to-nominal between cells): `s/iter ≈ -0.678 + 0.01158·tokens` (r=0.993), genuinely
  clean only ≲300–650 tok.
- At 500 tok HOT ≈ 1.7x COLD. Secondary finding: the shared h3/h4 `cooldownCapSeconds=120` is
  insufficient once cell duration scales with tokens; a dedicated 300 s cap helped (clean-nominal
  range 200→300 tok) but cells ≥700 tok never reach `nominal` even then.
- **NAX-ON rerun:** HOT `-1.476 + 13.52 ms/tok` (r=0.982), COLD `-0.489 + 7.15 ms/tok` (r=0.987)
  → 1.59x / 1.62x slope drop. Quotable: **hot-with-fix ≈ cold-without-fix** (curves overlap).

### h8 — GC granularity sweep (closed 2026-07-26)

`experiments/2026-07-26-ondevice-gc-granularity-h8.md`, memory `project_granularity_plan.md`.
K = consecutive blocks per checkpoint boundary, all 9 divisors of 36, cap 1024, 100 steps/cell.
**K=1 (per-block) is Pareto-best.** Peak memory rises monotonically 4126→5238 MB (+27%) K=1→6 —
the robust finding. Raw run-mean throughput looked like a clean −14% decline but that is mostly a
**within-run thermal-drift confound** (user pushback, correct): grouped by matched `thermal_state`
there is no trend at nominal/fair and only ~12% K=1→3 plateauing at serious. **K≥9 jetsams
outright** (hard OOM wall between 6 and 9). NAX-ON rerun: throughput lifted uniformly ~1.6x
(K=1 0.098→0.155 iter/s), **peak memory identical to the off round** (kernel changes dispatch, not
allocation), OOM wall unchanged on iOS 26.6.
This round also found the **`loraLayers = 28` bug** (SmolLM3-3B has 36 blocks and
`LoRAContainer.from` takes a *suffix*, so h1–h7 trained only the last 28) — fixed for h8 onward
(h10/h11 use 36); h1–h7 left as-is, h7's NAX rerun deliberately kept 28 for bug-compatibility.

### h9 — energy (closed 2026-07-29)

`experiments/2026-07-29-ondevice-energy-h9.md`, memory `project_energy_h9_plan.md`.
Method: unplugged (C2) `%drain × 3998 mAh × 3.87 V` (full battery ≈ **55,700 J**; capacity from
device model `MG8N4ZD/A`), minus a paired idle baseline's average power × duration.
- NAX-off: XS(405) **28,443 J ≈ 51%** of a battery; L(550) **48,398 J ≈ 87%**; XXL(987) **died at
  1460/2961 iterations (49.3%)** having consumed 50,398 J ≈ 90.5% — silent OS force-shutdown, no
  adapter saved. **One-charge ceiling sits between 550 and 987 examples**, and cost scales faster
  than iteration count.
- NAX-ON: XS **17,667 J = 31.7%** of a charge, 1.94 h, 2.79 W avg; L **35,590 J = 63.9%**, 3.62 h,
  **3.00 W avg vs off 3.06 — the sustained power envelope is unchanged, so time savings convert
  directly to energy savings** (confirms h10's power-capped reasoning). Unplugged speedups (XS
  1.20x, L 1.33x) are consistently *below* plugged per-op (1.5–1.9x) — candidate causes: battery
  clock caps, iOS 26.6 confound. Flag, do not overclaim.
- **Protocol rule: start energy runs from ≤85% charge, never 100%** — a 100%-start run sits on a
  fuel-gauge plateau that under-reads drain (demonstrated twice; the voided 100%-start XS attempt
  is kept as the citable demonstration).

### h10 — thermal cooldown, duty cycling, pacing (closed 2026-07-30)

Plan+results `experiments/2026-07-28-ondevice-thermal-cooldown-h10-plan.md`, memory
`project_thermal_cooldown_h10_plan.md`. Six arms.
- **Recovery is fast**: after a 60-min soak R = 2.146x, t50/t90/t95 = 180/327/346 s; after a
  10-min soak t50/t95 = 82/118 s.
- **60-min bursts are a wash**: best schedule 1.017 / 1.028 / 1.073x across three runs — the
  cold-start bonus (+13.2% over a 60-min burst) evaporates in ~2 min.
- **Run C (10-min soak) suggested 1.25x, and the sustained cycling arm (h10c, 10-on/2-off ×6)
  REFUTED it: 0.755x — cycling is ~25% *worse* than continuous.** Bursts settle at 10.7–11.1 s/iter,
  hotter than continuous training's own 9.98 plateau; restart overhead measured at 1–2 s/burst, far
  too small to explain it. Run C measured the one burst that starts from a genuinely cool chassis.
- **Self-limiting/pacing (h10d)**: inserting a delay after every iteration inside one continuous
  train call converges to 0.877x (12.3% throughput loss). Measured exchange rate
  **dc/dD = −0.303 s of compute bought per s of delay, against the −1 pacing needs**; throttle
  response is continuous (no governor steps >3%); equilibrium is path-independent (no hysteresis).
- **`thermalState` releases ~59 min LATE** after a long burst (~28 min after a short one) — gating
  cooldowns on `nominal` (h3/h4/h7/h8 all did) burns an hour for nothing.
- **Conclusion (NAX-off): no workload-scheduling strategy pays at any granularity.** Root cause:
  training already sits at the device's sustained dissipation envelope (~3.05 W measured vs a
  published 3–5 W), so there is no thermal headroom to reclaim. Platform framing: iOS exposes no
  DVFS API, so *when* to run is an app's only lever and it does not pay.
- **NAX-ON rerun flips the cycling verdict.** Thermal *structure* is kernel-invariant (R = 2.172x,
  recovery 164/312/344 s, Run C t50 = 82 s identical), only the level scales (plateau 9.98→6.505,
  cold ref 4.655→2.995). But the cycling arm now **settles cooler than continuous (5.13 vs 6.72
  s/iter) and wins: 0.1644 iter/s = 1.105x continuous, where it lost by 25% off.** So "no schedule
  pays" is **kernel-dependent**. Caveats before quoting: n=1, ambient uncontrolled (cycling ran
  04:49–05:59 vs the off round's 22:06), continuous reference is the A+B plateau mean. **A matched
  confirmation run is queued and required before the paper claims this.**

### h11 — per-op / per-phase iteration breakdown (closed 2026-08-06)

`experiments/2026-08-06-ondevice-perop-h11.md`, memory `project_perop_h11_plan.md`. Explicitly
descriptive, nothing pre-registered. Tier 1 = 6-phase eval-barrier decomposition; Tier 2 =
`GPU.startCapture` per-phase `.gputrace` read in Xcode's shader profiler.

**Tier 1 (12 cells, 264 iterations, zero errors):** backward **~78%** of an iteration at ≥250 tok,
forward ~21%, everything else <1%. `backward/forward = 3.5–3.9x`, above the textbook ~3x.
**Shares are thermally invariant** (≲1 pp cool-vs-hot at matched tokens) while absolute cost rises
~25%. `readback` is 0.000 s in all cells. `graph_build` flat ~0.031–0.038 s (pure CPU). Barrier
overhead Σphases/fused = 1.02–1.08. Barriered peak memory is *lower* than fused (0.87–0.985x).
Cold-ref 4.702 s/iter @500 tok reproduces h10's 4.609–4.710 — a cross-round rig check.
Data `results/ondevice/train_bench_metrics_perop_2026-08-04.jsonl`.

**Tier 2 kernel tables (all three phases ~100% transcribed):**
- Forward: frozen 4-bit base weights `affine_qmm_t_*` **86.3%**, LoRA `steel_gemm_*` **3.2%**,
  elementwise+copies 7.6%, SiLU 1.8%, attention+norm+RoPE 0.9% (**attention is negligible — the
  model is weight-bound at these lengths**), loss 0.25%.
- Backward: quantized matmul **89.95%**, elementwise 6.46%, LoRA gemm 2.21%, attn+norm+RoPE 0.80%.
  Captured as top-K blocks; category totals stable across K (90.59% @K=4 vs 89.95% @K=12), and a
  two-point fit separates the fixed lm_head term: `qmm_n/qmm_t = 25.9/K + 4.95` → at K=36,
  qmm_n ~76.5% / qmm_t ~13.5%, i.e. **GC recompute is ~13–14% of backward, not the 7.2% a K=4
  sample naively reads**.
- Optimizer: pure elementwise, no matmul (75.7% elementwise, 18.8% sqrt/square, 5.5% scalar).
- **SIMD-group count and cost rank disagree sharply** (`vvn_Multiply` runs 11x the groups of the
  dominant `affine_qmm_t` for 4.79% of cost) — optimizing by op count targets the wrong thing.
- **The number Tier 2 exists for:** adapter-attributable compute @250 tok = forward gemms 0.0209 s
  + backward gemms 0.0617 s + the entire optimizer phase 0.0662 s = **0.1488 s of a 3.5523 s
  iteration = 4.19%. ~95.8% of an on-device LoRA training iteration is the frozen base model.**
- **Reading for the paper: PEFT saves memory and storage, NOT compute.** Freezing 99% of parameters
  removes almost none of the work; the levers are the base-weight matmul path and the recompute.
- Unplanned finding that led straight to the NAX round: forward uses the **tiled**
  `affine_qmm_t_nax_...`, backward the **untiled** `affine_qmm_n_*`, costing 2.3x more per SIMD
  group.
- Kernel data `results/ondevice/perop_kernels_2026-08-04.json`; `.gputrace` bundles (~20 GB)
  deleted after transcription (re-capture is ~90 s).
- **NAX-ON rerun:** backward share **78% → 65–67%**, forward ~21% → ~32%, hot-500 fused
  8.96 → 5.15 s (1.74x). Data `..._perop_nax-on_2026-08-11.jsonl`.

### NAX kernel fix — MLX non-transposed quantized matmul (2026-08-06/07)

`experiments/2026-08-06-mlx-nax-qmm-n-backward.md`, upstream handoff
`experiments/2026-08-07-mlx-upstream-pr-handoff.md`, memory `project_mlx_nax_backward_patch.md`.
**PR #4051 filed 2026-08-07.**

**Premise:** MLX gates NAX on `transpose == true` (`quantized.cpp:694`), so backward's `dX` falls
to a generic 32×32-tiled kernel while `affine_qmm_n_nax` sits compiled and unreachable.
`QuantizedLinear` always computes `x·Wᵀ`, so `transpose=false` is reachable essentially only from a
backward pass — **a training-only path**. Every NAX quantized bug filed to date is an inference
path (#3925, #3887, #3797); a tracker search for `qmm_n` returns nothing.

**Two independent upstream defects, both in one function (= code that was never run):**
1. **Weight addressing uses the transposed layout** — `qmm_n_nax_tgp_impl` was copy-adapted from
   `qmm_t_nax` and never converted (`wl += y_col*K_w`, `scales += y_col*K_g`, leading dim `K`).
   Fixing 3 offsets + the leading dim makes it **bit-exact** (rel. error 1.5–7826 → 0.0). Also
   repairs the MoE path (`affine_gather_qmm_n_nax` shares the impl).
2. **No partial-M-tile handling** (`(void)M`, no `load_safe`/`store_safe`) — unaligned M **writes y
   out of bounds** (~49 KB at M=250) while still returning correct values in the valid region by
   luck. Fixed by porting `sgp_sm` + `dispatch_bool` + safe load/store from `qmm_t_nax`; verified
   across M ∈ {1,2,31,…,1025} incl. the 33–63 range where `sgp_sm` goes negative. **M is now
   unconstrained; the guard retains only `N % 64 == 0`** (structural for group_size ≥ 64).

**Speedups.** Per-op paired A/B at h11's exact grid: backward 1.65–2.13x (mean 45% removed), whole
iteration 1.45–1.68x (mean 1.55x). **E2E (the decisive arm, one full User-LoRA `u00008075`, 1215
iterations): 2.906 h → 1.503 h = 1.93x, same loss (0.8829 vs 0.8813, −0.18%), loss correlation
0.9999 over 1210 steps.** Quote **1.93x** as the realistic sustained figure; report per-op numbers
as the mechanism decomposition, not a competing estimate.
The **1.93x-vs-1.55x discrepancy**: the leading hypothesis (per-iteration arm alternation in the
per-op design) was tested with a pinned-arm run and **REFUTED** — pinned gives 1.52x, agreeing with
alternating. The gap must come from the long-session regime / 28-vs-36 loraLayers / real-data
composition; still open. The pinned run also **closed the "unexplained ~3% forward residual"**:
the forward control flips sign between designs (+5.7% pinned vs −3% alternating), so it is a
design artifact, not a kernel effect.
Data: `results/ondevice/train_bench_metrics_naxab*{,_h11grid,_e2e,_pinned}_*.jsonl`,
analysis `eval/naxab_aggregate.py`, `eval/plot_naxab_e2e.py`.

**Traps this round taught (each silently produces a FALSE NULL):**
- **The NAX kernels are JIT-compiled at runtime from `Source/Cmlx/mlx-generated/quantized_nax.cpp`** —
  editing `mlx/backend/metal/kernels/quantized_nax.h` does nothing in the mlx-swift build.
  **Upstream it is the reverse**: `kernels/quantized_nax.h` is the source of truth there.
- **SPM local-override identity**: the vendored directory must be named exactly `mlx-swift`
  (SPM matches local packages by directory basename against the remote URL tail), or SPM silently
  fetches unpatched MLX for `mlx-swift-lm`. Point both `mlx-swift-lm-local/Package.swift` and
  `mlx-swift-examples/Package.swift` at `.package(path:)`.
- **M = tokens − 1** (`LoRABatchIterator` slices inputs `[:, :-1]`), so an alignment grid must be
  `{65,129,257,…}`; every iteration records `seq_len`/`seq_len_aligned_64` so this is verified.
- **Vendor before building** — patch-in-place-then-vendor pays the long `Cmlx` rebuild twice.
- **Test-data conditioning nearly produced a false positive**: `sin(row*a + col*b)` test matrices
  make every row a smooth sinusoid, products cancel, the reference norm collapses and *every*
  relative error inflates (the known-good generic kernel scored 0.99). Use a two-stage fractional
  hash and log reference norms.
- **Paired A/B needs a seeded batch iterator** (`LoRATrain.shuffleSeed`, local opt-in, default nil).
  MLX's `LoRABatchIterator` otherwise uses Swift's unseeded system RNG, so the two arms consume
  different batch orders and the comparison becomes unfalsifiable.
- **Symmetric thermal protocol is necessary, not padding**: 60 min idle before *each* arm, baseline
  first so residual heat works against the patched arm. Both arms then start `nominal`.
- Device is left **SAFE**: `enable_nax_n()` defaults to 0, dispatch identical to stock upstream.
  Patch documented in `ios/mlx-swift/{VENDORED,LOCAL_PATCHES}.md`.

**Open upstream follow-ups:** audit the rest of the quantized training path (if `transpose=false`
is systematically untested, two defects in one kernel predicts more — check `gather_qmm_n_nax`,
`fp_quantized_nax`, K-tail handling); this could turn one bug into a finding about the class.
Note NAX cannot be validated on the M3 Mac — all validation was on the A19 Pro phone.

### h12 — on-device Task-LoRA (LaMP-7), the demonstration arm

Spec `experiments/2026-08-10-ondevice-task-adapter-lamp7-h12-plan.md` (implement from it alone;
Decisions table settled). Our A1-lamp recipe verbatim (not OPPU's); masked loss via Mac-side
pre-tokenized ids + assistant-mask span (**prompt-prefix masking is documented-broken for SmolLM3
in `train.py`**); effective batch 32 by accumulation, wd 0.0, clip 1.0; no gates, bare single-shot,
smoke first. Three descriptive arms: cluster reference (`pt_lamp7_1ep.json`, exists), Mac control
on the exact MLX 4-bit model, device overnight run (~7.3 h, 326 optimizer steps; corpus 10,437
examples, mean ~212 tok, 0% over the 1024 cap).

---

## Phase 3 — runbook

### Runtime decision (verified against official sources only)

**Runtime = MLX** (`mlx-swift` on device, `mlx-lm` on the Mac). llama.cpp / ExecuTorch rejected.
SmolLM3 is first-class in MLX (`mlx_lm/models/smollm3.py`, `Libraries/MLXLLM/Models/SmolLM3.swift`).
Apple Foundation Models' adapter toolkit is Apple-model-only (rank-32 LoRA bound to the OS model) —
cannot adapt SmolLM3. MLX is also the credible on-device *training* route (`mlx_lm.lora`, the
`LoRATrainingExample` app, Apple paper arXiv:2510.03425).

### Host / device / signing

- Mac: Apple **M3, 16 GB**. Xcode 26.5 at `/Applications/Xcode.app`, but the active dev dir is
  CommandLineTools — **prefix every Xcode/devicectl command** with
  `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- Signing: **Apple Development: andrew.geyko@icloud.com**, team **`JGW9U9Y36Y`** (free personal
  team; bundle IDs auto-disambiguated via `DISAMBIGUATOR=${DEVELOPMENT_TEAM}` in
  `Configuration/Build.xcconfig`).
- Device: **iPhone 17 Pro** (`iPhone18,1`), **iOS 26.6 (23G71)**, Developer Mode on.
  UDID **`00008150-000674C60A3B401C`** (also appears as `61A5D517-E8C8-5701-85BA-7515E5EA3550`).
  Apps: **`mlx.LLMEvalJGW9U9Y36Y`** (main) and `com.geyko.bgprobe` (BGProbe control, h6).
  List devices with `xcrun devicectl list devices`.
- **Model delivery = HF download on-device**: the app pulls its `modelConfiguration` repo (~1.73 GB
  for the 4-bit 3B, ~4.3 GB for 8B) into the sandbox on first generation, over Wi-Fi, once. It
  survives install-over, not uninstall.

### Mac-side MLX toolchain

venv **`.venv-mlx/`** (Python **3.11** — 3.14 has no MLX wheels), `mlx-lm` (mlx 0.31.2), gitignored.
```
.venv-mlx/bin/python -m mlx_lm convert --hf-path HuggingFaceTB/SmolLM3-3B \
  --mlx-path data/models/SmolLM3-3B-mlx-4bit -q --q-bits 4
.venv-mlx/bin/python -m mlx_lm generate --model data/models/SmolLM3-3B-mlx-4bit --prompt "..." --max-tokens 60
```
SmolLM3 has **thinking mode on by default** (emits `<think>…</think>`).

### iOS app — vendoring layout

- `ios/mlx-swift-examples/` — vendored via `git subtree` (upstream base `378f244`). Edit + commit
  normally. Bump: `git subtree pull --prefix=ios/mlx-swift-examples <url> <tag> --squash`.
- `ios/mlx-swift-lm-local/` — **local SPM override** of `ml-explore/mlx-swift-lm` (converted
  2026-06-30 for GC). Edits: `Models/SmolLM3.swift` (`checkpointGroupSize`), `LoraTrain.swift`
  (`LoRATrain.shuffleSeed`), `Load.swift` (`weights.removeAll()` before `eval(model)`).
- `ios/mlx-swift/` — **local SPM override of mlx-swift carrying the NAX patch.** Directory name
  must stay exactly `mlx-swift` (see SPM identity trap above).
- Harness code: `Applications/LLMEval/{ViewModels/LLMEvaluator.swift,Benchmark/*}`,
  `LLMEval-Info-Additions.plist` (carries `UIBackgroundModes`, `BGTaskSchedulerPermittedIdentifiers`,
  `MetalCaptureEnabled` — `INFOPLIST_KEY_*` synthesis cannot express array-valued keys).

**Build (device, signed)** — `-skipMacroValidation` is **required** (else it fails on
`MLXHuggingFaceMacros … must be enabled`):
```
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cd ios/mlx-swift-examples
xcodebuild -project mlx-swift-examples.xcodeproj -scheme LLMEval \
  -configuration Debug -destination 'id=00008150-000674C60A3B401C' \
  -derivedDataPath ./build -allowProvisioningUpdates -skipMacroValidation \
  DEVELOPMENT_TEAM=JGW9U9Y36Y build
```
**Install + launch (install-over, NEVER uninstall):**
```
xcrun devicectl device install app --device 00008150-000674C60A3B401C \
  build/Build/Products/Debug-iphoneos/LLMEval.app
xcrun devicectl device process launch --device 00008150-000674C60A3B401C mlx.LLMEvalJGW9U9Y36Y <args>
```
**Pull telemetry** (no live console: macOS `log stream` has no `--device`, `log collect --device`
needs root, `idevicesyslog` is not installed):
```
xcrun devicectl device copy from --device 00008150-000674C60A3B401C \
  --domain-type appDataContainer --domain-identifier mlx.LLMEvalJGW9U9Y36Y \
  --source Documents/<file>.jsonl --destination /tmp/devpull/<file>.jsonl
```

### Harness conventions

- `app_build` string per mode (`smollm3-ondevice-train-perop-h11`, `...-thermal-cooldown-h10`,
  `...-granularity`, `...-tokentime-h7`, `...-bg-h6`, `qwen3-8b-ondevice-bench-h3`, …). **Bump it
  whenever harness logic changes**; schema version bumps are separate and per-mode.
- **One JSONL per mode** in `Documents/` (`train_bench_metrics_{e2e,gc,granularity,thermal,
  selflimit,perop,naxab*,e2e_bg}.jsonl`) so an in-flight round's file is never touched. Dated pulls
  land in `results/ondevice/`; aggregates in `results/ondevice_*.json`.
- **`--nax-arm on|off` is honored globally in every bench mode** (commit `d75d940`): records carry
  `nax_arm`, `app_build` is suffixed `-nax-<arm>`, and JSONLs route to `_nax-<arm>` siblings
  (aggregators summarize whole files, so separation prevents kernel-mixing). `--pin-arms` keeps the
  arm constant per sub-block for A/B.
- Every record carries `git_commit`/`git_dirty`, thermal state, memory, and a passive sampler
  (10–30 s cadence depending on mode).
- Aggregators/plots (all in `eval/`, **note `.gitignore` has `eval/*` — use `git add -f`**):
  `bench_aggregate.py`, `e2e_aggregate.py`, `train_tokentime_aggregate.py`,
  `train_granularity_aggregate.py`, `thermal_aggregate.py`, `perop_aggregate.py`,
  `naxab_aggregate.py`, `bg_progress.py`, `bg_timeslice.py`, `dedupe_gputrace.py`,
  `plot_{thermal,granularity,energy,perop,naxab_e2e}.py`,
  `plot_{perop,tokentime,granularity}_overlay.py`.
- Sequencing scripts live in `scripts/` (`nax_rerun_night*.sh`, `run_granularity_sweep.sh`).

**Launch args by round** (all passed to `devicectl device process launch`):

| Round | Args |
|---|---|
| h2 inference | `--benchmark` (cold+prefill+decode), `--benchmark-tail` (realistic + 5-min stress), `--benchmark-cold` |
| h4 capped stress | `--benchmark-stress-capped` |
| h5 E2E | `--user <uid> --condition <C0\|C1\|C2\|C4>` |
| h6 background | `--bg-train-submit --user U --condition C`, `--bg-train-resubmit` (keeps checkpoint + cap origin), `--bg-train-cancel`, `--bg-train-validation` |
| h7 token-time | `--benchmark-train-tokentime`, `--benchmark-train-tokentime-cold` |
| h8 granularity | `--benchmark-train-granularity --granularity-k <K>` |
| h9 energy | `--benchmark-train-idle-baseline` (+ h5 args, `--condition C2`) |
| h10 thermal | `--benchmark-thermal-cooldown --soak-minutes M --probe-interval-s S`; `--benchmark-thermal-cycle --burst-minutes 10 --rest-seconds 120 --cycles 6`; `--benchmark-thermal-selflimit [--selflimit-delay D --selflimit-minutes M]` |
| h11 per-op | `--benchmark-train-perop [--idle-minutes N]`; `--benchmark-train-perop-capture [--capture-tokens N] [--capture-backward-layers K]` |
| NAX A/B | `--benchmark-nax-ab [--pin-arms]`; `--nax-arm on\|off` (global, every mode) |

**Canonical NAX-off data files** (the campaign's overlays compare against these):

| Round | Telemetry / aggregate |
|---|---|
| Inference 3B | `results/ondevice/bench_metrics_smollm3-4bit-base_2026-06-21.jsonl` → `results/ondevice_base_smollm3_4bit_2026-06-21.json` |
| Inference 8B | `bench_metrics_qwen3-8b-4bit-base_2026-06-22.jsonl` → `results/ondevice_base_qwen3_8b_4bit_2026-06-22.json` |
| Capped stress | `bench_metrics_{smollm3-4bit,qwen3-8b-4bit}-stresscap_2026-07-03.jsonl` → `results/ondevice_stresscap_*.json` |
| h7 | **hot = `results/ondevice_tokentime_smollm3_4bit_*_fine50.json`, cold = `*_cold_v4.json`** — several other aggregates exist and fit different slopes; these two are canonical |
| h8 | `train_bench_metrics_granularity_2026-07-26.jsonl`, figures `figures/granularity_2026-07-26.*` |
| h9 / E2E | `train_bench_metrics_e2e_smollm3_a1lamp_2026-07-29.jsonl` → `results/ondevice_e2e_smollm3_a1lamp_2026-07-29.json` (energy block is schema-v3-filtered) |
| h10 | `train_bench_metrics_thermal_2026-07-28.jsonl` → `results/ondevice_thermal_smollm3_4bit_2026-07-28.json` |
| h11 | `train_bench_metrics_perop_2026-08-04.jsonl` → `results/ondevice_perop_smollm3_4bit_2026-08-04.json` + `perop_kernels_2026-08-04.json` |
| NAX A/B | `train_bench_metrics_naxab{_h11grid,}_2026-08-06.jsonl`, `..._naxab_e2e_2026-08-07.jsonl` |

---

## Operational traps (each of these has already cost hours)

**Device / launch**
- A **resident app silently absorbs new launch args** — an idle instance makes a launch a no-op.
  **Kill any resident PID before every launch** (`devicectl device process signal --signal SIGKILL`).
- `devicectl … process launch --console` **propagates SIGTERM to the app** — killing the monitor
  kills training (`signal 15`). Launch detached and poll the device JSONL instead.
- Run long sequencers under `nohup` + `caffeinate -ims`, fully detached: **harness-tracked
  background tasks have been killed mid-experiment twice.**
- Device locked → `FBSOpenApplicationErrorDomain error 7`; `devicectl list devices` showing
  `unavailable` usually means asleep. Ask the user to unlock. Set Auto-Lock to Never for runs.
- A wedged devicectl tunnel stuck "connecting" is fixed by **killing user-level `remotepairingd`**
  (no sudo). A fully dropped wireless connection needs a fresh USB reconnect.
- **Never `devicectl device uninstall`** — it wipes the ~1.7 GB cached model *and* side-loaded
  `Documents/user_data/*.jsonl`, and resets the per-app trust record (requires a manual
  Settings ▸ VPN & Device Management tap). Install-over is always the right move. After any
  uninstall, re-warm the model cache with a foreground launch before trusting anything else.
- 8B weights get evicted by reinstalls; the next launch re-downloads ~4.3 GB and can appear hung —
  SIGKILL and relaunch.
- **"Plugged" is not guaranteed**: a 448-example C0 run drained 100→25% on Mac USB power. Check the
  charger source for plugged runs.

**Metal GPU capture (h11 Tier 2)**
- `MTLCaptureManager.stopCapture()` **finalises asynchronously** — starting the next capture too
  soon fails with `Already capturing`, an mlx-c fatal that kills the process, and the symptom points
  at a wedged device. Poll `isCapturing` (`awaitCaptureIdle()`); do not substitute a fixed sleep.
- **Never SIGKILL a capture-mode process** — it wedges the capture daemon persistently (survives
  device reboot; quitting Xcode releases it).
- `devicectl` **cannot read symlinks** (ELOOP, aborts the whole transfer) and Metal fills bundles
  with them. **Hard-link** on device (not copy — links point inside the same bundle, copying
  re-expands duplicates and blows past a 3 GB cap), pull, then re-collapse with
  `eval/dedupe_gputrace.py`.
- Deduping a bundle whose transfer is still running **silently corrupts it** (truncated files hash
  equal); the script now refuses to run on an unsettled directory.
- The replay ceiling is on **resource count, not bytes** (~1900–2200 files): a *smaller* backward
  trace fails where a *larger* forward one replays. Shrinking tokens never helps because backward's
  file count barely moves with sequence length — use `--capture-backward-layers K` instead.
- `MetalCaptureEnabled` **is** honoured under a `devicectl` launch. `GPU+Metal.swift`'s doc comment
  claiming `MLX_METAL_DEBUG` is required is stale.

**Measurement validity**
- **Check matched-thermal-state slices before publishing any run-mean trend** — within-run drift
  manufactures trends (h8). Memory `feedback_thermal_drift_confound.md`.
- **Phases must reach equilibrium**: the device needs 40–50 min to plateau; 20-min phases are ramps
  and ramps systematically *flatter* whatever intervention is being tested (h10d pilot).
- **Verify schedules by running them**: single-burst extrapolation predicted 1.25x, sustained
  cycling measured 0.755x (h10).
- **Score an arm against the right baseline, or it reports a false null**: a 10-min soak never
  reaches continuous training's throttled steady state, so scored against its *own* plateau (correct
  for the 60-min runs) h10 Run C read −0.1% instead of +50.6%. Fixed by `apply_reference_plateau()` —
  short-soak sessions borrow only the long-soak continuous baseline, keeping their own burst work.
- The **cold-reference probe is necessary but not sufficient** as a cross-run check — it measures
  die temperature, so three runs agreed within 2% while the bursts they preceded differed 5.4% in
  work. Report idle history alongside it.
- **`ProcessInfo.thermalState` is not a throttle proxy**: it lied `nominal` through a −53% decode
  throttle, reads `serious` from the first bucket of every training session, and releases ~59 min
  *late* after a burst. Do not gate on it.
- MLX fuses a whole training iteration into one lazy `eval` — per-phase timing needs explicit
  barriers. MLX dense/quantized kernels are value-independent, so timings do not depend on which
  weights are resident (which is why h11's no-op weight-snapshot bug did not affect timings; it was
  removed and fidelity is instead read as loss *continuity* across the mode boundary).
- `Module.update`'s leaf case swaps the handle *inside* an existing `MLXArray`, so
  `trainableParameters()` returns **aliases** of the live arrays — a "snapshot" taken that way
  tracks training and restores nothing. A genuine reset needs a deep copy.

---

## Phases 1 & 2 — frozen state

### Research questions

| Q | Status |
|---|---|
| **Q1** — does fine-tuning on LaMP help a 3B model at all? | **YES** — A1-lamp ckpt-1000: +0.11 / +0.07 / +0.13 on LaMP-3/4/7 test over the BM25 baseline. |
| Q2 — synthetic preference-conditional data | DROPPED (2026-06-02 pivot). |
| Q3 — general vs domain-specific Task-LoRA | DROPPED (2026-06-02 pivot). |
| **Q4** — per-user LoRA beyond Task-LoRA? | **YES for LaMP-3** (R5: ΔMAE −0.050, acc 0.680→0.730, RMSE 0.616→0.575, at MDE p≈0.10). **Null on LaMP-4** (R6: ΔR-1 +0.007, p=0.20). **Does not survive a base-adapter swap** (R8). |
| **Q5** — does the 3B two-LoRA stack survive a scale comparator? | **YES** — beats Llama-3.1-70B-Instruct + BM25 on all 7 LaMP tasks. |

### Phase 1 headline numbers (LaMP test, seed 0, greedy, BM25 k=4)

| Task | No-profile floor | Profile baseline | A1-lamp (ckpt-1000) | Δ |
|---|---|---|---|---|
| LaMP-3 (acc) | 0.4508 | 0.6964 | **0.8056** | +0.109 |
| LaMP-4 (rouge1) | 0.1393 | 0.1537 | **0.2259** | +0.072 |
| LaMP-7 (rouge1) | 0.4170 | 0.4372 | **0.5619** | +0.125 |
| BFCL AST overall | — | **0.8078** (base) | **0.7696** | −0.038 |

Results: `results/LaMP_{3,4,7}_test_a1_lamp_1ep_seed0_checkpoint-1000_bm25k4_seed0.{json,predictions.jsonl}`
(+ `_dev_*`), `results/bfcl_ast_*`. Test-split correction and the Pareto sweep narrative were
`experiments/2026-06-13-lamp-test-split-correction.md` and `2026-06-02-a1-lamp-1ep-pareto.md` —
**neither is in this working tree** (see the note under Round history).

### Canonical artifacts

- **A1-lamp Task-LoRA:** `train/checkpoints/a1_lamp_1ep_seed0/checkpoint-1000/` (frozen 2026-06-02).
  The 2-epoch `a1_lamp_seed0/` is Pareto-dominated, kept for provenance. `checkpoint-400` is the
  alternative if maximum BFCL retention dominates.
- **One-LoRA FT** (7-task Task-LoRA, R7): `train/checkpoints/a2_lamp_1ep_seed0/final/`.
  **Always called "One-LoRA FT" in prose, never "A2-lamp"** (explicit user preference).
- **100 LaMP-3 User-LoRAs (R5):** `train/checkpoints/user_lora_lamp3_<fp>_seed0/final/`,
  pool `data/lamp_user_stats/LaMP_3_top100_users.json`.
- **100 LaMP-4 User-LoRAs (R6):** `train/checkpoints/user_lora_lamp4_<fp>_oppu_seed0/final/`,
  pool `data/lamp_user_stats/LaMP_4_top100_users.json`.
- Training configs: `train/config/{a1_lamp,a1_lamp_1ep,a2_lamp_1ep,user_lora_*,pt_lamp7_1ep}.json`.
  R7 corpus `data/lamp_train_mixed7_bm25k4.jsonl`; R7 BFCL result
  `results/bfcl_ast_a2_lamp_1ep_seed0_final_seed0.json`.
- **Fused 4-bit device model:** HF `ageyko/SmolLM3-3B-a1lamp-4bit` (base: `mlx-community/SmolLM3-3B-4bit`).

### Round history

**Warning:** the `experiments/*.md` docs for R7, R8, R9, PT1, LL1, the Pareto sweep, the test-split
correction and the per-user-count analysis are **not present in this working tree** (nor are the
`project_user_lora_*` memories they cite). These lines, plus `results/*.json`, are the surviving
record — do not compress them further without re-creating the docs.

- **R1–R4** (single-user u00000011, LaMP-4): all failed pre-registered test gates, with a
  consistent dev/test asymmetry (dev Δ +0.030/+0.043/+0.047/+0.040 vs test +0.003/−0.004/+0.010/−0.018).
- **R5** (LaMP-3, K=100, OPPU recipe stacked on A1-lamp ckpt-1000): confirmed Q4 at MDE — ΔMAE
  −0.050, acc 0.680→0.730, 7/91/2 win/tie/loss, zero inference overhead.
- **R6** (LaMP-4, K=100, 2026-07-07, `experiments/2026-07-07-user-lora-lamp4-round6-multi.md`):
  mean R-1 0.235→0.242 (+0.007), **not significant** (paired-t p=0.20, Wilcoxon p=0.30, CI spans
  zero) — expected per the plan's own priors (OPPU's own LaMP-4 lift is +0.003).
- **R7** (LaMP coverage 3→7 tasks, One-LoRA FT, 2026-07-15): one shared Task-LoRA on a 72,062-example
  7-task corpus (1 epoch, 2250 steps, ~4h23m, loss 1.86→0.89). Beats BM25-only on **every** task;
  biggest lifts on citation ID, title generation and tweets (+0.12–0.14); the original 3 tasks are
  unchanged vs A1-lamp. **BFCL regresses hard: 0.633** (base 0.808, A1-lamp 0.767) — concentrated in
  multi-call categories (`multiple` 0.86→0.50, `parallel_multiple` 0.785→0.58). A checkpoint sweep
  (200/1000/1800) ruled out both overfitting and data-ordering: the collapse is sharp between step
  200 and 1000 and is a **format-fidelity collapse** — unparseable bare-dict output instead of
  `<tool_call>{…}</tool_call>` jumps 0.7%→15.3%→21.7%, i.e. SmolLM3's pretrained formatting habit
  resurfacing as the adapter's grip erodes. Training loss across the window is flat.
  **Decision 2026-07-16: One-LoRA FT stays canonical despite this** — a knowing trade-off, flagged
  before it was made.
- **R8** (LaMP-3 re-run on One-LoRA FT, 2026-07-17): **the personalization lift does not survive the
  base-adapter swap.** C2′ (One-LoRA FT + BM25) acc 0.71 / MAE 0.310 — fine on its own — but
  stacking the User-LoRA gives **C3′ = C2′ exactly** (mean_diff 0.0000, p=1.0, 5/90/5): 10 users'
  predictions changed, split exactly 5 wins / 5 losses, an **exact cancellation**, verified by
  diffing raw predictions and by re-checking provenance at every layer.
- **R9** (LaMP-4 on One-LoRA FT) is designed and pinned, not run. **PT-track** (Per-Task-LoRA, one
  adapter per task, + PT1/PT2 User-LoRA rounds) is designed, not run.
- **LL1** (LongLaMP Product Review Task-LoRA, 2026-07-25): floor 0.328 R-1 → BM25 0.343 → LongLaMP-LoRA
  **0.182 (regression)**. Root cause: the adapter degenerates into verbatim-sentence greedy-decoding
  repetition loops (~897 mean generated tokens vs ~375), present from checkpoint-100 onward; matches
  a documented LoRA/greedy interaction (`huggingface/peft#1003`). `repetition_penalty=1.3` applied
  identically to all arms made every arm worse and was rejected. BFCL 0.645. `max_seq_length` had to
  go 2048→8192 (2048 truncated 75% of examples). Separate harness:
  `data/download_longlamp.py`, `train/build_longlamp_dataset.py`, `eval/eval_longlamp.py`.
- **Llama scale comparison** (2026-06-30, extended to 7 tasks 2026-07-15): SmolLM3-3B + One-LoRA FT
  beats Llama-3.1-70B-Instruct + BM25 on all seven tasks (+0.01 to +0.13); Llama-8B trails 70B
  everywhere. The K=100 personalization-hard subset table is still LaMP-3-only.
- **Per-user viability, corrected 2026-07-25:** the earlier record-count analysis wrongly called
  LaMP-2-movies and LaMP-5 dead ends. Their profile entries match their task's input→output shape,
  so the **profile-entry reframing** `build_user_dataset.py` already implements for LaMP-3/4 applies.
  Revised: **LaMP-2-movies / 2-news / 5 are viable via supervised reframing; LaMP-1 and LaMP-7 are
  viable only via OPPU's unsupervised right-shifted-history recipe** (new code path, not built).
  LaMP-2-news has 321 users / 102 unseen / up to 211 records, natural K≈27.

### Model & training (cluster side)

Base `HuggingFaceTB/SmolLM3-3B`, bf16, frozen. HF Transformers + PEFT. **CE loss only — no KD, no
teacher co-loading, no base-weight modification.**

```python
LoraConfig(r=4, lora_alpha=8,
           target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
           lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
```
r=4 (not 64) because the value prop is on-device efficiency; alpha/r = 2. Training: AdamW lr=3e-4,
cosine + 3% warmup, per-device bs 4 × grad_accum 8 (effective 32), 2–3 epochs, checkpoint every 500
steps, metrics to `metrics.jsonl` + `train_meta.json`. W&B wired but off by default.

**User-LoRA (OPPU) recipe, settled — do not relitigate:** r=8, q+v only, alpha=16, dropout 0.05,
AdamW lr=1e-5, L2 1e-2, cosine + 3% warmup, 3 epochs, `save_strategy=epoch`, `save_total_limit=1`,
per_device 2 / grad_accum 4; base = SmolLM3-3B + A1-lamp ckpt-1000 via `--base-adapter`;
eval = BM25 k=4, greedy, seed 0, `enable_thinking=False`, `max_new_tokens=64`; smoke on the
smallest-profile user.

### Datasets

- `data/lamp/LaMP_{3,4,7}/` — **user-based** split (users disjoint across splits), used for A1-lamp.
- `data/lamp_time/` — **time-based** split (same users, chronological partition), used for User-LoRA;
  present for all 7 tasks. Profile entries carry `date`; test_outputs present.
- Built corpora: `data/lamp_train_{LaMP_3,LaMP_4,LaMP_7,mixed}_bm25k4.jsonl` (mixed = 42,964) and
  the 7-task `mixed7` (72,062). **`LEGACY_MIXED_TASKS` in `build_dataset.py` keeps the 3-task
  `mixed` file distinct from `mixed7`** so A1-lamp's training data is never silently touched;
  existing per-task files are reused read-only (`per_task_reused` sidecar).
- Per-user volume varies sharply by task (`experiments/2026-06-12-lamp-time-split-per-user-counts.md`).
  **LaMP-6 unsupported** (private Avocado corpus).

### Eval methodology (frozen)

- **Personalization channel = BM25 top-k (k=4)** of the user's profile into the `system` slot — not
  summarization (tried, reverted; rationale `notebooks/lamp_evaluation_approach.md`). **Same BM25, same k, same formatting, same role layout at
  training and eval time. Train/eval consistency is the cardinal rule.**
- **System-always prompt regime** (resolved 2026-05-31): the profile sits in `system` for every
  training example, so the adapter expects that shape at inference. Open hypothesis: an on-device
  User-LoRA could absorb the profile into weights and drop the +118…+482 token/query prompt tax.
- **BFCL = Path C** — install `bfcl-eval` in the image, generate with our own transformers stack,
  call `ast_checker` as a library. SmolLM3 isn't in `MODEL_CONFIG_MAPPING`, so we pass
  `model_name="meta-llama/Llama-3.1-8B-Instruct"` as a neutral placeholder (recorded as
  `scorer_model_name_placeholder`); `BFCL_PROJECT_ROOT` must be set before any `bfcl_eval` import
  (`eval_bfcl.py` sets it to `/tmp/bfcl_project_root`).
- **BFCL `irrelevance` skipped** (its `possible_answer` file doesn't ship); Java/JS type errors
  (~80) account for most of the 80.78-vs-92.3 base gap and were never investigated.

---

## Conventions

### Standard script patterns (all eval/train/data-prep scripts)

- **Provenance banner** on the first stdout line: task / split / condition / seed / commit / Condor
  IDs / host.
- **Provenance dict in every result record**: `git_commit`, `git_dirty`, `condor_cluster_id`,
  `condor_proc_id`, `hostname`, `timestamp_utc`, library versions.
- **Flat single-level JSON results** — every field scalar, so
  `pd.DataFrame([json.load(open(p)) for p in glob("results/*.json")])` works with no unnesting.
- **Per-example predictions in a sibling JSONL** (`{id, pred, gold}`; BFCL adds `category`,
  `pred_text`, `pred_parsed`, `valid`, `error_type`).
- **Refuse-to-overwrite by default** — `sys.exit(1)` unless `--overwrite`. Smoke runs (`--limit N`)
  get an `_limitN` filename suffix so they can never collide with full-run outputs.
- Condor IDs forwarded via the submit file's `environment`
  (`CONDOR_CLUSTER_ID=$(ClusterId) CONDOR_PROC_ID=$(ProcId)`).
- Under Condor, `Path(__file__).parent.parent` does **not** resolve — use the `PROJECT_ROOT`/`LAMP_DIR`
  env-var pattern `build_dataset.py` already uses.

### Experiment log format

Every run gets `experiments/YYYY-MM-DD-<slug>.md` with `## Hypothesis / ## Setup (command, config,
seed) / ## Result / ## Conclusion`. Note only a minority of `experiments/*` are force-added to git —
check `git ls-files experiments/` before assuming a doc is versioned.

### Paper writeup style

LaTeX lives in a **separate git repo at `~/Documents/Research/overleaf/6a2b1ada3ba0566171e752a2/`**
(sections in `sections/experiments/*.tex`, `sections/90-appendix.tex`; remote `git.overleaf.com`;
pull before editing, push when done — credentials are not always configured, so some sections sit
committed-but-unpushed). Match the plain, direct style of the existing sections: short declarative
sentences, first person plural, minimal jargon, number-first. Per explicit feedback, avoid: inline
research-question bookkeeping ("RQ1", "corroborates Q4"); introducing a shorthand as a parenthetical
aside mid-sentence; justification asides for choices that don't change the takeaway; any line that
over-explains rationale nobody asked for. See memory `feedback_terse_paper_style.md` and
`feedback_no_ai_sounding_commits.md` (no Claude co-author trailers; plain human voice in commits and
public PR/issue prose; no dashes as punctuation).

---

## Repo structure

```
/
├── CLAUDE.md, Dockerfile, requirements.txt, pyrightconfig.json
├── condor/          # submit files: build_dataset, download_model, download_llama, interactive,
│                    #   eval_lamp{,_floor,_llama,_llama_k100}, eval_bfcl, train{,_1ep}, chat.py
├── data/            # download_lamp.py (--split-type user|time), download_longlamp.py,
│                    #   lamp/, lamp_time/, lamp_user_stats{.py,/}, models/, lamp_train_*.jsonl
├── train/           # build_dataset.py, build_user_dataset.py, build_longlamp_dataset.py,
│                    #   train.py, config/, checkpoints/ (gitignored)
├── eval/            # eval_lamp.py, eval_bfcl.py, eval_longlamp.py, paired_compare*.py,
│                    #   tables.py, summary.py + all on-device aggregators/plots (gitignored, -f)
├── ios/             # mlx-swift-examples/ (subtree), mlx-swift-lm-local/, mlx-swift/ (NAX patch),
│                    #   BGProbe/
├── scripts/         # device sequencers (nax_rerun_night*.sh, run_granularity_sweep.sh)
├── results/         # flat scalar JSON + predictions JSONL; ondevice/ for device telemetry+figures
├── experiments/     # YYYY-MM-DD-<slug>.md per run (mostly gitignored)
├── workshop_odi2026/  # ODI paper draft
└── runlogs/, notebooks/  # gitignored
```

## Docker image

Current tag **`ghcr.io/gordofreemo/smollm3-train:ver4`** — base
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (Python 3.11, torch 2.5.1+cu124), `apt-get install git`
(for in-container `git_commit` provenance), `pip install -r requirements.txt`.
**When `requirements.txt` or the Dockerfile changes, bump the tag and update all ten sub files**
(`eval_lamp`, `eval_lamp_floor`, `eval_lamp_llama`, `eval_lamp_llama_k100`, `eval_bfcl`,
`build_dataset`, `train`, `interactive`, `download_model`, `download_llama`).

**GPU capability ceiling.** RTX PRO 6000 Blackwell (sm_120) nodes cannot run ver4's cu124 build —
jobs die at the first CUDA op ("no kernel image is available"). Llama submits constrain
`Capability >= 8.0 && Capability < 10.0` via `require_gpus`; LaMP-4 GPU subs now carry it too.
Harden others as they start failing. Known flaky hosts: `tyr1` (GPU-slot oversubscription — retry or
exclude), `modi` (uncorrectable ECC), `fornjoter` (Blackwell).

---

## Hard constraints

- **Never modify base model weights.** LoRA only.
- **No KD loss.** CE only. **No co-loading teacher and student.**
- **Reproducibility first** — every training run launchable from one CLI command with a fixed seed;
  log the full command in the experiment file.
- **No profile leakage between splits** — validate explicitly.
- "No on-device / mobile code" is **LIFTED for Phase 3** (historical framing for Phases 1–2 only).

## Key references (do not hallucinate URLs)

- SmolLM3-3B: `HuggingFaceTB/SmolLM3-3B` (HuggingFace)
- LaMP: lamp-benchmark.github.io · LongLaMP: longlamp-benchmark.github.io (arXiv:2407.11016)
- OPPU (per-user PEFT recipe): arXiv:2402.04401
- BFCL: gorilla.cs.berkeley.edu/leaderboard.html
- CDCDA-PLM (closest prior work): arXiv:2508.21313
- Apple on-device fine-tuning (memory-efficient backprop): arXiv:2510.03425
- MELT (per-op mobile inference benchmarks, MobiCom '24) — the model for h11's Tier 2
- EnerInfer arXiv:2606.23001, PELM (ACM 2026) — DVFS/config-selection edge-LLM levers
