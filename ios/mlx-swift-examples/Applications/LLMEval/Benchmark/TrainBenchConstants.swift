// On-device naive LoRA-training characterization benchmark — constants.
//
// Implements the locked design in
//   experiments/2026-06-29-ondevice-training-naive-plan.md
// Separate from the inference benchmark's `BenchConstants` so the two harnesses
// version independently. Orchestration lives in
// `LLMEvaluator+TrainBenchmark.swift`.

import Foundation

enum TrainBenchConstants {
    /// Bump on every harness-logic change so each JSONL ties back to a specific
    /// harness version (baked into every record as `app_build`).
    /// h1 (2026-06-29): first cut — naive (no systems optimizations) LoRA
    /// training cost baseline on SmolLM3-3B-4bit, OPPU recipe (r=8, q+v only).
    /// h2 (2026-06-29): added stderr milestone tracing (`tlog`) to localize a
    /// SIGKILL/jetsam that struck before the first record was written.
    /// h3 (2026-06-29): naive batch-1 at full deployment seq length (193–1511
    /// tok) jetsams on the FIRST backward step. PRIMARY AXIS changed from
    /// batch-size sweep to a sequence-length-cap sweep at batchSize=1 to find
    /// the feasible boundary + OOM threshold (records persist per window; a
    /// `cap_start` sentinel marks the OOM'd cap).
    /// h4 (2026-06-30): GRADIENT-CHECKPOINTING variant. Per-transformer-block
    /// gradient checkpointing (all 28 blocks) added via a local mlx-swift-lm SPM
    /// override (`ios/mlx-swift-lm-local`, `SmolLM3Model.checkpointGroupSize`).
    /// Same cap sweep as h3 + cap=1024 stretch. Writes a SEPARATE JSONL
    /// (`train_bench_metrics_gc.jsonl`) so a mid-run jetsam can't corrupt the
    /// naive records. See experiments/2026-06-29-ondevice-training-gc-plan.md.
    /// h5 (2026-07-06): E2E per-user run — NEW mode (`--benchmark-train-e2e
    /// --user <fp>`, `runE2ETrainBenchmark`). Trains a REAL top-100 LaMP-3
    /// User-LoRA to completion (3 epochs = `3 × n_user` iterations) on the
    /// side-loaded per-user data with the faithful R5 recipe (AdamW, GC on,
    /// cap 1024, save adapter), and captures training loss + timed battery.
    /// Writes a SEPARATE JSONL (`train_bench_metrics_e2e.jsonl`); records are
    /// written INCREMENTALLY (per window, from the train callback) so a jetsam
    /// in a multi-hour run leaves every completed window on disk. The h1–h4
    /// cap-sweep path (`runTrainBenchmark`) is untouched. See
    /// experiments/2026-07-03-ondevice-e2e-training-plan.md.
    /// h9 (2026-07-26): ENERGY characterization round — adds CPU-utilization
    /// sampling to the existing periodic `battery` record (`cpu_util_pct`,
    /// aggregate %busy since the previous sample via
    /// `host_statistics`/`HOST_CPU_LOAD_INFO` — a simpler aggregate-across-
    /// all-cores read than per-core `host_processor_info`, sufficient for a
    /// secondary sanity-check signal; no public per-process GPU-utilization
    /// API exists on iOS, so this can't be a full power model on its own) +
    /// a new `idle_baseline` mode (`--benchmark-idle-baseline
    /// --user <fp> --baseline-duration-seconds <N>`, `runIdleBaselineBenchmark`)
    /// for paired energy-baseline runs (screen on, no training, same
    /// sampling cadence) used to subtract non-training drain from the real
    /// C2 training runs. See experiments/2026-07-26-ondevice-energy-h9-plan.md
    /// (pinned via `/grill_me` 2026-07-26). `appBuild` kept UNCHANGED
    /// (schema-only bump, same convention as h6's v1-v10 progression under
    /// one `bgAppBuild`) since the e2e TRAINING path itself is not modified —
    /// only the periodic-sample record gains a field and a new sibling mode
    /// is added; the still-pending E2E (h5) backlog runs stay correctly
    /// tagged h5.
    static let appBuild = "smollm3-ondevice-train-e2e-h5"
    static let schemaVersion = 3

    /// Build-time git provenance, stamped by hand at build time (same discipline
    /// as the inference harness — avoids fragile project.pbxproj build-phase
    /// surgery). Update alongside `appBuild` when re-baking before a run.
    static let gitCommit = "457d30a"
    static let gitDirty = true

    // --- Gradient-checkpointing flags (h4) -----------------------------------
    /// Baked into every record so GC runs are unambiguously distinguishable from
    /// the naive (h3) baseline even though both share the cap-sweep schema.
    static let gradientCheckpointing = true
    /// Granularity of the checkpoint unit. "per_block" = one `mx.checkpoint`-
    /// equivalent boundary per transformer block (all 28), the maximum-savings
    /// configuration matching mlx_lm's `--grad-checkpoint`.
    static let checkpointGranularity = "per_block"

    /// Steps per run cell (decision 7/8): 200 training iterations.
    static let iterations = 200

    /// One system-metrics record every N steps (decision 7). 200/5 = 40 records
    /// per cell — fine enough to catch thermal onset, coarse enough to stay quiet.
    static let stepsPerReport = 5

    /// Original batch-size sweep (decision 8) — retained for reference but no
    /// longer the primary axis (naive batch-1 OOMs at full seq length, so a
    /// batch-size sweep is moot until a feasible seq length is established).
    static let batchSizes = [1, 2, 4]
    static let stressBatchSizes = [1]

    /// PRIMARY AXIS (h3): ascending sequence-length cap (tokens) at batchSize=1.
    /// Each cap is one cell; the sweep finds the largest cap that survives the
    /// first backward step before jetsam. Ascending so every feasible cap's
    /// records are on disk before an infeasible cap SIGKILLs the process.
    static let seqCaps = [32, 64, 128, 256, 512, 1024]
    static let trainBatchSize = 1
    /// Cap used by `--benchmark-train-stress` (sustained thermal run). Set to a
    /// likely-feasible value; revise to the measured feasible ceiling after the
    /// sweep identifies it.
    static let stressSeqCap = 128

    /// Inter-cell cooldown gated on `thermalState == nominal`, capped (decision 8).
    static let cooldownCapSeconds = 120.0

    // --- LoRA config = OPPU recipe (decision 4) ------------------------------
    /// All 28 SmolLM3 transformer layers.
    static let loraLayers = 28
    static let loraRank = 8
    /// alpha/r = 2 → scale = alpha = 16.0 (matches R5/R6 r=8, alpha=16).
    static let loraScale: Float = 16.0
    /// Target only q_proj + v_proj. NOTE: `LoRAContainer.replaceLayers` matches
    /// against `Module.namedModules()` keys, which are FULL dotted paths from the
    /// transformer block (verified against mlx-swift Module.visit). SmolLM3's
    /// attention submodule is keyed `self_attn`, so the keys must carry that
    /// prefix — bare ["q_proj","v_proj"] would match nothing.
    static let loraKeys = ["self_attn.q_proj", "self_attn.v_proj"]
    /// Human-readable form for the JSONL `lora_keys` field.
    static let loraKeysLabel = "q_proj,v_proj"

    /// Bundled training/validation resources (decision 3). The valid stub exists
    /// only to satisfy the `LoRATrain.train` signature; the loop forces one
    /// validation at iteration 0 regardless of `stepsPerEval`, so it is consumed
    /// exactly once (see harness note). Validation is otherwise disabled.
    static let trainResource = "lora_train"
    static let validResource = "lora_valid"

    /// Output JSONL in the app sandbox. Separate from BOTH the inference
    /// harness's `bench_metrics.jsonl` AND the naive (h3) run's
    /// `train_bench_metrics.jsonl`, so a GC jetsam can't corrupt naive records
    /// and the two runs can be pulled/aggregated independently.
    static let metricsFileName = "train_bench_metrics_gc.jsonl"

    // =========================================================================
    // E2E (h5) — real per-user to-completion training. Constants used only by
    // `runE2ETrainBenchmark`; the h1–h4 cap-sweep constants above are unchanged.
    // =========================================================================

    /// Separate JSONL for the E2E run (never mixes with the cap-sweep files, so
    /// a mid-run jetsam can't corrupt prior runs and each is pulled/aggregated
    /// independently).
    static let e2eMetricsFileName = "train_bench_metrics_e2e.jsonl"

    /// Side-loaded per-user data lives at
    /// `Documents/<e2eDataDirName>/lamp3_<fp>.jsonl` (pushed via
    /// `devicectl device copy to`). Rendered `{"text": ...}` lines.
    static let e2eDataDirName = "user_data"

    /// Saved adapters go to `Documents/<e2eAdapterDirName>/adapter_<fp>.safetensors`
    /// (weights persisted for the fidelity / optional-accuracy check).
    static let e2eAdapterDirName = "e2e_adapters"

    /// Epochs over the user's data. `iterations = e2eEpochs × n_user` at batch 1
    /// (the LoRABatchIterator reshuffles on exhaustion), matching R5's 3 epochs
    /// on a data-coverage basis.
    static let e2eEpochs = 3

    /// Sequence-length cap (GC ceiling; some LaMP-3 examples exceed this and are
    /// truncated — documented forced deviation from R5's uncapped 7168).
    static let e2eSeqCap = 1024

    /// Batch 1 (no grad-accum in the stock trainer; batch≥2 at real seq length
    /// jetsams). Forced deviation from R5's effective-8.
    static let e2eBatchSize = 1

    /// Loss-reporting / record-emission cadence. Matches R5's logging_steps=10;
    /// the reported loss is the mean over the window. For L (987×3≈2961 iters)
    /// this is ~296 records — fine-grained enough for the fidelity overlay,
    /// coarse enough to keep the JSONL small.
    static let e2eStepsPerReport = 10

    /// Faithful R5 optimizer: AdamW, LR 1e-5, weight-decay 0.01 (== R5 L2).
    static let e2eLearningRate: Float = 1e-5
    static let e2eWeightDecay: Float = 0.01
    /// R5 used `adamw_torch`, which bias-corrects the moment estimates; MLX's
    /// AdamW defaults `biasCorrection=false`. Set true to match PyTorch AdamW.
    static let e2eAdamBiasCorrection = true

    /// Timed battery-sampling cadence (wall-clock seconds). Independent of the
    /// per-window train records so the C2 (unplugged) drain curve has real
    /// wall-clock resolution over a multi-hour run. UIDevice is @MainActor, so
    /// these samples are taken on the main actor while training runs off-actor.
    static let e2eBatterySampleSeconds = 30.0

    // =========================================================================
    // Background-scheduled (h6) — real BGProcessingTask OS scheduling, chunked
    // resumable training. Constants used only by `LLMEvaluator+BGTrain.swift`;
    // h1–h5 constants above are unchanged. Reuses the h5 recipe constants
    // (loraRank, loraKeys, e2eLearningRate, e2eWeightDecay,
    // e2eAdamBiasCorrection, e2eSeqCap, e2eBatchSize, e2eEpochs,
    // gradientCheckpointing) verbatim — only the orchestration differs.
    //
    // h6 (2026-07-XX): trains a real top-100 LaMP-3 User-LoRA to completion
    // under real (non-forced) `BGProcessingTask` OS scheduling instead of a
    // foreground/screen-on session. `LoRATrain.train` is called in chunks of
    // `bgChunkIterations` (10) so the app can checkpoint between chunks and
    // survive being suspended/relaunched across many wakes. See
    // experiments/2026-07-13-ondevice-bg-training-plan.md.
    //
    // KNOWN, ACCEPTED DEVIATION: checkpoint/resume covers LoRA weights + the
    // iteration counter ONLY. `MLXOptimizers.AdamW`'s internal Adam moments
    // (m/v) are stored in an `internal`-access `stateStorage` dict with no
    // public getter/setter (verified by reading mlx-swift's
    // Source/MLXOptimizers/Optimizers.swift — only a read-only, unkeyed
    // `innerState() -> [MLXArray]` is exposed, nothing to round-trip through).
    // Vendoring mlx-swift locally (as we did for mlx-swift-lm, to get
    // gradient checkpointing) to expose it was considered and explicitly
    // rejected as disproportionate for this round. So a FRESH `AdamW` is
    // constructed every wake — first/second moments reset to zero at every
    // wake boundary. This is a deliberate scope decision, not an oversight:
    // the round's two headline questions (calendar-vs-device time,
    // wake-scheduling characterization) don't depend on optimizer
    // continuity. The secondary loss-curve-continuity deliverable WILL show
    // small real restart bumps at wake boundaries — report them as such in
    // the write-up rather than treating them as a bug.
    // h6 schema v2 (2026-07-13, mid-run): added persistent JSONL diagnostic
    // markers (`model_loaded`, `training_setup_complete`, `chunk_start`,
    // `resume_start`/`resumed`) after the first real submission made zero
    // progress across 3 wakes with no visibility into why — deliberately
    // NOT more `tlog()`, since tlog's stderr is unreadable during a real
    // unattended wake (no console attached), only readable during a
    // `devicectl --console` launch or an Xcode-debug session. Also switched
    // `loadLoRAWeights`'s `Module.update` call from the `verify: .none`
    // convenience wrapper to the throwing `verify: .shapeMismatch` overload
    // — a real latent gap (silent corruption instead of a catchable error
    // on any shape mismatch), found while investigating two consecutive
    // real wakes that died silently right around the weight-resume step.
    // h6 schema v3 (2026-07-13, mid-run, same day as v2): after v2 STILL
    // showed 4/4 consecutive resume-needing wakes dying in the same narrow
    // window (right after `model_loaded`, before even `resume_start`) while
    // the 1 fresh-start wake sailed through it — added `lora_apply_start`/
    // `lora_apply_complete` markers bracketing the one call both paths
    // share, PLUS a heartbeat mechanism (`bgHeartbeatFileName`, overwritten
    // ~1x/sec by a concurrent Task, independent of which milestone marker
    // last fired) to directly answer "how long was this wake actually alive
    // before it died" — a question the coarse milestone markers alone can't
    // answer precisely, since a wake can die between any two of them.
    // h6 schema v4 (2026-07-14): after v3's new markers showed 30+
    // consecutive wakes dying in an extremely tight ~9.6-9.8s band (not just
    // "somewhere in the early region" — a near-fixed cutoff), always
    // silently (no `wake_end`, so `setTaskCompleted` never gets called) —
    // added `Task.isCancelled` checks at every step boundary in the setup
    // path (not just the training while-loop, which is where the ONLY
    // successful cancellation-catch so far happened, on wake 0) plus a
    // loose wall-clock backstop (`bgWallClockBackstopS`) independent of
    // `Task.isCancelled`, in case cancellation doesn't propagate through a
    // synchronous call. Testable hypothesis, not a confirmed fix: silently
    // dying without ever calling `setTaskCompleted` may be training iOS's
    // scheduler to keep granting minimal "probation" windows — reaching a
    // clean `return` (even a `wall_clock_backstop` one with zero training
    // done) at least gives the scheduler an acknowledged completion signal
    // every time, which the current behavior never has.
    // h6 schema v5 (2026-07-14): the wall-clock-backstop catch (v4) confirmed
    // the fix's mechanism works when it fires, but the very next wake reverted
    // to the same ~10s silent death with no visible grant-size recovery — so
    // "reach a clean return eventually" isn't enough on its own to test the
    // scheduler-trust hypothesis; the app was still trying to grab MULTIPLE
    // `bgChunkIterations` chunks per wake whenever it got the chance (wake 0
    // ran 4 chunks back-to-back before being cancelled), i.e. always asking
    // for as much as it could get. Changed `runBGTrainWake()` to attempt
    // exactly ONE chunk per wake, checkpoint, then voluntarily return
    // (`voluntary_yield`) rather than looping for more even if time remains.
    // Explicitly a hypothesis test, not a confirmed fix: does a consistently
    // small, quick, always-completes-cleanly request pattern earn steadier
    // scheduling than a greedy one? Doesn't address the current dominant
    // failure mode (dying during model load/LoRA setup, before any chunk is
    // reached) — it's a complementary, lower-priority experiment layered on
    // top of the existing Task.isCancelled/wall-clock-backstop safety net,
    // which is unchanged.
    // h6 schema v6 (2026-07-15): diagnostic-only, no behavior change. A
    // standalone control app (`ios/BGProbe/` — registers a trivial
    // BGProcessingTask, no model load, no heavy allocation) got 240s+
    // grants at the EXACT SAME wake instants (matched to the second) that
    // LLMEval died at its usual ~9.5-10s — ruling out a platform/OS-level
    // ceiling and pointing squarely at LLMEval's own resource footprint
    // (most likely memory pressure from loading the ~1.7GB 4-bit model) as
    // the proximate cause of the short grants. Added `peak_mem_bytes`/
    // `active_mem_bytes` (via `Memory.snapshot()`) to the `model_loaded`,
    // `lora_apply_start`, and `lora_apply_complete` markers, plus a single
    // `GPU.resetPeakMemory()` near wake start for a clean per-wake
    // baseline so readings are comparable. Also bumped the heartbeat
    // cadence 1s→200ms and added the same memory fields to every heartbeat
    // tick — prior investigation localized death to within ~0.1-0.5s of
    // `lora_apply_start`, too fast for 1s heartbeat resolution to pin down
    // further; the finer cadence plus memory fields means the LAST
    // heartbeat tick before a silent death now gives both a tighter
    // elapsed-time bound AND an actual memory reading at approximately the
    // moment of death.
    // h6 schema v7 (2026-07-15): first REAL behavior change (not just
    // instrumentation) since the footprint investigation began. Traced the
    // vendored model-load path (`mlx-swift-lm-local/Libraries/
    // MLXLMCommon/Load.swift`): the loaded-weights dictionary stays alive
    // through `model.update(parameters:)` and the final `eval(model)` that
    // materializes the whole model at once — if `update` doesn't just swap
    // references, both the loaded-from-disk copy and the model's own
    // parameter storage could be resident simultaneously right at the peak
    // moment. Added one line, `weights.removeAll()`, right after `update`
    // succeeds and before `eval(model)`, to let ARC release the loaded
    // copy before the materialization spike.
    // RESULT: no-op. Two v7 wakes both showed peak_mem_bytes IDENTICAL to
    // the pre-fix baseline (3,459,923,896 bytes, to the byte). Explained
    // by reading `Module.update` in mlx-swift's `Module.swift`: the
    // leaf-array case calls `p._updateInternal(newArray)`, a reference
    // swap, not a copy — the model's own parameter storage and
    // `weights[key]` already point at the SAME MLXArray after `update`,
    // so dropping the `weights` dict reference frees nothing (the model
    // still holds the only reference that matters). Also confirmed
    // on-device there's no duplicate model file inflating this (pulled
    // `Library/Caches/huggingface/hub/.../blobs/` directly — exactly one
    // ~1.73GB blob, matching the published repo's `model.safetensors`
    // byte-for-byte, no stray/duplicate copies on disk).
    // h6 schema v8 (2026-07-15): removed the disproven `weights.removeAll()`
    // line. Added `LoadWeightsDiagnostics` (new type in vendored
    // `Load.swift`) — a memory snapshot at each of `loadWeights`'s 5
    // internal stages (`loadWeights_entry`, `safetensors_read`,
    // `sanitize_complete`, `quantize_applied`, `parameters_updated`,
    // `eval_complete`), drained by `runBGTrainWake()` right after `load()`
    // returns and logged as one `load_stage` JSONL record per stage
    // (`peak_mem_bytes`/`active_mem_bytes`/`wake_elapsed_s` each). Directly
    // localizes which specific stage the ~2x jump (1.73GB on-disk → 3.46GB
    // resident) happens at, rather than continuing to guess from the
    // outside. Leading candidate going in: `quantize(model:)` calls
    // `QuantizedLinear.init(weight:...)` on the model's own freshly
    // constructed (random-init, full-precision, lazy) Linear/Embedding
    // layers BEFORE the real loaded weights are applied — if that graph
    // gets materialized somehow before being replaced by `update`, it'd
    // produce a same-sized second quantized copy. Unconfirmed; this is
    // what the v8 trace is for.
    // h6 schema v9 (2026-07-15): v8's own first wake ANSWERED the
    // question — `load_stage` showed `loadWeights` ran TWICE within one
    // wake: a first load ~514s before `wakeStart` (timing matches a
    // `--bg-train-resubmit` foreground launch shortly before), cleanly
    // materializing to 1.73GB, immediately followed by the wake's own
    // load STARTING from that already-resident 1.73GB baseline and
    // reaching 3.46GB at its own `eval_complete`. Root cause, confirmed
    // by reading the code (not guessed): `handleBGTrainTask` instantiated
    // a brand-new `LLMEvaluator()` on every wake
    // (`await LLMEvaluator().runBGTrainWake()`), completely independent
    // of `ContentView`'s own `@State var llm = LLMEvaluator()`, which
    // eagerly loads the model via `.task { llm.load() }` on every normal
    // app launch. If the process stays resident between a foreground
    // launch and the next OS-granted wake, both models end up loaded
    // simultaneously. Fix: `LLMEvaluator.shared` (new static singleton,
    // `ViewModels/LLMEvaluator.swift`) — both `ContentView` and
    // `handleBGTrainTask` now reference the SAME instance. `load()`
    // already had correct caching via its `loadState` enum
    // (idle/loading/loaded) — it just never had a chance to apply across
    // instances before. This also matches the realistic deployment story
    // better than the artificial one this investigation ran under: a real
    // user opens the app, backgrounds it, and the eventual wake reuses
    // the already-warm model — one load, correct footprint, and the
    // ~9.5-10s load cost is skipped entirely inside the tight granted
    // window rather than repeated every wake.
    // h6 schema v10 (2026-07-16): v9 fixed the memory doubling, but 32
    // consecutive wakes over ~16.5h on the clean v9 restart still banked
    // ZERO checkpoints — 13/32 reached `chunk_start` (~29s in) but none
    // finished a 10-iteration chunk, ruling out memory pressure as the
    // (sole) driver of the dominant failure. Two real, complementary
    // fixes, not more diagnostics: (1) `cachedCapExamples` (new, near
    // `capExamples`) disk-caches the tokenize+truncate+detokenize pass
    // over all `nUser` examples — measured at ~20-27s on every wake
    // (the dominant chunk of the `lora_apply_complete` → `chunk_start`
    // gap) despite being pure harness overhead unrelated to the recipe,
    // paid fresh every wake for no reason. Cached to
    // `bg_capped_examples_<user>.json`, keyed by `cap`/`count` so a
    // config or data change can't silently reuse stale output; adds a
    // `cap_examples_cache_hit` field to `training_setup_complete`. (2)
    // `bgChunkIterations` dropped 10→3 (see its own doc comment) — needs
    // proportionally less training time to reach a checkpoint, so a
    // partial window has a real chance to bank something instead of an
    // all-or-nothing loss. Neither change touches the recipe itself
    // (LR/rank/batch/seq_cap/epochs all unchanged) — both are pure
    // harness efficiency, matching the standing rule that fixes here
    // shouldn't compromise the faithful-recipe comparison to h5.
    static let bgAppBuild = "smollm3-ondevice-train-bg-h6"
    static let bgSchemaVersion = 10

    /// Defensive wall-clock ceiling (from `wakeStart`) checked alongside
    /// `Task.isCancelled` at every setup-path step boundary — independent
    /// backstop in case cancellation doesn't propagate through a
    /// synchronous call in time. Set well above the observed ~10s death
    /// zone (30+ consecutive real wakes) so it never preempts a
    /// legitimately long wake (wake 0 ran ~324s) — this is a safety net,
    /// not a mechanism for trying to predict/beat the real kill.
    static let bgWallClockBackstopS: Double = 25.0

    /// `BGTaskSchedulerPermittedIdentifiers` entry (Info.plist) + the id
    /// passed to `.backgroundTask(.processing(id:))` — must match exactly.
    static let bgTaskIdentifier = "mlx.LLMEval.bgtrain"

    /// Separate JSONL for the BG run (never mixes with h1–h5 files).
    static let bgMetricsFileName = "train_bench_metrics_e2e_bg.jsonl"

    /// Overwritten (not appended) ~1x/sec throughout a wake by a concurrent
    /// heartbeat `Task`, independent of which milestone marker last fired.
    /// After a wake dies silently, this file's last-written
    /// `wake_elapsed_s` is the actual OS-granted time-slice length for that
    /// wake — the milestone JSONL markers alone can only bound death to
    /// "somewhere between marker A and marker B," which can be a wide gap.
    static let bgHeartbeatFileName = "bg_heartbeat.json"

    /// Run-level summary, mirrors the cluster-side `train_meta.json`
    /// convention (rewritten at the end of every wake).
    static let bgRunMetaFileName = "bg_run_meta.json"

    /// Per-user checkpoint subdir under Documents:
    /// `bg_checkpoints/<fp>/weights.safetensors` (the LoRA adapter itself —
    /// doubles as both the running checkpoint and the final saved adapter)
    /// + `bg_checkpoints/<fp>/checkpoint_meta.json` (iteration counter, wake
    /// number, cumulative device-compute seconds). No optimizer-state file
    /// — see the deviation note above.
    static let bgCheckpointDirName = "bg_checkpoints"

    /// Submission-time config written to `Documents/<bgConfigFileName>` (user
    /// fingerprint, condition, computed iterations_total) — read by the wake
    /// handler since `BGProcessingTaskRequest` carries no custom payload.
    static let bgConfigFileName = "bg_train_config.json"

    /// Chunk size for the ONE `LoRATrain.train(iterations: bgChunkIterations)`
    /// call attempted per wake (schema v5 — previously looped for multiple
    /// chunks per wake whenever time allowed; now always stops after exactly
    /// one, see the v5 changelog above). `LoRATrain.train` is a single
    /// blocking call — checkpointing only happens after the chunk completes,
    /// so worst case `bgChunkIterations` iterations of work is lost on a
    /// hard SIGKILL rather than a clean expiration.
    //
    // h6 schema v10 (2026-07-16): dropped 10→3. After the clean v9 restart,
    // 32 consecutive wakes over ~16.5h banked ZERO checkpoints — 13/32 got
    // as far as `chunk_start` (~29s in) but none finished a 10-iteration
    // chunk (~45-56s of training on top of that, needing a ~75-85s window
    // that never showed up). A smaller chunk needs proportionally less
    // training time to reach a checkpoint, giving partial-window wakes a
    // real chance to bank something instead of an all-or-nothing loss.
    // Paired with the `capExamples` disk-cache below (same schema bump) —
    // together they attack both halves of "does this wake get far enough
    // AND finish once it's training." Quadruples chunk count for the
    // 1215-iteration run (~122 → ~405) but that's cheap relative to a wake
    // (checkpoint write is fast); worth it if completion rate actually
    // moves off zero.
    static let bgChunkIterations = 3

    // =========================================================================
    // Token-time cost model (h7) — per-iteration wall-time as a function of
    // synthetic example token count, under the exact h5 recipe (AdamW lr=1e-5
    // wd=0.01 bias-corrected, r=8 q+v, GC on, batch=1). Constants used only by
    // `runTokenTimeBenchmark` in `LLMEvaluator+TrainBenchmark.swift`; h1-h6
    // constants above are unchanged. Purpose: fit `seconds/iter ≈ a + b×tokens`
    // to predict real E2E (h5) per-user wall time from a user's example
    // token-length distribution. Pinned via `/grill_me` 2026-07-24; see
    // experiments/2026-07-24-ondevice-tokentime-plan.md for the full design
    // rationale (locked decisions, don't relitigate without cause).
    static let tokentimeAppBuild = "smollm3-ondevice-train-tokentime-h7"
    /// v2 (2026-07-24, same day): grid changed from the coarse `seqCaps`
    /// (32,64,128,256,512,1024 — log-ish spacing) to a uniform 50-token-step
    /// grid, 50...1000 (20 cells), per explicit user request for finer
    /// resolution. All 6 caps up to 1024 were already OOM-free under GC
    /// (v1's own result), so this is purely a resolution change, not a new
    /// feasibility question. Upper bound rounded down to 1000 (nearest
    /// multiple of 50) rather than tacking a non-uniform 1024 onto the end.
    /// v3 (2026-07-24, same day): added the COLD regime (`runTokenTimeColdBenchmark`,
    /// `tokentimeColdMetricsFileName`) — see that constant's doc comment.
    /// v4 (2026-07-25): raised the cold regime's cooldown cap
    /// (`tokentimeColdCooldownCapSeconds`, 120s→300s) after v3's own data
    /// showed 120s was insufficient above ~200 tokens — see that constant's
    /// doc comment for the evidence.
    static let tokentimeSchemaVersion = 4

    /// Uniform 50-token-step grid, 50...1000 (20 cells) — locked design v2.
    static let tokentimeTokenCounts = Array(stride(from: 50, through: 1000, by: 50))

    /// 21 iterations/cell; iteration index 0 is discarded (MLX graph compile /
    /// first-allocation overhead, plus `LoRATrain.train`'s forced validation
    /// pass which always fires at iteration 0 regardless of `stepsPerEval`),
    /// the remaining 20 are kept for the per-cell mean + stddev.
    static let tokentimeIterationsPerCell = 21

    /// Filler phrase tiled to exceed each target token count, then truncated
    /// via tokenizer encode/decode to the *exact* count (locked design:
    /// synthetic, not real corpus text — content is irrelevant to
    /// attention/FFN compute cost, only sequence shape matters).
    static let tokentimeFillerPhrase = "The quick brown fox jumps over the lazy dog. "

    /// Deliberate warm-up burst before cell 1 (discarded, no records written)
    /// so ALL cells — including the smallest token count — are measured under
    /// representative sustained-training thermal conditions, not an
    /// artificial cold-start advantage for the first cell (locked design:
    /// explicit user call, opposite of h3/h4's cooldown-to-nominal pattern —
    /// this cost model is meant to represent real sustained E2E training).
    /// Duration matches the inference benchmark's observed ~90s throttle knee.
    static let tokentimeWarmupSeconds: Double = 75.0
    static let tokentimeWarmupSeqCap = 512
    /// Upper bound on warm-up iterations so a stalled/slow device can't loop
    /// forever; the progress callback stops early via `.stop` once
    /// `tokentimeWarmupSeconds` elapses — this is just a safety ceiling.
    static let tokentimeWarmupMaxIterations = 400

    /// Output JSONL — separate from every other harness file (h1-h6
    /// convention: a mid-run failure in one harness can't corrupt another's
    /// data, and each is pulled/aggregated independently).
    static let tokentimeMetricsFileName = "train_bench_metrics_tokentime.jsonl"

    /// v3 (2026-07-24, same day): COLD variant — explicit user request for
    /// the opposite thermal regime from the hot/no-cooldown sweep above. Each
    /// cell is isolated by a cooldown-to-nominal gate (reuses `trainCooldown()`,
    /// same mechanism/cap as h3/h4) both BEFORE the sweep starts (in case the
    /// device is still hot from a prior run) and AFTER every cell — no
    /// deliberate warm-up burst (that would defeat the point: this run
    /// measures each token count from as close to a clean nominal baseline as
    /// the device will give in `cooldownCapSeconds`). Same grid, same 21
    /// iterations/cell, same recipe as the hot sweep — only the thermal
    /// regime between cells differs, so the two are directly comparable.
    /// Separate JSONL (own file, not a wipe-and-reuse of the hot file) so
    /// both regimes' data persist independently and can't be conflated by an
    /// aggregator that groups by token count alone.
    static let tokentimeColdMetricsFileName = "train_bench_metrics_tokentime_cold.jsonl"

    /// v4 (2026-07-25): the v3 cold run's own data showed the shared h3/h4
    /// `cooldownCapSeconds` (120s) is genuinely too short once cell training
    /// time (and therefore heat output) grows with token count — verified
    /// against the raw per-iteration `thermal_state` series, not guessed:
    /// cells ≤200 tokens recovered to `nominal` every time (clean), but
    /// cap=250 flipped `nominal`→`fair` mid-cell, cap=400 was `fair` for 19/20
    /// iterations then `serious`, and cap=950 never got below `serious` at
    /// all despite the 120s wait before it. A dedicated (larger) cap for this
    /// regime only — NOT a change to the shared `cooldownCapSeconds`, which
    /// h3/h4 still use unmodified via `trainCooldown()`'s default parameter.
    /// True recovery time at the high end of the grid is unknown (this is the
    /// first attempt at raising it), so 300s (2.5×) is a first bump, not a
    /// verified-sufficient value — the rerun's own thermal_state column is
    /// the check for whether it was enough.
    static let tokentimeColdCooldownCapSeconds: Double = 300.0

    // =========================================================================
    // GC granularity sweep (h8) — how per-iteration throughput, peak activation
    // memory, and thermal trajectory respond to gradient-checkpointing
    // GRANULARITY: K consecutive transformer blocks grouped between checkpoint
    // boundaries (`SmolLM3Model.checkpointGroupSize`), at a fixed cap=1024.
    // Direct follow-on to h4 (per-block GC, K=1 only). No pre-registered
    // hypothesis — pure systems characterization. Constants used only by
    // `runGranularityBenchmark` in `LLMEvaluator+TrainBenchmark.swift`; h1-h7
    // constants above are unchanged. Pinned via `/grill_me` 2026-07-25; see
    // experiments/2026-07-25-ondevice-gc-granularity-plan.md.
    //
    // KNOWN, ACCEPTED DEVIATION FROM h1-h7: the shared `loraLayers` constant
    // above (28) was discovered to be a bug during h8 implementation — the
    // real SmolLM3-3B has 36 hidden layers (verified against
    // data/models/SmolLM3-3B-mlx-4bit/config.json's `num_hidden_layers`), and
    // `LoRAContainer.from` takes a SUFFIX of `numLayers` blocks (verified
    // against `LoRAContainer.swift`'s `lora.loraLayers.suffix(configuration
    // .numLayers)`), so every on-device round through h7 actually trained
    // LoRA on only the LAST 28 of 36 blocks, not "all layers" as their doc
    // comments claim. The cluster-side R5/R6 recipe this was supposed to
    // match (`train/config/user_lora_lamp3_oppu_template.json`) has no such
    // restriction — PEFT's `target_modules: ["q_proj","v_proj"]` applies to
    // every matching submodule, i.e. genuinely all 36. Explicit user call:
    // fix this FOR h8 ONLY (`granularityLoraLayers` below, =36), leaving the
    // shared `loraLayers`/h1-h7 untouched (their results are closed/written
    // up and not being re-run). Because of this, h8 also does NOT reuse h4's
    // K=1 cell verbatim (the design doc's original plan) — K=1 is re-run
    // fresh under h8's own (correct, 36-layer) LoRA config so all 9 K values
    // in the sweep are mutually apples-to-apples.
    static let granularityAppBuild = "smollm3-ondevice-train-granularity-h8"
    static let granularitySchemaVersion = 1

    /// h8-specific LoRA layer count — see the deviation note above. NOT the
    /// same as the shared `loraLayers` (28, still used unmodified by h1-h7).
    static let granularityLoraLayers = 36

    /// All 9 divisors of 36, ascending. K=1 is re-run fresh (not reused from
    /// h4 — see deviation note above), so every cell in this list is a real
    /// on-device run.
    static let granularityKValues = [1, 2, 3, 4, 6, 9, 12, 18, 36]

    /// Fixed sequence-length cap for every cell (locked design: the most
    /// demanding/thermally-bound cell from h4 — 0.07 iter/s, 4014 MB peak,
    /// `serious` throughout at K=1 — where a compute/memory tradeoff knob
    /// matters most, and closest to real LaMP user-history lengths). No cap
    /// sweep this round.
    static let granularitySeqCap = 1024

    /// 100 steps/cell — half of h4's 200 (h4's steady-state stats stabilize
    /// well before 200 steps end; halves wall-clock/thermal cost per cell and
    /// surfaces an OOM sooner if one occurs).
    static let granularityIterations = 100

    /// One system-metrics record every 5 steps → 20 windows/cell.
    static let granularityStepsPerReport = 5

    /// AdamW lr=1e-5 (locked design). Weight decay / bias correction left at
    /// `AdamW`'s own defaults (0.01 / false) — the design doc specifies only
    /// the learning rate, and this round measures compute/memory/thermal
    /// cost, not loss, so the optimizer's other hyperparameters don't affect
    /// the measured quantities.
    static let granularityLearningRate: Float = 1e-5

    /// Inter-cell cooldown: poll `thermalState` once/minute until `nominal`
    /// (UNCAPPED — no timed backstop, unlike h3/h4's 120s or h7 cold's 300s,
    /// both found insufficient at high heat/token counts; explicit user call:
    /// "impossible for the phone not to cool down"), then wait this many
    /// additional seconds as a fixed buffer before starting the next cell.
    static let granularityCooldownPollSeconds: Double = 60.0
    static let granularityCooldownBufferSeconds: Double = 600.0

    /// Output JSONL — separate from every other harness file (h1-h7
    /// convention: a mid-run failure in one harness can't corrupt another's
    /// data). One process launch per K (Mac-driven orchestration — see the
    /// driver script), so every cell's records persist independently even if
    /// a later K jetsams; a jetsam/OOM at a given K is itself valid data, not
    /// a failure to retry.
    static let granularityMetricsFileName = "train_bench_metrics_granularity.jsonl"
}
