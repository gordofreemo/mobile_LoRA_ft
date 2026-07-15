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
    /// override (`ios/mlx-swift-lm-local`, `SmolLM3Model.useGradientCheckpoint`).
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
    static let appBuild = "smollm3-ondevice-train-e2e-h5"
    static let schemaVersion = 2

    /// Build-time git provenance, stamped by hand at build time (same discipline
    /// as the inference harness — avoids fragile project.pbxproj build-phase
    /// surgery). Update alongside `appBuild` when re-baking before a run.
    static let gitCommit = "3047aff"
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
    static let bgAppBuild = "smollm3-ondevice-train-bg-h6"
    static let bgSchemaVersion = 6

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
    /// one, see the v5 changelog above). Matches `e2eStepsPerReport` (10) so
    /// one JSONL record = one checkpoint = one chunk = one wake's worth of
    /// work. `LoRATrain.train` is a single blocking call — checkpointing only
    /// happens after the chunk completes, so worst case ~10 iterations of
    /// work is lost on a hard SIGKILL rather than a clean expiration.
    static let bgChunkIterations = 10
}
