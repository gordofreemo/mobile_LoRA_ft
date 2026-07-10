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
    static let gitCommit = "9ba9f06"
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
}
