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
    static let appBuild = "smollm3-ondevice-train-bench-h3"
    static let schemaVersion = 1

    /// Build-time git provenance, stamped by hand at build time (same discipline
    /// as the inference harness — avoids fragile project.pbxproj build-phase
    /// surgery). Update alongside `appBuild` when re-baking before a run.
    static let gitCommit = "b748761"
    static let gitDirty = true

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

    /// Output JSONL in the app sandbox (separate from inference `bench_metrics.jsonl`).
    static let metricsFileName = "train_bench_metrics.jsonl"
}
