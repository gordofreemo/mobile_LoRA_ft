// On-device naive LoRA-training characterization benchmark — orchestration.
//
// Auto-started by a launch arg (`--benchmark-train` / `--benchmark-train-stress`)
// so the whole run is one Mac-driven `devicectl process launch --console`
// command (decision 2/8). Writes one flat-JSON record per stepsPerReport window
// to Documents/train_bench_metrics.jsonl, then exits so `--console` unblocks.
//
// Goal (decision 1): a COST BASELINE for systems-optimization comparison — raw
// time / memory / thermal cost of on-device LoRA fine-tuning with zero systems
// tricks (no gradient checkpointing, no quantized optimizer states, no
// activation offloading). Feasibility is a byproduct.
//
// See experiments/2026-06-29-ondevice-training-naive-plan.md for the locked
// design. Implementation-time deviations from the plan, all forced by the actual
// mlx-swift-lm source (verified by reading it):
//
//  * LoRA target keys are FULL dotted module paths — ["self_attn.q_proj",
//    "self_attn.v_proj"], not bare ["q_proj","v_proj"]. `LoRAContainer`'s
//    `replaceLayers` matches `Module.namedModules()` keys, which are dotted
//    from the transformer block (mlx-swift Module.visit uses flattened
//    prefixes). Bare keys match nothing. (TrainBenchConstants.loraKeys.)
//  * Validation cannot be fully disabled: `LoRATrain.train` forces one
//    validation at `iteration == 0` regardless of `stepsPerEval`. So the valid
//    stub IS consumed once at step 0; its forward pass is folded into the first
//    report window (step 5). `stepsPerEval = iterations+1` suppresses all the
//    others. We ignore the validation Progress event (no record written).
//  * Per-window battery/charging is unavailable: `UIDevice` is `@MainActor`
//    (NS_SWIFT_UI_ACTOR) and the training loop runs on the ModelContainer's
//    background actor, so the synchronous progress callback can't read UIKit.
//    Battery + charging are sampled at cell boundaries on the main actor
//    (`battery_level` = cell start, `battery_level_end` = cell end); thermal /
//    low-power / peak-mem / throughput remain per-window (ProcessInfo + MLX are
//    safe off-main). At 200 steps the battery delta is at/under the 1% floor
//    anyway (decision 10).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
// h11 Tier 2 only: `MTLCaptureManager.supportsDestination(_:)` is checked
// before any `GPU.startCapture`, so a missing `MetalCaptureEnabled` Info.plist
// key degrades to a logged marker instead of an uncatchable mlx-c error exit.
import Metal

#if canImport(UIKit)
    import UIKit
#endif

/// Serializes concurrent appends to the E2E JSONL from the off-actor train
/// callback and the main-actor battery sampler. File scope so it is reachable
/// from `nonisolated` writers (a `@MainActor`-class static `let` would not be).
private let e2eFileLock = NSLock()

/// Serializes appends to the token-time (h7) JSONL. No concurrent writer
/// (single sequential train callback, no battery sampler) but kept for
/// consistency with the E2E write path and as cheap insurance.
private let tokentimeFileLock = NSLock()

/// Serializes appends to the granularity-sweep (h8) JSONL. No concurrent
/// writer (one cell per process launch, single sequential train callback)
/// but kept for consistency with the other write paths.
private let granularityFileLock = NSLock()

/// Serializes appends to the thermal-cooldown (h10) JSONL. Genuinely
/// concurrent here, unlike h7/h8: the 10s passive sampler task and the
/// soak/probe train callbacks write interleaved for the whole run.
private let thermalFileLock = NSLock()

/// Serializes appends to the per-op (h11) JSONL. Concurrent here like h10's:
/// the 30s passive sampler task and the cell train loops write interleaved.
private let peropFileLock = NSLock()

/// Serializes appends to the task-adapter (h12) JSONL. Genuinely concurrent:
/// the 30s passive sampler task and the off-actor training loop write
/// interleaved for the whole (multi-hour) run.
private let taskAdapterFileLock = NSLock()

extension LLMEvaluator {

    // MARK: - Launch mode

    /// True when the app was launched to run the training benchmark.
    static var trainBenchmarkLaunchMode: TrainBenchLaunchMode? {
        let args = CommandLine.arguments
        // E2E (h5), token-time (h7, hot/cold), and granularity (h8) are
        // distinct exact args — check the more specific "-cold" flag before
        // the plain one.
        if args.contains("--benchmark-idle-baseline") { return .idleBaseline }
        if args.contains("--benchmark-train-e2e") { return .e2e }
        if args.contains("--benchmark-train-tokentime-cold") { return .tokentimeCold }
        if args.contains("--benchmark-train-tokentime") { return .tokentime }
        if args.contains("--benchmark-train-granularity") { return .granularity }
        // h11: the capture mode is a distinct exact arg, but check it first
        // anyway so the pair reads unambiguously.
        if args.contains("--benchmark-train-perop-capture") { return .peropCapture }
        if args.contains("--benchmark-train-perop") { return .perop }
        // NAX A/B: h11's cell machinery with per-iteration arm alternation.
        if args.contains("--benchmark-nax-ab") { return .naxAB }
        // h12: on-device Per-Task-LoRA (LaMP-7) training to completion.
        if args.contains("--benchmark-train-taskadapter") { return .taskAdapter }
        // NAX qmm_n numerical check: no model, no training — see
        // runNaxVerifyBenchmark().
        if args.contains("--verify-qmm-n") { return .verifyQmmN }
        if args.contains("--benchmark-thermal-selflimit") { return .thermalSelfLimit }
        if args.contains("--benchmark-thermal-cycle") { return .thermalCycle }
        if args.contains("--benchmark-thermal-cooldown") { return .thermalCooldown }
        if args.contains("--benchmark-train-stress") { return .stress }
        if args.contains("--benchmark-train") { return .full }
        return nil
    }

    enum TrainBenchLaunchMode {
        /// Batch-size sweep: 200 steps at each of batchSizes, cooldown between.
        case full
        /// Sustained single run: 200 steps at batchSize=1 only, no interruptions.
        case stress
        /// E2E (h5): one real user trained to completion (3×n_user iters), save
        /// adapter, capture loss + timed battery. See runE2ETrainBenchmark.
        case e2e
        /// Token-time (h7), HOT regime: per-iteration wall-time vs synthetic
        /// example token count, warm-up burst + no cooldown between cells
        /// (sustained-training conditions). See runTokenTimeBenchmark.
        case tokentime
        /// Token-time (h7), COLD regime: same grid/recipe/iterations, but
        /// each cell isolated by a cooldown-to-nominal gate before and after
        /// (clean per-token-count measurement). See runTokenTimeColdBenchmark.
        case tokentimeCold
        /// GC granularity sweep (h8): ONE cell (fixed K, via `--granularity-k`)
        /// per process launch — Mac-driven orchestration loops K across many
        /// launches. See runGranularityBenchmark.
        case granularity
        /// Idle energy baseline (h9): no model load, no training — just the
        /// same battery/thermal/CPU sampling cadence as the E2E battery
        /// sampler, for a fixed wall-clock duration
        /// (`--baseline-duration-seconds`). Paired with a real C2 training
        /// run (same `--user`, matched starting charge band) so its drain
        /// rate can be subtracted out. See runIdleBaselineBenchmark.
        case idleBaseline
        /// Thermal cooldown trajectory (h10): cold-reference probe → fixed
        /// training soak → fixed 90-minute observation window probed at a
        /// fixed cadence, measuring how training throughput RECOVERS after a
        /// burst. See runThermalCooldownBenchmark.
        case thermalCooldown
        /// Sustained-cycling arm (h10c): repeat burst/rest cycles and measure
        /// whether iterations-per-burst decays, i.e. whether the schedule
        /// extrapolated from a single burst actually holds up over repeats.
        /// See runThermalCycleBenchmark.
        case thermalCycle
        /// Self-limiting arm (h10d): ONE continuous training call, paced with a
        /// fixed inter-iteration delay that steps through a phase schedule.
        /// Tests whether holding the device below its throttle point sustains
        /// more throughput than letting the governor throttle it.
        /// See runThermalSelfLimitBenchmark.
        case thermalSelfLimit
        /// Per-op / per-phase decomposition (h11), Tier 1: the token grid run
        /// twice (cool pass, then hot pass), each cell measured both FUSED
        /// (stock trainer, the validity control) and BARRIERED (six phases
        /// separated by explicit evals). See runPerOpBenchmark.
        case perop
        /// Per-op (h11), Tier 2: a single 500-token iteration with
        /// `GPU.startCapture` brackets around each phase's eval, producing
        /// three `.gputrace` bundles for hand analysis in Xcode's Metal
        /// debugger. Separate launch — capture perturbs timing.
        /// See runPerOpCaptureBenchmark.
        case peropCapture
        /// Numerical verification of the vendored MLX patch that routes
        /// non-transposed quantized matmul (backward's `dX`) onto the NAX
        /// kernel. No model load, no training — synthesises weights and
        /// compares kernels against dequantized references.
        /// See runNaxVerifyBenchmark.
        case verifyQmmN
        /// NAX A/B (follow-on to h11): the per-op decomposition run on an
        /// ALIGNED token grid with `MLX_ENABLE_NAX_N` alternated per iteration,
        /// so each cell yields paired on/off measurements at the same die
        /// temperature. See runPerOpBenchmark(idleMinutes:naxAB:).
        case naxAB
        /// Task-adapter training (h12): the Per-Task-LoRA (LaMP-7) trained to
        /// completion on-device with the canonical task recipe (r=4, all seven
        /// projections, cosine LR, effective batch 32 via accumulation) on
        /// pre-tokenized side-loaded data with an assistant-masked loss.
        /// `--max-steps 20` is the smoke form. See runTaskAdapterBenchmark.
        case taskAdapter
    }

    /// Value of a `--flag <value>` launch arg, or nil if absent/trailing.
    /// `nonisolated`: reads only the process-global `CommandLine.arguments`,
    /// and the NAX-arm accessors below are needed from `nonisolated static`
    /// record builders.
    private nonisolated static func launchArgValue(_ flag: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    /// `--nax-arm on|off` — NAX A/B arm for an E2E run. Sets MLX_ENABLE_NAX_N,
    /// tags every record, routes to a separate JSONL and an arm-specific adapter
    /// path, and enables the seeded batch shuffle so both arms see an IDENTICAL
    /// batch sequence (otherwise the loss curves would differ by data order
    /// rather than by the kernel under test). Absent → ordinary E2E run.
    ///
    /// 2026-08-11 NAX-ON rerun campaign: this arg is now honoured GLOBALLY —
    /// `runTrainBenchmark` sets `MLX_ENABLE_NAX_N` once at entry for EVERY
    /// mode, each mode's base record carries `nax_arm`, its `app_build` gains
    /// a `-nax-<arm>` suffix, and its on-device JSONL routes to a `_nax-<arm>`
    /// sibling file (see `naxArmFileName`) so pre-campaign data is never mixed
    /// into a rerun pull. Absent → byte-identical behaviour to every prior
    /// round.
    nonisolated static var trainBenchmarkNaxArm: String? {
        guard let v = launchArgValue("--nax-arm"), v == "on" || v == "off" else { return nil }
        return v
    }

    /// `--pin-arms` — variant of `--benchmark-nax-ab` where the arm is held
    /// CONSTANT within each sub-block instead of alternating per iteration.
    /// Exists to close the 1.93x (E2E, one arm throughout) vs ~1.56x (per-op,
    /// per-iteration alternation) discrepancy: if switching the dispatch arm
    /// costs anything (recompiled graphs, cache state), the alternating design
    /// UNDERSTATED the speedup, and this design measures it without switching
    /// while keeping the cells thermally paired at the block level.
    nonisolated static var trainBenchmarkPinArms: Bool {
        CommandLine.arguments.contains("--pin-arms")
    }

    /// On-device JSONL filename for the current global NAX arm:
    /// "x.jsonl" → "x_nax-on.jsonl". Identity when `--nax-arm` is absent, so
    /// every existing round's file routing is untouched. A separate FILE (not
    /// just a tag) because the Mac-side aggregators summarise whole files —
    /// appending rerun records to the original JSONLs would mix kernels in one
    /// pull.
    nonisolated static func naxArmFileName(_ name: String) -> String {
        guard let arm = trainBenchmarkNaxArm else { return name }
        guard name.hasSuffix(".jsonl") else { return name + "_nax-\(arm)" }
        return String(name.dropLast(".jsonl".count)) + "_nax-\(arm).jsonl"
    }

    /// `app_build` for the current global NAX arm: `base` → `base-nax-on`.
    /// Identity when `--nax-arm` is absent.
    nonisolated static func naxArmAppBuild(_ base: String) -> String {
        guard let arm = trainBenchmarkNaxArm else { return base }
        return base + "-nax-\(arm)"
    }

    /// `--user <fingerprint>` — which side-loaded per-user dataset to train (E2E).
    static var trainBenchmarkUser: String? { launchArgValue("--user") }

    /// `--condition <label>` — staged physical condition (C0/C1/C2/C4); logged
    /// verbatim in every E2E record. Defaults to "unlabeled".
    static var trainBenchmarkCondition: String { launchArgValue("--condition") ?? "unlabeled" }

    /// `--max-iters <N>` — cap the iteration count (for the SHORT smoke run of
    /// execution-order step 1). Absent → full 3×n_user.
    static var trainBenchmarkMaxIters: Int? {
        guard let v = launchArgValue("--max-iters") else { return nil }
        return Int(v)
    }

    /// `--max-steps <N>` — cap the OPTIMIZER-step count for the h12 task-adapter
    /// run (the smoke protocol runs 20). Distinct from `--max-iters`, which caps
    /// E2E microbatch iterations. The LR schedule is always computed against the
    /// FULL-corpus step count, so a smoke run is literally the first N steps of
    /// the real schedule.
    static var trainBenchmarkMaxSteps: Int? {
        guard let v = launchArgValue("--max-steps") else { return nil }
        return Int(v)
    }

    /// `--granularity-k <K>` — checkpoint group size for this process's single
    /// cell (h8). Must be a divisor of 36; validated in runGranularityBenchmark.
    static var trainBenchmarkGranularityK: Int? {
        guard let v = launchArgValue("--granularity-k") else { return nil }
        return Int(v)
    }

    /// `--baseline-duration-seconds <N>` — target wall-clock duration for the
    /// idle energy baseline (h9). Required for `.idleBaseline`; no default,
    /// since it should be chosen to roughly match its paired training run's
    /// expected duration (Mac-side orchestration's job, not baked in here).
    static var trainBenchmarkBaselineDurationSeconds: Double? {
        guard let v = launchArgValue("--baseline-duration-seconds") else { return nil }
        return Double(v)
    }

    /// `--soak-minutes <M>` — h10 heat-soak duration. Defaults to Run A's 60
    /// minutes; Run C of the matrix passes 10.
    static var trainBenchmarkSoakMinutes: Double {
        guard let v = launchArgValue("--soak-minutes"), let d = Double(v), d > 0 else {
            return TrainBenchConstants.thermalDefaultSoakMinutes
        }
        return d
    }

    /// `--burst-minutes <M>` / `--rest-seconds <S>` / `--cycles <N>` — the
    /// h10c sustained-cycling schedule. Defaults are 10 min / 120 s / 6.
    static var trainBenchmarkBurstMinutes: Double {
        guard let v = launchArgValue("--burst-minutes"), let d = Double(v), d > 0 else {
            return TrainBenchConstants.thermalCycleBurstSeconds / 60.0
        }
        return d
    }

    static var trainBenchmarkRestSeconds: Double {
        guard let v = launchArgValue("--rest-seconds"), let d = Double(v), d >= 0 else {
            return TrainBenchConstants.thermalCycleRestSeconds
        }
        return d
    }

    static var trainBenchmarkCycles: Int {
        guard let v = launchArgValue("--cycles"), let n = Int(v), n > 0 else {
            return TrainBenchConstants.thermalCycleCount
        }
        return n
    }

    /// `--selflimit-delay <D>` / `--selflimit-minutes <M>` — run the
    /// self-limiting arm as a SINGLE phase at a fixed delay instead of the
    /// built-in ascending schedule. This is the cold-start form of the
    /// experiment: the ascending schedule can only show whether pacing cools
    /// an already-throttled device, which is a different question from whether
    /// pacing prevents throttling in the first place. Both args must be given
    /// together; otherwise the built-in schedule is used.
    static var trainBenchmarkSelfLimitSinglePhase: (delay: Double, seconds: Double)? {
        guard let d = launchArgValue("--selflimit-delay"), let delay = Double(d),
            let m = launchArgValue("--selflimit-minutes"), let mins = Double(m),
            delay >= 0, mins > 0
        else { return nil }
        return (delay: delay, seconds: mins * 60.0)
    }

    /// `--idle-minutes <M>` — h11 honest-user-input approximate idle time since
    /// the device's last heavy use, recorded VERBATIM in `run_start` and never
    /// inferred. Exists because h10 Run B showed the cold-reference probe
    /// (which measures die temperature) is necessary but NOT sufficient as a
    /// cross-run comparability check: three sessions' cold refs agreed within
    /// 2% while the bursts that followed differed 5.4% in total work, the
    /// difference being how deeply idle the device had been beforehand.
    /// `nil` when not passed — recorded as null, not as a guess.
    static var trainBenchmarkIdleMinutes: Double? {
        guard let v = launchArgValue("--idle-minutes") else { return nil }
        return Double(v)
    }

    /// `--capture-backward-layers <K>` — capture the backward of only the TOP
    /// K transformer blocks (see the partial-backward note in
    /// `runPerOpCaptureBenchmark`). Absent → full backward, which is not
    /// replayable on this device.
    static var trainBenchmarkCaptureBackwardLayers: Int? {
        guard let v = launchArgValue("--capture-backward-layers"), let n = Int(v), n > 0 else {
            return nil
        }
        return n
    }

    /// Transformer-block index embedded in a parameter path such as
    /// `model.layers.31.self_attn.q_proj.lora_a`. `nil` for parameters that are
    /// not inside a block (lm_head, embeddings) — those are always kept, since
    /// the loss/lm_head backward is part of any backward pass.
    private nonisolated static func layerIndex(in key: String) -> Int? {
        let parts = key.split(separator: ".")
        guard let i = parts.firstIndex(of: "layers"), i + 1 < parts.count else { return nil }
        return Int(parts[i + 1])
    }

    /// `--capture-tokens <N>` — h11 Tier-2 token count. Defaults to 500 (the
    /// h7/h10 canonical anchor); the pre-registered fallback if a 500-token
    /// capture is too large or fails is to retry at 250.
    static var trainBenchmarkCaptureTokens: Int {
        guard let v = launchArgValue("--capture-tokens"), let n = Int(v), n > 0 else {
            return TrainBenchConstants.peropCaptureTokens
        }
        return n
    }

    /// `--probe-interval-s <S>` — h10 probe cadence during the observation
    /// window. Defaults to Run A's 120s; Run B (the self-heating control)
    /// passes 240.
    static var trainBenchmarkProbeIntervalSeconds: Double {
        guard let v = launchArgValue("--probe-interval-s"), let d = Double(v), d > 0 else {
            return TrainBenchConstants.thermalDefaultProbeIntervalSeconds
        }
        return d
    }

    // MARK: - Per-window sample (collected off-actor, written on main)

    /// One stepsPerReport window. All fields value types → `Sendable`, so the
    /// array crosses the `perform` isolation boundary in `TrainCellResult`.
    struct TrainWindowSample: Sendable {
        let step: Int
        let iterPerSec: Double
        let tokPerSec: Double
        let elapsedS: Double
        let peakMemBytes: Int
        let thermalState: String
        let lowPowerMode: Bool
    }

    struct TrainCellResult: Sendable {
        let samples: [TrainWindowSample]
    }

    /// Battery snapshot taken on the main actor at a cell boundary.
    private struct BatterySnapshot {
        let level: Double
        let charging: Bool
    }

    // MARK: - Entry point

    /// Run the training benchmark and exit. Safe to call once on launch.
    func runTrainBenchmark(mode: TrainBenchLaunchMode) async {
        // 2026-08-11 NAX-ON rerun campaign: honour `--nax-arm on|off` for EVERY
        // mode by setting the dispatch env once at entry. Modes that manage the
        // arm themselves (`.naxAB` alternation, `.e2e`'s own setenv) simply
        // overwrite it — harmless. Absent arg → env untouched → stock dispatch,
        // byte-identical to every prior round.
        if let arm = Self.trainBenchmarkNaxArm {
            setenv("MLX_ENABLE_NAX_N", arm == "on" ? "1" : "0", 1)
            tlog("global NAX arm=\(arm) (records tagged nax_arm, JSONLs suffixed _nax-\(arm))")
        }
        // NAX qmm_n verification: no model, no training, seconds not minutes.
        // Checked first so it can never be shadowed by the model-loading paths.
        if mode == .verifyQmmN {
            await runNaxVerifyBenchmark()
            return
        }
        // Idle energy baseline (h9) is a separate orchestration path (no
        // model, no training); the cap-sweep below is untouched.
        if mode == .idleBaseline {
            await runIdleBaselineBenchmark(
                user: Self.trainBenchmarkUser,
                condition: Self.trainBenchmarkCondition,
                durationSeconds: Self.trainBenchmarkBaselineDurationSeconds)
            return
        }
        // E2E (h5) is a separate orchestration path (real user, to completion,
        // save adapter, timed battery); the cap-sweep below is untouched.
        if mode == .e2e {
            await runE2ETrainBenchmark(
                user: Self.trainBenchmarkUser,
                condition: Self.trainBenchmarkCondition,
                maxIters: Self.trainBenchmarkMaxIters)
            return
        }
        // Token-time (h7) is also a separate orchestration path (synthetic
        // exact-length sweep); the cap-sweep below is untouched. Two regimes,
        // two entry points — see their doc comments.
        if mode == .tokentime {
            await runTokenTimeBenchmark()
            return
        }
        if mode == .tokentimeCold {
            await runTokenTimeColdBenchmark()
            return
        }
        // Granularity (h8) is also a separate orchestration path (one fixed-K
        // cell per process launch); the cap-sweep below is untouched.
        if mode == .granularity {
            await runGranularityBenchmark(k: Self.trainBenchmarkGranularityK)
            return
        }
        // Thermal cooldown (h10) is also a separate orchestration path
        // (cold-ref probe → soak → probed observation window); the cap-sweep
        // below is untouched.
        if mode == .thermalCooldown {
            await runThermalCooldownBenchmark(
                soakMinutes: Self.trainBenchmarkSoakMinutes,
                probeIntervalSeconds: Self.trainBenchmarkProbeIntervalSeconds)
            return
        }
        if mode == .thermalSelfLimit {
            await runThermalSelfLimitBenchmark()
            return
        }
        // Per-op decomposition (h11) — two separate orchestration paths (the
        // Tier-1 sweep and the Tier-2 Metal capture); everything above and the
        // cap-sweep below are untouched.
        if mode == .naxAB {
            await runPerOpBenchmark(
                idleMinutes: Self.trainBenchmarkIdleMinutes, naxAB: true)
            return
        }
        if mode == .perop {
            await runPerOpBenchmark(idleMinutes: Self.trainBenchmarkIdleMinutes)
            return
        }
        if mode == .peropCapture {
            await runPerOpCaptureBenchmark(targetTokens: Self.trainBenchmarkCaptureTokens)
            return
        }
        // Task-adapter (h12) — separate orchestration path (pre-tokenized data,
        // custom accumulation loop); everything above and below is untouched.
        if mode == .taskAdapter {
            await runTaskAdapterBenchmark(maxSteps: Self.trainBenchmarkMaxSteps)
            return
        }
        if mode == .thermalCycle {
            await runThermalCycleBenchmark(
                burstMinutes: Self.trainBenchmarkBurstMinutes,
                restSeconds: Self.trainBenchmarkRestSeconds,
                cycles: Self.trainBenchmarkCycles)
            return
        }

        enableThinking = false

        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        tlog("start session=\(sessionId) mode=\(mode) build=\(TrainBenchConstants.appBuild)")
        benchLogLine(
            "train-benchmark start session=\(sessionId) mode=\(mode) "
                + "build=\(TrainBenchConstants.appBuild)")

        guard
            let trainData = Self.loadBundledLoRAData(TrainBenchConstants.trainResource),
            let validData = Self.loadBundledLoRAData(TrainBenchConstants.validResource)
        else {
            tlog("FAILED to load bundled LoRA data")
            benchLogLine("train-benchmark FAILED to load bundled LoRA data")
            finishTrainBenchmark()
            return
        }
        tlog("loaded train=\(trainData.count) valid=\(validData.count) examples")
        benchLogLine("loaded train=\(trainData.count) valid=\(validData.count) examples")

        // PRIMARY AXIS = sequence-length cap at batchSize=1 (revised 2026-06-29).
        // The original batch-size sweep is moot: naive batch-1 training at full
        // deployment sequence lengths (193–1511 tok) SIGKILLs (jetsam) on the
        // FIRST backward step. So we sweep an ascending token cap to find the
        // feasible boundary + the OOM threshold. Records persist to the JSONL
        // per window, so when a cap OOMs (uncatchable SIGKILL) every smaller cap
        // already on disk survives — one launch yields the whole curve. A
        // `cap_start` sentinel is written before each cell so the OOM'd cap
        // (sentinel present, no train records) is pinpointable.
        let caps =
            (mode == .stress)
            ? [TrainBenchConstants.stressSeqCap] : TrainBenchConstants.seqCaps

        for (i, cap) in caps.enumerated() {
            await runTrainCell(
                batchSize: TrainBenchConstants.trainBatchSize, seqCap: cap,
                trainData: trainData, validData: validData, sessionId: sessionId)
            if mode == .full && i < caps.count - 1 {
                await trainCooldown()
            }
        }

        tlog("train-benchmark complete session=\(sessionId)")
        benchLogLine("train-benchmark complete session=\(sessionId)")
        finishTrainBenchmark()
    }

    /// Truncate each rendered example to at most `cap` tokens (token space, via
    /// the model tokenizer), then decode back to a string for `LoRABatchIterator`
    /// to re-tokenize. This caps the autograd-graph memory (≈ linear in sequence
    /// length) so the naive config can run. The assistant turn / EOS may be cut
    /// — irrelevant to a COST benchmark (loss is not recorded); only the
    /// per-step compute/memory cost, which depends on sequence length, matters.
    private nonisolated static func capExamples(
        _ data: [String], cap: Int, tokenizer: Tokenizer
    ) -> [String] {
        data.map { s in
            let toks = tokenizer.encode(text: s)
            guard toks.count > cap else { return s }
            return tokenizer.decode(tokenIds: Array(toks.prefix(cap)))
        }
    }

    // MARK: - Cell runner

    /// One run cell: a FRESH base model + fresh LoRA adapters, trained for
    /// `iterations` steps at `batchSize`. Reloading per cell ensures each batch
    /// size is measured from the same base state (no accumulated adapter drift).
    /// Model load is outside the measured window.
    private func runTrainCell(
        batchSize: Int, seqCap: Int, trainData: [String], validData: [String],
        sessionId: String
    ) async {
        tlog("cell bs=\(batchSize) cap=\(seqCap): loading model ...")
        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("cell bs=\(batchSize) cap=\(seqCap): model load failed: \(error)")
            benchLogLine("cell bs=\(batchSize) cap=\(seqCap): model load failed: \(error)")
            return
        }
        tlog("cell bs=\(batchSize) cap=\(seqCap): model loaded")

        let batteryStart = Self.batterySnapshot()
        // Sentinel BEFORE training: if the cell then OOMs (SIGKILL, no train
        // records), this marks which cap was in flight.
        writeCapStartRecord(
            batchSize: batchSize, seqCap: seqCap, sessionId: sessionId,
            battery: batteryStart, nTrain: trainData.count)

        let iterations = TrainBenchConstants.iterations
        let stepsPerReport = TrainBenchConstants.stepsPerReport

        // decision 9: wrap LoRA-apply + train in do/catch. OOM may surface as a
        // thrown MLX allocation error (caught here) — or, on Metal, as a hard
        // abort that no Swift catch can intercept. Best-effort.
        do {
            let result = try await container.perform {
                ctx throws -> TrainCellResult in

                // Apply OPPU LoRA (r=8, q+v only, all 28 layers). Mutates the
                // model in place: freezes base, replaces q/v projections.
                let config = LoRAConfiguration(
                    numLayers: TrainBenchConstants.loraLayers,
                    loraParameters: .init(
                        rank: TrainBenchConstants.loraRank,
                        scale: TrainBenchConstants.loraScale,
                        keys: TrainBenchConstants.loraKeys))
                _ = try LoRAContainer.from(model: ctx.model, configuration: config)

                // h4: enable per-transformer-block gradient checkpointing on the
                // model (no-op for any non-SmolLM3 model). The model's forward
                // reads each block's trainable params at call time, so this is
                // set after LoRA is applied. See TrainBenchConstants /
                // SmolLM3Model.checkpointGroupSize.
                if TrainBenchConstants.gradientCheckpointing {
                    (ctx.model as? SmolLM3Model)?.checkpointGroupSize = 1
                }

                // Cap sequence length (see capExamples). Done inside perform so
                // the model tokenizer is in scope.
                let capTrain = Self.capExamples(
                    trainData, cap: seqCap, tokenizer: ctx.tokenizer)
                let capValid = Self.capExamples(
                    validData, cap: seqCap, tokenizer: ctx.tokenizer)
                self.tlog("cell bs=\(batchSize) cap=\(seqCap): LoRA applied, starting train")

                let params = LoRATrain.Parameters(
                    batchSize: batchSize,
                    iterations: iterations,
                    stepsPerReport: stepsPerReport,
                    // Suppress periodic validation (one still fires at iter 0).
                    stepsPerEval: iterations + 1,
                    validationBatches: 0,
                    // No periodic saves — we don't keep the weights.
                    saveEvery: iterations + 1,
                    adapterURL: nil)

                let optimizer = Adam(learningRate: 1e-5)

                var samples: [TrainWindowSample] = []
                let loopStart = Date.timeIntervalSinceReferenceDate
                // decision 5: arm the first window's peak counter just before
                // the loop; re-armed at the END of each callback below.
                GPU.resetPeakMemory()

                try LoRATrain.train(
                    model: ctx.model, train: capTrain, validate: capValid,
                    optimizer: optimizer, tokenizer: ctx.tokenizer, parameters: params
                ) { progress in
                    switch progress {
                    case .train(let iter, _, let ips, let tps):
                        samples.append(
                            TrainWindowSample(
                                step: iter + 1,
                                iterPerSec: ips,
                                tokPerSec: tps,
                                elapsedS: Date.timeIntervalSinceReferenceDate - loopStart,
                                peakMemBytes: Memory.snapshot().peakMemory,
                                thermalState: Self.thermalString(),
                                lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled))
                        if iter + 1 == TrainBenchConstants.stepsPerReport {
                            self.tlog(
                                "cell bs=\(batchSize) cap=\(seqCap): first window @step "
                                    + "\(iter + 1) ips=\(ips) tps=\(tps)")
                        }
                        // Re-arm: the next window's peak starts from here
                        // (decision 5 / implementation verification: reset AFTER
                        // reading so the just-read peak isn't clobbered).
                        GPU.resetPeakMemory()
                    case .validation, .save:
                        break
                    }
                    return .more
                }

                return TrainCellResult(samples: samples)
            }

            let batteryEnd = Self.batterySnapshot()
            for s in result.samples {
                writeTrainRecord(
                    sample: s, batchSize: batchSize, seqCap: seqCap, sessionId: sessionId,
                    batteryStart: batteryStart, batteryEnd: batteryEnd,
                    nTrain: trainData.count)
            }
            tlog("cell bs=\(batchSize) cap=\(seqCap) complete windows=\(result.samples.count)")
            benchLogLine(
                "cell bs=\(batchSize) cap=\(seqCap) complete windows=\(result.samples.count) "
                    + "lastThermal=\(result.samples.last?.thermalState ?? "n/a")")
        } catch {
            tlog("cell bs=\(batchSize) cap=\(seqCap): training error (possible OOM): \(error)")
            benchLogLine(
                "cell bs=\(batchSize) cap=\(seqCap): training error (possible OOM): \(error)")
            let batteryEnd = Self.batterySnapshot()
            writeOOMRecord(
                batchSize: batchSize, seqCap: seqCap, sessionId: sessionId,
                batteryStart: batteryStart, batteryEnd: batteryEnd,
                nTrain: trainData.count)
        }
    }

    /// Cooldown gated on `thermalState == nominal`, capped (decision 8).
    /// `capSeconds` defaults to the original h3/h4 cap so existing callers are
    /// unaffected; the token-time COLD regime (h7 v4) passes a much larger
    /// value — see `tokentimeColdCooldownCapSeconds`'s doc comment for why.
    private func trainCooldown(capSeconds: Double = TrainBenchConstants.cooldownCapSeconds) async {
        let start = Date.timeIntervalSinceReferenceDate
        while ProcessInfo.processInfo.thermalState != .nominal {
            let elapsed = Date.timeIntervalSinceReferenceDate - start
            if elapsed > capSeconds {
                tlog("cooldown cap hit after \(Int(elapsed))s (still \(Self.thermalString()))")
                benchLogLine("train cooldown cap hit (still non-nominal)")
                break
            }
            try? await Task.sleep(for: .seconds(2))
        }
    }

    private func finishTrainBenchmark() {
        tlog("exiting after train benchmark")
        benchLogLine("exiting after train benchmark")
        exit(0)
    }

    // MARK: - E2E (h5) — real per-user to-completion training

    /// Immutable per-run context, built once on the main actor and passed into
    /// the nonisolated record builders (so the off-actor train callback and the
    /// main-actor battery sampler stamp identical run metadata).
    struct E2ERunContext: Sendable {
        let user: String
        let profileSize: Int
        let condition: String
        let nUser: Int
        let iterations: Int
        let seqCap: Int
        let modelName: String
        let sessionId: String
        /// NAX A/B arm for this run ("on"/"off"), or nil for a normal E2E run.
        /// When set, records go to a SEPARATE JSONL and the adapter path is
        /// arm-specific, so the h5 backlog data and adapters are untouched.
        var naxArm: String? = nil
    }

    /// Train ONE real user's LaMP-3 User-LoRA to completion (3 epochs =
    /// `3 × n_user` iterations, or `maxIters` for the smoke run), stacked on the
    /// fused A1-lamp 4-bit base, with the faithful R5 recipe (AdamW wd=0.01, r=8
    /// q+v, GC on, cap 1024). Saves the adapter, captures training loss per
    /// window, and samples battery on a wall-clock timer. Records are written
    /// INCREMENTALLY so a jetsam in a multi-hour run leaves every completed
    /// window on disk. Then exits.
    func runE2ETrainBenchmark(user: String?, condition: String, maxIters: Int?) async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        guard let user, !user.isEmpty else {
            tlog("E2E: missing --user <fingerprint>")
            benchLogLine("E2E FAILED: missing --user <fingerprint>")
            finishTrainBenchmark()
            return
        }
        // NAX A/B: set the dispatch arm and pin the batch order BEFORE any
        // model load or training so both runs are identical apart from the
        // kernel. The env read in the vendored MLX is live (not cached), and
        // dispatch is decided at eval time, so setting it here covers the run.
        let naxArm = Self.trainBenchmarkNaxArm
        if let naxArm {
            setenv("MLX_ENABLE_NAX_N", naxArm == "on" ? "1" : "0", 1)
            LoRATrain.shuffleSeed = TrainBenchConstants.naxABShuffleSeed
            tlog("E2E NAX arm=\(naxArm) shuffleSeed=\(TrainBenchConstants.naxABShuffleSeed)")
        }
        tlog("E2E start session=\(sessionId) user=\(user) condition=\(condition) "
            + "build=\(TrainBenchConstants.appBuild)")
        benchLogLine("E2E start session=\(sessionId) user=\(user) condition=\(condition)")

        // Side-loaded per-user data from Documents (pushed via devicectl copy).
        guard let trainData = Self.loadE2EUserData(user: user), !trainData.isEmpty else {
            tlog("E2E FAILED to load user data for \(user) "
                + "(expected Documents/\(TrainBenchConstants.e2eDataDirName)/lamp3_\(user).jsonl)")
            benchLogLine("E2E FAILED to load user data for \(user)")
            finishTrainBenchmark()
            return
        }
        let nUser = trainData.count
        var iterations = TrainBenchConstants.e2eEpochs * nUser
        if let cap = maxIters, cap > 0 { iterations = min(iterations, cap) }
        tlog("E2E user=\(user) nUser=\(nUser) iterations=\(iterations) "
            + "(maxIters=\(maxIters.map(String.init) ?? "none"))")
        benchLogLine("E2E loaded nUser=\(nUser) iterations=\(iterations)")

        // Adapter save path (Documents/e2e_adapters/adapter_<fp>.safetensors).
        let adapterURL = Self.e2eAdapterURL(user: user, naxArm: naxArm)
        try? FileManager.default.createDirectory(
            at: adapterURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        // Load model (outside the measured window).
        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("E2E model load failed: \(error)")
            benchLogLine("E2E model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        tlog("E2E model loaded")

        let modelName =
            modelConfiguration.name.components(separatedBy: "/").last
            ?? modelConfiguration.name
        var ctx = E2ERunContext(
            user: user, profileSize: nUser, condition: condition, nUser: nUser,
            iterations: iterations, seqCap: TrainBenchConstants.e2eSeqCap,
            modelName: modelName, sessionId: sessionId)
        ctx.naxArm = naxArm

        // A wall-clock origin shared by every elapsed_s column (battery + train).
        let benchStart = Date.timeIntervalSinceReferenceDate

        let batteryStart = Self.batterySnapshot()
        Self.appendE2EMarker(
            ctx, recordType: "run_start",
            extra: [
                "battery_level": batteryStart.level,
                "charging": batteryStart.charging,
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                "thermal_state": Self.thermalString(),
                "adapter_url": adapterURL.lastPathComponent,
            ])

        // Timed battery sampler on the main actor (UIDevice is @MainActor). Runs
        // concurrently with the off-actor training loop; cancelled at the end.
        let batterySampler = Task { @MainActor in
            // h9: threaded across iterations so each sample's cpu_util_pct is
            // %busy since the PREVIOUS sample, not since boot.
            var previousCPUTicks = Self.cpuTicks()
            while !Task.isCancelled {
                let snap = Self.batterySnapshot()
                let (cpuPct, newTicks) = Self.cpuUtilizationPercent(previous: previousCPUTicks)
                previousCPUTicks = newTicks
                Self.appendE2EBattery(
                    ctx,
                    elapsed: Date.timeIntervalSinceReferenceDate - benchStart,
                    level: snap.level, charging: snap.charging,
                    thermal: Self.thermalString(),
                    lpm: ProcessInfo.processInfo.isLowPowerModeEnabled,
                    peak: Memory.snapshot().peakMemory,
                    cpuUtilPct: cpuPct)
                try? await Task.sleep(for: .seconds(TrainBenchConstants.e2eBatterySampleSeconds))
            }
        }

        var trainError: String? = nil
        do {
            try await container.perform { mc in
                // OPPU LoRA: r=8, q+v only, all 28 layers (fresh adapter stacked
                // on the already-fused A1-lamp base weights).
                let config = LoRAConfiguration(
                    numLayers: TrainBenchConstants.loraLayers,
                    loraParameters: .init(
                        rank: TrainBenchConstants.loraRank,
                        scale: TrainBenchConstants.loraScale,
                        keys: TrainBenchConstants.loraKeys))
                _ = try LoRAContainer.from(model: mc.model, configuration: config)

                // Per-block gradient checkpointing (h4 infra) — required to fit
                // real seq lengths at cap 1024.
                if TrainBenchConstants.gradientCheckpointing {
                    (mc.model as? SmolLM3Model)?.checkpointGroupSize = 1
                }

                let capTrain = Self.capExamples(
                    trainData, cap: ctx.seqCap, tokenizer: mc.tokenizer)
                // 1-example validation stub: LoRATrain forces one validation at
                // iter 0 regardless of stepsPerEval; its event is ignored.
                let capValid = Array(capTrain.prefix(1))
                self.tlog("E2E LoRA applied, starting train iterations=\(iterations)")

                let params = LoRATrain.Parameters(
                    batchSize: TrainBenchConstants.e2eBatchSize,
                    iterations: iterations,
                    stepsPerReport: TrainBenchConstants.e2eStepsPerReport,
                    stepsPerEval: iterations + 1,
                    validationBatches: 0,
                    // Save once at the final iteration: (iter+1) % saveEvery == 0
                    // fires exactly at the last step.
                    saveEvery: iterations,
                    adapterURL: adapterURL)

                // Faithful R5 optimizer: AdamW, LR 1e-5, wd 0.01, bias-corrected
                // to match `adamw_torch`.
                let optimizer = AdamW(
                    learningRate: TrainBenchConstants.e2eLearningRate,
                    weightDecay: TrainBenchConstants.e2eWeightDecay,
                    biasCorrection: TrainBenchConstants.e2eAdamBiasCorrection)

                GPU.resetPeakMemory()
                try LoRATrain.train(
                    model: mc.model, train: capTrain, validate: capValid,
                    optimizer: optimizer, tokenizer: mc.tokenizer, parameters: params
                ) { progress in
                    switch progress {
                    case .train(let iter, let loss, let ips, let tps):
                        Self.appendE2ETrainWindow(
                            ctx, step: iter + 1, loss: loss, ips: ips, tps: tps,
                            elapsed: Date.timeIntervalSinceReferenceDate - benchStart,
                            peak: Memory.snapshot().peakMemory,
                            thermal: Self.thermalString(),
                            lpm: ProcessInfo.processInfo.isLowPowerModeEnabled)
                        // Re-arm peak counter for the next window (read-then-reset).
                        GPU.resetPeakMemory()
                    case .save(let it, let url):
                        self.tlog("E2E saved adapter @iter \(it + 1) -> \(url.lastPathComponent)")
                    case .validation:
                        break
                    }
                    return .more
                }
            }
        } catch {
            trainError = "\(error)"
            tlog("E2E training error (possible OOM/jetsam): \(error)")
            benchLogLine("E2E training error: \(error)")
        }

        batterySampler.cancel()
        let batteryEnd = Self.batterySnapshot()
        let adapterSaved = FileManager.default.fileExists(atPath: adapterURL.path)
        Self.appendE2EMarker(
            ctx, recordType: trainError == nil ? "run_end" : "error",
            extra: [
                "battery_level": batteryStart.level,
                "battery_level_end": batteryEnd.level,
                "charging": batteryEnd.charging,
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                "thermal_state": Self.thermalString(),
                "elapsed_s": Date.timeIntervalSinceReferenceDate - benchStart,
                "adapter_saved": adapterSaved,
                "error": trainError ?? NSNull(),
            ])
        tlog("E2E complete user=\(user) adapter_saved=\(adapterSaved) error=\(trainError ?? "none")")
        benchLogLine("E2E complete user=\(user) adapter_saved=\(adapterSaved)")
        finishTrainBenchmark()
    }

    // MARK: - Idle energy baseline (h9)

    struct IdleBaselineContext: Sendable {
        let user: String
        let condition: String
        let sessionId: String
    }

    /// No model load, no training — just screen-on idle with the same
    /// battery/thermal/CPU sampling cadence as the E2E battery sampler, for a
    /// fixed target duration. Paired with a real C2 training run (same
    /// `--user`, matched starting charge band per the h9 design) so the
    /// training run's drain rate can have this baseline's non-training drain
    /// rate subtracted out. Writes into the SAME
    /// `train_bench_metrics_e2e.jsonl` as the E2E path (h9 extends h5's
    /// schema rather than adding a parallel file), tagged with distinct
    /// `record_type`s (`idle_baseline_start`/`idle_baseline`/
    /// `idle_baseline_end`) so an aggregator can't confuse it with a real
    /// training run.
    func runIdleBaselineBenchmark(user: String?, condition: String, durationSeconds: Double?) async {
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        guard let user, !user.isEmpty else {
            tlog("idle-baseline: missing --user <fingerprint>")
            benchLogLine("idle-baseline FAILED: missing --user <fingerprint>")
            finishTrainBenchmark()
            return
        }
        guard let durationSeconds, durationSeconds > 0 else {
            tlog("idle-baseline: missing/invalid --baseline-duration-seconds <N>")
            benchLogLine("idle-baseline FAILED: missing/invalid --baseline-duration-seconds")
            finishTrainBenchmark()
            return
        }
        tlog("idle-baseline start session=\(sessionId) user=\(user) condition=\(condition) "
            + "target_duration=\(durationSeconds)s build=\(TrainBenchConstants.appBuild)")
        benchLogLine(
            "idle-baseline start session=\(sessionId) user=\(user) duration=\(durationSeconds)s")

        let ctx = IdleBaselineContext(user: user, condition: condition, sessionId: sessionId)
        let benchStart = Date.timeIntervalSinceReferenceDate

        let batteryStart = Self.batterySnapshot()
        Self.appendIdleBaselineMarker(
            ctx, recordType: "idle_baseline_start",
            extra: [
                "battery_level": batteryStart.level,
                "charging": batteryStart.charging,
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                "thermal_state": Self.thermalString(),
                "target_duration_s": durationSeconds,
            ])

        var previousCPUTicks = Self.cpuTicks()
        while Date.timeIntervalSinceReferenceDate - benchStart < durationSeconds {
            try? await Task.sleep(for: .seconds(TrainBenchConstants.e2eBatterySampleSeconds))
            let snap = Self.batterySnapshot()
            let (cpuPct, newTicks) = Self.cpuUtilizationPercent(previous: previousCPUTicks)
            previousCPUTicks = newTicks
            Self.appendIdleBaselineSample(
                ctx,
                elapsed: Date.timeIntervalSinceReferenceDate - benchStart,
                level: snap.level, charging: snap.charging,
                thermal: Self.thermalString(),
                lpm: ProcessInfo.processInfo.isLowPowerModeEnabled,
                cpuUtilPct: cpuPct)
        }

        let elapsed = Date.timeIntervalSinceReferenceDate - benchStart
        let batteryEnd = Self.batterySnapshot()
        Self.appendIdleBaselineMarker(
            ctx, recordType: "idle_baseline_end",
            extra: [
                "battery_level": batteryStart.level,
                "battery_level_end": batteryEnd.level,
                "charging": batteryEnd.charging,
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                "thermal_state": Self.thermalString(),
                "elapsed_s": elapsed,
            ])
        tlog("idle-baseline complete user=\(user) elapsed=\(elapsed)s")
        benchLogLine("idle-baseline complete user=\(user) elapsed=\(elapsed)s")
        finishTrainBenchmark()
    }

    private nonisolated static func idleBaselineBaseRecord(
        _ c: IdleBaselineContext, recordType: String
    ) -> [String: Any] {
        [
            "record_type": recordType,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "user_fingerprint": c.user,
            "condition": c.condition,
            "app_build": TrainBenchConstants.appBuild,
            "bench_schema_version": TrainBenchConstants.schemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": c.sessionId,
            "device_model": trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    private nonisolated static func appendIdleBaselineSample(
        _ c: IdleBaselineContext, elapsed: Double, level: Double, charging: Bool,
        thermal: String, lpm: Bool, cpuUtilPct: Double?
    ) {
        var r = idleBaselineBaseRecord(c, recordType: "idle_baseline")
        r["elapsed_s"] = elapsed
        r["battery_level"] = level
        r["charging"] = charging
        r["thermal_state"] = thermal
        r["low_power_mode"] = lpm
        r["cpu_util_pct"] = cpuUtilPct ?? NSNull()
        emitE2E(r)
    }

    private nonisolated static func appendIdleBaselineMarker(
        _ c: IdleBaselineContext, recordType: String, extra: [String: Any]
    ) {
        var r = idleBaselineBaseRecord(c, recordType: recordType)
        for (k, v) in extra { r[k] = v }
        emitE2E(r)
    }

    // MARK: - E2E data + adapter paths

    /// Load the side-loaded `{"text": ...}` per-user dataset from
    /// `Documents/<e2eDataDirName>/lamp3_<fp>.jsonl` (no bundle fallback — the
    /// per-user data is device-local and pushed via devicectl).
    private nonisolated static func loadE2EUserData(user: String) -> [String]? {
        let url = URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.e2eDataDirName)
            .appendingPathComponent("lamp3_\(user).jsonl")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try? MLXLLM.loadLoRAData(url: url)
    }

    private nonisolated static func e2eAdapterURL(user: String, naxArm: String? = nil) -> URL {
        // Arm-specific filename so the two A/B runs do not overwrite each
        // other's adapter and the resulting models can be compared directly.
        let suffix = naxArm.map { "_nax-\($0)" } ?? ""
        return URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.e2eAdapterDirName)
            .appendingPathComponent("adapter_\(user)\(suffix).safetensors")
    }

    // MARK: - E2E record builders (nonisolated → callable from the train callback)

    /// Output file for this run: the NAX A/B round writes to its own JSONL so
    /// `train_bench_metrics_e2e.jsonl` (h5 backlog) stays untouched.
    private nonisolated static func e2eFileName(_ c: E2ERunContext) -> String? {
        c.naxArm == nil ? nil : TrainBenchConstants.naxABE2EMetricsFileName
    }

    private nonisolated static func e2eBaseRecord(_ c: E2ERunContext, recordType: String) -> [String: Any] {
        [
            "record_type": recordType,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "user_fingerprint": c.user,
            "nax_arm": c.naxArm ?? NSNull(),
            "shuffle_seed": c.naxArm == nil
                ? NSNull() : TrainBenchConstants.naxABShuffleSeed,
            "profile_size": c.profileSize,
            "condition": c.condition,
            "n_user": c.nUser,
            "iterations_total": c.iterations,
            "seq_cap": c.seqCap,
            "batch_size": TrainBenchConstants.e2eBatchSize,
            "epochs": TrainBenchConstants.e2eEpochs,
            "model": c.modelName,
            "lora_rank": TrainBenchConstants.loraRank,
            "lora_keys": TrainBenchConstants.loraKeysLabel,
            "num_lora_layers": TrainBenchConstants.loraLayers,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "checkpoint_granularity": TrainBenchConstants.checkpointGranularity,
            "optimizer": "adamw",
            "learning_rate": TrainBenchConstants.e2eLearningRate,
            "weight_decay": TrainBenchConstants.e2eWeightDecay,
            "adam_bias_correction": TrainBenchConstants.e2eAdamBiasCorrection,
            "steps_per_report": TrainBenchConstants.e2eStepsPerReport,
            "app_build": TrainBenchConstants.appBuild,
            "bench_schema_version": TrainBenchConstants.schemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": c.sessionId,
            "device_model": trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    private nonisolated static func appendE2ETrainWindow(
        _ c: E2ERunContext, step: Int, loss: Float, ips: Double, tps: Double,
        elapsed: Double, peak: Int, thermal: String, lpm: Bool
    ) {
        var r = e2eBaseRecord(c, recordType: "train")
        r["step"] = step
        r["training_loss"] = Double(loss)
        r["iter_per_sec"] = ips
        r["tok_per_sec"] = tps
        r["elapsed_s"] = elapsed
        r["peak_mem_bytes"] = peak
        r["thermal_state"] = thermal
        r["low_power_mode"] = lpm
        emitE2E(r, fileName: e2eFileName(c))
    }

    private nonisolated static func appendE2EBattery(
        _ c: E2ERunContext, elapsed: Double, level: Double, charging: Bool,
        thermal: String, lpm: Bool, peak: Int, cpuUtilPct: Double?
    ) {
        var r = e2eBaseRecord(c, recordType: "battery")
        r["elapsed_s"] = elapsed
        r["battery_level"] = level
        r["charging"] = charging
        r["thermal_state"] = thermal
        r["low_power_mode"] = lpm
        r["peak_mem_bytes"] = peak
        // h9: secondary diagnostic only, see cpuUtilizationPercent's doc comment.
        r["cpu_util_pct"] = cpuUtilPct ?? NSNull()
        emitE2E(r, fileName: e2eFileName(c))
    }

    private nonisolated static func appendE2EMarker(
        _ c: E2ERunContext, recordType: String, extra: [String: Any]
    ) {
        var r = e2eBaseRecord(c, recordType: recordType)
        for (k, v) in extra { r[k] = v }
        emitE2E(r, fileName: e2eFileName(c))
    }

    /// Serialize + append one E2E record to `train_bench_metrics_e2e.jsonl`.
    /// nonisolated + lock-guarded (see file-scope `e2eFileLock`): the off-actor
    /// train callback and the main-actor battery sampler both write concurrently.
    private nonisolated static func emitE2E(_ record: [String: Any], fileName: String? = nil) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(
            fileName ?? TrainBenchConstants.e2eMetricsFileName)
        e2eFileLock.lock()
        defer { e2eFileLock.unlock() }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                if let d = line.data(using: .utf8) { try handle.write(contentsOf: d) }
            } else {
                try line.write(to: url, atomically: true, encoding: .utf8)
            }
        } catch {
            // best-effort; a failed line must not crash a multi-hour run
        }
    }

    /// Unbuffered stderr line — visible in `devicectl ... launch --console`
    /// (os_log via `benchLogLine` is NOT). Used for milestone tracing so a hard
    /// jetsam/SIGKILL leaves a breadcrumb trail in the console.
    nonisolated func tlog(_ message: String) {
        FileHandle.standardError.write(Data(("[trainbench] " + message + "\n").utf8))
    }

    // MARK: - Token-time (h7) — per-iteration wall-time vs token count

    /// Run the token-time cost-model sweep: a discarded warm-up burst, then
    /// the token-count grid back-to-back with NO cooldown gate (locked
    /// design — this cost model is meant to represent real sustained E2E
    /// training, not a cool per-cell reset). Each cell trains a single
    /// synthetic example (exact token count) for
    /// `tokentimeIterationsPerCell` iterations with `stepsPerReport = 1` so
    /// every progress callback reports a true per-step (not window-averaged)
    /// rate. See experiments/2026-07-24-ondevice-tokentime-plan.md.
    func runTokenTimeBenchmark() async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        tlog("tokentime start session=\(sessionId) build=\(TrainBenchConstants.tokentimeAppBuild)")
        benchLogLine(
            "tokentime start session=\(sessionId) build=\(TrainBenchConstants.tokentimeAppBuild)")

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("tokentime model load failed: \(error)")
            benchLogLine("tokentime model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        tlog("tokentime model loaded")

        let modelName =
            modelConfiguration.name.components(separatedBy: "/").last
            ?? modelConfiguration.name

        await runTokenTimeWarmup(container: container, sessionId: sessionId)

        for tokens in TrainBenchConstants.tokentimeTokenCounts {
            await runTokenTimeCell(
                container: container, targetTokens: tokens, modelName: modelName,
                sessionId: sessionId, fileName: TrainBenchConstants.tokentimeMetricsFileName,
                regime: "hot_sustained_no_cooldown")
        }

        tlog("tokentime complete session=\(sessionId)")
        benchLogLine("tokentime complete session=\(sessionId)")
        finishTrainBenchmark()
    }

    /// COLD regime (v3): same grid/recipe/iterations as `runTokenTimeBenchmark`,
    /// but each cell is isolated by a cooldown-to-nominal gate before and
    /// after, instead of a warm-up burst + no cooldown. Explicit user request
    /// for the opposite thermal regime, to compare against the hot sweep.
    /// Writes to a SEPARATE JSONL (`tokentimeColdMetricsFileName`) so the two
    /// regimes' data never mix in one aggregation.
    func runTokenTimeColdBenchmark() async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        tlog(
            "tokentime-cold start session=\(sessionId) "
                + "build=\(TrainBenchConstants.tokentimeAppBuild)")
        benchLogLine(
            "tokentime-cold start session=\(sessionId) "
                + "build=\(TrainBenchConstants.tokentimeAppBuild)")

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("tokentime-cold model load failed: \(error)")
            benchLogLine("tokentime-cold model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        tlog("tokentime-cold model loaded")

        let modelName =
            modelConfiguration.name.components(separatedBy: "/").last
            ?? modelConfiguration.name

        // Ensure a clean nominal baseline before cell 1 too (the device may
        // still be hot from a prior run) — no warm-up burst here, that would
        // defeat the point of this regime.
        tlog("tokentime-cold: pre-sweep cooldown")
        await trainCooldown(capSeconds: TrainBenchConstants.tokentimeColdCooldownCapSeconds)

        for tokens in TrainBenchConstants.tokentimeTokenCounts {
            await runTokenTimeCell(
                container: container, targetTokens: tokens, modelName: modelName,
                sessionId: sessionId, fileName: TrainBenchConstants.tokentimeColdMetricsFileName,
                regime: "cold_isolated")
            tlog("tokentime-cold cell tokens=\(tokens): cooling down")
            await trainCooldown(capSeconds: TrainBenchConstants.tokentimeColdCooldownCapSeconds)
        }

        tlog("tokentime-cold complete session=\(sessionId)")
        benchLogLine("tokentime-cold complete session=\(sessionId)")
        finishTrainBenchmark()
    }

    /// Discarded throwaway training burst at `tokentimeWarmupSeqCap` for
    /// ~`tokentimeWarmupSeconds` (locked design: no records written) so the
    /// real sweep below starts already thermally representative of sustained
    /// training, rather than giving the smallest token count an artificial
    /// cold-start advantage.
    private func runTokenTimeWarmup(container: ModelContainer, sessionId: String) async {
        tlog(
            "tokentime warmup: starting (~\(TrainBenchConstants.tokentimeWarmupSeconds)s "
                + "@tokens=\(TrainBenchConstants.tokentimeWarmupSeqCap))")
        do {
            try await container.perform { ctx throws -> Void in
                let config = LoRAConfiguration(
                    numLayers: TrainBenchConstants.loraLayers,
                    loraParameters: .init(
                        rank: TrainBenchConstants.loraRank,
                        scale: TrainBenchConstants.loraScale,
                        keys: TrainBenchConstants.loraKeys))
                _ = try LoRAContainer.from(model: ctx.model, configuration: config)
                if TrainBenchConstants.gradientCheckpointing {
                    (ctx.model as? SmolLM3Model)?.checkpointGroupSize = 1
                }

                let example = Self.syntheticExample(
                    targetTokens: TrainBenchConstants.tokentimeWarmupSeqCap,
                    tokenizer: ctx.tokenizer)
                let data = [example]

                let params = LoRATrain.Parameters(
                    batchSize: TrainBenchConstants.trainBatchSize,
                    iterations: TrainBenchConstants.tokentimeWarmupMaxIterations,
                    stepsPerReport: 1,
                    stepsPerEval: TrainBenchConstants.tokentimeWarmupMaxIterations + 1,
                    validationBatches: 0,
                    saveEvery: TrainBenchConstants.tokentimeWarmupMaxIterations + 1,
                    adapterURL: nil)

                let optimizer = AdamW(
                    learningRate: TrainBenchConstants.e2eLearningRate,
                    weightDecay: TrainBenchConstants.e2eWeightDecay,
                    biasCorrection: TrainBenchConstants.e2eAdamBiasCorrection)

                let warmupStart = Date.timeIntervalSinceReferenceDate
                try LoRATrain.train(
                    model: ctx.model, train: data, validate: data,
                    optimizer: optimizer, tokenizer: ctx.tokenizer, parameters: params
                ) { progress in
                    switch progress {
                    case .train:
                        let elapsed = Date.timeIntervalSinceReferenceDate - warmupStart
                        if elapsed >= TrainBenchConstants.tokentimeWarmupSeconds {
                            return .stop
                        }
                    case .validation, .save:
                        break
                    }
                    return .more
                }
            }
            tlog("tokentime warmup: complete")
            benchLogLine("tokentime warmup complete")
        } catch {
            // Best-effort — a warmup failure shouldn't abort the real sweep;
            // each cell below applies its own fresh LoRA/GC state regardless.
            tlog("tokentime warmup error (continuing to sweep anyway): \(error)")
            benchLogLine("tokentime warmup error (continuing): \(error)")
        }
    }

    /// One token-length cell: `tokentimeIterationsPerCell` iterations on a
    /// single synthetic example of exactly `targetTokens` tokens, discarding
    /// iteration index 0 (compile warm-up + the forced iteration-0
    /// validation pass folded into the same window). Records are written
    /// incrementally (per kept step) so a mid-sweep failure on a later cell
    /// leaves every earlier cell's data on disk.
    private func runTokenTimeCell(
        container: ModelContainer, targetTokens: Int, modelName: String, sessionId: String,
        fileName: String, regime: String
    ) async {
        tlog("tokentime cell tokens=\(targetTokens): starting")
        do {
            try await container.perform { ctx throws -> Void in
                let config = LoRAConfiguration(
                    numLayers: TrainBenchConstants.loraLayers,
                    loraParameters: .init(
                        rank: TrainBenchConstants.loraRank,
                        scale: TrainBenchConstants.loraScale,
                        keys: TrainBenchConstants.loraKeys))
                _ = try LoRAContainer.from(model: ctx.model, configuration: config)
                if TrainBenchConstants.gradientCheckpointing {
                    (ctx.model as? SmolLM3Model)?.checkpointGroupSize = 1
                }

                let example = Self.syntheticExample(
                    targetTokens: targetTokens, tokenizer: ctx.tokenizer)
                let data = [example]

                let params = LoRATrain.Parameters(
                    batchSize: TrainBenchConstants.trainBatchSize,
                    iterations: TrainBenchConstants.tokentimeIterationsPerCell,
                    stepsPerReport: 1,
                    stepsPerEval: TrainBenchConstants.tokentimeIterationsPerCell + 1,
                    validationBatches: 0,
                    saveEvery: TrainBenchConstants.tokentimeIterationsPerCell + 1,
                    adapterURL: nil)

                let optimizer = AdamW(
                    learningRate: TrainBenchConstants.e2eLearningRate,
                    weightDecay: TrainBenchConstants.e2eWeightDecay,
                    biasCorrection: TrainBenchConstants.e2eAdamBiasCorrection)

                GPU.resetPeakMemory()
                try LoRATrain.train(
                    model: ctx.model, train: data, validate: data,
                    optimizer: optimizer, tokenizer: ctx.tokenizer, parameters: params
                ) { progress in
                    switch progress {
                    case .train(let iteration, _, let ips, let tps):
                        if iteration == 0 {
                            // Discard: MLX graph compile / first-allocation
                            // overhead, plus the forced iteration-0
                            // validation pass folded into this same window.
                            return .more
                        }
                        Self.appendTokenTimeRecord(
                            targetTokens: targetTokens, iteration: iteration,
                            secondsPerIter: 1.0 / ips, tokPerSec: tps,
                            peakMemBytes: Memory.snapshot().peakMemory,
                            thermalState: Self.thermalString(),
                            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                            modelName: modelName, sessionId: sessionId, fileName: fileName,
                            regime: regime)
                    case .validation, .save:
                        break
                    }
                    return .more
                }
            }
            tlog("tokentime cell tokens=\(targetTokens): complete")
            benchLogLine("tokentime cell tokens=\(targetTokens) complete")
        } catch {
            tlog("tokentime cell tokens=\(targetTokens): error: \(error)")
            benchLogLine("tokentime cell tokens=\(targetTokens) error: \(error)")
        }
    }

    /// Build ONE synthetic example whose token count is EXACTLY `target`, by
    /// tiling `tokentimeFillerPhrase` until it exceeds `target` tokens, then
    /// truncating via tokenizer encode/decode (same mechanism as
    /// `capExamples`). Content is irrelevant to attention/FFN compute cost —
    /// only sequence shape matters.
    private nonisolated static func syntheticExample(
        targetTokens target: Int, tokenizer: Tokenizer
    ) -> String {
        var text = ""
        var toks = tokenizer.encode(text: text)
        while toks.count <= target {
            text += TrainBenchConstants.tokentimeFillerPhrase
            toks = tokenizer.encode(text: text)
        }
        return tokenizer.decode(tokenIds: Array(toks.prefix(target)))
    }

    /// Serialize + append one record to the token-time JSONL (`fileName`
    /// selects hot vs cold — see `runTokenTimeBenchmark`/
    /// `runTokenTimeColdBenchmark`).
    private nonisolated static func appendTokenTimeRecord(
        targetTokens: Int, iteration: Int, secondsPerIter: Double, tokPerSec: Double,
        peakMemBytes: Int, thermalState: String, lowPowerMode: Bool, modelName: String,
        sessionId: String, fileName: String, regime: String
    ) {
        let record: [String: Any] = [
            "record_type": "train",
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "target_tokens": targetTokens,
            "iteration": iteration,
            "seconds_per_iter": secondsPerIter,
            "tok_per_sec": tokPerSec,
            "peak_mem_bytes": peakMemBytes,
            "thermal_state": thermalState,
            "low_power_mode": lowPowerMode,
            "model": modelName,
            "regime": regime,
            "batch_size": TrainBenchConstants.trainBatchSize,
            "iterations_per_cell": TrainBenchConstants.tokentimeIterationsPerCell,
            "lora_rank": TrainBenchConstants.loraRank,
            "lora_keys": TrainBenchConstants.loraKeysLabel,
            "num_lora_layers": TrainBenchConstants.loraLayers,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "checkpoint_granularity": TrainBenchConstants.checkpointGranularity,
            "optimizer": "adamw",
            "learning_rate": TrainBenchConstants.e2eLearningRate,
            "weight_decay": TrainBenchConstants.e2eWeightDecay,
            "adam_bias_correction": TrainBenchConstants.e2eAdamBiasCorrection,
            "warmup_seconds": TrainBenchConstants.tokentimeWarmupSeconds,
            "warmup_seq_cap": TrainBenchConstants.tokentimeWarmupSeqCap,
            "cooldown_cap_seconds": TrainBenchConstants.cooldownCapSeconds,
            "app_build": naxArmAppBuild(TrainBenchConstants.tokentimeAppBuild),
            "nax_arm": trainBenchmarkNaxArm ?? NSNull(),
            "bench_schema_version": TrainBenchConstants.tokentimeSchemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": sessionId,
            "device_model": Self.trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
        emitTokenTime(record, fileName: fileName)
    }

    /// Append one token-time JSONL line to `fileName` (see `tokentimeFileLock`).
    private nonisolated static func emitTokenTime(_ record: [String: Any], fileName: String) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(naxArmFileName(fileName))
        tokentimeFileLock.lock()
        defer { tokentimeFileLock.unlock() }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                if let d = line.data(using: .utf8) { try handle.write(contentsOf: d) }
            } else {
                try line.write(to: url, atomically: true, encoding: .utf8)
            }
        } catch {
            // best-effort; a failed line must not crash the run
        }
    }

    // MARK: - GC granularity sweep (h8) — one fixed-K cell per process launch

    /// Run ONE granularity cell (`checkpointGroupSize = k`) at the fixed
    /// `granularitySeqCap`/`granularityIterations` recipe, then exit. Mac-side
    /// orchestration launches this once per K in `granularityKValues`,
    /// blocking on each launch's exit before the next — see
    /// `scripts/run_granularity_sweep.sh`. A jetsam/OOM at this K is valid
    /// data (the `k_start` sentinel written before training, mirroring h1-h4's
    /// `cap_start`, pinpoints which K was in flight if the process dies
    /// without ever reaching a `train` record); the orchestrator moves on to
    /// the next K regardless of how this process exits.
    func runGranularityBenchmark(k: Int?) async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        guard let k, k > 0, 36 % k == 0 else {
            tlog("granularity: missing/invalid --granularity-k (must be a divisor of 36)")
            benchLogLine("granularity FAILED: missing/invalid --granularity-k")
            finishTrainBenchmark()
            return
        }
        tlog(
            "granularity start session=\(sessionId) k=\(k) "
                + "build=\(TrainBenchConstants.granularityAppBuild)")
        benchLogLine("granularity start session=\(sessionId) k=\(k)")

        guard
            let trainData = Self.loadBundledLoRAData(TrainBenchConstants.trainResource),
            let validData = Self.loadBundledLoRAData(TrainBenchConstants.validResource)
        else {
            tlog("granularity k=\(k): FAILED to load bundled LoRA data")
            benchLogLine("granularity k=\(k) FAILED to load bundled LoRA data")
            finishTrainBenchmark()
            return
        }

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("granularity k=\(k): model load failed: \(error)")
            benchLogLine("granularity k=\(k) model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        tlog("granularity k=\(k): model loaded")

        tlog(
            "granularity k=\(k): cooldown to nominal (uncapped) + "
                + "\(Int(TrainBenchConstants.granularityCooldownBufferSeconds))s buffer")
        await granularityCooldown()

        let modelName =
            modelConfiguration.name.components(separatedBy: "/").last
            ?? modelConfiguration.name

        writeGranularityKStartRecord(k: k, modelName: modelName, sessionId: sessionId)

        do {
            try await container.perform { ctx throws -> Void in
                let config = LoRAConfiguration(
                    numLayers: TrainBenchConstants.granularityLoraLayers,
                    loraParameters: .init(
                        rank: TrainBenchConstants.loraRank,
                        scale: TrainBenchConstants.loraScale,
                        keys: TrainBenchConstants.loraKeys))
                _ = try LoRAContainer.from(model: ctx.model, configuration: config)

                (ctx.model as? SmolLM3Model)?.checkpointGroupSize = k

                let capTrain = Self.capExamples(
                    trainData, cap: TrainBenchConstants.granularitySeqCap, tokenizer: ctx.tokenizer)
                let capValid = Self.capExamples(
                    validData, cap: TrainBenchConstants.granularitySeqCap, tokenizer: ctx.tokenizer)
                self.tlog(
                    "granularity k=\(k): LoRA applied (numLayers="
                        + "\(TrainBenchConstants.granularityLoraLayers)), starting train")

                let iterations = TrainBenchConstants.granularityIterations
                let params = LoRATrain.Parameters(
                    batchSize: TrainBenchConstants.trainBatchSize,
                    iterations: iterations,
                    stepsPerReport: TrainBenchConstants.granularityStepsPerReport,
                    stepsPerEval: iterations + 1,
                    validationBatches: 0,
                    saveEvery: iterations + 1,
                    adapterURL: nil)

                let optimizer = AdamW(learningRate: TrainBenchConstants.granularityLearningRate)

                let loopStart = Date.timeIntervalSinceReferenceDate
                GPU.resetPeakMemory()
                try LoRATrain.train(
                    model: ctx.model, train: capTrain, validate: capValid,
                    optimizer: optimizer, tokenizer: ctx.tokenizer, parameters: params
                ) { progress in
                    switch progress {
                    case .train(let iter, _, let ips, let tps):
                        Self.appendGranularityTrainRecord(
                            k: k, step: iter + 1, iterPerSec: ips, tokPerSec: tps,
                            elapsed: Date.timeIntervalSinceReferenceDate - loopStart,
                            peakMemBytes: Memory.snapshot().peakMemory,
                            thermalState: Self.thermalString(),
                            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                            modelName: modelName, sessionId: sessionId)
                        GPU.resetPeakMemory()
                    case .validation, .save:
                        break
                    }
                    return .more
                }
            }
            tlog("granularity k=\(k): complete")
            benchLogLine("granularity k=\(k) complete")
        } catch {
            tlog("granularity k=\(k): training error (possible OOM): \(error)")
            benchLogLine("granularity k=\(k) training error (possible OOM): \(error)")
            writeGranularityErrorRecord(
                k: k, error: "\(error)", modelName: modelName, sessionId: sessionId)
        }

        tlog("granularity complete session=\(sessionId) k=\(k)")
        benchLogLine("granularity complete session=\(sessionId) k=\(k)")
        finishTrainBenchmark()
    }

    /// Uncapped cooldown (h8): poll `thermalState` once every
    /// `granularityCooldownPollSeconds` until `nominal` — NO timed backstop
    /// (locked design, unlike `trainCooldown`'s capped wait) — then wait a
    /// fixed additional `granularityCooldownBufferSeconds` buffer.
    private func granularityCooldown() async {
        let start = Date.timeIntervalSinceReferenceDate
        while ProcessInfo.processInfo.thermalState != .nominal {
            try? await Task.sleep(for: .seconds(TrainBenchConstants.granularityCooldownPollSeconds))
        }
        let elapsedToNominal = Date.timeIntervalSinceReferenceDate - start
        tlog(
            "granularity cooldown: reached nominal after \(Int(elapsedToNominal))s, "
                + "waiting \(Int(TrainBenchConstants.granularityCooldownBufferSeconds))s buffer")
        try? await Task.sleep(for: .seconds(TrainBenchConstants.granularityCooldownBufferSeconds))
    }

    // MARK: - Granularity record builders

    private nonisolated static func granularityBaseRecord(
        k: Int, recordType: String, modelName: String, sessionId: String
    ) -> [String: Any] {
        [
            "record_type": recordType,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "checkpoint_granularity": k,
            "num_checkpoint_groups": 36 / k,
            "seq_cap": TrainBenchConstants.granularitySeqCap,
            "batch_size": TrainBenchConstants.trainBatchSize,
            "iterations_total": TrainBenchConstants.granularityIterations,
            "steps_per_report": TrainBenchConstants.granularityStepsPerReport,
            "model": modelName,
            "lora_rank": TrainBenchConstants.loraRank,
            "lora_keys": TrainBenchConstants.loraKeysLabel,
            "num_lora_layers": TrainBenchConstants.granularityLoraLayers,
            "gradient_checkpointing": true,
            "optimizer": "adamw",
            "learning_rate": TrainBenchConstants.granularityLearningRate,
            "app_build": naxArmAppBuild(TrainBenchConstants.granularityAppBuild),
            "nax_arm": trainBenchmarkNaxArm ?? NSNull(),
            "bench_schema_version": TrainBenchConstants.granularitySchemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": sessionId,
            "device_model": trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    private nonisolated static func appendGranularityTrainRecord(
        k: Int, step: Int, iterPerSec: Double, tokPerSec: Double, elapsed: Double,
        peakMemBytes: Int, thermalState: String, lowPowerMode: Bool, modelName: String,
        sessionId: String
    ) {
        var r = granularityBaseRecord(
            k: k, recordType: "train", modelName: modelName, sessionId: sessionId)
        r["step"] = step
        r["iter_per_sec"] = iterPerSec
        r["tok_per_sec"] = tokPerSec
        r["elapsed_s"] = elapsed
        r["peak_mem_bytes"] = peakMemBytes
        r["thermal_state"] = thermalState
        r["low_power_mode"] = lowPowerMode
        emitGranularity(r)
    }

    /// Sentinel written BEFORE a cell trains (mirrors h1-h4's `cap_start`). If
    /// the process then dies (uncatchable SIGKILL/jetsam, no `train` records),
    /// this marks which K was in flight.
    private func writeGranularityKStartRecord(k: Int, modelName: String, sessionId: String) {
        var r = Self.granularityBaseRecord(
            k: k, recordType: "k_start", modelName: modelName, sessionId: sessionId)
        r["thermal_state"] = Self.thermalString()
        r["low_power_mode"] = ProcessInfo.processInfo.isLowPowerModeEnabled
        Self.emitGranularity(r)
    }

    private func writeGranularityErrorRecord(
        k: Int, error: String, modelName: String, sessionId: String
    ) {
        var r = Self.granularityBaseRecord(
            k: k, recordType: "error", modelName: modelName, sessionId: sessionId)
        r["error"] = error
        r["thermal_state"] = Self.thermalString()
        r["low_power_mode"] = ProcessInfo.processInfo.isLowPowerModeEnabled
        Self.emitGranularity(r)
    }

    /// Serialize + append one record to `train_bench_metrics_granularity.jsonl`.
    /// No concurrent writer within a single process (one cell per launch,
    /// sequential train callback) but lock-guarded for consistency with the
    /// other harnesses' write paths.
    private nonisolated static func emitGranularity(_ record: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(
            naxArmFileName(TrainBenchConstants.granularityMetricsFileName))
        granularityFileLock.lock()
        defer { granularityFileLock.unlock() }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                if let d = line.data(using: .utf8) { try handle.write(contentsOf: d) }
            } else {
                try line.write(to: url, atomically: true, encoding: .utf8)
            }
        } catch {
            // best-effort; a failed line must not crash the run
        }
    }

    // MARK: - Thermal cooldown trajectory (h10)

    /// Immutable per-run context for the h10 records.
    struct ThermalRunContext: Sendable {
        let sessionId: String
        let soakSeconds: Double
        let probeIntervalSeconds: Double
        let modelName: String
    }

    /// Measure how training throughput RECOVERS after a training burst, and
    /// whether any burst-and-cool schedule beats running continuously.
    ///
    /// Sequence, one process launch = one run of the 3-run matrix:
    ///   1. cold-reference probe (`cold_ref`) — the session's own 100%-recovered
    ///      baseline, taken before any soak. Load-bearing: it is also the only
    ///      cross-run comparability check, since ambient temperature is
    ///      deliberately not recorded.
    ///   2. heat soak (`soak`) — continuous training for `soakMinutes` at
    ///      `thermalSoakTokens`, per-iteration records. Doubles as the HOT
    ///      measurement: its steady state vs. the cold reference IS the
    ///      in-session cold/hot ratio the duty-cycle arithmetic needs.
    ///   3. observation window (`probe`) — a fixed 90 minutes, probed every
    ///      `probeIntervalSeconds`, NO early stop.
    /// A 10s passive sampler (`sample`) runs across all three.
    ///
    /// See experiments/2026-07-28-ondevice-thermal-cooldown-h10-plan.md.
    func runThermalCooldownBenchmark(soakMinutes: Double, probeIntervalSeconds: Double) async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        let soakSeconds = soakMinutes * 60.0
        tlog(
            "thermal start session=\(sessionId) soak=\(soakMinutes)min "
                + "probe_interval=\(probeIntervalSeconds)s "
                + "build=\(TrainBenchConstants.thermalAppBuild)")
        benchLogLine(
            "thermal start session=\(sessionId) soak=\(soakMinutes)min "
                + "probe_interval=\(probeIntervalSeconds)s")

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("thermal: model load failed: \(error)")
            benchLogLine("thermal model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        tlog("thermal: model loaded")

        let ctx = ThermalRunContext(
            sessionId: sessionId, soakSeconds: soakSeconds,
            probeIntervalSeconds: probeIntervalSeconds,
            modelName: modelConfiguration.name.components(separatedBy: "/").last
                ?? modelConfiguration.name)

        let runStart = Date.timeIntervalSinceReferenceDate
        let batteryStart = Self.batterySnapshot()
        Self.appendThermalMarker(
            ctx, recordType: "run_start",
            extra: [
                "battery_level": batteryStart.level,
                "charging": batteryStart.charging,
                "thermal_state": Self.thermalString(),
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            ])

        // Passive 10s sampler, independent of the probe cadence. Runs for the
        // whole session (cold ref + soak + observation) and is cancelled on
        // every exit path.
        let sampler = Task { [ctx] in
            var previousCPUTicks = Self.cpuTicks()
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(TrainBenchConstants.thermalSampleSeconds))
                if Task.isCancelled { break }
                let snap = Self.batterySnapshot()
                let (cpuPct, newTicks) = Self.cpuUtilizationPercent(previous: previousCPUTicks)
                previousCPUTicks = newTicks
                Self.appendThermalSample(
                    ctx, elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                    level: snap.level, charging: snap.charging,
                    thermal: Self.thermalString(),
                    lpm: ProcessInfo.processInfo.isLowPowerModeEnabled,
                    cpuUtilPct: cpuPct)
            }
        }
        defer { sampler.cancel() }

        // 1. Cold reference.
        tlog("thermal: cold-reference probe")
        await runThermalProbe(
            container: container, ctx: ctx, recordType: "cold_ref", probeIndex: -1,
            cooldownElapsed: -1.0, runStart: runStart)

        // 2. Heat soak.
        tlog("thermal: soak starting (\(Int(soakMinutes))min @\(TrainBenchConstants.thermalSoakTokens) tok)")
        benchLogLine("thermal soak starting (\(Int(soakMinutes))min)")
        await runThermalSoak(container: container, ctx: ctx, runStart: runStart)
        let soakEnd = Date.timeIntervalSinceReferenceDate
        Self.appendThermalMarker(
            ctx, recordType: "soak_end",
            extra: [
                "elapsed_s": soakEnd - runStart,
                "thermal_state": Self.thermalString(),
                "battery_level": Self.batterySnapshot().level,
            ])
        tlog("thermal: soak complete after \(Int(soakEnd - runStart))s")

        // 3. Observation window — fixed duration, no early stop. The probe
        // itself takes real time, so sleep the REMAINDER of each interval
        // rather than the full interval; that keeps the cadence (and hence
        // the probe duty cycle, the thing Run B controls for) honest.
        var probeIndex = 0
        while true {
            let cooldownElapsed = Date.timeIntervalSinceReferenceDate - soakEnd
            if cooldownElapsed >= TrainBenchConstants.thermalObservationSeconds { break }
            let probeStart = Date.timeIntervalSinceReferenceDate
            await runThermalProbe(
                container: container, ctx: ctx, recordType: "probe", probeIndex: probeIndex,
                cooldownElapsed: cooldownElapsed, runStart: runStart)
            probeIndex += 1
            let probeDuration = Date.timeIntervalSinceReferenceDate - probeStart
            let remaining = probeIntervalSeconds - probeDuration
            if remaining > 0 {
                try? await Task.sleep(for: .seconds(remaining))
            }
        }

        sampler.cancel()
        let elapsed = Date.timeIntervalSinceReferenceDate - runStart
        let batteryEnd = Self.batterySnapshot()
        Self.appendThermalMarker(
            ctx, recordType: "run_end",
            extra: [
                "elapsed_s": elapsed,
                "probe_count": probeIndex,
                "battery_level": batteryStart.level,
                "battery_level_end": batteryEnd.level,
                "charging": batteryEnd.charging,
                "thermal_state": Self.thermalString(),
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            ])
        tlog("thermal complete session=\(sessionId) elapsed=\(Int(elapsed))s probes=\(probeIndex)")
        benchLogLine("thermal complete session=\(sessionId) probes=\(probeIndex)")
        finishTrainBenchmark()
    }

    /// Continuous training at `thermalSoakTokens` until `soakSeconds` of wall
    /// clock elapses (the iteration count is only a runaway backstop). Every
    /// iteration is logged, so the soak doubles as the HOT-regime measurement
    /// and its own heat-up curve.
    private func runThermalSoak(
        container: ModelContainer, ctx: ThermalRunContext, runStart: Double,
        durationSeconds: Double? = nil, recordType: String = "soak", cycleIndex: Int = -1
    ) async {
        let burstSeconds = durationSeconds ?? ctx.soakSeconds
        do {
            try await container.perform { c throws -> Void in
                try Self.applyThermalLoRA(to: c.model)
                let data = [
                    Self.syntheticExample(
                        targetTokens: TrainBenchConstants.thermalSoakTokens,
                        tokenizer: c.tokenizer)
                ]
                let params = LoRATrain.Parameters(
                    batchSize: TrainBenchConstants.trainBatchSize,
                    iterations: TrainBenchConstants.thermalSoakMaxIterations,
                    stepsPerReport: TrainBenchConstants.thermalStepsPerReport,
                    stepsPerEval: TrainBenchConstants.thermalSoakMaxIterations + 1,
                    validationBatches: 0,
                    saveEvery: TrainBenchConstants.thermalSoakMaxIterations + 1,
                    adapterURL: nil)
                let optimizer = AdamW(
                    learningRate: TrainBenchConstants.e2eLearningRate,
                    weightDecay: TrainBenchConstants.e2eWeightDecay,
                    biasCorrection: TrainBenchConstants.e2eAdamBiasCorrection)

                let soakStart = Date.timeIntervalSinceReferenceDate
                GPU.resetPeakMemory()
                try LoRATrain.train(
                    model: c.model, train: data, validate: data,
                    optimizer: optimizer, tokenizer: c.tokenizer, parameters: params
                ) { progress in
                    switch progress {
                    case .train(let iteration, _, let ips, let tps):
                        Self.appendThermalTrainRecord(
                            ctx, recordType: recordType, probeIndex: cycleIndex,
                            iterIndex: iteration,
                            secondsPerIter: 1.0 / ips, tokPerSec: tps,
                            targetTokens: TrainBenchConstants.thermalSoakTokens,
                            elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                            cooldownElapsed: -1.0)
                        if Date.timeIntervalSinceReferenceDate - soakStart >= burstSeconds {
                            return .stop
                        }
                    case .validation, .save:
                        break
                    }
                    return .more
                }
            }
        } catch {
            tlog("thermal soak error: \(error)")
            benchLogLine("thermal soak error: \(error)")
        }
    }

    /// Self-limiting arm (h10d). ONE continuous `LoRATrain.train` call, paced
    /// by sleeping a fixed delay inside the progress callback. The delay steps
    /// through `thermalSelfLimitPhases` on wall clock, so the training loop
    /// never stops and this arm carries none of h10c's restart/boost confound.
    ///
    /// The sleep is taken inside the callback deliberately: `LoraTrain.swift`
    /// computes `iterationsPerSecond` before invoking the callback and resets
    /// its timing origin after the callback returns, so the imposed delay is
    /// excluded from the reported rate. Every record therefore carries the
    /// device's true compute time per iteration (`seconds_per_iter`) alongside
    /// the delay that was in force (`delay_s`), and the effective throughput
    /// is `1 / (seconds_per_iter + delay_s)`.
    func runThermalSelfLimitBenchmark() async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        // Single-phase (cold-start) form if both launch args are present,
        // otherwise the built-in ascending schedule.
        let phases: [(delay: Double, seconds: Double)] =
            Self.trainBenchmarkSelfLimitSinglePhase.map { [$0] }
            ?? TrainBenchConstants.thermalSelfLimitPhases
        let total = phases.reduce(0.0) { $0 + $1.seconds }
        let sessionId = UUID().uuidString
        tlog(
            "thermal-selflimit start session=\(sessionId) phases=\(phases.count) "
                + "total=\(Int(total))s build=\(TrainBenchConstants.thermalSelfLimitAppBuild)")
        benchLogLine(
            "thermal-selflimit start session=\(sessionId) total=\(Int(total))s")

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("thermal-selflimit: model load failed: \(error)")
            benchLogLine("thermal-selflimit model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        tlog("thermal-selflimit: model loaded")

        let ctx = ThermalRunContext(
            sessionId: sessionId, soakSeconds: total, probeIntervalSeconds: 0,
            modelName: modelConfiguration.name.components(separatedBy: "/").last
                ?? modelConfiguration.name)

        let runStart = Date.timeIntervalSinceReferenceDate
        let batteryStart = Self.batterySnapshot()
        Self.appendSelfLimitMarker(
            ctx, recordType: "selflimit_run_start",
            extra: [
                "battery_level": batteryStart.level,
                "charging": batteryStart.charging,
                "thermal_state": Self.thermalString(),
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                "phase_delays_s": phases.map { $0.delay },
                "phase_durations_s": phases.map { $0.seconds },
                "total_planned_s": total,
            ])

        let sampler = Task { [ctx] in
            var previousCPUTicks = Self.cpuTicks()
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(TrainBenchConstants.thermalSampleSeconds))
                if Task.isCancelled { break }
                let snap = Self.batterySnapshot()
                let (cpuPct, newTicks) = Self.cpuUtilizationPercent(previous: previousCPUTicks)
                previousCPUTicks = newTicks
                Self.appendSelfLimitSample(
                    ctx, elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                    level: snap.level, charging: snap.charging,
                    thermal: Self.thermalString(),
                    lpm: ProcessInfo.processInfo.isLowPowerModeEnabled,
                    cpuUtilPct: cpuPct)
            }
        }
        defer { sampler.cancel() }

        // Cold reference, unpaced — the in-session baseline for "fast".
        tlog("thermal-selflimit: cold-reference probe")
        await runSelfLimitColdRef(container: container, ctx: ctx, runStart: runStart)

        do {
            try await container.perform { c throws -> Void in
                try Self.applyThermalLoRA(to: c.model)
                let data = [
                    Self.syntheticExample(
                        targetTokens: TrainBenchConstants.thermalSoakTokens,
                        tokenizer: c.tokenizer)
                ]
                let maxIters = TrainBenchConstants.thermalSelfLimitMaxIterations
                let params = LoRATrain.Parameters(
                    batchSize: TrainBenchConstants.trainBatchSize,
                    iterations: maxIters,
                    stepsPerReport: TrainBenchConstants.thermalSelfLimitStepsPerReport,
                    stepsPerEval: maxIters + 1,
                    validationBatches: 0,
                    saveEvery: maxIters + 1,
                    adapterURL: nil)
                let optimizer = AdamW(
                    learningRate: TrainBenchConstants.e2eLearningRate,
                    weightDecay: TrainBenchConstants.e2eWeightDecay,
                    biasCorrection: TrainBenchConstants.e2eAdamBiasCorrection)

                let trainStart = Date.timeIntervalSinceReferenceDate
                var phaseIndex = 0
                var phaseEnds: [Double] = []
                var acc = 0.0
                for p in phases {
                    acc += p.seconds
                    phaseEnds.append(acc)
                }

                GPU.resetPeakMemory()
                try LoRATrain.train(
                    model: c.model, train: data, validate: data,
                    optimizer: optimizer, tokenizer: c.tokenizer, parameters: params
                ) { progress in
                    switch progress {
                    case .train(let iteration, _, let ips, let tps):
                        let sinceTrain = Date.timeIntervalSinceReferenceDate - trainStart
                        // Advance the phase on wall clock, logging each boundary.
                        while phaseIndex < phaseEnds.count - 1,
                            sinceTrain >= phaseEnds[phaseIndex]
                        {
                            phaseIndex += 1
                            Self.appendSelfLimitMarker(
                                ctx, recordType: "phase_start",
                                extra: [
                                    "phase_index": phaseIndex,
                                    "delay_s": phases[phaseIndex].delay,
                                    "elapsed_s": Date.timeIntervalSinceReferenceDate - runStart,
                                    "thermal_state": Self.thermalString(),
                                ])
                        }
                        let delay = phases[phaseIndex].delay
                        Self.appendSelfLimitTrainRecord(
                            ctx, phaseIndex: phaseIndex, delaySeconds: delay,
                            iterIndex: iteration, secondsPerIter: 1.0 / ips, tokPerSec: tps,
                            elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                            phaseElapsed: sinceTrain
                                - (phaseIndex == 0 ? 0.0 : phaseEnds[phaseIndex - 1]))
                        if sinceTrain >= acc { return .stop }
                        // Pace. Synchronous on purpose: this blocks the model
                        // actor's thread (not the main actor, so the sampler
                        // keeps running) and falls outside LoraTrain's timing
                        // window, so it does not corrupt seconds_per_iter.
                        if delay > 0 { Thread.sleep(forTimeInterval: delay) }
                    case .validation, .save:
                        break
                    }
                    return .more
                }
            }
        } catch {
            tlog("thermal-selflimit error: \(error)")
            benchLogLine("thermal-selflimit error: \(error)")
        }

        sampler.cancel()
        let elapsed = Date.timeIntervalSinceReferenceDate - runStart
        let batteryEnd = Self.batterySnapshot()
        Self.appendSelfLimitMarker(
            ctx, recordType: "selflimit_run_end",
            extra: [
                "elapsed_s": elapsed,
                "battery_level": batteryStart.level,
                "battery_level_end": batteryEnd.level,
                "charging": batteryEnd.charging,
                "thermal_state": Self.thermalString(),
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            ])
        tlog("thermal-selflimit complete session=\(sessionId) elapsed=\(Int(elapsed))s")
        benchLogLine("thermal-selflimit complete session=\(sessionId)")
        finishTrainBenchmark()
    }

    /// Unpaced 2-iteration cold reference, written to the self-limit JSONL.
    private func runSelfLimitColdRef(
        container: ModelContainer, ctx: ThermalRunContext, runStart: Double
    ) async {
        do {
            try await container.perform { c throws -> Void in
                try Self.applyThermalLoRA(to: c.model)
                let data = [
                    Self.syntheticExample(
                        targetTokens: TrainBenchConstants.thermalProbeTokens,
                        tokenizer: c.tokenizer)
                ]
                let iters = TrainBenchConstants.thermalProbeIterations
                let params = LoRATrain.Parameters(
                    batchSize: TrainBenchConstants.trainBatchSize, iterations: iters,
                    stepsPerReport: 1, stepsPerEval: iters + 1, validationBatches: 0,
                    saveEvery: iters + 1, adapterURL: nil)
                let optimizer = AdamW(
                    learningRate: TrainBenchConstants.e2eLearningRate,
                    weightDecay: TrainBenchConstants.e2eWeightDecay,
                    biasCorrection: TrainBenchConstants.e2eAdamBiasCorrection)
                GPU.resetPeakMemory()
                try LoRATrain.train(
                    model: c.model, train: data, validate: data,
                    optimizer: optimizer, tokenizer: c.tokenizer, parameters: params
                ) { progress in
                    if case .train(let iteration, _, let ips, let tps) = progress {
                        Self.appendSelfLimitTrainRecord(
                            ctx, phaseIndex: -1, delaySeconds: 0.0, iterIndex: iteration,
                            secondsPerIter: 1.0 / ips, tokPerSec: tps,
                            elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                            phaseElapsed: -1.0, recordType: "cold_ref")
                    }
                    return .more
                }
            }
        } catch {
            tlog("thermal-selflimit cold-ref error: \(error)")
        }
    }

    // MARK: - Self-limit (h10d) record builders

    private nonisolated static func selfLimitBaseRecord(
        _ c: ThermalRunContext, recordType: String
    ) -> [String: Any] {
        [
            "record_type": recordType,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "total_planned_s": c.soakSeconds,
            "probe_tokens": TrainBenchConstants.thermalProbeTokens,
            "train_tokens": TrainBenchConstants.thermalSoakTokens,
            "model": c.modelName,
            "batch_size": TrainBenchConstants.trainBatchSize,
            "lora_rank": TrainBenchConstants.loraRank,
            "lora_keys": TrainBenchConstants.loraKeysLabel,
            "num_lora_layers": TrainBenchConstants.thermalLoraLayers,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "checkpoint_granularity": TrainBenchConstants.checkpointGranularity,
            "optimizer": "adamw",
            "learning_rate": TrainBenchConstants.e2eLearningRate,
            "weight_decay": TrainBenchConstants.e2eWeightDecay,
            "adam_bias_correction": TrainBenchConstants.e2eAdamBiasCorrection,
            "app_build": naxArmAppBuild(TrainBenchConstants.thermalSelfLimitAppBuild),
            "nax_arm": trainBenchmarkNaxArm ?? NSNull(),
            "bench_schema_version": TrainBenchConstants.thermalSchemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": c.sessionId,
            "device_model": trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    private nonisolated static func appendSelfLimitTrainRecord(
        _ c: ThermalRunContext, phaseIndex: Int, delaySeconds: Double, iterIndex: Int,
        secondsPerIter: Double, tokPerSec: Double, elapsed: Double, phaseElapsed: Double,
        recordType: String = "selflimit_train"
    ) {
        var r = selfLimitBaseRecord(c, recordType: recordType)
        r["phase_index"] = phaseIndex
        r["delay_s"] = delaySeconds
        r["iter_index"] = iterIndex
        r["seconds_per_iter"] = secondsPerIter
        r["tok_per_sec"] = tokPerSec
        r["elapsed_s"] = elapsed
        r["phase_elapsed_s"] = phaseElapsed
        // Device compute + imposed pacing. This is the quantity the whole arm
        // exists to compare against continuous training's ~9.98 s/iter.
        r["effective_seconds_per_iter"] = secondsPerIter + delaySeconds
        r["peak_mem_bytes"] = Memory.snapshot().peakMemory
        r["thermal_state"] = thermalString()
        r["low_power_mode"] = ProcessInfo.processInfo.isLowPowerModeEnabled
        emitSelfLimit(r)
    }

    private nonisolated static func appendSelfLimitSample(
        _ c: ThermalRunContext, elapsed: Double, level: Double, charging: Bool,
        thermal: String, lpm: Bool, cpuUtilPct: Double?
    ) {
        var r = selfLimitBaseRecord(c, recordType: "sample")
        r["elapsed_s"] = elapsed
        r["battery_level"] = level
        r["charging"] = charging
        r["thermal_state"] = thermal
        r["low_power_mode"] = lpm
        r["cpu_util_pct"] = cpuUtilPct ?? NSNull()
        emitSelfLimit(r)
    }

    private nonisolated static func appendSelfLimitMarker(
        _ c: ThermalRunContext, recordType: String, extra: [String: Any]
    ) {
        var r = selfLimitBaseRecord(c, recordType: recordType)
        for (k, v) in extra { r[k] = v }
        emitSelfLimit(r)
    }

    private nonisolated static func emitSelfLimit(_ record: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(
            naxArmFileName(TrainBenchConstants.thermalSelfLimitMetricsFileName))
        thermalFileLock.lock()
        defer { thermalFileLock.unlock() }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                if let d = line.data(using: .utf8) { try handle.write(contentsOf: d) }
            } else {
                try line.write(to: url, atomically: true, encoding: .utf8)
            }
        } catch {
            // best-effort; a failed line must not crash the run
        }
    }

    /// Sustained-cycling arm (h10c). Repeat `cycles` × (train for
    /// `burstMinutes`, idle for `restSeconds`) and log every training
    /// iteration tagged with its cycle index, so the aggregator can ask the
    /// question Runs A/B/C could not: does iterations-per-burst hold up, or
    /// decay as chassis heat accumulates across cycles?
    ///
    /// Runs A/B/C measured ONE burst and the recovery after it; the ~1.25×
    /// figure for a 10-on/2-off schedule was extrapolated from that. Run B
    /// showed burst output depends on retained chassis heat, which a
    /// 2-minute gap does not clear even though throughput recovers — so that
    /// extrapolation is an upper bound and this arm tests it directly.
    ///
    /// A cold-reference probe is taken first (same as the other arms) so the
    /// run is comparable to A/B/C, and a short probe is taken during each
    /// rest gap to record what throughput has recovered to before the next
    /// burst starts.
    func runThermalCycleBenchmark(
        burstMinutes: Double, restSeconds: Double, cycles: Int
    ) async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        let burstSeconds = burstMinutes * 60.0
        tlog(
            "thermal-cycle start session=\(sessionId) burst=\(burstMinutes)min "
                + "rest=\(restSeconds)s cycles=\(cycles) "
                + "build=\(TrainBenchConstants.thermalAppBuild)")
        benchLogLine(
            "thermal-cycle start session=\(sessionId) burst=\(burstMinutes)min "
                + "rest=\(restSeconds)s cycles=\(cycles)")

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("thermal-cycle: model load failed: \(error)")
            benchLogLine("thermal-cycle model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        tlog("thermal-cycle: model loaded")

        // soakSeconds carries the BURST length so every record self-describes
        // the schedule; cycle-specific fields are added by the markers below.
        let ctx = ThermalRunContext(
            sessionId: sessionId, soakSeconds: burstSeconds,
            probeIntervalSeconds: restSeconds,
            modelName: modelConfiguration.name.components(separatedBy: "/").last
                ?? modelConfiguration.name)

        let runStart = Date.timeIntervalSinceReferenceDate
        let batteryStart = Self.batterySnapshot()
        Self.appendThermalMarker(
            ctx, recordType: "cycle_run_start",
            extra: [
                "battery_level": batteryStart.level,
                "charging": batteryStart.charging,
                "thermal_state": Self.thermalString(),
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                "burst_seconds": burstSeconds,
                "rest_seconds": restSeconds,
                "cycles_planned": cycles,
            ])

        let sampler = Task { [ctx] in
            var previousCPUTicks = Self.cpuTicks()
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(TrainBenchConstants.thermalSampleSeconds))
                if Task.isCancelled { break }
                let snap = Self.batterySnapshot()
                let (cpuPct, newTicks) = Self.cpuUtilizationPercent(previous: previousCPUTicks)
                previousCPUTicks = newTicks
                Self.appendThermalSample(
                    ctx, elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                    level: snap.level, charging: snap.charging,
                    thermal: Self.thermalString(),
                    lpm: ProcessInfo.processInfo.isLowPowerModeEnabled,
                    cpuUtilPct: cpuPct)
            }
        }
        defer { sampler.cancel() }

        tlog("thermal-cycle: cold-reference probe")
        await runThermalProbe(
            container: container, ctx: ctx, recordType: "cold_ref", probeIndex: -1,
            cooldownElapsed: -1.0, runStart: runStart)

        for cycle in 0..<cycles {
            let burstStart = Date.timeIntervalSinceReferenceDate
            tlog("thermal-cycle: burst \(cycle + 1)/\(cycles) starting")
            await runThermalSoak(
                container: container, ctx: ctx, runStart: runStart,
                durationSeconds: burstSeconds, recordType: "cycle_burst",
                cycleIndex: cycle)
            let burstEnd = Date.timeIntervalSinceReferenceDate
            Self.appendThermalMarker(
                ctx, recordType: "cycle_burst_end",
                extra: [
                    "cycle_index": cycle,
                    "elapsed_s": burstEnd - runStart,
                    "burst_duration_s": burstEnd - burstStart,
                    "thermal_state": Self.thermalString(),
                    "battery_level": Self.batterySnapshot().level,
                ])
            tlog(
                "thermal-cycle: burst \(cycle + 1) done in "
                    + "\(Int(burstEnd - burstStart))s, resting \(Int(restSeconds))s")

            guard cycle < cycles - 1 else { break }

            // Rest gap. Probe near its end so we record what throughput has
            // actually recovered to before the next burst begins — the
            // quantity that decides whether the schedule holds up. The probe
            // is ~13s of training, so it is taken with enough of the gap
            // remaining that it does not overrun into the next burst.
            let probeLead = min(20.0, restSeconds * 0.25)
            let sleepBeforeProbe = max(0.0, restSeconds - probeLead)
            if sleepBeforeProbe > 0 {
                try? await Task.sleep(for: .seconds(sleepBeforeProbe))
            }
            await runThermalProbe(
                container: container, ctx: ctx, recordType: "cycle_rest_probe",
                probeIndex: cycle,
                cooldownElapsed: Date.timeIntervalSinceReferenceDate - burstEnd,
                runStart: runStart)
        }

        sampler.cancel()
        let elapsed = Date.timeIntervalSinceReferenceDate - runStart
        let batteryEnd = Self.batterySnapshot()
        Self.appendThermalMarker(
            ctx, recordType: "cycle_run_end",
            extra: [
                "elapsed_s": elapsed,
                "cycles_completed": cycles,
                "battery_level": batteryStart.level,
                "battery_level_end": batteryEnd.level,
                "charging": batteryEnd.charging,
                "thermal_state": Self.thermalString(),
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            ])
        tlog("thermal-cycle complete session=\(sessionId) elapsed=\(Int(elapsed))s")
        benchLogLine("thermal-cycle complete session=\(sessionId)")
        finishTrainBenchmark()
    }

    /// One probe: `thermalProbeIterations` iterations at `thermalProbeTokens`.
    /// EVERY iteration is logged (including index 0) — the discard rule lives
    /// in the aggregator, per this repo's log-raw-decide-later convention.
    private func runThermalProbe(
        container: ModelContainer, ctx: ThermalRunContext, recordType: String,
        probeIndex: Int, cooldownElapsed: Double, runStart: Double
    ) async {
        do {
            try await container.perform { c throws -> Void in
                try Self.applyThermalLoRA(to: c.model)
                let data = [
                    Self.syntheticExample(
                        targetTokens: TrainBenchConstants.thermalProbeTokens,
                        tokenizer: c.tokenizer)
                ]
                let iters = TrainBenchConstants.thermalProbeIterations
                let params = LoRATrain.Parameters(
                    batchSize: TrainBenchConstants.trainBatchSize,
                    iterations: iters,
                    stepsPerReport: TrainBenchConstants.thermalStepsPerReport,
                    stepsPerEval: iters + 1,
                    validationBatches: 0,
                    saveEvery: iters + 1,
                    adapterURL: nil)
                let optimizer = AdamW(
                    learningRate: TrainBenchConstants.e2eLearningRate,
                    weightDecay: TrainBenchConstants.e2eWeightDecay,
                    biasCorrection: TrainBenchConstants.e2eAdamBiasCorrection)

                GPU.resetPeakMemory()
                try LoRATrain.train(
                    model: c.model, train: data, validate: data,
                    optimizer: optimizer, tokenizer: c.tokenizer, parameters: params
                ) { progress in
                    switch progress {
                    case .train(let iteration, _, let ips, let tps):
                        Self.appendThermalTrainRecord(
                            ctx, recordType: recordType, probeIndex: probeIndex,
                            iterIndex: iteration, secondsPerIter: 1.0 / ips, tokPerSec: tps,
                            targetTokens: TrainBenchConstants.thermalProbeTokens,
                            elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                            cooldownElapsed: cooldownElapsed)
                    case .validation, .save:
                        break
                    }
                    return .more
                }
            }
        } catch {
            tlog("thermal probe \(probeIndex) error: \(error)")
            benchLogLine("thermal probe \(probeIndex) error: \(error)")
        }
    }

    /// Fresh LoRA + per-block GC on the model. h10 uses `thermalLoraLayers`
    /// (36 — the FIXED count, unlike h1-h7's buggy 28); see that constant's
    /// doc comment for why that is safe for this round's headline.
    private nonisolated static func applyThermalLoRA(to model: LanguageModel) throws {
        let config = LoRAConfiguration(
            numLayers: TrainBenchConstants.thermalLoraLayers,
            loraParameters: .init(
                rank: TrainBenchConstants.loraRank,
                scale: TrainBenchConstants.loraScale,
                keys: TrainBenchConstants.loraKeys))
        _ = try LoRAContainer.from(model: model, configuration: config)
        if TrainBenchConstants.gradientCheckpointing {
            (model as? SmolLM3Model)?.checkpointGroupSize = 1
        }
    }

    // MARK: - Thermal (h10) record builders

    private nonisolated static func thermalBaseRecord(
        _ c: ThermalRunContext, recordType: String
    ) -> [String: Any] {
        [
            "record_type": recordType,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "soak_seconds": c.soakSeconds,
            "probe_interval_s": c.probeIntervalSeconds,
            "probe_tokens": TrainBenchConstants.thermalProbeTokens,
            "probe_iterations": TrainBenchConstants.thermalProbeIterations,
            "soak_tokens": TrainBenchConstants.thermalSoakTokens,
            "observation_seconds": TrainBenchConstants.thermalObservationSeconds,
            "model": c.modelName,
            "batch_size": TrainBenchConstants.trainBatchSize,
            "lora_rank": TrainBenchConstants.loraRank,
            "lora_keys": TrainBenchConstants.loraKeysLabel,
            "num_lora_layers": TrainBenchConstants.thermalLoraLayers,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "checkpoint_granularity": TrainBenchConstants.checkpointGranularity,
            "optimizer": "adamw",
            "learning_rate": TrainBenchConstants.e2eLearningRate,
            "weight_decay": TrainBenchConstants.e2eWeightDecay,
            "adam_bias_correction": TrainBenchConstants.e2eAdamBiasCorrection,
            "app_build": naxArmAppBuild(TrainBenchConstants.thermalAppBuild),
            "nax_arm": trainBenchmarkNaxArm ?? NSNull(),
            "bench_schema_version": TrainBenchConstants.thermalSchemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": c.sessionId,
            "device_model": trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    /// One training iteration, from either the soak or a probe. `probe_index`
    /// is -1 for soak/cold-ref rows; `cooldown_elapsed_s` is -1 for anything
    /// before the observation window starts.
    private nonisolated static func appendThermalTrainRecord(
        _ c: ThermalRunContext, recordType: String, probeIndex: Int, iterIndex: Int,
        secondsPerIter: Double, tokPerSec: Double, targetTokens: Int, elapsed: Double,
        cooldownElapsed: Double
    ) {
        var r = thermalBaseRecord(c, recordType: recordType)
        r["probe_index"] = probeIndex
        r["iter_index"] = iterIndex
        r["seconds_per_iter"] = secondsPerIter
        r["tok_per_sec"] = tokPerSec
        r["target_tokens"] = targetTokens
        r["elapsed_s"] = elapsed
        r["cooldown_elapsed_s"] = cooldownElapsed
        r["peak_mem_bytes"] = Memory.snapshot().peakMemory
        r["thermal_state"] = thermalString()
        r["low_power_mode"] = ProcessInfo.processInfo.isLowPowerModeEnabled
        emitThermal(r)
    }

    private nonisolated static func appendThermalSample(
        _ c: ThermalRunContext, elapsed: Double, level: Double, charging: Bool,
        thermal: String, lpm: Bool, cpuUtilPct: Double?
    ) {
        var r = thermalBaseRecord(c, recordType: "sample")
        r["elapsed_s"] = elapsed
        r["battery_level"] = level
        r["charging"] = charging
        r["thermal_state"] = thermal
        r["low_power_mode"] = lpm
        r["cpu_util_pct"] = cpuUtilPct ?? NSNull()
        emitThermal(r)
    }

    private nonisolated static func appendThermalMarker(
        _ c: ThermalRunContext, recordType: String, extra: [String: Any]
    ) {
        var r = thermalBaseRecord(c, recordType: recordType)
        for (k, v) in extra { r[k] = v }
        emitThermal(r)
    }

    /// Append one h10 JSONL line (see `thermalFileLock` — genuinely concurrent
    /// writers here, unlike h7/h8).
    private nonisolated static func emitThermal(_ record: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(
            naxArmFileName(TrainBenchConstants.thermalMetricsFileName))
        thermalFileLock.lock()
        defer { thermalFileLock.unlock() }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                if let d = line.data(using: .utf8) { try handle.write(contentsOf: d) }
            } else {
                try line.write(to: url, atomically: true, encoding: .utf8)
            }
        } catch {
            // best-effort; a failed line must not crash the run
        }
    }

    // MARK: - Per-op / per-phase decomposition (h11)

    /// Immutable per-run context for the h11 records.
    struct PerOpRunContext: Sendable {
        let sessionId: String
        let modelName: String
        /// Honest user input via `--idle-minutes`, recorded verbatim; `nil`
        /// when the arg was omitted (recorded as null, never inferred).
        let idleMinutes: Double?
        /// NAX A/B round: alternate `MLX_ENABLE_NAX_N` per iteration and write
        /// to a separate JSONL under a separate build string, so h11's data and
        /// provenance are untouched. Defaults false — h11 behaviour verbatim.
        var naxAB: Bool = false
        /// `--pin-arms` variant of the A/B: the arm is constant within each
        /// sub-block (one block per arm) instead of alternating per iteration.
        /// Separate JSONL + `-pinned` build suffix. See trainBenchmarkPinArms.
        var pinnedArms: Bool = false
    }

    /// The six per-phase times of ONE barriered training iteration, seconds.
    struct PerOpPhaseTimes: Sendable {
        let dataPrep: Double
        let graphBuild: Double
        let forward: Double
        let backward: Double
        let optimizer: Double
        let readback: Double

        var total: Double {
            dataPrep + graphBuild + forward + backward + optimizer + readback
        }
    }

    /// Tier 1: where does the time inside one LoRA training iteration go, as a
    /// function of token count and heat?
    ///
    /// Session shape, one launch:
    ///   1. cold-reference probe (`cold_ref`, 2 iterations @500 tok — h10's
    ///      exact design) for cross-run comparability.
    ///   2. pass `cool` — the token grid ascending, no cooldown gates.
    ///   3. pass `hot` — the identical grid re-run immediately, on the now-hot
    ///      device. Matched cell order, so both passes share a drift profile.
    /// A 30s passive sampler runs throughout.
    ///
    /// Each cell measures the same recipe twice: FUSED (stock
    /// `LoRATrain.train`, h7's measurement mode — the validity control) and
    /// BARRIERED (this file's replica of the trainer's internals with an
    /// `eval` between each phase). Σ(phases)/fused is the decomposition
    /// overhead, reported per cell by the aggregator.
    ///
    /// See experiments/2026-08-04-ondevice-perop-h11-plan.md.
    /// `naxAB: true` runs the NAX A/B variant: aligned token grid, per-iteration
    /// `MLX_ENABLE_NAX_N` alternation, separate JSONL and build string. The
    /// cold-reference probe stays pinned OFF so it remains directly comparable
    /// to h11's 4.702 s/iter as a cross-round reproducibility anchor.
    func runPerOpBenchmark(idleMinutes: Double?, naxAB: Bool = false) async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        tlog(
            "perop start session=\(sessionId) build=\(TrainBenchConstants.peropAppBuild) "
                + "idle_minutes=\(idleMinutes.map { String($0) } ?? "unset")")
        benchLogLine("perop start session=\(sessionId)")

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("perop: model load failed: \(error)")
            benchLogLine("perop model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        tlog("perop: model loaded")

        let ctx = PerOpRunContext(
            sessionId: sessionId,
            modelName: modelConfiguration.name.components(separatedBy: "/").last
                ?? modelConfiguration.name,
            idleMinutes: idleMinutes,
            naxAB: naxAB,
            pinnedArms: naxAB && Self.trainBenchmarkPinArms)

        let runStart = Date.timeIntervalSinceReferenceDate
        let batteryStart = Self.batterySnapshot()
        Self.appendPerOpMarker(
            ctx, recordType: "run_start",
            extra: [
                "elapsed_s": 0.0,
                "battery_level": batteryStart.level,
                "charging": batteryStart.charging,
                "thermal_state": Self.thermalString(),
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            ])

        // Passive 30s sampler (h5/h9 cadence — h10's 10s existed to resolve
        // `thermalState` transitions, which is not a deliverable here).
        let sampler = Task { [ctx] in
            var previousCPUTicks = Self.cpuTicks()
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(TrainBenchConstants.peropSampleSeconds))
                if Task.isCancelled { break }
                let snap = Self.batterySnapshot()
                let (cpuPct, newTicks) = Self.cpuUtilizationPercent(previous: previousCPUTicks)
                previousCPUTicks = newTicks
                Self.appendPerOpSample(
                    ctx, elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                    level: snap.level, charging: snap.charging,
                    thermal: Self.thermalString(),
                    lpm: ProcessInfo.processInfo.isLowPowerModeEnabled,
                    cpuUtilPct: cpuPct)
            }
        }
        defer { sampler.cancel() }

        // 1. Cold reference (h10 design, 2 iterations @500 tok).
        tlog("perop: cold-reference probe")
        // Pinned OFF: this probe's whole job is comparability with h11's
        // 4.702 s/iter, so it must run stock dispatch in both rounds. The
        // same reasoning holds for the global `--nax-arm` rerun — the anchor
        // chain (h10 4.6xx / h11 4.702) only means something if every round's
        // cold ref runs the same kernel.
        _ = Self.setNaxArm(on: false, enabled: naxAB)
        if Self.trainBenchmarkNaxArm != nil { setenv("MLX_ENABLE_NAX_N", "0", 1) }
        await runPerOpColdRef(container: container, ctx: ctx, runStart: runStart)
        if let arm = Self.trainBenchmarkNaxArm {
            setenv("MLX_ENABLE_NAX_N", arm == "on" ? "1" : "0", 1)
        }

        // 2+3. Both passes, ascending tokens, no cooldown gate anywhere.
        let grid =
            naxAB
            ? TrainBenchConstants.naxABTokenCounts
            : TrainBenchConstants.peropTokenCounts
        var cellIndex = 0
        for pass in TrainBenchConstants.peropPasses {
            for tokens in grid {
                await runPerOpCell(
                    container: container, ctx: ctx, targetTokens: tokens, pass: pass,
                    runStart: runStart, cellIndex: cellIndex)
                cellIndex += 1
            }
        }

        sampler.cancel()
        let elapsed = Date.timeIntervalSinceReferenceDate - runStart
        let batteryEnd = Self.batterySnapshot()
        Self.appendPerOpMarker(
            ctx, recordType: "run_end",
            extra: [
                "elapsed_s": elapsed,
                "battery_level": batteryStart.level,
                "battery_level_end": batteryEnd.level,
                "charging": batteryEnd.charging,
                "thermal_state": Self.thermalString(),
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            ])
        tlog("perop complete session=\(sessionId) elapsed=\(Int(elapsed))s")
        benchLogLine("perop complete session=\(sessionId)")
        finishTrainBenchmark()
    }

    /// Cold-reference probe: `peropColdRefIterations` fused iterations at
    /// `peropColdRefTokens`, through the stock trainer. Both iterations are
    /// logged (log-raw-decide-later); index 0 carries the compile/first-alloc
    /// cost and is flagged `warmup: true` for the aggregator to drop.
    private func runPerOpColdRef(
        container: ModelContainer, ctx: PerOpRunContext, runStart: Double
    ) async {
        do {
            try await container.perform { c throws -> Void in
                try Self.applyPerOpLoRA(to: c.model)
                let data = [
                    Self.syntheticExample(
                        targetTokens: TrainBenchConstants.peropColdRefTokens,
                        tokenizer: c.tokenizer)
                ]
                let iters = TrainBenchConstants.peropColdRefIterations
                let params = LoRATrain.Parameters(
                    batchSize: TrainBenchConstants.trainBatchSize,
                    iterations: iters, stepsPerReport: 1, stepsPerEval: iters + 1,
                    validationBatches: 0, saveEvery: iters + 1, adapterURL: nil)
                let optimizer = Self.perOpOptimizer()

                GPU.resetPeakMemory()
                try LoRATrain.train(
                    model: c.model, train: data, validate: data,
                    optimizer: optimizer, tokenizer: c.tokenizer, parameters: params
                ) { progress in
                    if case .train(let iteration, let loss, let ips, let tps) = progress {
                        // The probe is pinned OFF in every arm-aware round (see
                        // runPerOpBenchmark), so its rows say so explicitly
                        // rather than inheriting the base record's global arm.
                        let coldRefArm: String? =
                            (ctx.naxAB || Self.trainBenchmarkNaxArm != nil) ? "off" : nil
                        Self.appendPerOpIterRecord(
                            ctx, recordType: "cold_ref", mode: "fused", pass: "cold_ref",
                            targetTokens: TrainBenchConstants.peropColdRefTokens,
                            iterIndex: iteration, warmup: iteration < 1,
                            iterSeconds: 1.0 / ips, tokPerSec: tps, loss: loss,
                            phases: nil,
                            elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                            arm: coldRefArm)
                    }
                    return .more
                }
            }
        } catch {
            tlog("perop cold_ref error: \(error)")
            benchLogLine("perop cold_ref error: \(error)")
        }
    }

    /// One cell = one (token count, pass). Fresh LoRA + fresh optimizer, then
    /// the fused sub-block, then the barriered sub-block, back-to-back (NOT
    /// interleaved step-by-step — alternating would thrash lazy-eval/cache
    /// state and blur each sub-block's warmup).
    ///
    /// The barriered sub-block CONTINUES from the weights the fused sub-block
    /// left behind; it does not restart from the adapter's initial state.
    ///
    /// This started as a snapshot/restore of `model.trainableParameters()`
    /// intended to give both modes an identical starting point (so their loss
    /// sequences could be compared step-for-step). That restore was a NO-OP and
    /// was removed after the 2026-08-04 run showed why: `Module.update`'s
    /// leaf-array case calls `p._updateInternal(newArray)`, which swaps the
    /// handle INSIDE the existing MLXArray object rather than replacing the
    /// dictionary's reference — so `trainableParameters()` hands back ALIASES
    /// of the model's live arrays, and the "snapshot" tracked the weights right
    /// through training. (The same aliasing the h6 v7 changelog in
    /// TrainBenchConstants.swift documents for the model-load path.) A genuine
    /// reset would need a deep copy of every adapter array.
    ///
    /// Plan risk #1 is checked from the continuity of the loss curve across the
    /// mode boundary instead: the barriered block picks the trajectory up where
    /// the fused block left it, so a faithful replica continues the same
    /// per-step decay, while a broken one (wrong loss, skipped optimizer step,
    /// defeated checkpointing) would show a step change at the seam. Measured
    /// 2026-08-04: residuals |Δ| ≤ 0.0008 across all 12 cells, i.e. ≲2.5% of the
    /// step size. `eval/perop_aggregate.py`'s `loss_continuity` computes it.
    ///
    /// Note this never affected the phase TIMINGS — MLX's dense and quantized
    /// kernels are value-independent, so per-iteration cost does not depend on
    /// which weights happen to be resident.
    private func runPerOpCell(
        container: ModelContainer, ctx: PerOpRunContext, targetTokens: Int, pass: String,
        runStart: Double, cellIndex: Int = 0
    ) async {
        let cellStart = Date.timeIntervalSinceReferenceDate
        let battery = Self.batterySnapshot()
        // Pinned-arm block order alternates by cell so slow thermal drift
        // cancels across the grid: even cells run OFF first, odd cells ON
        // first. `armOrder` is unused (and unrecorded) outside pinned mode.
        let armOrder: [Bool] = cellIndex % 2 == 0 ? [false, true] : [true, false]
        var cellStartExtra: [String: Any] = [
            "target_tokens": targetTokens,
            "pass": pass,
            "cell_index": cellIndex,
            "elapsed_s": cellStart - runStart,
            "thermal_state": Self.thermalString(),
            "battery_level": battery.level,
            "charging": battery.charging,
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
        ]
        if ctx.pinnedArms {
            cellStartExtra["arm_first"] = armOrder[0] ? "on" : "off"
        }
        Self.appendPerOpMarker(ctx, recordType: "cell_start", extra: cellStartExtra)
        tlog("perop cell tokens=\(targetTokens) pass=\(pass): starting")

        do {
            try await container.perform { c throws -> Void in
                try Self.applyPerOpLoRA(to: c.model)
                let model: Module = c.model

                let example = Self.syntheticExample(
                    targetTokens: targetTokens, tokenizer: c.tokenizer)
                let total =
                    TrainBenchConstants.peropWarmupIterations
                    + TrainBenchConstants.peropKeptIterations
                let warmupCount = TrainBenchConstants.peropWarmupIterations

                // --- fused sub-block (validity control) ----------------------
                // Stock `LoRATrain.train` with stepsPerReport = 1 — h7's exact
                // measurement mode, so these times are directly comparable to
                // h7's fits. Iteration 0 also absorbs the trainer's forced
                // iteration-0 validation pass (LoraTrain.swift:314), which is
                // why it is the discarded one.
                let fusedParams = LoRATrain.Parameters(
                    batchSize: TrainBenchConstants.trainBatchSize,
                    iterations: total, stepsPerReport: 1, stepsPerEval: total + 1,
                    validationBatches: 0, saveEvery: total + 1, adapterURL: nil)
                GPU.resetPeakMemory()
                if ctx.pinnedArms {
                    // Pinned arms (--pin-arms): one fused block PER ARM, arm
                    // constant throughout — the design the alternating A/B is
                    // being checked against. Each block's iteration 0 absorbs
                    // the one arm-switch seam (recompile/first-alloc) and is
                    // dropped as warmup, so no switch cost contaminates the
                    // kept iterations. Block order = `armOrder` (alternates by
                    // cell so drift cancels across the grid).
                    for armOn in armOrder {
                        let armLabel = Self.setNaxArm(on: armOn, enabled: true)
                        try LoRATrain.train(
                            model: model, train: [example], validate: [example],
                            optimizer: Self.perOpOptimizer(), tokenizer: c.tokenizer,
                            parameters: fusedParams
                        ) { progress in
                            if case .train(let iteration, let loss, let ips, let tps) =
                                progress
                            {
                                Self.appendPerOpIterRecord(
                                    ctx, recordType: "iter", mode: "fused", pass: pass,
                                    targetTokens: targetTokens, iterIndex: iteration,
                                    warmup: iteration < warmupCount,
                                    iterSeconds: 1.0 / ips, tokPerSec: tps, loss: loss,
                                    phases: nil,
                                    elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                                    arm: armLabel)
                            }
                            return .more
                        }
                    }
                } else {
                    // NAX A/B: alternate the arm per iteration. The callback fires
                    // AFTER an iteration completes, so it records the arm that just
                    // ran and arms the NEXT one; iteration 0's arm is set here.
                    // Even iterations are ON so each cell starts on the patched path.
                    var fusedArm = Self.setNaxArm(on: true, enabled: ctx.naxAB)
                    try LoRATrain.train(
                        model: model, train: [example], validate: [example],
                        optimizer: Self.perOpOptimizer(), tokenizer: c.tokenizer,
                        parameters: fusedParams
                    ) { progress in
                        if case .train(let iteration, let loss, let ips, let tps) = progress {
                            Self.appendPerOpIterRecord(
                                ctx, recordType: "iter", mode: "fused", pass: pass,
                                targetTokens: targetTokens, iterIndex: iteration,
                                warmup: iteration < warmupCount,
                                iterSeconds: 1.0 / ips, tokPerSec: tps, loss: loss,
                                phases: nil,
                                elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                                arm: fusedArm)
                            fusedArm = Self.setNaxArm(
                                on: (iteration + 1) % 2 == 0, enabled: ctx.naxAB)
                        }
                        return .more
                    }
                }

                // --- barriered sub-block (the decomposition) -----------------
                // Continues from the fused sub-block's weights by design (see
                // the doc comment). A fresh AdamW restarts the Adam moments at
                // the seam, which perturbs the first barriered step slightly —
                // expected, and accounted for in the continuity check.
                let lossValueGrad = valueAndGrad(model: model) {
                    (m: Module, arrays: [MLXArray]) -> [MLXArray] in
                    let (ce, ntoks) = LoRATrain.loss(
                        model: m, inputs: arrays[0], targets: arrays[1], lengths: arrays[2])
                    return [ce, ntoks]
                }

                GPU.resetPeakMemory()
                if ctx.pinnedArms {
                    // Pinned arms: one barriered block per arm, same order as
                    // the fused blocks. Fresh AdamW per block (timing is
                    // value-independent; the loss trajectory just restarts its
                    // moments at each seam, as at the fused/barriered seam).
                    for armOn in armOrder {
                        let armLabel = Self.setNaxArm(on: armOn, enabled: true)
                        let optimizer = Self.perOpOptimizer()
                        for iteration in 0 ..< total {
                            let (phases, loss, ntokens, seqLen) =
                                Self.perOpBarrieredIteration(
                                    model: model, tokenizer: c.tokenizer, example: example,
                                    lossValueGrad: lossValueGrad, optimizer: optimizer)
                            Self.appendPerOpIterRecord(
                                ctx, recordType: "iter", mode: "barriered", pass: pass,
                                targetTokens: targetTokens, iterIndex: iteration,
                                warmup: iteration < warmupCount,
                                iterSeconds: phases.total,
                                tokPerSec: Double(ntokens) / phases.total, loss: loss,
                                phases: phases,
                                elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                                arm: armLabel, seqLen: seqLen)
                        }
                    }
                } else {
                    let optimizer = Self.perOpOptimizer()
                    for iteration in 0 ..< total {
                        // Arm set BEFORE the iteration runs — unlike the fused
                        // block, this loop brackets each iteration directly.
                        let arm = Self.setNaxArm(on: iteration % 2 == 0, enabled: ctx.naxAB)
                        let (phases, loss, ntokens, seqLen) = Self.perOpBarrieredIteration(
                            model: model, tokenizer: c.tokenizer, example: example,
                            lossValueGrad: lossValueGrad, optimizer: optimizer)
                        Self.appendPerOpIterRecord(
                            ctx, recordType: "iter", mode: "barriered", pass: pass,
                            targetTokens: targetTokens, iterIndex: iteration,
                            warmup: iteration < warmupCount,
                            iterSeconds: phases.total,
                            tokPerSec: Double(ntokens) / phases.total, loss: loss,
                            phases: phases,
                            elapsed: Date.timeIntervalSinceReferenceDate - runStart,
                            arm: arm, seqLen: seqLen)
                    }
                }
            }
            tlog("perop cell tokens=\(targetTokens) pass=\(pass): complete")
            benchLogLine("perop cell tokens=\(targetTokens) pass=\(pass) complete")
        } catch {
            tlog("perop cell tokens=\(targetTokens) pass=\(pass) error: \(error)")
            benchLogLine("perop cell tokens=\(targetTokens) pass=\(pass) error: \(error)")
        }

        let cellEnd = Date.timeIntervalSinceReferenceDate
        Self.appendPerOpMarker(
            ctx, recordType: "cell_end",
            extra: [
                "target_tokens": targetTokens,
                "pass": pass,
                "elapsed_s": cellEnd - runStart,
                "cell_seconds": cellEnd - cellStart,
                "thermal_state": Self.thermalString(),
                "battery_level": Self.batterySnapshot().level,
                "peak_mem_bytes": Memory.snapshot().peakMemory,
            ])
    }

    /// Set the `MLX_ENABLE_NAX_N` arm for the iteration about to run and return
    /// its label ("on"/"off"), or nil when alternation is disabled (h11 path).
    ///
    /// Safe to call between iterations because the vendored `env::enable_nax_n()`
    /// reads the variable live rather than caching it in a function-local static
    /// the way every sibling accessor does. Dispatch is decided in the C++
    /// backend at EVAL time, so the arm in force when an iteration's graph is
    /// evaluated is the one that runs — which is why this is called immediately
    /// before the iteration and never mid-iteration.
    ///
    /// With the arm "off" the guard collapses to stock upstream MLX exactly, so
    /// the off arm is the true control, not an approximation of one.
    private nonisolated static func setNaxArm(on: Bool, enabled: Bool) -> String? {
        guard enabled else { return nil }
        setenv("MLX_ENABLE_NAX_N", on ? "1" : "0", 1)
        return on ? "on" : "off"
    }

    /// ONE barriered training iteration: `LoRATrain.train`'s per-iteration body
    /// (LoraTrain.swift:275-289) re-expressed with an explicit `eval` between
    /// each phase. Deliberately a REPLICA in the harness, not a fork of
    /// `LoraTrain.swift` — h4/h7's no-fork principle.
    ///
    /// Phase boundaries, and what each one is actually timing:
    ///  1. `data_prep`  — tokenize + pad + build the batch arrays, i.e. what
    ///     `LoRABatchIterator.next()` does for a batch of 1. No barrier: the
    ///     dominant cost here is the tokenizer (synchronous CPU). The array
    ///     construction itself is lazy, so materializing a `[1, N]` Int32
    ///     array lands in `forward` — microseconds, noted rather than fixed,
    ///     since adding a barrier here would change what `data_prep` means.
    ///  2. `graph_build` — the `lossValueGrad(...)` call. Builds the lazy
    ///     forward+backward graph on the CPU and returns; no barrier needed
    ///     because nothing has executed yet.
    ///  3. `forward`   — `eval(lvalue)`. Runs the forward pass and materializes
    ///     the loss. With GC on, only block-boundary activations are retained.
    ///  4. `backward`  — `eval(grad)`. Backward proper PLUS the gradient-
    ///     checkpoint recompute; indistinguishable at this granularity by
    ///     construction (GC-on is the only regime — see the constants block).
    ///  5. `optimizer` — `optimizer.update(model:gradients:)` + `eval(model,
    ///     optimizer)`. AdamW's elementwise math and moment-state update.
    ///  6. `readback`  — `.item()` on the loss and the token count. Expected
    ///     ≈0 since everything is already materialized; timed separately to
    ///     PROVE that rather than assert it.
    ///
    /// NOT sub-divided per transformer block: 36 extra syncs inside
    /// forward/backward would distort the measurement. Per-block/per-kernel
    /// resolution is Tier 2's job.
    private nonisolated static func perOpBarrieredIteration(
        model: Module, tokenizer: Tokenizer, example: String,
        lossValueGrad: (Module, [MLXArray]) -> ([MLXArray], ModuleParameters),
        optimizer: AdamW
    ) -> (phases: PerOpPhaseTimes, loss: Float, ntokens: Int, seqLen: Int) {
        let t0 = Date.timeIntervalSinceReferenceDate

        // 1. data_prep — LoRABatchIterator.next() for batchSize = 1.
        let toks = tokenizer.encode(text: example)
        let length = toks.count
        let batchArray = MLXArray.zeros([1, length], type: Int32.self)
        batchArray[0, 0 ..< length] = MLXArray(toks)
        let inputs = batchArray[0..., .stride(to: -1)]
        let targets = batchArray[0..., 1...]
        let lengths = MLXArray([length])
        let t1 = Date.timeIntervalSinceReferenceDate

        // 2. graph_build
        let (resultArray, grad) = lossValueGrad(model, [inputs, targets, lengths])
        let lvalue = resultArray[0]
        let tokenCount = resultArray[1]
        let t2 = Date.timeIntervalSinceReferenceDate

        // 3. forward
        eval(lvalue)
        let t3 = Date.timeIntervalSinceReferenceDate

        // 4. backward. `grad` is a nested ModuleParameters; flattening to the
        // leaf arrays picks exactly the same set `eval(_:)`'s own collector
        // would, without relying on overload resolution over `Any`.
        eval(grad.flattened().map { $0.1 })
        let t4 = Date.timeIntervalSinceReferenceDate

        // 5. optimizer
        optimizer.update(model: model, gradients: grad)
        eval(model, optimizer)
        let t5 = Date.timeIntervalSinceReferenceDate

        // 6. readback
        let loss = lvalue.item(Float.self)
        let ntokens = tokenCount.item(Int.self)
        let t6 = Date.timeIntervalSinceReferenceDate

        return (
            PerOpPhaseTimes(
                dataPrep: t1 - t0, graphBuild: t2 - t1, forward: t3 - t2,
                backward: t4 - t3, optimizer: t5 - t4, readback: t6 - t5),
            loss, ntokens, inputs.dim(1)
        )
    }

    /// Tier 2: kernel-level Metal capture. A SEPARATE launch — capture
    /// perturbs timing, so nothing but a `capture_run` marker is written to the
    /// Tier-1 JSONL.
    ///
    /// Produces three `.gputrace` bundles (forward / backward / optimizer) from
    /// ONE iteration at `--capture-tokens` (default 500), after
    /// `peropCaptureWarmupIterations` discarded iterations so compile caches
    /// are hot. This mirrors MELT's per-stage kernel split (their
    /// embed/prefill/decode ⇒ our forward/backward/optimizer). The bundles are
    /// pulled to the Mac and read by hand in Xcode's Metal debugger — the
    /// per-kernel table is GUI-only, outside this harness.
    ///
    /// Programmatic capture needs `MetalCaptureEnabled` in the app's Info.plist
    /// (added to LLMEval-Info-Additions.plist). That is checked here up front
    /// via `MTLCaptureManager.supportsDestination(_:)` rather than assumed: a
    /// `GPU.startCapture` that MLX cannot start raises through mlx-c's error
    /// path, which is not catchable from Swift. If capture is unavailable the
    /// run degrades to a logged, explicit "attempted, blocked" marker — the
    /// pre-registered fallback ladder (retry at 250 tok, then `xctrace
    /// --template 'Metal System Trace'` from the Mac) is a manual next step.
    func runPerOpCaptureBenchmark(targetTokens: Int) async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        tlog("perop-capture start session=\(sessionId) tokens=\(targetTokens)")
        benchLogLine("perop-capture start session=\(sessionId) tokens=\(targetTokens)")

        let supported = MTLCaptureManager.shared().supportsDestination(.gpuTraceDocument)
        tlog("perop-capture: gpuTraceDocument supported=\(supported)")

        // Recover a wedged capture session.
        //
        // Killing a capture-mode process between startCapture and stopCapture
        // leaves the session open, and every later run then dies on
        // `[metal::start_capture] Failed to start: Already capturing` — an
        // mlx-c fatal, so it takes the process with it. Observed 2026-08-04 to
        // survive a full device reboot, quitting Xcode, restarting the Mac's
        // gputoolsserviced, and a two-minute quiet period, which is what makes
        // it worth handling in-app rather than operationally.
        //
        // `isCapturing` is checked first because stopping when nothing is
        // capturing is itself an error. If this reports true in a FRESH
        // process, the stuck session belongs to this app and we can cancel it;
        // if it reports false while start still fails, the session is held
        // somewhere outside the app and this cannot fix it — either way the
        // log line resolves the question.
        let wasCapturing = MTLCaptureManager.shared().isCapturing
        tlog("perop-capture: isCapturing at entry=\(wasCapturing)")
        if wasCapturing {
            MTLCaptureManager.shared().stopCapture()
            tlog("perop-capture: stopped a pre-existing capture session")
        }

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("perop-capture: model load failed: \(error)")
            finishTrainBenchmark()
            return
        }

        let ctx = PerOpRunContext(
            sessionId: sessionId,
            modelName: modelConfiguration.name.components(separatedBy: "/").last
                ?? modelConfiguration.name,
            idleMinutes: nil)

        let dir = URL.documentsDirectory.appendingPathComponent(
            TrainBenchConstants.peropCaptureDirName)
        // Start clean. Captures are large and there is no `devicectl` delete
        // subcommand, so without this every run's bundles accumulate on the
        // phone with no way to reclaim the space short of an uninstall — and
        // uninstall is off-limits here (it wipes the model cache and the
        // side-loaded per-user data; h6 lesson).
        if FileManager.default.fileExists(atPath: dir.path) {
            let previous = Self.directorySizeBytes(dir)
            try? FileManager.default.removeItem(at: dir)
            tlog("perop-capture: cleared previous captures (~\(previous / 1_048_576) MB)")
        }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let stamp = Int(Date().timeIntervalSince1970)
        let urls = [
            "forward": dir.appendingPathComponent("forward_\(stamp).gputrace"),
            "backward": dir.appendingPathComponent("backward_\(stamp).gputrace"),
            "optimizer": dir.appendingPathComponent("optimizer_\(stamp).gputrace"),
        ]

        guard supported else {
            Self.appendPerOpMarker(
                ctx, recordType: "capture_run",
                extra: [
                    "target_tokens": targetTokens,
                    "capture_supported": false,
                    "captured": false,
                    "note":
                        "MTLCaptureManager.supportsDestination(.gpuTraceDocument) == false — "
                        + "MetalCaptureEnabled missing/ineffective under this launch; "
                        + "fall back to xctrace Metal System Trace",
                    "thermal_state": Self.thermalString(),
                ])
            benchLogLine("perop-capture BLOCKED: gpuTraceDocument unsupported")
            finishTrainBenchmark()
            return
        }

        var cacheBefore = 0
        var cacheAfter = 0
        var extraCaptureInfo: [String: Any] = [:]
        // Read before entering the container closure: these statics are
        // main-actor isolated and `perform` runs off-actor.
        let backwardLayers = Self.trainBenchmarkCaptureBackwardLayers
        do {
            try await container.perform { c throws -> Void in
                try Self.applyPerOpLoRA(to: c.model)
                let model: Module = c.model
                let example = Self.syntheticExample(
                    targetTokens: targetTokens, tokenizer: c.tokenizer)
                let optimizer = Self.perOpOptimizer()
                let lossValueGrad = valueAndGrad(model: model) {
                    (m: Module, arrays: [MLXArray]) -> [MLXArray] in
                    let (ce, ntoks) = LoRATrain.loss(
                        model: m, inputs: arrays[0], targets: arrays[1], lengths: arrays[2])
                    return [ce, ntoks]
                }

                // Discarded warm iterations so the captured one shows
                // steady-state kernels, not first-run compiles.
                for _ in 0 ..< TrainBenchConstants.peropCaptureWarmupIterations {
                    _ = Self.perOpBarrieredIteration(
                        model: model, tokenizer: c.tokenizer, example: example,
                        lossValueGrad: lossValueGrad, optimizer: optimizer)
                }

                // Drop MLX's buffer cache before capturing. Replay
                // RE-ALLOCATES ALL CAPTURED GPU HEAP STATE, so any buffer MLX
                // merely holds for reuse would become memory the replay guest
                // has to find — and that guest is Apple-signed, so it does not
                // carry this app's `increased-memory-limit` entitlement and
                // dies (`guest app crashed (512)`) well below what this app can
                // allocate. Measured ceiling 2026-08-04: forward @250 tok
                // replays at 1.96 GB; forward @500 (2.13 GB) and backward @250
                // (2.20 GB) both kill the guest.
                //
                // TESTED AND FOUND TO BE A NO-OP HERE, kept as cheap insurance
                // and to record the negative result: `cacheMemory` reads 0 at
                // this point, and the flattened link counts are identical
                // either side of the change (669/1980/71). The barriered
                // iteration's per-phase evals already release everything, so
                // there is no allocator slack in these traces. Unlike vLLM's
                // Metal backend — which fixes the same guest-crash by shrinking
                // a 22 GB KV cache via VLLM_METAL_MEMORY_FRACTION — our trace
                // size is the 4-bit model weights themselves plus live
                // activations, and is therefore irreducible at fixed model and
                // sequence length. Token count is the only lever left.
                cacheBefore = Memory.snapshot().cacheMemory
                Memory.cacheLimit = 0
                Memory.clearCache()
                cacheAfter = Memory.snapshot().cacheMemory

                // The captured iteration — same six phases, with a capture
                // bracketing each of the three GPU-bearing ones.
                let toks = c.tokenizer.encode(text: example)
                let length = toks.count
                let batchArray = MLXArray.zeros([1, length], type: Int32.self)
                batchArray[0, 0 ..< length] = MLXArray(toks)
                let inputs = batchArray[0..., .stride(to: -1)]
                let targets = batchArray[0..., 1...]
                let lengths = MLXArray([length])

                let (resultArray, grad) = lossValueGrad(model, [inputs, targets, lengths])
                let lvalue = resultArray[0]

                GPU.startCapture(url: urls["forward"]!)
                eval(lvalue)
                GPU.stopCapture(url: urls["forward"]!)
                Self.awaitCaptureIdle()

                // PARTIAL BACKWARD when `--capture-backward-layers K` is given.
                //
                // A full backward capture cannot be replayed on this device,
                // and the blocker is RESOURCE COUNT, not bytes — measured
                // 2026-08-04: optimizer 1629 files / forward 1921 files both
                // replay at 1.96 GB, while backward fails at 2792 files /
                // 1.92 GB and 3045 / 1.99 GB. Backward's count is set by the
                // number of distinct tensors across all 36 blocks and barely
                // moves with sequence length (2792 @50 tok vs 3045 @100), so
                // shrinking tokens can never reach the ~1900-file ceiling.
                //
                // The gradient for a LoRA parameter in block N only needs
                // backprop from the loss down to block N — not through the
                // blocks beneath it. Evaluating just the top K blocks'
                // gradients therefore materialises a small subgraph whose
                // kernels are exactly the per-block backward kernels, and
                // SmolLM3's 36 blocks are architecturally identical, so the
                // category shares carry over.
                //
                // Documented deviation: this subgraph includes the lm_head and
                // loss backward (which the full pass also does, once) and
                // excludes the remaining 36-K blocks (identical in structure,
                // repeated). It is a representative sample of backward, not a
                // capture of the whole phase.
                let gradArrays: [MLXArray]
                if let k = backwardLayers {
                    let keep = (TrainBenchConstants.peropLoraLayers - k)...
                    gradArrays = grad.flattened().filter { key, _ in
                        guard let n = Self.layerIndex(in: key) else { return true }
                        return keep.contains(n)
                    }.map { $0.1 }
                } else {
                    gradArrays = grad.flattened().map { $0.1 }
                }
                extraCaptureInfo["backward_grad_arrays"] = gradArrays.count
                extraCaptureInfo["backward_grad_arrays_total"] = grad.flattened().count

                GPU.startCapture(url: urls["backward"]!)
                eval(gradArrays)
                GPU.stopCapture(url: urls["backward"]!)
                Self.awaitCaptureIdle()

                GPU.startCapture(url: urls["optimizer"]!)
                optimizer.update(model: model, gradients: grad)
                eval(model, optimizer)
                GPU.stopCapture(url: urls["optimizer"]!)
                Self.awaitCaptureIdle()

                _ = lvalue.item(Float.self)
            }
        } catch {
            tlog("perop-capture error: \(error)")
            benchLogLine("perop-capture error: \(error)")
        }

        // Record what actually landed on disk — a capture that silently
        // produced nothing is the failure mode the fallback ladder exists for.
        var extra: [String: Any] = [
            "target_tokens": targetTokens,
            "capture_supported": true,
            "capture_warmup_iterations": TrainBenchConstants.peropCaptureWarmupIterations,
            "thermal_state": Self.thermalString(),
            // How much dead buffer cache was dropped before capturing — the
            // difference between a trace the replay guest can re-allocate and
            // one that kills it. See the note at the clearCache call.
            "cache_bytes_before_capture": cacheBefore,
            "cache_bytes_after_clear": cacheAfter,
            "capture_backward_layers": backwardLayers ?? NSNull(),
        ]
        for (k, v) in extraCaptureInfo { extra[k] = v }
        var allPresent = true
        for (phase, url) in urls {
            let exists = FileManager.default.fileExists(atPath: url.path)
            allPresent = allPresent && exists
            extra["capture_\(phase)_path"] =
                "\(TrainBenchConstants.peropCaptureDirName)/\(url.lastPathComponent)"
            extra["capture_\(phase)_exists"] = exists
            extra["capture_\(phase)_bytes"] = Self.directorySizeBytes(url)
            // Without this the bundle cannot leave the device at all — see
            // flattenSymlinks. Done after timing, so it cannot perturb the
            // captured iteration.
            if exists {
                let stats = Self.flattenSymlinks(
                    in: url, budgetBytes: TrainBenchConstants.peropCaptureFlattenBudgetBytes)
                for (k, v) in stats { extra["capture_\(phase)_\(k)"] = v }
                extra["capture_\(phase)_bytes_flattened"] = Self.directorySizeBytes(url)
                tlog("perop-capture flatten \(phase): \(stats)")
            }
        }
        extra["captured"] = allPresent
        Self.appendPerOpMarker(ctx, recordType: "capture_run", extra: extra)
        tlog("perop-capture complete captured=\(allPresent)")
        benchLogLine("perop-capture complete captured=\(allPresent)")
        finishTrainBenchmark()
    }

    /// Replace every symbolic link inside a `.gputrace` bundle with a real copy
    /// of its target, so the bundle can be pulled off the device.
    ///
    /// WHY THIS EXISTS (found 2026-08-04, after the first successful capture):
    /// Metal writes most buffer contents as SYMLINKS — 1247 of the 1629 entries
    /// in a 500-token optimizer capture were `MTLBuffer-*` symlinks, which
    /// `devicectl device info files` labels `SymbolicLink` outright.
    /// `devicectl device copy from` cannot read them: it fails the whole
    /// transfer with `openat(2) POSIX error 62` (ELOOP) on the first one, and
    /// there is no flag to follow or skip links. So a capture that succeeds on
    /// device is still unretrievable until the links are flattened here.
    ///
    /// Uses HARD LINKS, not copies. Measured 2026-08-04: the links point at
    /// other buffers INSIDE the same bundle (`MTLBuffer-26430-0 ->
    /// MTLBuffer-24997-0`) — Metal is deduplicating identical buffers, not
    /// referencing anything outside. Copying each target therefore re-expands
    /// every duplicate: the first attempt blew through a 3 GB/bundle cap with
    /// 273 (forward) and 1816 (backward) links still unresolved, heading for
    /// ~4.4 GB and ~11 GB. A hard link is indistinguishable from a regular file
    /// to `openat`, so `devicectl` reads it happily while the phone keeps the
    /// dedup and the bundle stays its original size. Expansion then happens
    /// only in the Mac-side copy, where there is room for it.
    ///
    /// `budgetBytes` now guards only the copy FALLBACK (used if `linkItem`
    /// fails, e.g. across filesystems). Over-budget links are left in place and
    /// counted, so a partial flatten is visible in the record, not hidden.
    private nonisolated static func flattenSymlinks(
        in bundle: URL, budgetBytes: Int
    ) -> [String: Any] {
        var resolved = 0
        var linked = 0
        var copied = 0
        var failed = 0
        var skippedOverBudget = 0
        var bytes = 0
        var examples: [String] = []

        guard
            let e = FileManager.default.enumerator(
                at: bundle, includingPropertiesForKeys: [.isSymbolicLinkKey],
                options: [.skipsSubdirectoryDescendants])
        else {
            return ["flatten_error": "enumerator failed"]
        }

        for case let url as URL in e {
            let isLink =
                (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) ?? false
            guard isLink == true else { continue }
            do {
                let target = try FileManager.default.destinationOfSymbolicLink(atPath: url.path)
                let targetURL =
                    target.hasPrefix("/")
                    ? URL(fileURLWithPath: target)
                    : URL(fileURLWithPath: target, relativeTo: url.deletingLastPathComponent())
                        .standardizedFileURL
                if examples.count < 3 {
                    examples.append("\(url.lastPathComponent) -> \(target)")
                }
                let size =
                    (try? targetURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) as? Int ?? 0
                try FileManager.default.removeItem(at: url)
                do {
                    // Preferred: costs no additional storage on device.
                    try FileManager.default.linkItem(at: targetURL, to: url)
                    linked += 1
                } catch {
                    // Fallback only; this is the path that can fill the phone,
                    // hence the budget.
                    if bytes + size > budgetBytes {
                        skippedOverBudget += 1
                        continue
                    }
                    try FileManager.default.copyItem(at: targetURL, to: url)
                    copied += 1
                    bytes += size
                }
                resolved += 1
            } catch {
                failed += 1
            }
        }
        return [
            "flatten_resolved": resolved,
            "flatten_linked": linked,
            "flatten_copied": copied,
            "flatten_failed": failed,
            "flatten_skipped_over_budget": skippedOverBudget,
            "flatten_copied_bytes": bytes,
            "flatten_examples": examples,
        ]
    }

    /// Block until the capture manager is idle again after `stopCapture`.
    ///
    /// `stopCapture` FINALISES ASYNCHRONOUSLY. Starting the next phase's
    /// capture too soon fails with `[metal::start_capture] Failed to start:
    /// Already capturing` — an mlx-c fatal, so it kills the process.
    ///
    /// This cost most of an evening on 2026-08-04 because the symptom points
    /// the wrong way: the run dies on the THIRD startCapture (optimizer) while
    /// forward and backward have already written their bundles, so the console
    /// shows the fatal right after the last log line and it reads like a
    /// wedged device. It only appeared once `--capture-backward-layers` made
    /// `eval(grad)` fast enough to close the gap between backward's stop and
    /// optimizer's start; the full-backward path was slow enough to hide it.
    /// Do not "fix" this with a fixed sleep — poll the actual state.
    private nonisolated static func awaitCaptureIdle(timeout: Double = 30.0) {
        let start = Date.timeIntervalSinceReferenceDate
        while MTLCaptureManager.shared().isCapturing {
            if Date.timeIntervalSinceReferenceDate - start > timeout { break }
            usleep(50_000)
        }
        usleep(200_000)  // let the trace document finish landing on disk
    }

    /// Total bytes of a `.gputrace` (a bundle directory), or -1 if absent.
    ///
    /// NOTE: `.fileSizeKey` follows symlinks, so before `flattenSymlinks` this
    /// counts link TARGETS and overstates what is actually stored in the
    /// bundle. The 2026-08-04 capture reported 1.9-2.4 GB per bundle this way.
    private nonisolated static func directorySizeBytes(_ url: URL) -> Int {
        guard FileManager.default.fileExists(atPath: url.path) else { return -1 }
        guard
            let e = FileManager.default.enumerator(
                at: url, includingPropertiesForKeys: [.fileSizeKey])
        else { return -1 }
        var total = 0
        for case let f as URL in e {
            total += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) as? Int ?? 0
        }
        return total
    }

    /// Fresh AdamW on the shared h5/h7/h10 recipe (lr 1e-5, wd 0.01,
    /// bias-corrected to match R5's `adamw_torch`).
    private nonisolated static func perOpOptimizer() -> AdamW {
        AdamW(
            learningRate: TrainBenchConstants.e2eLearningRate,
            weightDecay: TrainBenchConstants.e2eWeightDecay,
            biasCorrection: TrainBenchConstants.e2eAdamBiasCorrection)
    }

    /// Fresh LoRA + per-block GC. Uses `peropLoraLayers` (36 — the FIXED
    /// count, like h8/h10, unlike h1-h7's buggy 28). Re-applying replaces any
    /// previous adapter rather than stacking: `LoRALinear.from` reads the
    /// target layer's base `weight`/`bias` and builds a new layer from them,
    /// so each call yields a freshly initialised adapter (random A, zero B).
    private nonisolated static func applyPerOpLoRA(to model: LanguageModel) throws {
        let config = LoRAConfiguration(
            numLayers: TrainBenchConstants.peropLoraLayers,
            loraParameters: .init(
                rank: TrainBenchConstants.loraRank,
                scale: TrainBenchConstants.loraScale,
                keys: TrainBenchConstants.loraKeys))
        _ = try LoRAContainer.from(model: model, configuration: config)
        if TrainBenchConstants.gradientCheckpointing {
            (model as? SmolLM3Model)?.checkpointGroupSize = 1
        }
    }

    // MARK: - Per-op (h11) record builders

    private nonisolated static func perOpBaseRecord(
        _ c: PerOpRunContext, recordType: String
    ) -> [String: Any] {
        [
            "record_type": recordType,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "idle_minutes": c.idleMinutes ?? NSNull(),
            "token_grid": c.naxAB
                ? TrainBenchConstants.naxABTokenCounts
                : TrainBenchConstants.peropTokenCounts,
            "nax_ab": c.naxAB,
            "warmup_iterations": TrainBenchConstants.peropWarmupIterations,
            "kept_iterations": TrainBenchConstants.peropKeptIterations,
            "cold_ref_tokens": TrainBenchConstants.peropColdRefTokens,
            "cold_ref_iterations": TrainBenchConstants.peropColdRefIterations,
            "sample_interval_s": TrainBenchConstants.peropSampleSeconds,
            "model": c.modelName,
            "batch_size": TrainBenchConstants.trainBatchSize,
            "lora_rank": TrainBenchConstants.loraRank,
            "lora_keys": TrainBenchConstants.loraKeysLabel,
            "num_lora_layers": TrainBenchConstants.peropLoraLayers,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "checkpoint_granularity": TrainBenchConstants.checkpointGranularity,
            "optimizer": "adamw",
            "learning_rate": TrainBenchConstants.e2eLearningRate,
            "weight_decay": TrainBenchConstants.e2eWeightDecay,
            "adam_bias_correction": TrainBenchConstants.e2eAdamBiasCorrection,
            "app_build": c.naxAB
                ? (c.pinnedArms
                    ? TrainBenchConstants.naxABAppBuild + "-pinned"
                    : TrainBenchConstants.naxABAppBuild)
                : naxArmAppBuild(TrainBenchConstants.peropAppBuild),
            "nax_arm": trainBenchmarkNaxArm ?? NSNull(),
            "pinned_arms": c.pinnedArms,
            "bench_schema_version": TrainBenchConstants.peropSchemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": c.sessionId,
            "device_model": trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    /// One training iteration, fused or barriered. `phases` is nil for fused
    /// rows (the whole point of fused mode is that the phases are not
    /// separable); barriered rows carry all six times plus their sum as
    /// `iter_seconds`.
    private nonisolated static func appendPerOpIterRecord(
        _ c: PerOpRunContext, recordType: String, mode: String, pass: String,
        targetTokens: Int, iterIndex: Int, warmup: Bool, iterSeconds: Double,
        tokPerSec: Double, loss: Float, phases: PerOpPhaseTimes?, elapsed: Double,
        arm: String? = nil, seqLen: Int? = nil
    ) {
        var r = perOpBaseRecord(c, recordType: recordType)
        r["mode"] = mode
        r["pass"] = pass
        // Which MLX_ENABLE_NAX_N arm ran THIS iteration ("on"/"off"); nil on
        // the h11 path, where the flag is never touched.
        r["arm"] = arm ?? NSNull()
        // The ACTUAL M seen by the quantized matmuls. LoRABatchIterator returns
        // inputs as batchArray[:, :-1], so M = tokens - 1, and the NAX
        // non-transposed dispatch requires M % 64 == 0. Recorded rather than
        // assumed: if this is not a multiple of 64 the run silently measures
        // the generic kernel in BOTH arms and means nothing.
        r["seq_len"] = seqLen ?? NSNull()
        r["seq_len_aligned_64"] = seqLen.map { $0 % 64 == 0 } ?? NSNull()
        r["target_tokens"] = targetTokens
        r["iter_index"] = iterIndex
        r["warmup"] = warmup
        r["iter_seconds"] = iterSeconds
        r["tok_per_sec"] = tokPerSec
        r["loss"] = loss
        r["elapsed_s"] = elapsed
        r["peak_mem_bytes"] = Memory.snapshot().peakMemory
        r["active_mem_bytes"] = Memory.snapshot().activeMemory
        r["thermal_state"] = thermalString()
        r["low_power_mode"] = ProcessInfo.processInfo.isLowPowerModeEnabled
        if let p = phases {
            r["phase_data_prep_s"] = p.dataPrep
            r["phase_graph_build_s"] = p.graphBuild
            r["phase_forward_s"] = p.forward
            r["phase_backward_s"] = p.backward
            r["phase_optimizer_s"] = p.optimizer
            r["phase_readback_s"] = p.readback
        }
        emitPerOp(r, naxAB: c.naxAB, pinned: c.pinnedArms)
    }

    private nonisolated static func appendPerOpSample(
        _ c: PerOpRunContext, elapsed: Double, level: Double, charging: Bool,
        thermal: String, lpm: Bool, cpuUtilPct: Double?
    ) {
        var r = perOpBaseRecord(c, recordType: "sample")
        r["elapsed_s"] = elapsed
        r["battery_level"] = level
        r["charging"] = charging
        r["thermal_state"] = thermal
        r["low_power_mode"] = lpm
        r["cpu_util_pct"] = cpuUtilPct ?? NSNull()
        emitPerOp(r, naxAB: c.naxAB, pinned: c.pinnedArms)
    }

    private nonisolated static func appendPerOpMarker(
        _ c: PerOpRunContext, recordType: String, extra: [String: Any]
    ) {
        var r = perOpBaseRecord(c, recordType: recordType)
        for (k, v) in extra { r[k] = v }
        emitPerOp(r, naxAB: c.naxAB, pinned: c.pinnedArms)
    }

    /// Append one h11 JSONL line. Its own file: the h9 L run is still pending
    /// against `train_bench_metrics_e2e.jsonl`, which this round must not
    /// touch.
    private nonisolated static func emitPerOp(
        _ record: [String: Any], naxAB: Bool = false, pinned: Bool = false
    ) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(
            naxAB
                ? (pinned
                    ? TrainBenchConstants.naxABPinnedMetricsFileName
                    : TrainBenchConstants.naxABMetricsFileName)
                : naxArmFileName(TrainBenchConstants.peropMetricsFileName))
        peropFileLock.lock()
        defer { peropFileLock.unlock() }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                if let d = line.data(using: .utf8) { try handle.write(contentsOf: d) }
            } else {
                try line.write(to: url, atomically: true, encoding: .utf8)
            }
        } catch {
            // best-effort; a failed line must not crash the run
        }
    }

    // MARK: - NAX qmm_n numerical verification (--verify-qmm-n)
    //
    // Checks the vendored MLX patch (ios/mlx-swift, `env::enable_nax_n()`) that
    // lets non-transposed quantized matmul reach `affine_qmm_n_nax`. Upstream
    // gates NAX on `transpose == true`, so that kernel — although compiled and
    // instantiated — was unreachable, and therefore never exercised by upstream
    // CI. "It compiles" says nothing about whether it is correct, so this runs
    // before any timing work.
    //
    // METHOD. Not an absolute tolerance: NAX for float32 is TF32-precision by
    // design (that is what the `env::enable_tf32() || dtype != float32` clause
    // in the dispatch guard means), so a fixed threshold would be a guess that
    // could fail a healthy kernel or pass a broken one. Instead we calibrate
    // against `affine_qmm_t_nax` — the NAX kernel already running in every
    // forward pass of every training run to date, all of which converge with
    // healthy loss curves, and which therefore DEFINES the precision already
    // empirically accepted on this device.
    //
    // Each kernel is compared against a reference built by dequantizing the
    // weights IT consumed, so 4-bit quantization error cancels out of both and
    // what remains is accumulation error — the only thing the patch could have
    // broken. The transposed and non-transposed cases need genuinely different
    // matrices ([N,K] vs [K,N]); transposing one is not equivalent, because
    // quantization groups along the last axis and would group different numbers.
    //
    // Structural note worth recording: for `transpose == false` the weight is
    // [K, N] and quantization groups along N, so N % 64 == 0 is REQUIRED merely
    // to construct the operand. The plan's worry about `qmm_nax` computing
    // `aligned = N % 64 == 0` but only using it for the transposed kernel name
    // is therefore moot — N is always aligned by construction. Only K can be
    // misaligned, and the surviving `K % 64 == 0` clause routes those to the
    // generic kernel anyway.

    /// One (K, N, M, batch) case under test.
    private struct NaxVerifyCase {
        let label: String
        let k: Int
        let n: Int
        let m: Int
        /// Leading batch dim; 1 means a 2-D `x` (the `batch_0` kernel variant).
        let batch: Int
    }

    /// Deterministic, equidistributed float32 test matrix of shape [r, c],
    /// values roughly uniform on [-1.73, 1.73] (unit variance).
    ///
    /// The data generator matters more than it looks. A first version used
    /// `sin(row*a + col*b)`, which made every row a smooth sinusoid; the
    /// products in `x·V` then cancelled systematically, collapsing the norm of
    /// the reference result and inflating EVERY relative error — the
    /// known-good generic kernel scored 0.99 on one shape and the transposed
    /// reference collapsed to exactly 0. Well-conditioned test data is a
    /// prerequisite for the comparison to mean anything, so this uses a
    /// two-stage fractional hash with no smooth structure.
    ///
    /// Built from separate row/column index vectors rather than a flat index:
    /// float32 represents integers exactly only to 2^24 = 16.7M, and the
    /// lm_head case has 128256*2048 = 263M elements, so a flat `arange` would
    /// silently alias. Row and column extents are both well under 2^24.
    private nonisolated static func naxTestMatrix(_ r: Int, _ c: Int) -> MLXArray {
        let ri = arange(r, dtype: .float32).reshaped([r, 1])
        let ci = arange(c, dtype: .float32).reshaped([1, c])
        // R2 low-discrepancy lattice, then a second decorrelating fract.
        let t = ri * 0.754_877_666_2 + ci * 0.569_840_290_9
        let f = t - floor(t)
        let g = f * 1234.5678 + ri * 0.314_159_265_3
        let h = g - floor(g)
        return (h - 0.5) * 3.4641
    }

    /// Frobenius norm, as a scalar Double. Logged alongside every error so a
    /// degenerate (near-zero) reference is visible rather than silently
    /// inflating the relative errors computed against it.
    private nonisolated static func naxNorm(_ a: MLXArray) -> Double {
        let f = a.asType(.float32)
        return Double(sqrt(sum(f * f)).item(Float.self))
    }

    /// Relative Frobenius error and max absolute deviation of `a` against `b`.
    private nonisolated static func naxRelError(_ a: MLXArray, _ b: MLXArray)
        -> (rel: Double, maxAbs: Double)
    {
        let af = a.asType(.float32)
        let bf = b.asType(.float32)
        let diff = af - bf
        let num = sqrt(sum(diff * diff)).item(Float.self)
        let den = sqrt(sum(bf * bf)).item(Float.self)
        let maxAbs = diff.abs().max().item(Float.self)
        return (Double(num) / Double(Swift.max(den, 1e-30)), Double(maxAbs))
    }

    /// Build and evaluate `make()` with the NAX non-transposed path forced on
    /// or off.
    ///
    /// The `eval` MUST happen inside this call. MLX is lazy: the dispatch
    /// decision is taken when the graph is EVALUATED, not when the op is
    /// queued, so deferring the eval past a later flip of the flag would let
    /// the env var in force at eval time decide — and both arms could silently
    /// take the same path, producing a spurious "identical results" pass.
    /// This is also why the vendored `env::enable_nax_n()` deliberately does
    /// NOT cache in a function-local static the way its siblings do.
    private nonisolated static func naxEval(arm on: Bool, _ make: () -> MLXArray) -> MLXArray {
        setenv("MLX_ENABLE_NAX_N", on ? "1" : "0", 1)
        let y = make()
        eval(y)
        return y
    }

    /// The shape grid: SmolLM3's real backward shapes, plus a batched case and
    /// a deliberately K-misaligned probe.
    ///
    /// For `dX = dY·W` the contraction is over out_features and the output is
    /// in_features, so (K,N) here are (out, in) of the forward projection.
    /// Ordered small→large, and every case streams its result to disk as it
    /// finishes, so a jetsam on the 1 GB lm_head reference costs only that row.
    private nonisolated static var naxVerifyCases: [NaxVerifyCase] {
        var cases: [NaxVerifyCase] = []
        let shapes: [(String, Int, Int)] = [
            ("k_v_proj", 512, 2048),      // 2048 -> 512
            ("q_o_proj", 2048, 2048),     // 2048 -> 2048
            ("down_proj", 2048, 11008),   // 11008 -> 2048
            ("gate_up_proj", 11008, 2048),  // 2048 -> 11008
        ]
        // M is swept ALIGNED and UNALIGNED in pairs. `qmm_n_nax_tgp_impl` does
        // `(void)M`, has its `num_els = min(BM, M - y_row)` bounds lines
        // COMMENTED OUT, and calls `Atile.load` / `Dtile.store` unconditionally
        // — where the transposed sibling carries `kAlignedM`/`kAlignedN` and
        // switches to `load_safe`/`store_safe` on partial tiles. So the kernel
        // should be correct exactly when M % BM == 0 (BM = 64) and garbage
        // otherwise. 250/500 (h11's token grid) are unaligned; 64/128/256/512
        // are aligned.
        for (label, k, n) in shapes {
            for m in [64, 128, 250, 256, 500, 512] {
                cases.append(NaxVerifyCase(label: label, k: k, n: n, m: m, batch: 1))
            }
        }
        // Dedicated M sweep stressing the ported partial-tile paths. BM = 64,
        // and each threadgroup is split into two SM = 32 simdgroup slices, so
        // the interesting values are the ones straddling 32 and 64: a tile with
        // a single live row, a tile one row short, and a slice where the SECOND
        // simdgroup is entirely past the end of the matrix (33..63), which is
        // where `sgp_sm` goes negative.
        for m in [1, 2, 31, 32, 33, 63, 65, 96, 97, 100, 127, 129, 191, 193, 255,
                  257, 511, 513, 999, 1000, 1023, 1025] {
            cases.append(NaxVerifyCase(label: "m_sweep", k: 2048, n: 2048, m: m, batch: 1))
        }
        // Unaligned M under batching, where the per-batch offset arithmetic
        // also has to stay correct.
        for m in [100, 250] {
            cases.append(NaxVerifyCase(label: "m_sweep_batched", k: 2048, n: 2048, m: m, batch: 2))
        }
        // M = 1 never reaches qmm() at all (dispatches to the matrix-VECTOR
        // kernel qmv), so both arms must agree exactly — a control on the
        // harness itself.
        cases.append(NaxVerifyCase(label: "m1_qmv_control", k: 2048, n: 2048, m: 1, batch: 1))
        // Batched variant — exercises the `batch_1` kernel instantiation, both
        // sides of the alignment boundary.
        cases.append(NaxVerifyCase(label: "q_o_batched_aligned", k: 2048, n: 2048, m: 256, batch: 2))
        cases.append(NaxVerifyCase(label: "q_o_batched_unaligned", k: 2048, n: 2048, m: 250, batch: 2))
        // K not a multiple of 64: the surviving `K % 64 == 0` clause should
        // route this to the GENERIC kernel even with the patch active, so both
        // arms must agree exactly. Confirms the patch did not widen the guard
        // further than intended.
        cases.append(NaxVerifyCase(label: "k_misaligned_probe", k: 100, n: 2048, m: 250, batch: 1))
        // lm_head last: its dequantized fp32 reference alone is ~1 GB, and it
        // jetsammed at M=250 on the first run — keep it to the aligned/unaligned
        // pair at small M.
        for m in [64, 250] {
            cases.append(NaxVerifyCase(label: "lm_head", k: 128_256, n: 2048, m: m, batch: 1))
        }
        return cases
    }

    /// Run the whole grid and write one JSONL row per case.
    func runNaxVerifyBenchmark() async {
        let started = Date()
        tlog("nax-verify: start, \(Self.naxVerifyCases.count) cases")
        Self.emitNaxVerify([
            "record_type": "run_start",
            "timestamp_utc": ISO8601DateFormatter().string(from: started),
            "app_build": TrainBenchConstants.peropAppBuild,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "group_size": 64,
            "bits": 4,
            "case_count": Self.naxVerifyCases.count,
        ])

        for c in Self.naxVerifyCases {
            GPU.resetPeakMemory()
            var row: [String: Any] = [
                "record_type": "case",
                "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
                "label": c.label,
                "k": c.k, "n": c.n, "m": c.m, "batch": c.batch,
                "k_aligned_64": c.k % 64 == 0,
            ]

            let xShape = c.batch == 1 ? [c.m, c.k] : [c.batch, c.m, c.k]
            let x = Self.naxTestMatrix(c.batch * c.m, c.k).reshaped(xShape)

            // --- non-transposed (the patched path): w is [K, N] -------------
            let v = Self.naxTestMatrix(c.k, c.n)
            let (vq, vs, vb) = quantized(v, groupSize: 64, bits: 4)
            let refN = matmul(x, dequantized(vq, scales: vs, biases: vb, groupSize: 64, bits: 4))
            eval(refN)

            let yNaxOn = Self.naxEval(arm: true) {
                quantizedMM(
                    x, vq, scales: vs, biases: vb, transpose: false, groupSize: 64, bits: 4)
            }
            let yNaxOff = Self.naxEval(arm: false) {
                quantizedMM(
                    x, vq, scales: vs, biases: vb, transpose: false, groupSize: 64, bits: 4)
            }

            let eOn = Self.naxRelError(yNaxOn, refN)
            let eOff = Self.naxRelError(yNaxOff, refN)
            let eArms = Self.naxRelError(yNaxOn, yNaxOff)
            // Reference scale, so a collapsed reference can never masquerade as
            // a kernel error. Expect ~sqrt(elements * K) for well-conditioned
            // unit-variance operands.
            row["ref_n_norm"] = Self.naxNorm(refN)
            row["y_n_nax_norm"] = Self.naxNorm(yNaxOn)
            row["y_n_generic_norm"] = Self.naxNorm(yNaxOff)
            row["err_n_nax_rel"] = eOn.rel
            row["err_n_nax_maxabs"] = eOn.maxAbs
            row["err_n_generic_rel"] = eOff.rel
            row["err_n_generic_maxabs"] = eOff.maxAbs
            row["err_arms_rel"] = eArms.rel
            row["err_arms_maxabs"] = eArms.maxAbs
            row["n_nax_finite"] = eOn.rel.isFinite && eOn.maxAbs.isFinite

            // --- transposed (the trusted yardstick): w is [N, K] ------------
            // Only constructible when K % 64 == 0, since quantization groups
            // along the last axis of the [N, K] operand.
            if c.k % 64 == 0 {
                let w = Self.naxTestMatrix(c.n, c.k)
                let (wq, ws, wbb) = quantized(w, groupSize: 64, bits: 4)
                let refT = matmul(
                    x,
                    dequantized(wq, scales: ws, biases: wbb, groupSize: 64, bits: 4)
                        .transposed(1, 0))
                let yT = quantizedMM(
                    x, wq, scales: ws, biases: wbb, transpose: true, groupSize: 64, bits: 4)
                eval(refT, yT)
                let eT = Self.naxRelError(yT, refT)
                row["err_t_nax_rel"] = eT.rel
                row["err_t_nax_maxabs"] = eT.maxAbs
                row["ref_t_norm"] = Self.naxNorm(refT)
                row["y_t_nax_norm"] = Self.naxNorm(yT)
                // The headline comparison: how does the newly-reachable kernel
                // compare to the one already trusted in production?
                row["ratio_n_over_t"] = eT.rel > 0 ? eOn.rel / eT.rel : Double.nan
            }

            row["peak_mem_bytes"] = Memory.snapshot().peakMemory
            // Log BEFORE emitting: if a row ever fails to serialise, the
            // console must still carry the numbers (an earlier version crashed
            // inside emit and lost the very case that was misbehaving).
            tlog(
                "nax-verify \(c.label) K=\(c.k) N=\(c.n) M=\(c.m) b=\(c.batch): "
                    + "n_nax=\(eOn.rel) n_gen=\(eOff.rel) "
                    + "t_nax=\(row["err_t_nax_rel"] as? Double ?? -1) "
                    + "arms=\(eArms.rel) refN=\(Self.naxNorm(refN)) "
                    + "refT=\(row["ref_t_norm"] as? Double ?? -1)")
            Self.emitNaxVerify(row)
        }

        // Leave the process with the NAX non-transposed path OFF. It was
        // measured WRONG (see the file-level note and
        // experiments/2026-08-06-mlx-nax-qmm-n-backward.md), so nothing
        // downstream may inherit it enabled.
        setenv("MLX_ENABLE_NAX_N", "0", 1)
        Self.emitNaxVerify([
            "record_type": "run_end",
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "elapsed_s": Date().timeIntervalSince(started),
        ])
        tlog("nax-verify: done in \(Date().timeIntervalSince(started))s")
        exit(0)
    }

    /// JSONSerialization throws `NSInvalidArgumentException` on NaN/Inf, which
    /// kills the process. A non-finite error value is exactly what a broken
    /// kernel would produce — i.e. it is the FINDING — so preserve it as a
    /// string rather than crashing the run or silently dropping the row.
    private nonisolated static func naxSanitize(_ record: [String: Any]) -> [String: Any] {
        var out: [String: Any] = [:]
        for (k, v) in record {
            if let d = v as? Double, !d.isFinite {
                out[k] = d.isNaN ? "nan" : (d > 0 ? "inf" : "-inf")
            } else {
                out[k] = v
            }
        }
        return out
    }

    /// Append one verification row. Its own file — this must not touch the h11
    /// per-op JSONL, whose timing run is still to come.
    private nonisolated static func emitNaxVerify(_ record: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: naxSanitize(record), options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent("nax_verify.jsonl")
        peropFileLock.lock()
        defer { peropFileLock.unlock() }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                if let d = line.data(using: .utf8) { try handle.write(contentsOf: d) }
            } else {
                try line.write(to: url, atomically: true, encoding: .utf8)
            }
        } catch {
            // best-effort; a failed line must not crash the run
        }
    }

    // MARK: - Task-adapter training (h12) — Per-Task-LoRA (LaMP-7) to completion

    /// One pre-tokenized example from the side-loaded corpus. `ids` is the full
    /// rendered token sequence; `[lossStart, lossEnd)` is the assistant-
    /// supervised span in `ids` (built Mac-side via HF
    /// `return_assistant_tokens_mask` — see data/build_task_device_data.py).
    struct TaskAdapterExample: Sendable {
        let id: String
        let ids: [Int32]
        let lossStart: Int
        let lossEnd: Int
    }

    /// Reference-captured carrier for the current microbatch's supervised
    /// position range: `valueAndGrad`'s closure signature only admits
    /// MLXArrays, and slicing needs Swift ints. Mutated and read on the same
    /// (container) actor, strictly between calls — hence @unchecked.
    final class TaskAdapterSpanBox: @unchecked Sendable {
        var range: Range<Int> = 0 ..< 1
    }

    /// Harness-owned AdamW (h12 v3). Exists because checkpoint/resume needs
    /// the optimizer moments, and `MLXOptimizers.AdamW` keeps its state in an
    /// internal store with no public round-trip (h6 finding, still true).
    /// Same math as adamw_torch / MLX AdamW with biasCorrection, wd = 0:
    ///   m ← β1·m + (1−β1)·g;  v ← β2·v + (1−β2)·g²;  t ← t+1
    ///   p ← p − lr · (m/(1−β1ᵗ)) / (√(v/(1−β2ᵗ)) + eps)
    /// Verified by trajectory match against the pre-v3 MLXOptimizers runs
    /// (identical losses through the first 10 steps) and by the resume test
    /// (kill at a checkpoint, resume, losses continue the same trajectory).
    final class TaskAdapterAdamW: @unchecked Sendable {
        var m: [String: MLXArray] = [:]
        var v: [String: MLXArray] = [:]
        var t: Int = 0
        let beta1 = TrainBenchConstants.taskAdapterAdamBeta1
        let beta2 = TrainBenchConstants.taskAdapterAdamBeta2
        let eps = TrainBenchConstants.taskAdapterAdamEps

        func update(model: Module, grads: [(String, MLXArray)], learningRate lr: Float) {
            t += 1
            let bc1 = 1 - pow(beta1, Float(t))
            let bc2 = 1 - pow(beta2, Float(t))
            let current = Dictionary(
                uniqueKeysWithValues: model.trainableParameters().flattened())
            var newParams: [(String, MLXArray)] = []
            for (k, g) in grads {
                let mNew = beta1 * (m[k] ?? MLXArray.zeros(g.shape, dtype: g.dtype))
                    + (1 - beta1) * g
                let vNew = beta2 * (v[k] ?? MLXArray.zeros(g.shape, dtype: g.dtype))
                    + (1 - beta2) * (g * g)
                m[k] = mNew
                v[k] = vNew
                let mHat = mNew / bc1
                let vHat = vNew / bc2
                guard let p = current[k] else { continue }
                newParams.append((k, p - lr * mHat / (MLX.sqrt(vHat) + eps)))
            }
            model.update(parameters: ModuleParameters.unflattened(newParams))
            eval(model)
            eval(Array(m.values) + Array(v.values))
        }
    }

    struct TaskAdapterRunContext: Sendable {
        let sessionId: String
        let modelName: String
        let nExamples: Int
        /// Steps this run will execute (smoke: maxSteps; full: ceil(n/32)).
        let totalSteps: Int
        /// Steps the LR schedule is computed against — ALWAYS the full-corpus
        /// count, so a smoke run is the first N steps of the real schedule.
        let scheduleTotalSteps: Int
        let runName: String
        let naxArm: String?
        let condition: String
        let idleMinutes: Double?
    }

    /// Train the Per-Task-LoRA (LaMP-7) on-device with the canonical task
    /// recipe. Faithful-by-construction pieces, each matching the cluster
    /// reference (`train/checkpoints/per_task_lamp7_1ep_seed0`):
    ///
    ///  * data: pre-tokenized ids + assistant-mask span, rendered with the
    ///    byte-identical tokenizer/chat template the cluster used; the file's
    ///    order IS the seed-0 shuffle (baked in by the builder), consumed
    ///    sequentially — one epoch, one permutation, like HF's seeded sampler.
    ///  * loss: CE summed over the masked span per microbatch; the window's
    ///    gradient is Σ(grads of CE-sums)/Σ(masked tokens) — HF Trainer's
    ///    token-weighted num_items_in_batch normalization.
    ///  * accumulation: 32 microbatches of batch 1 per optimizer step; the
    ///    last window of the epoch is partial (5), matching HF
    ///    drop_last=False (10,437 = 326×32 + 5 → 327 steps).
    ///  * clip: global-norm 1.0 over the normalized grads, pre-optimizer.
    ///  * LR: cosine with ceil(0.03×total) warmup steps, applied per
    ///    optimizer step (verified against the cluster metrics.jsonl).
    ///  * optimizer: AdamW β 0.9/0.999, eps 1e-8, weight decay 0.0
    ///    (explicitly — the harness-legacy e2e constant 0.01 is WRONG here),
    ///    bias-corrected to match adamw_torch.
    ///
    /// Declared deviations (plan Decisions table): 4-bit base, no dropout,
    /// batch 1×32 accumulation ordering, cap 1024 (moot — corpus max 597).
    func runTaskAdapterBenchmark(maxSteps: Int?) async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        // h12 v3: duplicate stderr into a pullable file. The overnight
        // failures die TRACELESSLY under a detached launch — and mlx-c's
        // fatal path prints its message to stderr then exit(-1)s (uncatchable
        // from Swift, no crash report; h11 documented this for start_capture).
        // A detached launch discards stderr, so the one line naming the killer
        // was being thrown away. tlog() also lands here now.
        let stderrURL = URL.documentsDirectory
            .appendingPathComponent("taskadapter_stderr.log")
        freopen(stderrURL.path, "a", stderr)
        setvbuf(stderr, nil, _IONBF, 0)
        FileHandle.standardError.write(
            Data("\n===== launch \(ISO8601DateFormatter().string(from: Date())) =====\n".utf8))

        let sessionId = UUID().uuidString
        let naxArm = Self.trainBenchmarkNaxArm
        if let naxArm {
            setenv("MLX_ENABLE_NAX_N", naxArm == "on" ? "1" : "0", 1)
        }

        // Hard model override — the h5–h11 rounds leave `modelConfiguration`
        // pointing at the a1lamp-FUSED model, which would silently train the
        // wrong artifact (h8's stale-configuration lesson). The task adapter
        // trains on the PLAIN 4-bit base.
        modelConfiguration = ModelConfiguration(
            id: TrainBenchConstants.taskAdapterModelId,
            defaultPrompt: "Why is the sky blue?")

        tlog("taskadapter start session=\(sessionId) naxArm=\(naxArm ?? "absent") "
            + "maxSteps=\(maxSteps.map(String.init) ?? "none") "
            + "build=\(TrainBenchConstants.taskAdapterAppBuild)")
        benchLogLine("taskadapter start session=\(sessionId)")

        guard let examples = Self.loadTaskAdapterData(), !examples.isEmpty else {
            tlog("taskadapter FAILED to load pre-tokenized data "
                + "(expected Documents/\(TrainBenchConstants.taskAdapterDataDirName)/"
                + "\(TrainBenchConstants.taskAdapterDataFileName))")
            benchLogLine("taskadapter FAILED to load data")
            finishTrainBenchmark()
            return
        }
        // Lifecycle forensics (added after the 2026-08-10 freezes): the first
        // full-run attempts froze ~15-20 min in with a signature that cannot
        // distinguish "screen locked → app suspended → GPU revoked" from "MLX/
        // Metal wedge with the training thread holding the allocator". These
        // markers decide it: a `lifecycle` record BEFORE the silence means the
        // OS took the app out (protectedDataWillBecomeUnavailable = the device
        // LOCKED; willResignActive/didEnterBackground = foreground loss);
        // silence with NO lifecycle marker means an in-process wedge. The
        // observer writes synchronously on the main actor — cheap, and worth
        // it: this is systems-characterization data in its own right.
        #if canImport(UIKit)
            let lifecycleEvents: [(Notification.Name, String)] = [
                (UIApplication.willResignActiveNotification, "will_resign_active"),
                (UIApplication.didBecomeActiveNotification, "did_become_active"),
                (UIApplication.didEnterBackgroundNotification, "did_enter_background"),
                (UIApplication.willEnterForegroundNotification, "will_enter_foreground"),
                (UIApplication.protectedDataWillBecomeUnavailableNotification, "device_will_lock"),
                (UIApplication.protectedDataDidBecomeAvailableNotification, "device_unlocked"),
                (ProcessInfo.thermalStateDidChangeNotification, "thermal_change"),
            ]
            let lifecycleSessionId = sessionId
            for (name, label) in lifecycleEvents {
                NotificationCenter.default.addObserver(
                    forName: name, object: nil, queue: .main
                ) { _ in
                    Self.emitTaskAdapter([
                        "record_type": "lifecycle",
                        "event": label,
                        "thermal_state": Self.thermalString(),
                        "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
                        "bench_session_id": lifecycleSessionId,
                        "run_name": "lifecycle",
                    ])
                }
            }
        #endif

        let n = examples.count
        let window = TrainBenchConstants.taskAdapterAccumWindow
        let scheduleTotalSteps = (n + window - 1) / window
        var totalSteps = scheduleTotalSteps
        if let maxSteps, maxSteps > 0 { totalSteps = min(totalSteps, maxSteps) }
        let runName =
            (maxSteps != nil && totalSteps < scheduleTotalSteps)
            ? TrainBenchConstants.taskAdapterRunNameSmoke
            : TrainBenchConstants.taskAdapterRunNameFull

        let adapterDir = URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.taskAdapterAdapterDirName)
            .appendingPathComponent(runName)
        try? FileManager.default.createDirectory(
            at: adapterDir, withIntermediateDirectories: true)
        let adapterURL = adapterDir.appendingPathComponent("adapters.safetensors")

        // Load model (outside any measured window).
        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("taskadapter model load failed: \(error)")
            benchLogLine("taskadapter model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        let modelName =
            modelConfiguration.name.components(separatedBy: "/").last
            ?? modelConfiguration.name
        tlog("taskadapter model loaded: \(modelName) nExamples=\(n) "
            + "totalSteps=\(totalSteps)/\(scheduleTotalSteps)")

        let ctx = TaskAdapterRunContext(
            sessionId: sessionId, modelName: modelName, nExamples: n,
            totalSteps: totalSteps, scheduleTotalSteps: scheduleTotalSteps,
            runName: runName, naxArm: naxArm,
            condition: Self.trainBenchmarkCondition,
            idleMinutes: Self.trainBenchmarkIdleMinutes)

        let benchStart = Date.timeIntervalSinceReferenceDate
        let batteryStart = Self.batterySnapshot()
        Self.appendTaskAdapterMarker(
            ctx, recordType: "run_start",
            extra: [
                "battery_level": batteryStart.level,
                "charging": batteryStart.charging,
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                "thermal_state": Self.thermalString(),
                "adapter_url": adapterURL.lastPathComponent,
                // Order fingerprint: catches a stale/wrong side-loaded file and
                // verifies the Mac control consumed the identical sequence.
                "first_example_ids": examples.prefix(3).map { $0.id },
                "total_corpus_tokens": examples.reduce(0) { $0 + $1.ids.count },
            ])

        // Passive 30s sampler on the main actor (UIDevice is @MainActor) —
        // battery/thermal/CPU/memory over the whole multi-hour session. This
        // run doubles as sustained-training characterization data, so the
        // sampler is a primary deliverable, not bookkeeping.
        let sampler = Task { @MainActor in
            var previousCPUTicks = Self.cpuTicks()
            while !Task.isCancelled {
                let snap = Self.batterySnapshot()
                let (cpuPct, newTicks) = Self.cpuUtilizationPercent(previous: previousCPUTicks)
                previousCPUTicks = newTicks
                Self.appendTaskAdapterSample(
                    ctx,
                    elapsed: Date.timeIntervalSinceReferenceDate - benchStart,
                    level: snap.level, charging: snap.charging,
                    thermal: Self.thermalString(),
                    lpm: ProcessInfo.processInfo.isLowPowerModeEnabled,
                    peak: Memory.snapshot().peakMemory,
                    cpuUtilPct: cpuPct)
                try? await Task.sleep(
                    for: .seconds(TrainBenchConstants.taskAdapterSampleSeconds))
            }
        }

        var trainError: String? = nil
        do {
            try await container.perform { mc in
                // Canonical task LoRA: r=4, scale 2.0 (α8/r4), all seven
                // projections, ALL 36 layers.
                let config = LoRAConfiguration(
                    numLayers: TrainBenchConstants.taskAdapterLoraLayers,
                    loraParameters: .init(
                        rank: TrainBenchConstants.taskAdapterLoraRank,
                        scale: TrainBenchConstants.taskAdapterLoraScale,
                        keys: TrainBenchConstants.taskAdapterLoraKeys))
                _ = try LoRAContainer.from(model: mc.model, configuration: config)

                // Per-block gradient checkpointing (h4/h8: K=1 is Pareto-best).
                if TrainBenchConstants.gradientCheckpointing {
                    (mc.model as? SmolLM3Model)?.checkpointGroupSize = 1
                }
                self.tlog("taskadapter LoRA applied (r=4, 7 proj, 36 layers), GC on")

                // Masked loss via SLICED lm_head (h12b, 2026-08-11): logits are
                // materialised ONLY for the assistant span — `arrays[1]` is the
                // pre-sliced target tokens ids[lossStart..<lossEnd] and
                // `spanBox.range` the matching shifted positions. Identical
                // values and gradients to full-logits-plus-mask (outside-span
                // positions have exactly zero cotangent), but removes the
                // full-sequence [seq, vocab] fp32 transient that put long-
                // example windows on the jetsam wall (measured 6.07 GB at 449
                // tok vs 4.7–5.1 GB baseline; 4/5 sessions died on that step).
                // The span bounds ride in a reference-captured box because
                // valueAndGrad's closure only takes MLXArrays.
                let spanBox = TaskAdapterSpanBox()
                let lossValueGrad = valueAndGrad(model: mc.model) { model, arrays in
                    let llm = model as! SmolLM3Model
                    let logits = llm.logits(arrays[0], positionRange: spanBox.range)
                        .asType(.float32)
                    let ceSum = crossEntropy(logits: logits, targets: arrays[1]).sum()
                    return [ceSum]
                }

                let optimizer = TaskAdapterAdamW()

                // v3 resume: if a valid checkpoint exists in this run's dir,
                // restore adapter weights + Adam moments + step counter and
                // continue. Data order is deterministic (file order), the LR
                // schedule is stateless in the step index, and the moments
                // round-trip exactly — so a resumed run computes the same
                // math as an uninterrupted one; only thermal/timing differ.
                var startStep = 0
                if let ck = Self.loadTaskAdapterCheckpoint(dir: adapterDir) {
                    do {
                        try mc.model.update(
                            parameters: ModuleParameters.unflattened(
                                Array(ck.params)),
                            verify: .noUnusedKeys)
                        optimizer.m = ck.m
                        optimizer.v = ck.v
                        optimizer.t = ck.adamT
                        startStep = ck.nextStep
                        self.tlog("taskadapter RESUMED from checkpoint: "
                            + "next_step=\(ck.nextStep) adam_t=\(ck.adamT)")
                    } catch {
                        self.tlog("taskadapter checkpoint restore FAILED: \(error) "
                            + "— starting fresh")
                        startStep = 0
                    }
                }
                Self.appendTaskAdapterMarker(
                    ctx, recordType: "train_begin",
                    extra: ["resumed_from_step": startStep])

                GPU.resetPeakMemory()

                var exampleIndex = startStep * window
                for step in startStep ..< totalSteps {
                    let stepStart = Date.timeIntervalSinceReferenceDate
                    let microCount = min(window, n - exampleIndex)

                    var accum: [String: MLXArray] = [:]
                    var windowCESum: Float = 0
                    var windowMaskedTokens: Float = 0
                    var windowSeqTokens = 0
                    var microLosses: [Double] = []
                    var microSeqLens: [Int] = []
                    var microIterS: [Double] = []

                    for _ in 0 ..< microCount {
                        let microStart = Date.timeIntervalSinceReferenceDate
                        let ex = examples[exampleIndex]
                        exampleIndex += 1

                        let nTok = ex.ids.count
                        let full = MLXArray(ex.ids).reshaped([1, nTok])
                        let inputs = full[0..., .stride(to: -1)]
                        // Sliced-span loss: logits at shifted positions
                        // [lossStart−1, lossEnd−1) predict exactly the tokens
                        // ids[lossStart..<lossEnd]. Same off-by-one as the
                        // original mask construction, verified against the Mac
                        // reference (smoke check 1) in both implementations.
                        spanBox.range = (ex.lossStart - 1) ..< (ex.lossEnd - 1)
                        let targetsSlice = full[0..., ex.lossStart ..< ex.lossEnd]
                        let spanTokens = Float(ex.lossEnd - ex.lossStart)

                        let (vals, grad) = lossValueGrad(
                            mc.model, [inputs, targetsSlice])
                        let flat = grad.flattened()
                        if accum.isEmpty {
                            for (k, g) in flat { accum[k] = g }
                        } else {
                            for (k, g) in flat { accum[k] = accum[k]! + g }
                        }
                        // Barrier per microbatch so the lazy graph never spans
                        // the accumulation window.
                        eval(Array(accum.values))
                        let ceSum = vals[0].item(Float.self)
                        let ntoks = spanTokens

                        windowCESum += ceSum
                        windowMaskedTokens += ntoks
                        windowSeqTokens += nTok
                        microLosses.append(ntoks > 0 ? Double(ceSum / ntoks) : 0)
                        microSeqLens.append(nTok)
                        microIterS.append(
                            Date.timeIntervalSinceReferenceDate - microStart)
                    }

                    // Token-weighted normalization + global-norm clip 1.0
                    // (norm computed on the NORMALIZED grads, like HF's
                    // clip_grad_norm_ after loss averaging).
                    var sq = MLXArray(Float(0))
                    for g in accum.values { sq = sq + (g * g).sum() }
                    let rawNorm = MLX.sqrt(sq).item(Float.self)
                    let gradNorm = rawNorm / windowMaskedTokens
                    let clip = TrainBenchConstants.taskAdapterGradClipNorm
                    let clipScale: Float = gradNorm > clip ? clip / gradNorm : 1.0
                    let finalScale = clipScale / windowMaskedTokens

                    let lr = Self.taskAdapterLR(
                        step: step, scheduleTotalSteps: scheduleTotalSteps)
                    let scaled = accum.map { ($0.key, $0.value * finalScale) }
                    optimizer.update(
                        model: mc.model, grads: scaled, learningRate: lr)

                    let now = Date.timeIntervalSinceReferenceDate
                    let windowLoss = windowMaskedTokens > 0
                        ? windowCESum / windowMaskedTokens : 0
                    let peak = Memory.snapshot().peakMemory
                    GPU.resetPeakMemory()
                    Self.appendTaskAdapterStep(
                        ctx, step: step + 1, loss: windowLoss, lr: lr,
                        gradNorm: gradNorm, clipScale: clipScale,
                        nMicro: microCount,
                        windowMaskedTokens: Int(windowMaskedTokens),
                        windowSeqTokens: windowSeqTokens,
                        windowS: now - stepStart,
                        elapsed: now - benchStart, peak: peak,
                        thermal: Self.thermalString(),
                        lpm: ProcessInfo.processInfo.isLowPowerModeEnabled,
                        microLosses: microLosses, microSeqLens: microSeqLens,
                        microIterS: microIterS)
                    if (step + 1) % 10 == 0 || step == 0 {
                        self.tlog("taskadapter step \(step + 1)/\(totalSteps) "
                            + "loss=\(windowLoss) lr=\(lr) gradNorm=\(gradNorm)")
                    }
                    if (step + 1) % TrainBenchConstants.taskAdapterCheckpointEverySteps == 0
                        || step + 1 == totalSteps
                    {
                        do {
                            try Self.saveTaskAdapterCheckpoint(
                                dir: adapterDir, model: mc.model,
                                optimizer: optimizer, nextStep: step + 1)
                        } catch {
                            self.tlog("taskadapter checkpoint save FAILED: \(error)")
                        }
                    }
                }

                try LoRATrain.saveLoRAWeights(model: mc.model, url: adapterURL)
                self.tlog("taskadapter adapter saved -> \(adapterURL.path)")
            }
        } catch {
            trainError = "\(error)"
            tlog("taskadapter training error (possible OOM/jetsam): \(error)")
            benchLogLine("taskadapter training error: \(error)")
        }

        sampler.cancel()
        let batteryEnd = Self.batterySnapshot()
        let adapterSaved = FileManager.default.fileExists(atPath: adapterURL.path)
        if trainError == nil {
            Self.writeTaskAdapterMeta(
                ctx, adapterDir: adapterDir,
                elapsed: Date.timeIntervalSinceReferenceDate - benchStart)
        }
        Self.appendTaskAdapterMarker(
            ctx, recordType: trainError == nil ? "run_end" : "error",
            extra: [
                "battery_level": batteryStart.level,
                "battery_level_end": batteryEnd.level,
                "charging": batteryEnd.charging,
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                "thermal_state": Self.thermalString(),
                "elapsed_s": Date.timeIntervalSinceReferenceDate - benchStart,
                "adapter_saved": adapterSaved,
                "error": trainError ?? NSNull(),
            ])
        tlog("taskadapter complete adapter_saved=\(adapterSaved) "
            + "error=\(trainError ?? "none")")
        benchLogLine("taskadapter complete adapter_saved=\(adapterSaved)")
        finishTrainBenchmark()
    }

    // MARK: - Task-adapter (h12 v3) checkpoint/resume

    struct TaskAdapterCheckpoint {
        let params: [String: MLXArray]
        let m: [String: MLXArray]
        let v: [String: MLXArray]
        let adamT: Int
        let nextStep: Int
    }

    /// Crash-safe save: everything is written into `checkpoint.new/`, which is
    /// swapped in only when complete (meta.json written last). A kill at ANY
    /// point leaves either the previous consistent checkpoint or a `.new` dir
    /// the loader ignores — never a torn state where new weights pair with an
    /// old step counter (which would silently re-run optimizer steps on
    /// already-updated weights).
    private nonisolated static func saveTaskAdapterCheckpoint(
        dir: URL, model: Module, optimizer: TaskAdapterAdamW, nextStep: Int
    ) throws {
        let fm = FileManager.default
        let newDir = dir.appendingPathComponent("checkpoint.new")
        let ckDir = dir.appendingPathComponent("checkpoint")
        let oldDir = dir.appendingPathComponent("checkpoint.old")
        try? fm.removeItem(at: newDir)
        try fm.createDirectory(at: newDir, withIntermediateDirectories: true)

        let params = Dictionary(
            uniqueKeysWithValues: model.trainableParameters().flattened())
        try save(arrays: params, url: newDir.appendingPathComponent("adapter.safetensors"))
        try save(arrays: optimizer.m, url: newDir.appendingPathComponent("adam_m.safetensors"))
        try save(arrays: optimizer.v, url: newDir.appendingPathComponent("adam_v.safetensors"))
        let meta: [String: Any] = [
            "next_step": nextStep,
            "adam_t": optimizer.t,
            "n_params": params.count,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
        ]
        let metaData = try JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys])
        try metaData.write(to: newDir.appendingPathComponent("meta.json"))

        try? fm.removeItem(at: oldDir)
        if fm.fileExists(atPath: ckDir.path) {
            try fm.moveItem(at: ckDir, to: oldDir)
        }
        try fm.moveItem(at: newDir, to: ckDir)
        try? fm.removeItem(at: oldDir)
    }

    /// Load `checkpoint/`, falling back to `checkpoint.old/` if the primary is
    /// torn (missing meta or unreadable arrays). Returns nil when neither is
    /// usable — the caller starts fresh.
    private nonisolated static func loadTaskAdapterCheckpoint(dir: URL) -> TaskAdapterCheckpoint? {
        for name in ["checkpoint", "checkpoint.old"] {
            let ck = dir.appendingPathComponent(name)
            let metaURL = ck.appendingPathComponent("meta.json")
            guard
                let metaData = try? Data(contentsOf: metaURL),
                let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
                let nextStep = meta["next_step"] as? Int,
                let adamT = meta["adam_t"] as? Int,
                let nParams = meta["n_params"] as? Int,
                let params = try? MLX.loadArrays(url: ck.appendingPathComponent("adapter.safetensors")),
                let m = try? MLX.loadArrays(url: ck.appendingPathComponent("adam_m.safetensors")),
                let v = try? MLX.loadArrays(url: ck.appendingPathComponent("adam_v.safetensors")),
                params.count == nParams, m.count == nParams, v.count == nParams
            else { continue }
            return TaskAdapterCheckpoint(
                params: params, m: m, v: v, adamT: adamT, nextStep: nextStep)
        }
        return nil
    }

    /// HF cosine-with-warmup, 0-based step index. Verified against the cluster
    /// reference's metrics.jsonl (see TrainBenchConstants doc).
    private nonisolated static func taskAdapterLR(
        step: Int, scheduleTotalSteps: Int
    ) -> Float {
        let warmup = Int(
            ceil(Double(scheduleTotalSteps) * TrainBenchConstants.taskAdapterWarmupRatio))
        if step < warmup {
            return TrainBenchConstants.taskAdapterBaseLR * Float(step) / Float(max(1, warmup))
        }
        let progress =
            Double(step - warmup) / Double(max(1, scheduleTotalSteps - warmup))
        return TrainBenchConstants.taskAdapterBaseLR
            * Float(0.5 * (1.0 + cos(Double.pi * progress)))
    }

    /// Parse the side-loaded pre-tokenized corpus. Returns nil on ANY malformed
    /// line — a partially-consumed corpus would silently change the step count
    /// and data order, which is worse than failing loudly.
    private nonisolated static func loadTaskAdapterData() -> [TaskAdapterExample]? {
        let url = URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.taskAdapterDataDirName)
            .appendingPathComponent(TrainBenchConstants.taskAdapterDataFileName)
        guard let content = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var out: [TaskAdapterExample] = []
        for line in content.split(separator: "\n") {
            guard
                let d = line.data(using: .utf8),
                let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                let ids = obj["input_ids"] as? [Int],
                let lossStart = obj["loss_start"] as? Int,
                let lossEnd = obj["loss_end"] as? Int,
                ids.count >= 2, lossStart >= 1, lossEnd > lossStart, lossEnd <= ids.count
            else { return nil }
            let id =
                (obj["id"] as? NSNumber)?.stringValue ?? (obj["id"] as? String ?? "?")
            out.append(
                TaskAdapterExample(
                    id: id, ids: ids.map(Int32.init),
                    lossStart: lossStart, lossEnd: lossEnd))
        }
        return out
    }

    // MARK: - Task-adapter (h12) record builders

    private nonisolated static func taskAdapterBaseRecord(
        _ c: TaskAdapterRunContext, recordType: String
    ) -> [String: Any] {
        [
            "record_type": recordType,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "task": "LaMP_7",
            "run_name": c.runName,
            "condition": c.condition,
            "nax_arm": c.naxArm ?? NSNull(),
            "idle_minutes": c.idleMinutes ?? NSNull(),
            "n_examples": c.nExamples,
            "total_steps": c.totalSteps,
            "schedule_total_steps": c.scheduleTotalSteps,
            "accum_window": TrainBenchConstants.taskAdapterAccumWindow,
            "model": c.modelName,
            "lora_rank": TrainBenchConstants.taskAdapterLoraRank,
            "lora_scale": TrainBenchConstants.taskAdapterLoraScale,
            "lora_keys": TrainBenchConstants.taskAdapterLoraKeysLabel,
            "num_lora_layers": TrainBenchConstants.taskAdapterLoraLayers,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "optimizer": "adamw",
            "base_learning_rate": TrainBenchConstants.taskAdapterBaseLR,
            "lr_schedule": "cosine_warmup0.03",
            "weight_decay": TrainBenchConstants.taskAdapterWeightDecay,
            "grad_clip_norm": TrainBenchConstants.taskAdapterGradClipNorm,
            "adam_bias_correction": true,
            // Schema v2 (2026-08-11): loss computes logits only on the sliced
            // assistant span (see SmolLM3Model.logits(_:positionRange:)) —
            // numerically identical to v1's full-logits+mask, but without the
            // full-sequence fp32 logits transient.
            "loss_impl": "sliced_lm_head",
            "app_build": TrainBenchConstants.taskAdapterAppBuild,
            "bench_schema_version": TrainBenchConstants.taskAdapterSchemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": c.sessionId,
            "device_model": trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    private nonisolated static func appendTaskAdapterStep(
        _ c: TaskAdapterRunContext, step: Int, loss: Float, lr: Float,
        gradNorm: Float, clipScale: Float, nMicro: Int,
        windowMaskedTokens: Int, windowSeqTokens: Int, windowS: Double,
        elapsed: Double, peak: Int, thermal: String, lpm: Bool,
        microLosses: [Double], microSeqLens: [Int], microIterS: [Double]
    ) {
        var r = taskAdapterBaseRecord(c, recordType: "opt_step")
        r["step"] = step
        r["loss"] = Double(loss)
        r["learning_rate"] = Double(lr)
        r["grad_norm_preclip"] = Double(gradNorm)
        r["clip_scale"] = Double(clipScale)
        r["n_micro"] = nMicro
        r["window_masked_tokens"] = windowMaskedTokens
        r["window_seq_tokens"] = windowSeqTokens
        r["window_s"] = windowS
        r["elapsed_s"] = elapsed
        r["peak_mem_bytes"] = peak
        r["thermal_state"] = thermal
        r["low_power_mode"] = lpm
        r["micro_losses"] = microLosses
        r["micro_seq_lens"] = microSeqLens
        r["micro_iter_s"] = microIterS
        emitTaskAdapter(r)
    }

    private nonisolated static func appendTaskAdapterSample(
        _ c: TaskAdapterRunContext, elapsed: Double, level: Double, charging: Bool,
        thermal: String, lpm: Bool, peak: Int, cpuUtilPct: Double?
    ) {
        var r = taskAdapterBaseRecord(c, recordType: "sample")
        r["elapsed_s"] = elapsed
        r["battery_level"] = level
        r["charging"] = charging
        r["thermal_state"] = thermal
        r["low_power_mode"] = lpm
        r["peak_mem_bytes"] = peak
        r["cpu_util_pct"] = cpuUtilPct ?? NSNull()
        emitTaskAdapter(r)
    }

    private nonisolated static func appendTaskAdapterMarker(
        _ c: TaskAdapterRunContext, recordType: String, extra: [String: Any]
    ) {
        var r = taskAdapterBaseRecord(c, recordType: recordType)
        for (k, v) in extra { r[k] = v }
        emitTaskAdapter(r)
    }

    /// Sidecar next to the saved adapter with everything the Mac-side MLX→PEFT
    /// converter needs (rank/scale/keys, provenance).
    private nonisolated static func writeTaskAdapterMeta(
        _ c: TaskAdapterRunContext, adapterDir: URL, elapsed: Double
    ) {
        let meta: [String: Any] = [
            "run_name": c.runName,
            "task": "LaMP_7",
            "session_id": c.sessionId,
            "model": c.modelName,
            "lora_rank": TrainBenchConstants.taskAdapterLoraRank,
            "lora_alpha": TrainBenchConstants.taskAdapterLoraScale
                * Float(TrainBenchConstants.taskAdapterLoraRank),
            "lora_scale": TrainBenchConstants.taskAdapterLoraScale,
            "lora_keys": TrainBenchConstants.taskAdapterLoraKeys,
            "num_lora_layers": TrainBenchConstants.taskAdapterLoraLayers,
            "total_steps": c.totalSteps,
            "n_examples": c.nExamples,
            "nax_arm": c.naxArm ?? "absent",
            "elapsed_s": elapsed,
            "app_build": TrainBenchConstants.taskAdapterAppBuild,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
        ]
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: meta, options: [.sortedKeys, .prettyPrinted])
        else { return }
        try? data.write(to: adapterDir.appendingPathComponent("adapter_meta.json"))
    }

    private nonisolated static func emitTaskAdapter(_ record: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(
            TrainBenchConstants.taskAdapterMetricsFileName)
        taskAdapterFileLock.lock()
        defer { taskAdapterFileLock.unlock() }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                if let d = line.data(using: .utf8) { try handle.write(contentsOf: d) }
            } else {
                try line.write(to: url, atomically: true, encoding: .utf8)
            }
        } catch {
            // best-effort; a failed line must not crash a multi-hour run
        }
    }

    // MARK: - Bundled data

    /// Load a bundled `<name>.jsonl` of `{"text": ...}` lines via MLXLLM's parser
    /// (same path LoRATrainingExample uses).
    private static func loadBundledLoRAData(_ name: String) -> [String]? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "jsonl") else {
            return nil
        }
        return try? MLXLLM.loadLoRAData(url: url)
    }

    // MARK: - Off-main helpers (nonisolated so the train callback can call them)

    private nonisolated static func thermalString() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// Hardware identifier, e.g. "iPhone18,1". Local copy (the inference
    /// harness's is `private` in another file).
    private nonisolated static func trainHwModel() -> String {
        var sysinfo = utsname()
        uname(&sysinfo)
        let mirror = Mirror(reflecting: sysinfo.machine)
        let id = mirror.children.compactMap { ($0.value as? Int8) }
            .filter { $0 != 0 }.map { String(UnicodeScalar(UInt8($0))) }.joined()
        return id.isEmpty ? "unknown" : id
    }

    // MARK: - CPU utilization (h9 energy round — secondary diagnostic only,
    // NOT part of the joules computation; see
    // experiments/2026-07-26-ondevice-energy-h9-plan.md. Aggregate across all
    // cores via `host_statistics`/`HOST_CPU_LOAD_INFO` (simpler than the
    // per-core `host_processor_info`, which needs a dynamically-allocated
    // out-array + `vm_deallocate` — unnecessary for a sanity-check signal).
    // No public per-process GPU-utilization API exists on iOS, so this can
    // never be a full power model by itself.

    /// Raw cumulative tick counts since boot, aggregated across all cores.
    /// `nil` on the (unexpected) failure path so a bad read degrades to a
    /// missing `cpu_util_pct` rather than a crash.
    private nonisolated static func cpuTicks() -> host_cpu_load_info_data_t? {
        var info = host_cpu_load_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return info
    }

    /// %busy (user+system+nice / total) over the interval between `previous`
    /// and a fresh read taken now. Returns the fresh read alongside so the
    /// caller can thread it into the next call — ticks are cumulative since
    /// boot, so a single snapshot alone says nothing about the sampling
    /// window. `pct` is `nil` on the first call of a run (no previous
    /// reading yet) or on a read failure.
    private nonisolated static func cpuUtilizationPercent(
        previous: host_cpu_load_info_data_t?
    ) -> (pct: Double?, current: host_cpu_load_info_data_t?) {
        guard let current = cpuTicks() else { return (nil, nil) }
        guard let previous else { return (nil, current) }
        let userDelta = Double(current.cpu_ticks.0 &- previous.cpu_ticks.0)
        let systemDelta = Double(current.cpu_ticks.1 &- previous.cpu_ticks.1)
        let idleDelta = Double(current.cpu_ticks.2 &- previous.cpu_ticks.2)
        let niceDelta = Double(current.cpu_ticks.3 &- previous.cpu_ticks.3)
        let total = userDelta + systemDelta + idleDelta + niceDelta
        guard total > 0 else { return (nil, current) }
        return (100.0 * (userDelta + systemDelta + niceDelta) / total, current)
    }

    // MARK: - Battery (main actor — UIDevice is @MainActor)

    private static func batterySnapshot() -> BatterySnapshot {
        var level = -1.0
        var charging = false
        #if canImport(UIKit)
            let device = UIDevice.current
            level = Double(device.batteryLevel)
            switch device.batteryState {
            case .charging, .full: charging = true
            default: charging = false
            }
        #endif
        return BatterySnapshot(level: level, charging: charging)
    }

    // MARK: - Record writers

    private func writeTrainRecord(
        sample s: TrainWindowSample, batchSize: Int, seqCap: Int, sessionId: String,
        batteryStart: BatterySnapshot, batteryEnd: BatterySnapshot, nTrain: Int
    ) {
        let modelName =
            modelConfiguration.name.components(separatedBy: "/").last
            ?? modelConfiguration.name

        let record: [String: Any] = [
            "record_type": "train",
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "step": s.step,
            "batch_size": batchSize,
            "seq_cap": seqCap,
            "iter_per_sec": s.iterPerSec,
            "tok_per_sec": s.tokPerSec,
            "elapsed_s": s.elapsedS,
            "peak_mem_bytes": s.peakMemBytes,
            "thermal_state": s.thermalState,
            "battery_level": batteryStart.level,
            "battery_level_end": batteryEnd.level,
            "charging": batteryStart.charging,
            "low_power_mode": s.lowPowerMode,
            "model": modelName,
            "lora_rank": TrainBenchConstants.loraRank,
            "lora_keys": TrainBenchConstants.loraKeysLabel,
            "num_lora_layers": TrainBenchConstants.loraLayers,
            "num_train_examples": nTrain,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "checkpoint_granularity": TrainBenchConstants.checkpointGranularity,
            "iterations_total": TrainBenchConstants.iterations,
            "steps_per_report": TrainBenchConstants.stepsPerReport,
            "app_build": TrainBenchConstants.appBuild,
            "bench_schema_version": TrainBenchConstants.schemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": sessionId,
            "device_model": Self.trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
        emit(record)
    }

    private func writeOOMRecord(
        batchSize: Int, seqCap: Int, sessionId: String,
        batteryStart: BatterySnapshot, batteryEnd: BatterySnapshot, nTrain: Int
    ) {
        let modelName =
            modelConfiguration.name.components(separatedBy: "/").last
            ?? modelConfiguration.name

        let record: [String: Any] = [
            "record_type": "oom",
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            // step at failure is not recoverable through the thrown error; the
            // failing (batch_size, seq_cap) is the actionable field (decision 9).
            "step": NSNull(),
            "batch_size": batchSize,
            "seq_cap": seqCap,
            "iter_per_sec": NSNull(),
            "tok_per_sec": NSNull(),
            "elapsed_s": NSNull(),
            "peak_mem_bytes": NSNull(),
            "thermal_state": Self.thermalString(),
            "battery_level": batteryStart.level,
            "battery_level_end": batteryEnd.level,
            "charging": batteryStart.charging,
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "model": modelName,
            "lora_rank": TrainBenchConstants.loraRank,
            "lora_keys": TrainBenchConstants.loraKeysLabel,
            "num_lora_layers": TrainBenchConstants.loraLayers,
            "num_train_examples": nTrain,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "checkpoint_granularity": TrainBenchConstants.checkpointGranularity,
            "iterations_total": TrainBenchConstants.iterations,
            "steps_per_report": TrainBenchConstants.stepsPerReport,
            "app_build": TrainBenchConstants.appBuild,
            "bench_schema_version": TrainBenchConstants.schemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": sessionId,
            "device_model": Self.trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
        emit(record)
    }

    /// Sentinel written BEFORE a cell trains (decision: cap-sweep). If the cell
    /// then SIGKILLs (jetsam, uncatchable) the JSONL has this `cap_start` but no
    /// `train` rows for the cap — pinpointing the OOM threshold.
    private func writeCapStartRecord(
        batchSize: Int, seqCap: Int, sessionId: String, battery: BatterySnapshot,
        nTrain: Int
    ) {
        let modelName =
            modelConfiguration.name.components(separatedBy: "/").last
            ?? modelConfiguration.name
        let record: [String: Any] = [
            "record_type": "cap_start",
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "batch_size": batchSize,
            "seq_cap": seqCap,
            "thermal_state": Self.thermalString(),
            "battery_level": battery.level,
            "charging": battery.charging,
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "model": modelName,
            "lora_rank": TrainBenchConstants.loraRank,
            "lora_keys": TrainBenchConstants.loraKeysLabel,
            "num_lora_layers": TrainBenchConstants.loraLayers,
            "num_train_examples": nTrain,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "checkpoint_granularity": TrainBenchConstants.checkpointGranularity,
            "iterations_total": TrainBenchConstants.iterations,
            "steps_per_report": TrainBenchConstants.stepsPerReport,
            "app_build": TrainBenchConstants.appBuild,
            "bench_schema_version": TrainBenchConstants.schemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": sessionId,
            "device_model": Self.trainHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
        emit(record)
    }

    /// Serialize + append one record to the training-benchmark JSONL (separate
    /// file from the inference harness's `bench_metrics.jsonl`).
    private func emit(_ record: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else {
            benchLogLine("failed to serialize train bench record")
            return
        }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(
            TrainBenchConstants.metricsFileName)
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                defer { try? handle.close() }
                try handle.seekToEnd()
                if let d = line.data(using: .utf8) { try handle.write(contentsOf: d) }
            } else {
                try line.write(to: url, atomically: true, encoding: .utf8)
            }
        } catch {
            benchLogLine("failed to write train bench record: \(error)")
        }
    }
}
