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
    }

    /// Value of a `--flag <value>` launch arg, or nil if absent/trailing.
    private static func launchArgValue(_ flag: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
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
        if mode == .perop {
            await runPerOpBenchmark(idleMinutes: Self.trainBenchmarkIdleMinutes)
            return
        }
        if mode == .peropCapture {
            await runPerOpCaptureBenchmark(targetTokens: Self.trainBenchmarkCaptureTokens)
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
        let adapterURL = Self.e2eAdapterURL(user: user)
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
        let ctx = E2ERunContext(
            user: user, profileSize: nUser, condition: condition, nUser: nUser,
            iterations: iterations, seqCap: TrainBenchConstants.e2eSeqCap,
            modelName: modelName, sessionId: sessionId)

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

    private nonisolated static func e2eAdapterURL(user: String) -> URL {
        URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.e2eAdapterDirName)
            .appendingPathComponent("adapter_\(user).safetensors")
    }

    // MARK: - E2E record builders (nonisolated → callable from the train callback)

    private nonisolated static func e2eBaseRecord(_ c: E2ERunContext, recordType: String) -> [String: Any] {
        [
            "record_type": recordType,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "user_fingerprint": c.user,
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
        emitE2E(r)
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
        emitE2E(r)
    }

    private nonisolated static func appendE2EMarker(
        _ c: E2ERunContext, recordType: String, extra: [String: Any]
    ) {
        var r = e2eBaseRecord(c, recordType: recordType)
        for (k, v) in extra { r[k] = v }
        emitE2E(r)
    }

    /// Serialize + append one E2E record to `train_bench_metrics_e2e.jsonl`.
    /// nonisolated + lock-guarded (see file-scope `e2eFileLock`): the off-actor
    /// train callback and the main-actor battery sampler both write concurrently.
    private nonisolated static func emitE2E(_ record: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(
            TrainBenchConstants.e2eMetricsFileName)
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
            "app_build": TrainBenchConstants.tokentimeAppBuild,
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
        let url = URL.documentsDirectory.appendingPathComponent(fileName)
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
            "app_build": TrainBenchConstants.granularityAppBuild,
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
            TrainBenchConstants.granularityMetricsFileName)
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
            "app_build": TrainBenchConstants.thermalSelfLimitAppBuild,
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
            TrainBenchConstants.thermalSelfLimitMetricsFileName)
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
            "app_build": TrainBenchConstants.thermalAppBuild,
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
            TrainBenchConstants.thermalMetricsFileName)
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
    func runPerOpBenchmark(idleMinutes: Double?) async {
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
            idleMinutes: idleMinutes)

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
        await runPerOpColdRef(container: container, ctx: ctx, runStart: runStart)

        // 2+3. Both passes, ascending tokens, no cooldown gate anywhere.
        for pass in TrainBenchConstants.peropPasses {
            for tokens in TrainBenchConstants.peropTokenCounts {
                await runPerOpCell(
                    container: container, ctx: ctx, targetTokens: tokens, pass: pass,
                    runStart: runStart)
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
                        Self.appendPerOpIterRecord(
                            ctx, recordType: "cold_ref", mode: "fused", pass: "cold_ref",
                            targetTokens: TrainBenchConstants.peropColdRefTokens,
                            iterIndex: iteration, warmup: iteration < 1,
                            iterSeconds: 1.0 / ips, tokPerSec: tps, loss: loss,
                            phases: nil,
                            elapsed: Date.timeIntervalSinceReferenceDate - runStart)
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
        runStart: Double
    ) async {
        let cellStart = Date.timeIntervalSinceReferenceDate
        let battery = Self.batterySnapshot()
        Self.appendPerOpMarker(
            ctx, recordType: "cell_start",
            extra: [
                "target_tokens": targetTokens,
                "pass": pass,
                "elapsed_s": cellStart - runStart,
                "thermal_state": Self.thermalString(),
                "battery_level": battery.level,
                "charging": battery.charging,
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            ])
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
                            elapsed: Date.timeIntervalSinceReferenceDate - runStart)
                    }
                    return .more
                }

                // --- barriered sub-block (the decomposition) -----------------
                // Continues from the fused sub-block's weights by design (see
                // the doc comment). A fresh AdamW restarts the Adam moments at
                // the seam, which perturbs the first barriered step slightly —
                // expected, and accounted for in the continuity check.
                let optimizer = Self.perOpOptimizer()
                let lossValueGrad = valueAndGrad(model: model) {
                    (m: Module, arrays: [MLXArray]) -> [MLXArray] in
                    let (ce, ntoks) = LoRATrain.loss(
                        model: m, inputs: arrays[0], targets: arrays[1], lengths: arrays[2])
                    return [ce, ntoks]
                }

                GPU.resetPeakMemory()
                for iteration in 0 ..< total {
                    let (phases, loss, ntokens) = Self.perOpBarrieredIteration(
                        model: model, tokenizer: c.tokenizer, example: example,
                        lossValueGrad: lossValueGrad, optimizer: optimizer)
                    Self.appendPerOpIterRecord(
                        ctx, recordType: "iter", mode: "barriered", pass: pass,
                        targetTokens: targetTokens, iterIndex: iteration,
                        warmup: iteration < warmupCount,
                        iterSeconds: phases.total,
                        tokPerSec: Double(ntokens) / phases.total, loss: loss,
                        phases: phases,
                        elapsed: Date.timeIntervalSinceReferenceDate - runStart)
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
    ) -> (phases: PerOpPhaseTimes, loss: Float, ntokens: Int) {
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
            loss, ntokens
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

                GPU.startCapture(url: urls["backward"]!)
                eval(grad.flattened().map { $0.1 })
                GPU.stopCapture(url: urls["backward"]!)

                GPU.startCapture(url: urls["optimizer"]!)
                optimizer.update(model: model, gradients: grad)
                eval(model, optimizer)
                GPU.stopCapture(url: urls["optimizer"]!)

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
        ]
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
    /// Capped by `budgetBytes`: a symlink may point at something large (the
    /// mmap-backed model weights are the obvious candidate), and silently
    /// inflating a bundle to several GB on a phone is not acceptable. On
    /// exceeding the cap it stops and reports, leaving the remaining links in
    /// place — a partial flatten is visible in the record rather than hidden.
    private nonisolated static func flattenSymlinks(
        in bundle: URL, budgetBytes: Int
    ) -> [String: Any] {
        var resolved = 0
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
                if bytes + size > budgetBytes {
                    skippedOverBudget += 1
                    continue
                }
                try FileManager.default.removeItem(at: url)
                try FileManager.default.copyItem(at: targetURL, to: url)
                resolved += 1
                bytes += size
            } catch {
                failed += 1
            }
        }
        return [
            "flatten_resolved": resolved,
            "flatten_failed": failed,
            "flatten_skipped_over_budget": skippedOverBudget,
            "flatten_bytes": bytes,
            "flatten_examples": examples,
        ]
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
            "token_grid": TrainBenchConstants.peropTokenCounts,
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
            "app_build": TrainBenchConstants.peropAppBuild,
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
        tokPerSec: Double, loss: Float, phases: PerOpPhaseTimes?, elapsed: Double
    ) {
        var r = perOpBaseRecord(c, recordType: recordType)
        r["mode"] = mode
        r["pass"] = pass
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
        emitPerOp(r)
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
        emitPerOp(r)
    }

    private nonisolated static func appendPerOpMarker(
        _ c: PerOpRunContext, recordType: String, extra: [String: Any]
    ) {
        var r = perOpBaseRecord(c, recordType: recordType)
        for (k, v) in extra { r[k] = v }
        emitPerOp(r)
    }

    /// Append one h11 JSONL line. Its own file: the h9 L run is still pending
    /// against `train_bench_metrics_e2e.jsonl`, which this round must not
    /// touch.
    private nonisolated static func emitPerOp(_ record: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(
            TrainBenchConstants.peropMetricsFileName)
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
