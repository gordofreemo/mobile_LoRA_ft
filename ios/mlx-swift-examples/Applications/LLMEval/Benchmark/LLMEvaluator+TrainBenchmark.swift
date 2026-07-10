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
import MLXOptimizers

#if canImport(UIKit)
    import UIKit
#endif

/// Serializes concurrent appends to the E2E JSONL from the off-actor train
/// callback and the main-actor battery sampler. File scope so it is reachable
/// from `nonisolated` writers (a `@MainActor`-class static `let` would not be).
private let e2eFileLock = NSLock()

extension LLMEvaluator {

    // MARK: - Launch mode

    /// True when the app was launched to run the training benchmark.
    static var trainBenchmarkLaunchMode: TrainBenchLaunchMode? {
        let args = CommandLine.arguments
        // E2E (h5) is a distinct exact arg — check first.
        if args.contains("--benchmark-train-e2e") { return .e2e }
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
        // E2E (h5) is a separate orchestration path (real user, to completion,
        // save adapter, timed battery); the cap-sweep below is untouched.
        if mode == .e2e {
            await runE2ETrainBenchmark(
                user: Self.trainBenchmarkUser,
                condition: Self.trainBenchmarkCondition,
                maxIters: Self.trainBenchmarkMaxIters)
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
                // SmolLM3Model.useGradientCheckpoint.
                if TrainBenchConstants.gradientCheckpointing {
                    (ctx.model as? SmolLM3Model)?.useGradientCheckpoint = true
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
    private func trainCooldown() async {
        let start = Date.timeIntervalSinceReferenceDate
        while ProcessInfo.processInfo.thermalState != .nominal {
            if Date.timeIntervalSinceReferenceDate - start
                > TrainBenchConstants.cooldownCapSeconds
            {
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
            while !Task.isCancelled {
                let snap = Self.batterySnapshot()
                Self.appendE2EBattery(
                    ctx,
                    elapsed: Date.timeIntervalSinceReferenceDate - benchStart,
                    level: snap.level, charging: snap.charging,
                    thermal: Self.thermalString(),
                    lpm: ProcessInfo.processInfo.isLowPowerModeEnabled,
                    peak: Memory.snapshot().peakMemory)
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
                    (mc.model as? SmolLM3Model)?.useGradientCheckpoint = true
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
        thermal: String, lpm: Bool, peak: Int
    ) {
        var r = e2eBaseRecord(c, recordType: "battery")
        r["elapsed_s"] = elapsed
        r["battery_level"] = level
        r["charging"] = charging
        r["thermal_state"] = thermal
        r["low_power_mode"] = lpm
        r["peak_mem_bytes"] = peak
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
