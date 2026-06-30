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

extension LLMEvaluator {

    // MARK: - Launch mode

    /// True when the app was launched to run the training benchmark.
    static var trainBenchmarkLaunchMode: TrainBenchLaunchMode? {
        let args = CommandLine.arguments
        if args.contains("--benchmark-train-stress") { return .stress }
        if args.contains("--benchmark-train") { return .full }
        return nil
    }

    enum TrainBenchLaunchMode {
        /// Batch-size sweep: 200 steps at each of batchSizes, cooldown between.
        case full
        /// Sustained single run: 200 steps at batchSize=1 only, no interruptions.
        case stress
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
