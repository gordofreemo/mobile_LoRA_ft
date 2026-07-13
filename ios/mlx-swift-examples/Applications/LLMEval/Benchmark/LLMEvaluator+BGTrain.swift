// On-device background-scheduled (BGProcessingTask) LoRA-training — h6.
//
// Follow-on to the h5 E2E round (`LLMEvaluator+TrainBenchmark.swift`'s E2E
// section): same real top-100 LaMP-3 User-LoRA recipe, but orchestrated as a
// series of OS-scheduled background wakes instead of one long foreground
// session. `LoRATrain.train` is called in `TrainBenchConstants.bgChunkIterations`
// chunks so the app can checkpoint (LoRA weights + iteration counter) between
// chunks and resume across wakes, process relaunches, and (best-effort) hard
// kills. See experiments/2026-07-13-ondevice-bg-training-plan.md and the
// KNOWN, ACCEPTED DEVIATION note on `TrainBenchConstants.bgAppBuild`: Adam
// moments are NOT checkpointed and reset to zero every wake.
//
// Two entry points, both launch-arg driven (same "one Mac-driven devicectl
// launch, do one thing, exit or suspend" convention as every other harness
// mode in this file/target):
//   * `--bg-train-submit --user <fp> [--condition <label>]` — one-shot: writes
//     `bg_train_config.json`, submits a `BGProcessingTaskRequest`, exits.
//     Dispatched from ContentView's existing launch-mode `.task {}`.
//   * `handleBGTrainTask(_:)`, registered via `BGTaskScheduler.register` in
//     `LLMEvalApp.init()` — fires on every real OS wake, runs
//     `runBGTrainWake()` on a FRESH `LLMEvaluator` (background wakes have no
//     guaranteed view hierarchy to piggyback on). NOTE: SwiftUI's
//     `.backgroundTask` scene modifier only covers `.appRefresh`/`.urlSession`
//     — there is no `.processing` case, so `BGProcessingTaskRequest` needs
//     the traditional register/expirationHandler/setTaskCompleted API
//     (verified against the actual BackgroundTasks.framework header after an
//     initial wrong assumption in the design doc — see LLMEvalApp.swift).

import BackgroundTasks
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers

#if canImport(UIKit)
    import UIKit
#endif

/// Serializes concurrent appends to the BG JSONL. File scope (not an
/// extension member) so it's reachable from `nonisolated` writers, mirroring
/// the E2E file's `e2eFileLock`. Deliberately separate from that lock since
/// the two harnesses never run in the same process at the same time but
/// sharing a lock type across files would be a spurious coupling.
private let bgFileLock = NSLock()

extension LLMEvaluator {

    // MARK: - Launch args

    /// True when launched to SUBMIT a BGProcessingTaskRequest (one-shot,
    /// exits immediately after). The wake itself is handled separately by
    /// the `.backgroundTask` scene modifier, not a launch arg.
    static var bgTrainSubmitLaunchMode: Bool {
        CommandLine.arguments.contains("--bg-train-submit")
    }

    /// `--bg-train-validation` — stamped into every record this session so
    /// Xcode debug-forced wake cycles (`_simulateLaunchForTaskWithIdentifier:`)
    /// are excluded from the real 7-day-run analysis.
    private static var bgValidationFlag: Bool {
        CommandLine.arguments.contains("--bg-train-validation")
    }

    private static func launchArgValue(_ flag: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    static var bgTrainSubmitUser: String? { launchArgValue("--user") }
    static var bgTrainSubmitCondition: String { launchArgValue("--condition") ?? "bg_overnight" }

    // MARK: - Submission-time config (Documents/bg_train_config.json)

    private struct BGTrainConfig: Codable {
        let user: String
        let condition: String
        let nUser: Int
        let iterationsTotal: Int
        let submittedAtUtc: String
        let requiresExternalPowerConnected: Bool
        let requiresNetworkConnectivity: Bool
    }

    private nonisolated static var bgConfigURL: URL {
        URL.documentsDirectory.appendingPathComponent(TrainBenchConstants.bgConfigFileName)
    }

    /// Submit a `BGProcessingTaskRequest` for `user` (one real user, trained
    /// to completion under real OS scheduling). Writes the config the wake
    /// handler reads (the request itself carries no custom payload), then
    /// the caller (ContentView's launch-mode dispatch) exits.
    func submitBGTrainRequest(user: String?, condition: String) {
        guard let user, !user.isEmpty else {
            tlog("BG submit: missing --user <fingerprint>")
            benchLogLine("BG submit FAILED: missing --user <fingerprint>")
            return
        }
        guard let trainData = Self.loadBGUserData(user: user), !trainData.isEmpty else {
            tlog("BG submit FAILED to load user data for \(user)")
            benchLogLine("BG submit FAILED to load user data for \(user)")
            return
        }
        let nUser = trainData.count
        let iterationsTotal = TrainBenchConstants.e2eEpochs * nUser

        let config = BGTrainConfig(
            user: user, condition: condition, nUser: nUser,
            iterationsTotal: iterationsTotal,
            submittedAtUtc: ISO8601DateFormatter().string(from: Date()),
            requiresExternalPowerConnected: true,
            requiresNetworkConnectivity: false)
        guard let data = try? JSONEncoder().encode(config) else {
            tlog("BG submit FAILED to encode config")
            benchLogLine("BG submit FAILED to encode config")
            return
        }
        do {
            try data.write(to: Self.bgConfigURL, options: .atomic)
        } catch {
            tlog("BG submit FAILED to write config: \(error)")
            benchLogLine("BG submit FAILED to write config: \(error)")
            return
        }

        // Fresh checkpoint dir for a fresh submission (a re-submit for the
        // same user starts the adapter over rather than silently resuming
        // stale weights from an earlier, unrelated submission).
        let ckptDir = Self.bgCheckpointDir(user: user)
        try? FileManager.default.removeItem(at: ckptDir)
        try? FileManager.default.createDirectory(
            at: ckptDir, withIntermediateDirectories: true)

        let request = BGProcessingTaskRequest(identifier: TrainBenchConstants.bgTaskIdentifier)
        request.requiresExternalPower = true
        request.requiresNetworkConnectivity = false
        do {
            try BGTaskScheduler.shared.submit(request)
            tlog("BG submit OK user=\(user) nUser=\(nUser) iterationsTotal=\(iterationsTotal) "
                + "condition=\(condition)")
            benchLogLine("BG submit OK user=\(user) iterationsTotal=\(iterationsTotal)")
        } catch {
            tlog("BG submit FAILED to schedule request: \(error)")
            benchLogLine("BG submit FAILED to schedule request: \(error)")
        }
    }

    // MARK: - Checkpoint (weights + iteration counter — NOT optimizer state)

    private struct BGCheckpointMeta: Codable {
        var iterationsCompleted: Int
        var wakeNumber: Int
        var cumulativeDeviceElapsedS: Double
        var updatedAtUtc: String
    }

    private nonisolated static func bgCheckpointDir(user: String) -> URL {
        URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.bgCheckpointDirName)
            .appendingPathComponent(user)
    }

    private nonisolated static func bgWeightsURL(user: String) -> URL {
        bgCheckpointDir(user: user).appendingPathComponent("weights.safetensors")
    }

    private nonisolated static func bgCheckpointMetaURL(user: String) -> URL {
        bgCheckpointDir(user: user).appendingPathComponent("checkpoint_meta.json")
    }

    private nonisolated static func loadBGCheckpointMeta(user: String) -> BGCheckpointMeta? {
        guard let data = try? Data(contentsOf: bgCheckpointMetaURL(user: user)) else { return nil }
        return try? JSONDecoder().decode(BGCheckpointMeta.self, from: data)
    }

    /// Write updated iteration counter/wake number/cumulative device time.
    /// LoRA weights themselves are written separately via `saveLoRAWeights`
    /// (reused from `LoraTrain.swift` — same file IS both the running
    /// checkpoint and, at completion, the final saved adapter).
    private nonisolated static func writeBGCheckpointMeta(_ meta: BGCheckpointMeta, user: String) {
        guard let data = try? JSONEncoder().encode(meta) else { return }
        try? data.write(to: bgCheckpointMetaURL(user: user), options: .atomic)
    }

    /// `LoraTrain.swift` only exposes `saveLoRAWeights` — no load counterpart
    /// upstream (a stale doc-comment references one that was never added).
    /// Same primitive (`loadArrays` / `Module.update(parameters:)`), just the
    /// missing other half.
    private nonisolated static func loadLoRAWeights(model: Module, url: URL) throws {
        let arrays = try loadArrays(url: url)
        _ = model.update(parameters: ModuleParameters.unflattened(arrays))
    }

    // MARK: - Per-run context (immutable, built once per wake on the main actor)

    private struct BGRunContext: Sendable {
        let user: String
        let condition: String
        let nUser: Int
        let iterationsTotal: Int
        let seqCap: Int
        let modelName: String
        let sessionId: String
        let wakeNumber: Int
        let validation: Bool
    }

    /// Returned by the `container.perform` closure in `runBGTrainWake()` —
    /// crosses the isolation boundary back to the caller instead of the
    /// closure mutating captured outer vars directly (see the capture note
    /// at the call site).
    private struct BGWakeResult: Sendable {
        let iterationsCompleted: Int
        let cumulativeDeviceElapsedS: Double
        let terminationReason: String
    }

    // MARK: - BGProcessingTask lifecycle (registered in LLMEvalApp.init())

    /// Entry point handed to `BGTaskScheduler.register(...)`. `runBGTrainWake()`
    /// (below) already checks `Task.isCancelled` between chunks, so expiration
    /// is wired the standard way: run the wake as a child `Task`, cancel it
    /// from `expirationHandler`, and call `setTaskCompleted` once it (or the
    /// cancellation) finishes. `success: true` regardless of how much of the
    /// adapter got trained — it reflects "did we exit cleanly", not "did we
    /// finish all iterations" (the JSONL's `wake_termination_reason` carries
    /// the latter).
    static func handleBGTrainTask(_ task: BGProcessingTask) {
        let work = Task {
            await LLMEvaluator().runBGTrainWake()
        }
        task.expirationHandler = {
            work.cancel()
        }
        Task {
            _ = await work.value
            task.setTaskCompleted(success: true)
        }
    }

    // MARK: - Wake entry point

    /// Called from `handleBGTrainTask` on a fresh `LLMEvaluator` for every
    /// real OS wake (and, during validation, every Xcode debug-forced wake).
    /// Re-submits the next request FIRST so the chain survives even if this
    /// wake errors, then resumes training from the last checkpoint (weights
    /// + iteration counter; optimizer state is NOT resumed — fresh `AdamW`
    /// every wake, see `TrainBenchConstants.bgAppBuild`'s doc comment).
    func runBGTrainWake() async {
        enableThinking = false
        #if canImport(UIKit)
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        // Re-arm the next wake before doing any work — a crash/expiration
        // mid-chunk must not silently end the whole multi-day chain.
        let nextRequest = BGProcessingTaskRequest(identifier: TrainBenchConstants.bgTaskIdentifier)
        nextRequest.requiresExternalPower = true
        nextRequest.requiresNetworkConnectivity = false
        try? BGTaskScheduler.shared.submit(nextRequest)

        guard let configData = try? Data(contentsOf: Self.bgConfigURL),
            let config = try? JSONDecoder().decode(BGTrainConfig.self, from: configData)
        else {
            tlog("BG wake: no bg_train_config.json — nothing submitted yet")
            benchLogLine("BG wake: no config, skipping")
            return
        }

        guard let trainData = Self.loadBGUserData(user: config.user), !trainData.isEmpty else {
            tlog("BG wake FAILED to load user data for \(config.user)")
            benchLogLine("BG wake FAILED to load user data for \(config.user)")
            return
        }

        let priorMeta = Self.loadBGCheckpointMeta(user: config.user)
        let wakeNumber = (priorMeta?.wakeNumber ?? -1) + 1
        var iterationsCompleted = priorMeta?.iterationsCompleted ?? 0
        var cumulativeDeviceElapsedS = priorMeta?.cumulativeDeviceElapsedS ?? 0

        if iterationsCompleted >= config.iterationsTotal {
            tlog("BG wake: user=\(config.user) already complete "
                + "(\(iterationsCompleted)/\(config.iterationsTotal))")
            benchLogLine("BG wake: already complete, nothing to do")
            return
        }

        let sessionId = UUID().uuidString
        let modelName = modelConfiguration.name.components(separatedBy: "/").last
            ?? modelConfiguration.name
        let ctx = BGRunContext(
            user: config.user, condition: config.condition, nUser: config.nUser,
            iterationsTotal: config.iterationsTotal, seqCap: TrainBenchConstants.e2eSeqCap,
            modelName: modelName, sessionId: sessionId, wakeNumber: wakeNumber,
            validation: Self.bgValidationFlag)

        tlog("BG wake start user=\(config.user) wake=\(wakeNumber) "
            + "resumeAt=\(iterationsCompleted)/\(config.iterationsTotal) build=\(TrainBenchConstants.bgAppBuild)")
        benchLogLine("BG wake start user=\(config.user) wake=\(wakeNumber) resumeAt=\(iterationsCompleted)")

        let calendarOrigin = Self.parseISO8601(config.submittedAtUtc) ?? Date()
        let wakeStartDate = Date()
        let wakeStart = Date.timeIntervalSinceReferenceDate
        let batteryAtWakeStart = Self.bgBatterySnapshot()
        let thermalAtWakeStart = Self.thermalString()

        Self.appendBGMarker(
            ctx, recordType: "wake_start",
            extra: [
                "battery_level": batteryAtWakeStart.level,
                "charging": batteryAtWakeStart.charging,
                "thermal_state": thermalAtWakeStart,
                "requires_external_power": true,
                "calendar_elapsed_s": wakeStartDate.timeIntervalSince(calendarOrigin),
                "cumulative_device_elapsed_s": cumulativeDeviceElapsedS,
                "iterations_completed_total": iterationsCompleted,
            ])

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("BG wake model load failed: \(error)")
            benchLogLine("BG wake model load failed: \(error)")
            Self.appendBGMarker(
                ctx, recordType: "error",
                extra: ["error": "model load failed: \(error)", "wake_termination_reason": "error"])
            return
        }

        var terminationReason = "chunk_loop_finished_early"
        var wakeError: String? = nil
        // Snapshotted for capture into the `perform` closure below — that
        // closure owns its OWN local copies and returns them via
        // `BGWakeResult` rather than mutating these captured vars directly
        // (SWIFT_STRICT_CONCURRENCY = complete on this target forbids
        // mutating a captured outer `var` from inside a `@Sendable` closure).
        let initialIterationsCompleted = iterationsCompleted
        let initialCumulativeDeviceElapsedS = cumulativeDeviceElapsedS

        do {
            let result = try await container.perform { mc throws -> BGWakeResult in
                let config2 = LoRAConfiguration(
                    numLayers: TrainBenchConstants.loraLayers,
                    loraParameters: .init(
                        rank: TrainBenchConstants.loraRank,
                        scale: TrainBenchConstants.loraScale,
                        keys: TrainBenchConstants.loraKeys))
                _ = try LoRAContainer.from(model: mc.model, configuration: config2)

                if TrainBenchConstants.gradientCheckpointing {
                    (mc.model as? SmolLM3Model)?.useGradientCheckpoint = true
                }

                let weightsURL = Self.bgWeightsURL(user: ctx.user)
                if initialIterationsCompleted > 0,
                    FileManager.default.fileExists(atPath: weightsURL.path)
                {
                    try Self.loadLoRAWeights(model: mc.model, url: weightsURL)
                    self.tlog("BG wake: resumed weights from \(weightsURL.lastPathComponent)")
                }

                let capTrain = Self.capExamples(trainData, cap: ctx.seqCap, tokenizer: mc.tokenizer)
                let capValid = Array(capTrain.prefix(1))

                // Fresh optimizer every wake — see the accepted deviation
                // note on TrainBenchConstants.bgAppBuild. Moments start at
                // zero; only the model weights and iteration counter resume.
                let optimizer = AdamW(
                    learningRate: TrainBenchConstants.e2eLearningRate,
                    weightDecay: TrainBenchConstants.e2eWeightDecay,
                    biasCorrection: TrainBenchConstants.e2eAdamBiasCorrection)

                var localIterationsCompleted = initialIterationsCompleted
                var localCumulativeDeviceElapsedS = initialCumulativeDeviceElapsedS
                var localTerminationReason = "chunk_loop_finished_early"

                while localIterationsCompleted < ctx.iterationsTotal {
                    if Task.isCancelled {
                        localTerminationReason = "expiration_handler"
                        break
                    }
                    let remaining = ctx.iterationsTotal - localIterationsCompleted
                    let chunk = min(TrainBenchConstants.bgChunkIterations, remaining)
                    let iterationsCompletedBeforeChunk = localIterationsCompleted

                    let params = LoRATrain.Parameters(
                        batchSize: TrainBenchConstants.e2eBatchSize,
                        iterations: chunk,
                        stepsPerReport: chunk,
                        stepsPerEval: chunk + 1,
                        validationBatches: 0,
                        saveEvery: chunk + 1,
                        adapterURL: nil)

                    GPU.resetPeakMemory()
                    let chunkStart = Date.timeIntervalSinceReferenceDate
                    let cumulativeBeforeChunk = localCumulativeDeviceElapsedS
                    try LoRATrain.train(
                        model: mc.model, train: capTrain, validate: capValid,
                        optimizer: optimizer, tokenizer: mc.tokenizer, parameters: params
                    ) { progress in
                        switch progress {
                        case .train(let iter, let loss, let ips, let tps):
                            let now = Date.timeIntervalSinceReferenceDate
                            Self.appendBGTrainWindow(
                                ctx, step: iterationsCompletedBeforeChunk + iter + 1, loss: loss,
                                ips: ips, tps: tps, wakeElapsed: now - wakeStart,
                                cumulativeDeviceElapsed: cumulativeBeforeChunk + (now - chunkStart),
                                calendarElapsed: Date().timeIntervalSince(calendarOrigin),
                                checkpointWriteMs: 0, peak: Memory.snapshot().peakMemory,
                                thermal: Self.thermalString(),
                                lpm: ProcessInfo.processInfo.isLowPowerModeEnabled)
                        case .validation, .save:
                            break
                        }
                        return .more
                    }
                    GPU.resetPeakMemory()
                    let chunkElapsed = Date.timeIntervalSinceReferenceDate - chunkStart
                    localCumulativeDeviceElapsedS += chunkElapsed
                    localIterationsCompleted += chunk

                    let ckptStart = Date.timeIntervalSinceReferenceDate
                    try? FileManager.default.createDirectory(
                        at: Self.bgCheckpointDir(user: ctx.user), withIntermediateDirectories: true)
                    try LoRATrain.saveLoRAWeights(model: mc.model, url: weightsURL)
                    let checkpointWriteMs =
                        (Date.timeIntervalSinceReferenceDate - ckptStart) * 1000
                    Self.writeBGCheckpointMeta(
                        BGCheckpointMeta(
                            iterationsCompleted: localIterationsCompleted, wakeNumber: ctx.wakeNumber,
                            cumulativeDeviceElapsedS: localCumulativeDeviceElapsedS,
                            updatedAtUtc: ISO8601DateFormatter().string(from: Date())),
                        user: ctx.user)

                    Self.appendBGMarker(
                        ctx, recordType: "checkpoint",
                        extra: [
                            "step": localIterationsCompleted,
                            "checkpoint_write_ms": checkpointWriteMs,
                            "cumulative_device_elapsed_s": localCumulativeDeviceElapsedS,
                            "calendar_elapsed_s": Date().timeIntervalSince(calendarOrigin),
                        ])
                    // Milestone trace (visible live in Xcode's console during
                    // debug-forced validation, unlike the JSONL/meta writes
                    // above) — one line per chunk so progress is observable
                    // without pulling files mid-session.
                    self.tlog(
                        "BG wake: checkpoint iter=\(localIterationsCompleted)/\(ctx.iterationsTotal) "
                            + "chunkElapsed=\(String(format: "%.1f", chunkElapsed))s "
                            + "ckptWriteMs=\(String(format: "%.0f", checkpointWriteMs))")

                    // Lightweight partial upsert into bg_run_meta.json after
                    // EVERY chunk (not just at wake end) — see the doc
                    // comment on `upsertBGRunMetaWake` for why: a hard kill
                    // mid-wake must still leave this wake in the file.
                    // `charging: true` is hardcoded rather than sampled (no
                    // MainActor hop available off-actor here) — matches
                    // requiresExternalPower=true, always true for this task.
                    Self.upsertBGRunMetaWake(
                        wakeNumber: ctx.wakeNumber, wakeStart: wakeStartDate, wakeEnd: Date(),
                        iterationsThisWake: localIterationsCompleted - initialIterationsCompleted,
                        iterationsCompletedTotal: localIterationsCompleted,
                        iterationsTotal: ctx.iterationsTotal, charging: true,
                        thermalAtStart: thermalAtWakeStart, thermalAtEnd: Self.thermalString(),
                        completed: localIterationsCompleted >= ctx.iterationsTotal,
                        calendarOrigin: calendarOrigin,
                        cumulativeDeviceElapsedS: localCumulativeDeviceElapsedS)

                    if localIterationsCompleted >= ctx.iterationsTotal {
                        localTerminationReason = "chunk_loop_finished_early"
                        break
                    }
                }

                return BGWakeResult(
                    iterationsCompleted: localIterationsCompleted,
                    cumulativeDeviceElapsedS: localCumulativeDeviceElapsedS,
                    terminationReason: localTerminationReason)
            }
            iterationsCompleted = result.iterationsCompleted
            cumulativeDeviceElapsedS = result.cumulativeDeviceElapsedS
            terminationReason = result.terminationReason
        } catch {
            wakeError = "\(error)"
            terminationReason = "error"
            tlog("BG wake training error: \(error)")
            benchLogLine("BG wake training error: \(error)")
        }

        let batteryAtWakeEnd = Self.bgBatterySnapshot()
        let wakeEndDate = Date()
        Self.appendBGMarker(
            ctx, recordType: wakeError == nil ? "wake_end" : "error",
            extra: [
                "battery_level": batteryAtWakeEnd.level,
                "charging": batteryAtWakeEnd.charging,
                "thermal_state": Self.thermalString(),
                "wake_elapsed_s": Date.timeIntervalSinceReferenceDate - wakeStart,
                "cumulative_device_elapsed_s": cumulativeDeviceElapsedS,
                "calendar_elapsed_s": wakeEndDate.timeIntervalSince(calendarOrigin),
                "iterations_completed_total": iterationsCompleted,
                "wake_termination_reason": terminationReason,
                "error": wakeError ?? NSNull(),
            ])

        Self.upsertBGRunMetaWake(
            wakeNumber: wakeNumber, wakeStart: wakeStartDate, wakeEnd: wakeEndDate,
            iterationsThisWake: iterationsCompleted - (priorMeta?.iterationsCompleted ?? 0),
            iterationsCompletedTotal: iterationsCompleted, iterationsTotal: config.iterationsTotal,
            charging: batteryAtWakeEnd.charging, thermalAtStart: thermalAtWakeStart,
            thermalAtEnd: Self.thermalString(),
            completed: iterationsCompleted >= config.iterationsTotal,
            calendarOrigin: calendarOrigin, cumulativeDeviceElapsedS: cumulativeDeviceElapsedS)

        tlog("BG wake end user=\(config.user) wake=\(wakeNumber) "
            + "completed=\(iterationsCompleted)/\(config.iterationsTotal) reason=\(terminationReason)")
        benchLogLine("BG wake end iterationsCompleted=\(iterationsCompleted) reason=\(terminationReason)")
    }

    // MARK: - User data (side-loaded, same layout as h5 E2E)

    private nonisolated static func loadBGUserData(user: String) -> [String]? {
        let url = URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.e2eDataDirName)
            .appendingPathComponent("lamp3_\(user).jsonl")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try? MLXLLM.loadLoRAData(url: url)
    }

    /// Truncate to `cap` tokens (local copy of the E2E file's helper —
    /// `private` there, not reachable across files; small enough to
    /// duplicate rather than couple the two harnesses together).
    private nonisolated static func capExamples(
        _ data: [String], cap: Int, tokenizer: Tokenizer
    ) -> [String] {
        data.map { s in
            let toks = tokenizer.encode(text: s)
            guard toks.count > cap else { return s }
            return tokenizer.decode(tokenIds: Array(toks.prefix(cap)))
        }
    }

    // MARK: - JSONL record builders (h6 schema)

    private nonisolated static func bgBaseRecord(_ c: BGRunContext, recordType: String) -> [String: Any] {
        [
            "record_type": recordType,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "user_fingerprint": c.user,
            "condition": c.condition,
            "n_user": c.nUser,
            "iterations_total": c.iterationsTotal,
            "seq_cap": c.seqCap,
            "batch_size": TrainBenchConstants.e2eBatchSize,
            "epochs": TrainBenchConstants.e2eEpochs,
            "model": c.modelName,
            "lora_rank": TrainBenchConstants.loraRank,
            "lora_keys": TrainBenchConstants.loraKeysLabel,
            "num_lora_layers": TrainBenchConstants.loraLayers,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "optimizer": "adamw",
            "learning_rate": TrainBenchConstants.e2eLearningRate,
            "weight_decay": TrainBenchConstants.e2eWeightDecay,
            "adam_bias_correction": TrainBenchConstants.e2eAdamBiasCorrection,
            "adam_moments_reset_per_wake": true,
            "requires_external_power_connected": true,
            "wake_number": c.wakeNumber,
            "validation": c.validation,
            "app_build": TrainBenchConstants.bgAppBuild,
            "bench_schema_version": TrainBenchConstants.bgSchemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": c.sessionId,
            "device_model": bgHwModel(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    private nonisolated static func appendBGTrainWindow(
        _ c: BGRunContext, step: Int, loss: Float, ips: Double, tps: Double,
        wakeElapsed: Double, cumulativeDeviceElapsed: Double, calendarElapsed: Double,
        checkpointWriteMs: Double, peak: Int, thermal: String, lpm: Bool
    ) {
        var r = bgBaseRecord(c, recordType: "train")
        r["step"] = step
        r["training_loss"] = Double(loss)
        r["iter_per_sec"] = ips
        r["tok_per_sec"] = tps
        r["wake_elapsed_s"] = wakeElapsed
        r["cumulative_device_elapsed_s"] = cumulativeDeviceElapsed
        r["calendar_elapsed_s"] = calendarElapsed
        r["checkpoint_write_ms"] = checkpointWriteMs
        r["peak_mem_bytes"] = peak
        r["thermal_state"] = thermal
        r["low_power_mode"] = lpm
        emitBG(r)
    }

    private nonisolated static func appendBGMarker(
        _ c: BGRunContext, recordType: String, extra: [String: Any]
    ) {
        var r = bgBaseRecord(c, recordType: recordType)
        for (k, v) in extra { r[k] = v }
        emitBG(r)
    }

    private nonisolated static func emitBG(_ record: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        let line = json + "\n"
        let url = URL.documentsDirectory.appendingPathComponent(TrainBenchConstants.bgMetricsFileName)
        bgFileLock.lock()
        defer { bgFileLock.unlock() }
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
            // best-effort; a failed line must not crash a multi-day run
        }
    }

    // MARK: - bg_run_meta.json (per-wake summaries + rolling run summary)

    private struct BGWakeSummary: Codable {
        let wakeNumber: Int
        let wakeStartUtc: String
        let wakeEndUtc: String
        let gapSincePreviousWakeS: Double?
        let iterationsThisWake: Int
        let iterationsCompletedTotal: Int
        let iterationsTotal: Int
        let charging: Bool
        let thermalStateAtWakeStart: String
        let thermalStateAtWakeEnd: String
    }

    private struct BGRunSummary: Codable {
        let totalCalendarDays: Double
        let totalWakes: Int
        let totalDeviceComputeS: Double
        let meanIterSBackgrounded: Double?
        let meanGapBetweenWakesS: Double?
        let completed: Bool
        let iterationsCompletedFinal: Int
        let capHit: Bool
    }

    private struct BGRunMetaFile: Codable {
        var wakes: [BGWakeSummary]
        var summary: BGRunSummary?
    }

    private nonisolated static var bgRunMetaURL: URL {
        URL.documentsDirectory.appendingPathComponent(TrainBenchConstants.bgRunMetaFileName)
    }

    /// Read-modify-write: UPSERT this wake's summary by `wakeNumber` (replace
    /// if an entry for this wake already exists, else append + keep sorted),
    /// recompute the rolling run-level summary. Upsert rather than blind
    /// append for two reasons, both confirmed by the first validation cycle:
    /// (1) called from TWO places per wake now (see below) so the same wake
    /// must not produce duplicate entries; (2) a debug-forced wake that's
    /// re-triggered via `_simulateLaunchForTaskWithIdentifier:` before the
    /// prior invocation finished (observed during validation — re-issuing
    /// the LLDB command mid-wake) would otherwise also duplicate.
    ///
    /// Called from TWO places in `runBGTrainWake()`:
    ///   - a lightweight partial upsert after EVERY chunk checkpoint (from
    ///     inside the `container.perform` closure, off-actor — so `charging`
    ///     is hardcoded `true` rather than sampled, matching the task's own
    ///     `requiresExternalPower` requirement, and thermal uses the
    ///     `nonisolated` `thermalString()` reader) — this is the fix for a
    ///     real gap found in validation: a hard kill mid-wake (Xcode Stop,
    ///     or a real SIGKILL) previously left NO trace of that wake in this
    ///     file at all, only in the per-chunk JSONL + `checkpoint_meta.json`,
    ///     undercounting wakes in the co-headline #2 wake-scheduling
    ///     analysis.
    ///   - the authoritative full upsert once the wake actually finishes
    ///     (graceful or error), with real end-of-wake battery/thermal —
    ///     overwrites the last partial entry with accurate final values.
    private nonisolated static func upsertBGRunMetaWake(
        wakeNumber: Int, wakeStart: Date, wakeEnd: Date, iterationsThisWake: Int,
        iterationsCompletedTotal: Int, iterationsTotal: Int, charging: Bool,
        thermalAtStart: String, thermalAtEnd: String, completed: Bool,
        calendarOrigin: Date, cumulativeDeviceElapsedS: Double
    ) {
        var file =
            (try? Data(contentsOf: bgRunMetaURL)).flatMap {
                try? JSONDecoder().decode(BGRunMetaFile.self, from: $0)
            } ?? BGRunMetaFile(wakes: [], summary: nil)

        // Gap is measured against the previous WAKE NUMBER's end, not simply
        // "the last array entry" — safe under upserts/out-of-order writes.
        let previousWakeEnd = file.wakes
            .filter { $0.wakeNumber < wakeNumber }
            .max(by: { $0.wakeNumber < $1.wakeNumber })
            .flatMap { parseISO8601($0.wakeEndUtc) }
        let gap = previousWakeEnd.map { wakeStart.timeIntervalSince($0) }

        let entry = BGWakeSummary(
            wakeNumber: wakeNumber,
            wakeStartUtc: ISO8601DateFormatter().string(from: wakeStart),
            wakeEndUtc: ISO8601DateFormatter().string(from: wakeEnd),
            gapSincePreviousWakeS: gap, iterationsThisWake: iterationsThisWake,
            iterationsCompletedTotal: iterationsCompletedTotal,
            iterationsTotal: iterationsTotal, charging: charging,
            thermalStateAtWakeStart: thermalAtStart, thermalStateAtWakeEnd: thermalAtEnd)

        if let idx = file.wakes.firstIndex(where: { $0.wakeNumber == wakeNumber }) {
            file.wakes[idx] = entry
        } else {
            file.wakes.append(entry)
            file.wakes.sort { $0.wakeNumber < $1.wakeNumber }
        }

        let gaps = file.wakes.compactMap { $0.gapSincePreviousWakeS }
        let totalCalendarDays = wakeEnd.timeIntervalSince(calendarOrigin) / 86400
        file.summary = BGRunSummary(
            totalCalendarDays: totalCalendarDays,
            totalWakes: file.wakes.count,
            totalDeviceComputeS: cumulativeDeviceElapsedS,
            meanIterSBackgrounded: cumulativeDeviceElapsedS > 0
                ? Double(iterationsCompletedTotal) / cumulativeDeviceElapsedS : nil,
            meanGapBetweenWakesS: gaps.isEmpty ? nil : gaps.reduce(0, +) / Double(gaps.count),
            completed: completed,
            iterationsCompletedFinal: iterationsCompletedTotal,
            capHit: totalCalendarDays >= 7.0 && !completed)

        guard let data = try? JSONEncoder().encode(file) else { return }
        try? data.write(to: bgRunMetaURL, options: .atomic)
    }

    // MARK: - Small local helpers (duplicated from the E2E file on purpose —
    // that file's are `private`, not reachable across files; matches this
    // codebase's existing precedent of small per-harness-file duplicates,
    // e.g. `trainHwModel()`'s own doc comment in the E2E file).

    private nonisolated static func parseISO8601(_ s: String) -> Date? {
        ISO8601DateFormatter().date(from: s)
    }

    private nonisolated static func thermalString() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private nonisolated static func bgHwModel() -> String {
        var sysinfo = utsname()
        uname(&sysinfo)
        let mirror = Mirror(reflecting: sysinfo.machine)
        let id = mirror.children.compactMap { ($0.value as? Int8) }
            .filter { $0 != 0 }.map { String(UnicodeScalar(UInt8($0))) }.joined()
        return id.isEmpty ? "unknown" : id
    }

    private struct BGBatterySnapshot {
        let level: Double
        let charging: Bool
    }

    private static func bgBatterySnapshot() -> BGBatterySnapshot {
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
        return BGBatterySnapshot(level: level, charging: charging)
    }
}
