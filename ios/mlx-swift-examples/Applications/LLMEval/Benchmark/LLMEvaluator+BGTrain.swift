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
    /// `handleBGTrainTask(_:)`, registered via `BGTaskScheduler.register` in
    /// `LLMEvalApp.init()` — not a launch arg.
    static var bgTrainSubmitLaunchMode: Bool {
        CommandLine.arguments.contains("--bg-train-submit")
    }

    /// True when launched to CANCEL all pending BGProcessingTaskRequests for
    /// this identifier (one-shot, exits immediately after). Does NOT touch
    /// any on-disk data (config/checkpoints/JSONL) — pair with a manual wipe
    /// or a fresh `--bg-train-submit` (which wipes the checkpoint dir itself)
    /// if a clean restart is the goal.
    static var bgTrainCancelLaunchMode: Bool {
        CommandLine.arguments.contains("--bg-train-cancel")
    }

    /// True when launched to submit a fresh `BGProcessingTaskRequest`
    /// WITHOUT touching `bg_train_config.json` or the checkpoint dir (unlike
    /// `--bg-train-submit`, which wipes both for a clean-slate start) and
    /// without resetting the original submission's calendar-cap origin
    /// timestamp. For re-arming the chain after a code update mid-run,
    /// where the goal is to keep existing progress and the original 7-day
    /// clock, not restart either.
    static var bgTrainResubmitLaunchMode: Bool {
        CommandLine.arguments.contains("--bg-train-resubmit")
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

    /// Cancel any pending `BGProcessingTaskRequest` for this identifier —
    /// stops the self-perpetuating wake chain (`runBGTrainWake()` re-arms the
    /// next request as its first action, so simply wiping on-disk data does
    /// NOT stop future wakes from firing; this does). Does not touch
    /// on-disk state.
    func cancelBGTrainRequest() {
        BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: TrainBenchConstants.bgTaskIdentifier)
        tlog("BG cancel: cancelled pending request for \(TrainBenchConstants.bgTaskIdentifier)")
        benchLogLine("BG cancel: cancelled pending request")
    }

    /// Submit a fresh `BGProcessingTaskRequest` — same request shape as
    /// `submitBGTrainRequest`/the in-wake re-arm — but WITHOUT touching
    /// `bg_train_config.json` or the checkpoint dir. For re-arming after a
    /// code update mid-run (e.g. cancel, reinstall, resubmit) where existing
    /// progress and the original submission's calendar-cap origin should
    /// both be preserved, unlike `--bg-train-submit` which wipes both for a
    /// deliberate clean-slate start.
    func resubmitBGTrainRequest() {
        let request = BGProcessingTaskRequest(identifier: TrainBenchConstants.bgTaskIdentifier)
        request.requiresExternalPower = true
        request.requiresNetworkConnectivity = false
        do {
            try BGTaskScheduler.shared.submit(request)
            tlog("BG resubmit OK (config/checkpoint untouched)")
            benchLogLine("BG resubmit OK")
        } catch {
            tlog("BG resubmit FAILED: \(error)")
            benchLogLine("BG resubmit FAILED: \(error)")
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

    /// Overwritten (not appended) ~1x/sec by the heartbeat `Task` in
    /// `runBGTrainWake()` — see `TrainBenchConstants.bgHeartbeatFileName`'s
    /// doc comment for why this exists (pinpointing actual OS-granted
    /// time-slice length, independent of which milestone marker last fired).
    private nonisolated static func writeBGHeartbeat(
        wakeNumber: Int, wakeElapsedS: Double, peakMemBytes: Int, activeMemBytes: Int
    ) {
        let payload: [String: Any] = [
            "wake_number": wakeNumber,
            "wake_elapsed_s": wakeElapsedS,
            "peak_mem_bytes": peakMemBytes,
            "active_mem_bytes": activeMemBytes,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return }
        let url = URL.documentsDirectory.appendingPathComponent(TrainBenchConstants.bgHeartbeatFileName)
        try? data.write(to: url, options: .atomic)
    }

    /// `LoraTrain.swift` only exposes `saveLoRAWeights` — no load counterpart
    /// upstream (a stale doc-comment references one that was never added).
    /// Same primitive (`loadArrays` / `Module.update(parameters:)`), just the
    /// missing other half.
    ///
    /// Uses the THROWING `update(parameters:verify:)` overload with
    /// `.shapeMismatch` rather than the convenience `update(parameters:)`
    /// wrapper (which hardcodes `verify: .none` and `try!`) — found while
    /// investigating a real-run wake that died silently right around this
    /// call: with `.none`, a shape mismatch wouldn't throw at all, it would
    /// silently assign the wrong-shaped array into the model's parameter
    /// storage, and the actual crash would surface later as an uncatchable
    /// native MLX abort the next time that parameter hits a matmul — which
    /// looks EXACTLY like what was observed (silent death, zero trace). Not
    /// confident this is the actual root cause (the same code path worked in
    /// Xcode-debug validation, and LoRA shapes should be deterministic given
    /// identical config both times) — but converting a silent-corruption
    /// failure mode into a catchable, loggable one is strictly safer either
    /// way, and costs nothing on the success path.
    private nonisolated static func loadLoRAWeights(model: Module, url: URL) throws {
        let arrays = try loadArrays(url: url)
        _ = try model.update(parameters: ModuleParameters.unflattened(arrays), verify: .shapeMismatch)
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

        // Diagnostic instrumentation (schema v6, added after the BGProbe
        // control experiment showed a trivial app gets 240s+ at the SAME
        // wake instants LLMEval dies at ~9.5-10s — ruling out a platform/OS
        // ceiling and pointing at LLMEval's own resource footprint, most
        // likely memory pressure from loading the ~1.7GB 4-bit model, as
        // the proximate cause. Diagnostic-only: no behavior change here,
        // just finer visibility into what that footprint looks like right
        // up to the moment of death.
        //
        // One clean per-wake baseline so every peak_mem_bytes reading below
        // (heartbeat ticks + the model_loaded/lora_apply_start/
        // lora_apply_complete markers) is comparable and monotonic within
        // this wake, not carrying over noise from a previous wake's process
        // if the OS happened to keep it warm.
        GPU.resetPeakMemory()

        // Heartbeat: overwrites a small file ~5x/sec (bumped from ~1x/sec —
        // prior investigation showed death lands within ~0.1-0.5s of
        // `lora_apply_start`, too fast for 1s resolution to localize
        // further), independent of which milestone marker last fired.
        // Now also carries peak/active memory so the LAST tick before a
        // silent death gives a real memory-footprint reading at
        // approximately the moment of death, not just an elapsed time.
        // `Task { ... }` is NOT a structured child of this function (no
        // automatic cancellation on return), so it's cancelled explicitly
        // via `defer` — covers every exit path below (model-load failure,
        // closure success, closure error) without duplicating the cancel
        // call at each one.
        let heartbeatWakeNumber = wakeNumber
        let heartbeatTask = Task {
            while !Task.isCancelled {
                let snap = Memory.snapshot()
                Self.writeBGHeartbeat(
                    wakeNumber: heartbeatWakeNumber,
                    wakeElapsedS: Date.timeIntervalSinceReferenceDate - wakeStart,
                    peakMemBytes: snap.peakMemory, activeMemBytes: snap.activeMemory)
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        defer { heartbeatTask.cancel() }

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
        // Diagnostic marker (persistent JSONL, NOT tlog — during a real
        // unattended wake there is no attached console to read tlog's
        // stderr from; only what's written to disk survives to be pulled
        // later). Pinpoints whether a dead wake ever got past model load.
        // `peak_mem_bytes` (schema v6) — footprint right after loading the
        // ~1.7GB 4-bit model, the leading suspect for why LLMEval dies at
        // the same wake instants a near-zero-footprint control app (see
        // ios/BGProbe/) gets 240s+.
        Self.appendBGMarker(
            ctx, recordType: "model_loaded",
            extra: [
                "wake_elapsed_s": Date.timeIntervalSinceReferenceDate - wakeStart,
                "peak_mem_bytes": Memory.snapshot().peakMemory,
                "active_mem_bytes": Memory.snapshot().activeMemory,
            ])

        var terminationReason = "voluntary_yield"
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
                // Every real wake since the first has died silently between
                // `model_loaded` and the next marker — never via a caught
                // `Task.isCancelled` + clean return, always via what looks
                // like a hard kill mid-synchronous-call. Two independent
                // checkpoints added at every step boundary below: (1)
                // `Task.isCancelled` — the OS's own cooperative-cancellation
                // signal, which DID work correctly on wake 0 (caught between
                // chunks, clean `expiration_handler` return); checking it
                // EARLIER, before the risky synchronous calls, gives it more
                // chances to catch an expiring wake before the hard kill.
                // (2) a loose wall-clock backstop, independent of (1), in
                // case cancellation doesn't propagate through some
                // synchronous call for whatever reason — threshold is set
                // well above the observed ~10s death zone so it never
                // preempts a legitimately long wake like wake 0's ~324s one.
                // The goal either way: reach a clean `return` so
                // `handleBGTrainTask`'s wrapper actually calls
                // `task.setTaskCompleted`, instead of the process just
                // vanishing with no acknowledgment sent to the scheduler —
                // untested hypothesis, but a real gap either way (see
                // memory/CLAUDE.md for the fuller reasoning).
                func earlyExit(_ reason: String) -> BGWakeResult {
                    BGWakeResult(
                        iterationsCompleted: initialIterationsCompleted,
                        cumulativeDeviceElapsedS: initialCumulativeDeviceElapsedS,
                        terminationReason: reason)
                }
                func expiring() -> Bool {
                    Task.isCancelled
                        || Date.timeIntervalSinceReferenceDate - wakeStart
                            > TrainBenchConstants.bgWallClockBackstopS
                }

                if expiring() {
                    return earlyExit(Task.isCancelled ? "expiration_handler" : "wall_clock_backstop")
                }

                let config2 = LoRAConfiguration(
                    numLayers: TrainBenchConstants.loraLayers,
                    loraParameters: .init(
                        rank: TrainBenchConstants.loraRank,
                        scale: TrainBenchConstants.loraScale,
                        keys: TrainBenchConstants.loraKeys))
                // Bracketed specifically because 4 consecutive real wakes
                // that needed to resume a checkpoint died somewhere between
                // `model_loaded` and the next marker, while the 1 fresh-start
                // wake (no resume needed) sailed through this same region —
                // this call is the one thing both paths share, so bracketing
                // it directly is the next disambiguation step.
                Self.appendBGMarker(
                    ctx, recordType: "lora_apply_start",
                    extra: [
                        "wake_elapsed_s": Date.timeIntervalSinceReferenceDate - wakeStart,
                        "peak_mem_bytes": Memory.snapshot().peakMemory,
                        "active_mem_bytes": Memory.snapshot().activeMemory,
                    ])
                _ = try LoRAContainer.from(model: mc.model, configuration: config2)
                Self.appendBGMarker(
                    ctx, recordType: "lora_apply_complete",
                    extra: [
                        "wake_elapsed_s": Date.timeIntervalSinceReferenceDate - wakeStart,
                        "peak_mem_bytes": Memory.snapshot().peakMemory,
                        "active_mem_bytes": Memory.snapshot().activeMemory,
                    ])

                if expiring() {
                    return earlyExit(Task.isCancelled ? "expiration_handler" : "wall_clock_backstop")
                }

                if TrainBenchConstants.gradientCheckpointing {
                    (mc.model as? SmolLM3Model)?.useGradientCheckpoint = true
                }

                let weightsURL = Self.bgWeightsURL(user: ctx.user)
                if initialIterationsCompleted > 0,
                    FileManager.default.fileExists(atPath: weightsURL.path)
                {
                    if expiring() {
                        return earlyExit(Task.isCancelled ? "expiration_handler" : "wall_clock_backstop")
                    }
                    // Bracketed with its own markers + peak-memory reading —
                    // this exact step (loading + assigning the saved
                    // checkpoint on top of an already-loaded model) is the
                    // prime suspect for two silent real-wake deaths (see the
                    // doc comment on `loadLoRAWeights`). If a future wake
                    // dies with `resume_start` but no `resumed`, this is
                    // confirmed as the bottleneck; the peak-mem reading on
                    // success helps judge whether memory pressure is
                    // plausible even when it doesn't outright kill the wake.
                    GPU.resetPeakMemory()
                    let resumeStart = Date.timeIntervalSinceReferenceDate
                    Self.appendBGMarker(
                        ctx, recordType: "resume_start",
                        extra: ["wake_elapsed_s": resumeStart - wakeStart])
                    try Self.loadLoRAWeights(model: mc.model, url: weightsURL)
                    self.tlog("BG wake: resumed weights from \(weightsURL.lastPathComponent)")
                    Self.appendBGMarker(
                        ctx, recordType: "resumed",
                        extra: [
                            "wake_elapsed_s": Date.timeIntervalSinceReferenceDate - wakeStart,
                            "step_elapsed_s": Date.timeIntervalSinceReferenceDate - resumeStart,
                            "peak_mem_bytes": Memory.snapshot().peakMemory,
                        ])
                    GPU.resetPeakMemory()
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
                var localTerminationReason = "voluntary_yield"

                // Diagnostic marker: LoRA applied, GC flag set, weights
                // resumed (if any) — everything before the training loop
                // itself is done. If a wake dies with `model_loaded` but no
                // `training_setup_complete`, the bottleneck is LoRA/GC setup
                // or the weight-resume step, not model load or training.
                Self.appendBGMarker(
                    ctx, recordType: "training_setup_complete",
                    extra: [
                        "resume_from_iter": initialIterationsCompleted,
                        "wake_elapsed_s": Date.timeIntervalSinceReferenceDate - wakeStart,
                    ])

                // ONE chunk per wake, then a voluntary clean return — schema
                // v5 (see TrainBenchConstants.bgAppBuild's changelog). This
                // used to be a `while`, looping for more chunks whenever time
                // allowed (wake 0 ran 4 back-to-back); now it always stops
                // after exactly one chunk regardless of remaining budget, to
                // test whether a consistently small, quick, always-completes
                // request pattern earns steadier scheduling than a greedy one.
                if localIterationsCompleted < ctx.iterationsTotal && !Task.isCancelled {
                    let remaining = ctx.iterationsTotal - localIterationsCompleted
                    let chunk = min(TrainBenchConstants.bgChunkIterations, remaining)
                    let iterationsCompletedBeforeChunk = localIterationsCompleted

                    // Diagnostic marker: about to attempt this chunk. If a
                    // wake dies with a `chunk_start` for iteration N but no
                    // matching `checkpoint`, the bottleneck is somewhere
                    // inside that specific `LoRATrain.train` call (the 10
                    // training steps themselves took too long / got evicted
                    // mid-chunk), not setup.
                    Self.appendBGMarker(
                        ctx, recordType: "chunk_start",
                        extra: [
                            "step_target": iterationsCompletedBeforeChunk + chunk,
                            "wake_elapsed_s": Date.timeIntervalSinceReferenceDate - wakeStart,
                        ])

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

                    localTerminationReason =
                        localIterationsCompleted >= ctx.iterationsTotal
                        ? "run_complete" : "voluntary_yield"
                } else if Task.isCancelled {
                    localTerminationReason = "expiration_handler"
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
