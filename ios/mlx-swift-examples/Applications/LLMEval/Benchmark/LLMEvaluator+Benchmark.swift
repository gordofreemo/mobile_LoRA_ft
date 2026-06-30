// On-device base-inference characterization benchmark — orchestration.
//
// Auto-started by a launch arg (`--benchmark` / `--benchmark-cold`) so the whole
// run is one Mac-driven `devicectl process launch --console` command, zero
// per-generation taps (decision 2). Writes one flat-JSON record per generation to
// Documents/bench_metrics.jsonl, then exits so `--console` unblocks and the JSONL
// can be pulled.
//
// See experiments/2026-06-21-ondevice-base-inference-plan.md for the locked design.

import Foundation
import MLX
import MLXLMCommon

#if canImport(UIKit)
    import UIKit
#endif

extension LLMEvaluator {

    /// True when the app was launched to run the benchmark instead of the UI.
    static var benchmarkLaunchMode: BenchLaunchMode? {
        let args = CommandLine.arguments
        if args.contains("--benchmark-cold") { return .coldOnly }
        if args.contains("--benchmark-tail") { return .tail }
        if args.contains("--benchmark") { return .full }
        return nil
    }

    enum BenchLaunchMode {
        case full
        case coldOnly
        /// Realistic tier + stress run only (no cold/prefill/decode). Lets the
        /// long-tail cells be captured in a separate, time-bounded launch when the
        /// full grid would otherwise overrun a single `--console` session.
        case tail
    }

    // MARK: - Entry point

    /// Run the benchmark and exit. Safe to call once on launch.
    func runBenchmark(mode: BenchLaunchMode) async {
        enableThinking = false  // decision 9: thinking off throughout

        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let sessionId = UUID().uuidString
        benchLogLine("benchmark start session=\(sessionId) mode=\(mode) build=\(BenchConstants.appBuild)")

        // 1. Cold start: time the model load, then one cold generation. This is the
        //    "app launch → first answer" number (decision 8). Never averaged into
        //    steady-state.
        let loadStart = Date.timeIntervalSinceReferenceDate
        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            benchLogLine("benchmark FAILED to load model: \(error)")
            finishBenchmark()
            return
        }
        let modelLoadMs = (Date.timeIntervalSinceReferenceDate - loadStart) * 1000

        // Tail mode: realistic + stress only (the long-tail cells), separate launch.
        if mode == .tail {
            await runCell(
                container: container, cell: .realistic, input: Self.realisticInput(),
                targetPrompt: nil, targetGen: nil,
                maxTokens: BenchConstants.realisticMaxTokens, forced: false,
                repeats: BenchConstants.realisticRuns, sessionId: sessionId)
            await cooldown()
            await runStress(container: container, sessionId: sessionId)
            benchLogLine("tail benchmark complete session=\(sessionId)")
            finishBenchmark()
            return
        }

        let coldInput = Self.realisticInput()
        await runOne(
            container: container, cell: .cold, input: coldInput,
            targetPrompt: nil, targetGen: 64, maxTokens: 64, forced: true,
            warmup: false, cold: true, modelLoadMs: modelLoadMs, repeatIdx: 0,
            sessionId: sessionId)

        if mode == .coldOnly {
            benchLogLine("cold-only benchmark complete")
            finishBenchmark()
            return
        }

        // 2. Prefill curve: prompt ∈ targets @ fixed gen.
        for target in BenchConstants.prefillPromptTargets {
            let text = await paddedUserText(container: container, targetPromptTokens: target)
            let input = Self.userOnlyInput(text)
            await runCell(
                container: container, cell: .prefill, input: input,
                targetPrompt: target, targetGen: BenchConstants.prefillGenTokens,
                maxTokens: BenchConstants.prefillGenTokens, forced: true, sessionId: sessionId)
            await cooldown()
        }

        // 3. Decode curve: gen ∈ targets @ fixed prompt.
        let decodeText = await paddedUserText(
            container: container, targetPromptTokens: BenchConstants.decodePromptTarget)
        for target in BenchConstants.decodeGenTargets {
            let input = Self.userOnlyInput(decodeText)
            await runCell(
                container: container, cell: .decode, input: input,
                targetPrompt: BenchConstants.decodePromptTarget, targetGen: target,
                maxTokens: target, forced: true, sessionId: sessionId)
            await cooldown()
        }

        // 4. Realistic tier: natural-EOS on a real LaMP-3 BM25-k4 prompt.
        await runCell(
            container: container, cell: .realistic, input: Self.realisticInput(),
            targetPrompt: nil, targetGen: nil,
            maxTokens: BenchConstants.realisticMaxTokens, forced: false,
            repeats: BenchConstants.realisticRuns, sessionId: sessionId)
        await cooldown()

        // 5. Sustained stress run (expected to leave nominal — that's the point).
        await runStress(container: container, sessionId: sessionId)

        benchLogLine("benchmark complete session=\(sessionId)")
        finishBenchmark()
    }

    // MARK: - Cell runners

    /// One cell = `measuredPerCell` measured runs + a discarded warmup that absorbs
    /// per-config Metal-shader compile (decision 8).
    private func runCell(
        container: ModelContainer, cell: BenchCell, input: UserInput,
        targetPrompt: Int?, targetGen: Int?, maxTokens: Int, forced: Bool,
        repeats: Int = BenchConstants.measuredPerCell, sessionId: String
    ) async {
        // Warmup (discarded).
        await runOne(
            container: container, cell: cell, input: input, targetPrompt: targetPrompt,
            targetGen: targetGen, maxTokens: maxTokens, forced: forced, warmup: true,
            cold: false, modelLoadMs: nil, repeatIdx: -1, sessionId: sessionId)

        for i in 0 ..< repeats {
            await runOne(
                container: container, cell: cell, input: input, targetPrompt: targetPrompt,
                targetGen: targetGen, maxTokens: maxTokens, forced: forced, warmup: false,
                cold: false, modelLoadMs: nil, repeatIdx: i, sessionId: sessionId)
        }
    }

    private func runOne(
        container: ModelContainer, cell: BenchCell, input: UserInput,
        targetPrompt: Int?, targetGen: Int?, maxTokens: Int, forced: Bool,
        warmup: Bool, cold: Bool, modelLoadMs: Double?, repeatIdx: Int, sessionId: String
    ) async {
        let device = DeviceState.snapshot()
        do {
            let m = try await container.perform(nonSendable: input) { ctx, ui in
                try await measureGeneration(
                    context: ctx, userInput: ui, maxTokens: maxTokens, forced: forced)
            }
            writeRecord(
                cell: cell, m: m, targetPrompt: targetPrompt, targetGen: targetGen,
                maxTokens: maxTokens, forced: forced, warmup: warmup, cold: cold,
                modelLoadMs: modelLoadMs, repeatIdx: repeatIdx, device: device,
                sessionId: sessionId)
        } catch {
            benchLogLine("run failed cell=\(cell.rawValue) repeat=\(repeatIdx): \(error)")
        }
    }

    /// Sustained ~5-min forced decode, one flat record per `stressSegmentTokens`
    /// segment (decision 6). No warmup — this run is about the thermal trajectory.
    private func runStress(container: ModelContainer, sessionId: String) async {
        let device = DeviceState.snapshot()
        let input = Self.userOnlyInput(
            await paddedUserText(
                container: container, targetPromptTokens: BenchConstants.decodePromptTarget))
        do {
            let m = try await container.perform(nonSendable: input) { ctx, ui in
                try await measureGeneration(
                    context: ctx, userInput: ui, maxTokens: BenchConstants.stressMaxTokens,
                    forced: true, segmentTokens: BenchConstants.stressSegmentTokens,
                    maxSeconds: BenchConstants.stressMaxSeconds)
            }
            for seg in m.segments {
                writeRecord(
                    cell: .stress, m: m, targetPrompt: BenchConstants.decodePromptTarget,
                    targetGen: nil, maxTokens: BenchConstants.stressMaxTokens, forced: true,
                    warmup: false, cold: false, modelLoadMs: nil, repeatIdx: 0,
                    device: device, sessionId: sessionId, segment: seg)
            }
            benchLogLine("stress complete segments=\(m.segments.count) totalTokens=\(m.genTokens)")
        } catch {
            benchLogLine("stress run failed: \(error)")
        }
    }

    /// Cooldown gated on `thermalState == nominal`, capped (decision 6).
    private func cooldown() async {
        let start = Date.timeIntervalSinceReferenceDate
        while ProcessInfo.processInfo.thermalState != .nominal {
            if Date.timeIntervalSinceReferenceDate - start > BenchConstants.cooldownCapSeconds {
                benchLogLine("cooldown cap hit (still non-nominal)")
                break
            }
            try? await Task.sleep(for: .seconds(2))
        }
    }

    private func finishBenchmark() {
        // Exit so `devicectl ... launch --console` unblocks and the JSONL can be pulled.
        benchLogLine("exiting after benchmark")
        exit(0)
    }

    // MARK: - Inputs

    private static func userOnlyInput(_ text: String) -> UserInput {
        UserInput(
            chat: [.user(text)],
            additionalContext: ["enable_thinking": false])
    }

    /// A real LaMP-3 BM25-k4-in-system prompt (the true deployment input shape,
    /// decision 4). System layout + per-entry format mirror eval/eval_lamp.py.
    private static func realisticInput() -> UserInput {
        let system =
            "The following are examples of this user's past activity. "
            + "Use them to match this user's preferences and writing style.\n\n"
            + "- Review: \"Arrived quickly and works exactly as described. Sturdy build, "
            + "would buy again.\" — the user rated it 5/5\n"
            + "- Review: \"Decent for the price but the battery drains faster than I "
            + "expected. Does the job.\" — the user rated it 3/5\n"
            + "- Review: \"Stopped working after two weeks. Customer support was no help "
            + "at all. Very disappointed.\" — the user rated it 1/5\n"
            + "- Review: \"Pretty good overall. A couple of minor annoyances with the "
            + "interface but nothing dealbreaking.\" — the user rated it 4/5"
        let user =
            "What is the score of the following review on a scale of 1 to 5? "
            + "just answer with 1, 2, 3, 4, or 5 without further explanation. review: "
            + "The product is okay. It does what it says but feels a little cheap and "
            + "the instructions were confusing. I might keep it for now."
        return UserInput(
            chat: [.system(system), .user(user)],
            additionalContext: ["enable_thinking": false])
    }

    /// Build a user message of approximately `targetPromptTokens` tokens by slicing
    /// the filler corpus in token space. Content is irrelevant to prefill cost
    /// (decision 4); we bin on the *measured* prompt_tokens, so an approximate hit
    /// is fine.
    private func paddedUserText(container: ModelContainer, targetPromptTokens: Int) async -> String {
        // Approx tokens added by the (system-less) chat template + generation prompt.
        let overhead = 12
        let want = max(targetPromptTokens - overhead, 1)
        let fillerTokens = await container.encode(Self.fillerText)
        let slice = Array(fillerTokens.prefix(want))
        return await container.decode(tokens: slice)
    }

    /// Regular filler — content-independent, only token count matters. Long enough
    /// to cover the 2048-token prefill cell.
    private static let fillerText: String = String(
        repeating:
            "The quick brown fox jumps over the lazy dog near the quiet river bank at dawn. ",
        count: 400)

    // MARK: - Record writer

    private func writeRecord(
        cell: BenchCell, m: GenMeasurement, targetPrompt: Int?, targetGen: Int?,
        maxTokens: Int, forced: Bool, warmup: Bool, cold: Bool, modelLoadMs: Double?,
        repeatIdx: Int, device: DeviceState, sessionId: String, segment: SegmentSample? = nil
    ) {
        let modelName = modelConfiguration.name.components(separatedBy: "/").last
            ?? modelConfiguration.name

        let record: [String: Any] = [
            // Existing schema (kept).
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "model": modelName,
            "prompt_tokens": m.promptTokens,
            "gen_tokens": segment?.cumulativeTokens ?? m.genTokens,
            "ttft_ms": m.ttftMs,
            "gen_tps": segment?.genTps ?? m.genTps,
            "prompt_tps": m.promptTps,
            "gen_time_s": m.genTimeS,
            "peak_mem_bytes": m.peakMemBytes,
            "max_tokens": maxTokens,
            "thinking": false,
            "truncated": m.truncated,
            "device_model": Self.hwModelIdentifier(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
            // Device/thermal state.
            "thermal_state": device.thermalState,
            "battery_level": device.batteryLevel,
            "charging": device.charging,
            "low_power_mode": device.lowPowerMode,
            // Run semantics.
            "cold": cold,
            "model_load_ms": modelLoadMs ?? NSNull(),
            "warmup": warmup,
            "forced": forced,
            "target_prompt_tokens": targetPrompt ?? NSNull(),
            "target_gen_tokens": targetGen ?? NSNull(),
            // Labeling / provenance.
            "bench_session_id": sessionId,
            "cell": cell.rawValue,
            "repeat_idx": repeatIdx,
            "app_build": BenchConstants.appBuild,
            "bench_schema_version": BenchConstants.schemaVersion,
            "git_commit": BenchConstants.gitCommit,
            "git_dirty": BenchConstants.gitDirty,
            // Stress segment fields (null elsewhere).
            "segment_idx": segment?.idx ?? NSNull(),
            "cumulative_tokens": segment?.cumulativeTokens ?? NSNull(),
        ]

        guard
            let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else {
            benchLogLine("failed to serialize bench record")
            return
        }
        appendBenchLine(json)
    }

    /// Hardware identifier, e.g. "iPhone18,1".
    private static func hwModelIdentifier() -> String {
        var sysinfo = utsname()
        uname(&sysinfo)
        let mirror = Mirror(reflecting: sysinfo.machine)
        let id = mirror.children.compactMap { ($0.value as? Int8) }
            .filter { $0 != 0 }.map { String(UnicodeScalar(UInt8($0))) }.joined()
        return id.isEmpty ? "unknown" : id
    }
}
