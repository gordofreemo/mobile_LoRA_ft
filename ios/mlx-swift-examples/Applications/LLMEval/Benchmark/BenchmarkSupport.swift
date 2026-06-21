// On-device base-inference characterization benchmark — support types.
//
// Implements the locked design in
//   experiments/2026-06-21-ondevice-base-inference-plan.md
// of the mobile_LoRA_ft project. This file holds the flat-JSON record schema,
// the device-state snapshot, the benchmark grid constants, and the low-level
// per-generation measurement routine (driven directly off `TokenIterator` so we
// can force a fixed token count by ignoring EOS — see `measureGeneration`).
//
// The orchestration (cold-start, prefill/decode curves, realistic tier, stress
// run, cooldown gating) lives in `LLMEvaluator+Benchmark.swift`.

import Foundation
import MLX
import MLXLMCommon

#if canImport(UIKit)
    import UIKit
#endif

// MARK: - Constants / grid

enum BenchConstants {
    /// Bump whenever the harness logic changes so every JSONL ties back to a
    /// specific harness version (baked into each record as `app_build`).
    static let appBuild = "smollm3-ondevice-bench-h2"
    static let schemaVersion = 2

    // Star grid (decision 5): one axis at a time, not a cross-product.
    static let prefillPromptTargets = [64, 256, 512, 1024, 2048]
    static let prefillGenTokens = 64

    static let decodeGenTargets = [128, 512, 1024, 2048]
    static let decodePromptTarget = 256

    /// Measured runs per cell + one discarded warmup (decision 5 / 8).
    static let measuredPerCell = 5
    static let realisticRuns = 5

    /// Natural-EOS cap for the realistic LaMP-3 tier — answers are ~1 token but
    /// guard against a rambling generation.
    static let realisticMaxTokens = 16

    /// Sustained stress run (decision 6): ~5 min of forced continuous decode,
    /// logging per-segment tok/s every `stressSegmentTokens`.
    static let stressMaxSeconds = 300.0
    static let stressSegmentTokens = 256
    static let stressMaxTokens = 60_000

    /// Inter-cell cooldown is gated on `thermalState == nominal`, capped here
    /// (decision 6) rather than a blind sleep.
    static let cooldownCapSeconds = 120.0
}

/// Which slice of the grid a record belongs to.
enum BenchCell: String {
    case prefill, decode, realistic, cold, stress
}

// MARK: - Device state

/// Per-run device/thermal/battery snapshot (decision 6 / 7). All fields logged
/// so primary (nominal-thermal) numbers stay separable from throttled ones.
struct DeviceState: Sendable {
    let thermalState: String
    let batteryLevel: Double
    let charging: Bool
    let lowPowerMode: Bool

    @MainActor
    static func snapshot() -> DeviceState {
        let info = ProcessInfo.processInfo
        let thermal: String
        switch info.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }

        var level = -1.0
        var charging = false
        #if canImport(UIKit)
            let device = UIDevice.current
            level = Double(device.batteryLevel)  // -1 if monitoring disabled
            switch device.batteryState {
            case .charging, .full: charging = true
            default: charging = false
            }
        #endif

        return DeviceState(
            thermalState: thermal,
            batteryLevel: level,
            charging: charging,
            lowPowerMode: info.isLowPowerModeEnabled)
    }
}

// MARK: - Measurement result

/// One stress-run segment: tok/s over a `stressSegmentTokens` window.
struct SegmentSample: Sendable {
    let idx: Int
    let cumulativeTokens: Int
    let genTps: Double
}

/// Scalar result of one generation (Sendable so it can cross the `perform`
/// isolation boundary — no MLXArrays escape).
struct GenMeasurement: Sendable {
    let promptTokens: Int
    let genTokens: Int
    let ttftMs: Double
    let genTps: Double
    let promptTps: Double
    let genTimeS: Double
    let peakMemBytes: Int
    let truncated: Bool
    /// Non-empty only for the stress cell.
    let segments: [SegmentSample]
}

/// EOS / stop token set, mirroring the private `buildStopTokenIds` in
/// MLXLMCommon's Evaluate.swift so natural-EOS runs stop exactly where the
/// library would.
private func stopTokenIds(_ context: ModelContext) -> Set<Int> {
    var stops = context.configuration.eosTokenIds
    if let e = context.tokenizer.eosTokenId { stops.insert(e) }
    for token in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(token) { stops.insert(id) }
    }
    return stops
}

/// Drive `TokenIterator` directly and measure one generation.
///
/// `forced == true` suppresses end-of-sequence: we keep pulling tokens until
/// `maxTokens` regardless of EOS (the EOS stop check lives in MLXLMCommon's loop
/// wrapper, not in `TokenIterator.next()` itself — verified against Evaluate.swift),
/// giving an exact gen-length axis. `forced == false` stops on the model's EOS
/// for the realistic tier.
///
/// TTFT brackets construction (prompt prefill) → first token, matching the prior
/// single-record convention. Decode tok/s = genTokens / (time after first token),
/// also matching the existing app's `generateTps`.
///
/// Must be called inside `ModelContainer.perform` so the model/tokenizer stay
/// within the serial-access isolation and all MLXArrays are evaluated before return.
func measureGeneration(
    context: ModelContext,
    userInput: UserInput,
    maxTokens: Int,
    forced: Bool,
    segmentTokens: Int? = nil,
    maxSeconds: Double? = nil
) async throws -> GenMeasurement {
    let lmInput = try await context.processor.prepare(input: userInput)
    let promptTokens = lmInput.text.tokens.size

    // Greedy (temperature 0 → ArgMaxSampler) for reproducibility; decode rate is
    // sampler-independent. EOS handled by us, not the sampler.
    let params = GenerateParameters(maxTokens: maxTokens, temperature: 0)
    let stops = stopTokenIds(context)
    let unknown = context.tokenizer.unknownTokenId

    let start = Date.timeIntervalSinceReferenceDate
    var iterator = try TokenIterator(input: lmInput, model: context.model, parameters: params)

    // First token = end of TTFT window.
    guard let firstTok = iterator.next() else {
        return GenMeasurement(
            promptTokens: promptTokens, genTokens: 0, ttftMs: 0, genTps: 0,
            promptTps: 0, genTimeS: 0, peakMemBytes: Memory.snapshot().peakMemory,
            truncated: false, segments: [])
    }
    let firstTick = Date.timeIntervalSinceReferenceDate
    let ttftMs = (firstTick - start) * 1000

    var genTokens = 1
    var segments: [SegmentSample] = []
    var segStart = firstTick
    var segCount = 1
    var segIdx = 0

    let firstIsStop = !forced && (stops.contains(firstTok) || firstTok == unknown)
    if !firstIsStop {
        loop: while let tok = iterator.next() {
            if !forced && (stops.contains(tok) || tok == unknown) { break loop }
            genTokens += 1
            segCount += 1

            if let segT = segmentTokens, segCount >= segT {
                let now = Date.timeIntervalSinceReferenceDate
                segIdx += 1
                segments.append(
                    SegmentSample(
                        idx: segIdx, cumulativeTokens: genTokens,
                        genTps: Double(segCount) / (now - segStart)))
                segStart = now
                segCount = 0
            }

            if let cap = maxSeconds, Date.timeIntervalSinceReferenceDate - firstTick > cap {
                break loop
            }
        }
    }

    let end = Date.timeIntervalSinceReferenceDate
    // TokenIterator uses asyncEval to keep the pipeline full; synchronize before
    // returning so pending GPU work doesn't trip teardown asserts.
    MLX.Stream().synchronize()

    let genTime = end - firstTick
    let genTps = genTime > 0 ? Double(genTokens) / genTime : 0
    let promptTps = ttftMs > 0 ? Double(promptTokens) / (ttftMs / 1000.0) : 0
    let truncated = !forced && genTokens >= maxTokens

    return GenMeasurement(
        promptTokens: promptTokens, genTokens: genTokens, ttftMs: ttftMs,
        genTps: genTps, promptTps: promptTps, genTimeS: genTime,
        peakMemBytes: Memory.snapshot().peakMemory, truncated: truncated,
        segments: segments)
}
