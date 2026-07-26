// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Diagnostic-only memory trace of `loadWeights`'s internal stages.
///
/// Added for the h6 on-device background-training investigation
/// (2026-07-15) into why model load costs ~2x the on-disk quantized
/// weight size (~3.46GB peak for a ~1.73GB 4-bit checkpoint) — the v7
/// fix (`weights.removeAll()` before `eval(model)`) didn't move the
/// number, which (per `Module.update`'s `_updateInternal` — a reference
/// swap, not a copy) is expected: dropping the `weights` dict reference
/// can't free memory the model's own parameter storage still points at.
/// This traces each internal stage so the caller (which owns the JSONL
/// bench logging) can localize exactly where the doubling first appears
/// — model construction, the safetensors read, `quantize(model:)`, or
/// `update(parameters:)`. Not used by `loadWeights` itself; purely a
/// side channel for the caller to drain and log.
public struct LoadWeightsStageSnapshot: Sendable {
    public let stage: String
    public let peakMemBytes: Int
    public let activeMemBytes: Int
    public let capturedAtReferenceDate: TimeInterval
}

public final class LoadWeightsDiagnostics: @unchecked Sendable {
    public static let shared = LoadWeightsDiagnostics()
    private let lock = NSLock()
    private var trace: [LoadWeightsStageSnapshot] = []

    public func record(stage: String) {
        let snap = Memory.snapshot()
        lock.lock()
        trace.append(
            LoadWeightsStageSnapshot(
                stage: stage, peakMemBytes: snap.peakMemory, activeMemBytes: snap.activeMemory,
                capturedAtReferenceDate: Date.timeIntervalSinceReferenceDate))
        lock.unlock()
    }

    /// Returns the accumulated trace and clears it, so each `loadWeights`
    /// call's stages can be attributed to that specific wake/call.
    public func drain() -> [LoadWeightsStageSnapshot] {
        lock.lock()
        defer { lock.unlock() }
        let t = trace
        trace = []
        return t
    }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    LoadWeightsDiagnostics.shared.record(stage: "loadWeights_entry")

    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let (w, m) = try loadArraysAndMetadata(url: url)
            for (key, value) in w {
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
        }
    }
    LoadWeightsDiagnostics.shared.record(stage: "safetensors_read")

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)
    LoadWeightsDiagnostics.shared.record(stage: "sanitize_complete")

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }
    LoadWeightsDiagnostics.shared.record(stage: "quantize_applied")

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])
    LoadWeightsDiagnostics.shared.record(stage: "parameters_updated")

    // v7 (2026-07-15) tried `weights.removeAll()` here, on the theory that
    // `weights` and the model's own parameter storage might be separate
    // live arrays after `update(parameters:)`. Confirmed a no-op by both
    // the next wake's telemetry (peak_mem_bytes identical to the pre-fix
    // baseline, to the byte) and by reading `Module.update` in mlx-swift
    // (`Module.swift`): the leaf-array case calls `p._updateInternal(newArray)`,
    // a reference swap, not a copy — the model's parameter storage and
    // `weights[key]` already point at the SAME MLXArray after `update`,
    // so dropping the `weights` reference here frees nothing. Removed.
    // Real localization now comes from `LoadWeightsDiagnostics` above,
    // which the caller drains after `loadWeights` returns to see which of
    // {safetensors_read, sanitize_complete, quantize_applied,
    // parameters_updated, eval_complete} is where peak memory actually
    // jumps to ~2x the ~1.73GB on-disk quantized weight size.
    eval(model)
    LoadWeightsDiagnostics.shared.record(stage: "eval_complete")
}
