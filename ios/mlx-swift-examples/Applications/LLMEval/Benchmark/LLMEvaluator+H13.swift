// h13 — on-device validation of the OPPU movie-tagging personalization effect.
//
// Two launch modes, both per-user:
//   --benchmark-h13-train --user <id>            train one User-LoRA to completion
//   --benchmark-h13-eval  --user <id> --arm <a>  generate this user's test queries
//
// Arms: rag (no adapter) | cluster | mac | device. All four generate on the
// phone, over the same 4-bit quantisation of THEIR merged movie task adapter,
// with the same sampler — so per-query pairing is clean and the comparison is
// internally consistent. The published bf16/HF numbers do not carry over.
//
// Spec: experiments/2026-08-19-ondevice-oppu-movie-validation-h13-plan.md

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers

#if canImport(UIKit)
    import UIKit
#endif

/// Serializes appends to the h13 JSONLs — the 30s passive sampler and the
/// training/eval loops write interleaved for the whole run.
private let h13FileLock = NSLock()

extension LLMEvaluator {

    // MARK: - Data

    /// One pre-tokenized training example. `ids` is the full sequence,
    /// `[lossStart, lossEnd)` the supervised span — exactly run_oppu.py's
    /// `generate_and_tokenize_prompt` (labels masked over the prompt prefix).
    struct H13Example: Sendable {
        let id: String
        let ids: [Int32]
        let lossStart: Int
        let lossEnd: Int
    }

    /// One pre-tokenized eval prompt. Tokenized Mac-side so the device never
    /// re-tokenizes and cannot drift from the cluster arm's text.
    struct H13Query: Sendable {
        let id: String
        let ids: [Int32]
    }

    final class H13SpanBox: @unchecked Sendable {
        var range: Range<Int> = 0 ..< 1
    }

    /// Harness-owned AdamW with DECOUPLED weight decay (their
    /// TrainingArguments weight_decay=1e-2, optim=adamw_torch):
    ///   m ← β1·m + (1−β1)·g;  v ← β2·v + (1−β2)·g²;  t ← t+1
    ///   p ← p − lr·wd·p − lr·(m/(1−β1ᵗ)) / (√(v/(1−β2ᵗ)) + eps)
    /// h12's optimizer omits the decay term because its recipe sets wd = 0.
    final class H13AdamW: @unchecked Sendable {
        var m: [String: MLXArray] = [:]
        var v: [String: MLXArray] = [:]
        var t: Int = 0
        let beta1 = TrainBenchConstants.h13AdamBeta1
        let beta2 = TrainBenchConstants.h13AdamBeta2
        let eps = TrainBenchConstants.h13AdamEps
        let weightDecay = TrainBenchConstants.h13WeightDecay

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
                guard let p = current[k] else { continue }
                let step = (mNew / bc1) / (MLX.sqrt(vNew / bc2) + eps)
                newParams.append((k, p - lr * (step + weightDecay * p)))
            }
            model.update(parameters: ModuleParameters.unflattened(newParams))
            eval(model)
            eval(Array(m.values) + Array(v.values))
        }
    }

    struct H13RunContext: Sendable {
        let sessionId: String
        let modelName: String
        let userId: String
        let arm: String
        let nExamples: Int
        let epochs: Int
        let totalSteps: Int
        let scheduleTotalSteps: Int
        let naxArm: String?
        let condition: String
    }

    // MARK: - Launch args

    /// `--arm rag|cluster|mac|device` — which adapter the h13 eval mode loads.
    /// Accepts a comma-separated list (or `all`); every arm in the list is
    /// evaluated inside ONE process launch, sharing a single model load. That
    /// load is ~30-60 s against ~0.4 s of generation per query, so running the
    /// arms separately would spend more wall-clock on loading than on the
    /// measurement it exists to take.
    static var h13Arms: [String]? {
        guard let v = launchArgValueH13("--arm") else { return nil }
        let all = ["rag", "cluster", "mac", "device"]
        let want = v == "all" ? all : v.split(separator: ",").map(String.init)
        let ok = want.filter { all.contains($0) }
        return ok.isEmpty ? nil : ok
    }

    private nonisolated static func launchArgValueH13(_ flag: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    /// Side-loaded model directory if present, else the Hub id. Side-loading is
    /// preferred: it keeps the merged movie model off the public Hub.
    nonisolated static func h13ModelConfiguration() -> ModelConfiguration {
        let dir = URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.h13ModelDirName)
        if FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path) {
            return ModelConfiguration(directory: dir, defaultPrompt: "Why is the sky blue?")
        }
        return ModelConfiguration(
            id: TrainBenchConstants.h13ModelId, defaultPrompt: "Why is the sky blue?")
    }

    nonisolated static func h13UserDir(_ user: String) -> URL {
        URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.h13DataDirName)
            .appendingPathComponent(user)
    }

    nonisolated static func h13AdapterURL(user: String, arm: String) -> URL {
        URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.h13AdapterDirName)
            .appendingPathComponent(user)
            .appendingPathComponent(arm)
            .appendingPathComponent("adapters.safetensors")
    }

    /// Parse the side-loaded training corpus. Returns nil on ANY malformed line
    /// — a partially-consumed corpus silently changes the step count and data
    /// order, which is worse than failing loudly.
    nonisolated static func loadH13Train(user: String) -> [H13Example]? {
        let url = h13UserDir(user).appendingPathComponent("train.jsonl")
        guard let content = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var out: [H13Example] = []
        for line in content.split(separator: "\n") {
            guard
                let d = line.data(using: .utf8),
                let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                let ids = obj["input_ids"] as? [Int],
                let lossStart = obj["loss_start"] as? Int,
                let lossEnd = obj["loss_end"] as? Int,
                ids.count >= 2, lossStart >= 1, lossEnd > lossStart, lossEnd <= ids.count
            else { return nil }
            let id = (obj["id"] as? String) ?? "?"
            out.append(
                H13Example(
                    id: id, ids: ids.map(Int32.init),
                    lossStart: lossStart, lossEnd: lossEnd))
        }
        return out
    }

    nonisolated static func loadH13Queries(user: String) -> [H13Query]? {
        let url = h13UserDir(user).appendingPathComponent("eval.jsonl")
        guard let content = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var out: [H13Query] = []
        for line in content.split(separator: "\n") {
            guard
                let d = line.data(using: .utf8),
                let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                let ids = obj["input_ids"] as? [Int], !ids.isEmpty
            else { return nil }
            let id =
                (obj["id"] as? String) ?? (obj["id"] as? NSNumber)?.stringValue ?? "?"
            out.append(H13Query(id: id, ids: ids.map(Int32.init)))
        }
        return out
    }

    /// Number of examples in ONE epoch (the corpus file holds `epochs` copies,
    /// each a distinct permutation baked in by the Mac builder).
    nonisolated static func h13ExamplesPerEpoch(_ n: Int) -> Int {
        n / TrainBenchConstants.h13Epochs
    }

    /// Accumulation windows, as (startIndex, count) into the flat corpus.
    /// Windows never straddle an epoch boundary — HF restarts accumulation at
    /// each epoch, so every epoch ends with a partial window.
    nonisolated static func h13Windows(nPerEpoch: Int) -> [(Int, Int)] {
        let w = TrainBenchConstants.h13AccumWindow
        var out: [(Int, Int)] = []
        for e in 0 ..< TrainBenchConstants.h13Epochs {
            var i = 0
            while i < nPerEpoch {
                out.append((e * nPerEpoch + i, min(w, nPerEpoch - i)))
                i += w
            }
        }
        return out
    }

    /// HF cosine-with-warmup, identical to h12's (verified there against the
    /// cluster metrics.jsonl): warmup = ceil(0.03 × total) steps, linear 0→base,
    /// then base × 0.5 × (1 + cos(π × progress)).
    nonisolated static func h13LR(step: Int, scheduleTotalSteps: Int) -> Float {
        let warmup = Int(
            ceil(Double(scheduleTotalSteps) * TrainBenchConstants.h13WarmupRatio))
        if step < warmup {
            return TrainBenchConstants.h13BaseLR * Float(step) / Float(max(1, warmup))
        }
        let progress = Double(step - warmup) / Double(max(1, scheduleTotalSteps - warmup))
        return TrainBenchConstants.h13BaseLR * Float(0.5 * (1.0 + cos(Double.pi * progress)))
    }

    // MARK: - Training

    func runH13TrainBenchmark(user: String?, maxSteps: Int?) async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif

        let stderrURL = URL.documentsDirectory.appendingPathComponent("h13_stderr.log")
        freopen(stderrURL.path, "a", stderr)
        setvbuf(stderr, nil, _IONBF, 0)
        FileHandle.standardError.write(
            Data("\n===== h13 train launch \(ISO8601DateFormatter().string(from: Date())) =====\n".utf8))

        guard let user else {
            tlog("h13 train: --user is required")
            finishTrainBenchmark()
            return
        }
        let sessionId = UUID().uuidString
        let naxArm = Self.trainBenchmarkNaxArm
        if let naxArm { setenv("MLX_ENABLE_NAX_N", naxArm == "on" ? "1" : "0", 1) }

        // Hard model override: h5–h11 leave `modelConfiguration` on the a1lamp
        // model. h13 MUST train over the quantised OPPU movie merge.
        modelConfiguration = Self.h13ModelConfiguration()

        guard let examples = Self.loadH13Train(user: user), !examples.isEmpty else {
            tlog("h13 train: failed to load Documents/\(TrainBenchConstants.h13DataDirName)/\(user)/train.jsonl")
            finishTrainBenchmark()
            return
        }
        let n = examples.count
        let nPerEpoch = Self.h13ExamplesPerEpoch(n)
        guard nPerEpoch > 0, nPerEpoch * TrainBenchConstants.h13Epochs == n else {
            tlog("h13 train: corpus size \(n) is not \(TrainBenchConstants.h13Epochs) whole epochs")
            finishTrainBenchmark()
            return
        }
        let windows = Self.h13Windows(nPerEpoch: nPerEpoch)
        let scheduleTotalSteps = windows.count
        var totalSteps = scheduleTotalSteps
        if let maxSteps, maxSteps > 0 { totalSteps = min(totalSteps, maxSteps) }

        let adapterDir = URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.h13AdapterDirName)
            .appendingPathComponent(user)
            .appendingPathComponent("device")
        try? FileManager.default.createDirectory(
            at: adapterDir, withIntermediateDirectories: true)
        let adapterURL = adapterDir.appendingPathComponent("adapters.safetensors")

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("h13 train: model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        let modelName =
            modelConfiguration.name.components(separatedBy: "/").last ?? modelConfiguration.name

        let ctx = H13RunContext(
            sessionId: sessionId, modelName: modelName, userId: user, arm: "device",
            nExamples: nPerEpoch, epochs: TrainBenchConstants.h13Epochs,
            totalSteps: totalSteps, scheduleTotalSteps: scheduleTotalSteps,
            naxArm: naxArm, condition: Self.trainBenchmarkCondition)

        tlog("h13 train user=\(user) n=\(nPerEpoch)x\(TrainBenchConstants.h13Epochs) "
            + "steps=\(totalSteps)/\(scheduleTotalSteps) model=\(modelName)")

        let benchStart = Date.timeIntervalSinceReferenceDate
        let batteryStart = Self.batterySnapshotH13()
        Self.appendH13Marker(
            ctx, recordType: "run_start", file: TrainBenchConstants.h13MetricsFileName,
            extra: [
                "battery_level": batteryStart.0,
                "charging": batteryStart.1,
                "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                "thermal_state": Self.thermalStringH13(),
                // Order fingerprint: catches a stale side-loaded corpus and lets
                // the Mac control prove it consumed the identical sequence.
                "first_example_ids": examples.prefix(3).map { $0.id },
                "total_corpus_tokens": examples.reduce(0) { $0 + $1.ids.count },
            ])

        let sampler = Task { @MainActor in
            while !Task.isCancelled {
                let snap = Self.batterySnapshotH13()
                Self.appendH13Sample(
                    ctx, elapsed: Date.timeIntervalSinceReferenceDate - benchStart,
                    level: snap.0, charging: snap.1, thermal: Self.thermalStringH13(),
                    peak: Memory.snapshot().peakMemory,
                    file: TrainBenchConstants.h13MetricsFileName)
                try? await Task.sleep(for: .seconds(TrainBenchConstants.h13SampleSeconds))
            }
        }

        var trainError: String? = nil
        do {
            try await container.perform { mc in
                let config = LoRAConfiguration(
                    numLayers: TrainBenchConstants.h13LoraLayers,
                    loraParameters: .init(
                        rank: TrainBenchConstants.h13LoraRank,
                        scale: TrainBenchConstants.h13LoraScale,
                        keys: TrainBenchConstants.h13LoraKeys))
                _ = try LoRAContainer.from(model: mc.model, configuration: config)
                if TrainBenchConstants.gradientCheckpointing {
                    (mc.model as? SmolLM3Model)?.checkpointGroupSize = 1
                }
                self.tlog("h13 LoRA applied (r=8, q+v, 36 layers, scale 2.0), GC on")

                // Sliced-lm_head masked loss (h12b): logits only for the
                // supervised span. Identical values and gradients to
                // full-logits-plus-mask, without the [seq, vocab] transient.
                let spanBox = H13SpanBox()
                let lossValueGrad = valueAndGrad(model: mc.model) { model, arrays in
                    let llm = model as! SmolLM3Model
                    let logits = llm.logits(arrays[0], positionRange: spanBox.range)
                        .asType(.float32)
                    return [crossEntropy(logits: logits, targets: arrays[1]).sum()]
                }

                let optimizer = H13AdamW()
                var startStep = 0
                if let ck = Self.loadH13Checkpoint(dir: adapterDir) {
                    do {
                        try mc.model.update(
                            parameters: ModuleParameters.unflattened(Array(ck.0)),
                            verify: .noUnusedKeys)
                        optimizer.m = ck.1
                        optimizer.v = ck.2
                        optimizer.t = ck.3
                        startStep = ck.4
                        self.tlog("h13 RESUMED from step \(startStep)")
                    } catch {
                        self.tlog("h13 checkpoint restore FAILED: \(error) — starting fresh")
                        startStep = 0
                    }
                }
                Self.appendH13Marker(
                    ctx, recordType: "train_begin",
                    file: TrainBenchConstants.h13MetricsFileName,
                    extra: ["resumed_from_step": startStep])

                GPU.resetPeakMemory()

                for step in startStep ..< totalSteps {
                    let stepStart = Date.timeIntervalSinceReferenceDate
                    let (windowStart, microCount) = windows[step]

                    var accum: [String: MLXArray] = [:]
                    var windowCESum: Float = 0
                    var windowMaskedTokens: Float = 0
                    var windowSeqTokens = 0

                    for j in 0 ..< microCount {
                        let ex = examples[windowStart + j]
                        let nTok = ex.ids.count
                        let full = MLXArray(ex.ids).reshaped([1, nTok])
                        let inputs = full[0..., .stride(to: -1)]
                        spanBox.range = (ex.lossStart - 1) ..< (ex.lossEnd - 1)
                        let targetsSlice = full[0..., ex.lossStart ..< ex.lossEnd]

                        let (vals, grad) = lossValueGrad(mc.model, [inputs, targetsSlice])
                        let flat = grad.flattened()
                        if accum.isEmpty {
                            for (k, g) in flat { accum[k] = g }
                        } else {
                            for (k, g) in flat { accum[k] = accum[k]! + g }
                        }
                        eval(Array(accum.values))
                        windowCESum += vals[0].item(Float.self)
                        windowMaskedTokens += Float(ex.lossEnd - ex.lossStart)
                        windowSeqTokens += nTok
                    }

                    // Token-weighted normalization (HF num_items_in_batch) then
                    // global-norm clip on the normalized grads.
                    var sq = MLXArray(Float(0))
                    for g in accum.values { sq = sq + (g * g).sum() }
                    let gradNorm = MLX.sqrt(sq).item(Float.self) / windowMaskedTokens
                    let clip = TrainBenchConstants.h13GradClipNorm
                    let clipScale: Float = gradNorm > clip ? clip / gradNorm : 1.0
                    let finalScale = clipScale / windowMaskedTokens

                    let lr = Self.h13LR(step: step, scheduleTotalSteps: scheduleTotalSteps)
                    optimizer.update(
                        model: mc.model,
                        grads: accum.map { ($0.key, $0.value * finalScale) },
                        learningRate: lr)

                    let now = Date.timeIntervalSinceReferenceDate
                    let peak = Memory.snapshot().peakMemory
                    GPU.resetPeakMemory()
                    Self.appendH13Step(
                        ctx, step: step + 1,
                        epoch: step / max(1, scheduleTotalSteps / TrainBenchConstants.h13Epochs),
                        loss: windowMaskedTokens > 0 ? windowCESum / windowMaskedTokens : 0,
                        lr: lr, gradNorm: gradNorm, clipScale: clipScale,
                        nMicro: microCount, windowMaskedTokens: Int(windowMaskedTokens),
                        windowSeqTokens: windowSeqTokens, windowS: now - stepStart,
                        elapsed: now - benchStart, peak: peak,
                        thermal: Self.thermalStringH13())
                    if (step + 1) % 10 == 0 || step == 0 {
                        self.tlog("h13 step \(step + 1)/\(totalSteps) "
                            + "loss=\(windowMaskedTokens > 0 ? windowCESum / windowMaskedTokens : 0) lr=\(lr)")
                    }
                    if (step + 1) % TrainBenchConstants.h13CheckpointEverySteps == 0
                        || step + 1 == totalSteps
                    {
                        try? Self.saveH13Checkpoint(
                            dir: adapterDir, model: mc.model, optimizer: optimizer,
                            nextStep: step + 1)
                    }
                }

                try LoRATrain.saveLoRAWeights(model: mc.model, url: adapterURL)
                self.tlog("h13 adapter saved -> \(adapterURL.path)")
            }
        } catch {
            trainError = "\(error)"
            tlog("h13 training error: \(error)")
        }

        sampler.cancel()
        let batteryEnd = Self.batterySnapshotH13()
        let saved = FileManager.default.fileExists(atPath: adapterURL.path)
        Self.appendH13Marker(
            ctx, recordType: trainError == nil ? "run_end" : "error",
            file: TrainBenchConstants.h13MetricsFileName,
            extra: [
                "battery_level": batteryStart.0, "battery_level_end": batteryEnd.0,
                "charging": batteryEnd.1,
                "thermal_state": Self.thermalStringH13(),
                "elapsed_s": Date.timeIntervalSinceReferenceDate - benchStart,
                "adapter_saved": saved, "error": trainError ?? NSNull(),
            ])
        tlog("h13 train complete user=\(user) adapter_saved=\(saved) error=\(trainError ?? "none")")
        finishTrainBenchmark()
    }

    // MARK: - On-device four-arm evaluation

    func runH13EvalBenchmark(user: String?, arms: [String]?) async {
        enableThinking = false
        #if canImport(UIKit)
            UIApplication.shared.isIdleTimerDisabled = true
            UIDevice.current.isBatteryMonitoringEnabled = true
        #endif
        let stderrURL = URL.documentsDirectory.appendingPathComponent("h13_stderr.log")
        freopen(stderrURL.path, "a", stderr)
        setvbuf(stderr, nil, _IONBF, 0)

        guard let user, let arms, !arms.isEmpty else {
            tlog("h13 eval: --user and --arm <rag|cluster|mac|device|all|a,b> are required")
            finishTrainBenchmark()
            return
        }
        let sessionId = UUID().uuidString
        if let naxArm = Self.trainBenchmarkNaxArm {
            setenv("MLX_ENABLE_NAX_N", naxArm == "on" ? "1" : "0", 1)
        }
        modelConfiguration = Self.h13ModelConfiguration()

        guard let queries = Self.loadH13Queries(user: user), !queries.isEmpty else {
            tlog("h13 eval: failed to load \(user)/eval.jsonl")
            finishTrainBenchmark()
            return
        }
        // Drop arms with no adapter rather than failing the whole launch — a
        // user's mac/device adapter may legitimately not exist yet.
        let runnable = arms.filter { arm in
            arm == "rag" || FileManager.default.fileExists(
                atPath: Self.h13AdapterURL(user: user, arm: arm).path)
        }
        for skipped in arms where !runnable.contains(skipped) {
            tlog("h13 eval: skipping arm=\(skipped) (no adapter for \(user))")
        }
        guard !runnable.isEmpty else {
            tlog("h13 eval: no runnable arms for \(user)")
            finishTrainBenchmark()
            return
        }

        let container: ModelContainer
        do {
            container = try await load()
        } catch {
            tlog("h13 eval: model load failed: \(error)")
            finishTrainBenchmark()
            return
        }
        let modelName =
            modelConfiguration.name.components(separatedBy: "/").last ?? modelConfiguration.name

        let predDir = URL.documentsDirectory
            .appendingPathComponent(TrainBenchConstants.h13PredDirName)
            .appendingPathComponent(user)
        try? FileManager.default.createDirectory(
            at: predDir, withIntermediateDirectories: true)

        // Captured before entering the container actor — these are main-actor
        // isolated and unreachable from inside `perform`.
        let evalNaxArm = Self.trainBenchmarkNaxArm
        let evalCondition = Self.trainBenchmarkCondition

        let benchStart = Date.timeIntervalSinceReferenceDate
        var evalError: String? = nil
        do {
            try await container.perform { mc in
                let config = LoRAConfiguration(
                    numLayers: TrainBenchConstants.h13LoraLayers,
                    loraParameters: .init(
                        rank: TrainBenchConstants.h13LoraRank,
                        scale: TrainBenchConstants.h13LoraScale,
                        keys: TrainBenchConstants.h13LoraKeys))
                _ = try LoRAContainer.from(model: mc.model, configuration: config)
                // `LoRAContainer.from` leaves lora_b at zero, so this snapshot IS
                // the rag arm: y + scale*(x@a)@0 == y, i.e. exactly the base model
                // through an identical graph. Restoring it between arms also
                // guarantees no adapter leaks into the next one.
                let pristine = Dictionary(
                    uniqueKeysWithValues: mc.model.trainableParameters().flattened()
                        .map { ($0.0, MLX.stopGradient($0.1)) })

                var stopTokens = mc.configuration.eosTokenIds
                if let e = mc.tokenizer.eosTokenId { stopTokens.insert(e) }
                let params = GenerateParameters(
                    maxTokens: TrainBenchConstants.h13MaxNewTokens,
                    temperature: TrainBenchConstants.h13Temperature,
                    topP: TrainBenchConstants.h13TopP,
                    topK: TrainBenchConstants.h13TopK)

                for arm in runnable {
                    let ctx = H13RunContext(
                        sessionId: sessionId, modelName: modelName, userId: user, arm: arm,
                        nExamples: 0, epochs: 0, totalSteps: queries.count,
                        scheduleTotalSteps: queries.count,
                        naxArm: evalNaxArm, condition: evalCondition)
                    let predURL = predDir.appendingPathComponent("\(arm).jsonl")
                    try? FileManager.default.removeItem(at: predURL)

                    try mc.model.update(
                        parameters: ModuleParameters.unflattened(Array(pristine)),
                        verify: .noUnusedKeys)
                    if arm != "rag" {
                        let url = Self.h13AdapterURL(user: user, arm: arm)
                        let weights = try MLX.loadArrays(url: url)
                        try mc.model.update(
                            parameters: ModuleParameters.unflattened(weights),
                            verify: .noUnusedKeys)
                        self.tlog("h13 eval: loaded \(weights.count) adapter tensors (arm=\(arm))")
                    }
                    eval(mc.model)

                    Self.appendH13Marker(
                        ctx, recordType: "eval_start",
                        file: TrainBenchConstants.h13EvalMetricsFileName,
                        extra: [
                            "n_queries": queries.count,
                            "thermal_state": Self.thermalStringH13(),
                            "battery_level": Self.batterySnapshotH13().0,
                        ])
                    let armStart = Date.timeIntervalSinceReferenceDate

                    for (qi, q) in queries.enumerated() {
                        // Seed per query so every arm sees identical randomness
                        // at the same query, and a re-run reproduces exactly.
                        MLXRandom.seed(TrainBenchConstants.h13EvalSeed &+ UInt64(qi))
                        let promptArray = MLXArray(q.ids)
                        let qStart = Date.timeIntervalSinceReferenceDate
                        var iterator = try TokenIterator(
                            input: LMInput(tokens: promptArray), model: mc.model,
                            parameters: params)
                        var outIds: [Int] = []
                        var firstTokenAt: Double = 0
                        while outIds.count < TrainBenchConstants.h13MaxNewTokens,
                            let token = iterator.next()
                        {
                            if outIds.isEmpty {
                                firstTokenAt = Date.timeIntervalSinceReferenceDate
                            }
                            if token == mc.tokenizer.unknownTokenId
                                || stopTokens.contains(token)
                            {
                                break
                            }
                            outIds.append(token)
                        }
                        let qEnd = Date.timeIntervalSinceReferenceDate
                        let text = mc.tokenizer.decode(tokenIds: outIds)

                        Self.appendH13Prediction(
                            url: predURL,
                            record: [
                                "id": q.id, "output": text, "output_ids": outIds,
                                "n_prompt_tokens": q.ids.count,
                                "n_gen_tokens": outIds.count,
                            ])
                        Self.appendH13Query(
                            ctx, queryIndex: qi, queryId: q.id,
                            promptTokens: q.ids.count, genTokens: outIds.count,
                            prefillS: firstTokenAt > 0 ? firstTokenAt - qStart : qEnd - qStart,
                            totalS: qEnd - qStart, elapsed: qEnd - benchStart,
                            peak: Memory.snapshot().peakMemory,
                            thermal: Self.thermalStringH13())
                    }

                    Self.appendH13Marker(
                        ctx, recordType: "eval_end",
                        file: TrainBenchConstants.h13EvalMetricsFileName,
                        extra: [
                            "elapsed_s": Date.timeIntervalSinceReferenceDate - armStart,
                            "thermal_state": Self.thermalStringH13(),
                            "battery_level": Self.batterySnapshotH13().0,
                            "predictions_file": predURL.lastPathComponent,
                        ])
                    self.tlog("h13 eval complete user=\(user) arm=\(arm) "
                        + "n=\(queries.count) "
                        + "s=\(Date.timeIntervalSinceReferenceDate - armStart)")
                }
            }
        } catch {
            evalError = "\(error)"
            tlog("h13 eval error: \(error)")
        }

        tlog("h13 eval ALL DONE user=\(user) arms=\(runnable.joined(separator: ",")) "
            + "error=\(evalError ?? "none")")
        finishTrainBenchmark()
    }


    // MARK: - Checkpoint / resume

    nonisolated static func saveH13Checkpoint(
        dir: URL, model: Module, optimizer: H13AdamW, nextStep: Int
    ) throws {
        let tmp = dir.appendingPathComponent("ckpt_tmp")
        try? FileManager.default.removeItem(at: tmp)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let params = Dictionary(
            uniqueKeysWithValues: model.trainableParameters().flattened())
        try MLX.save(arrays: params, url: tmp.appendingPathComponent("params.safetensors"))
        try MLX.save(arrays: optimizer.m, url: tmp.appendingPathComponent("adam_m.safetensors"))
        try MLX.save(arrays: optimizer.v, url: tmp.appendingPathComponent("adam_v.safetensors"))
        let meta = ["next_step": nextStep, "adam_t": optimizer.t]
        try JSONSerialization.data(withJSONObject: meta)
            .write(to: tmp.appendingPathComponent("state.json"))
        let final = dir.appendingPathComponent("ckpt")
        try? FileManager.default.removeItem(at: final)
        try FileManager.default.moveItem(at: tmp, to: final)
    }

    nonisolated static func loadH13Checkpoint(dir: URL)
        -> ([String: MLXArray], [String: MLXArray], [String: MLXArray], Int, Int)?
    {
        let ck = dir.appendingPathComponent("ckpt")
        guard
            let stateData = try? Data(contentsOf: ck.appendingPathComponent("state.json")),
            let state = try? JSONSerialization.jsonObject(with: stateData) as? [String: Int],
            let nextStep = state["next_step"], let adamT = state["adam_t"],
            let params = try? MLX.loadArrays(url: ck.appendingPathComponent("params.safetensors")),
            let m = try? MLX.loadArrays(url: ck.appendingPathComponent("adam_m.safetensors")),
            let v = try? MLX.loadArrays(url: ck.appendingPathComponent("adam_v.safetensors"))
        else { return nil }
        return (params, m, v, adamT, nextStep)
    }

    // MARK: - Records

    nonisolated static func thermalStringH13() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    nonisolated static func batterySnapshotH13() -> (Double, Bool) {
        #if canImport(UIKit)
            let level = Double(UIDevice.current.batteryLevel)
            let state = UIDevice.current.batteryState
            return (level, state == .charging || state == .full)
        #else
            return (-1, false)
        #endif
    }

    nonisolated static func h13BaseRecord(_ c: H13RunContext, recordType: String)
        -> [String: Any]
    {
        [
            "record_type": recordType,
            "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
            "task": "LaMP_2M",
            "protocol": "oppu_k1_r5",
            "user_id": c.userId,
            "arm": c.arm,
            "condition": c.condition,
            "nax_arm": c.naxArm ?? NSNull(),
            "n_examples": c.nExamples,
            "epochs": c.epochs,
            "total_steps": c.totalSteps,
            "schedule_total_steps": c.scheduleTotalSteps,
            "accum_window": TrainBenchConstants.h13AccumWindow,
            "model": c.modelName,
            "lora_rank": TrainBenchConstants.h13LoraRank,
            "lora_scale": TrainBenchConstants.h13LoraScale,
            "lora_keys": TrainBenchConstants.h13LoraKeysLabel,
            "num_lora_layers": TrainBenchConstants.h13LoraLayers,
            "gradient_checkpointing": TrainBenchConstants.gradientCheckpointing,
            "optimizer": "adamw_decoupled_wd",
            "base_learning_rate": TrainBenchConstants.h13BaseLR,
            "lr_schedule": "cosine_warmup0.03",
            "weight_decay": TrainBenchConstants.h13WeightDecay,
            "grad_clip_norm": TrainBenchConstants.h13GradClipNorm,
            "loss_impl": "sliced_lm_head",
            "decode": "sample_t0.1_topk10_topp0.9_max200",
            "app_build": naxArmAppBuild(TrainBenchConstants.h13AppBuild),
            "bench_schema_version": TrainBenchConstants.h13SchemaVersion,
            "git_commit": TrainBenchConstants.gitCommit,
            "git_dirty": TrainBenchConstants.gitDirty,
            "bench_session_id": c.sessionId,
            "device_model": trainHwModelH13(),
            "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
        ]
    }

    nonisolated static func trainHwModelH13() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        return String(cString: machine)
    }

    nonisolated static func appendH13Marker(
        _ c: H13RunContext, recordType: String, file: String, extra: [String: Any]
    ) {
        var r = h13BaseRecord(c, recordType: recordType)
        for (k, v) in extra { r[k] = v }
        emitH13(r, file: file)
    }

    nonisolated static func appendH13Step(
        _ c: H13RunContext, step: Int, epoch: Int, loss: Float, lr: Float,
        gradNorm: Float, clipScale: Float, nMicro: Int, windowMaskedTokens: Int,
        windowSeqTokens: Int, windowS: Double, elapsed: Double, peak: Int,
        thermal: String
    ) {
        var r = h13BaseRecord(c, recordType: "train_step")
        r["step"] = step
        r["epoch"] = epoch
        r["loss"] = Double(loss)
        r["learning_rate"] = Double(lr)
        r["grad_norm"] = Double(gradNorm)
        r["clip_scale"] = Double(clipScale)
        r["n_micro"] = nMicro
        r["window_masked_tokens"] = windowMaskedTokens
        r["window_seq_tokens"] = windowSeqTokens
        r["window_s"] = windowS
        r["elapsed_s"] = elapsed
        r["peak_mem_bytes"] = peak
        r["thermal_state"] = thermal
        emitH13(r, file: TrainBenchConstants.h13MetricsFileName)
    }

    nonisolated static func appendH13Query(
        _ c: H13RunContext, queryIndex: Int, queryId: String, promptTokens: Int,
        genTokens: Int, prefillS: Double, totalS: Double, elapsed: Double,
        peak: Int, thermal: String
    ) {
        var r = h13BaseRecord(c, recordType: "eval_query")
        r["query_index"] = queryIndex
        r["query_id"] = queryId
        r["n_prompt_tokens"] = promptTokens
        r["n_gen_tokens"] = genTokens
        r["prefill_s"] = prefillS
        r["total_s"] = totalS
        r["decode_tok_s"] = totalS > prefillS && genTokens > 1
            ? Double(genTokens - 1) / (totalS - prefillS) : 0
        r["elapsed_s"] = elapsed
        r["peak_mem_bytes"] = peak
        r["thermal_state"] = thermal
        emitH13(r, file: TrainBenchConstants.h13EvalMetricsFileName)
    }

    nonisolated static func appendH13Sample(
        _ c: H13RunContext, elapsed: Double, level: Double, charging: Bool,
        thermal: String, peak: Int, file: String
    ) {
        var r = h13BaseRecord(c, recordType: "sample")
        r["elapsed_s"] = elapsed
        r["battery_level"] = level
        r["charging"] = charging
        r["thermal_state"] = thermal
        r["peak_mem_bytes"] = peak
        emitH13(r, file: file)
    }

    nonisolated static func appendH13Prediction(url: URL, record: [String: Any]) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        appendLineH13(json + "\n", to: url)
    }

    nonisolated static func emitH13(_ record: [String: Any], file: String) {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: record, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else { return }
        appendLineH13(
            json + "\n",
            to: URL.documentsDirectory.appendingPathComponent(naxArmFileName(file)))
    }

    nonisolated static func appendLineH13(_ line: String, to url: URL) {
        h13FileLock.lock()
        defer { h13FileLock.unlock() }
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
            // best-effort; a failed line must not kill a multi-hour run
        }
    }
}
