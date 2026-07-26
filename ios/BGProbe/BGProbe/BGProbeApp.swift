// Minimal, standalone BGProcessingTask probe — NOT part of the LLMEval
// harness. Built to answer one question directly: are the ~9.5s
// BGProcessingTask grants seen in the h6 background-training round specific
// to LLMEval's heavy footprint (loading a ~1.7GB 4-bit model + LoRA/GC
// setup), or a platform/device-level ceiling that also hits a near-zero
// footprint app? This app does no model loading, no heavy allocation, no
// sustained CPU work — just registers the task and sleeps in a heartbeat
// loop, logging exactly how long it survives each wake.
//
// Deliberately a SEPARATE Xcode project (not a new target bolted onto
// mlx-swift-examples.xcodeproj) so it has zero risk of touching the live,
// multi-day h6 experiment, and so it has its own clean scheduling history
// with iOS (no prior silent-death wakes to bias any "trust" heuristic).

import BackgroundTasks
import Foundation
import Metal
import SwiftUI

let bgProbeTaskId = "mlx.bgprobe.test"
let bgProbeFileLock = NSLock()

// h6 investigation follow-up (2026-07-16): BGProbe's original heartbeat-only
// loop proved the OS grants 240s+ to a near-zero-footprint app at the SAME
// wake instants LLMEval's real GPU-compute work dies within ~1-30s. That
// ruled out a platform-level grant-length ceiling but never tested whether
// *any* real GPU submission — independent of LLMEval's model/LoRA/training
// specifics — is itself the trigger. External research (2026-07-16) found a
// real, current iOS behavior: a backgrounded app's in-flight Metal command
// buffer can have its GPU access revoked mid-flight
// (`MTLCommandBufferErrorDomain Code=8 'accessRevoked'`), and on iOS 26.2+
// this is reported to cause a hard process abort rather than a graceful,
// catchable error — plain `BGProcessingTaskRequest` (what both BGProbe and
// LLMEval use) has no background-GPU resource assertion mechanism at all
// (that's `BGContinuedProcessingTaskRequest`'s job, a different,
// foreground-initiated API). Raw Metal (not MLX) deliberately, to keep this
// a minimal, dependency-free, LLMEval-independent test of the platform
// mechanism itself.
let mtlDevice = MTLCreateSystemDefaultDevice()
let mtlQueue = mtlDevice?.makeCommandQueue()

/// One trivial real GPU submission: allocate a small shared buffer,
/// blit-fill it, commit, wait for completion. If iOS revokes GPU access
/// mid-command-buffer as a hard abort, this call (or the process itself)
/// may never return — in that case the heartbeat log will simply stop
/// mid-tick, the same silent-death signature seen in every LLMEval death.
/// If it fails gracefully instead, `cmdBuffer.error`'s description is
/// logged directly, which would name the exact Metal error (e.g.
/// `accessRevoked`) rather than leaving it inferred.
func runTrivialGPUOp() -> (success: Bool, detail: String) {
    guard let device = mtlDevice else { return (false, "no MTLDevice") }
    guard let queue = mtlQueue else { return (false, "no command queue") }
    guard let buffer = device.makeBuffer(length: 4096, options: .storageModeShared) else {
        return (false, "no buffer")
    }
    guard let cmdBuffer = queue.makeCommandBuffer() else {
        return (false, "no command buffer")
    }
    guard let blit = cmdBuffer.makeBlitCommandEncoder() else {
        return (false, "no blit encoder")
    }
    blit.fill(buffer: buffer, range: 0..<4096, value: 7)
    blit.endEncoding()
    cmdBuffer.commit()
    cmdBuffer.waitUntilCompleted()
    if let error = cmdBuffer.error {
        return (false, "\(error)")
    }
    return (true, "status=\(cmdBuffer.status.rawValue)")
}

@main
struct BGProbeApp: App {
    init() {
        let ok = BGTaskScheduler.shared.register(
            forTaskWithIdentifier: bgProbeTaskId, using: nil
        ) { task in
            handleProbeTask(task as! BGProcessingTask)
        }
        assert(ok, "BGTaskScheduler.register failed for \(bgProbeTaskId)")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

func handleProbeTask(_ task: BGProcessingTask) {
    let work = Task {
        await runProbeWake()
    }
    task.expirationHandler = {
        work.cancel()
    }
    Task {
        _ = await work.value
        task.setTaskCompleted(success: true)
    }
}

func submitProbeRequest() {
    let request = BGProcessingTaskRequest(identifier: bgProbeTaskId)
    request.requiresExternalPower = true
    request.requiresNetworkConnectivity = false
    do {
        try BGTaskScheduler.shared.submit(request)
        appendProbeLine(["event": "submit", "ok": true, "ts": isoNow()])
    } catch {
        appendProbeLine(["event": "submit", "ok": false, "error": "\(error)", "ts": isoNow()])
    }
}

func runProbeWake() async {
    // Re-arm first, same discipline as the real training harness — a wake
    // that dies must not silently end the chain.
    submitProbeRequest()

    let wakeStart = Date.timeIntervalSinceReferenceDate
    let sessionId = UUID().uuidString
    appendProbeLine(["event": "wake_start", "session": sessionId, "ts": isoNow()])

    // Heartbeat loop, now with one trivial real Metal GPU submission per
    // tick (see `runTrivialGPUOp` above) — isolates whether real GPU work
    // itself (not LLMEval's model/LoRA/training specifics) is what
    // triggers termination during a plain BGProcessingTask wake.
    var tick = 0
    while !Task.isCancelled {
        let elapsed = Date.timeIntervalSinceReferenceDate - wakeStart
        let gpuResult = runTrivialGPUOp()
        appendProbeLine([
            "event": "heartbeat", "session": sessionId, "elapsed_s": elapsed,
            "tick": tick, "ts": isoNow(),
            "gpu_op_success": gpuResult.success, "gpu_op_detail": gpuResult.detail,
        ])
        tick += 1
        if elapsed > 240 { break }  // defensive cap, well above anything observed so far
        try? await Task.sleep(nanoseconds: 200_000_000)
    }

    let elapsed = Date.timeIntervalSinceReferenceDate - wakeStart
    appendProbeLine([
        "event": "wake_end", "session": sessionId, "elapsed_s": elapsed,
        "cancelled": Task.isCancelled, "ts": isoNow(),
    ])
}

func isoNow() -> String { ISO8601DateFormatter().string(from: Date()) }

func appendProbeLine(_ record: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
        let json = String(data: data, encoding: .utf8)
    else { return }
    let line = json + "\n"
    let url = URL.documentsDirectory.appendingPathComponent("bgprobe.jsonl")
    bgProbeFileLock.lock()
    defer { bgProbeFileLock.unlock() }
    if FileManager.default.fileExists(atPath: url.path) {
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            try? handle.seekToEnd()
            if let d = line.data(using: .utf8) { try? handle.write(contentsOf: d) }
        }
    } else {
        try? line.write(to: url, atomically: true, encoding: .utf8)
    }
}
