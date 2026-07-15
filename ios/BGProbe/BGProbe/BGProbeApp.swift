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
import SwiftUI

let bgProbeTaskId = "mlx.bgprobe.test"
let bgProbeFileLock = NSLock()

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

    // Trivial work only: a 0.2s-resolution heartbeat loop, no model, no
    // real CPU/GPU load. If this also dies at ~9.5s, the LLMEval-specific
    // resource footprint is ruled out as the cause.
    var tick = 0
    while !Task.isCancelled {
        let elapsed = Date.timeIntervalSinceReferenceDate - wakeStart
        appendProbeLine([
            "event": "heartbeat", "session": sessionId, "elapsed_s": elapsed,
            "tick": tick, "ts": isoNow(),
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
