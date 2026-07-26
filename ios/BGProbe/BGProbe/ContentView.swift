import BackgroundTasks
import SwiftUI
import UIKit

struct ContentView: View {
    var body: some View {
        VStack(spacing: 12) {
            Text("BGProbe")
                .font(.title)
            Text("Minimal BGProcessingTask probe for the h6 grant-size investigation. No UI interaction needed beyond first launch.")
                .font(.caption)
                .multilineTextAlignment(.center)
                .padding()
        }
        .padding()
        .task {
            submitProbeRequest()
            logSupportedResources()
        }
    }

    // h6 pivot check (2026-07-16): before investing in
    // BGContinuedProcessingTaskRequest as the real fix for sustained
    // background GPU access, checked whether THIS device even supports it
    // — external research found no published device list and at least one
    // report of an iPhone 16 Pro Max NOT supporting it, so this is a hard
    // runtime gate, not something to assume. Logged once per foreground
    // launch (cheap, no background wake needed to get an answer).
    private func logSupportedResources() {
        if #available(iOS 26.0, *) {
            let resources = BGTaskScheduler.supportedResources
            appendProbeLine([
                "event": "supported_resources_check",
                "has_gpu": resources.contains(.gpu),
                "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
                "device_model": UIDevice.current.model,
                "ts": isoNow(),
            ])
        } else {
            appendProbeLine([
                "event": "supported_resources_check",
                "has_gpu": false,
                "note": "BGTaskScheduler.supportedResources requires iOS 26+",
                "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
                "ts": isoNow(),
            ])
        }
    }
}
