// Copyright © 2024 Apple Inc.

import BackgroundTasks
import SwiftUI

@main
struct LLMEvalApp: App {
    // h6 background-scheduled training (see LLMEvaluator+BGTrain.swift).
    // SwiftUI's `.backgroundTask` scene modifier only covers
    // `.appRefresh`/`.urlSession` — there is no `.processing` case, so
    // `BGProcessingTaskRequest` still needs the traditional
    // `BGTaskScheduler.register(forTaskWithIdentifier:using:launchHandler:)`
    // API. Per Apple's docs this must be registered before the app finishes
    // launching; `init()` is the standard place for a SwiftUI `App` with no
    // `AppDelegate`.
    init() {
        // `register` returns NO (silently, no exception here) if the
        // identifier isn't present in the built app's
        // BGTaskSchedulerPermittedIdentifiers Info.plist array — that failure
        // otherwise only surfaces later as a cryptic OS-level crash from
        // `_simulateLaunchForTaskWithIdentifier:` ("No launch handler
        // registered"). Fail loudly here instead, in debug builds.
        let registered = BGTaskScheduler.shared.register(
            forTaskWithIdentifier: TrainBenchConstants.bgTaskIdentifier, using: nil
        ) { task in
            LLMEvaluator.handleBGTrainTask(task as! BGProcessingTask)
        }
        assert(
            registered,
            "BGTaskScheduler.register failed for \(TrainBenchConstants.bgTaskIdentifier) — "
                + "check BGTaskSchedulerPermittedIdentifiers in the built app's Info.plist "
                + "(stale derived data is a common cause; Clean Build Folder and retry).")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(DeviceStat())
        }
    }
}
