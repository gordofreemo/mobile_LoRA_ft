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
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: TrainBenchConstants.bgTaskIdentifier, using: nil
        ) { task in
            LLMEvaluator.handleBGTrainTask(task as! BGProcessingTask)
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(DeviceStat())
        }
    }
}
