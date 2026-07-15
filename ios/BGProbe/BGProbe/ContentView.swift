import SwiftUI

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
        }
    }
}
