import Foundation

protocol WearablesAudioService {
    func routeCoachAudio(text: String) async throws -> WearablesAudioRouteResult
}

final class MockWearablesAudioService: WearablesAudioService {
    func routeCoachAudio(text: String) async throws -> WearablesAudioRouteResult {
        // Future Meta iOS integration:
        // 1) Bind to glasses audio output stream/session.
        // 2) Stream synthesized coaching audio to glasses speakers.
        try await Task.sleep(nanoseconds: 120_000_000)
        return WearablesAudioRouteResult(
            route: "glasses_future",
            note: text.isEmpty
                ? "No coach text yet. Audio route is ready."
                : "Prepared future glasses audio route for current coach response."
        )
    }
}

