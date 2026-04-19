import Foundation

protocol WearablesCameraService {
    func captureBoardFrame() async throws -> WearablesCaptureResult
}

final class MockWearablesCameraService: WearablesCameraService {
    func captureBoardFrame() async throws -> WearablesCaptureResult {
        // Future Meta iOS integration:
        // 1) Request latest frame from Meta Wearables Device Access Toolkit.
        // 2) Convert frame -> backend payload for /vision/board.
        try await Task.sleep(nanoseconds: 150_000_000)
        return WearablesCaptureResult(
            source: "meta_glasses_mock_camera",
            note: "Simulated glasses board capture completed."
        )
    }
}

