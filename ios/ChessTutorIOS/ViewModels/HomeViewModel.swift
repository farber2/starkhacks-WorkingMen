import Foundation

@MainActor
final class HomeViewModel: ObservableObject {
    @Published var connectionState: ConnectionState = .disconnected
    @Published var coachText: String = "Tap Get Help to fetch coaching from your Python backend."
    @Published var backendStatusText: String = "Unknown"
    @Published var cameraStatusText: String = "Manual mode active"
    @Published var audioStatusText: String = "Local speaker route"
    @Published var infoMessage: String = "Ready."
    @Published var isLoading: Bool = false
    @Published var metaModeEnabled: Bool = true

    private let apiClient: APIClient
    private let connectionService: WearablesConnectionService
    private let cameraService: WearablesCameraService
    private let audioService: WearablesAudioService

    init(
        apiClient: APIClient,
        connectionService: WearablesConnectionService,
        cameraService: WearablesCameraService,
        audioService: WearablesAudioService
    ) {
        self.apiClient = apiClient
        self.connectionService = connectionService
        self.cameraService = cameraService
        self.audioService = audioService
    }

    func onAppear() async {
        await refreshState()
    }

    func toggleConnection() async {
        isLoading = true
        defer { isLoading = false }
        do {
            if connectionState == .connected || connectionState == .mock {
                await connectionService.disconnect()
                connectionState = .disconnected
                infoMessage = "Meta glasses disconnected."
            } else {
                connectionState = try await connectionService.connect()
                infoMessage = "Meta glasses mock session is active."
            }
        } catch {
            connectionState = .disconnected
            infoMessage = "Connection failed: \(error.localizedDescription)"
        }
    }

    func refreshState() async {
        do {
            let state = try await apiClient.getState()
            updateFromBackend(status: state.status, fallbackCoachText: state.latestCoachText)
        } catch {
            backendStatusText = "Offline"
            infoMessage = "Could not reach backend: \(error.localizedDescription)"
        }
    }

    func captureBoard() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let capture = try await cameraService.captureBoardFrame()
            cameraStatusText = capture.note
            let response = try await apiClient.postVisionBoard(source: capture.source)
            infoMessage = response.message ?? "Vision placeholder call completed."
        } catch {
            infoMessage = "Capture failed: \(error.localizedDescription)"
        }
    }

    func requestHelp() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let help = try await apiClient.requestHelp(speak: false)
            let text = (help.coachRecommendation?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)
                ? help.coachRecommendation!
                : (help.latestCoachText ?? "No coach text returned.")
            coachText = text
            updateFromBackend(status: help.status, fallbackCoachText: help.latestCoachText)
            infoMessage = "Help recommendation updated."
        } catch {
            infoMessage = "Help request failed: \(error.localizedDescription)"
        }
    }

    func routeAudioToGlasses() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let local = try await audioService.routeCoachAudio(text: coachText)
            let backend = try await apiClient.postGlassesAudio(text: coachText, route: local.route)
            audioStatusText = backend.message ?? local.note
            infoMessage = "Audio route placeholder synced with backend."
        } catch {
            infoMessage = "Audio route failed: \(error.localizedDescription)"
        }
    }

    private func updateFromBackend(status: BackendStatus?, fallbackCoachText: String?) {
        if let status {
            let engine = status.engine ?? "unknown"
            let ai = status.localAI ?? "unknown"
            let tts = status.tts ?? "unknown"
            backendStatusText = "Engine: \(engine), AI: \(ai), TTS: \(tts)"
        } else {
            backendStatusText = "Status unavailable"
        }

        if let fallbackCoachText, !fallbackCoachText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            coachText = fallbackCoachText
        }
    }
}

