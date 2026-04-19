import SwiftUI

@main
struct ChessTutorIOSApp: App {
    // Simulator uses localhost; physical iPhone uses your Mac LAN IP.
    private var appConfig: AppConfig {
        #if targetEnvironment(simulator)
        return .defaultLocal
        #else
        return .defaultLAN
        #endif
    }

    var body: some Scene {
        WindowGroup {
            ContentView(
                viewModel: HomeViewModel(
                    apiClient: APIClient(config: appConfig),
                    connectionService: MockWearablesConnectionService(),
                    cameraService: MockWearablesCameraService(),
                    audioService: MockWearablesAudioService()
                )
            )
        }
    }
}
