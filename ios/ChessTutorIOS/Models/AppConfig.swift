import Foundation

struct AppConfig {
    let backendBaseURL: URL
    let useMockWearables: Bool

    // For iOS simulator on the same Mac, localhost works.
    static let defaultLocal = AppConfig(
        backendBaseURL: URL(string: "http://127.0.0.1:8000")!,
        useMockWearables: true
    )

    // For a physical iPhone, use your Mac's LAN IP on the same Wi-Fi.
    // Example: http://192.168.1.42:8000
    static let defaultLAN = AppConfig(
        backendBaseURL: URL(string: "http://192.168.4.32:8000")!,
        useMockWearables: true
    )
}
