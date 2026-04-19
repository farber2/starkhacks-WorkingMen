import Foundation

enum ConnectionState: String {
    case disconnected = "Disconnected"
    case connecting = "Connecting"
    case connected = "Connected"
    case mock = "Mock Connected"
}

struct WearablesCaptureResult {
    let source: String
    let note: String
}

struct WearablesAudioRouteResult {
    let route: String
    let note: String
}

