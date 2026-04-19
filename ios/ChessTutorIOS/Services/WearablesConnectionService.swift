import Foundation

protocol WearablesConnectionService {
    var state: ConnectionState { get }
    func connect() async throws -> ConnectionState
    func disconnect() async
}

final class MockWearablesConnectionService: WearablesConnectionService {
    private(set) var state: ConnectionState = .disconnected

    func connect() async throws -> ConnectionState {
        state = .connecting
        try await Task.sleep(nanoseconds: 200_000_000)
        state = .mock
        return state
    }

    func disconnect() async {
        state = .disconnected
    }
}

