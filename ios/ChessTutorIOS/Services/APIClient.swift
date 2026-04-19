import Foundation

final class APIClient {
    private let config: AppConfig
    private let session: URLSession
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()

    init(config: AppConfig, session: URLSession = .shared) {
        self.config = config
        self.session = session
    }

    func getState() async throws -> StateResponse {
        let request = try makeRequest(path: "/state", method: "GET")
        return try await execute(request, decodeAs: StateResponse.self)
    }

    func requestHelp(speak: Bool = false) async throws -> HelpResponse {
        let payload = HelpRequest(speak: speak)
        let request = try makeRequest(path: "/help", method: "POST", body: payload)
        return try await execute(request, decodeAs: HelpResponse.self)
    }

    func postVisionBoard(source: String, simulateFEN: String? = nil) async throws -> VisionBoardResponse {
        let payload = VisionBoardRequest(source: source, simulateFEN: simulateFEN)
        let request = try makeRequest(path: "/vision/board", method: "POST", body: payload)
        return try await execute(request, decodeAs: VisionBoardResponse.self)
    }

    func postGlassesAudio(text: String, route: String = "glasses_future") async throws -> GlassesAudioResponse {
        let payload = GlassesAudioRequest(event: "coach_output", text: text, route: route)
        let request = try makeRequest(path: "/glasses/audio", method: "POST", body: payload)
        return try await execute(request, decodeAs: GlassesAudioResponse.self)
    }

    private func makeRequest<T: Encodable>(
        path: String,
        method: String,
        body: T? = nil
    ) throws -> URLRequest {
        let url = config.backendBaseURL.appendingPathComponent(path.trimmingCharacters(in: CharacterSet(charactersIn: "/")))
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let body {
            request.httpBody = try encoder.encode(body)
        }
        return request
    }

    private func execute<T: Decodable>(_ request: URLRequest, decodeAs: T.Type) async throws -> T {
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        guard (200...299).contains(httpResponse.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw APIError.httpStatus(httpResponse.statusCode, body)
        }
        return try decoder.decode(decodeAs, from: data)
    }
}

enum APIError: LocalizedError {
    case invalidResponse
    case httpStatus(Int, String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid backend response."
        case .httpStatus(let code, let body):
            return "Backend error (\(code)): \(body)"
        }
    }
}

