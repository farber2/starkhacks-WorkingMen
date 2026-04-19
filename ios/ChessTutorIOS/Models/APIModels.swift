import Foundation

struct BackendStatus: Decodable {
    let engine: String?
    let localAI: String?
    let tts: String?

    enum CodingKeys: String, CodingKey {
        case engine
        case localAI = "local_ai"
        case tts
    }
}

struct StateResponse: Decodable {
    let latestCoachText: String?
    let status: BackendStatus?

    enum CodingKeys: String, CodingKey {
        case latestCoachText = "latest_coach_text"
        case status
    }
}

struct HelpRequest: Encodable {
    let speak: Bool
}

struct HelpResponse: Decodable {
    let coachRecommendation: String?
    let latestCoachText: String?
    let status: BackendStatus?

    enum CodingKeys: String, CodingKey {
        case coachRecommendation = "coach_recommendation"
        case latestCoachText = "latest_coach_text"
        case status
    }
}

struct VisionBoardRequest: Encodable {
    let source: String
    let simulateFEN: String?

    enum CodingKeys: String, CodingKey {
        case source
        case simulateFEN = "simulate_fen"
    }
}

struct VisionBoardResponse: Decodable {
    let ok: Bool
    let message: String?
    let recognizedFEN: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case message
        case recognizedFEN = "recognized_fen"
    }
}

struct GlassesAudioRequest: Encodable {
    let event: String
    let text: String
    let route: String
}

struct GlassesAudioResponse: Decodable {
    let ok: Bool
    let message: String?
    let route: String?
}

