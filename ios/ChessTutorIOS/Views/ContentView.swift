import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel: HomeViewModel

    init(viewModel: HomeViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 14) {
                    connectionCard
                    actionCard
                    coachCard
                    statusCard
                    infoCard
                }
                .padding()
            }
            .background(
                LinearGradient(
                    colors: [Color(red: 0.12, green: 0.15, blue: 0.19), Color(red: 0.08, green: 0.1, blue: 0.13)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
            )
            .navigationTitle("Chess Tutor Glasses Bridge")
            .task {
                await viewModel.onAppear()
            }
        }
    }

    private var connectionCard: some View {
        CardView(title: "Meta Glasses Connection") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("State")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(viewModel.connectionState.rawValue)
                        .fontWeight(.semibold)
                }
                Toggle("Meta Glasses Mode", isOn: $viewModel.metaModeEnabled)
                    .tint(.green)
                Button(viewModel.connectionState == .disconnected ? "Connect (Mock)" : "Disconnect") {
                    Task { await viewModel.toggleConnection() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isLoading)
            }
        }
    }

    private var actionCard: some View {
        CardView(title: "Actions") {
            VStack(spacing: 10) {
                Button("Capture Board") {
                    Task { await viewModel.captureBoard() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isLoading)

                Button("Get Help") {
                    Task { await viewModel.requestHelp() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isLoading)

                Button("Route Audio to Glasses (Placeholder)") {
                    Task { await viewModel.routeAudioToGlasses() }
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.isLoading)
            }
        }
    }

    private var coachCard: some View {
        CardView(title: "Coach Response") {
            Text(viewModel.coachText)
                .frame(maxWidth: .infinity, alignment: .leading)
                .foregroundStyle(Color.white.opacity(0.94))
        }
    }

    private var statusCard: some View {
        CardView(title: "Status") {
            VStack(alignment: .leading, spacing: 8) {
                Label(viewModel.backendStatusText, systemImage: "network")
                Label(viewModel.cameraStatusText, systemImage: "camera")
                Label(viewModel.audioStatusText, systemImage: "speaker.wave.2")
                Text("Current flow: iPhone app -> FastAPI backend -> local AI/TTS. Future flow: Meta glasses camera/audio bridged via iOS service layer.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var infoCard: some View {
        CardView(title: "Demo Notes") {
            Text(viewModel.infoMessage)
                .frame(maxWidth: .infinity, alignment: .leading)
                .foregroundStyle(.secondary)
        }
    }
}

struct CardView<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.headline)
                .foregroundStyle(.white)
            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.white.opacity(0.08))
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .stroke(Color.white.opacity(0.12), lineWidth: 1)
                )
        )
    }
}

#Preview {
    ContentView(
        viewModel: HomeViewModel(
            apiClient: APIClient(config: .defaultLocal),
            connectionService: MockWearablesConnectionService(),
            cameraService: MockWearablesCameraService(),
            audioService: MockWearablesAudioService()
        )
    )
}
