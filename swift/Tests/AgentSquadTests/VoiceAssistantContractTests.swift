import Foundation
import Testing

@testable import AgentSquad

@Suite struct VoiceAssistantContractTests {
    private final class StubSession: VoiceAssistant {
        let modality = RealtimeModality()
        let events: AsyncStream<RealtimeEvent>
        private let continuation: AsyncStream<RealtimeEvent>.Continuation
        init() { (events, continuation) = AsyncStream.makeStream() }
        func emit(_ event: RealtimeEvent) { continuation.yield(event) }
        func start() async throws { continuation.yield(.state(.listening)) }
        func sendAudio(_ pcm16: Data) async {}
        func sendText(_ text: String) async { continuation.yield(.userTranscript(text, final: true)) }
        func interrupt() async { continuation.yield(.audioDone(interrupted: true)) }
        func stop() async { continuation.finish() }
    }

    @Test func sessionStreamsEventsInOrder() async throws {
        let session = StubSession()
        try await session.start()
        await session.sendText("odds?")
        await session.interrupt()
        await session.stop()

        var collected: [RealtimeEvent] = []
        for await event in session.events { collected.append(event) }

        #expect(collected.count == 3)
        if case .state(.listening) = collected[0] {} else { Issue.record("expected .state(.listening)") }
        if case .userTranscript("odds?", true) = collected[1] {} else { Issue.record("expected user transcript") }
        if case .audioDone(true) = collected[2] {} else { Issue.record("expected interrupted audioDone") }
    }

    // Compile-time guard: both concrete actors must declare VoiceAssistant conformance.
    // If the conformance is ever dropped this test will fail to compile.
    @Test func voiceAssistantsConform() {
        func requireVoiceAssistant(_: any VoiceAssistant.Type) {}
        requireVoiceAssistant(OpenAIVoiceAssistant.self)
        requireVoiceAssistant(OpenAIGroundedVoiceAssistant.self)
    }

    // The stream is non-throwing: an in-band `.error` must not end iteration.
    @Test func inBandErrorDoesNotTerminateStream() async throws {
        let session = StubSession()
        try await session.start()
        session.emit(.error(code: nil, message: "transient"))
        session.emit(.state(.ready))
        await session.stop()

        var sawError = false
        var sawStateAfterError = false
        for await event in session.events {
            if case .error = event { sawError = true }
            if sawError, case .state(.ready) = event { sawStateAfterError = true }
        }
        #expect(sawError)
        #expect(sawStateAfterError)
    }
}
