import Foundation
import Testing

@testable import AgentSquad

@Suite struct RealtimeRuntimeTests {
    // MARK: - Mocks

    private final class MockSession: VoiceAssistant, @unchecked Sendable {
        nonisolated let modality = RealtimeModality()
        let events: AsyncStream<RealtimeEvent>
        private let cont: AsyncStream<RealtimeEvent>.Continuation
        private let lock = NSLock()
        private var _sentAudio: [Data] = []
        private var _sentText: [String] = []
        private var _interrupts = 0
        private var _stopped = false
        private var _clock: (@Sendable () async -> Double?)?

        init() { (events, cont) = AsyncStream.makeStream(of: RealtimeEvent.self) }
        var sentAudio: [Data] { lock.withLock { _sentAudio } }
        var sentText: [String] { lock.withLock { _sentText } }
        var interrupts: Int { lock.withLock { _interrupts } }
        var stopped: Bool { lock.withLock { _stopped } }
        var clock: (@Sendable () async -> Double?)? { lock.withLock { _clock } }

        func start() async throws {}
        func sendAudio(_ pcm16: Data) async { lock.withLock { _sentAudio.append(pcm16) } }
        func sendText(_ text: String) async { lock.withLock { _sentText.append(text) } }
        func interrupt() async { lock.withLock { _interrupts += 1 }; cont.yield(.audioDone(interrupted: true)) }
        func stop() async { lock.withLock { _stopped = true }; cont.finish() }
        func setPlaybackClock(_ playedMilliseconds: @escaping @Sendable () async -> Double?) async {
            lock.withLock { _clock = playedMilliseconds }
        }
        func push(_ event: RealtimeEvent) { cont.yield(event) }
    }

    private final class MockInput: AudioInput, @unchecked Sendable {
        let frames: AsyncStream<Data>
        private let cont: AsyncStream<Data>.Continuation
        private let lock = NSLock()
        private var _stopped = false
        init() { (frames, cont) = AsyncStream.makeStream(of: Data.self) }
        var stopped: Bool { lock.withLock { _stopped } }
        func start() async throws {}
        func stop() async { lock.withLock { _stopped = true }; cont.finish() }
        func push(_ data: Data) { cont.yield(data) }
    }

    private final class MockOutput: AudioOutput, @unchecked Sendable {
        private let lock = NSLock()
        private var _enqueued: [Data] = []
        private var _flushes = 0
        private var _stopped = false
        private var _calls: [String] = []   // ordering assertions (flush vs played)
        var enqueued: [Data] { lock.withLock { _enqueued } }
        var flushes: Int { lock.withLock { _flushes } }
        var stopped: Bool { lock.withLock { _stopped } }
        var calls: [String] { lock.withLock { _calls } }
        func start() async throws {}
        func enqueue(_ pcm16: Data) async { lock.withLock { _enqueued.append(pcm16) } }
        func flush() async { lock.withLock { _flushes += 1; _calls.append("flush") } }
        func stop() async { lock.withLock { _stopped = true } }
        func playedMilliseconds() async -> Double? { lock.withLock { _calls.append("played") }; return 42 }
    }

    private final class EventLog: @unchecked Sendable {
        private let lock = NSLock()
        private var _events: [RealtimeEvent] = []
        private var _finished = false
        func start(_ runtime: RealtimeRuntime) {
            Task {
                for await e in runtime.events { self.lock.withLock { self._events.append(e) } }
                self.lock.withLock { self._finished = true }
            }
        }
        var finished: Bool { lock.withLock { _finished } }
        var states: [RealtimePhase] { lock.withLock { _events }.compactMap { if case .state(let p) = $0 { return p } else { return nil } } }
        var transcripts: [String] { lock.withLock { _events }.compactMap { if case .userTranscript(let t, _) = $0 { return t } else { return nil } } }
        /// Short tags in arrival order, for ordering assertions.
        var tags: [String] {
            lock.withLock { _events }.map {
                switch $0 {
                case .state: return "state"
                case .audio: return "audio"
                case .userTranscript: return "transcript"
                case .presenterText: return "presenter"
                case .widget: return "widget"
                case .audioDone: return "audioDone"
                case .error: return "error"
                }
            }
        }
    }

    // MARK: - Helpers

    private func eventually(_ condition: @escaping @Sendable () -> Bool, within seconds: Double = 2) async {
        let deadline = ContinuousClock().now + .seconds(seconds)
        while ContinuousClock().now < deadline {
            if condition() { return }
            try? await Task.sleep(for: .milliseconds(3))
        }
    }

    private func make() -> (RealtimeRuntime, MockSession, MockInput, MockOutput) {
        let s = MockSession(); let i = MockInput(); let o = MockOutput()
        return (RealtimeRuntime(session: s, input: i, output: o), s, i, o)
    }

    // MARK: - Tests

    @Test func installedPlaybackClockSamplesThenCutsPlayback() async throws {
        // Sample BEFORE the flush: a flush re-arms the burst, so a stale pump enqueue racing in
        // after it would zero the total. The cut still happens inside the closure — before the
        // session sends the truncate frame (the docs' stop-before-truncate).
        let (runtime, session, _, output) = make()
        try await runtime.start()
        let clock = try #require(session.clock)
        let played = await clock()
        #expect(played == 42)
        #expect(output.calls == ["played", "flush"])
    }

    @Test func forwardsMicFramesToSession() async throws {
        let (runtime, session, input, _) = make()
        try await runtime.start()
        input.push(Data([1, 2]))
        input.push(Data([3, 4]))
        await eventually { session.sentAudio.count == 2 }
        #expect(session.sentAudio == [Data([1, 2]), Data([3, 4])])
    }

    @Test func routesSessionAudioToPlayback() async throws {
        let (runtime, session, _, output) = make()
        try await runtime.start()
        session.push(.audio(Data([9])))
        await eventually { output.enqueued == [Data([9])] }
    }

    @Test func bargeInAudioDoneFlushesPlayback() async throws {
        let (runtime, session, _, output) = make()
        try await runtime.start()
        session.push(.audioDone(interrupted: true))
        await eventually { output.flushes == 1 }
    }

    @Test func cleanAudioDoneDoesNotFlush() async throws {
        let (runtime, session, _, output) = make()
        let log = EventLog(); log.start(runtime)
        try await runtime.start()
        session.push(.audioDone(interrupted: false))
        session.push(.state(.listening))
        await eventually { log.states.contains(.listening) }   // event was processed…
        #expect(output.flushes == 0)                           // …without flushing (audio plays out)
    }

    @Test func reBroadcastsSessionEventsForTheUI() async throws {
        let (runtime, session, _, _) = make()
        let log = EventLog(); log.start(runtime)
        try await runtime.start()
        session.push(.state(.presenting))
        session.push(.userTranscript("hi", final: true))
        await eventually { log.states.contains(.presenting) && log.transcripts.contains("hi") }
    }

    @Test func interruptFlushesEagerlyThenAgainViaSession() async throws {
        // Eager flush (for latency) + the session's `.audioDone(interrupted:true)` routed flush —
        // an intended, idempotent double-flush. Both are kept on purpose.
        let (runtime, session, _, output) = make()
        try await runtime.start()
        await runtime.interrupt()
        #expect(session.interrupts == 1)
        await eventually { output.flushes == 2 }
    }

    @Test func stopEndsTheRuntimeEventStream() async throws {
        let (runtime, _, _, _) = make()
        let log = EventLog(); log.start(runtime)
        try await runtime.start()
        await runtime.stop()
        await eventually { log.finished }
    }

    @Test func micFramesStopBeingForwardedAfterStop() async throws {
        let (runtime, session, input, _) = make()
        try await runtime.start()
        input.push(Data([1]))
        await eventually { session.sentAudio.count == 1 }
        await runtime.stop()
        input.push(Data([2]))   // after stop — must not reach the session
        try? await Task.sleep(for: .milliseconds(20))
        #expect(session.sentAudio == [Data([1])])
    }

    @Test func preservesEventOrderIncludingInterleavedAudio() async throws {
        let (runtime, session, _, output) = make()
        let log = EventLog(); log.start(runtime)
        try await runtime.start()
        session.push(.audio(Data([0xA])))
        session.push(.state(.presenting))
        session.push(.audio(Data([0xB])))
        session.push(.userTranscript("x", final: true))
        await eventually { log.tags.count == 4 }
        #expect(log.tags == ["audio", "state", "audio", "transcript"])   // order preserved
        #expect(output.enqueued == [Data([0xA]), Data([0xB])])           // audio not reordered/dropped
    }

    @Test func sendTextForwardsToSession() async throws {
        let (runtime, session, _, _) = make()
        try await runtime.start()
        await runtime.sendText("hello")
        #expect(session.sentText == ["hello"])
    }

    @Test func stopTearsDownSessionInputAndOutput() async throws {
        let (runtime, session, input, output) = make()
        try await runtime.start()
        await runtime.stop()
        #expect(session.stopped)
        #expect(input.stopped)
        #expect(output.stopped)
    }
}
