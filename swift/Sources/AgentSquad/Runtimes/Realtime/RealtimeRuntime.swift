import Foundation

/// Wires a `VoiceAssistant` to audio I/O: forwards mic frames into the session and routes its audio
/// out to playback, flushing on barge-in. The one object an app starts for a voice conversation; it
/// re-broadcasts the session's events on its own `events` stream (it is the sole consumer of
/// `session.events`), so the UI observes transcripts/state/widgets here. Takes pre-built
/// `VoiceAssistant` + `AudioInput`/`AudioOutput` (the AVFoundation ones, or mocks) so it's testable
/// without hardware.
public actor RealtimeRuntime {
    public nonisolated let events: AsyncStream<RealtimeEvent>
    private nonisolated let continuation: AsyncStream<RealtimeEvent>.Continuation

    private let session: any VoiceAssistant
    private let input: any AudioInput
    private let output: any AudioOutput
    private var micPump: Task<Void, Never>?
    private var eventPump: Task<Void, Never>?

    /// The session's fixed modality — tells the UI which event mix to expect (audio / text / both).
    public nonisolated var modality: RealtimeModality { session.modality }

    public init(session: any VoiceAssistant, input: any AudioInput, output: any AudioOutput) {
        self.session = session
        self.input = input
        self.output = output
        (self.events, self.continuation) = AsyncStream.makeStream(of: RealtimeEvent.self)
    }

    public func start() async throws {
        // Wire the output's played-ms clock into the session so barge-in can send
        // `conversation.item.truncate` (WebSocket transports: the client manages playback).
        // Sample THEN cut: a flush re-arms the burst, so a stale pump `enqueue` racing in after
        // it would zero the total — sampling first makes the measurement immune to that, and the
        // cut still lands here, before the session sends the truncate frame (the docs' stop-
        // before-truncate). On server-VAD barge-in this is the earliest possible cut (the
        // session hasn't emitted `.audioDone` yet); on app-initiated interrupt the runtime
        // already flushed and this one is an idempotent repeat.
        await session.setPlaybackClock { [output] in
            let played = await output.playedMilliseconds()
            await output.flush()
            return played
        }
        try await output.start()
        try await session.start()
        try await input.start()
        // Route the session's events to playback + re-broadcast for the UI.
        eventPump = Task { [weak self] in
            guard let self else { return }
            for await event in self.session.events { await self.route(event) }
            self.finish()
        }
        // Forward mic audio into the session.
        micPump = Task { [weak self] in
            guard let self else { return }
            for await frame in self.input.frames { await self.session.sendAudio(frame) }
        }
    }

    /// Typed input within the session (barge-in is handled inside the session on a typed turn).
    public func sendText(_ text: String) async { await session.sendText(text) }

    /// Explicit barge-in (e.g. a stop button) — cuts playback and stops the in-flight response.
    public func interrupt() async {
        // Flush eagerly for latency, and because a custom session without a playback clock never
        // flushes on its own. The session's clock closure (sample + flush) and the
        // `.audioDone(interrupted: true)` route are idempotent repeats. Keep all three.
        await output.flush()
        await session.interrupt()
    }

    public func stop() async {
        // Tear down producer-first so each pump's `for await` ends (on its stream finishing) and
        // drains before the next layer stops — otherwise an in-flight `sendAudio`/`enqueue` could
        // run against an already-stopped session/engine. Awaiting a pump before finishing its source
        // stream would deadlock, hence this ordering.
        micPump?.cancel(); eventPump?.cancel()
        await input.stop()        // finishes `input.frames` → micPump loop ends
        await micPump?.value      // no `sendAudio` in flight now
        await session.stop()      // finishes `session.events` → eventPump loop ends
        await eventPump?.value    // no `route`/`enqueue` in flight now
        await output.stop()
        micPump = nil
        eventPump = nil
        finish()   // idempotent with the eventPump's own finish() when session.events ended
    }

    // MARK: - Internals

    private func route(_ event: RealtimeEvent) async {
        switch event {
        case .audio(let pcm16):
            await output.enqueue(pcm16)
        case .audioDone(let interrupted):
            // Flush only on barge-in; a clean end lets the already-queued audio play out.
            if interrupted { await output.flush() }
        default:
            break
        }
        continuation.yield(event)
    }

    private nonisolated func finish() { continuation.finish() }
}
