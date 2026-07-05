import Foundation

/// Microphone capture seam: PCM16 @ 24 kHz mono chunks the `RealtimeRuntime` forwards to the session.
/// Testable without a mic; the AVFoundation implementation (`MicCapture`) lives in `AgentSquadAudio`.
public protocol AudioInput: Sendable {
    /// Captured PCM16 chunks, **bounded drop-oldest** so a slow consumer never blocks the audio thread. Finishes when capture stops.
    var frames: AsyncStream<Data> { get }
    func start() async throws
    func stop() async
}

/// Speaker playback seam: the `RealtimeRuntime` enqueues PCM16 @ 24 kHz frames and flushes on barge-in.
/// AVFoundation implementation (`AudioPlayback`) lives in `AgentSquadAudio`. No error channel yet —
/// runtime errors (route changes, hardware) are handled inside the implementation.
public protocol AudioOutput: Sendable {
    func start() async throws
    /// Queue one PCM16 frame for playback.
    func enqueue(_ pcm16: Data) async
    /// Drop all queued/playing audio immediately (barge-in cut).
    func flush() async
    func stop() async
    /// Milliseconds actually played of the current playback burst (a burst starts when audio is
    /// enqueued onto an empty queue), or `nil` when the implementation can't measure playback.
    /// Must survive `flush()` — the session samples it right after a barge-in cut to send
    /// `conversation.item.truncate`. Default: `nil` (truncation is skipped).
    func playedMilliseconds() async -> Double?
}

public extension AudioOutput {
    func playedMilliseconds() async -> Double? { nil }
}
