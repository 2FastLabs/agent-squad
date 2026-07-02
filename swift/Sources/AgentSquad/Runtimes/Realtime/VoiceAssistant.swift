import Foundation

/// The phase a realtime session is in — drives the UI's status indicator.
public enum RealtimePhase: String, Sendable {
    case idle         // not started
    case ready        // connected, awaiting input (e.g. text mode, which never "listens")
    case listening    // capturing the user's voice
    case thinking     // agent turn: gathering via tools
    case presenting   // grounded presenter speaking from curated data
    case speaking     // direct (no-tool) reply
}

/// The session's fixed input/output modality — tells the consumer which event mix to expect (e.g. `text` output means no `.audio`).
public struct RealtimeModality: Sendable, Equatable {
    public enum Input: Sendable { case speech, text }
    public enum Output: Sendable { case audio, text, audioAndText }
    public let input: Input
    public let output: Output
    public init(input: Input = .speech, output: Output = .audio) {
        self.input = input
        self.output = output
    }

    /// Coarse `modality` trace marker: `"audio"` when both sides speak, `"text"` when neither, else `"mixed"`.
    var traceMarker: String {
        let audioIn = input == .speech
        let audioOut = output != .text
        if audioIn && audioOut { return "audio" }
        if !audioIn && !audioOut { return "text" }
        return "mixed"
    }
}

/// How the server decides the user's turn has ended (the session's `turn_detection`).
public enum RealtimeTurnDetection: Sendable, Equatable {
    /// Model-based end-of-turn detection (the default). `eagerness` trades latency against letting
    /// the user pause mid-thought: `.high` replies sooner, `.low` waits longer; `nil` leaves the
    /// server default (`auto`).
    case semanticVAD(eagerness: Eagerness? = nil)
    /// Silence-based detection with tunable thresholds; `nil` fields keep the server defaults.
    case serverVAD(threshold: Double? = nil, prefixPaddingMs: Int? = nil, silenceDurationMs: Int? = nil)
    /// No automatic turns — only for text-driven sessions (`sendText`, which creates its response
    /// explicitly). Spoken push-to-talk would additionally need a buffer-commit API the framework
    /// doesn't expose yet; with speech input this leaves the session inert.
    case disabled

    public enum Eagerness: String, Sendable {
        case low, medium, high, auto
    }
}

/// An event from a realtime voice session. Its **own** type, deliberately not `AgentEvent`:
/// continuous audio, server-driven turn boundaries and barge-in don't fit the turn-based contract.
/// Errors arrive in-band as `.error`, so the stream is non-throwing.
public enum RealtimeEvent: Sendable {
    case state(RealtimePhase)
    case userTranscript(String, final: Bool)       // what the user said (STT)
    case presenterText(String, final: Bool)         // the spoken reply, as text (audio+text mode)
    case widget(UIPayload)                          // MCP Apps UI for this turn
    /// PCM16 @ 24 kHz to play (only for the response currently playing — the session drops stale audio on `interrupt()`). Drain promptly into a bounded playback channel.
    case audio(Data)
    case audioDone(interrupted: Bool)               // flush playback (barge-in or natural end)
    case error(String)
}

/// A long-lived, bidirectional voice session (e.g. OpenAI Realtime). A peer of `Orchestrator`, not
/// an `AgentProtocol` — it owns its own control loop and reuses only the shared `ToolProvider` /
/// `ChatStorage` / `Tracer` contracts.
public protocol VoiceAssistant: Sendable {
    /// The fixed modality this session was built with (what event mix to expect).
    var modality: RealtimeModality { get }
    /// Open the connection and begin listening.
    func start() async throws
    /// Forward one chunk of mic audio (PCM16 @ 24 kHz). Called continuously while the user speaks.
    func sendAudio(_ pcm16: Data) async
    /// Typed input within the session (when the user types instead of speaking).
    func sendText(_ text: String) async
    /// Barge-in: flush playback, cancel the in-flight response and drop its stale audio, without tearing down the connection.
    func interrupt() async
    /// The session's event stream.
    var events: AsyncStream<RealtimeEvent> { get }
    /// Close the connection.
    func stop() async
}
