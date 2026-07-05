import AVFoundation
import AgentSquad

/// `AudioOutput` over `AVAudioEngine` + `AVAudioPlayerNode`: schedules incoming PCM16 @ 24 kHz frames
/// (converted to float) for playback; `flush` stops and clears the queue for an instant barge-in cut.
///
/// `@unchecked Sendable`: not safe to call concurrently — callers must serialize. `RealtimeRuntime`
/// satisfies this by driving every method from its single event pump, so the engine/player are
/// never touched concurrently.
public final class AudioPlayback: AudioOutput, @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private let sessionPolicy: AudioSessionPolicy
    private let configureEngine: (@Sendable (AVAudioEngine) throws -> Void)?
    private var isStarted = false

    /// `configureEngine` runs after the player is attached/connected, before the engine starts —
    /// the escape hatch to any `AVAudioEngine` API AgentSquad doesn't wrap. Do NOT enable voice
    /// processing here: it belongs on the capture engine (`MicCapture`), not playback.
    public init(
        sampleRate: Double = 24_000,
        sessionPolicy: AudioSessionPolicy = .managed,
        configureEngine: (@Sendable (AVAudioEngine) throws -> Void)? = nil
    ) {
        self.format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!   // float32, non-interleaved
        self.sessionPolicy = sessionPolicy
        self.configureEngine = configureEngine
    }

    public func start() async throws {
        guard !isStarted else { return }   // start-once: a second engine.attach(player) would crash
        #if os(iOS)
        try VoiceAudioSession.activate(policy: sessionPolicy)
        #endif
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        do {
            try configureEngine?(engine)
            engine.prepare()
            try engine.start()
        } catch {
            engine.detach(player)   // roll back so a retry's attach(player) doesn't crash
            throw error
        }
        player.play()
        isStarted = true
    }

    public func enqueue(_ pcm16: Data) async {
        guard let buffer = makeBuffer(pcm16) else { return }
        schedule(buffer)
    }

    /// Sync hand-off so the non-async `scheduleBuffer` overload is selected (we must NOT await
    /// playback completion — that would serialize enqueue to real-time playback speed).
    private func schedule(_ buffer: AVAudioPCMBuffer) {
        player.scheduleBuffer(buffer, completionHandler: nil)
    }

    public func flush() async {
        // `stop()` discards all scheduled buffers — the instant barge-in cut — then re-arm for next.
        player.stop()
        player.play()
    }

    public func stop() async {
        player.stop()
        engine.stop()
        isStarted = false
    }

    private func makeBuffer(_ data: Data) -> AVAudioPCMBuffer? {
        PCM16.floatBuffer(fromPCM16: data, format: format)
    }

    /// PCM16 little-endian mono bytes → float32 samples in `[-1, 1]`. Pure (no engine) so the
    /// endianness/scaling — where a silent regression would corrupt all playback — is unit-testable.
    static func floatSamples(fromPCM16 data: Data) -> [Float] {
        let count = data.count / MemoryLayout<Int16>.size
        guard count > 0 else { return [] }
        return data.withUnsafeBytes { raw in
            let ints = raw.bindMemory(to: Int16.self)
            return (0..<count).map { Float(Int16(littleEndian: ints[$0])) / 32_768.0 }
        }
    }
}
