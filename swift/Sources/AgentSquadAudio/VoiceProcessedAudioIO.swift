import AVFoundation
import AgentSquad
import os

/// Capture AND playback on **one** `AVAudioEngine` with the Voice-Processing I/O unit enabled:
/// the assistant's audio renders through the same engine's voice-processed output, so it is by
/// construction in the echo canceller's reference path — the configuration Apple's VP unit is
/// designed around. Pass ONE instance as both `input:` and `output:` to `RealtimeRuntime`.
///
/// Prefer this over the separate `MicCapture` + `AudioPlayback` pair for voice sessions; the
/// split pair relies on the device-level echo reference, which is route-dependent.
///
/// `start()`/`stop()` are idempotent — the runtime calls each twice (once per protocol role).
/// One instance serves one session: after `stop()` the frames stream is finished for good;
/// create a new instance to start again.
///
/// `@unchecked Sendable`: `isStarted` and the player/engine calls in `enqueue`/`flush`/`stop` are
/// serialized by an internal unfair lock — actor reentrancy in `RealtimeRuntime` lets an in-flight
/// `enqueue` overlap `stop()` (both roles hit this one object). `start()` itself must not be
/// called concurrently with itself (the runtime awaits it sequentially, before any pump runs).
/// The tap thread never takes the lock: it touches only the closure-owned converter and the
/// Sendable continuation.
public final class VoiceProcessedAudioIO: AudioInput, AudioOutput, @unchecked Sendable {
    public let frames: AsyncStream<Data>
    private let continuation: AsyncStream<Data>.Continuation
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let captureFormat: AVAudioFormat    // PCM16 interleaved mono
    private let playbackFormat: AVAudioFormat   // float32 non-interleaved mono
    private let voiceProcessing: VoiceProcessing
    private let sessionPolicy: AudioSessionPolicy
    private let configureEngine: (@Sendable (AVAudioEngine) throws -> Void)?
    private let lock = OSAllocatedUnfairLock()
    private let clock = PlaybackClock()   // its own lock; callbacks never take `lock`
    private var isStarted = false   // guarded by `lock`

    /// `voiceProcessing` is non-optional here — raw capture defeats this class's purpose; use
    /// `MicCapture(voiceProcessing: nil)` for that. `configureEngine` runs after voice processing
    /// is enabled and the player is wired, before the tap is installed.
    public init(
        sampleRate: Double = 24_000,
        maxBufferedFrames: Int = 16,
        voiceProcessing: VoiceProcessing = .default,
        sessionPolicy: AudioSessionPolicy = .managed,
        configureEngine: (@Sendable (AVAudioEngine) throws -> Void)? = nil
    ) {
        self.captureFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: sampleRate, channels: 1, interleaved: true)!
        self.playbackFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        self.voiceProcessing = voiceProcessing
        self.sessionPolicy = sessionPolicy
        self.configureEngine = configureEngine
        (self.frames, self.continuation) = AsyncStream.makeStream(of: Data.self, bufferingPolicy: .bufferingNewest(maxBufferedFrames))
    }

    public func start() async throws {
        guard lock.withLockUnchecked({ !isStarted }) else { return }   // idempotent: the runtime starts this as output, then as input
        #if os(iOS)
        try VoiceAudioSession.activate(policy: sessionPolicy)
        let granted = await withCheckedContinuation { (c: CheckedContinuation<Bool, Never>) in
            // `AVAudioApplication.requestRecordPermission` is iOS 17+; fall back to the (now-deprecated)
            // `AVAudioSession` call on iOS 16. No deprecation warning fires at the iOS-16 floor.
            if #available(iOS 17, *) {
                AVAudioApplication.requestRecordPermission { c.resume(returning: $0) }
            } else {
                AVAudioSession.sharedInstance().requestRecordPermission { c.resume(returning: $0) }
            }
        }
        guard granted else { throw MicCaptureError.permissionDenied }
        #endif

        let input = engine.inputNode
        do {
            // MUST precede the inputFormat read below — enabling VP changes the node's format.
            try input.setVoiceProcessingEnabled(true)
        } catch {
            throw MicCaptureError.voiceProcessingUnavailable(String(describing: error))
        }
        input.isVoiceProcessingAGCEnabled = voiceProcessing.automaticGainControl
        if #available(iOS 17.0, *) {
            input.voiceProcessingOtherAudioDuckingConfiguration = .init(
                enableAdvancedDucking: false, duckingLevel: voiceProcessing.duckingLevel.avLevel)
        }

        // Playback wired into the SAME engine — this is the whole point: the player's audio goes
        // out through the voice-processed output and lands in the AEC reference.
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: playbackFormat)

        // Roll back so a failed start leaves the engine raw/detached and a retry starts clean.
        func rollBack(tapInstalled: Bool) {
            if tapInstalled { input.removeTap(onBus: 0) }
            engine.detach(player)
            try? input.setVoiceProcessingEnabled(false)
        }

        do {
            try configureEngine?(engine)
        } catch {
            rollBack(tapInstalled: false)
            throw error
        }

        let inputFormat = input.inputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: inputFormat, to: captureFormat) else {
            rollBack(tapInstalled: false)
            throw MicCaptureError.converterUnavailable
        }
        // Converter owned by the tap closure, not a shared field — see MicCapture for the rationale.
        input.installTap(onBus: 0, bufferSize: 2_048, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            PCM16.convertAndYield(buffer, converter: converter, targetFormat: self.captureFormat, continuation: self.continuation)
        }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            rollBack(tapInstalled: true)
            throw error
        }
        player.play()
        lock.withLockUnchecked { isStarted = true }
    }

    public func enqueue(_ pcm16: Data) async {
        guard let buffer = PCM16.floatBuffer(fromPCM16: pcm16, format: playbackFormat) else { return }
        let durationMs = Double(buffer.frameLength) / playbackFormat.sampleRate * 1_000
        lock.withLockUnchecked {
            guard isStarted else { return }   // covers pre-start AND racing/after stop()
            let token = clock.willSchedule()
            // Handler-based (non-async) overload: never await playback. `.dataPlayedBack` feeds the
            // played-ms clock behind `playedMilliseconds()`.
            player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [clock] _ in
                clock.completed(durationMs: durationMs, token: token)
            }
        }
    }

    public func flush() async {
        lock.withLockUnchecked {
            guard isStarted else { return }
            // `stop()` discards all scheduled buffers — the instant barge-in cut — then re-arm for
            // next. The clock keeps its played total (voiding only pending buffers) so the session
            // can still sample it for `conversation.item.truncate`.
            clock.flushed()
            player.stop()
            player.play()
        }
    }

    public func stop() async {
        lock.withLockUnchecked {
            guard isStarted else { return }   // idempotent: the runtime stops this as input, then as output
            isStarted = false
            player.stop()
            // Halt the render thread BEFORE removing the tap so no tap block starts mid-teardown.
            engine.stop()
            engine.inputNode.removeTap(onBus: 0)
            clock.reset()
            continuation.finish()
        }
    }

    public func playedMilliseconds() async -> Double? {
        clock.milliseconds()
    }
}
