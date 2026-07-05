import AVFoundation
import AgentSquad

/// `AudioInput` over `AVAudioEngine`: taps the mic, converts to PCM16 @ 24 kHz mono, and yields
/// frames into a **bounded, drop-oldest** `AsyncStream` (the real-time tap thread must never block).
///
/// By default the capture runs through Apple's **Voice-Processing I/O** unit — echo cancellation
/// using the speaker signal as hardware reference, noise suppression, AGC. Pass
/// `voiceProcessing: nil` for raw capture. AEC is inert in the simulator; validate on a device.
///
/// `@unchecked Sendable`: `start`/`stop` are not safe to call concurrently (callers must serialize;
/// `RealtimeRuntime` does). The converter lives in the tap closure and is touched only on the audio
/// thread; the stream `continuation` is itself Sendable.
public final class MicCapture: AudioInput, @unchecked Sendable {
    public let frames: AsyncStream<Data>
    private let continuation: AsyncStream<Data>.Continuation
    private let engine = AVAudioEngine()
    private let targetFormat: AVAudioFormat
    private let voiceProcessing: VoiceProcessing?
    private let sessionPolicy: AudioSessionPolicy
    private let configureEngine: (@Sendable (AVAudioEngine) throws -> Void)?
    private var isStarted = false

    /// `maxBufferedFrames` bounds the queue — under back-pressure the oldest frames are dropped.
    /// `configureEngine` runs after voice processing is enabled but before the tap is installed —
    /// the escape hatch to any `AVAudioEngine` API AgentSquad doesn't wrap.
    public init(
        sampleRate: Double = 24_000,
        maxBufferedFrames: Int = 16,
        voiceProcessing: VoiceProcessing? = .default,
        sessionPolicy: AudioSessionPolicy = .managed,
        configureEngine: (@Sendable (AVAudioEngine) throws -> Void)? = nil
    ) {
        self.targetFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: sampleRate, channels: 1, interleaved: true)!
        self.voiceProcessing = voiceProcessing
        self.sessionPolicy = sessionPolicy
        self.configureEngine = configureEngine
        (self.frames, self.continuation) = AsyncStream.makeStream(of: Data.self, bufferingPolicy: .bufferingNewest(maxBufferedFrames))
    }

    public func start() async throws {
        guard !isStarted else { return }   // start-once: a second installTap/engine.start would crash
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
        if let vp = voiceProcessing {
            do {
                // MUST precede the inputFormat read below — enabling VP changes the node's format.
                try input.setVoiceProcessingEnabled(true)
            } catch {
                throw MicCaptureError.voiceProcessingUnavailable(String(describing: error))
            }
            input.isVoiceProcessingAGCEnabled = vp.automaticGainControl
            if #available(iOS 17.0, *) {
                input.voiceProcessingOtherAudioDuckingConfiguration = .init(
                    enableAdvancedDucking: false, duckingLevel: vp.duckingLevel.avLevel)
            }
        }
        // Roll back so a failed start leaves the node raw and a retry starts clean.
        func rollBackVoiceProcessing() {
            if voiceProcessing != nil { try? input.setVoiceProcessingEnabled(false) }
        }

        do {
            try configureEngine?(engine)
        } catch {
            rollBackVoiceProcessing()
            throw error
        }

        let inputFormat = input.inputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            rollBackVoiceProcessing()
            throw MicCaptureError.converterUnavailable
        }
        // Capture the converter into the tap closure, not a shared field: the tap runs on the
        // real-time audio thread (a stored `var` would be a cross-thread read), and the closure
        // being its sole owner means it can't be freed under an in-flight `process` (no UAF on teardown).
        input.installTap(onBus: 0, bufferSize: 2_048, format: inputFormat) { [weak self] buffer, _ in
            self?.process(buffer, with: converter)
        }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)   // roll back so a retry's installTap doesn't hit a live tap
            rollBackVoiceProcessing()
            throw error
        }
        isStarted = true
    }

    public func stop() async {
        // Halt the render thread BEFORE removing the tap so no tap block starts mid-teardown. The
        // converter is owned by the tap closure, not a shared field, so removing the tap can't race
        // a `process` call — an in-flight one holds the converter on its stack.
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        continuation.finish()
        isStarted = false
    }

    /// Runs on the real-time audio thread: convert one captured buffer to PCM16/24k and yield it.
    private func process(_ buffer: AVAudioPCMBuffer, with converter: AVAudioConverter) {
        PCM16.convertAndYield(buffer, converter: converter, targetFormat: targetFormat, continuation: continuation)
    }
}

public enum MicCaptureError: Error, Equatable {
    case permissionDenied
    case converterUnavailable
    /// `setVoiceProcessingEnabled(true)` failed; payload is the underlying error's description.
    /// Catch it and retry with `voiceProcessing: nil` to degrade to raw capture deliberately.
    case voiceProcessingUnavailable(String)
}
