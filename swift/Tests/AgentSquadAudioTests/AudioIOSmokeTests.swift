import AVFoundation
import Foundation
import Testing

@testable import AgentSquadAudio
import AgentSquad

/// The AVFoundation impls can't be meaningfully unit-tested (mic/speaker/engine). These only confirm
/// they construct and conform to the seams without touching hardware (no `start()`).
@Suite struct AudioIOSmokeTests {
    @Test func micCaptureConstructsAndExposesFrames() {
        let mic = MicCapture()
        let input: any AudioInput = mic
        // The frames stream exists pre-start; iterating it isn't needed for this smoke check.
        _ = input.frames
    }

    @Test func audioPlaybackConstructs() {
        let playback = AudioPlayback()
        let _: any AudioOutput = playback
    }

    @Test func micCaptureErrorIsEquatable() {
        #expect(MicCaptureError.permissionDenied == .permissionDenied)
        #expect(MicCaptureError.permissionDenied != .converterUnavailable)
        #expect(MicCaptureError.voiceProcessingUnavailable("x") == .voiceProcessingUnavailable("x"))
        #expect(MicCaptureError.voiceProcessingUnavailable("x") != .voiceProcessingUnavailable("y"))
    }

    @Test func voiceProcessingDefaults() {
        #expect(VoiceProcessing.default == VoiceProcessing())
        #expect(VoiceProcessing.default.automaticGainControl)
        #expect(VoiceProcessing.default.duckingLevel == .default)
        #expect(VoiceProcessing(automaticGainControl: false, duckingLevel: .min) != .default)
    }

    @Test func micCaptureConstructsAcrossConfigurations() {
        // Raw capture, external session, engine hook — constructing must not touch hardware.
        _ = MicCapture(voiceProcessing: nil)
        _ = MicCapture(sessionPolicy: .external, configureEngine: { _ in })
        _ = MicCapture(voiceProcessing: .init(automaticGainControl: false, duckingLevel: .max))
        #if os(iOS)
        _ = MicCapture(sessionPolicy: .custom { _ in })
        #endif
    }

    @Test func audioPlaybackConstructsAcrossConfigurations() {
        _ = AudioPlayback(sessionPolicy: .external)
        _ = AudioPlayback(sessionPolicy: .managed, configureEngine: { _ in })
    }

    @Test func voiceProcessedAudioIOServesBothRoles() {
        let io = VoiceProcessedAudioIO()
        let input: any AudioInput = io
        let output: any AudioOutput = io
        _ = input.frames
        _ = output
        _ = VoiceProcessedAudioIO(voiceProcessing: .init(duckingLevel: .min), sessionPolicy: .external, configureEngine: { _ in })
    }

    @Test func voiceProcessedAudioIOIsInertBeforeStart() async {
        // enqueue/flush/stop before start must be safe no-ops (the runtime can race teardown).
        let io = VoiceProcessedAudioIO()
        await io.enqueue(Data([0x00, 0x00]))
        await io.flush()
        await io.stop()
    }

    @Test func playbackClockAccumulatesPlayedBuffers() {
        let clock = PlaybackClock()
        let t1 = clock.willSchedule()
        let t2 = clock.willSchedule()
        clock.completed(durationMs: 40, token: t1)
        clock.completed(durationMs: 60, token: t2)
        #expect(clock.milliseconds() == 100)
    }

    @Test func playbackClockSurvivesFlushAndVoidsPendingBuffers() {
        // The realistic barge-in: deltas outpace playback, so the queue still holds buffers at the cut.
        let clock = PlaybackClock()
        let played = clock.willSchedule()
        let pending = clock.willSchedule()   // still queued when the user barges in
        clock.completed(durationMs: 40, token: played)
        clock.flushed()
        // player.stop() fires the pending buffer's callback too — it must not count.
        clock.completed(durationMs: 500, token: pending)
        #expect(clock.milliseconds() == 40)   // the played total survives the flush for sampling
    }

    @Test func staleEnqueueAfterFlushZeroesTheTotal() {
        // Why the runtime's clock closure samples BEFORE it flushes: a flush re-arms the burst,
        // and any stale enqueue racing in afterwards starts a new one, wiping the played total.
        let clock = PlaybackClock()
        let t = clock.willSchedule()
        clock.completed(durationMs: 40, token: t)
        clock.flushed()
        #expect(clock.milliseconds() == 40)   // survives the flush…
        _ = clock.willSchedule()              // …until a stale enqueue lands
        #expect(clock.milliseconds() == 0)
    }

    @Test func playbackClockUnderReportsAfterAMidResponseStall() {
        // Queue drains mid-response (stall) → the next buffer starts a new burst and the total
        // restarts. Deliberate: truncation then under-reports, which never claims unheard audio
        // was played (over-reporting is the server-error case).
        let clock = PlaybackClock()
        let t1 = clock.willSchedule()
        clock.completed(durationMs: 40, token: t1)   // drained
        let t2 = clock.willSchedule()                // stalled response resumes
        clock.completed(durationMs: 10, token: t2)
        #expect(clock.milliseconds() == 10)
    }

    @Test func playbackClockStartsANewBurstAfterDrainOrFlush() {
        let clock = PlaybackClock()
        let t1 = clock.willSchedule()
        clock.completed(durationMs: 40, token: t1)         // queue drained → idle
        let t2 = clock.willSchedule()                       // new burst → total resets
        clock.completed(durationMs: 10, token: t2)
        #expect(clock.milliseconds() == 10)

        clock.flushed()                                     // barge-in → idle
        let t3 = clock.willSchedule()
        clock.completed(durationMs: 5, token: t3)
        #expect(clock.milliseconds() == 5)
    }

    @Test func outputsExposeAPlayedClockAndInputsDefaultToNil() async {
        // Pre-start: 0 from the built-ins (measured, nothing played), nil from the protocol default.
        struct BareOutput: AudioOutput {
            func start() async throws {}
            func enqueue(_ pcm16: Data) async {}
            func flush() async {}
            func stop() async {}
        }
        #expect(await AudioPlayback().playedMilliseconds() == 0)
        #expect(await VoiceProcessedAudioIO().playedMilliseconds() == 0)
        #expect(await BareOutput().playedMilliseconds() == nil)
    }

    @Test func pcm16FloatBufferMatchesFloatSamples() {
        let format = AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)!
        let buffer = PCM16.floatBuffer(fromPCM16: Data([0x00, 0x00, 0xFF, 0x7F, 0x00, 0x80]), format: format)
        #expect(buffer?.frameLength == 3)
        #expect(buffer?.floatChannelData?[0][2] == -1.0)
        #expect(PCM16.floatBuffer(fromPCM16: Data(), format: format) == nil)
    }

    @Test func pcm16ToFloatScalingAndEndianness() {
        // Little-endian Int16: 0 → 0.0 ; +32767 (FF 7F) → ~+1 ; -32768 (00 80) → -1.0
        let samples = AudioPlayback.floatSamples(fromPCM16: Data([0x00, 0x00, 0xFF, 0x7F, 0x00, 0x80]))
        #expect(samples.count == 3)
        #expect(samples[0] == 0.0)
        #expect(abs(samples[1] - 0.99997) < 0.001)
        #expect(samples[2] == -1.0)
        #expect(AudioPlayback.floatSamples(fromPCM16: Data([0x01])).isEmpty)   // odd byte count → no samples
    }
}
