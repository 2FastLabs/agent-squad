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
