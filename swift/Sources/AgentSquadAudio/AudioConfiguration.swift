import AVFoundation

/// Apple Voice-Processing I/O configuration for `MicCapture`: hardware echo cancellation (the
/// speaker signal is the subtraction reference), noise suppression, and gain control. `nil` on
/// `MicCapture.init` = raw capture.
public struct VoiceProcessing: Sendable, Equatable {
    /// Automatic gain control on the captured speech.
    public var automaticGainControl: Bool

    /// How hard the system ducks other audio while capture runs (iOS 17+/macOS 14+, ignored
    /// elsewhere). `.min` counters the "speaker got quieter" side effect of voice processing.
    public var duckingLevel: DuckingLevel

    public enum DuckingLevel: Sendable, Equatable {
        case `default`, min, mid, max
    }

    public init(automaticGainControl: Bool = true, duckingLevel: DuckingLevel = .default) {
        self.automaticGainControl = automaticGainControl
        self.duckingLevel = duckingLevel
    }

    public static let `default` = VoiceProcessing()
}

extension VoiceProcessing.DuckingLevel {
    @available(iOS 17.0, macOS 14.0, *)
    var avLevel: AVAudioVoiceProcessingOtherAudioDuckingConfiguration.Level {
        switch self {
        case .default: .default
        case .min: .min
        case .mid: .mid
        case .max: .max
        }
    }
}

/// Who configures the `AVAudioSession`. All audio classes take a policy; if you use the split
/// `MicCapture`/`AudioPlayback` pair, give both the same one so they can't fight over the
/// session. No effect on macOS (no `AVAudioSession` there).
public enum AudioSessionPolicy: Sendable {
    /// AgentSquad configures it: `.playAndRecord`, `.voiceChat`, speaker output, Bluetooth HFP.
    case managed
    #if os(iOS)
    /// AgentSquad calls your closure (on every `start()`) instead of its own setup.
    case custom(@Sendable (AVAudioSession) throws -> Void)
    #endif
    /// AgentSquad never touches the session — the app owns category, mode, and activation, and
    /// must activate it before `start()`.
    case external
}
