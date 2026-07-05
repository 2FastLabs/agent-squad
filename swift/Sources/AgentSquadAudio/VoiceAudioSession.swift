#if os(iOS)
import AVFoundation

/// Shared `AVAudioSession` setup — one entry point so `MicCapture` and `AudioPlayback` can't
/// drift to conflicting categories. `.managed` is idempotent (re-activating an already-active
/// session is a no-op); a `.custom` closure runs on every `start()` and owns its own idempotency.
enum VoiceAudioSession {
    static func activate(policy: AudioSessionPolicy) throws {
        switch policy {
        case .managed:
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
            try session.setActive(true)
        case .custom(let configure):
            try configure(AVAudioSession.sharedInstance())
        case .external:
            break
        }
    }
}
#endif
