---
title: MicCapture
description: AVAudioEngine-backed AudioInput that taps the microphone, converts to PCM16 @ 24 kHz mono, and yields frames into a bounded AsyncStream.
---

`MicCapture` is part of the `AgentSquadAudio` product. It taps `AVAudioEngine`'s input node, converts the hardware stream to **PCM16 @ 24 kHz mono**, and yields frames through a bounded `AsyncStream<Data>`.

By default it captures through Apple's **Voice-Processing I/O** unit: echo cancellation that uses the speaker signal as a hardware reference to subtract the assistant's own voice from the mic, plus noise suppression and automatic gain control.

:::tip
For voice sessions, prefer [`VoiceProcessedAudioIO`](/agent-squad/swift/audio/built-in/voice-processed-audio-io/) — capture and playback on **one** engine, which guarantees the assistant's audio is in the AEC reference path. With the split `MicCapture`/`AudioPlayback` pair the reference is device-level and route-dependent.
:::

```swift
import AgentSquadAudio
```

---

## Init

```swift
public init(
    sampleRate: Double = 24_000,
    maxBufferedFrames: Int = 16,
    voiceProcessing: VoiceProcessing? = .default,
    sessionPolicy: AudioSessionPolicy = .managed,
    configureEngine: (@Sendable (AVAudioEngine) throws -> Void)? = nil
)
```

| Parameter | Default | Notes |
|---|---|---|
| `sampleRate` | `24_000` | Target sample rate in Hz. Must match what the realtime runtime expects. |
| `maxBufferedFrames` | `16` | Capacity of the internal `AsyncStream`. Under back-pressure the **oldest** frames are dropped — the tap thread never blocks. |
| `voiceProcessing` | `.default` | Apple Voice-Processing I/O configuration (AEC + noise suppression + AGC). Pass `nil` for raw, unprocessed capture. |
| `sessionPolicy` | `.managed` | Who configures the `AVAudioSession` — see [AudioSessionPolicy](#audiosessionpolicy-ios-only) below. |
| `configureEngine` | `nil` | Escape hatch: runs with the raw `AVAudioEngine` after voice processing is enabled, before the tap is installed. |

---

## VoiceProcessing

Without voice processing, whatever `AudioPlayback` sends to the speaker leaks back into the mic — with server-side VAD, the assistant hears itself and interrupts its own answers. Voice processing is therefore **on by default**:

```swift
public struct VoiceProcessing: Sendable, Equatable {
    public var automaticGainControl: Bool          // default true
    public var duckingLevel: DuckingLevel          // default .default; iOS 17+/macOS 14+, ignored elsewhere
    public enum DuckingLevel { case `default`, min, mid, max }
    public static let `default`: VoiceProcessing
}
```

```swift
MicCapture()                                                              // AEC on — the default
MicCapture(voiceProcessing: .init(duckingLevel: .min))                    // AEC on, playback stays louder
MicCapture(voiceProcessing: .init(automaticGainControl: false))           // AEC on, no gain control
MicCapture(voiceProcessing: nil)                                          // raw capture (previous behavior)
```

Voice-processed audio sounds "call-like" and the speaker output gets quieter — `duckingLevel: .min` counters that. Enabling voice processing changes the input node's hardware format; `MicCapture` handles the ordering internally (`setVoiceProcessingEnabled` before the format read), which is why you should not enable it yourself from `configureEngine`.

:::caution
The **simulator performs no echo cancellation** — AEC must be validated on a real device. And never enable voice processing on the playback engine (`AudioPlayback`); it belongs on capture only.
:::

---

## Public surface

```swift
public let frames: AsyncStream<Data>   // yields PCM16 little-endian mono frames

public func start() async throws       // installs tap, starts engine, requests mic permission (iOS)
public func stop()  async              // stops engine, removes tap, finishes the stream
```

`start()` is idempotent — a second call before `stop()` is a no-op. Calling `start()` again after `stop()` is not supported; create a new instance.

---

## MicCaptureError

```swift
public enum MicCaptureError: Error, Equatable {
    case permissionDenied                     // user denied mic access (iOS only)
    case converterUnavailable                 // AVAudioConverter could not be initialised for the hardware format
    case voiceProcessingUnavailable(String)   // setVoiceProcessingEnabled(true) failed; payload = underlying error
}
```

`voiceProcessingUnavailable` is thrown rather than silently degrading to raw (echo-prone) capture. If raw capture is an acceptable fallback for your app, catch it and retry with `voiceProcessing: nil`.

---

## iOS mic permission

On iOS, `start()` requests microphone access before installing the tap:

- **iOS 17+** — uses `AVAudioApplication.requestRecordPermission`
- **iOS 16** — falls back to `AVAudioSession.requestRecordPermission` (the deprecated overload; no compiler warning fires at the iOS 16 deployment floor)

If the user denies permission, `start()` throws `MicCaptureError.permissionDenied`.

:::caution
Add `NSMicrophoneUsageDescription` to your `Info.plist`. Without it the system permission prompt will not appear, and the request is silently treated as denied.
:::

---

## AudioSessionPolicy (iOS only)

`sessionPolicy` decides who configures the shared `AVAudioSession` when `start()` runs. Pass the **same policy** to `MicCapture` and `AudioPlayback` so the two can't fight over the session. On macOS the session handling is compiled out and the system default audio device is used.

```swift
public enum AudioSessionPolicy: Sendable {
    case managed                                              // AgentSquad configures it (the default)
    case custom(@Sendable (AVAudioSession) throws -> Void)    // AgentSquad calls YOUR closure instead
    case external                                             // AgentSquad never touches the session
}
```

- **`.managed`** — the default; sets `category: .playAndRecord`, `mode: .voiceChat`, `options: [.defaultToSpeaker, .allowBluetoothHFP]` and activates the session. Idempotent — re-activating an already-active session is a no-op.
- **`.custom`** — AgentSquad drives the timing (on every `start()`) but with your configuration:

  ```swift
  let policy = AudioSessionPolicy.custom { session in
      try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.allowBluetoothHFP])
      try session.setActive(true)
  }
  ```

- **`.external`** — for apps that already manage their `AVAudioSession` (music, video, CallKit…). AgentSquad never touches it; you must configure **and activate** the session yourself *before* calling `start()`, otherwise the input format can read back as 0 Hz and capture fails.

---

## Usage

```swift
let mic = MicCapture()          // 24 kHz, 16-frame queue, echo-cancelled
try await mic.start()

for await frame in mic.frames {
    // frame: Data containing PCM16 little-endian mono samples
}

await mic.stop()
```

### Wiring to the voice runtime

Pass `MicCapture` directly to `RealtimeRuntime` — the runtime only cares about the `AudioInput` protocol:

```swift
let runtime = RealtimeRuntime(
    input:  MicCapture(),
    output: AudioPlayback(),
    // ... other config
)
```

:::caution
`start()` and `stop()` are not safe to call concurrently. `RealtimeRuntime` serialises them internally. If you drive `MicCapture` directly, serialise calls yourself.
:::

---

## Related pages

- [Audio overview](/agent-squad/swift/audio/overview/) — the `AudioInput` protocol and how it fits into the runtime
- [AudioPlayback](/agent-squad/swift/audio/built-in/audio-playback/) — the companion `AudioOutput` built-in
- [Custom audio](/agent-squad/swift/audio/custom/) — rolling your own `AudioInput` conformance
- [Voice overview](/agent-squad/swift/voice/overview/) — the `RealtimeRuntime` that consumes `MicCapture`
