---
title: VoiceProcessedAudioIO
description: Capture and playback on one voice-processed AVAudioEngine — echo cancellation with the assistant's audio guaranteed in the reference path. The recommended wiring for voice sessions.
---

`VoiceProcessedAudioIO` is part of the `AgentSquadAudio` product. It runs microphone capture **and** assistant playback on a **single** `AVAudioEngine` with Apple's Voice-Processing I/O unit enabled. Because the assistant's audio renders through the same engine's voice-processed output, it is by construction in the echo canceller's reference path — the configuration the VP unit is designed around.

Conforms to **both** `AudioInput` and `AudioOutput`: pass **one instance** as both `input:` and `output:`.

```swift
import AgentSquadAudio

let io = VoiceProcessedAudioIO()
let runtime = RealtimeRuntime(session: assistant, input: io, output: io)
try await runtime.start()
```

Prefer this over the separate [`MicCapture`](/agent-squad/swift/audio/built-in/mic-capture/) + [`AudioPlayback`](/agent-squad/swift/audio/built-in/audio-playback/) pair for voice sessions — with two engines the echo reference is taken at the device level, which is route-dependent.

---

## Init

```swift
public init(
    sampleRate: Double = 24_000,
    maxBufferedFrames: Int = 16,
    voiceProcessing: VoiceProcessing = .default,
    sessionPolicy: AudioSessionPolicy = .managed,
    configureEngine: (@Sendable (AVAudioEngine) throws -> Void)? = nil
)
```

| Parameter | Default | Notes |
|---|---|---|
| `sampleRate` | `24_000` | Both capture and playback rate. Must match what the realtime session expects (OpenAI Realtime: PCM is always 24 kHz). |
| `maxBufferedFrames` | `16` | Capacity of the capture `AsyncStream`; oldest frames dropped under back-pressure. |
| `voiceProcessing` | `.default` | AEC + noise suppression + AGC tuning. **Non-optional** — raw capture defeats this class's purpose; use `MicCapture(voiceProcessing: nil)` for that. |
| `sessionPolicy` | `.managed` | Who configures the `AVAudioSession` — see [AudioSessionPolicy](/agent-squad/swift/audio/built-in/mic-capture/#audiosessionpolicy-ios-only). |
| `configureEngine` | `nil` | Escape hatch: runs with the raw `AVAudioEngine` after voice processing is enabled and the player is wired, before the tap is installed. |

---

## Public surface

```swift
public let frames: AsyncStream<Data>    // AudioInput — captured PCM16 LE mono frames

public func start() async throws        // both roles; idempotent
public func enqueue(_ pcm16: Data) async // AudioOutput — schedule one frame
public func flush() async                // AudioOutput — instant barge-in cut
public func stop()  async                // both roles; idempotent
public func playedMilliseconds() async -> Double?  // ms actually played of the current burst (survives flush)
```

`playedMilliseconds()` feeds the session's `conversation.item.truncate` on barge-in, keeping the server's context aligned with what the user actually heard.

`start()` and `stop()` are **idempotent** — `RealtimeRuntime` calls each twice on the same instance (once through the `AudioOutput` role, once through `AudioInput`), and the second call is a no-op. `enqueue`/`flush` before `start()` or after `stop()` are safe no-ops. One instance serves **one session**: `stop()` finishes the `frames` stream for good — create a new instance to start again.

Failure modes are the shared [`MicCaptureError`](/agent-squad/swift/audio/built-in/mic-capture/#miccaptureerror) cases: `permissionDenied`, `converterUnavailable`, `voiceProcessingUnavailable`.

:::caution
The **simulator performs no echo cancellation** — validate AEC on a real device. Voice-processed output sounds "call-like" and slightly quieter; counter with `duckingLevel: .min`.
:::

---

## Related pages

- [Audio overview](/agent-squad/swift/audio/overview/) — the `AudioInput`/`AudioOutput` protocols
- [MicCapture](/agent-squad/swift/audio/built-in/mic-capture/) — capture-only built-in (split-pair wiring, `VoiceProcessing`, `AudioSessionPolicy` docs)
- [AudioPlayback](/agent-squad/swift/audio/built-in/audio-playback/) — playback-only built-in
- [Voice overview](/agent-squad/swift/voice/overview/) — the `RealtimeRuntime` that consumes this class
