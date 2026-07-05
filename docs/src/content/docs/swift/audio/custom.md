---
title: Custom audio
description: Implement AudioInput and AudioOutput from the core AgentSquad module to replace MicCapture and AudioPlayback with any custom capture or playback strategy.
---

`AudioInput` and `AudioOutput` are declared in the core `AgentSquad` module (`AudioIO.swift`). You can conform to either without importing `AgentSquadAudio` — useful for unit tests, file-based input, remote audio bridges, or platforms where AVFoundation is unavailable.

```swift
import AgentSquad   // AudioInput, AudioOutput — no AVFoundation dependency
```

---

## Custom AudioInput — replaying pre-recorded PCM frames

```swift
import AgentSquad
import Foundation

/// Replays a pre-recorded PCM16 @ 24 kHz mono buffer in fixed-size chunks.
/// Useful in unit tests and integration harnesses that need deterministic audio.
final class FileAudioInput: AudioInput, @unchecked Sendable {

    let frames: AsyncStream<Data>

    private let continuation: AsyncStream<Data>.Continuation
    private let pcm16: Data          // full recording, PCM16 little-endian mono
    private let chunkBytes: Int      // bytes per chunk (e.g. 4800 samples × 2 = 9600 bytes per 200 ms)
    private var task: Task<Void, Never>?

    init(pcm16: Data, chunkBytes: Int = 9_600) {
        self.pcm16 = pcm16
        self.chunkBytes = chunkBytes
        (self.frames, self.continuation) = AsyncStream.makeStream(
            of: Data.self,
            bufferingPolicy: .bufferingNewest(16)
        )
    }

    func start() async throws {
        task = Task { [pcm16, chunkBytes, continuation] in
            var offset = 0
            while offset < pcm16.count {
                let end = min(offset + chunkBytes, pcm16.count)
                continuation.yield(pcm16[offset..<end])
                offset = end
                try? await Task.sleep(nanoseconds: 200_000_000)  // pace to ~5 chunks/s
                if Task.isCancelled { break }
            }
            continuation.finish()
        }
    }

    func stop() async {
        task?.cancel()
        task = nil
        continuation.finish()
    }
}
```

---

## Custom AudioOutput — capturing rendered audio in tests

```swift
import AgentSquad
import Foundation

/// Collects every enqueued PCM16 frame into an in-memory buffer.
/// Lets tests assert on exactly what the runtime tried to play back.
actor RecordingAudioOutput: AudioOutput {

    private(set) var recorded: [Data] = []
    private(set) var flushCount = 0

    func start() async throws {
        recorded = []
        flushCount = 0
    }

    func enqueue(_ pcm16: Data) async {
        recorded.append(pcm16)
    }

    func flush() async {
        flushCount += 1
        recorded.removeAll()
    }

    func stop() async {}
}
```

:::note
`RecordingAudioOutput` is declared as an `actor`, which satisfies `Sendable` implicitly. For a `class`-based conformance — like `MicCapture` and `AudioPlayback` — mark it `@unchecked Sendable` and serialise access yourself (or route all calls through a single `Task`).
:::

`playedMilliseconds()` is not implemented above — the protocol defaults it to `nil`, which skips barge-in truncation (`conversation.item.truncate`) and changes nothing else. Implement it (ms actually played of the current burst, surviving `flush()`) if your output plays audio for real and you want the model's context to keep only what the user heard.

---

## Wiring a custom implementation

Pass your conformance directly to `RealtimeRuntime`; the runtime only cares about the protocol:

```swift
let pcm = try Data(contentsOf: URL(fileURLWithPath: "sample.raw"))

let runtime = RealtimeRuntime(
    input:  FileAudioInput(pcm16: pcm),
    output: RecordingAudioOutput(),
    // ... other config
)
```

---

## Related pages

- [Audio overview](/agent-squad/swift/audio/overview/) — the `AudioInput` / `AudioOutput` protocols in full
- [MicCapture](/agent-squad/swift/audio/built-in/mic-capture/) — built-in `AudioInput` over AVAudioEngine
- [AudioPlayback](/agent-squad/swift/audio/built-in/audio-playback/) — built-in `AudioOutput` over AVAudioEngine
- [Voice overview](/agent-squad/swift/voice/overview/) — the `RealtimeRuntime` that consumes both protocols
