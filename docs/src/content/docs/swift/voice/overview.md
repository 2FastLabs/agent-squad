---
title: Voice overview
description: Long-lived bidirectional voice runtime built on OpenAI Realtime — a peer to Orchestrator that owns its own WebSocket, tool loop, audio I/O, and tracing.
---

The realtime voice stack is a peer runtime to the [Orchestrator](/agent-squad/swift/orchestrator/overview/) — not an agent. It owns a persistent WebSocket, drives its own turn loop, and routes audio directly to playback. You instantiate a `RealtimeRuntime`, hand it a `VoiceAssistant` and audio I/O, call `start()`, and observe events.

## Core types at a glance

| Type | Role |
|---|---|
| `RealtimeRuntime` | Wires a `VoiceAssistant` to `AudioInput`/`AudioOutput`; the object the app holds |
| `VoiceAssistant` | Protocol — start, stop, send audio/text, interrupt, events stream |
| `OpenAIVoiceAssistant` | Single-LLM implementation — calls tools then speaks the answer |
| `OpenAIGroundedVoiceAssistant` | Gatherer→presenter — tools first, then isolated presenter speaks |
| `RealtimeTransport` | Frame channel protocol; `URLSessionWebSocketTransport` is the live implementation |
| `AudioInput` / `AudioOutput` | PCM16 capture and playback seams — AVFoundation implementations in `AgentSquadAudio` |

---

## VoiceAssistant protocol

`VoiceAssistant` is the contract every voice session satisfies. `RealtimeRuntime` consumes it; your app does not call it directly — use `RealtimeRuntime`'s surface instead.

```swift
public protocol VoiceAssistant: Sendable {
    var modality: RealtimeModality { get }
    var events: AsyncStream<RealtimeEvent> { get }
    func start() async throws
    func sendAudio(_ pcm16: Data) async
    func sendText(_ text: String) async
    func interrupt() async
    func stop() async
}
```

- `sendAudio` expects PCM16 at 24 kHz mono chunks — exactly what `AudioInput.frames` produces.
- `sendText` triggers a typed turn (no VAD); the reply is text-only in `OpenAIVoiceAssistant`.
- `interrupt` is an explicit barge-in: flushes playback and cancels the in-flight server response without tearing down the connection.

---

## RealtimeRuntime

`RealtimeRuntime` is the object an app instantiates. It wires the session to `AudioInput`/`AudioOutput`, re-broadcasts the session's events, and handles audio routing including barge-in flushes.

```swift
public actor RealtimeRuntime {
    public nonisolated let events: AsyncStream<RealtimeEvent>
    public nonisolated var modality: RealtimeModality { get }

    public init(
        session: any VoiceAssistant,
        input: any AudioInput,
        output: any AudioOutput
    )

    public func start() async throws
    public func sendText(_ text: String) async
    public func interrupt() async
    public func stop() async
}
```

`start()` starts the output engine, then the session, then the mic — in that order — and launches two internal tasks: one pumping the session's events to playback and re-broadcasting them, and one forwarding mic frames to the session. `stop()` tears down in producer-first order to avoid in-flight sends against an already-stopped session.

`interrupt()` flushes `AudioOutput` immediately for minimum latency before awaiting the server round-trip; the session's own `interrupt()` then emits `.audioDone(interrupted: true)`, which causes a second flush — harmless and idempotent.

:::caution
`stop()` is the only way to release the mic and WebSocket. Always call it — e.g. in `onDisappear` or a view model's `deinit`.
:::

### Minimal setup

```swift
let transport = URLSessionWebSocketTransport(
    apiKey: "sk-..."
)

let assistant = OpenAIVoiceAssistant(
    name: "voice-assistant",
    transport: transport,
    tools: myToolProvider,
    userId: "u1",
    sessionId: UUID().uuidString
)

// AudioInput / AudioOutput from AgentSquadAudio:
let runtime = RealtimeRuntime(
    session: assistant,
    input: MicCapture(),
    output: AudioPlayback()
)

try await runtime.start()

for await event in runtime.events {
    switch event {
    case .state(let phase):                        updateUI(phase)
    case .userTranscript(let text, final: true):   showTranscript(text)
    case .presenterText(let text, final: true):    showReply(text)
    case .error(let msg):                          print("error:", msg)
    default: break
    }
}
```

---

## RealtimeModality

Controls what the session produces and consumes.

```swift
public struct RealtimeModality: Sendable, Equatable {
    public enum Input: Sendable  { case speech, text }
    public enum Output: Sendable { case audio, text, audioAndText }
    public let input: Input
    public let output: Output
    public init(input: Input = .speech, output: Output = .audio)
}
```

The default `RealtimeModality()` is speech in / audio out. `output: .audioAndText` makes the session emit both `.audio` frames and `.presenterText` deltas in parallel.

---

## RealtimeEvent

The non-throwing event stream. All events arrive on `RealtimeRuntime.events` (which re-broadcasts the session's own stream).

```swift
public enum RealtimeEvent: Sendable {
    case state(RealtimePhase)
    case userTranscript(String, final: Bool)   // STT delta or final
    case presenterText(String, final: Bool)    // spoken reply as text (audio+text or text mode)
    case widget(UIPayload)                     // MCP Apps UI payload for this turn
    case audio(Data)                           // PCM16 @ 24 kHz — drain promptly
    case audioDone(interrupted: Bool)          // flush playback: barge-in or natural end
    case error(String)
}
```

- `userTranscript` streams incrementally (`final: false`) then fires once more with `final: true`. `presenterText` mirrors that pattern.
- `audio` frames arrive continuously while the model speaks — `RealtimeRuntime` routes them to `AudioOutput` automatically; an app observing `events` directly must drain them promptly.

---

## RealtimePhase

```swift
public enum RealtimePhase: String, Sendable {
    case idle        // not started
    case ready       // connected, awaiting typed input (text-input mode)
    case listening   // capturing the user's voice
    case thinking    // agent turn: calling tools
    case presenting  // grounded presenter speaking from curated data
    case speaking    // direct (no-tool) reply
}
```

---

## History seeding

When `store` is provided at init, `start()` replays prior turns from `ChatStorage` as conversation items before the pump starts handling inbound frames. Each completed turn (user transcript + spoken reply) is saved under `slugify(name)`. See [Storage overview](/agent-squad/swift/storage/overview/).

---

## Text-only typed turns

`sendText` (called on `RealtimeRuntime`) marks the turn as text-only: the assistant's tool→continue loop stays in text, and `.state(.speaking)` is never emitted. Useful for non-voice UIs sharing the same session.

---

## Tracing

By default each session is one trace: a `voice.session` root span opened on the first turn, with a `voice.turn` child per turn. Tool calls appear as `tool.<name>` children of the turn. Token usage (including per-modality audio token breakdown) is attached to the generation span inside each turn. Set `traceTranscripts: false` to keep spoken content off the trace while span structure and token counts still flow.

Set `tracePerTurn: true` to instead root **each turn** as its own trace (no shared `voice.session`): a backend that finalizes a trace when its root span ends (e.g. LangSmith) then renders each turn the moment it completes, rather than only after the session closes. The turns still group into one conversation via shared span metadata (e.g. a `thread_id` a custom `Redactor` stamps on every span), not a shared trace id. Use `traceName` to label the root trace with something human (e.g. the match teams) instead of `voice.session` / `voice.turn`. See [Tracing overview](/agent-squad/swift/tracing/overview/).

---

## Audio I/O

`AudioInput` and `AudioOutput` are protocol seams so the runtime is testable without hardware. The concrete AVFoundation implementations (`MicCapture`, `AudioPlayback`) live in the `AgentSquadAudio` module.

```swift
public protocol AudioInput: Sendable {
    var frames: AsyncStream<Data> { get }   // PCM16 @ 24 kHz mono, bounded drop-oldest
    func start() async throws
    func stop() async
}

public protocol AudioOutput: Sendable {
    func start() async throws
    func enqueue(_ pcm16: Data) async       // queue one PCM16 frame for playback
    func flush() async                      // drop all queued/playing audio (barge-in cut)
    func stop() async
}
```

`frames` is a bounded drop-oldest stream so a slow consumer never blocks the audio capture thread. `flush()` is the barge-in cut — it drops both queued and currently-playing audio immediately.

:::note
`AudioInput` and `AudioOutput` are defined in `AgentSquad` core. The AVFoundation implementations that use the microphone and speaker require `AgentSquadAudio`, which is a separate Swift package target to keep the core free of AVFoundation. See the [Audio overview](/agent-squad/swift/audio/overview/).
:::

---

## Grounded vs. standard — when to use each

| | `OpenAIVoiceAssistant` | `OpenAIGroundedVoiceAssistant` |
|---|---|---|
| Turn structure | Single response: tools then speak | Gatherer (tools, silent) → presenter (speaks) |
| Hallucination risk | Standard — model can mix tool data with priors | Low — presenter sees only curated tool output |
| Latency | Lower (one response) | Higher (two responses per tool-using turn) |
| Direct (no-tool) turns | Model speaks directly | Model speaks directly (`directInstructions`) |
| Phase events | `.thinking`, `.speaking` | `.thinking`, `.presenting`, `.speaking` |

Use `OpenAIGroundedVoiceAssistant` when factual accuracy matters and the answer derives from tool data. Use `OpenAIVoiceAssistant` for low-latency assistants where the model's parametric knowledge is acceptable.

---

## What's next

- [OpenAIVoiceAssistant](/agent-squad/swift/voice/built-in/openai-voice/) — single-LLM built-in
- [OpenAIGroundedVoiceAssistant](/agent-squad/swift/voice/built-in/openai-grounded-voice/) — grounded two-phase built-in
- [WebSocket transport](/agent-squad/swift/voice/built-in/websocket-transport/) — `URLSessionWebSocketTransport` and the `RealtimeTransport` protocol
- [Custom transport](/agent-squad/swift/voice/custom/) — mock and production custom transports
- [Audio overview](/agent-squad/swift/audio/overview/) — `MicCapture` and `AudioPlayback`
