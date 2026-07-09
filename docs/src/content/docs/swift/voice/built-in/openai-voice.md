---
title: OpenAIVoiceAssistant
description: Single-LLM realtime voice assistant — calls tools and speaks the answer in one response.
---

`OpenAIVoiceAssistant` is the single-LLM `VoiceAssistant` implementation. One OpenAI Realtime WebSocket handles the entire agent turn: the model calls any needed tools and then speaks the answer directly. For the two-phase gatherer→presenter variant see [OpenAIGroundedVoiceAssistant](/agent-squad/swift/voice/built-in/openai-grounded-voice/).

---

## Init

```swift
public actor OpenAIVoiceAssistant: VoiceAssistant {
    public init(
        name: String,
        transport: any RealtimeTransport,
        tools: any ToolProvider,
        userId: String,
        sessionId: String,
        store: (any ChatStorage)? = nil,
        tracer: any Tracer = OSLogTracer(),
        traceTranscripts: Bool = true,
        tracePerTurn: Bool = false,
        traceName: String? = nil,
        maxMessages: Int? = ChatStorageDefaults.maxMessages,
        modality: RealtimeModality = RealtimeModality(),
        instructions: String = OpenAIVoiceAssistant.defaultInstructions,
        model: String = "gpt-realtime",
        voice: String = "marin",
        language: String? = nil,
        sampleRate: Int = 24_000,
        transcriptionModel: String = "gpt-4o-mini-transcribe",
        turnDetection: RealtimeTurnDetection = .semanticVAD(),
        sessionOverrides: [String: JSONValue] = [:]
    )
}
```

### Parameters

| Parameter | Default | Notes |
|---|---|---|
| `name` | — | Identifies this assistant in storage and traces; slugified for the storage key |
| `transport` | — | Any `RealtimeTransport`; use `URLSessionWebSocketTransport` in production |
| `tools` | — | Tool registry the model may call during a turn |
| `userId` | — | Scopes storage and tracing |
| `sessionId` | — | Scopes storage and tracing |
| `store` | `nil` | Enables history seeding and persistence; see [Storage overview](/agent-squad/swift/storage/overview/) |
| `tracer` | `OSLogTracer()` | Span sink; see [Tracing overview](/agent-squad/swift/tracing/overview/) |
| `traceTranscripts` | `true` | When `false`, span input/output fields are omitted while structure and token counts still flow |
| `tracePerTurn` | `false` | When `true`, each turn is its own root trace instead of nesting under one `voice.session` root — so a backend that finalizes a trace on its root's end (e.g. LangSmith) renders each turn as it completes, mid-session. Turns still group into a conversation via shared span metadata (e.g. a `thread_id`), not a shared trace id |
| `traceName` | `nil` | Overrides the root trace name with a human label (e.g. the match teams). `nil` keeps the defaults: `voice.session` (session root) / `voice.turn` (per-turn root) |
| `maxMessages` | `ChatStorageDefaults.maxMessages` | Cap on replayed history items |
| `modality` | `RealtimeModality()` (speech in / audio out) | Controls the event mix; see [Voice overview](/agent-squad/swift/voice/overview/) |
| `instructions` | `defaultInstructions` | System prompt — tells the model to use tools and reply concisely in speech |
| `model` | `"gpt-realtime"` | Passed as the `?model=` query parameter by the transport |
| `voice` | `"marin"` | OpenAI Realtime voice name |
| `language` | `nil` | BCP-47 language code; `nil` lets the model auto-detect |
| `sampleRate` | `24_000` | PCM16 sample rate; must match `AudioInput`/`AudioOutput` |
| `transcriptionModel` | `"gpt-4o-mini-transcribe"` | Speech-to-text model for the *user's* transcript (captions + persisted user turns); e.g. `"gpt-4o-transcribe"` for better accuracy at higher cost. Does not affect the assistant's replies |
| `turnDetection` | `.semanticVAD()` | When the server ends the user's turn: `.semanticVAD(eagerness: .low/.medium/.high/.auto)` to trade patience vs latency, `.serverVAD(threshold:prefixPaddingMs:silenceDurationMs:)` for silence-based detection, `.disabled` for text-driven sessions only (`sendText`; spoken turns need VAD) |
| `sessionOverrides` | `[:]` | Escape hatch: deep-merged into the generated `session.update` object last — set any session key the parameters above don't model (e.g. `audio.input.noise_reduction`). Nested objects merge; scalars and arrays replace |

---

## Session tuning

Session config is layered: **typed parameters** for the everyday knobs, and **`sessionOverrides`** as
the forward-compatibility valve. Overrides are an arbitrary JSON object deep-merged into the
generated `session` payload *last*, so when OpenAI ships a new session parameter you can set it from
your app immediately — no AgentSquad release needed:

```swift
let assistant = OpenAIVoiceAssistant(
    name: "Sport", transport: transport, tools: tools,
    userId: "u1", sessionId: "s1",
    transcriptionModel: "gpt-4o-transcribe",             // better STT accuracy, higher cost
    turnDetection: .serverVAD(silenceDurationMs: 500),   // idle_timeout_ms below only applies to server_vad
    sessionOverrides: [
        "audio": .object(["input": .object([
            "noise_reduction": .object(["type": .string("near_field")]),
            "turn_detection": .object(["idle_timeout_ms": .int(6_000)]),   // key not (yet) modeled — just set it
        ])]),
        "max_output_tokens": .int(512),
    ]
)
```

Merge rules:

- **Any key you set wins**, including generated ones (`instructions`, `output_modalities`, the
  transcription model, …) — you can effectively author the whole session config yourself.
- **Keys you don't set keep their generated values** — objects merge key-by-key, so overriding
  `audio.input.noise_reduction` doesn't clobber `audio.input.format` or `transcription`.
- **Clear a key** by setting it to `.null` (the Realtime API's way of disabling things, e.g.
  `turn_detection`); keys can be nulled but never removed from the payload.
- Overrides patch the **`session.update` frame only** — per-response `response.create` frames are
  untouched, so an override can't corrupt the turn machinery.

---

## History seeding

When `store` is provided, `start()` replays prior turns from `ChatStorage` as conversation items before the pump handles inbound frames. Each completed turn (user transcript + spoken reply) is saved under `slugify(name)`. See [Storage overview](/agent-squad/swift/storage/overview/).

---

## Text-only typed turns

`sendText` marks the turn as text-only: the tool→continue loop stays in text and `.state(.speaking)` is never emitted. This is useful for non-voice UIs sharing the same session or for injecting context without audio output.

---

## Usage

```swift
import AgentSquad

let transport = URLSessionWebSocketTransport(
    apiKey: "sk-..."
)

let assistant = OpenAIVoiceAssistant(
    name: "support-voice",
    transport: transport,
    tools: myToolProvider,
    userId: currentUser.id,
    sessionId: UUID().uuidString,
    store: InMemoryChatStorage(),
    voice: "marin",
    language: "en"
)

let runtime = RealtimeRuntime(
    session: assistant,
    input: MicCapture(),
    output: AudioPlayback()
)

try await runtime.start()

for await event in runtime.events {
    switch event {
    case .state(let phase):
        updateStatusIndicator(phase)
    case .userTranscript(let text, final: true):
        showUserBubble(text)
    case .presenterText(let text, final: true):
        showAssistantBubble(text)
    case .error(let code, let message):
        print("voice error [\(code ?? "unknown")]:", message)
    default:
        break
    }
}
```

:::note
`RealtimeRuntime` is the only caller of `start()`, `stop()`, `sendAudio()`, and `interrupt()` on the assistant. Your app interacts exclusively with `RealtimeRuntime`. See [Voice overview](/agent-squad/swift/voice/overview/).
:::

---

## Phase events emitted

| Phase | When |
|---|---|
| `.listening` | After `start()` and after each completed or interrupted turn |
| `.thinking` | When a typed turn (`sendText`) starts — VAD turns emit `.speaking` directly |
| `.speaking` | When the model begins its spoken response |

---

## Related pages

- [Voice overview](/agent-squad/swift/voice/overview/) — `RealtimeRuntime`, protocols, and event reference
- [OpenAIGroundedVoiceAssistant](/agent-squad/swift/voice/built-in/openai-grounded-voice/) — two-phase grounded variant
- [WebSocket transport](/agent-squad/swift/voice/built-in/websocket-transport/) — `URLSessionWebSocketTransport`
- [Custom transport](/agent-squad/swift/voice/custom/) — mock and custom `RealtimeTransport` implementations
- [Audio overview](/agent-squad/swift/audio/overview/) — `MicCapture` and `AudioPlayback`
