---
title: OpenAIGroundedVoiceAssistant
description: Two-phase grounded voice assistant — a silent gatherer calls tools, then an isolated presenter speaks from curated data only.
---

`OpenAIGroundedVoiceAssistant` is the grounded `VoiceAssistant` implementation. Each tool-using turn runs two responses on one WebSocket: a silent **gatherer** that calls tools and accumulates results, then an isolated **presenter** that speaks solely from the curated facts. This mirrors [GroundedAgent](/agent-squad/swift/agents/built-in/grounded-agent/) in a voice context. For the single-LLM variant see [OpenAIVoiceAssistant](/agent-squad/swift/voice/built-in/openai-voice/).

---

## Init

```swift
public actor OpenAIGroundedVoiceAssistant: VoiceAssistant {
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
        curator: any ToolOutputCurator = .dataBlock,
        presenterPrompt: PresenterPrompt = .default,
        agentInstructions: String = OpenAIGroundedVoiceAssistant.defaultAgentInstructions,
        directInstructions: String = OpenAIGroundedVoiceAssistant.defaultDirectInstructions,
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
| `tools` | — | Tool registry the gatherer may call |
| `userId` | — | Scopes storage and tracing |
| `sessionId` | — | Scopes storage and tracing |
| `store` | `nil` | Enables history seeding and persistence; see [Storage overview](/agent-squad/swift/storage/overview/) |
| `tracer` | `OSLogTracer()` | Span sink; see [Tracing overview](/agent-squad/swift/tracing/overview/) |
| `traceTranscripts` | `true` | When `false`, span input/output fields are omitted |
| `tracePerTurn` | `false` | When `true`, each turn is its own root trace instead of nesting under one `voice.session` root — so a backend that finalizes a trace on its root's end (e.g. LangSmith) renders each turn as it completes, mid-session. Turns still group into a conversation via shared span metadata (e.g. a `thread_id`), not a shared trace id |
| `traceName` | `nil` | Overrides the root trace name with a human label (e.g. the match teams). `nil` keeps the defaults: `voice.session` (session root) / `voice.turn` (per-turn root) |
| `maxMessages` | `ChatStorageDefaults.maxMessages` | Cap on replayed history items |
| `modality` | `RealtimeModality()` (speech in / audio out) | Controls the event mix; see [Voice overview](/agent-squad/swift/voice/overview/) |
| `curator` | `.dataBlock` | Shapes tool output fed to the presenter; see [UI overview](/agent-squad/swift/ui/overview/) |
| `presenterPrompt` | `.default` | Prompt wrapper injected before the curated data block |
| `presenterInput` | `.questionAndData` | What the presenter sees besides its prompt: the question plus the curated feed (default), or `.dataOnly` for the feed alone |
| `agentInstructions` | `defaultAgentInstructions` | System prompt for the gatherer — instructs it to call tools and not speak |
| `directInstructions` | `defaultDirectInstructions` | System prompt for the direct-response path (no tools called) |
| `model` | `"gpt-realtime"` | Passed as the `?model=` query parameter by the transport |
| `voice` | `"marin"` | OpenAI Realtime voice name |
| `language` | `nil` | BCP-47 language code; `nil` lets the model auto-detect |
| `sampleRate` | `24_000` | PCM16 sample rate; must match `AudioInput`/`AudioOutput` |
| `transcriptionModel` | `"gpt-4o-mini-transcribe"` | Speech-to-text model for the *user's* transcript (captions + persisted user turns); e.g. `"gpt-4o-transcribe"` for better accuracy at higher cost. Does not affect the gatherer or presenter |
| `turnDetection` | `.semanticVAD()` | When the server ends the user's turn: `.semanticVAD(eagerness: .low/.medium/.high/.auto)` to trade patience vs latency, `.serverVAD(threshold:prefixPaddingMs:silenceDurationMs:)` for silence-based detection, `.disabled` for text-driven sessions only (`sendText`; spoken turns need VAD) |
| `sessionOverrides` | `[:]` | Escape hatch: deep-merged into the generated `session.update` object last — set any session key the parameters above don't model (e.g. `audio.input.noise_reduction`). Nested objects merge; scalars and arrays replace |

---

## Session tuning

Session config is layered: **typed parameters** for the everyday knobs, and **`sessionOverrides`** as
the forward-compatibility valve. Overrides are an arbitrary JSON object deep-merged into the
generated `session` payload *last*, so when OpenAI ships a new session parameter you can set it from
your app immediately — no AgentSquad release needed:

```swift
let assistant = OpenAIGroundedVoiceAssistant(
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
- Overrides patch the **`session.update` frame only** — the presenter and direct replies ride
  per-response `response.create` frames, deliberately out of reach, so an override can't corrupt
  the grounded turn machinery. Their prompts are the `presenterPrompt` / `directInstructions`
  parameters instead.

---

## Turn structure

A tool-using turn runs in two phases:

1. **Gather** — the gatherer response runs text-only (it never speaks). It calls tools and accumulates results. The session emits `.state(.thinking)` during this phase.
2. **Present** — tool results are curated by `curator` and passed to an isolated presenter response. The presenter speaks from the curated block only. The session emits `.state(.presenting)`.

When no tools are called, the gatherer's accumulated text is used for a **direct response** — the model speaks from conversation history without the grounding step. `directInstructions` governs that path.

:::note
`OpenAIGroundedVoiceAssistant` emits `.state(.thinking)` during the gather phase and `.state(.presenting)` when the presenter speaks, so the UI can distinguish the two phases from `.state(.speaking)` in the direct path.
:::

---

## Usage

```swift
import AgentSquad

let transport = URLSessionWebSocketTransport(
    apiKey: "sk-..."
)

let assistant = OpenAIGroundedVoiceAssistant(
    name: "grounded-voice",
    transport: transport,
    tools: myToolProvider,
    userId: currentUser.id,
    sessionId: UUID().uuidString,
    store: InMemoryChatStorage(),
    curator: .dataBlock,
    voice: "marin"
)

let runtime = RealtimeRuntime(
    session: assistant,
    input: MicCapture(),
    output: AudioPlayback()
)

try await runtime.start()

for await event in runtime.events {
    switch event {
    case .state(.thinking):
        showSpinner("Looking it up…")
    case .state(.presenting):
        showSpinner("Speaking…")
    case .state(.listening):
        hideSpinner()
    case .userTranscript(let text, final: true):
        showUserBubble(text)
    case .presenterText(let text, final: true):
        showAssistantBubble(text)
    case .widget(let payload):
        renderWidget(payload)
    case .error(let code, let message):
        print("voice error [\(code ?? "unknown")]:", message)
    default:
        break
    }
}
```

---

## Phase events emitted

| Phase | When |
|---|---|
| `.listening` | After `start()` and after each completed or interrupted turn |
| `.thinking` | Gatherer response is running (tool calls in flight) |
| `.presenting` | Presenter response is speaking from curated data |
| `.speaking` | Direct response (no tools called) — the model speaks from history |

---

## Default instructions

```swift
OpenAIGroundedVoiceAssistant.defaultAgentInstructions
// "You gather the facts needed to answer the user by calling the available tools. Do NOT speak
//  the final answer yourself — a separate presenter will. Call the tools you need and nothing else."

OpenAIGroundedVoiceAssistant.defaultDirectInstructions
// "You are a friendly, concise voice assistant. Reply naturally to the user. Do not call tools."
```

---

## Related pages

- [Voice overview](/agent-squad/swift/voice/overview/) — `RealtimeRuntime`, protocols, and event reference
- [OpenAIVoiceAssistant](/agent-squad/swift/voice/built-in/openai-voice/) — single-LLM variant
- [GroundedAgent](/agent-squad/swift/agents/built-in/grounded-agent/) — the text-turn analogue of this pattern
- [WebSocket transport](/agent-squad/swift/voice/built-in/websocket-transport/) — `URLSessionWebSocketTransport`
- [Custom transport](/agent-squad/swift/voice/custom/) — mock and custom `RealtimeTransport` implementations
- [Audio overview](/agent-squad/swift/audio/overview/) — `MicCapture` and `AudioPlayback`
