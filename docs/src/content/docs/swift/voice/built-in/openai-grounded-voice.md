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
        maxMessages: Int? = ChatStorageDefaults.maxMessages,
        modality: RealtimeModality = RealtimeModality(),
        curator: any ToolOutputCurator = .dataBlock,
        presenterPrompt: PresenterPrompt = .default,
        agentInstructions: String = OpenAIGroundedVoiceAssistant.defaultAgentInstructions,
        directInstructions: String = OpenAIGroundedVoiceAssistant.defaultDirectInstructions,
        model: String = "gpt-realtime",
        voice: String = "marin",
        language: String? = nil,
        sampleRate: Int = 24_000
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
    case .error(let msg):
        print("voice error:", msg)
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
