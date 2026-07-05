---
name: agent-squad-swift
description: >-
  Use when building or modifying a Swift app that uses the AgentSquad Swift framework — on-device
  multi-agent orchestration for iOS 16+ / macOS 14+: orchestrator, agents (Agent, GroundedAgent),
  classifier routing, LLM clients (OpenAI-compatible), tools (native + MCP), tool UIs/widgets,
  on-device storage, tracing, and realtime voice — built-in types and custom implementations.
---

# AgentSquad Swift — assistant guide

Protocol-driven, on-device multi-agent framework (Swift 6.2, iOS 16+ / macOS 14+; persistence via
`FileChatStorage` on iOS 16+, `DeviceChatStorage` on iOS 17+). This is guidance and
a map — **not an API reference**. Read the exact signatures from the source (`swift/Sources/AgentSquad/`)
and the worked recipes from the docs site sources (`docs/src/content/docs/swift/`); this file tells
you *what to use, when, and what to watch out for*.

## When to use what

- **One assistant** → an `Agent` (or `GroundedAgent`) + an `Orchestrator` with no classifier. No routing hop.
- **Several specialists** → multiple agents + an `LLMClassifier`; the orchestrator routes each turn.
- **Answers must not drift from data** (prices, stock, balances) → `GroundedAgent`: a Brain calls
  tools, an isolated Presenter speaks only from the curated results (it can be a smaller/local model).
- **Voice** → a `VoiceAssistant` (a peer of the orchestrator, not an agent): `OpenAIVoiceAssistant`
  (single LLM + tools, speaks directly — the spoken analog of `Agent`) or
  `OpenAIGroundedVoiceAssistant` (Brain → Presenter, can't drift from data — analog of `GroundedAgent`).

Every component is a `Sendable` protocol with one built-in implementation — swap in your own anywhere.

## Modules (import only what you use)

| Import | Pulls in | Contents |
|---|---|---|
| `AgentSquad` | nothing external | protocols, `Agent`, `GroundedAgent`, `Orchestrator`, `LLMClassifier`, `ChatCompletionsClient`, `FileChatStorage`, `DeviceChatStorage`, `InMemoryChatStorage`, `OSLogTracer`, OTLP export |
| `AgentSquadMCP` | MCP Swift SDK | `MCPServer` (= `MCPToolProvider`), `SDKMCPClient` |
| `AgentSquadAudio` | AVFoundation | `VoiceProcessedAudioIO` (capture+playback, one engine, AEC — the recommended wiring), `MicCapture` (voice-processed/AEC by default), `AudioPlayback`, `VoiceProcessing`, `AudioSessionPolicy` (needs `NSMicrophoneUsageDescription`) |

SwiftPM: `.package(url: "https://github.com/2FastLabs/agent-squad", branch: "main")`.

## How a turn works

Two peer runtimes share contracts but not a control loop: a turn-based **`Orchestrator`**
(`classify? → run agent → stream → persist`) and a long-lived **`VoiceAssistant`** for voice. Either
way you consume an `AsyncThrowingStream<AgentEvent, any Error>` — the one idiom worth memorizing:

```swift
for try await event in orchestrator.route(.text("hello"), userId: "u1", sessionId: "s1") {
    switch event {
    case .textDelta(let token): /* stream tokens */
    case .final(let message):   /* the message that was persisted */
    case .toolCall, .widget, .thinking, .error: break   // .error is a user-facing string
    }
}
```

`.error` carries a *user-facing* message; real programmer/transport failures **throw** through the
stream. `.final` is what the orchestrator persists. Inputs/messages are value types
(`AgentInput.text`, `ConversationMessage`, `ContentPart`, `JSONValue`) in `Sources/AgentSquad/Core/`.

## The pieces

- **`Orchestrator`** drives a turn. The **classifier is optional** — omit it for a single agent.
- **`Agent`** is one LLM with an internal tool loop. **`GroundedAgent`** is two LLMs (Brain + isolated
  Presenter) for answers that must stay grounded in tool results. The Presenter never sees chat
  history or the Brain's transcript; `presenterInput` picks `.questionAndData` (default) or `.dataOnly`.
- **`ChatCompletionsClient`** speaks the OpenAI wire — point its `baseURL` at OpenAI, Azure,
  OpenRouter, Groq, or a local Ollama/llama.cpp. Implement `LLMClient` for anything else.
- **Tools** come from a `ToolProvider`. Built-ins: **`ToolKit`** holds native tools — `Tool.local`
  (Swift closure) and `Tool.http`/`Tool.get`/`.post` (declarative HTTP, with a `ToolParameter` DSL so
  you don't hand-write JSON Schema); **`HTTPToolGroup(baseURL:…)`** declares one API's shared
  config once, then one line per endpoint; **`MCPServer(url:)`** connects an MCP server; and
  **`AggregateToolProvider`** composes any mix behind one seam. A `ToolResult` is three-part: text →
  the model's context, `structuredContent` → curator/UI data, `ui` → an optional widget.
- **`FileChatStorage`** (JSON files, iOS 16+) and **`DeviceChatStorage`** (SwiftData, iOS 17+) persist history on-device; **`InMemoryChatStorage`** is a non-persistent, seedable single-conversation store. **`OSLogTracer`** is the default
  tracer; wire `ProcessingTracer` + `OTLPExporter` to ship traces to Langfuse/LangSmith/Datadog/…
- **Voice**: two `VoiceAssistant`s over a WebSocket — `OpenAIVoiceAssistant` (single LLM, speaks
  directly) and `OpenAIGroundedVoiceAssistant` (grounded Brain → Presenter). Both are self-sufficient
  (own `tracer`/`store`/`userId`/`sessionId`; with a `store`, completed turns persist and prior
  history seeds on `start()`), wired to the mic/speaker by `RealtimeRuntime`. Preferred audio
  wiring: ONE `VoiceProcessedAudioIO` instance passed as both `input:` and `output:` — capture and
  playback share one voice-processed `AVAudioEngine`, so the assistant's audio is guaranteed to be
  in the echo canceller's reference path. The split `MicCapture`/`AudioPlayback` pair also works
  (capture is voice-processed by default; the AEC reference is then device-level/route-dependent;
  `MicCapture(voiceProcessing: nil)` = raw capture). All three audio classes take an
  `AudioSessionPolicy` (`.managed` / `.custom` / `.external` for apps that own the
  `AVAudioSession`) and a `configureEngine` hook exposing the raw `AVAudioEngine`.
  Session tuning on both: `transcriptionModel` (the user's STT only), `turnDetection`
  (`.semanticVAD(eagerness:)` / `.serverVAD(threshold:…)` / `.disabled`), and `sessionOverrides`
  (deep-merged into the generated `session.update` last — the escape hatch for unmodeled keys like
  `audio.input.noise_reduction`).

## Custom implementations

Conform to the protocol and pass your type where the built-in goes. Each seam has a worked example on
its doc page (paths below are under `docs/src/content/docs/swift/`, published at `/agent-squad/swift/…`);
signatures live in `Sources/AgentSquad/`.

| Seam | Protocol | Source · doc |
|---|---|---|
| Agent | `AgentProtocol` | `Core/AgentProtocol.swift` · `agents/custom` |
| Classifier | `Classifier` (return an agent from the passed list, or `nil`) | `Core/Classifier/` · `classifiers/custom` |
| LLM client | `LLMClient` | `Core/LLMClient.swift` · `llm/custom` |
| Tools | `ToolProvider` | `Core/Tooling/` · `tools/custom` |
| Tool-output curator | `ToolOutputCurator` (where you trim oversized output) | `Core/Presenter/` · `ui/built-in/curators` |
| Presenter prompt | `PresenterPrompt` | `Core/Presenter/` · `agents/built-in/grounded-agent` |
| Storage | `ChatStorage` | `Core/Storage/` · `storage/custom` |
| Tracing | `TraceExporter` (easiest) / `SpanProcessor` / `Tracer` / `Redactor` | `Core/Tracing/` · `tracing/custom` |
| Realtime transport | `RealtimeTransport` | `Runtimes/Realtime/` · `voice/custom` |
| Audio I/O | `AudioInput` / `AudioOutput` | `Runtimes/Realtime/AudioIO.swift` · `audio/custom` |

## Gotchas

- **`maxToolRounds`**: `Agent`/`GroundedAgent` default to `20`; the `AgentProtocol` default is `1`. A
  custom agent that injects tools but leaves `1` silently disables its tool loop.
- **Classifier is optional**: no classifier ⇒ no routing hop / no extra model call. A `nil` selection
  falls back to the default agent (no confidence threshold).
- **Persistence**: only turns ending in `.final` are saved.
- **`ChatCompletionsClient`**: retries only *before the first event*; some local runtimes reject
  `stream_options`/unknown body keys — override via `extraBody`.
- **`JSONValue`**: whole-number doubles decode to `.int`; carry large IDs as `.string`.
- **Storage**: `FileChatStorage` (JSON, iOS 16+, scopes per-call by `userId`/`sessionId`/`agentId` — e.g. `sessionId` to isolate per match) or `DeviceChatStorage` (SwiftData, iOS 17+, bound to one `userId`). Both default to Library/Caches (disposable). `InMemoryChatStorage` (iOS 16+) is non-persistent and holds one conversation — construct it empty or seeded with a prior conversation to load one into a session.
- **Tracing lifecycle**: nothing drains the tracer for you — flush on background, shut down on
  termination. `OSLogTracer` logs no payloads. `Redaction` hashes ids + clips strings but does **not**
  pattern-scrub PII — supply a custom `Redactor` for that.
- **Realtime** is a peer runtime, not an agent; its `events` stream is non-throwing; needs
  `NSMicrophoneUsageDescription`; always `stop()`.
- **Voice processing (AEC)**: on by default; if it can't be enabled `start()` throws
  `.voiceProcessingUnavailable` (degrade deliberately with `MicCapture(voiceProcessing: nil)`).
  For guaranteed echo cancellation use `VoiceProcessedAudioIO` and pass the **same instance** as
  input and output (its `start()`/`stop()` are idempotent — the runtime calls each twice). The
  simulator does **no** AEC — validate on a device. VP quiets the speaker (counter with
  `duckingLevel: .min`); never enable VP on a playback-only engine. With
  `sessionPolicy: .external` the app must configure **and activate** its `AVAudioSession` before
  `start()`, and should use the same policy everywhere.
- **`ContentPart` Codable** keys off case + label names — renaming breaks stored history.

## Go deeper

- **Prose & recipes** — the Starlight docs under `docs/src/content/docs/swift/` (run the site from
  `docs/` with `npm run dev`): `quick-start`, `orchestrator/overview`,
  `agents/built-in/grounded-agent`, `mcp/overview`, `ui/overview`, `storage/built-in/device`,
  `tracing/built-in/otlp-exporter`, `voice/built-in/openai-voice`,
  `voice/built-in/openai-grounded-voice`, `guides/*`.
- **Exact signatures** — `swift/Sources/AgentSquad/` (`Core/`, `Agents/`, `Core/LLM/`,
  `Core/Tooling/`, `Core/Tracing/`, `Runtimes/Realtime/`).
