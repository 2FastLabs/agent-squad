import Foundation

/// The shared half of the OpenAI Realtime voice runtimes. A peer of `Orchestrator` (not an
/// `AgentProtocol`): one WebSocket behind `RealtimeTransport`, its own tracing and persistence
/// scoped by `userId`/`sessionId`. `OpenAIVoiceAssistant` (single-LLM) and
/// `OpenAIGroundedVoiceAssistant` (gatherer→presenter) conform and add their per-turn brain.
///
/// Actors can't inherit, so the shared logic lives here as default methods on an `Actor`-bound
/// protocol: invoked on the conforming actor they run on its executor, synchronously, with no
/// added suspension — same ordering as if written inline. Storage stays in each actor (a protocol
/// has no stored properties) and is surfaced through the requirements below. This protocol and its
/// requirements are internal — plumbing, not public API (the actors' public surface is their
/// `VoiceAssistant` conformance).
protocol OpenAIRealtimeSession: Actor {
    // Config (set once at init).
    nonisolated var modality: RealtimeModality { get }
    nonisolated var name: String { get }
    nonisolated var transport: any RealtimeTransport { get }
    nonisolated var tools: any ToolProvider { get }
    nonisolated var tracer: any Tracer { get }
    nonisolated var userId: String { get }
    nonisolated var sessionId: String { get }
    nonisolated var store: (any ChatStorage)? { get }
    nonisolated var maxMessages: Int? { get }
    nonisolated var traceTranscripts: Bool { get }
    /// When true, each turn is its own root trace instead of a child of one `voice.session` root.
    /// Lets backends that finalize a trace on its root's end (e.g. LangSmith) render turns live
    /// mid-session; turns still group into a conversation via any shared metadata (e.g. a thread id).
    nonisolated var tracePerTurn: Bool { get }
    /// Overrides the root trace's name (e.g. a human label like the match teams). `nil` keeps the
    /// defaults: `voice.session` for the session root, `voice.turn` for a per-turn root.
    nonisolated var traceName: String? { get }
    nonisolated var model: String { get }
    nonisolated var voice: String { get }
    nonisolated var language: String? { get }
    nonisolated var sampleRate: Int { get }
    nonisolated var continuation: AsyncStream<RealtimeEvent>.Continuation { get }

    // Shared per-turn scaffold (cleared by each actor's `resetTurn`).
    var callsPerResponse: [String: Int] { get set }
    var fnNames: [String: String] { get set }

    // Lifecycle. The whole conversation is one trace: `sessionSpan` is a `voice.session` root
    // opened lazily on the first turn and closed in `endSession()`; each `turnSpan` is its child.
    var sessionSpan: (any SpanHandle)? { get set }
    var turnSpan: (any SpanHandle)? { get set }
    var pump: Task<Void, Never>? { get set }
    var stopped: Bool { get set }

    // Per-actor seams.
    /// The session-config frame sent on `start()` — differs in its instruction/output args.
    func sessionUpdateFrame(tools: [AgentTool]) -> String
    /// Post-tool-result step: the primitive emits `.widget`; the grounded session defers it and records the call.
    func afterToolResult(name: String, result: ToolResult)
    /// End the current turn's trace span on an incomplete turn-ending path (barge-in / `stop()`).
    /// Per-actor: the grounded session falls back to its pending presenter pair for the question.
    func endTurn(error: (any Error)?)
    /// Inbound frame dispatch — the turn brain, divergent enough to stay per-actor.
    func handle(_ frame: String) async
}

extension OpenAIRealtimeSession {
    // MARK: - Session lifecycle (shared)

    func runSession() async throws {
        try await transport.connect()
        let toolDefs = (try? await tools.listTools()) ?? []
        try await transport.send(sessionUpdateFrame(tools: toolDefs))
        await seedHistory()   // replay prior turns before the pump handles any inbound (no-op without a store)
        pump = Task { [weak self] in
            guard let self else { return }
            for await frame in self.transport.events {
                await self.handle(frame)
            }
            await self.transportDidClose()
        }
        emit(.state(modality.input == .speech ? .listening : .ready))
    }

    /// The transport's inbound stream ended without `stop()` — a dead socket (offline, server
    /// drop, an asynchronously-rejected handshake). Without this epilogue the failure is total
    /// silence: the open turn/session spans are never ended (so never exported), and the app's
    /// `events` loop just stalls. Close what's open with the failure, tell the app in-band, and
    /// finish `events` so the consumer gets a real end-of-session signal.
    func transportDidClose() async {
        guard !stopped else { return }   // `stop()` also finishes the stream — that teardown is clean
        stopped = true                    // the session is unusable; bar any late frame from reopening a span
        let cause = await transport.lastReceiveError()
        let error = RealtimeSessionError.transportClosed(underlying: cause.map { String(describing: $0) })
        endTurn(error: error)
        sessionSpan?.end(output: nil, error: error)
        sessionSpan = nil
        emit(.error(code: "transport_closed", message: error.debugDescription))
        continuation.finish()
    }

    /// Close the connection and release the pump. The app must call this when done.
    func endSession() async {
        stopped = true   // first: bar any buffered frame from reopening a span post-teardown
        endTurn(error: nil)
        sessionSpan?.end(output: nil, error: nil)   // close the conversation trace (no-op if no turn opened it)
        sessionSpan = nil
        pump?.cancel()
        await transport.close()
        continuation.finish()
    }

    // MARK: - Shared inbound steps

    func runTool(responseId: String, callId: String, name: String?, arguments: String) async {
        let toolName = name ?? fnNames[callId] ?? ""
        callsPerResponse[responseId, default: 0] += 1
        let args = RealtimeCoding.parseJSON(arguments)
        let span = turnSpan?.span("tool.\(toolName)", input: args)
        let result: ToolResult
        do { result = try await tools.call(toolName, arguments: args) }
        catch { result = .failure("tool \(toolName) failed: \(error)") }
        // A failed call still feeds the model (it decides how to recover), but the span must say
        // it failed — `error: nil` here exported every broken tool call as a success.
        let failure = result.isError ? RealtimeSessionError.toolFailed(RealtimeCoding.failureText(result)) : nil
        span?.end(output: result.structuredContent, error: failure)
        afterToolResult(name: toolName, result: result)
        try? await transport.send(RealtimeWire.functionOutput(callId: callId, output: RealtimeCoding.outputJSON(result)))
    }

    // MARK: - Helpers (shared)

    /// A conversation transcript for a span payload — `nil` when empty, or when `traceTranscripts`
    /// is off, so spoken content stays off the trace while span structure and token usage still flow.
    func transcript(_ text: String) -> JSONValue? {
        traceTranscripts && !text.isEmpty ? .string(text) : nil
    }

    /// Open the turn's span. By default a `voice.turn` child of a lazily-opened `voice.session` root
    /// (the whole conversation is one trace, the root carrying the `modality` marker and living until
    /// `stop()`). With `tracePerTurn`, each turn is instead its own root — no session root is opened —
    /// so it exports the moment the turn ends (grouping into a conversation is then left to shared
    /// metadata, e.g. a thread id, rather than a shared trace). `traceName` overrides the root name.
    func openTurnSpan() -> (any SpanHandle)? {
        let marker: JSONValue = .object(["modality": .string(modality.traceMarker)])
        if tracePerTurn {
            return tracer.startTrace(name: traceName ?? "voice.turn", userId: userId, sessionId: sessionId, metadata: marker)
        }
        if sessionSpan == nil {
            sessionSpan = tracer.startTrace(name: traceName ?? "voice.session", userId: userId, sessionId: sessionId, metadata: marker)
        }
        return sessionSpan?.span("voice.turn", input: nil)
    }

    func emit(_ event: RealtimeEvent) { continuation.yield(event) }

    /// Save the completed turn (user transcript + spoken reply) under `slugify(name)`, when a store
    /// is configured and both halves are non-empty. Only complete pairs are saved — a barge-in or a
    /// silent turn produces no pair. Callers snapshot the strings before resetting.
    func persist(user rawUser: String, reply rawReply: String) async {
        guard let store else { return }
        let user = rawUser.trimmingCharacters(in: .whitespacesAndNewlines)
        let reply = rawReply.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !user.isEmpty, !reply.isEmpty else { return }
        try? await store.saveMessages(
            [ConversationMessage(role: .user, text: user), ConversationMessage(role: .assistant, text: reply)],
            userId: userId, sessionId: sessionId, agentId: slugify(name), maxMessages: maxMessages
        )
    }

    /// Seed prior history into the session for context across connections (text parts only).
    /// Sent as conversation items; they don't trigger a response.
    func seedHistory() async {
        guard let store else { return }
        let history = (try? await store.fetch(userId: userId, sessionId: sessionId,
                                              agentId: slugify(name), maxMessages: maxMessages)) ?? []
        for message in history {
            let text = message.parts.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined(separator: "\n")
            guard !text.isEmpty else { continue }
            switch message.role {
            case .user:      try? await transport.send(RealtimeWire.userMessage(text))
            case .assistant: try? await transport.send(RealtimeWire.assistantMessage(text))
            case .system, .tool: break
            }
        }
    }
}

/// Stateless coding shared by the realtime sessions.
enum RealtimeCoding {
    static func parseJSON(_ raw: String) -> JSONValue {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let value = try? JSONDecoder().decode(JSONValue.self, from: Data(trimmed.utf8)) else {
            return .object([:])
        }
        return value
    }

    /// The tool result fed back to the model: its model-facing text, else its structured data.
    static func outputJSON(_ result: ToolResult) -> String {
        let text = (result.content ?? []).compactMap { part in
            if case .text(let value) = part { return value } else { return nil }
        }.joined(separator: "\n")
        let payload: JSONValue = text.isEmpty ? result.structuredContent : .string(text)
        guard let data = try? JSONEncoder().encode(payload) else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }

    /// A failed result's message, for the tool span's error (`.failure(_:)` puts it in `content`).
    static func failureText(_ result: ToolResult) -> String {
        let text = (result.content ?? []).compactMap { part in
            if case .text(let value) = part { return value } else { return nil }
        }.joined(separator: "\n")
        return text.isEmpty ? "tool reported an error" : text
    }
}
