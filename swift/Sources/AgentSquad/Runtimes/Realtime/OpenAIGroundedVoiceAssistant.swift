import Foundation

/// The OpenAI Realtime grounded session — a peer runtime of `Orchestrator`, not an `AgentProtocol`.
/// One WebSocket (behind `RealtimeTransport`) runs both halves of the grounded pattern as two
/// `response.create`s: a text-only **gatherer** that calls tools, then an isolated **presenter** that
/// speaks grounded only on the curated facts. A no-tool turn answers directly. Voice and text are one
/// engine — `RealtimeModality` flips the response output and whether turns are VAD- or text-triggered.
///
/// Reuses the shared grounding core (`Grounding`, `ToolOutputCurator`, `PresenterPrompt`) verbatim, and
/// the shared session plumbing in `OpenAIRealtimeSession`; this type holds the two-phase turn brain.
public actor OpenAIGroundedVoiceAssistant: OpenAIRealtimeSession, VoiceAssistant {
    public nonisolated let modality: RealtimeModality
    public nonisolated let events: AsyncStream<RealtimeEvent>
    nonisolated let continuation: AsyncStream<RealtimeEvent>.Continuation

    nonisolated let name: String
    nonisolated let transport: any RealtimeTransport
    nonisolated let tools: any ToolProvider
    nonisolated let tracer: any Tracer
    nonisolated let userId: String
    nonisolated let sessionId: String
    nonisolated let store: (any ChatStorage)?
    nonisolated let maxMessages: Int?
    nonisolated let traceTranscripts: Bool
    private let curator: any ToolOutputCurator
    private let presenterPrompt: PresenterPrompt
    private let presenterInput: PresenterInput
    private let agentInstructions: String
    private let directInstructions: String
    nonisolated let model: String
    nonisolated let voice: String
    nonisolated let language: String?
    nonisolated let sampleRate: Int
    private let transcriptionModel: String
    private let turnDetection: RealtimeTurnDetection
    private let sessionOverrides: [String: JSONValue]

    // Per-turn state (cleared by `resetTurn`).
    private var toolResults: [CapturedCall] = []
    private var userText = ""
    private var agentText = ""
    var callsPerResponse: [String: Int] = [:]
    var fnNames: [String: String] = [:]
    // Cross-turn state — survives `resetTurn` to gate incoming deltas to the response we're hearing.
    private var presenterId: String?
    private var presenterResponses: Set<String> = []
    private var presenterActive = false
    // Barge-in truncation: which audio item is playing + how much was sent, vs. the runtime's
    // played-ms clock. NOT cleared by `resetTurn` — `interrupt()` reads it after resetting the turn.
    // Only the in-band `direct` reply is truncatable: the presenter is out-of-band
    // (`conversation: "none"`), its items never enter the conversation, and truncating one errors.
    private var truncation = AudioTruncationState()
    private var spokenResponseIsInBand = false
    private var playbackClock: (@Sendable () async -> Double?)?
    // The turn's user question + spoken reply, captured for persistence. They survive `resetTurn`
    // because present()/speakDirectly() reset before the presenter speaks (the reset-before-await).
    private var pendingUserText = ""
    private var replyText = ""

    var sessionSpan: (any SpanHandle)?
    var turnSpan: (any SpanHandle)?
    var pump: Task<Void, Never>?
    var stopped = false

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
        presenterInput: PresenterInput = .questionAndData,
        agentInstructions: String = OpenAIGroundedVoiceAssistant.defaultAgentInstructions,
        directInstructions: String = OpenAIGroundedVoiceAssistant.defaultDirectInstructions,
        model: String = "gpt-realtime",
        voice: String = "marin",
        language: String? = nil,
        sampleRate: Int = 24_000,
        transcriptionModel: String = "gpt-4o-mini-transcribe",
        turnDetection: RealtimeTurnDetection = .semanticVAD(),
        sessionOverrides: [String: JSONValue] = [:]
    ) {
        self.name = name
        self.transport = transport
        self.tools = tools
        self.userId = userId
        self.sessionId = sessionId
        self.store = store
        self.maxMessages = maxMessages
        self.tracer = tracer
        self.traceTranscripts = traceTranscripts
        self.modality = modality
        self.curator = curator
        self.presenterPrompt = presenterPrompt
        self.presenterInput = presenterInput
        self.agentInstructions = agentInstructions
        self.directInstructions = directInstructions
        self.model = model
        self.voice = voice
        self.language = language
        self.sampleRate = sampleRate
        self.transcriptionModel = transcriptionModel
        self.turnDetection = turnDetection
        self.sessionOverrides = sessionOverrides
        (self.events, self.continuation) = AsyncStream.makeStream(of: RealtimeEvent.self)
    }

    // MARK: - VoiceAssistant (public surface)

    public func start() async throws { try await runSession() }
    public func stop() async { await endSession() }
    public func sendAudio(_ pcm16: Data) async {
        try? await transport.send(RealtimeWire.appendAudio(pcm16.base64EncodedString()))
    }
    public func setPlaybackClock(_ playedMilliseconds: @escaping @Sendable () async -> Double?) async {
        playbackClock = playedMilliseconds
    }

    // MARK: - Session seams

    func sessionUpdateFrame(tools: [AgentTool]) -> String {
        RealtimeWire.sessionUpdate(
            model: model, instructions: agentInstructions, voice: voice,
            language: language, tools: tools, sampleRate: sampleRate,
            transcriptionModel: transcriptionModel, turnDetection: turnDetection, overrides: sessionOverrides
        )
    }

    func afterToolResult(name: String, result: ToolResult) {
        toolResults.append(CapturedCall(name: name, result: result))   // widget deferred to present()
    }

    // MARK: - VoiceAssistant (turn brain)

    public func sendText(_ text: String) async {
        // New-turn boundary (text mode has no VAD `speech_started`): close the prior turn.
        // `interrupt()` ends its span; otherwise force-close a stale span so the new turn doesn't leak it.
        if presenterActive { await interrupt() } else { endTurn(error: nil) }
        userText = text
        emit(.userTranscript(text, final: true))
        try? await transport.send(RealtimeWire.userMessage(text))
        try? await transport.send(RealtimeWire.createResponse())   // manual agent turn (no VAD in text mode)
        emit(.state(.thinking))
    }

    public func interrupt() async {
        endTurn(error: nil)   // barge-in (the common path) must close the turn's trace span
        resetTurn()
        pendingUserText = ""   // abandon the in-flight turn's pending persistence
        replyText = ""
        presenterId = nil   // arms the stale-drop: later deltas for the cancelled response no longer match
        let wasActive = presenterActive
        presenterActive = false   // clear before the await so an interleaved frame sees the final state
        if wasActive {
            // Barge-in steps 2+3 (WebSocket): the clock closure reports what was heard and cuts
            // playback; the truncate then lets the server drop the unplayed audio + transcript.
            if let clock = playbackClock, let frame = truncation.truncateFrame(playedMs: await clock()) {
                try? await transport.send(frame)
            }
            try? await transport.send(RealtimeWire.cancelResponse())
        }
        truncation.reset()
        // Always emit, even when nothing was playing (a fresh-turn speech_started also lands here) —
        // the consumer treats a flush-when-idle as a no-op.
        emit(.audioDone(interrupted: true))
        emit(.state(.listening))
    }

    // MARK: - Inbound dispatch

    func handle(_ frame: String) async {
        guard !stopped else { return }   // ignore frames delivered after teardown — never reopen a span
        guard let event = RealtimeWire.decode(frame) else { return }
        switch event {
        case .speechStarted:
            await interrupt()
        case .userTranscriptDelta(let text):
            emit(.userTranscript(text, final: false))
        case .userTranscriptCompleted(let text):
            userText = text
            emit(.userTranscript(text, final: true))
        case .functionCallNamed(let callId, let name):
            fnNames[callId] = name
        case .functionCallArguments(let responseId, let callId, let name, let arguments):
            await runTool(responseId: responseId, callId: callId, name: name, arguments: arguments)
        case .outputTextDelta(let responseId, let text):
            if responseId == presenterId { emit(.presenterText(text, final: false)) } else { agentText += text }
        case .outputTextDone(let responseId, let text):
            if responseId == presenterId { replyText = text; emit(.presenterText(text, final: true)) }
        case .audioDelta(let responseId, let itemId, let base64):
            if responseId == presenterId, let data = Data(base64Encoded: base64) {
                if spokenResponseIsInBand {   // presenter is out-of-band — nothing to truncate server-side
                    truncation.record(itemId: itemId, pcm16ByteCount: data.count, sampleRate: sampleRate)
                }
                emit(.audio(data))
            }
        case .audioTranscriptDelta(let responseId, let text):
            if responseId == presenterId, modality.output == .audioAndText { emit(.presenterText(text, final: false)) }
        case .audioTranscriptDone(let responseId, let text):
            if responseId == presenterId {
                replyText = text   // capture for persistence even in pure-audio mode
                if modality.output == .audioAndText { emit(.presenterText(text, final: true)) }
            }
        case .responseCreated(let id, let role):
            if turnSpan == nil, !id.isEmpty {   // first response of the turn (the gatherer) opens the turn span
                turnSpan = openTurnSpan()
            }
            // The server emits `response.created` before any delta for that response, so setting the
            // gate here is what lets the presenter/direct deltas through (and the agent's — which never
            // gets a tracked id — accumulate silently). Guard empty so an id-less frame can't open the gate.
            if (role == "presenter" || role == "direct"), !id.isEmpty {
                presenterResponses.insert(id)
                presenterId = id
                spokenResponseIsInBand = role == "direct"
            }
        case .responseDone(let id, let usage):
            await responseDone(id, usage: usage)
        case .error(let code):
            if code == "response_cancel_not_active" { return }   // benign barge-in race
            emit(.error(code ?? "realtime error"))
        }
    }

    private func responseDone(_ id: String, usage: RealtimeUsage?) async {
        if presenterResponses.contains(id) {
            presenterResponses.remove(id)
            if id == presenterId {   // the response we're hearing finished cleanly
                presenterId = nil
                presenterActive = false
                truncation.reset()   // clean finish — nothing left to truncate
                let (user, reply) = (pendingUserText, replyText)   // snapshot + clear before the store await
                // Record the spoken answer as a `presenter` generation (question in, reply out, presenter
                // usage) under the turn, then close the turn. The gatherer's tool calls show as the turn's
                // `tool.*` child spans, but its token usage is dropped this pass — so this count is the
                // presenter's, not the turn total.
                if let turn = turnSpan {
                    let presenter = turn.generation("presenter", model: model, input: transcript(user))
                    presenter.usage(promptTokens: usage?.inputTokens, completionTokens: usage?.outputTokens)
                    if let breakdown = usage?.breakdownMetadata { presenter.setMetadata(breakdown) }
                    presenter.end(output: transcript(reply), error: nil)
                    // The user transcript arrives after the turn span opens, so set the root's input now.
                    if let input = transcript(user) { turn.setInput(input) }
                    turn.end(output: transcript(reply), error: nil)
                    turnSpan = nil
                }
                emit(.audioDone(interrupted: false))
                emit(.state(.listening))
                pendingUserText = ""
                replyText = ""
                await persist(user: user, reply: reply)
            }
            // else: a response we cancelled on barge-in (`presenterId` already cleared). The server still
            // emits its `done`, consumed here — to keep `presenterResponses` bounded and so it isn't
            // mistaken for an agent turn settling below.
            return
        }
        if let calls = callsPerResponse[id], calls > 0 {   // agent issued tools; outputs are in → continue
            callsPerResponse[id] = nil
            try? await transport.send(RealtimeWire.createResponse())
            return
        }
        // Agent turn settled: present what was gathered, or speak directly when nothing was.
        if toolResults.isEmpty { await speakDirectly() } else { await present() }
    }

    // MARK: - The two grounded responses (shared core)

    private func present() async {
        let feed = curator.curate(toolResults.map(\.curatorView))
        let primary = Grounding.primary(of: toolResults)
        if let payload = primary?.result.ui { emit(.widget(payload)) }
        let prompt = presenterPrompt.resolve(primaryTool: primary?.name)
        let message = Grounding.presenterMessage(question: userText, data: feed, mode: presenterInput)

        emit(.state(.presenting))
        presenterActive = true
        pendingUserText = userText   // capture for persistence before resetTurn wipes the turn state
        replyText = ""
        let frame = RealtimeWire.presenterResponse(instructions: prompt, feed: message, output: modality.output, voice: voice)
        // Reset BEFORE the await: `send` suspends the actor, so a next-turn frame could interleave —
        // resetting after would wipe the new turn's captured state. Everything we need is in `frame`.
        resetTurn()
        try? await transport.send(frame)
    }

    private func speakDirectly() async {
        guard !agentText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            endTurn(error: nil)        // nothing to say (e.g. noise) — close the turn's trace
            emit(.state(.listening))   // stay silent
            resetTurn()
            return
        }
        emit(.state(.speaking))
        presenterActive = true
        pendingUserText = userText   // capture for persistence before resetTurn (see present())
        replyText = ""
        let frame = RealtimeWire.directResponse(instructions: directInstructions, output: modality.output, voice: voice)
        resetTurn()   // before the await — see present()
        try? await transport.send(frame)
    }

    // MARK: - Turn lifecycle

    /// End the current turn's trace span (idempotent) on an incomplete turn-ending path — barge-in,
    /// the no-text silent return, or `stop()` (a clean presenter finish closes the span itself). Records
    /// the user's question — from the gather turn (`userText`) or the pending presenter pair
    /// (`pendingUserText`, after `resetTurn`) — and any partial reply, so an interrupted turn isn't empty.
    func endTurn(error: (any Error)?) {
        guard let turn = turnSpan else { return }
        let question = userText.isEmpty ? pendingUserText : userText
        if let input = transcript(question) { turn.setInput(input) }
        turn.end(output: transcript(replyText), error: error)
        turnSpan = nil
    }

    private func resetTurn() {
        toolResults.removeAll()
        userText = ""
        agentText = ""
        callsPerResponse.removeAll()
        fnNames.removeAll()
        // presenterId / presenterResponses / presenterActive / pendingUserText / replyText intentionally
        // survive — the presenter speaks after this reset, and persistence reads them when it finishes.
    }

    public static let defaultAgentInstructions = """
        You gather the facts needed to answer the user by calling the available tools. Do NOT speak \
        the final answer yourself — a separate presenter will. Call the tools you need and nothing else.
        """

    public static let defaultDirectInstructions = """
        You are a friendly, concise voice assistant. Reply naturally to the user. Do not call tools.
        """
}
