import Foundation

/// The single-LLM realtime voice runtime. One WebSocket (behind `RealtimeTransport`); the agent
/// turn uses audio output + tools and speaks the answer directly: call the tools it needs, then
/// reply aloud. For the grounded, anti-hallucination two-phase pattern use
/// `OpenAIGroundedVoiceAssistant`.
///
/// A self-sufficient peer runtime (not an `AgentProtocol`): like `Orchestrator` it owns its tracing
/// and persistence, scoped by `userId`/`sessionId`. With a `store`, each completed turn (user
/// transcript + spoken reply) is saved under `slugify(name)`, and prior history is seeded on `start()`.
/// Shared session plumbing lives in `OpenAIRealtimeSession`; this type holds the single-response turn brain.
public actor OpenAIVoiceAssistant: OpenAIRealtimeSession, VoiceAssistant {
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
    private let instructions: String
    nonisolated let model: String
    nonisolated let voice: String
    nonisolated let language: String?
    nonisolated let sampleRate: Int
    private let transcriptionModel: String
    private let turnDetection: RealtimeTurnDetection
    private let sessionOverrides: [String: JSONValue]

    // Per-turn state (cleared by `resetTurn`).
    private var userText = ""
    private var replyText = ""   // the spoken reply, captured for persistence in every output mode
    var callsPerResponse: [String: Int] = [:]
    var fnNames: [String: String] = [:]
    // A typed turn (`sendText`) replies text-only — so we don't announce `.speaking`, and the
    // tool→continue response stays text-only too.
    private var currentTurnIsTextOnly = false
    // Audio-relay gating: the spoken response is the bare agent turn, so track the current agent
    // response id and relay its audio, re-pointing on every `response.created` across the
    // tool→continue loop. On barge-in `currentResponseId` is cleared so late audio drops.
    private var liveResponses: Set<String> = []
    private var currentResponseId: String?
    private var isSpeaking = false

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
        instructions: String = OpenAIVoiceAssistant.defaultInstructions,
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
        self.instructions = instructions
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

    // MARK: - Session seams

    func sessionUpdateFrame(tools: [AgentTool]) -> String {
        // The agent turn speaks: its session-level output modality matches ours (audio / text).
        RealtimeWire.sessionUpdate(
            model: model, instructions: instructions, voice: voice,
            language: language, tools: tools, sampleRate: sampleRate, agentOutput: modality.output,
            transcriptionModel: transcriptionModel, turnDetection: turnDetection, overrides: sessionOverrides
        )
    }

    func afterToolResult(name: String, result: ToolResult) {
        // A tool may advertise UI — surface it as a widget alongside the spoken reply.
        if let payload = result.ui { emit(.widget(payload)) }
    }

    // MARK: - VoiceAssistant (turn brain)

    public func sendText(_ text: String) async {
        // New-turn boundary (text mode has no VAD `speech_started`): close the prior turn.
        // `interrupt()` ends its span; otherwise force-close a stale span so the new turn doesn't leak it.
        if isSpeaking { await interrupt() } else { endTurn(error: nil) }
        userText = text
        currentTurnIsTextOnly = true   // a typed turn replies in text, not speech
        emit(.userTranscript(text, final: true))
        try? await transport.send(RealtimeWire.userMessage(text))
        // Typed turns reply text-only; spoken/VAD turns keep the session default (audio). Tools are
        // unaffected — output modality doesn't change availability.
        try? await transport.send(RealtimeWire.createResponse(output: .text))   // manual turn (no VAD in text mode)
        emit(.state(.thinking))
    }

    public func interrupt() async {
        endTurn(error: nil)            // close the turn's trace span (barge-in is the common path)
        resetTurn()
        currentResponseId = nil        // arm the stale-drop: later audio for the cancelled response is ignored
        let wasSpeaking = isSpeaking
        isSpeaking = false             // clear before the await so an interleaved frame sees the final state
        if wasSpeaking {
            try? await transport.send(RealtimeWire.cancelResponse())
        }
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
            if responseId == currentResponseId { emit(.presenterText(text, final: false)) }
        case .outputTextDone(let responseId, let text):
            if responseId == currentResponseId { replyText = text; emit(.presenterText(text, final: true)) }
        case .audioDelta(let responseId, let base64):
            if responseId == currentResponseId, let data = Data(base64Encoded: base64) { emit(.audio(data)) }
        case .audioTranscriptDelta(let responseId, let text):
            if responseId == currentResponseId, modality.output == .audioAndText { emit(.presenterText(text, final: false)) }
        case .audioTranscriptDone(let responseId, let text):
            // Capture the spoken reply for persistence even in pure-audio mode; emit only when asked.
            if responseId == currentResponseId {
                replyText = text
                if modality.output == .audioAndText { emit(.presenterText(text, final: true)) }
            }
        case .responseCreated(let id, _):
            // The spoken response is the bare agent turn (no role) — relay its audio. Re-point on
            // every response across the tool→continue loop; open the turn span once.
            guard !id.isEmpty else { break }
            if turnSpan == nil {   // first response of the turn — open the turn span, announce speaking once
                turnSpan = openTurnSpan()
                if !currentTurnIsTextOnly { emit(.state(.speaking)) }   // a typed turn replies in text — not speaking
            }
            liveResponses.insert(id)
            currentResponseId = id
            isSpeaking = true
        case .responseDone(let id, let usage):
            await responseDone(id, usage: usage)
        case .error(let code):
            if code == "response_cancel_not_active" { return }   // benign barge-in race
            emit(.error(code ?? "realtime error"))
        }
    }

    private func responseDone(_ id: String, usage: RealtimeUsage?) async {
        guard liveResponses.contains(id) else { return }   // unknown / stale (cancelled) response
        liveResponses.remove(id)
        if let calls = callsPerResponse[id], calls > 0 {    // this response called tools → continue the turn
            callsPerResponse[id] = nil
            // Keep the continue response in the turn's modality — a typed turn stays text-only.
            try? await transport.send(currentTurnIsTextOnly ? RealtimeWire.createResponse(output: .text) : RealtimeWire.createResponse())
            return
        }
        // A response that called no tools is the spoken answer — the turn is complete.
        guard id == currentResponseId else { return }
        currentResponseId = nil
        isSpeaking = false
        // Snapshot the pair before the store await — a next turn interleaving during it (e.g. a typed
        // `sendText`) would otherwise have its captured state wiped by resetTurn.
        let (user, reply) = (userText, replyText)
        // Record the spoken exchange (transcript in/out + usage) as a generation under the turn, then
        // close the turn with the reply — so the trace shows what was said, not an empty span.
        if let turn = turnSpan {
            let exchange = turn.generation("response", model: model, input: transcript(user))
            exchange.usage(promptTokens: usage?.inputTokens, completionTokens: usage?.outputTokens)
            if let breakdown = usage?.breakdownMetadata { exchange.setMetadata(breakdown) }
            exchange.end(output: transcript(reply), error: nil)
            // The user transcript arrives after the turn span opens, so set the root's input now.
            if let input = transcript(user) { turn.setInput(input) }
            turn.end(output: transcript(reply), error: nil)
            turnSpan = nil
        }
        emit(.audioDone(interrupted: false))
        emit(.state(.listening))
        resetTurn()
        await persist(user: user, reply: reply)
    }

    // MARK: - Turn lifecycle

    /// End the current turn's trace span (idempotent) on an incomplete turn-ending path — barge-in,
    /// a stale-span close, or `stop()` (a clean finish closes the span itself, with its generation).
    /// Records what the user asked and any partial reply first, so an interrupted turn isn't empty.
    func endTurn(error: (any Error)?) {
        guard let turn = turnSpan else { return }
        if let input = transcript(userText) { turn.setInput(input) }
        turn.end(output: transcript(replyText), error: error)
        turnSpan = nil
    }

    private func resetTurn() {
        userText = ""
        replyText = ""
        callsPerResponse.removeAll()
        fnNames.removeAll()
        currentTurnIsTextOnly = false   // next turn defaults to the session modality unless `sendText` opts in
        // liveResponses / currentResponseId / isSpeaking are managed by responseDone / interrupt.
    }

    public static let defaultInstructions = """
        You are a friendly, concise voice assistant. Use the available tools when they help answer the \
        user, then reply naturally and concisely in the user's language. Keep replies short and spoken.
        """
}
