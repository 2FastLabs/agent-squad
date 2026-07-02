import Foundation

/// Drives a turn end to end: select agent → fetch history → run the agent → relay its events →
/// persist the result, all under a trace.
///
/// Agents are passed as one list — a single agent, or several to route between. The **first agent is
/// the default**: `classifier == nil` routes straight to it with no extra model call, and it's the
/// fallback when the classifier selects nothing.
public struct Orchestrator: Sendable {
    /// The fallback / single-agent target — the first element of `agents`.
    private let defaultAgent: any AgentProtocol
    private let agentsById: [String: any AgentProtocol]
    private let classifier: (any Classifier)?
    private let store: any ChatStorage
    private let tracer: any Tracer
    private let maxMessages: Int?

    /// - Parameters:
    ///   - agents: the agents to serve — **must be non-empty**, with **unique `id`s** (on a duplicate
    ///     id the first occurrence wins and the later one is unroutable). The first agent is the
    ///     default: the single-agent target, and the classifier's fallback when it selects nothing.
    ///   - classifier: routes each turn across the agents. When `nil` there is no routing — **only the
    ///     first agent is used** and any others are ignored (no extra model call).
    public init(
        agents: [any AgentProtocol],
        classifier: (any Classifier)? = nil,
        store: any ChatStorage,
        tracer: any Tracer = OSLogTracer(),
        maxMessages: Int? = ChatStorageDefaults.maxMessages
    ) {
        precondition(!agents.isEmpty, "Orchestrator requires at least one agent — the first is the default/fallback.")
        self.defaultAgent = agents[0]
        self.agentsById = Dictionary(
            agents.map { ($0.id, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        self.classifier = classifier
        self.store = store
        self.tracer = tracer
        self.maxMessages = maxMessages
    }

    public func route(
        _ input: AgentInput,
        userId: String,
        sessionId: String
    ) -> AsyncThrowingStream<AgentEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                await runTurn(input, userId: userId, sessionId: sessionId) { continuation.yield($0) }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Turn

    private func runTurn(
        _ input: AgentInput,
        userId: String,
        sessionId: String,
        emit: @Sendable (AgentEvent) -> Void
    ) async {
        let root = tracer.startTrace(name: "chat.turn", userId: userId, sessionId: sessionId, metadata: .object(["modality": .string("text")]))
        var agentSpan: (any SpanHandle)?
        var sawError = false
        do {
            let agent = try await selectAgent(input, userId: userId, sessionId: sessionId)
            let history = try await store.fetch(userId: userId, sessionId: sessionId, agentId: agent.id, maxMessages: maxMessages)
            // Stamped before the agent runs so timestamp-ordered views (`fetchAllChats`) keep
            // user → assistant order; the reply's message is created later, mid-stream.
            let userMessage = ConversationMessage(role: .user, text: input.text)

            let span = root.span("agent.\(agent.id)", input: nil)
            agentSpan = span
            let context = AgentContext(userId: userId, sessionId: sessionId, span: span)

            var finalMessage: ConversationMessage?
            for try await event in agent.process(input, history: history, context: context) {
                emit(event)
                if case .error = event { sawError = true }
                if case .final(let message) = event { finalMessage = message }
            }
            // Close the agent span on success so a later persist failure isn't misattributed to it,
            // and clear the handle so `catch` can't re-end it.
            span.end(output: nil, error: nil)
            agentSpan = nil

            // Persist user message + reply only on `.final`, so a mid-stream failure leaves no orphaned
            // user message. A failure here is post-answer: record it on the trace but don't emit
            // `.error`, since the user already got their reply.
            if let finalMessage, agent.saveChat {
                do {
                    try await store.saveMessages(
                        [userMessage, finalMessage],
                        userId: userId, sessionId: sessionId, agentId: agent.id, maxMessages: maxMessages
                    )
                } catch {
                    root.end(output: nil, error: error)
                    return
                }
            }
            root.end(output: nil, error: nil)
        } catch {
            // Never crash the stream: record the real error on the trace and surface a friendly one,
            // unless the agent already surfaced its own.
            agentSpan?.end(output: nil, error: error)
            if !sawError {
                emit(.error("Sorry — something went wrong handling that. Please try again."))
            }
            root.end(output: nil, error: error)
        }
    }

    /// `classifier == nil` → the default agent (no model call). Otherwise classify against the
    /// merged history and fall back to the default only on a null selection (confidence is never
    /// thresholded here).
    private func selectAgent(
        _ input: AgentInput,
        userId: String,
        sessionId: String
    ) async throws -> any AgentProtocol {
        guard let classifier else { return defaultAgent }
        let history = try await store.fetchAllChats(userId: userId, sessionId: sessionId)
        let result = try await classifier.classify(input.text, history: history, agents: Array(agentsById.values))
        return result.selectedAgent ?? defaultAgent
    }
}
