import Foundation

/// The 2-LLM grounded pattern as an `AgentProtocol`: a gatherer LLM calls tools (sees the raw
/// results, never speaks), then an isolated presenter LLM writes the answer grounded only on the
/// curated facts, so it can't hallucinate beyond what was fetched. A chit-chat turn that calls no
/// tools is answered in one pass by the gatherer, skipping the presenter.
public struct GroundedAgent: AgentProtocol {
    public let name: String
    public let description: String
    public let saveChat: Bool

    private let gatherer: any LLMClient
    private let presenter: any LLMClient
    private let tools: any ToolProvider
    private let curator: any ToolOutputCurator
    private let gathererPrompt: String?
    private let presenterPrompt: PresenterPrompt
    private let presenterInput: PresenterInput
    private let uiPolicy: UIPolicy
    private let toolRoundCap: Int

    public var maxToolRounds: Int { toolRoundCap }

    public init(
        name: String,
        description: String = "",
        gatherer: any LLMClient,
        presenter: any LLMClient,
        tools: any ToolProvider,
        curator: any ToolOutputCurator = .dataBlock,
        gathererPrompt: String? = nil,
        presenterPrompt: PresenterPrompt = .default,
        presenterInput: PresenterInput = .questionAndData,
        ui: UIPolicy = .forward,
        maxToolRounds: Int = 20,
        saveChat: Bool = true
    ) {
        self.name = name
        self.description = description
        self.gatherer = gatherer
        self.presenter = presenter
        self.tools = tools
        self.curator = curator
        self.gathererPrompt = gathererPrompt
        self.presenterPrompt = presenterPrompt
        self.presenterInput = presenterInput
        self.uiPolicy = ui
        self.toolRoundCap = maxToolRounds
        self.saveChat = saveChat
    }

    public func process(
        _ input: AgentInput,
        history: [ConversationMessage],
        context: AgentContext
    ) -> AsyncThrowingStream<AgentEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await runGrounded(input, history: history, context: context) { continuation.yield($0) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Turn

    private func runGrounded(
        _ input: AgentInput,
        history: [ConversationMessage],
        context: AgentContext,
        emit: @Sendable (AgentEvent) -> Void
    ) async throws {
        guard case .text(let question) = input else { return }
        var phaseSpan: (any SpanHandle)?
        do {
            // 1. Gather — the gatherer runs its tool loop against the capturing provider. Its tool
            //    calls are forwarded for observability; its own draft reply is buffered (used only if
            //    no tools were called). Widgets are suppressed here — we emit the primary one below.
            let capturing = CapturingToolProvider(tools)
            let gathererAgent = Agent(
                name: "\(name).gatherer", model: gatherer, tools: capturing,
                systemPrompt: gathererPrompt, ui: .suppress, maxToolRounds: toolRoundCap, saveChat: false
            )
            phaseSpan = context.span?.span("gatherer", input: nil)
            let gathererContext = AgentContext(userId: context.userId, sessionId: context.sessionId, params: context.params, span: phaseSpan)

            var draft = ""
            for try await event in gathererAgent.process(input, history: history, context: gathererContext) {
                switch event {
                case .toolCall(let id, let name, let arguments): emit(.toolCall(id: id, name: name, arguments: arguments))
                case .final(let message): draft = message.text
                case .textDelta, .thinking, .widget, .error: break
                }
            }
            phaseSpan?.end(output: nil, error: nil); phaseSpan = nil
            let captured = await capturing.captured
            try Task.checkCancellation()   // honor barge-in before the (synchronous) curate/present setup

            // 2. No tools called → chit-chat: speak the gatherer's own reply, skip the presenter.
            //    Re-emitted as one `.textDelta`, since live deltas on a tool-calling turn may include a
            //    suppressed intent line.
            guard !captured.isEmpty else {
                if !draft.isEmpty { emit(.textDelta(draft)) }
                emit(.final(ConversationMessage(role: .assistant, parts: [.text(draft)])))
                return
            }

            // 3. Curate the facts + pick the primary tool (drives the presenter prompt and the widget).
            let feed = curator.curate(captured.map(\.curatorView))
            let primary = Grounding.primary(of: captured)
            if uiPolicy == .forward, let payload = primary?.result.ui {
                emit(.widget(payload))
            }

            // 4. Present — the presenter is fully isolated: no tools, no chat history, no gatherer
            //    transcript. It speaks only from the curated feed (plus the question, by mode), so it
            //    cannot fetch or invent beyond what was gathered this turn.
            let presenterAgent = Agent(
                name: "\(name).presenter", model: presenter, tools: nil,
                systemPrompt: presenterPrompt.resolve(primaryTool: primary?.name), ui: .suppress, saveChat: false
            )
            phaseSpan = context.span?.span("presenter", input: nil)
            let presenterContext = AgentContext(userId: context.userId, sessionId: context.sessionId, params: context.params, span: phaseSpan)

            let presenterMessage = AgentInput.text(Grounding.presenterMessage(question: question, data: feed, mode: presenterInput))
            var finalMessage: ConversationMessage?
            for try await event in presenterAgent.process(presenterMessage, history: [], context: presenterContext) {
                switch event {
                case .textDelta(let delta): emit(.textDelta(delta))
                case .final(let message): finalMessage = message
                case .thinking, .toolCall, .widget, .error: break
                }
            }
            phaseSpan?.end(output: nil, error: nil); phaseSpan = nil
            emit(.final(finalMessage ?? ConversationMessage(role: .assistant, parts: [.text("")])))
        } catch {
            phaseSpan?.end(output: nil, error: error)
            throw error
        }
    }
}
