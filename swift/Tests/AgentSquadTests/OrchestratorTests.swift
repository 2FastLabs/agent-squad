import Foundation
import Testing

@testable import AgentSquad

@Suite struct OrchestratorTests {
    // MARK: - Test doubles

    /// Records that a trace was started; its spans are no-ops.
    private final class RecordingTracer: Tracer, @unchecked Sendable {
        private struct Span: GenerationHandle {
            let id: String
            func span(_ name: String, input: JSONValue?) -> any SpanHandle { Span(id: name) }
            func generation(_ name: String, model: String, input: JSONValue?) -> any GenerationHandle { Span(id: name) }
            func end(output: JSONValue?, error: (any Error)?) {}
            func usage(promptTokens: Int?, completionTokens: Int?) {}
        }
        private let lock = NSLock()
        private var count = 0
        var traceCount: Int { lock.lock(); defer { lock.unlock() }; return count }

        func startTrace(name: String, userId: String?, sessionId: String?, metadata: JSONValue?) -> any SpanHandle {
            lock.lock(); count += 1; lock.unlock()
            return Span(id: "root")
        }
    }

    /// Reports, via its reply text, whether it received a non-nil trace span.
    private struct SpanReportingAgent: AgentProtocol {
        let name = "reporter"
        let description = ""
        func process(_ input: AgentInput, history: [ConversationMessage], context: AgentContext) -> AsyncThrowingStream<AgentEvent, any Error> {
            let hasSpan = context.span != nil
            return AsyncThrowingStream { continuation in
                continuation.yield(.final(ConversationMessage(role: .assistant, text: hasSpan ? "has-span" : "no-span")))
                continuation.finish()
            }
        }
    }

    private struct ThrowingAgent: AgentProtocol {
        struct Boom: Error {}
        let name = "boom"
        let description = ""
        func process(_ input: AgentInput, history: [ConversationMessage], context: AgentContext) -> AsyncThrowingStream<AgentEvent, any Error> {
            AsyncThrowingStream { $0.finish(throwing: Boom()) }
        }
    }

    private struct PickClassifier: Classifier {
        let pick: String?
        func classify(_ input: String, history: [ConversationMessage], agents: [any AgentProtocol]) async throws -> ClassifierResult {
            let agent = agents.first { $0.id == pick }
            return ClassifierResult(selectedAgent: agent, confidence: agent == nil ? 0 : 0.9)
        }
    }

    /// Captures the history it was handed, then routes to `pick`.
    private final class RecordingClassifier: Classifier, @unchecked Sendable {
        let pick: String
        private let lock = NSLock()
        private var history: [ConversationMessage] = []
        init(pick: String) { self.pick = pick }
        func received() -> [ConversationMessage] { lock.withLock { history } }
        func classify(_ input: String, history: [ConversationMessage], agents: [any AgentProtocol]) async throws -> ClassifierResult {
            lock.withLock { self.history = history }
            return ClassifierResult(selectedAgent: agents.first { $0.id == pick }, confidence: 0.9)
        }
    }

    // MARK: - Helpers

    private func store() throws -> DeviceChatStorage { try DeviceChatStorage(userId: "u", inMemory: true) }

    private func sportsAgent(_ events: [LLMStreamEvent]) -> Agent {
        Agent(name: "sports", model: MockLLMClient(events))
    }

    // MARK: - Tests

    @Test func singleAgentTurnRelaysAndPersists() async throws {
        let store = try store()
        let orchestrator = Orchestrator(
            agents: [sportsAgent([.textDelta("hi"), .done(reason: .stop, usage: nil)])],
            store: store
        )
        let events = try await collect(orchestrator.route(.text("q"), userId: "u", sessionId: "s"))
        #expect(events.contains { if case .textDelta = $0 { return true } else { return false } })
        #expect(events.contains { $0.isFinal })

        let persisted = try await store.fetch(userId: "u", sessionId: "s", agentId: "sports", maxMessages: 100)
        #expect(persisted.map(\.role) == [.user, .assistant])
        #expect(persisted.first?.parts == [.text("q")])
        #expect(persisted.last?.parts == [.text("hi")])
    }

    @Test func priorHistoryIsFetchedAndFedToTheAgent() async throws {
        let store = try store()
        try await store.save(.init(role: .user, text: "old"), userId: "u", sessionId: "s", agentId: "sports", maxMessages: 100)
        try await store.save(.init(role: .assistant, text: "oldA"), userId: "u", sessionId: "s", agentId: "sports", maxMessages: 100)

        let model = MockLLMClient([.textDelta("new"), .done(reason: .stop, usage: nil)])
        let orchestrator = Orchestrator(agents: [Agent(name: "sports", model: model)], store: store)
        _ = try await collect(orchestrator.route(.text("now"), userId: "u", sessionId: "s"))

        let sent = model.capturedRequests().first
        #expect(sent?.messages.map(\.parts) == [[.text("old")], [.text("oldA")], [.text("now")]])
    }

    @Test func classifierRoutesToSelectedAgent() async throws {
        let orchestrator = Orchestrator(
            agents: [
                Agent(name: "sports", model: MockLLMClient([.textDelta("sports"), .done(reason: .stop, usage: nil)])),
                Agent(name: "casino", model: MockLLMClient([.textDelta("casino"), .done(reason: .stop, usage: nil)])),
            ],
            classifier: PickClassifier(pick: "casino"),
            store: try store()
        )
        let events = try await collect(orchestrator.route(.text("x"), userId: "u", sessionId: "s"))
        guard case .final(let message)? = events.last(where: { $0.isFinal })
        else { Issue.record("expected .final"); return }
        #expect(message.parts == [.text("casino")])
    }

    @Test func classifierCanActivelySelectTheFirstAgent() async throws {
        // The first agent is the default, but it's also part of the routable pool — the classifier can
        // pick it like any other (not only reach it via fallback). Pins the candidate-set contract.
        let orchestrator = Orchestrator(
            agents: [
                Agent(name: "sports", model: MockLLMClient([.textDelta("sports"), .done(reason: .stop, usage: nil)])),
                Agent(name: "casino", model: MockLLMClient([.textDelta("casino"), .done(reason: .stop, usage: nil)])),
            ],
            classifier: PickClassifier(pick: "sports"),   // actively selects the first/default agent
            store: try store()
        )
        let events = try await collect(orchestrator.route(.text("x"), userId: "u", sessionId: "s"))
        guard case .final(let message)? = events.last(where: { $0.isFinal })
        else { Issue.record("expected .final"); return }
        #expect(message.parts == [.text("sports")])
    }

    @Test func nullClassifierSelectionFallsBackToDefault() async throws {
        let orchestrator = Orchestrator(
            agents: [
                Agent(name: "sports", model: MockLLMClient([.textDelta("sports"), .done(reason: .stop, usage: nil)])),
                Agent(name: "casino", model: MockLLMClient([.textDelta("casino"), .done(reason: .stop, usage: nil)])),
            ],
            classifier: PickClassifier(pick: nil),   // matches nothing → falls back to the first agent
            store: try store()
        )
        let events = try await collect(orchestrator.route(.text("x"), userId: "u", sessionId: "s"))
        guard case .final(let message)? = events.last(where: { $0.isFinal })
        else { Issue.record("expected .final"); return }
        #expect(message.parts == [.text("sports")])   // default
    }

    @Test func agentFailureEmitsErrorAndFinishesWithoutThrowing() async throws {
        let orchestrator = Orchestrator(agents: [ThrowingAgent()], store: try store())
        let events = try await collect(orchestrator.route(.text("x"), userId: "u", sessionId: "s"))
        #expect(events.contains { $0.isError })
    }

    @Test func saveChatFalseSkipsPersistence() async throws {
        let store = try store()
        let agent = Agent(name: "sports", model: MockLLMClient([.textDelta("hi"), .done(reason: .stop, usage: nil)]), saveChat: false)
        _ = try await collect(Orchestrator(agents: [agent], store: store).route(.text("q"), userId: "u", sessionId: "s"))
        let persisted = try await store.fetch(userId: "u", sessionId: "s", agentId: "sports", maxMessages: 100)
        #expect(persisted.isEmpty)
    }

    @Test func classifierReceivesMergedCrossAgentHistory() async throws {
        let store = try store()
        try await store.save(.init(role: .assistant, text: "from sports"), userId: "u", sessionId: "s", agentId: "sports", maxMessages: 100)
        try await store.save(.init(role: .assistant, text: "from casino"), userId: "u", sessionId: "s", agentId: "casino", maxMessages: 100)

        let classifier = RecordingClassifier(pick: "sports")
        let orchestrator = Orchestrator(
            agents: [
                Agent(name: "sports", model: MockLLMClient([.textDelta("ok"), .done(reason: .stop, usage: nil)])),
                Agent(name: "casino", model: MockLLMClient([.textDelta("ok"), .done(reason: .stop, usage: nil)])),
            ],
            classifier: classifier,
            store: store
        )
        _ = try await collect(orchestrator.route(.text("x"), userId: "u", sessionId: "s"))

        let seen = classifier.received().flatMap(\.parts)
        // fetchAllChats prefixes assistant replies with the owning agent's id, merged across agents.
        #expect(seen.contains { if case .text(let t) = $0 { return t.contains("[sports]") } else { return false } })
        #expect(seen.contains { if case .text(let t) = $0 { return t.contains("[casino]") } else { return false } })
    }

    @Test func multipleTurnsAccumulateWithoutDuplicatingUserMessage() async throws {
        let store = try store()
        func run(_ q: String) async throws {
            let orchestrator = Orchestrator(agents: [sportsAgent([.textDelta("a"), .done(reason: .stop, usage: nil)])], store: store)
            _ = try await collect(orchestrator.route(.text(q), userId: "u", sessionId: "s"))
        }
        try await run("q1")
        try await run("q2")
        let persisted = try await store.fetch(userId: "u", sessionId: "s", agentId: "sports", maxMessages: 100)
        #expect(persisted.map(\.role) == [.user, .assistant, .user, .assistant])
    }

    @Test func persistedTurnKeepsUserBeforeAssistantInTimestampOrderedViews() async throws {
        // Regression: the user message must be stamped before the agent runs, or the reply's earlier
        // mid-stream timestamp inverts the pair in the timestamp-sorted `fetchAllChats` merge.
        let store = try store()
        let orchestrator = Orchestrator(agents: [sportsAgent([.textDelta("a"), .done(reason: .stop, usage: nil)])], store: store)
        _ = try await collect(orchestrator.route(.text("q1"), userId: "u", sessionId: "s"))
        _ = try await collect(orchestrator.route(.text("q2"), userId: "u", sessionId: "s"))

        let merged = try await store.fetchAllChats(userId: "u", sessionId: "s")
        #expect(merged.map(\.role) == [.user, .assistant, .user, .assistant])
        let pairs = zip(merged, merged.dropFirst())
        #expect(pairs.allSatisfy { $0.timestamp <= $1.timestamp })
    }

    @Test func createsRootTraceAndPassesSpanToTheAgent() async throws {
        let tracer = RecordingTracer()
        let orchestrator = Orchestrator(agents: [SpanReportingAgent()], store: try store(), tracer: tracer)
        let events = try await collect(orchestrator.route(.text("x"), userId: "u", sessionId: "s"))
        #expect(tracer.traceCount == 1)
        guard case .final(let message)? = events.last(where: { $0.isFinal })
        else { Issue.record("expected .final"); return }
        #expect(message.parts == [.text("has-span")])
    }
}
