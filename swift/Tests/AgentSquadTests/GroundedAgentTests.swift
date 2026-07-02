import Foundation
import Testing

@testable import AgentSquad

@Suite struct GroundedAgentTests {
    // MARK: - Test doubles

    /// Records the captured tools it was handed, then returns a sentinel feed.
    private final class SentinelCurator: ToolOutputCurator, @unchecked Sendable {
        func curate(_ results: [CapturedTool]) -> String {
            "CURATED[" + results.map(\.name).joined(separator: ",") + "]"
        }
    }

    // MARK: - Helpers

    private let context = AgentContext(userId: "u", sessionId: "s")

    /// A gatherer that calls one tool on round 1, then stops on round 2.
    private func gathererCalling(_ tool: String) -> MockLLMClient {
        MockLLMClient(turns: [
            [.toolCall(id: "c1", name: tool, arguments: .object([:])), .done(reason: .toolCalls, usage: nil)],
            [.textDelta("(gatherer draft)"), .done(reason: .stop, usage: nil)]
        ])
    }

    private func oddsProvider(ui: UIPayload? = nil) -> StubToolProvider {
        StubToolProvider(
            tools: [AgentTool(name: "odds", description: "match odds")],
            results: ["odds": ToolResult(content: [.text("PSG 2.5")], structuredContent: .object(["home": .double(2.5)]), ui: ui)]
        )
    }

    // MARK: - Tests

    @Test func presenterIsGroundedOnlyOnTheCuratedFeed() async throws {
        let presenter = MockLLMClient([.textDelta("PSG are 2.5"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter,
            tools: oddsProvider(), curator: SentinelCurator()
        )
        let events = try await collect(agent.process(.text("odds for PSG?"), history: [], context: context))

        #expect(finalText(events) == "PSG are 2.5")

        let request = try #require(presenter.capturedRequests().first)
        #expect(request.tools.isEmpty)   // presenter has no tools — can't fetch or invent
        let presented = request.messages.last?.parts.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined() ?? ""
        #expect(presented.contains("odds for PSG?"))   // the question…
        #expect(presented.contains("CURATED[odds]"))   // …and only the curated feed
    }

    @Test func noToolTurnSpeaksDirectlyAndSkipsThePresenter() async throws {
        let presenter = MockLLMClient([.textDelta("should not run"), .done(reason: .stop, usage: nil)])
        let gatherer = MockLLMClient([.textDelta("hello there"), .done(reason: .stop, usage: nil)])   // no tool call
        let agent = GroundedAgent(
            name: "sport", gatherer: gatherer, presenter: presenter,
            tools: oddsProvider()
        )
        let events = try await collect(agent.process(.text("hi"), history: [], context: context))

        #expect(finalText(events) == "hello there")        // the gatherer's own reply
        #expect(presenter.capturedRequests().isEmpty)       // presenter never called

        // …and it rides the same .textDelta channel as the presenter, so streaming consumers render it.
        let streamed = events.compactMap { if case .textDelta(let t) = $0 { return t } else { return nil } }.joined()
        #expect(streamed == "hello there")
    }

    @Test func gathererWithNoPromptSendsNoSystemPrompt() async throws {
        // GroundedAgent passes its (nil) gatherer prompt verbatim, so the base Agent's default
        // system prompt must NOT leak into the gatherer's LLM call.
        let gatherer = MockLLMClient([.textDelta("hi"), .done(reason: .stop, usage: nil)])
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(name: "sport", gatherer: gatherer, presenter: presenter, tools: oddsProvider())
        _ = try await collect(agent.process(.text("hi"), history: [], context: context))
        #expect(gatherer.capturedRequests().first?.system == nil)
    }

    @Test func emptyDraftChitChatEmitsFinalButNoEmptyDelta() async throws {
        let presenter = MockLLMClient([.textDelta("should not run"), .done(reason: .stop, usage: nil)])
        let gatherer = MockLLMClient([.done(reason: .stop, usage: nil)])   // no tool call, no text
        let agent = GroundedAgent(
            name: "sport", gatherer: gatherer, presenter: presenter,
            tools: oddsProvider()
        )
        let events = try await collect(agent.process(.text("hi"), history: [], context: context))

        #expect(finalText(events) == "")                    // a final is still emitted (for persistence)
        let deltas = events.filter { if case .textDelta = $0 { return true } else { return false } }
        #expect(deltas.isEmpty)                             // …but no zero-length delta
        #expect(presenter.capturedRequests().isEmpty)
    }

    @Test func presenterPromptIsResolvedByPrimaryTool() async throws {
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter,
            tools: oddsProvider(),
            presenterPrompt: PresenterPrompt(default: "DEFAULT", perTool: ["odds": "ODDS PROMPT"])
        )
        _ = try await collect(agent.process(.text("q"), history: [], context: context))
        #expect(presenter.capturedRequests().first?.system == "ODDS PROMPT")
    }

    @Test func curatorOutputIsWhatThePresenterReceives() async throws {
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter,
            tools: oddsProvider(), curator: .dataBlock   // the default faithful curator
        )
        _ = try await collect(agent.process(.text("q"), history: [], context: context))
        let presented = presenter.capturedRequests().first?.messages.last?.parts.compactMap {
            if case .text(let t) = $0 { return t } else { return nil }
        }.joined() ?? ""
        #expect(presented.contains("### odds"))     // DataBlockCurator's per-tool section…
        #expect(presented.contains("PSG 2.5"))      // …carrying the tool's faithful content
    }

    @Test func forwardPolicyEmitsThePrimaryWidget() async throws {
        let ui = UIPayload(resourceURI: "ui://sport/odds", mimeType: "text/html;profile=mcp-app")
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter,
            tools: oddsProvider(ui: ui), ui: .forward
        )
        let events = try await collect(agent.process(.text("q"), history: [], context: context))
        let widget = events.compactMap { if case .widget(let payload) = $0 { return payload } else { return nil } }.first
        #expect(widget?.resourceURI == "ui://sport/odds")
    }

    @Test func suppressPolicyEmitsNoWidgetButStillGroundsTheAnswer() async throws {
        let ui = UIPayload(resourceURI: "ui://sport/odds", mimeType: "text/html;profile=mcp-app")
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter,
            tools: oddsProvider(ui: ui), ui: .suppress
        )
        let events = try await collect(agent.process(.text("q"), history: [], context: context))
        #expect(!events.contains { $0.isWidget })
        // data is redirected into the feed, never dropped
        let presented = presenter.capturedRequests().first?.messages.last?.parts.compactMap {
            if case .text(let t) = $0 { return t } else { return nil }
        }.joined() ?? ""
        #expect(presented.contains("PSG 2.5"))
    }

    @Test func forwardsGathererToolCallsForObservability() async throws {
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter,
            tools: oddsProvider()
        )
        let events = try await collect(agent.process(.text("q"), history: [], context: context))
        #expect(events.contains { if case .toolCall(_, "odds", _) = $0 { return true } else { return false } })
    }

    @Test func primaryIsTheLastUITool_NotJustTheLastTool() async throws {
        // Gatherer calls a UI tool ("odds") first, then a non-UI tool ("stats") last. Primary must
        // be the last tool *with a UI* (odds), driving both the widget and the per-tool prompt.
        let ui = UIPayload(resourceURI: "ui://sport/odds", mimeType: "text/html;profile=mcp-app")
        let provider = StubToolProvider(
            tools: [AgentTool(name: "odds", description: ""), AgentTool(name: "stats", description: "")],
            results: [
                "odds": ToolResult(content: [.text("PSG 2.5")], ui: ui),
                "stats": ToolResult(content: [.text("possession 60%")])   // no UI, called last
            ]
        )
        let gatherer = MockLLMClient(turns: [
            [.toolCall(id: "c1", name: "odds", arguments: .object([:])),
             .toolCall(id: "c2", name: "stats", arguments: .object([:])),
             .done(reason: .toolCalls, usage: nil)],
            [.done(reason: .stop, usage: nil)]
        ])
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gatherer, presenter: presenter, tools: provider,
            presenterPrompt: PresenterPrompt(default: "DEF", perTool: ["odds": "ODDS", "stats": "STATS"])
        )
        let events = try await collect(agent.process(.text("q"), history: [], context: context))

        let widget = events.compactMap { if case .widget(let p) = $0 { return p } else { return nil } }.first
        #expect(widget?.resourceURI == "ui://sport/odds")              // last UI tool, not last tool
        #expect(presenter.capturedRequests().first?.system == "ODDS")   // prompt keyed on the primary
    }

    @Test func gathererDraftIsNotLeakedWhenToolsWereCalled() async throws {
        // gathererCalling emits "(gatherer draft)" as its .final — it must NOT reach the user.
        let presenter = MockLLMClient([.textDelta("grounded answer"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter, tools: oddsProvider()
        )
        let events = try await collect(agent.process(.text("q"), history: [], context: context))
        #expect(finalText(events) == "grounded answer")
        #expect(!(finalText(events) ?? "").contains("gatherer draft"))
    }

    @Test func gathererFailurePropagates() async throws {
        let agent = GroundedAgent(
            name: "sport", gatherer: FailingLLMClient(),
            presenter: MockLLMClient([.done(reason: .stop, usage: nil)]), tools: oddsProvider()
        )
        await #expect(throws: (any Error).self) {
            _ = try await collect(agent.process(.text("q"), history: [], context: context))
        }
    }

    @Test func presenterFailurePropagates() async throws {
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"),
            presenter: FailingLLMClient(), tools: oddsProvider()
        )
        await #expect(throws: (any Error).self) {
            _ = try await collect(agent.process(.text("q"), history: [], context: context))
        }
    }

    @Test func presenterNeverSeesChatHistory() async throws {
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter, tools: oddsProvider()
        )
        let history = [ConversationMessage(role: .user, text: "earlier turn")]
        _ = try await collect(agent.process(.text("q"), history: history, context: context))

        let messages = presenter.capturedRequests().first?.messages ?? []
        let texts = messages.flatMap(\.parts).compactMap { if case .text(let t) = $0 { return t } else { return nil } }
        #expect(!texts.joined().contains("earlier turn"))
        #expect(messages.count == 1)   // exactly the curated feed message, nothing else
    }

    @Test func presenterMessageCarriesTheQuestionByDefault() async throws {
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter, tools: oddsProvider()
        )
        _ = try await collect(agent.process(.text("odds for PSG?"), history: [], context: context))

        let text = presenter.capturedRequests().first?.messages.first?.text ?? ""
        #expect(text.contains("<user question>"))
        #expect(text.contains("odds for PSG?"))
        #expect(text.contains("<data to present>"))
    }

    @Test func dataOnlyPresenterInputOmitsTheQuestion() async throws {
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter,
            tools: oddsProvider(), presenterInput: .dataOnly
        )
        _ = try await collect(agent.process(.text("odds for PSG?"), history: [], context: context))

        let text = presenter.capturedRequests().first?.messages.first?.text ?? ""
        #expect(!text.contains("<user question>"))
        #expect(!text.contains("odds for PSG?"))
        #expect(text.contains("<data to present>"))
    }

    @Test func widgetIsEmittedBeforeTheAnswerText() async throws {
        let ui = UIPayload(resourceURI: "ui://sport/odds", mimeType: "text/html;profile=mcp-app")
        let presenter = MockLLMClient([.textDelta("answer"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter,
            tools: oddsProvider(ui: ui), ui: .forward
        )
        let events = try await collect(agent.process(.text("q"), history: [], context: context))
        let widgetIndex = events.firstIndex { if case .widget = $0 { return true } else { return false } }
        let textIndex = events.firstIndex { if case .textDelta = $0 { return true } else { return false } }
        #expect(widgetIndex != nil && textIndex != nil && widgetIndex! < textIndex!)
    }

    @Test func toolErrorResultIsPresentedNotTreatedAsChitChat() async throws {
        let provider = StubToolProvider(
            tools: [AgentTool(name: "odds", description: "")],
            results: ["odds": .failure("odds unavailable")]   // isError result — still a capture
        )
        let presenter = MockLLMClient([.textDelta("x"), .done(reason: .stop, usage: nil)])
        let agent = GroundedAgent(
            name: "sport", gatherer: gathererCalling("odds"), presenter: presenter, tools: provider
        )
        _ = try await collect(agent.process(.text("q"), history: [], context: context))

        #expect(!presenter.capturedRequests().isEmpty)   // presenter ran (not the no-tool path)
        let presented = presenter.capturedRequests().first?.messages.last?.parts.compactMap {
            if case .text(let t) = $0 { return t } else { return nil }
        }.joined() ?? ""
        #expect(presented.contains("odds unavailable"))
    }
}
