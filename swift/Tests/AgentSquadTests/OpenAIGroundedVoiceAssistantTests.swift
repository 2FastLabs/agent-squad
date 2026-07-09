import Foundation
import Testing

@testable import AgentSquad

@Suite struct OpenAIGroundedVoiceAssistantTests {
    private func session(_ transport: MockRealtimeTransport, tools: any ToolProvider, modality: RealtimeModality = RealtimeModality(input: .speech, output: .audio), tracer: any Tracer = OSLogTracer(), store: (any ChatStorage)? = nil) -> OpenAIGroundedVoiceAssistant {
        OpenAIGroundedVoiceAssistant(name: "grounded", transport: transport, tools: tools, userId: "u1", sessionId: "s1", store: store, tracer: tracer, modality: modality, presenterPrompt: PresenterPrompt(default: "DEFAULT", perTool: ["odds": "ODDS PROMPT"]))
    }

    @Test func startSendsSessionUpdateAndListens() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        await eventually { sentTypes(transport).contains("session.update") }
        #expect(sentTypes(transport).first == "session.update")
        await eventually { log.states.contains(.listening) }
    }

    @Test func voiceTurnGathersThenPresentsGroundedWithWidget() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        // Agent calls the odds tool, settles, then a continue round settles with no more tools.
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))   // tools were called → continue agent turn
        transport.push(responseDone("r2"))   // settled with results → present

        await eventually { sentTypes(transport).filter { $0 == "response.create" }.count >= 2 }
        let types = sentTypes(transport)
        #expect(types.contains("conversation.item.create"))   // function_call_output fed back
        #expect(types.filter { $0 == "response.create" }.count == 2)   // continue + presenter

        // The presenter response.create is grounded + out-of-band, with the primary tool's prompt.
        let presenter = transport.sent.first { $0.contains("\"role\":\"presenter\"") }
        let presenterObj = try #require(presenter.flatMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] })
        let response = try #require(presenterObj["response"] as? [String: Any])
        #expect(response["instructions"] as? String == "ODDS PROMPT")
        let feed = (((response["input"] as? [[String: Any]])?.first?["content"]) as? [[String: Any]])?.first?["text"] as? String
        #expect(feed?.contains("PSG 2.5") == true)            // grounded on the curated data

        await eventually { !log.widgets.isEmpty }
        #expect(log.widgets.first?.resourceURI == "ui://odds")
        #expect(log.states.contains(.presenting))
    }

    @Test func relaysPresenterAudioAndTextThenFinishesListening() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools(), modality: RealtimeModality(input: .speech, output: .audioAndText))
        log.start(session)
        try await session.start()

        transport.push(responseCreated("p1", role: "presenter"))   // now hearing p1
        transport.push(audioDelta("p1", "pcm"))
        transport.push(#"{"type":"response.output_audio_transcript.delta","response_id":"p1","delta":"PSG are 2.5"}"#)
        transport.push(responseDone("p1"))

        await eventually { log.audioDones.contains(false) }
        #expect(log.audioCount == 1)
        #expect(log.presenterTexts.contains("PSG are 2.5"))   // audio+text mode relays transcript
        #expect(log.states.last == .listening)
    }

    @Test func presenterBargeInSendsNoTruncate() async throws {
        // The presenter is out-of-band (`conversation: "none"`) — its items never enter the
        // conversation, so truncating one is a server error. Barge-in must skip the frame.
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        await session.setPlaybackClock { 40 }
        log.start(session)
        try await session.start()

        // Drive a real grounded turn so the presenter is active (tool → settle → present).
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))
        transport.push(responseDone("r2"))
        await eventually { transport.sent.contains { $0.contains("\"role\":\"presenter\"") } }
        transport.push(responseCreated("p1", role: "presenter"))
        transport.push(audioDelta("p1", String(repeating: "x", count: 4_800)))   // 100 ms received
        await eventually { log.audioCount == 1 }

        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)
        await eventually { sentTypes(transport).contains("response.cancel") }
        #expect(!sentTypes(transport).contains("conversation.item.truncate"))
    }

    @Test func directReplyBargeInTruncatesTheHeardPortion() async throws {
        // The direct (no-tool) reply is in-band — barge-in truncates like the plain assistant.
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        await session.setPlaybackClock { 40 }
        log.start(session)
        try await session.start()

        // Agent settles with no tools but with text → speakDirectly() creates the direct reply.
        transport.push(responseCreated("r1"))
        transport.push(#"{"type":"response.output_text.delta","response_id":"r1","delta":"hi"}"#)
        transport.push(responseDone("r1"))
        await eventually { transport.sent.contains { $0.contains("\"role\":\"direct\"") } }
        transport.push(responseCreated("d1", role: "direct"))
        transport.push(audioDelta("d1", String(repeating: "x", count: 4_800), item: "item_3"))
        await eventually { log.audioCount == 1 }

        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)
        await eventually { sentTypes(transport).contains("response.cancel") }
        let frame = try #require(transport.sent.first { $0.contains("conversation.item.truncate") })
        #expect(frame.contains(#""item_id":"item_3""#))
        #expect(frame.contains(#""audio_end_ms":40"#))
    }

    @Test func agentTextIsAccumulatedNotEmittedWhileNoPresenterIsActive() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        // No presenter id set yet → this text belongs to the (text-only) agent turn, not the user.
        transport.push(#"{"type":"response.output_text.delta","response_id":"agent","delta":"thinking..."}"#)
        try? await Task.sleep(for: .milliseconds(20))
        #expect(log.presenterTexts.isEmpty)   // agent text never surfaces as presenter text
    }

    @Test func noToolTurnSpeaksDirectly() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        transport.push(#"{"type":"response.output_text.delta","response_id":"agent","delta":"hello there"}"#)  // agent answers directly
        transport.push(responseDone("agent"))   // settled, no tools → speak directly

        await eventually { sentTypes(transport).filter { $0 == "response.create" }.count >= 1 }
        let direct = transport.sent.first { $0.contains("\"role\":\"direct\"") }
        #expect(direct != nil)
        await eventually { log.states.contains(.speaking) }
    }

    @Test func emptyAgentTurnStaysSilent() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        transport.push(responseDone("agent"))   // no tools, no text → stay silent
        await eventually { log.states.contains(.listening) }
        #expect(!transport.sent.contains { $0.contains("\"role\":\"direct\"") })   // no response created
    }

    @Test func textInputTriggersAgentTurnManually() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools(), modality: RealtimeModality(input: .text, output: .text))
        log.start(session)
        try await session.start()

        await session.sendText("odds for PSG?")
        await eventually { log.states.contains(.thinking) }

        let types = sentTypes(transport)
        #expect(types.contains("conversation.item.create"))   // the user message
        #expect(types.contains("response.create"))            // manual agent trigger
    }

    @Test func bargeInCancelsAndDropsStaleAudio() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        // Drive a real grounded turn so a presenter response is in flight (presenterActive == true)…
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))
        transport.push(responseDone("r2"))
        await eventually { transport.sent.contains { $0.contains("\"role\":\"presenter\"") } }
        transport.push(responseCreated("p1", role: "presenter"))   // …and we're hearing it
        transport.push(audioDelta("p1", "a"))
        await eventually { log.audioCount == 1 }

        // …user barges in.
        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)
        await eventually { log.audioDones.contains(true) }
        #expect(sentTypes(transport).contains("response.cancel"))

        // Stale audio for the cancelled response is now dropped.
        transport.push(audioDelta("p1", "b"))
        try? await Task.sleep(for: .milliseconds(20))
        #expect(log.audioCount == 1)   // not 2 — the post-barge-in frame was dropped
    }

    @Test func bargeInMidGatherDropsTheAbandonedRound() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        // Tools are in flight (the gatherer round hasn't settled, so no presenter is active)…
        transport.push(funcArgs("r1", "c1", "odds"))
        await eventually { sentTypes(transport).contains("conversation.item.create") }   // output fed back

        // …user barges in before the round's `response.done`. No presenter is playing, so nothing is
        // cancelled, but the gathered state is wiped.
        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)
        await eventually { log.audioDones.contains(true) }
        #expect(!sentTypes(transport).contains("response.cancel"))   // nothing was playing to cancel

        // The abandoned round's `done` now arrives. With state wiped it must NOT present or speak.
        transport.push(responseDone("r1"))
        try? await Task.sleep(for: .milliseconds(20))
        #expect(!transport.sent.contains { $0.contains("\"role\":\"presenter\"") })
        #expect(!transport.sent.contains { $0.contains("\"role\":\"direct\"") })
        #expect(log.states.last == .listening)
    }

    @Test func continueAgentLoopAcrossMultipleToolRounds() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        transport.push(funcArgs("r1", "c1", "odds"))   // round 1
        transport.push(responseDone("r1"))             // tools → continue
        transport.push(funcArgs("r2", "c2", "odds"))   // round 2
        transport.push(responseDone("r2"))             // tools → continue
        transport.push(responseDone("r3"))             // settled → present

        await eventually { sentTypes(transport).filter { $0 == "response.create" }.count >= 3 }
        let types = sentTypes(transport)
        #expect(types.filter { $0 == "response.create" }.count == 3)            // 2 continues + presenter
        #expect(types.filter { $0 == "conversation.item.create" }.count == 2)   // 2 function_call_outputs
        #expect(transport.sent.contains { $0.contains("\"role\":\"presenter\"") })
    }

    @Test func presenterTextRelayedInTextOutputMode() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools(), modality: RealtimeModality(input: .text, output: .text))
        log.start(session)
        try await session.start()

        transport.push(responseCreated("p1", role: "presenter"))
        transport.push(#"{"type":"response.output_text.delta","response_id":"p1","delta":"answer"}"#)
        await eventually { log.presenterTexts.contains("answer") }
    }

    @Test func directResponseFinishesCleanly() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        transport.push(#"{"type":"response.output_text.delta","response_id":"agent","delta":"hi"}"#)
        transport.push(responseDone("agent"))   // → speak directly
        await eventually { sentTypes(transport).contains("response.create") }
        transport.push(responseCreated("d1", role: "direct"))
        transport.push(responseDone("d1"))       // direct finished

        await eventually { log.audioDones.contains(false) }
        #expect(log.states.last == .listening)
    }

    @Test func agentAudioIsNeverRelayed() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        // No presenter id set → an audio delta on a non-presenter (agent) response must be dropped.
        transport.push(audioDelta("agent", "x"))
        try? await Task.sleep(for: .milliseconds(20))
        #expect(log.audioCount == 0)
    }

    @Test func stopClosesTransport() async throws {
        let transport = MockRealtimeTransport()
        let session = session(transport, tools: oddsTools())
        try await session.start()
        await session.stop()
        #expect(transport.closed)
    }

    @Test func swallowsCancelNotActiveButSurfacesOtherErrors() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        transport.push(#"{"type":"error","error":{"code":"response_cancel_not_active"}}"#)
        transport.push(#"{"type":"error","error":{"code":"server_error"}}"#)
        await eventually { !log.errors.isEmpty }
        #expect(log.errors == ["server_error"])   // the benign cancel race was swallowed
    }

    // MARK: - Tracing

    @Test func opensTurnOnGatherAndClosesItWhenThePresenterFinishes() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = session(transport, tools: oddsTools(), tracer: tracer)
        try await session.start()

        transport.push(responseCreated("r1"))            // gatherer turn begins → opens voice.turn
        transport.push(funcArgs("r1", "c1", "odds"))     // tool call → tool.odds child span
        transport.push(responseDone("r1"))               // tools → continue
        transport.push(responseDone("r2"))               // settled with results → present()
        transport.push(responseCreated("p1", role: "presenter"))   // the presenter response
        transport.push(responseDone("p1"))               // presenter finished → end the turn

        await eventually { tracer.recorder.ended.contains("voice.turn") }
        #expect(tracer.recorder.opened.filter { $0 == "voice.turn" }.count == 1)   // opened once (not by the presenter)
        #expect(tracer.recorder.opened.contains("tool.odds"))                      // tool span is a child of the turn
    }

    @Test func recordsThePresenterExchangeAndUsageOnTheTurnTrace() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = session(transport, tools: oddsTools(), tracer: tracer)
        try await session.start()

        transport.push(userSaid("what are tonight's odds?"))
        transport.push(responseCreated("r1"))                       // gatherer turn
        transport.push(funcArgs("r1", "c1", "odds"))                // tool call
        transport.push(responseDone("r1", inputTokens: 80, outputTokens: 20))   // gatherer usage — dropped this pass
        transport.push(responseDone("r2"))                          // settled with results → present()
        transport.push(responseCreated("p1", role: "presenter"))    // the presenter response
        transport.push(audioTranscriptDone("p1", "PSG is favourite at 2.5"))
        transport.push(responseDone("p1", inputTokens: 12, outputTokens: 34))   // presenter finished

        await eventually { tracer.recorder.ended.contains("voice.turn") }
        // The spoken answer is captured as a `presenter` generation: question in, reply out, presenter
        // token usage recorded. (The gatherer's tools are the turn's `tool.*` children; its tokens are
        // intentionally not recorded this pass — so usage here is the presenter's, not the turn total.)
        #expect(tracer.recorder.opened.contains("presenter"))
        #expect(tracer.recorder.input("presenter") == .string("what are tonight's odds?"))
        #expect(tracer.recorder.output("presenter") == .string("PSG is favourite at 2.5"))
        #expect(tracer.recorder.usage("presenter")?.0 == 12)
        #expect(tracer.recorder.usage("presenter")?.1 == 34)
        // The root turn shows the conversation at the top level: question in, answer out.
        #expect(tracer.recorder.input("voice.turn") == .string("what are tonight's odds?"))
        #expect(tracer.recorder.output("voice.turn") == .string("PSG is favourite at 2.5"))
    }

    @Test func presenterGenerationSpanMeasuresRealLatencyFromResponseCreated() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let log = EventLog()
        let session = session(transport, tools: oddsTools(), tracer: tracer)
        log.start(session)
        try await session.start()

        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))                          // tools → continue
        transport.push(responseDone("r2"))                          // settled → present()
        await eventually { log.states.contains(.presenting) }       // gatherer drained, pump idle
        transport.push(responseCreated("p1", role: "presenter"))    // presenter starts (span start anchor)
        try await Task.sleep(for: .milliseconds(50))                // …presenter "generates"…
        transport.push(responseDone("p1"))                          // presenter finishes (span end anchor)

        await eventually { tracer.recorder.ended.contains("presenter") }
        // Backdated to the presenter response.created, so the generation carries the real latency
        // instead of the ~0s it showed when created-and-ended at response.done.
        let latency = try #require(tracer.recorder.duration("presenter"))
        #expect(latency >= 0.02)
    }

    @Test func directGenerationSpanMeasuresRealLatencyFromResponseCreated() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let log = EventLog()
        let session = session(transport, tools: oddsTools(), tracer: tracer)
        log.start(session)
        try await session.start()

        transport.push(#"{"type":"response.output_text.delta","response_id":"agent","delta":"hi"}"#)
        transport.push(responseDone("agent"))                       // no tools, has text → speakDirectly()
        await eventually { log.states.contains(.speaking) }         // speakDirectly ran, pump idle
        transport.push(responseCreated("d1", role: "direct"))       // direct reply starts (span start anchor)
        try await Task.sleep(for: .milliseconds(50))                // …direct reply "generates"…
        transport.push(responseDone("d1"))                          // direct reply finishes (span end anchor)

        await eventually { tracer.recorder.ended.contains("presenter") }   // the direct answer is the `presenter` generation
        let latency = try #require(tracer.recorder.duration("presenter"))
        #expect(latency >= 0.02)
    }

    @Test func attachesAudioTokenBreakdownToThePresenterGeneration() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = session(transport, tools: oddsTools(), tracer: tracer)
        try await session.start()

        transport.push(userSaid("what are tonight's odds?"))
        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))                          // gatherer settled
        transport.push(responseDone("r2"))                          // → present()
        transport.push(responseCreated("p1", role: "presenter"))
        transport.push(audioTranscriptDone("p1", "PSG is favourite at 2.5"))
        // The presenter's response.done carries the per-modality split.
        transport.push(#"{"type":"response.done","response":{"id":"p1","usage":{"input_tokens":1240,"output_tokens":820,"input_token_details":{"audio_tokens":1000,"cached_tokens":200},"output_token_details":{"audio_tokens":700}}}}"#)

        await eventually { tracer.recorder.ended.contains("voice.turn") }
        #expect(tracer.recorder.usage("presenter")?.0 == 1240)
        #expect(tracer.recorder.metadata("presenter") == .object([
            "gen_ai.usage.input_audio_tokens": .int(1000),
            "gen_ai.usage.output_audio_tokens": .int(700),
            "gen_ai.usage.cache_read_input_tokens": .int(200),
        ]))
    }

    @Test func traceTranscriptsOffOmitsThePresenterTranscriptButKeepsStructureAndUsage() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = OpenAIGroundedVoiceAssistant(name: "voice", transport: transport, tools: oddsTools(),
                                                   userId: "u1", sessionId: "s1", tracer: tracer, traceTranscripts: false)
        try await session.start()

        transport.push(userSaid("what are tonight's odds?"))
        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))
        transport.push(responseDone("r2"))                          // → present()
        transport.push(responseCreated("p1", role: "presenter"))
        transport.push(audioTranscriptDone("p1", "PSG is favourite at 2.5"))
        transport.push(responseDone("p1", inputTokens: 12, outputTokens: 34))

        await eventually { tracer.recorder.ended.contains("voice.turn") }
        #expect(tracer.recorder.opened.contains("presenter"))       // span structure kept…
        #expect(tracer.recorder.usage("presenter")?.0 == 12)        // …and usage…
        #expect(tracer.recorder.input("presenter") == nil)          // …but the spoken transcript is omitted
        #expect(tracer.recorder.output("presenter") == nil)
        #expect(tracer.recorder.input("voice.turn") == nil)
        #expect(tracer.recorder.opened.contains("tool.odds"))       // tool spans still flow (not a transcript)
    }

    @Test func gatherAndPresentNestUnderOneSessionRoot() async throws {
        let transport = MockRealtimeTransport()
        let tracer = StructureTracer()
        let session = session(transport, tools: oddsTools(), tracer: tracer)
        try await session.start()

        // One two-phase turn: gatherer calls a tool, then the presenter speaks.
        transport.push(userSaid("what are tonight's odds?"))
        transport.push(responseCreated("r1"))           // gatherer opens the turn
        transport.push(funcArgs("r1", "c1", "odds"))    // tool.odds child
        transport.push(responseDone("r1"))              // tools → continue
        transport.push(responseDone("r2"))              // settled → present()
        transport.push(responseCreated("p1", role: "presenter"))
        transport.push(audioTranscriptDone("p1", "PSG is favourite at 2.5"))
        transport.push(responseDone("p1"))
        await eventually { tracer.recorder.opened.contains { $0.name == "presenter" } }
        await session.stop()

        let nodes = tracer.recorder.opened
        let root = try #require(nodes.first { $0.name == "voice.session" })
        let turn = try #require(nodes.first { $0.name == "voice.turn" })
        #expect(nodes.filter { $0.name == "voice.session" }.count == 1)         // one session root across the two-phase turn
        #expect(root.metadata == .object(["modality": .string("audio")]))
        #expect(turn.parentId == root.id)                                       // turn under the session
        // Both the gatherer's tool span and the presenter generation hang off the one turn span.
        #expect(nodes.first { $0.name == "tool.odds" }?.parentId == turn.id)
        #expect(nodes.first { $0.name == "presenter" }?.parentId == turn.id)
        #expect(Set(nodes.map(\.traceId)).count == 1)                           // one trace id throughout
        #expect(tracer.recorder.ended.filter { $0 == root.id }.count == 1)      // session closed once on stop()
    }

    @Test func endsTheTurnTraceOnBargeIn() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = session(transport, tools: oddsTools(), tracer: tracer)
        try await session.start()

        transport.push(responseCreated("r1"))            // a turn is open
        await eventually { tracer.recorder.opened.contains("voice.turn") }
        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)   // barge-in
        await eventually { tracer.recorder.ended.contains("voice.turn") }
        #expect(tracer.recorder.ended.contains("voice.turn"))              // the common path must not leak the span
    }

    @Test func bargeInKeepsTheUserQueryOnTheInterruptedTurnTrace() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = session(transport, tools: oddsTools(), tracer: tracer)
        try await session.start()

        transport.push(userSaid("odds for PSG?"))
        transport.push(responseCreated("r1"))            // gatherer turn open, question captured in userText
        await eventually { tracer.recorder.opened.contains("voice.turn") }
        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)   // barge-in mid-gather
        await eventually { tracer.recorder.ended.contains("voice.turn") }
        #expect(tracer.recorder.input("voice.turn") == .string("odds for PSG?"))   // not an empty span
    }

    // MARK: - Persistence

    @Test func persistsAGroundedTurn() async throws {
        let transport = MockRealtimeTransport()
        let store = RecordingStore()
        let session = session(transport, tools: oddsTools(), store: store)
        try await session.start()

        transport.push(userSaid("odds for PSG?"))
        transport.push(responseCreated("r1"))                     // gatherer turn
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))                        // tools → continue
        transport.push(responseDone("r2"))                        // settled → present()
        transport.push(responseCreated("p1", role: "presenter"))
        transport.push(audioTranscriptDone("p1", "PSG are 2.5"))  // the presenter's spoken reply
        transport.push(responseDone("p1"))                        // presenter done → persist the pair

        await eventually { store.saved.count == 2 }
        #expect(store.saved.map(\.role) == [.user, .assistant])
        #expect(messageText(store.saved[0]) == "odds for PSG?")   // the user question, captured before resetTurn
        #expect(messageText(store.saved[1]) == "PSG are 2.5")     // the presenter's grounded reply
    }

    @Test func seedsPriorHistoryOnStart() async throws {
        let transport = MockRealtimeTransport()
        let store = RecordingStore(seed: [
            ConversationMessage(role: .user, text: "earlier question"),
            ConversationMessage(role: .assistant, text: "earlier answer"),
        ])
        let session = session(transport, tools: oddsTools(), store: store)
        try await session.start()

        await eventually { sentTypes(transport).filter { $0 == "conversation.item.create" }.count >= 2 }
        #expect(transport.sent.contains { $0.contains("earlier question") && $0.contains("input_text") })
        #expect(transport.sent.contains { $0.contains("earlier answer") && $0.contains("\"role\":\"assistant\"") })
    }

    @Test func doesNotPersistABargedInPresenter() async throws {
        let transport = MockRealtimeTransport()
        let store = RecordingStore()
        let session = session(transport, tools: oddsTools(), store: store)
        try await session.start()

        transport.push(userSaid("odds?"))
        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))
        transport.push(responseDone("r2"))                        // → present()
        transport.push(responseCreated("p1", role: "presenter"))
        transport.push(audioTranscriptDone("p1", "PSG are 2.5"))
        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)   // barge-in mid-presenter
        transport.push(responseDone("p1"))                        // late done for the cancelled presenter
        try? await Task.sleep(for: .milliseconds(20))
        #expect(store.saved.isEmpty)   // a barged-in presenter produces no persisted pair
    }

    // MARK: - Failed responses

    @Test func failedGathererResponseEmitsErrorInsteadOfPresenting() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        transport.push(userSaid("odds?"))
        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))       // tools → continue
        transport.push(responseFailed("r2"))     // the continue died server-side

        await eventually { !log.errors.isEmpty }
        #expect(log.errorCodes == ["response_failed"])
        #expect(log.errors == ["server_error: internal_error"])   // status_details.error, decoded
        #expect(log.states.last == .listening)
        // The turn never presented: the only response.create is the tool-continue one.
        #expect(sentTypes(transport).filter { $0 == "response.create" }.count == 1)
        #expect(!transport.sent.contains { $0.contains(#""role":"presenter""#) })
    }

    @Test func failedPresenterResponseSurfacesAndDoesNotPersist() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let store = RecordingStore()
        let session = session(transport, tools: oddsTools(), store: store)
        log.start(session)
        try await session.start()

        transport.push(userSaid("odds?"))
        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))
        transport.push(responseDone("r2"))                        // settled → present()
        transport.push(responseCreated("p1", role: "presenter"))
        transport.push(responseFailed("p1"))                      // the presenter died mid-reply

        await eventually { !log.errors.isEmpty }
        #expect(log.errorCodes == ["response_failed"])
        #expect(log.states.last == .listening)
        try? await Task.sleep(for: .milliseconds(20))
        #expect(store.saved.isEmpty)   // no reply was spoken — nothing to persist
    }

    @Test func lateFailedDoneForABargedInPresenterDoesNotFailTheNewTurn() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport, tools: oddsTools())
        log.start(session)
        try await session.start()

        transport.push(userSaid("odds?"))
        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))
        transport.push(responseDone("r2"))                        // → present()
        transport.push(responseCreated("p1", role: "presenter"))
        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)   // barge-in mid-presenter
        transport.push(userSaid("lineups?"))                      // a new turn is under way
        transport.push(responseCreated("r3"))                     // its gatherer
        transport.push(responseFailed("p1"))   // cancel raced a server-side failure — `done` lands failed
        transport.push(funcArgs("r3", "c2", "odds"))
        transport.push(responseDone("r3"))     // tools in → the new turn must still continue

        // The stale presenter's failure neither surfaced nor wiped the new turn's state: r3's
        // tool-continue response.create still goes out (present()'s was the first).
        await eventually { sentTypes(transport).filter { $0 == "response.create" }.count >= 2 }
        #expect(log.errors.isEmpty)
    }

    @Test func transportDeathEndsSpansWithTheErrorAndFinishesEvents() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let log = EventLog()
        let session = session(transport, tools: oddsTools(), tracer: tracer)
        log.start(session)
        try await session.start()

        transport.push(userSaid("odds?"))
        transport.push(responseCreated("r1"))   // gatherer turn (and the session root) open
        await eventually { tracer.recorder.opened.contains("voice.turn") }
        transport.die(with: URLError(.networkConnectionLost))   // socket drops mid-gather

        await eventually { log.errorCodes.contains("transport_closed") }
        #expect(tracer.recorder.ended.contains("voice.turn"))
        #expect(tracer.recorder.ended.contains("voice.session"))
        #expect(tracer.recorder.error("voice.turn")?.contains("transport closed") == true)
        #expect(tracer.recorder.input("voice.turn") == .string("odds?"))   // the question still rides the span
        await eventually { log.finished }
        #expect(log.finished)
    }
}
