import Foundation
import Testing

@testable import AgentSquad

@Suite struct OpenAIVoiceAssistantTests {
    private func session(_ transport: MockRealtimeTransport, tracer: any Tracer = OSLogTracer(), store: (any ChatStorage)? = nil) -> OpenAIVoiceAssistant {
        OpenAIVoiceAssistant(name: "voice", transport: transport, tools: oddsTools(),
                             userId: "u1", sessionId: "s1", store: store, tracer: tracer)
    }

    @Test func startSendsAudioAgentSessionUpdateWithTools() async throws {
        let transport = MockRealtimeTransport()
        let session = session(transport)
        try await session.start()

        await eventually { sentTypes(transport).contains("session.update") }
        let update = try #require(transport.sent.first { $0.contains("session.update") })
        let obj = try #require(try? JSONSerialization.jsonObject(with: Data(update.utf8)) as? [String: Any])
        let sess = try #require(obj["session"] as? [String: Any])
        #expect(sess["output_modalities"] as? [String] == ["audio"])   // the agent turn itself speaks
        #expect(sess["tools"] != nil)                                   // MCP/native tools advertised
    }

    @Test func typedTurnRequestsATextOnlyReply() async throws {
        let transport = MockRealtimeTransport()
        let session = session(transport)   // audio session by default
        try await session.start()

        await session.sendText("what are tonight's odds?")

        await eventually { sentTypes(transport).contains("response.create") }
        // Even in an audio session, a typed turn overrides its response to text-only.
        let frame = try #require(transport.sent.first { frameType(of: $0) == "response.create" })
        let obj = try #require(try? JSONSerialization.jsonObject(with: Data(frame.utf8)) as? [String: Any])
        let response = try #require(obj["response"] as? [String: Any])
        #expect(response["output_modalities"] as? [String] == ["text"])
    }

    @Test func typedTurnDeliversTextReplyWithoutSpeakingOrAudio() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport)   // audio session
        log.start(session)
        try await session.start()

        await session.sendText("what are tonight's odds?")
        transport.push(responseCreated("r1"))
        transport.push(#"{"type":"response.output_text.done","response_id":"r1","text":"PSG are favourites"}"#)
        transport.push(responseDone("r1"))

        await eventually { log.presenterTexts.contains("PSG are favourites") }
        // The reply arrives as text, with no audio, and the assistant never enters `.speaking`.
        #expect(log.presenterTexts.contains("PSG are favourites"))
        #expect(log.audioCount == 0)
        #expect(!log.states.contains(.speaking))
    }

    @Test func toolUsingTypedTurnKeepsTheContinueResponseTextOnly() async throws {
        let transport = MockRealtimeTransport()
        let session = session(transport)   // audio session, with the odds tool
        try await session.start()

        await session.sendText("what are tonight's odds?")
        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))   // tool call → continue
        transport.push(responseDone("r1"))             // tools were called → continue response

        await eventually { sentTypes(transport).filter { $0 == "response.create" }.count >= 2 }
        // Both the initial typed response AND the post-tool continue are text-only — a typed turn
        // never reverts to the session's audio default partway through.
        let creates = transport.sent.filter { frameType(of: $0) == "response.create" }
        for frame in creates {
            let obj = try #require(try? JSONSerialization.jsonObject(with: Data(frame.utf8)) as? [String: Any])
            let response = try #require(obj["response"] as? [String: Any])
            #expect(response["output_modalities"] as? [String] == ["text"])
        }
    }

    @Test func runsToolThenContinues() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport)
        log.start(session)
        try await session.start()

        transport.push(responseCreated("r1"))           // agent turn begins (speaks audio)
        transport.push(funcArgs("r1", "c1", "odds"))     // it calls a tool
        transport.push(responseDone("r1"))               // tool was called → continue speaking

        await eventually { sentTypes(transport).contains("response.create") }
        let types = sentTypes(transport)
        #expect(types.contains("conversation.item.create"))             // function_call_output fed back
        #expect(types.contains("response.create"))                      // continued the turn
        await eventually { !log.widgets.isEmpty }
        #expect(log.widgets.first?.resourceURI == "ui://odds")          // tool UI surfaced as a widget
    }

    @Test func relaysOnlyTheCurrentResponseAudio() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport)
        log.start(session)
        try await session.start()

        transport.push(responseCreated("r1"))
        transport.push(audioDelta("r1", "a"))
        await eventually { log.audioCount == 1 }
        transport.push(audioDelta("other", "b"))         // not the current response → dropped
        try? await Task.sleep(for: .milliseconds(20))
        #expect(log.audioCount == 1)
    }

    @Test func bargeInCancelsAndDropsStaleAudio() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport)
        log.start(session)
        try await session.start()

        transport.push(responseCreated("r1"))
        transport.push(audioDelta("r1", "a"))
        await eventually { log.audioCount == 1 }

        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)   // user barges in
        await eventually { log.audioDones.contains(true) }
        #expect(sentTypes(transport).contains("response.cancel"))

        transport.push(audioDelta("r1", "b"))            // stale audio for the cancelled response → dropped
        try? await Task.sleep(for: .milliseconds(20))
        #expect(log.audioCount == 1)
    }

    @Test func bargeInTruncatesTheHeardPortion() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport)
        await session.setPlaybackClock { 40 }   // 40 ms actually played
        log.start(session)
        try await session.start()

        transport.push(responseCreated("r1"))
        // 4 800 PCM16 bytes = 100 ms @ 24 kHz — more than the 40 ms played, so truncation applies.
        transport.push(audioDelta("r1", String(repeating: "x", count: 4_800), item: "item_7"))
        await eventually { log.audioCount == 1 }   // delta processed → truncation state recorded

        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)
        await eventually { sentTypes(transport).contains("response.cancel") }

        let truncate = transport.sent.first { $0.contains("conversation.item.truncate") }
        let frame = try #require(truncate)
        #expect(frame.contains(#""item_id":"item_7""#))
        #expect(frame.contains(#""audio_end_ms":40"#))
        #expect(frame.contains(#""content_index":0"#))
        // Ordering: truncate goes out before the cancel.
        let types = sentTypes(transport)
        let truncateIdx = try #require(types.firstIndex(of: "conversation.item.truncate"))
        let cancelIdx = try #require(types.firstIndex(of: "response.cancel"))
        #expect(truncateIdx < cancelIdx)
    }

    @Test func bargeInSkipsTruncateWhenEverythingWasPlayed() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport)
        await session.setPlaybackClock { 500 }   // clock ≥ received (100 ms) — stale or fully played
        log.start(session)
        try await session.start()

        transport.push(responseCreated("r1"))
        transport.push(audioDelta("r1", String(repeating: "x", count: 4_800)))
        await eventually { log.audioCount == 1 }

        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)
        await eventually { sentTypes(transport).contains("response.cancel") }
        // audio_end_ms greater than the actual duration is a server error (client-events spec) — never sent.
        #expect(!sentTypes(transport).contains("conversation.item.truncate"))
    }

    @Test func bargeInWithoutAClockSendsNoTruncate() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport)   // no setPlaybackClock
        log.start(session)
        try await session.start()

        transport.push(responseCreated("r1"))
        transport.push(audioDelta("r1", String(repeating: "x", count: 4_800)))
        await eventually { log.audioCount == 1 }

        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)
        await eventually { sentTypes(transport).contains("response.cancel") }
        #expect(!sentTypes(transport).contains("conversation.item.truncate"))
    }

    @Test func noToolTurnSpeaksAndFinishes() async throws {
        let transport = MockRealtimeTransport()
        let log = EventLog()
        let session = session(transport)
        log.start(session)
        try await session.start()

        transport.push(responseCreated("r1"))
        transport.push(audioDelta("r1", "hi"))
        transport.push(responseDone("r1"))               // no tools → the spoken answer; turn done
        await eventually { log.audioDones.contains(false) }
        #expect(log.states.last == .listening)
    }

    @Test func opensAndClosesAPerTurnTraceWithToolSpan() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = session(transport, tracer: tracer)
        try await session.start()

        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))               // tools → continue
        transport.push(responseCreated("r2"))
        transport.push(responseDone("r2"))               // no tools → turn complete

        await eventually { tracer.recorder.ended.contains("voice.turn") }
        #expect(tracer.recorder.opened.contains("voice.turn"))   // one turn trace
        #expect(tracer.recorder.opened.contains("tool.odds"))    // a child span per tool call
        #expect(tracer.recorder.opened.filter { $0 == "voice.turn" }.count == 1)   // not reopened on continue
    }

    @Test func recordsTheSpokenExchangeAndUsageOnTheTurnTrace() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = session(transport, tracer: tracer)
        try await session.start()

        transport.push(userSaid("what are tonight's odds?"))
        transport.push(responseCreated("r1"))
        transport.push(audioTranscriptDone("r1", "PSG is favourite at 2.5"))
        transport.push(responseDone("r1", inputTokens: 12, outputTokens: 34))   // no tools → spoken answer

        await eventually { tracer.recorder.ended.contains("voice.turn") }
        // The spoken exchange is captured as a `response` generation under the turn: question in, reply
        // out, token usage recorded — so the trace shows what was said instead of an empty span.
        #expect(tracer.recorder.opened.contains("response"))
        #expect(tracer.recorder.input("response") == .string("what are tonight's odds?"))
        #expect(tracer.recorder.output("response") == .string("PSG is favourite at 2.5"))
        #expect(tracer.recorder.usage("response")?.0 == 12)
        #expect(tracer.recorder.usage("response")?.1 == 34)
        // The root turn shows the conversation at the top level: question in (set late, after the
        // transcript arrives), answer out.
        #expect(tracer.recorder.input("voice.turn") == .string("what are tonight's odds?"))
        #expect(tracer.recorder.output("voice.turn") == .string("PSG is favourite at 2.5"))
    }

    @Test func traceTranscriptsOffOmitsSpokenTextButKeepsStructureAndUsage() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = OpenAIVoiceAssistant(name: "voice", transport: transport, tools: oddsTools(),
                                           userId: "u1", sessionId: "s1", tracer: tracer, traceTranscripts: false)
        try await session.start()

        transport.push(userSaid("what are tonight's odds?"))
        transport.push(responseCreated("r1"))
        transport.push(audioTranscriptDone("r1", "PSG is favourite at 2.5"))
        transport.push(responseDone("r1", inputTokens: 12, outputTokens: 34))

        await eventually { tracer.recorder.ended.contains("voice.turn") }
        #expect(tracer.recorder.opened.contains("response"))          // the span structure is still there…
        #expect(tracer.recorder.usage("response")?.0 == 12)           // …and token usage…
        #expect(tracer.recorder.input("response") == nil)             // …but the spoken transcript is omitted
        #expect(tracer.recorder.output("response") == nil)
        #expect(tracer.recorder.input("voice.turn") == nil)
        #expect(tracer.recorder.output("voice.turn") == nil)
    }

    @Test func attachesAudioTokenBreakdownAsGenerationMetadata() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = session(transport, tracer: tracer)
        try await session.start()

        transport.push(userSaid("what are tonight's odds?"))
        transport.push(responseCreated("r1"))
        transport.push(audioTranscriptDone("r1", "PSG is favourite at 2.5"))
        // response.done carrying the per-modality split the Realtime API reports.
        transport.push(#"{"type":"response.done","response":{"id":"r1","usage":{"input_tokens":1240,"output_tokens":820,"input_token_details":{"audio_tokens":1000,"cached_tokens":200},"output_token_details":{"audio_tokens":700}}}}"#)

        await eventually { tracer.recorder.ended.contains("voice.turn") }
        // Totals ride the usage seam; the audio/cached breakdown rides metadata (GenAI keys) so the
        // backend can cost the turn correctly — audio tokens price differently from text.
        #expect(tracer.recorder.usage("response")?.0 == 1240)
        #expect(tracer.recorder.usage("response")?.1 == 820)
        #expect(tracer.recorder.metadata("response") == .object([
            "gen_ai.usage.input_audio_tokens": .int(1000),
            "gen_ai.usage.output_audio_tokens": .int(700),
            "gen_ai.usage.cache_read_input_tokens": .int(200),
        ]))
    }

    @Test func turnsNestUnderOneSessionRootSharingItsTraceId() async throws {
        let transport = MockRealtimeTransport()
        let tracer = StructureTracer()
        let session = session(transport, tracer: tracer)
        try await session.start()

        // Two complete spoken turns over one connection.
        for (rid, q, a) in [("r1", "first?", "one"), ("r2", "second?", "two")] {
            transport.push(userSaid(q))
            transport.push(responseCreated(rid))
            transport.push(audioTranscriptDone(rid, a))
            transport.push(responseDone(rid))
            await eventually { tracer.recorder.opened.filter { $0.name == "voice.turn" }.count
                == (rid == "r1" ? 1 : 2) }
        }
        await session.stop()

        let nodes = tracer.recorder.opened
        let sessions = nodes.filter { $0.name == "voice.session" }
        let turns = nodes.filter { $0.name == "voice.turn" }
        let root = try #require(sessions.first)
        #expect(sessions.count == 1)                                   // one session root, opened once across both turns
        #expect(root.parentId == nil)                                  // it's the trace root
        #expect(root.metadata == .object(["modality": .string("audio")]))   // tagged as a voice conversation
        #expect(turns.count == 2)
        #expect(turns.allSatisfy { $0.parentId == root.id })           // each turn nests under the session
        #expect(Set(nodes.map(\.traceId)).count == 1)                  // everything shares one trace id
        #expect(tracer.recorder.ended.contains(root.id))               // the session is closed on stop()
        #expect(tracer.recorder.ended.filter { $0 == root.id }.count == 1)
    }

    @Test func framesAfterStopDoNotReopenTheSessionTrace() async throws {
        let transport = MockRealtimeTransport()
        let tracer = StructureTracer()
        let session = session(transport, tracer: tracer)
        try await session.start()

        transport.push(userSaid("q"))
        transport.push(responseCreated("r1"))
        transport.push(audioTranscriptDone("r1", "a"))
        transport.push(responseDone("r1"))
        await eventually { tracer.recorder.opened.contains { $0.name == "voice.session" } }
        await session.stop()

        // A late frame (e.g. one buffered in the pump at teardown) must not reopen a session root that
        // nothing would ever close — `stop()` is final.
        transport.push(responseCreated("r2"))
        try? await Task.sleep(for: .milliseconds(20))
        #expect(tracer.recorder.opened.filter { $0.name == "voice.session" }.count == 1)
    }

    @Test func endsTheTurnTraceOnBargeIn() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = session(transport, tracer: tracer)
        try await session.start()

        transport.push(responseCreated("r1"))
        await eventually { tracer.recorder.opened.contains("voice.turn") }
        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)   // barge-in
        await eventually { tracer.recorder.ended.contains("voice.turn") }
        #expect(tracer.recorder.ended.contains("voice.turn"))              // no span leak on the common path
    }

    @Test func bargeInKeepsTheUserQueryOnTheInterruptedTurnTrace() async throws {
        let transport = MockRealtimeTransport()
        let tracer = RecordingTracer()
        let session = session(transport, tracer: tracer)
        try await session.start()

        transport.push(userSaid("what are tonight's odds?"))
        transport.push(responseCreated("r1"))
        transport.push(audioTranscriptDone("r1", "PSG is favo"))            // partial spoken reply
        await eventually { tracer.recorder.opened.contains("voice.turn") }
        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#)    // user barges in
        await eventually { tracer.recorder.ended.contains("voice.turn") }
        // Even interrupted, the turn records what was asked (and the partial reply) — not an empty span.
        #expect(tracer.recorder.input("voice.turn") == .string("what are tonight's odds?"))
        #expect(tracer.recorder.output("voice.turn") == .string("PSG is favo"))
    }

    @Test func persistsACompletedTurn() async throws {
        let transport = MockRealtimeTransport()
        let store = RecordingStore()
        let session = session(transport, store: store)
        try await session.start()

        transport.push(userSaid("what are the odds?"))
        transport.push(responseCreated("r1"))
        transport.push(audioTranscriptDone("r1", "PSG are 2.5"))   // captured for persistence even in audio mode
        transport.push(responseDone("r1"))                          // no tools → spoken answer; turn complete

        await eventually { store.saved.count == 2 }
        #expect(store.saved.map(\.role) == [.user, .assistant])
        #expect(messageText(store.saved[0]) == "what are the odds?")
        #expect(messageText(store.saved[1]) == "PSG are 2.5")
    }

    @Test func seedsPriorHistoryOnStart() async throws {
        let transport = MockRealtimeTransport()
        let store = RecordingStore(seed: [
            ConversationMessage(role: .user, text: "earlier question"),
            ConversationMessage(role: .assistant, text: "earlier answer"),
        ])
        let session = session(transport, store: store)
        try await session.start()

        await eventually { sentTypes(transport).filter { $0 == "conversation.item.create" }.count >= 2 }
        #expect(transport.sent.contains { $0.contains("earlier question") && $0.contains("input_text") })
        #expect(transport.sent.contains { $0.contains("earlier answer") && $0.contains("\"role\":\"assistant\"") })
    }

    @Test func doesNotPersistABargedInTurn() async throws {
        let transport = MockRealtimeTransport()
        let store = RecordingStore()
        let session = session(transport, store: store)
        try await session.start()

        transport.push(userSaid("hi"))
        transport.push(responseCreated("r1"))
        transport.push(audioTranscriptDone("r1", "partial…"))            // started speaking
        transport.push(#"{"type":"input_audio_buffer.speech_started"}"#) // barge-in before responseDone
        transport.push(responseDone("r1"))                               // late done for the cancelled response
        try? await Task.sleep(for: .milliseconds(20))
        #expect(store.saved.isEmpty)   // an interrupted turn produces no complete pair
    }

    @Test func stopClosesTransport() async throws {
        let transport = MockRealtimeTransport()
        let session = session(transport)
        try await session.start()
        await session.stop()
        #expect(transport.closed)
    }

    // MARK: - Reasoning effort

    private func reasoningEffort(of frame: String) -> String? {
        guard let obj = try? JSONSerialization.jsonObject(with: Data(frame.utf8)) as? [String: Any],
              let response = obj["response"] as? [String: Any],
              let reasoning = response["reasoning"] as? [String: Any] else { return nil }
        return reasoning["effort"] as? String
    }

    @Test func configuredToolEscalatesTheContinuationReasoning() async throws {
        let transport = MockRealtimeTransport()
        let session = OpenAIVoiceAssistant(name: "voice", transport: transport, tools: oddsTools(),
                                           userId: "u1", sessionId: "s1",
                                           toolReasoningEffort: ["odds": .medium])
        try await session.start()

        await session.sendText("should I bet on PSG?")
        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))     // the configured tool runs
        transport.push(responseDone("r1"))               // → continuation

        await eventually { sentTypes(transport).filter { $0 == "response.create" }.count >= 2 }
        let creates = transport.sent.filter { frameType(of: $0) == "response.create" }
        // The initial response runs at the session default; the response that synthesizes the
        // tool's payload thinks at the configured effort.
        #expect(reasoningEffort(of: creates[0]) == nil)
        #expect(reasoningEffort(of: try #require(creates.last)) == "medium")
    }

    @Test func reasoningEscalationIsTurnStickyAndResetsOnTheNextTurn() async throws {
        let transport = MockRealtimeTransport()
        let session = OpenAIVoiceAssistant(name: "voice", transport: transport, tools: oddsTools(),
                                           userId: "u1", sessionId: "s1",
                                           toolReasoningEffort: ["odds": .medium])
        try await session.start()

        await session.sendText("should I bet on PSG?")
        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))     // escalates the turn
        transport.push(responseDone("r1"))               // continuation 1
        transport.push(responseCreated("r2"))
        transport.push(funcArgs("r2", "c2", "odds"))     // a second tool round, same turn
        transport.push(responseDone("r2"))               // continuation 2
        transport.push(responseCreated("r3"))
        transport.push(responseDone("r3"))               // no tools → turn complete, escalation resets

        await eventually { sentTypes(transport).filter { $0 == "response.create" }.count >= 3 }
        await session.sendText("and the score?")         // a fresh turn
        await eventually { sentTypes(transport).filter { $0 == "response.create" }.count >= 4 }

        let creates = transport.sent.filter { frameType(of: $0) == "response.create" }
        #expect(reasoningEffort(of: creates[1]) == "medium")   // sticky across the turn's rounds…
        #expect(reasoningEffort(of: creates[2]) == "medium")
        #expect(reasoningEffort(of: creates[3]) == nil)        // …but never leaks into the next turn
    }

    @Test func unconfiguredToolLeavesTheContinuationAtTheSessionDefault() async throws {
        let transport = MockRealtimeTransport()
        let session = session(transport)   // no toolReasoningEffort
        try await session.start()

        transport.push(responseCreated("r1"))
        transport.push(funcArgs("r1", "c1", "odds"))
        transport.push(responseDone("r1"))

        await eventually { sentTypes(transport).contains("response.create") }
        let creates = transport.sent.filter { frameType(of: $0) == "response.create" }
        #expect(creates.allSatisfy { reasoningEffort(of: $0) == nil })   // wire unchanged for existing users
    }

    @Test func sessionLevelReasoningRidesTheSessionUpdate() async throws {
        let transport = MockRealtimeTransport()
        let session = OpenAIVoiceAssistant(name: "voice", transport: transport, tools: oddsTools(),
                                           userId: "u1", sessionId: "s1", reasoning: .high)
        try await session.start()

        await eventually { sentTypes(transport).contains("session.update") }
        let update = try #require(transport.sent.first { $0.contains("session.update") })
        let obj = try #require(try? JSONSerialization.jsonObject(with: Data(update.utf8)) as? [String: Any])
        let sess = try #require(obj["session"] as? [String: Any])
        let reasoning = try #require(sess["reasoning"] as? [String: Any])
        #expect(reasoning["effort"] as? String == "high")
    }
}
