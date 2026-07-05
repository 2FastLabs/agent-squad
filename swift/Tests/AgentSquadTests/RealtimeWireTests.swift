import Foundation
import Testing

@testable import AgentSquad

@Suite struct RealtimeWireTests {
    private func object(_ json: String) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
    }

    // MARK: - Outbound

    @Test func sessionUpdateConfig() throws {
        let json = RealtimeWire.sessionUpdate(
            model: "gpt-realtime", instructions: "gather", voice: "marin",
            language: "fr", tools: [AgentTool(name: "odds", description: "match odds")], sampleRate: 24_000
        )
        let root = try object(json)
        #expect(root["type"] as? String == "session.update")
        let session = try #require(root["session"] as? [String: Any])
        #expect(session["output_modalities"] as? [String] == ["text"])   // agent turn is text-only
        let audio = try #require(session["audio"] as? [String: Any])
        let input = try #require(audio["input"] as? [String: Any])
        #expect(((input["turn_detection"] as? [String: Any])?["type"]) as? String == "semantic_vad")
        #expect(((input["transcription"] as? [String: Any])?["language"]) as? String == "fr")
        let tools = try #require(session["tools"] as? [[String: Any]])
        #expect(tools.first?["name"] as? String == "odds")            // flat function shape
        #expect(session["tool_choice"] as? String == "auto")
        #expect(((input["transcription"] as? [String: Any])?["model"]) as? String == "gpt-4o-mini-transcribe")
        #expect(((input["turn_detection"] as? [String: Any])?["eagerness"]) == nil)   // server default when unset
    }

    private func inputSection(_ json: String) throws -> [String: Any] {
        let session = try #require(try object(json)["session"] as? [String: Any])
        let audio = try #require(session["audio"] as? [String: Any])
        return try #require(audio["input"] as? [String: Any])
    }

    @Test func sessionUpdateSemanticVADEagerness() throws {
        let json = RealtimeWire.sessionUpdate(
            model: "m", instructions: "i", voice: "v", language: nil, tools: [], sampleRate: 24_000,
            turnDetection: .semanticVAD(eagerness: .low)
        )
        let vad = try #require(try inputSection(json)["turn_detection"] as? [String: Any])
        #expect(vad["type"] as? String == "semantic_vad")
        #expect(vad["eagerness"] as? String == "low")
        #expect(vad["create_response"] as? Bool == true)
    }

    @Test func sessionUpdateServerVAD() throws {
        let json = RealtimeWire.sessionUpdate(
            model: "m", instructions: "i", voice: "v", language: nil, tools: [], sampleRate: 24_000,
            turnDetection: .serverVAD(threshold: 0.6, prefixPaddingMs: 300, silenceDurationMs: 500)
        )
        let vad = try #require(try inputSection(json)["turn_detection"] as? [String: Any])
        #expect(vad["type"] as? String == "server_vad")
        #expect(vad["threshold"] as? Double == 0.6)
        #expect(vad["prefix_padding_ms"] as? Int == 300)
        #expect(vad["silence_duration_ms"] as? Int == 500)
    }

    @Test func sessionUpdateDisabledTurnDetectionIsJSONNull() throws {
        let json = RealtimeWire.sessionUpdate(
            model: "m", instructions: "i", voice: "v", language: nil, tools: [], sampleRate: 24_000,
            turnDetection: .disabled
        )
        #expect(try inputSection(json)["turn_detection"] is NSNull)
    }

    @Test func sessionUpdateCustomTranscriptionModel() throws {
        let json = RealtimeWire.sessionUpdate(
            model: "m", instructions: "i", voice: "v", language: nil, tools: [], sampleRate: 24_000,
            transcriptionModel: "gpt-4o-transcribe"
        )
        #expect(((try inputSection(json)["transcription"] as? [String: Any])?["model"]) as? String == "gpt-4o-transcribe")
    }

    @Test func sessionUpdateOverridesDeepMergeIntoSession() throws {
        // Overrides add keys the signature doesn't model and win over generated ones — without
        // clobbering the siblings of the objects they touch.
        let json = RealtimeWire.sessionUpdate(
            model: "m", instructions: "i", voice: "v", language: "fr", tools: [], sampleRate: 24_000,
            overrides: [
                "audio": .object(["input": .object([
                    "noise_reduction": .object(["type": .string("near_field")]),
                    "transcription": .object(["model": .string("whisper-1")]),
                ])]),
                "max_output_tokens": .int(512),
            ]
        )
        let session = try #require(try object(json)["session"] as? [String: Any])
        #expect(session["max_output_tokens"] as? Int == 512)                       // added at top level
        let input = try inputSection(json)
        #expect((input["noise_reduction"] as? [String: Any])?["type"] as? String == "near_field")   // added nested
        let transcription = try #require(input["transcription"] as? [String: Any])
        #expect(transcription["model"] as? String == "whisper-1")                  // override wins
        #expect(transcription["language"] as? String == "fr")                      // sibling preserved
        #expect(input["format"] != nil)                                            // untouched siblings survive
        #expect((session["audio"] as? [String: Any])?["output"] != nil)
    }

    @Test func presenterResponseIsOutOfBandWithCuratedInput() throws {
        let json = RealtimeWire.presenterResponse(instructions: "present", feed: "FEED", output: .audio, voice: "marin")
        let response = try #require(try object(json)["response"] as? [String: Any])
        #expect(response["conversation"] as? String == "none")
        #expect(((response["metadata"] as? [String: Any])?["role"]) as? String == "presenter")
        #expect(response["output_modalities"] as? [String] == ["audio"])
        #expect(((response["audio"] as? [String: Any])?["output"] as? [String: Any])?["voice"] as? String == "marin")
        let input = try #require(response["input"] as? [[String: Any]])
        let content = try #require(input.first?["content"] as? [[String: Any]])
        #expect(content.first?["text"] as? String == "FEED")
    }

    @Test func directResponseIsInBandWithNoInput() throws {
        let json = RealtimeWire.directResponse(instructions: "chat", output: .text, voice: "marin")
        let response = try #require(try object(json)["response"] as? [String: Any])
        #expect(response["conversation"] == nil)                       // in-band — sees history
        #expect(((response["metadata"] as? [String: Any])?["role"]) as? String == "direct")
        #expect(response["input"] == nil)
        #expect(response["output_modalities"] as? [String] == ["text"])
    }

    @Test func createResponseBareUsesSessionConfigAndTextOverridesToTextOnly() throws {
        // Bare: no `response` object → the response runs under the session's modalities (the agent
        // turn / tool-continue path).
        let bare = try object(RealtimeWire.createResponse())
        #expect(bare["type"] as? String == "response.create")
        #expect(bare["response"] == nil)

        // `output: .text` overrides just this response to text-only — a typed turn in an audio session.
        let text = try object(RealtimeWire.createResponse(output: .text))
        let response = try #require(text["response"] as? [String: Any])
        #expect(response["output_modalities"] as? [String] == ["text"])
        #expect(response["audio"] == nil)   // text-only → no voice/audio block
    }

    @Test func appendAudioCarriesBase64Payload() throws {
        let base64 = Data("pcm16".utf8).base64EncodedString()
        let root = try object(RealtimeWire.appendAudio(base64))
        #expect(root["type"] as? String == "input_audio_buffer.append")
        #expect(root["audio"] as? String == base64)
    }

    @Test func functionOutputCarriesCallIdAndStringOutput() throws {
        let json = RealtimeWire.functionOutput(callId: "c1", output: "{\"x\":1}")
        let item = try #require(try object(json)["item"] as? [String: Any])
        #expect(item["type"] as? String == "function_call_output")
        #expect(item["call_id"] as? String == "c1")
        #expect(item["output"] as? String == "{\"x\":1}")
    }

    @Test func seededHistoryItemsUseDirectionalContentTypes() throws {
        // Realtime content parts are typed by direction. The GA API rejects the old beta `text` on an
        // assistant item with `invalid_value`, which broke reopening a chat with a stored reply.
        func contentType(_ json: String) throws -> String? {
            let item = try #require(try object(json)["item"] as? [String: Any])
            let content = try #require(item["content"] as? [[String: Any]])
            return content.first?["type"] as? String
        }
        #expect(try contentType(RealtimeWire.userMessage("hi")) == "input_text")        // user → input
        #expect(try contentType(RealtimeWire.assistantMessage("hello")) == "output_text")   // assistant → output
    }

    // MARK: - Inbound decode

    @Test func decodesGAAndLegacyAudioDeltaToTheSameEvent() {
        for type in ["response.output_audio.delta", "response.audio.delta"] {
            let event = RealtimeWire.decode(#"{"type":"\#(type)","response_id":"p1","item_id":"item_9","delta":"AAAA"}"#)
            #expect(event == .audioDelta(responseId: "p1", itemId: "item_9", base64: "AAAA"))
        }
    }

    @Test func truncationStateIsPerItemAcrossABurst() {
        // Tool→continue turns speak several items back-to-back without playback draining: the
        // burst clock covers all of them, so the frame must subtract the previous items' audio.
        var state = AudioTruncationState()
        state.record(itemId: "item_A", pcm16ByteCount: 24_000, sampleRate: 24_000)   // 500 ms
        state.record(itemId: "item_B", pcm16ByteCount: 144_000, sampleRate: 24_000)  // 3 000 ms
        // Burst played 800 ms total = all of A (500) + 300 of B → truncate B at 300, not 800.
        let frame = state.truncateFrame(playedMs: 800)
        #expect(frame?.contains(#""item_id":"item_B""#) == true)
        #expect(frame?.contains(#""audio_end_ms":300"#) == true)
        // Clock restarted (burst drained between items) → played < offset → skip, never negative.
        #expect(state.truncateFrame(playedMs: 200) == nil)
        // Stale clock beyond this item's received audio → skip (server errors past the duration).
        #expect(state.truncateFrame(playedMs: 4_000) == nil)
        #expect(state.truncateFrame(playedMs: nil) == nil)
    }

    @Test func truncateItemFrameMatchesTheSpec() throws {
        // reference/client-events.md: required fields type/item_id/content_index/audio_end_ms; content_index is 0.
        let json = try object(RealtimeWire.truncateItem(itemId: "item_002", audioEndMs: 1_500))
        #expect(json["type"] as? String == "conversation.item.truncate")
        #expect(json["item_id"] as? String == "item_002")
        #expect(json["content_index"] as? Int == 0)
        #expect(json["audio_end_ms"] as? Int == 1_500)
    }

    @Test func audioTranscriptDeltaIsNotMistakenForAudioDelta() {
        let event = RealtimeWire.decode(#"{"type":"response.output_audio_transcript.delta","response_id":"p1","delta":"hi"}"#)
        #expect(event == .audioTranscriptDelta(responseId: "p1", text: "hi"))
    }

    @Test func decodesTheHandledEventSet() {
        #expect(RealtimeWire.decode(#"{"type":"input_audio_buffer.speech_started"}"#) == .speechStarted)
        #expect(RealtimeWire.decode(#"{"type":"conversation.item.input_audio_transcription.completed","transcript":"hello"}"#) == .userTranscriptCompleted("hello"))
        #expect(RealtimeWire.decode(#"{"type":"response.function_call_arguments.done","response_id":"r1","call_id":"c1","name":"odds","arguments":"{}"}"#)
            == .functionCallArguments(responseId: "r1", callId: "c1", name: "odds", arguments: "{}"))
        #expect(RealtimeWire.decode(#"{"type":"response.output_item.added","item":{"type":"function_call","call_id":"c1","name":"odds"}}"#)
            == .functionCallNamed(callId: "c1", name: "odds"))
        #expect(RealtimeWire.decode(#"{"type":"response.output_text.delta","response_id":"r1","delta":"x"}"#) == .outputTextDelta(responseId: "r1", text: "x"))
        #expect(RealtimeWire.decode(#"{"type":"response.text.delta","response_id":"r1","delta":"x"}"#) == .outputTextDelta(responseId: "r1", text: "x"))
        #expect(RealtimeWire.decode(#"{"type":"response.created","response":{"id":"p1","metadata":{"role":"presenter"}}}"#) == .responseCreated(id: "p1", role: "presenter"))
        #expect(RealtimeWire.decode(#"{"type":"response.done","response":{"id":"r1"}}"#) == .responseDone(id: "r1", usage: nil))
        #expect(RealtimeWire.decode(#"{"type":"response.done","response":{"id":"r1","usage":{"input_tokens":12,"output_tokens":34}}}"#)
            == .responseDone(id: "r1", usage: RealtimeUsage(inputTokens: 12, outputTokens: 34, inputAudioTokens: nil, outputAudioTokens: nil, cachedInputTokens: nil)))
        // Per-field tolerance: a usage object missing one count decodes that field to nil, not a failure.
        #expect(RealtimeWire.decode(#"{"type":"response.done","response":{"id":"r1","usage":{"input_tokens":12}}}"#)
            == .responseDone(id: "r1", usage: RealtimeUsage(inputTokens: 12, outputTokens: nil, inputAudioTokens: nil, outputAudioTokens: nil, cachedInputTokens: nil)))
        // The per-modality breakdown (audio / cached) is decoded from the nested token-detail objects.
        #expect(RealtimeWire.decode(#"{"type":"response.done","response":{"id":"r1","usage":{"input_tokens":1240,"output_tokens":820,"input_token_details":{"audio_tokens":1000,"cached_tokens":200},"output_token_details":{"audio_tokens":700}}}}"#)
            == .responseDone(id: "r1", usage: RealtimeUsage(inputTokens: 1240, outputTokens: 820, inputAudioTokens: 1000, outputAudioTokens: 700, cachedInputTokens: 200)))
        #expect(RealtimeWire.decode(#"{"type":"error","error":{"code":"response_cancel_not_active"}}"#) == .error(code: "response_cancel_not_active"))
    }

    @Test func ignoresUnhandledEvents() {
        #expect(RealtimeWire.decode(#"{"type":"rate_limits.updated"}"#) == nil)
        #expect(RealtimeWire.decode("not json") == nil)
    }

    @Test func usageBreakdownProjectsOnlyPresentKeysAsMetadata() {
        // The breakdown maps to GenAI-convention metadata keys; absent fields are omitted.
        let full = RealtimeUsage(inputTokens: 1240, outputTokens: 820, inputAudioTokens: 1000, outputAudioTokens: 700, cachedInputTokens: 200)
        #expect(full.breakdownMetadata == .object([
            "gen_ai.usage.input_audio_tokens": .int(1000),
            "gen_ai.usage.output_audio_tokens": .int(700),
            "gen_ai.usage.cache_read_input_tokens": .int(200),
        ]))
        // Partial: only one field present → an object with exactly that key, the others omitted.
        #expect(RealtimeUsage(inputTokens: 12, outputTokens: 34, inputAudioTokens: nil, outputAudioTokens: nil, cachedInputTokens: 200).breakdownMetadata
            == .object(["gen_ai.usage.cache_read_input_tokens": .int(200)]))
        // Totals-only (no detail objects) → no breakdown metadata, so the caller skips setMetadata.
        #expect(RealtimeUsage(inputTokens: 12, outputTokens: 34, inputAudioTokens: nil, outputAudioTokens: nil, cachedInputTokens: nil).breakdownMetadata == nil)
    }
}
