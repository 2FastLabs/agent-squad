import Foundation

/// Wire translation for the OpenAI **Realtime API** (GA shape): builds the JSON frames the session
/// sends and decodes the ones it receives. Kept separate from the transport (raw frames) and the
/// session (control logic) so this pure mapping is unit-testable. Inbound event names are matched by
/// **suffix** so both GA (`response.output_audio.delta`) and legacy (`response.audio.delta`) names
/// resolve to the same `ServerEvent`.
enum RealtimeWire {
    static let audioMimeType = "audio/pcm"

    // MARK: - Outbound frames

    /// `session.update` — sent once at connect. `agentOutput` sets the agent turn's modality at the
    /// session level: `.text` (the grounded gatherer — never speaks; audio comes from a per-response
    /// presenter) or `.audio` (the primitive — the agent turn itself speaks). VAD auto-creates the
    /// agent response on speech end. `overrides` deep-merges into the generated `session` object
    /// last — the escape hatch for keys this signature doesn't model (noise reduction, idle timeout, …).
    static func sessionUpdate(
        model: String, instructions: String, voice: String,
        language: String?, tools: [AgentTool], sampleRate: Int,
        agentOutput: RealtimeModality.Output = .text,
        transcriptionModel: String = "gpt-4o-mini-transcribe",
        turnDetection: RealtimeTurnDetection = .semanticVAD(),
        overrides: [String: JSONValue] = [:]
    ) -> String {
        var transcription: [String: JSONValue] = ["model": .string(transcriptionModel)]
        if let language { transcription["language"] = .string(language) }

        let agentModality: JSONValue = agentOutput == .text ? .string("text") : .string("audio")
        var session: [String: JSONValue] = [
            "type": .string("realtime"),
            "model": .string(model),
            "output_modalities": .array([agentModality]),
            "instructions": .string(instructions),
            "audio": .object([
                "input": .object([
                    "format": .object(["type": .string(audioMimeType), "rate": .int(sampleRate)]),
                    "transcription": .object(transcription),
                    "turn_detection": turnDetectionValue(turnDetection),
                ]),
                "output": .object([
                    "format": .object(["type": .string(audioMimeType), "rate": .int(sampleRate)]),
                    "voice": .string(voice),
                ]),
            ]),
        ]
        if !tools.isEmpty {
            session["tools"] = .array(tools.map(functionSchema))
            session["tool_choice"] = .string("auto")
        }
        let merged = overrides.isEmpty
            ? JSONValue.object(session)
            : JSONValue.object(session).deepMerging(.object(overrides))
        return frame(["type": .string("session.update"), "session": merged])
    }

    static func appendAudio(_ base64: String) -> String {
        frame(["type": .string("input_audio_buffer.append"), "audio": .string(base64)])
    }

    static func userMessage(_ text: String) -> String {
        frame([
            "type": .string("conversation.item.create"),
            "item": .object([
                "type": .string("message"),
                "role": .string("user"),
                "content": .array([.object(["type": .string("input_text"), "text": .string(text)])]),
            ]),
        ])
    }

    /// A prior assistant message, for seeding history. Realtime content parts are typed by direction:
    /// an assistant item uses `output_text` (it's model-produced), a user item uses `input_text`. The
    /// GA `gpt-realtime` API rejects the old beta value `text` here with `invalid_value` — which only
    /// surfaced on reopening a chat that had a stored assistant turn to replay.
    static func assistantMessage(_ text: String) -> String {
        frame([
            "type": .string("conversation.item.create"),
            "item": .object([
                "type": .string("message"),
                "role": .string("assistant"),
                "content": .array([.object(["type": .string("output_text"), "text": .string(text)])]),
            ]),
        ])
    }

    /// A tool result fed back to the model — `output` is a JSON-encoded string.
    static func functionOutput(callId: String, output: String) -> String {
        frame([
            "type": .string("conversation.item.create"),
            "item": .object([
                "type": .string("function_call_output"),
                "call_id": .string(callId),
                "output": .string(output),
            ]),
        ])
    }

    /// `response.create`. With no `output`, runs under the session config; pass `output: .text` to override this one response to text-only. Audio responses go through `presenterResponse`/`directResponse`.
    static func createResponse(output: RealtimeModality.Output? = nil) -> String {
        switch output {
        case nil:
            return frame(["type": .string("response.create")])
        case .text:
            return frame([
                "type": .string("response.create"),
                "response": .object(["output_modalities": .array([.string("text")])]),
            ])
        case .audio, .audioAndText:
            preconditionFailure("createResponse supports a text-only override; audio responses use presenterResponse/directResponse")
        }
    }

    /// The grounded presenter: out-of-band (`conversation:"none"`), fed only the curated message.
    static func presenterResponse(instructions: String, feed: String, output: RealtimeModality.Output, voice: String) -> String {
        var response: [String: JSONValue] = [
            "conversation": .string("none"),
            "metadata": .object(["role": .string("presenter")]),
            "instructions": .string(instructions),
            "input": .array([.object([
                "type": .string("message"),
                "role": .string("user"),
                "content": .array([.object(["type": .string("input_text"), "text": .string(feed)])]),
            ])]),
        ]
        for (key, value) in responseModalities(output, voice: voice) { response[key] = value }
        return frame(["type": .string("response.create"), "response": .object(response)])
    }

    /// The direct (no-tool) reply: in-band (sees history), no curated input.
    static func directResponse(instructions: String, output: RealtimeModality.Output, voice: String) -> String {
        var response: [String: JSONValue] = [
            "metadata": .object(["role": .string("direct")]),
            "instructions": .string(instructions),
        ]
        for (key, value) in responseModalities(output, voice: voice) { response[key] = value }
        return frame(["type": .string("response.create"), "response": .object(response)])
    }

    static func cancelResponse() -> String {
        frame(["type": .string("response.cancel")])
    }

    // MARK: - Inbound

    /// Decode one inbound frame to a `ServerEvent`, or `nil` for events we don't handle.
    static func decode(_ json: String) -> ServerEvent? {
        guard let envelope = try? JSONDecoder().decode(Envelope.self, from: Data(json.utf8)) else { return nil }
        return classify(envelope)
    }

    // MARK: - Internals

    /// `create_response: true` is deliberate on both VAD types — the session's turn brain relies on
    /// the server opening the agent response; `.disabled` maps to JSON `null` (manual turns only).
    private static func turnDetectionValue(_ turnDetection: RealtimeTurnDetection) -> JSONValue {
        switch turnDetection {
        case .semanticVAD(let eagerness):
            var config: [String: JSONValue] = ["type": .string("semantic_vad"), "create_response": .bool(true)]
            if let eagerness { config["eagerness"] = .string(eagerness.rawValue) }
            return .object(config)
        case .serverVAD(let threshold, let prefixPaddingMs, let silenceDurationMs):
            var config: [String: JSONValue] = ["type": .string("server_vad"), "create_response": .bool(true)]
            if let threshold { config["threshold"] = .double(threshold) }
            if let prefixPaddingMs { config["prefix_padding_ms"] = .int(prefixPaddingMs) }
            if let silenceDurationMs { config["silence_duration_ms"] = .int(silenceDurationMs) }
            return .object(config)
        case .disabled:
            return .null
        }
    }

    private static func responseModalities(_ output: RealtimeModality.Output, voice: String) -> [String: JSONValue] {
        switch output {
        case .text:
            return ["output_modalities": .array([.string("text")])]
        case .audio, .audioAndText:
            return [
                "output_modalities": .array([.string("audio")]),
                "audio": .object(["output": .object(["voice": .string(voice)])]),
            ]
        }
    }

    private static func functionSchema(_ tool: AgentTool) -> JSONValue {
        .object([
            "type": .string("function"),
            "name": .string(tool.name),
            "description": .string(tool.description),
            "parameters": tool.inputSchema,
        ])
    }

    private static func frame(_ object: [String: JSONValue]) -> String {
        guard let data = try? JSONEncoder().encode(JSONValue.object(object)) else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }

    private static func classify(_ event: Envelope) -> ServerEvent? {
        let type = event.type
        switch type {
        case "input_audio_buffer.speech_started":
            return .speechStarted
        case "response.created":
            return .responseCreated(id: event.response?.id ?? "", role: event.response?.metadata?.role)
        case "response.done":
            return .responseDone(id: event.response?.id ?? "", usage: event.response?.usage.map { usage in
                RealtimeUsage(
                    inputTokens: usage.inputTokens,
                    outputTokens: usage.outputTokens,
                    inputAudioTokens: usage.inputTokenDetails?.audioTokens,
                    outputAudioTokens: usage.outputTokenDetails?.audioTokens,
                    cachedInputTokens: usage.inputTokenDetails?.cachedTokens
                )
            })
        case "error":
            return .error(code: event.error?.code)
        case "response.function_call_arguments.done":
            return .functionCallArguments(responseId: event.responseId ?? "", callId: event.callId ?? "", name: event.name, arguments: event.arguments ?? "{}")
        case "response.output_item.added":
            guard event.item?.type == "function_call", let callId = event.item?.callId ?? event.item?.id else { return nil }
            return .functionCallNamed(callId: callId, name: event.item?.name ?? "")
        default:
            break
        }
        // Suffix matches across GA + legacy names. Order matters: the transcript suffixes are tested
        // before the broad `audio.delta` so `output_audio_transcript.delta` isn't swallowed by it.
        // `audio.delta` stays intentionally broad to catch both GA (`output_audio.delta`) and legacy
        // (`audio.delta`); unhandled future names fall through to `nil` (the ignored long tail).
        if type.contains("input_audio_transcription") {
            if type.hasSuffix(".delta") { return .userTranscriptDelta(event.delta ?? "") }
            if type.hasSuffix(".completed") { return .userTranscriptCompleted(event.transcript ?? "") }
        }
        if type.hasSuffix("output_audio_transcript.delta") {
            return .audioTranscriptDelta(responseId: event.responseId ?? "", text: event.delta ?? "")
        }
        if type.hasSuffix("output_audio_transcript.done") {
            return .audioTranscriptDone(responseId: event.responseId ?? "", text: event.transcript ?? "")
        }
        if type.hasSuffix("audio.delta") {
            return .audioDelta(responseId: event.responseId ?? "", base64: event.delta ?? "")
        }
        if type.hasSuffix("output_text.delta") || type == "response.text.delta" {
            return .outputTextDelta(responseId: event.responseId ?? "", text: event.delta ?? "")
        }
        if type.hasSuffix("output_text.done") {
            return .outputTextDone(responseId: event.responseId ?? "", text: event.text ?? "")
        }
        return nil
    }

    private struct Envelope: Decodable {
        let type: String
        let delta: String?
        let transcript: String?
        let text: String?
        let responseId: String?
        let callId: String?
        let name: String?
        let arguments: String?
        let response: ResponseObject?
        let item: ItemObject?
        let error: ErrorObject?

        enum CodingKeys: String, CodingKey {
            case type, delta, transcript, text, name, arguments, response, item, error
            case responseId = "response_id"
            case callId = "call_id"
        }

        struct ResponseObject: Decodable {
            let id: String?
            let metadata: Metadata?
            let usage: Usage?
            struct Metadata: Decodable { let role: String? }
            // Scalar totals feed the tracer's `usage(...)`; the per-modality breakdown
            // (`input_token_details.audio_tokens`/`.cached_tokens`, `output_token_details.audio_tokens`)
            // rides span metadata — see `RealtimeUsage`.
            struct Usage: Decodable {
                let inputTokens: Int?
                let outputTokens: Int?
                let inputTokenDetails: TokenDetails?
                let outputTokenDetails: TokenDetails?
                enum CodingKeys: String, CodingKey {
                    case inputTokens = "input_tokens"
                    case outputTokens = "output_tokens"
                    case inputTokenDetails = "input_token_details"
                    case outputTokenDetails = "output_token_details"
                }
                // Canonical GA shape (verified against the Realtime SDK types): the cached *total*
                // is `input_token_details.cached_tokens` — a further `cached_tokens_details` splits it
                // by modality, which we don't surface. `cached_tokens` is absent on output details.
                struct TokenDetails: Decodable {
                    let audioTokens: Int?
                    let cachedTokens: Int?
                    enum CodingKeys: String, CodingKey {
                        case audioTokens = "audio_tokens"
                        case cachedTokens = "cached_tokens"
                    }
                }
            }
        }
        struct ItemObject: Decodable {
            let type: String?
            let id: String?
            let callId: String?
            let name: String?
            enum CodingKeys: String, CodingKey { case type, id, name; case callId = "call_id" }
        }
        struct ErrorObject: Decodable { let code: String? }
    }
}

/// An inbound Realtime API event the session acts on (the long tail is decoded to `nil` and ignored).
enum ServerEvent: Sendable, Equatable {
    case speechStarted
    case userTranscriptDelta(String)
    case userTranscriptCompleted(String)
    case functionCallNamed(callId: String, name: String)
    case functionCallArguments(responseId: String, callId: String, name: String?, arguments: String)
    case outputTextDelta(responseId: String, text: String)
    case outputTextDone(responseId: String, text: String)
    case audioDelta(responseId: String, base64: String)
    case audioTranscriptDelta(responseId: String, text: String)
    case audioTranscriptDone(responseId: String, text: String)
    case responseCreated(id: String, role: String?)
    case responseDone(id: String, usage: RealtimeUsage?)
    case error(code: String?)
}

/// Token accounting from a Realtime `response.done`: scalar totals plus the per-modality breakdown.
struct RealtimeUsage: Sendable, Equatable {
    let inputTokens: Int?
    let outputTokens: Int?
    let inputAudioTokens: Int?
    let outputAudioTokens: Int?
    let cachedInputTokens: Int?

    /// The breakdown as a span-metadata object (the usage seam itself is totals-only), or `nil` when the API reported none.
    ///
    /// Key names are a deliberate, now-frozen choice (they ride to every backend as attributes):
    /// stable OTel GenAI semconv has no audio-token keys, so `input_audio_tokens`/`output_audio_tokens`
    /// are AgentSquad extensions in that namespace; cached tokens use `cache_read_input_tokens` to
    /// match the prevailing GenAI-instrumentation convention (rather than inventing `cached_input_tokens`).
    var breakdownMetadata: JSONValue? {
        var fields: [String: JSONValue] = [:]
        if let v = inputAudioTokens { fields["gen_ai.usage.input_audio_tokens"] = .int(v) }
        if let v = outputAudioTokens { fields["gen_ai.usage.output_audio_tokens"] = .int(v) }
        if let v = cachedInputTokens { fields["gen_ai.usage.cache_read_input_tokens"] = .int(v) }
        return fields.isEmpty ? nil : .object(fields)
    }
}
