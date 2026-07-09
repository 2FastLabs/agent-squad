import Foundation

@testable import AgentSquad

// MARK: - Transport

/// Records sent frames and lets a test push inbound frames into the session's pump.
final class MockRealtimeTransport: RealtimeTransport, @unchecked Sendable {
    let events: AsyncStream<String>
    private let inbound: AsyncStream<String>.Continuation
    private let lock = NSLock()
    private var _sent: [String] = []
    private var _closed = false
    init() { (events, inbound) = AsyncStream.makeStream(of: String.self) }
    var sent: [String] { lock.withLock { _sent } }
    var closed: Bool { lock.withLock { _closed } }
    func connect() async throws {}
    func send(_ json: String) async throws { lock.withLock { _sent.append(json) } }
    func close() async { lock.withLock { _closed = true }; inbound.finish() }
    func push(_ json: String) { inbound.yield(json) }
}

// MARK: - Event log

/// What `EventLog` consumes — the realtime event stream. The OpenAI assistants expose this without
/// formally conforming to `VoiceAssistant` (their public surface is structural), so the test target
/// declares the conformance it needs.
protocol RealtimeEventSource: Sendable {
    var events: AsyncStream<RealtimeEvent> { get }
}

extension OpenAIVoiceAssistant: RealtimeEventSource {}
extension OpenAIGroundedVoiceAssistant: RealtimeEventSource {}

/// Collects a voice session's emitted events (the sole consumer of its single-shot stream), with
/// typed accessors for the cases tests assert on.
final class EventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _events: [RealtimeEvent] = []
    func start(_ session: some RealtimeEventSource) { Task { for await event in session.events { self.append(event) } } }
    private func append(_ event: RealtimeEvent) { lock.withLock { _events.append(event) } }
    var all: [RealtimeEvent] { lock.withLock { _events } }
    var states: [RealtimePhase] { all.compactMap { if case .state(let p) = $0 { return p } else { return nil } } }
    var presenterTexts: [String] { all.compactMap { if case .presenterText(let t, _) = $0 { return t } else { return nil } } }
    var widgets: [UIPayload] { all.compactMap { if case .widget(let p) = $0 { return p } else { return nil } } }
    var audioDones: [Bool] { all.compactMap { if case .audioDone(let i) = $0 { return i } else { return nil } } }
    var errors: [String] { all.compactMap { if case .error(_, let m) = $0 { return m } else { return nil } } }
    var errorCodes: [String?] { all.compactMap { if case .error(let c, _) = $0 { return .some(c) } else { return nil } } }
    var audioCount: Int { all.filter(\.isAudio).count }
}

// MARK: - Tracers

/// Records the per-turn trace lifecycle — which spans were opened/ended, plus the input/output and
/// token usage attached to each — so both the structure and the captured payloads can be asserted.
final class RecordingTracer: Tracer, @unchecked Sendable {
    final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _opened: [String] = []
        private var _ended: [String] = []
        private var _input: [String: JSONValue] = [:]
        private var _output: [String: JSONValue] = [:]
        private var _usage: [String: (Int?, Int?)] = [:]
        private var _metadata: [String: JSONValue] = [:]
        func open(_ n: String, input: JSONValue? = nil) { lock.withLock { _opened.append(n); if let input { _input[n] = input } } }
        func setInput(_ n: String, _ input: JSONValue) { lock.withLock { _input[n] = input } }
        func setMetadata(_ n: String, _ metadata: JSONValue) { lock.withLock { _metadata[n] = metadata } }
        func close(_ n: String, output: JSONValue?) { lock.withLock { _ended.append(n); if let output { _output[n] = output } } }
        func record(_ n: String, prompt: Int?, completion: Int?) { lock.withLock { _usage[n] = (prompt, completion) } }
        var opened: [String] { lock.withLock { _opened } }
        var ended: [String] { lock.withLock { _ended } }
        func input(_ n: String) -> JSONValue? { lock.withLock { _input[n] } }
        func output(_ n: String) -> JSONValue? { lock.withLock { _output[n] } }
        func usage(_ n: String) -> (Int?, Int?)? { lock.withLock { _usage[n] } }
        func metadata(_ n: String) -> JSONValue? { lock.withLock { _metadata[n] } }
    }
    final class Span: GenerationHandle, @unchecked Sendable {
        let id: String
        let recorder: Recorder
        init(id: String, recorder: Recorder) { self.id = id; self.recorder = recorder }
        func span(_ name: String, input: JSONValue?) -> any SpanHandle { recorder.open(name, input: input); return Span(id: name, recorder: recorder) }
        func generation(_ name: String, model: String, input: JSONValue?) -> any GenerationHandle { recorder.open(name, input: input); return Span(id: name, recorder: recorder) }
        func setInput(_ input: JSONValue) { recorder.setInput(id, input) }
        func setMetadata(_ metadata: JSONValue) { recorder.setMetadata(id, metadata) }
        func end(output: JSONValue?, error: (any Error)?) { recorder.close(id, output: output) }
        func usage(promptTokens: Int?, completionTokens: Int?) { recorder.record(id, prompt: promptTokens, completion: completionTokens) }
    }
    let recorder = Recorder()
    func startTrace(name: String, userId: String?, sessionId: String?, metadata: JSONValue?) -> any SpanHandle {
        recorder.open(name); return Span(id: name, recorder: recorder)
    }
}

/// Captures span identity (`id`/`parentId`/`traceId`) and metadata, so the session-as-root
/// *structure* — turns (and their tool/presenter children) nesting under one session trace — can be
/// asserted.
final class StructureTracer: Tracer, @unchecked Sendable {
    struct Node: Sendable { let name: String; let id: String; let parentId: String?; let traceId: String; let metadata: JSONValue? }
    final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _opened: [Node] = []
        private var _ended: [String] = []
        func open(_ node: Node) { lock.withLock { _opened.append(node) } }
        func end(_ id: String) { lock.withLock { _ended.append(id) } }
        var opened: [Node] { lock.withLock { _opened } }
        var ended: [String] { lock.withLock { _ended } }
    }
    final class Span: GenerationHandle, @unchecked Sendable {
        let id: String; let traceId: String; let recorder: Recorder
        init(id: String, traceId: String, recorder: Recorder) { self.id = id; self.traceId = traceId; self.recorder = recorder }
        private func child(_ name: String) -> Span {
            let c = Span(id: UUID().uuidString, traceId: traceId, recorder: recorder)
            recorder.open(Node(name: name, id: c.id, parentId: id, traceId: traceId, metadata: nil))
            return c
        }
        func span(_ name: String, input: JSONValue?) -> any SpanHandle { child(name) }
        func generation(_ name: String, model: String, input: JSONValue?) -> any GenerationHandle { child(name) }
        func end(output: JSONValue?, error: (any Error)?) { recorder.end(id) }
        func usage(promptTokens: Int?, completionTokens: Int?) {}
    }
    let recorder = Recorder()
    func startTrace(name: String, userId: String?, sessionId: String?, metadata: JSONValue?) -> any SpanHandle {
        let id = UUID().uuidString
        recorder.open(Node(name: name, id: id, parentId: nil, traceId: id, metadata: metadata))
        return Span(id: id, traceId: id, recorder: recorder)
    }
}

// MARK: - Storage

/// A `ChatStorage` that records what's saved and returns a fixed seed on fetch.
final class RecordingStore: ChatStorage, @unchecked Sendable {
    private let lock = NSLock()
    private var _saved: [ConversationMessage] = []
    private let seed: [ConversationMessage]
    init(seed: [ConversationMessage] = []) { self.seed = seed }
    var saved: [ConversationMessage] { lock.withLock { _saved } }
    func fetch(userId: String, sessionId: String, agentId: String, maxMessages: Int?) async throws -> [ConversationMessage] { seed }
    func save(_ message: ConversationMessage, userId: String, sessionId: String, agentId: String, maxMessages: Int?) async throws { lock.withLock { _saved.append(message) } }
    func saveMessages(_ messages: [ConversationMessage], userId: String, sessionId: String, agentId: String, maxMessages: Int?) async throws { lock.withLock { _saved.append(contentsOf: messages) } }
    func fetchAllChats(userId: String, sessionId: String) async throws -> [ConversationMessage] { lock.withLock { _saved } }
}

// MARK: - Polling

/// Polls `condition` until it holds or the deadline passes — for asserting on async event delivery.
func eventually(_ condition: @escaping @Sendable () -> Bool, within seconds: Double = 2) async {
    let deadline = ContinuousClock().now + .seconds(seconds)
    while ContinuousClock().now < deadline {
        if condition() { return }
        try? await Task.sleep(for: .milliseconds(3))
    }
}

// MARK: - Frame inspection

/// The `"type"` field of a JSON frame, if present.
func frameType(of frame: String) -> String? {
    (try? JSONSerialization.jsonObject(with: Data(frame.utf8)) as? [String: Any])?["type"] as? String
}

/// The `"type"` of every frame the transport has sent, in order.
func sentTypes(_ transport: MockRealtimeTransport) -> [String] { transport.sent.compactMap(frameType(of:)) }

/// Text parts of a message, joined.
func messageText(_ m: ConversationMessage) -> String {
    m.parts.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined()
}

// MARK: - Tools

/// A provider with one UI-bearing `odds` tool returning a fixed result + widget.
func oddsTools() -> StubToolProvider {
    let ui = UIPayload(resourceURI: "ui://odds", mimeType: "text/html;profile=mcp-app")
    return StubToolProvider(
        tool: AgentTool(name: "odds", description: "match odds", ui: "ui://odds"),
        result: ToolResult(content: [.text("PSG 2.5")], structuredContent: .object(["home": .double(2.5)]), ui: ui)
    )
}

// MARK: - Inbound frame builders

func funcArgs(_ rid: String, _ callId: String, _ name: String) -> String {
    #"{"type":"response.function_call_arguments.done","response_id":"\#(rid)","call_id":"\#(callId)","name":"\#(name)","arguments":"{}"}"#
}

func responseDone(_ id: String, inputTokens: Int? = nil, outputTokens: Int? = nil) -> String {
    if let i = inputTokens, let o = outputTokens {
        return #"{"type":"response.done","response":{"id":"\#(id)","usage":{"input_tokens":\#(i),"output_tokens":\#(o)}}}"#
    }
    return #"{"type":"response.done","response":{"id":"\#(id)"}}"#
}

func responseFailed(_ id: String, errorType: String = "server_error", errorCode: String? = "internal_error") -> String {
    let error = errorCode.map { #"{"type":"\#(errorType)","code":"\#($0)"}"# } ?? #"{"type":"\#(errorType)"}"#
    return #"{"type":"response.done","response":{"id":"\#(id)","status":"failed","status_details":{"type":"failed","error":\#(error)}}}"#
}

func serverError(_ code: String?, message: String) -> String {
    let codeJSON = code.map { #""\#($0)""# } ?? "null"
    return #"{"type":"error","error":{"type":"invalid_request_error","code":\#(codeJSON),"message":"\#(message)"}}"#
}

func responseCreated(_ id: String, role: String? = nil) -> String {
    if let role { return #"{"type":"response.created","response":{"id":"\#(id)","metadata":{"role":"\#(role)"}}}"# }
    return #"{"type":"response.created","response":{"id":"\#(id)"}}"#   // the gatherer/agent turn — no role
}

func audioDelta(_ id: String, _ bytes: String, item: String = "item_1") -> String {
    // `item_id` is required on the real event (see reference/server-events.md).
    #"{"type":"response.output_audio.delta","response_id":"\#(id)","item_id":"\#(item)","delta":"\#(Data(bytes.utf8).base64EncodedString())"}"#
}

func userSaid(_ text: String) -> String {
    #"{"type":"conversation.item.input_audio_transcription.completed","transcript":"\#(text)"}"#
}

func audioTranscriptDone(_ id: String, _ text: String) -> String {
    #"{"type":"response.output_audio_transcript.done","response_id":"\#(id)","transcript":"\#(text)"}"#
}
