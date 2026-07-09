import Foundation

/// Creates traces. The one thing an app wires; agents open child spans from the `SpanHandle` on
/// `AgentContext`. `startTrace` returns the trace's root span — ending it ends the trace.
public protocol Tracer: Sendable {
    func startTrace(
        name: String,
        userId: String?,
        sessionId: String?,
        metadata: JSONValue?
    ) -> any SpanHandle

    /// Export everything buffered now (e.g. on app-background).
    func flush() async throws
    /// Final drain + release resources (e.g. on app termination).
    func shutdown() async
}

extension Tracer {
    public func flush() async throws {}
    public func shutdown() async {}
}

/// A live span in a trace. The parent is passed explicitly (on `AgentContext`), never via
/// `@TaskLocal` — that doesn't survive `AsyncThrowingStream` producer tasks.
///
/// `setInput`/`setMetadata` apply only before `end`; a call after `end` is dropped. Both are no-op
/// by default, so a stateless tracer that doesn't retain open spans can ignore them.
public protocol SpanHandle: Sendable {
    /// Unique id; correlates exported events and wires parent/child links. The root span's `id` is
    /// the trace id, so `Tracer.startTrace`'s caller can deep-link with it.
    var id: String { get }
    /// Open a child span — a tool call, a runtime step.
    func span(_ name: String, input: JSONValue?) -> any SpanHandle
    /// Open a child generation — an LLM call (records model + token usage).
    func generation(_ name: String, model: String, input: JSONValue?) -> any GenerationHandle
    /// Open a child generation whose start is backdated to `startedAt` — for a call whose real start
    /// (e.g. a realtime `response.created`) precedes the moment we can materialize the span. Lets the
    /// exported span carry the true call latency instead of a ~0 duration. Defaults to ignoring
    /// `startedAt` (stamping now), so a stateless tracer need not implement it.
    func generation(_ name: String, model: String, input: JSONValue?, startedAt: Date) -> any GenerationHandle
    /// Set the input once it's known later than open (e.g. a transcript arriving after the turn span).
    func setInput(_ input: JSONValue)
    /// Attach metadata — a `.object` whose top-level keys ride to the backend as span attributes
    /// (e.g. `modality`, an audio/cached token breakdown `usage` can't hold). Last write wins.
    func setMetadata(_ metadata: JSONValue)
    /// Close this span, optionally recording its output or the error that ended it. A stateful tracer
    /// makes a second `end` (or a child opened after `end`) harmless, exporting each span once.
    func end(output: JSONValue?, error: (any Error)?)
}

extension SpanHandle {
    public func setInput(_ input: JSONValue) {}
    public func setMetadata(_ metadata: JSONValue) {}
    /// Default: ignore the backdate and stamp the start now — implementers that don't track a
    /// separate start time keep working unchanged.
    public func generation(_ name: String, model: String, input: JSONValue?, startedAt: Date) -> any GenerationHandle {
        generation(name, model: model, input: input)
    }
}

/// A span for an LLM call; adds token accounting.
public protocol GenerationHandle: SpanHandle {
    func usage(promptTokens: Int?, completionTokens: Int?)
}

/// The tracing pipeline's middle layer: receives span lifecycle from the `Tracer` and decides what
/// to do with finished spans (typically batch + hand to a `TraceExporter`). Swap it to change the
/// batching/forwarding strategy without touching the tracer or exporter.
///
/// Contract for the synchronous lifecycle hooks (`onOpen`/`onUsage`/`onSetInput`/`onSetMetadata`/`onEnd`):
/// - Non-blocking: they run on the agent's hot path; buffer the message and return (no network, I/O,
///   or blocking locks). `BatchSpanProcessor` does this by being an actor that only enqueues.
/// - Order-preserving per span id: `onOpen` → … → `onEnd` must keep arrival order, else token
///   attribution and parent linkage corrupt. A per-message detached `Task` or unordered fan-out breaks this.
public protocol SpanProcessor: Sendable {
    func onOpen(_ span: SpanData)
    func onUsage(id: String, promptTokens: Int?, completionTokens: Int?)
    /// Set/replace an open span's input. Applied only before its `onEnd`; a call for an
    /// ended/unknown id is dropped. No-op by default for processors that don't retain open spans.
    func onSetInput(id: String, input: JSONValue)
    /// Set/replace an open span's metadata. Applied only before its `onEnd`; a call for an
    /// ended/unknown id is dropped. No-op by default for processors that don't retain open spans.
    func onSetMetadata(id: String, metadata: JSONValue)
    func onEnd(id: String, endedAt: Date, output: JSONValue?, error: String?)
    /// Export everything buffered now.
    func flush() async throws
    /// Final drain + release the exporter.
    func shutdown() async
}

extension SpanProcessor {
    public func onSetInput(id: String, input: JSONValue) {}
    public func onSetMetadata(id: String, metadata: JSONValue) {}
}

/// A span's open-time facts, handed to `onOpen`. Usage and end arrive later via `onUsage`/`onEnd`.
public struct SpanData: Sendable {
    public let id: String
    public let traceId: String
    public let parentId: String?
    public let kind: TraceEvent.Kind
    public let name: String
    public let startedAt: Date
    public let input: JSONValue?
    public let model: String?
    public let userId: String?
    public let sessionId: String?
    public let metadata: JSONValue?

    public init(
        id: String,
        traceId: String,
        parentId: String?,
        kind: TraceEvent.Kind,
        name: String,
        startedAt: Date,
        input: JSONValue? = nil,
        model: String? = nil,
        userId: String? = nil,
        sessionId: String? = nil,
        metadata: JSONValue? = nil
    ) {
        self.id = id
        self.traceId = traceId
        self.parentId = parentId
        self.kind = kind
        self.name = name
        self.startedAt = startedAt
        self.input = input
        self.model = model
        self.userId = userId
        self.sessionId = sessionId
        self.metadata = metadata
    }
}

/// Where trace events go. The single variation point behind `BatchSpanProcessor`: batching,
/// ordering, and redaction are shared; an integration implements only `export`.
public protocol TraceExporter: Sendable {
    func export(_ batch: [TraceEvent]) async throws
    /// Drain anything buffered now (e.g. on app-background).
    func flush() async throws
    /// Final drain + release resources (e.g. on app termination).
    func shutdown() async
}

extension TraceExporter {
    public func flush() async throws {}
    public func shutdown() async {}
}
