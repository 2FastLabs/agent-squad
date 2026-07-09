import Foundation

/// The production `Tracer`: routes spans through a `SpanProcessor`. Span creation lives here;
/// batching/redaction/export live in the processor and its `TraceExporter`, each swappable.
///
/// Nothing drains automatically: the app must `flush()` on background and `shutdown()` on terminate,
/// else sub-`batchSize` spans are lost and the consumer task is never released.
public struct ProcessingTracer: Tracer {
    private let processor: any SpanProcessor

    public init(processor: any SpanProcessor) {
        self.processor = processor
    }

    /// Convenience: batch spans and export them — wires a `BatchSpanProcessor` around the exporter.
    public init(exporter: any TraceExporter, batchSize: Int = 64, redaction: any Redactor = Redaction.default) {
        self.init(processor: BatchSpanProcessor(exporter: exporter, batchSize: batchSize, redaction: redaction))
    }

    public func startTrace(
        name: String, userId: String?, sessionId: String?, metadata: JSONValue?
    ) -> any SpanHandle {
        let id = UUID().uuidString
        let span = ProcessedSpan(traceId: id, id: id, parentId: nil, startedAt: Date(), userId: userId, sessionId: sessionId, processor: processor)
        processor.onOpen(SpanData(
            id: id, traceId: id, parentId: nil, kind: .trace, name: name,
            startedAt: span.startedAt, userId: userId, sessionId: sessionId, metadata: metadata
        ))
        return span
    }

    public func flush() async throws { try await processor.flush() }
    public func shutdown() async { await processor.shutdown() }
}

/// One node in a `ProcessingTracer` tree. A `Sendable` value forwarding its lifecycle to the shared
/// `SpanProcessor` synchronously (order-preserving); `startedAt` is captured at creation.
struct ProcessedSpan: GenerationHandle {
    let traceId: String
    let id: String
    let parentId: String?
    let startedAt: Date
    // Carried so children inherit them and stay correlatable even if a long-lived root never ends.
    let userId: String?
    let sessionId: String?
    let processor: any SpanProcessor

    func span(_ name: String, input: JSONValue?) -> any SpanHandle {
        child(kind: .span, name: name, input: input, model: nil)
    }

    func generation(_ name: String, model: String, input: JSONValue?) -> any GenerationHandle {
        child(kind: .generation, name: name, input: input, model: model)
    }

    func generation(_ name: String, model: String, input: JSONValue?, startedAt: Date) -> any GenerationHandle {
        child(kind: .generation, name: name, input: input, model: model, startedAt: startedAt)
    }

    func usage(promptTokens: Int?, completionTokens: Int?) {
        processor.onUsage(id: id, promptTokens: promptTokens, completionTokens: completionTokens)
    }

    func setInput(_ input: JSONValue) {
        processor.onSetInput(id: id, input: input)
    }

    func setMetadata(_ metadata: JSONValue) {
        processor.onSetMetadata(id: id, metadata: metadata)
    }

    func end(output: JSONValue?, error: (any Error)?) {
        // Stringify the error synchronously so the message is Sendable and stays ordered before later ones.
        processor.onEnd(id: id, endedAt: Date(), output: output, error: error.map { String(reflecting: $0) })
    }

    private func child(kind: TraceEvent.Kind, name: String, input: JSONValue?, model: String?, startedAt: Date = Date()) -> ProcessedSpan {
        let child = ProcessedSpan(traceId: traceId, id: UUID().uuidString, parentId: id, startedAt: startedAt, userId: userId, sessionId: sessionId, processor: processor)
        processor.onOpen(SpanData(
            id: child.id, traceId: traceId, parentId: child.parentId, kind: kind, name: name,
            startedAt: child.startedAt, input: input, model: model, userId: userId, sessionId: sessionId
        ))
        return child
    }
}
