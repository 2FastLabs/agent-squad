import Foundation
import Testing

@testable import AgentSquad

@Suite struct ProcessingTracerTests {
    // MARK: - Test doubles

    /// Collects exported batches; can be told to fail to prove the buffer never crashes the caller
    /// on the automatic path and does surface failures on an explicit flush.
    private actor RecordingExporter: TraceExporter {
        private(set) var batches: [[TraceEvent]] = []
        private(set) var flushed = 0
        private(set) var didShutdown = false
        private let failExport: Bool

        init(failExport: Bool = false) { self.failExport = failExport }

        var events: [TraceEvent] { batches.flatMap { $0 } }

        func export(_ batch: [TraceEvent]) async throws {
            batches.append(batch)
            if failExport { throw CancellationError() }
        }
        func flush() async throws { flushed += 1 }
        func shutdown() async { didShutdown = true }
    }

    /// Records the lifecycle calls a `ProcessingTracer` forwards — proves the swappable seam exists.
    private final class RecordingProcessor: SpanProcessor, @unchecked Sendable {
        enum Call: Equatable { case open(id: String, name: String, parent: String?), usage(id: String), end(id: String), flush, shutdown }
        private let lock = NSLock()
        private var _calls: [Call] = []
        var calls: [Call] { lock.withLock { _calls } }

        func onOpen(_ span: SpanData) { lock.withLock { _calls.append(.open(id: span.id, name: span.name, parent: span.parentId)) } }
        func onUsage(id: String, promptTokens: Int?, completionTokens: Int?) { lock.withLock { _calls.append(.usage(id: id)) } }
        func onEnd(id: String, endedAt: Date, output: JSONValue?, error: String?) { lock.withLock { _calls.append(.end(id: id)) } }
        func flush() async throws { lock.withLock { _calls.append(.flush) } }
        func shutdown() async { lock.withLock { _calls.append(.shutdown) } }
    }

    // MARK: - Tests

    @Test func exportsWhenBatchSizeReached() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 2)

        tracer.startTrace(name: "a", userId: nil, sessionId: nil, metadata: nil).end(output: nil, error: nil)
        tracer.startTrace(name: "b", userId: nil, sessionId: nil, metadata: nil).end(output: nil, error: nil)
        try await tracer.flush()   // barrier: both closes are processed (auto-export fires) before this returns

        #expect(await Set(exporter.events.map(\.name)) == ["a", "b"])
        #expect(await exporter.batches.first?.count == 2)   // exported as one batch, not singly
    }

    @Test func flushDrainsPartialBatch() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)   // never auto-flushes

        tracer.startTrace(name: "solo", userId: nil, sessionId: nil, metadata: nil).end(output: nil, error: nil)
        try await tracer.flush()

        #expect(await exporter.events.map(\.name) == ["solo"])
        #expect(await exporter.flushed == 1)
    }

    @Test func flushWithNothingPendingStillFlushesExporterOnce() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        try await tracer.flush()

        #expect(await exporter.events.isEmpty)        // no spurious empty batch exported
        #expect(await exporter.flushed == 1)
    }

    @Test func shutdownDrainsAndReleasesExporter() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        tracer.startTrace(name: "t", userId: nil, sessionId: nil, metadata: nil).end(output: nil, error: nil)
        await tracer.shutdown()

        #expect(await exporter.events.map(\.name) == ["t"])
        #expect(await exporter.didShutdown)
    }

    @Test func nestsChildSpansUnderTheRootAndKeepsTokenCounts() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        let root = tracer.startTrace(name: "turn", userId: nil, sessionId: nil, metadata: nil)
        let gen = root.generation("llm", model: "gpt", input: nil)
        gen.usage(promptTokens: 10, completionTokens: 5)   // recorded immediately before end…
        gen.end(output: nil, error: nil)
        root.end(output: nil, error: nil)
        try await tracer.flush()

        let events = await exporter.events
        let rootEvent = try #require(events.first { $0.kind == .trace })
        let genEvent = try #require(events.first { $0.kind == .generation })
        #expect(genEvent.parentId == rootEvent.id)
        #expect(genEvent.traceId == rootEvent.id)
        #expect(genEvent.model == "gpt")
        #expect(genEvent.promptTokens == 10)        // …and ordering guarantees they survive finalize
        #expect(genEvent.completionTokens == 5)
    }

    @Test func generationBackdatesStartedAtToTheGivenTime() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        // A realtime answer whose call started well before we could materialize the span: backdating
        // `startedAt` makes the exported span carry the real latency instead of a ~0 duration.
        let start = Date(timeIntervalSince1970: 1_000)
        let root = tracer.startTrace(name: "turn", userId: nil, sessionId: nil, metadata: nil)
        let gen = root.generation("response", model: "gpt", input: nil, startedAt: start)
        gen.end(output: nil, error: nil)
        root.end(output: nil, error: nil)
        try await tracer.flush()

        let genEvent = try #require(await exporter.events.first { $0.kind == .generation })
        #expect(genEvent.startedAt == start)                        // the backdate survives finalize
        let ended = try #require(genEvent.endedAt)
        #expect(ended.timeIntervalSince(start) > 0)                 // and yields a non-zero latency
    }

    @Test func setInputAfterOpenIsAppliedToTheExportedSpan() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        // A root opened without input (e.g. a realtime turn whose transcript arrives later) can have
        // its input set before it ends; the exported event carries it.
        let root = tracer.startTrace(name: "turn", userId: nil, sessionId: nil, metadata: nil)
        root.setInput(.string("what are tonight's odds?"))
        root.end(output: .string("PSG 2.5"), error: nil)
        try await tracer.flush()

        let event = try #require(await exporter.events.first)
        #expect(event.input == .string("what are tonight's odds?"))
        #expect(event.output == .string("PSG 2.5"))
    }

    @Test func childSpansInheritUserAndSessionId() async throws {
        // Under a long-lived root (e.g. a realtime session), a child must stay correlatable on its
        // own — so userId/sessionId propagate down, not just onto the root.
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100, redaction: Redaction(hashUserIds: false))
        let root = tracer.startTrace(name: "session", userId: "u", sessionId: "s", metadata: nil)
        let turn = root.span("turn", input: nil)
        let gen = turn.generation("llm", model: "gpt", input: nil)
        gen.end(output: nil, error: nil)
        turn.end(output: nil, error: nil)
        root.end(output: nil, error: nil)
        try await tracer.flush()

        let events = await exporter.events
        #expect(events.count == 3)
        #expect(events.allSatisfy { $0.userId == "u" && $0.sessionId == "s" })   // every span, not just the root
    }

    @Test func setInputAfterEndIsDropped() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        let root = tracer.startTrace(name: "turn", userId: nil, sessionId: nil, metadata: nil)
        root.end(output: nil, error: nil)
        root.setInput(.string("too late"))   // ordered after close → no open span to apply it to
        try await tracer.flush()

        let events = await exporter.events
        #expect(events.count == 1)                 // the late setInput doesn't resurrect or duplicate the span
        #expect(events.first?.input == nil)        // and it's not applied retroactively
    }

    @Test func setMetadataAfterOpenIsAppliedToTheExportedSpan() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        // A realtime generation whose token breakdown arrives in `response.done` (after the span
        // opens) attaches it as metadata before ending; the exported event carries it.
        let root = tracer.startTrace(name: "turn", userId: nil, sessionId: nil, metadata: nil)
        let gen = root.generation("response", model: "gpt-realtime", input: nil)
        gen.usage(promptTokens: 1_240, completionTokens: 820)
        gen.setMetadata(.object(["gen_ai.usage.input_audio_tokens": .int(1_000),
                                 "gen_ai.usage.output_audio_tokens": .int(700)]))
        gen.end(output: nil, error: nil)
        root.end(output: nil, error: nil)
        try await tracer.flush()

        let genEvent = try #require(await exporter.events.first { $0.kind == .generation })
        #expect(genEvent.metadata == .object(["gen_ai.usage.input_audio_tokens": .int(1_000),
                                              "gen_ai.usage.output_audio_tokens": .int(700)]))
        #expect(genEvent.promptTokens == 1_240)   // totals still ride the typed seam
    }

    @Test func lastSetMetadataWinsAndReplaces() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        // Open-time root metadata is replaced wholesale by a later setMetadata (last write wins).
        let root = tracer.startTrace(name: "turn", userId: nil, sessionId: nil, metadata: .object(["modality": .string("text")]))
        root.setMetadata(.object(["modality": .string("audio")]))
        root.end(output: nil, error: nil)
        try await tracer.flush()

        let event = try #require(await exporter.events.first)
        #expect(event.metadata == .object(["modality": .string("audio")]))
    }

    @Test func setMetadataAfterEndIsDropped() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        let root = tracer.startTrace(name: "turn", userId: nil, sessionId: nil, metadata: nil)
        root.end(output: nil, error: nil)
        root.setMetadata(.object(["modality": .string("audio")]))   // ordered after close → dropped
        try await tracer.flush()

        let events = await exporter.events
        #expect(events.count == 1)                  // the late setMetadata doesn't resurrect or duplicate the span
        #expect(events.first?.metadata == nil)      // and it's not applied retroactively
    }

    @Test func recordsErrorStatus() async throws {
        struct Boom: Error {}
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        tracer.startTrace(name: "t", userId: nil, sessionId: nil, metadata: nil).end(output: nil, error: Boom())
        try await tracer.flush()

        let event = try #require(await exporter.events.first)
        #expect(event.status == .error)
        #expect(event.error?.contains("Boom") == true)
    }

    @Test func doubleEndExportsSpanOnce() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)

        let span = tracer.startTrace(name: "once", userId: nil, sessionId: nil, metadata: nil)
        span.end(output: nil, error: nil)
        span.end(output: nil, error: nil)
        try await tracer.flush()

        #expect(await exporter.events.count == 1)
    }

    @Test func exactlyBatchSizeExportsRemainderOnFlush() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 2)

        for name in ["a", "b", "c"] {   // 2 auto-export as one batch; "c" remains pending
            tracer.startTrace(name: name, userId: nil, sessionId: nil, metadata: nil).end(output: nil, error: nil)
        }
        try await tracer.flush()

        let batches = await exporter.batches
        #expect(batches.count == 2)
        #expect(batches.first?.count == 2)
        #expect(batches.last?.count == 1)
        #expect(await Set(exporter.events.map(\.name)) == ["a", "b", "c"])
    }

    @Test func automaticPathSwallowsExportFailures() async throws {
        let exporter = RecordingExporter(failExport: true)
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 1)

        tracer.startTrace(name: "t", userId: nil, sessionId: nil, metadata: nil).end(output: nil, error: nil)
        try await tracer.flush()   // the auto-export already failed-and-was-swallowed; flush itself won't throw

        #expect(await exporter.batches.isEmpty == false)   // export was attempted
    }

    @Test func flushSurfacesExportFailures() async throws {
        let exporter = RecordingExporter(failExport: true)
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)   // stays pending until flush

        tracer.startTrace(name: "t", userId: nil, sessionId: nil, metadata: nil).end(output: nil, error: nil)
        await #expect(throws: CancellationError.self) {
            try await tracer.flush()
        }
        // The failed batch is dropped, not re-enqueued — a follow-up flush has nothing to retry.
        try await tracer.flush()
        #expect(await exporter.batches.count == 1)
    }

    // MARK: - Redaction

    @Test func hashesUserIdAndClipsLongStrings() async throws {
        let redaction = Redaction(hashUserIds: true, maxStringLength: 5)
        let event = TraceEvent(
            traceId: "t", id: "t", kind: .trace, name: "n",
            startedAt: Date(), endedAt: Date(),
            output: .string("abcdefghij"),
            userId: "user-123"
        )
        let redacted = redaction.redact(event)

        #expect(redacted.userId != "user-123")
        #expect(redacted.userId?.count == 16)   // SHA-256 prefix, 8 bytes hex
        #expect(redacted.output == .string("abcde…"))
    }

    @Test func clipsStringsNestedInsideJSONObjectsAndArrays() async throws {
        let redaction = Redaction(hashUserIds: false, maxStringLength: 3)
        let event = TraceEvent(
            traceId: "t", id: "t", kind: .span, name: "n",
            startedAt: Date(), endedAt: Date(),
            metadata: .object([
                "key": .string("toolong"),
                "list": .array([.string("alsolong"), .int(7)])
            ])
        )
        let redacted = redaction.redact(event)
        #expect(redacted.metadata == .object([
            "key": .string("too…"),
            "list": .array([.string("als…"), .int(7)])
        ]))
    }

    @Test func numericMetadataSurvivesRedactionUnchanged() async throws {
        // Token-breakdown counts must reach the backend intact — clipping only touches strings, so
        // numeric metadata (kept as `.int`, never stringified) passes through even at a tiny limit.
        let redaction = Redaction(hashUserIds: true, maxStringLength: 1)
        let event = TraceEvent(
            traceId: "t", id: "t", kind: .generation, name: "response",
            startedAt: Date(), endedAt: Date(),
            metadata: .object([
                "gen_ai.usage.input_audio_tokens": .int(1_000),
                "gen_ai.usage.output_audio_tokens": .int(700),
                "modality": .string("audio")   // a short string survives; a long one would clip
            ])
        )
        #expect(redaction.redact(event).metadata == .object([
            "gen_ai.usage.input_audio_tokens": .int(1_000),
            "gen_ai.usage.output_audio_tokens": .int(700),
            "modality": .string("a…")
        ]))
    }

    @Test func leavesUserIdRawWhenHashingDisabled() async throws {
        let redaction = Redaction(hashUserIds: false, maxStringLength: nil)
        let event = TraceEvent(
            traceId: "t", id: "t", kind: .trace, name: "n",
            startedAt: Date(), endedAt: Date(), userId: "user-123"
        )
        #expect(redaction.redact(event).userId == "user-123")
    }

    @Test func usesACustomRedactorBeforeExport() async throws {
        // A user-supplied Redactor: drop the payloads entirely, keep everything else untouched.
        struct DropPayloads: Redactor {
            func redact(_ event: TraceEvent) -> TraceEvent {
                TraceEvent(
                    traceId: event.traceId, id: event.id, parentId: event.parentId,
                    kind: event.kind, name: event.name, status: event.status,
                    startedAt: event.startedAt, endedAt: event.endedAt,
                    input: nil, output: nil, error: event.error,
                    model: event.model, promptTokens: event.promptTokens,
                    completionTokens: event.completionTokens,
                    userId: event.userId, sessionId: event.sessionId, metadata: event.metadata
                )
            }
        }
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 1, redaction: DropPayloads())
        tracer.startTrace(name: "turn", userId: "u", sessionId: "s", metadata: nil)
            .end(output: "secret answer", error: nil)
        try await tracer.flush()

        let event = try #require(await exporter.events.first)
        #expect(event.output == nil)        // the custom redactor dropped the payload…
        #expect(event.userId == "u")        // …and left the user id raw (the built-in Redaction did not run)
    }

    // MARK: - The Tracer → SpanProcessor seam

    @Test func forwardsSpanLifecycleToTheProcessorInOrder() async throws {
        let processor = RecordingProcessor()
        let tracer = ProcessingTracer(processor: processor)

        let root = tracer.startTrace(name: "turn", userId: "u", sessionId: "s", metadata: nil)
        let gen = root.generation("llm", model: "gpt", input: nil)
        gen.usage(promptTokens: 1, completionTokens: 2)
        gen.end(output: nil, error: nil)
        root.end(output: nil, error: nil)

        let calls = processor.calls
        // Root opens first; the generation opens parented to the root; usage precedes the gen's end.
        guard case .open(let rootId, "turn", nil) = calls.first else { Issue.record("expected root open first"); return }
        guard case .open(let genId, "llm", rootId) = calls[1] else { Issue.record("expected gen open parented to root"); return }
        #expect(calls[2] == .usage(id: genId))
        #expect(calls[3] == .end(id: genId))
        #expect(calls[4] == .end(id: rootId))
    }

    @Test func forwardsFlushAndShutdownToTheProcessor() async throws {
        let processor = RecordingProcessor()
        let tracer = ProcessingTracer(processor: processor)
        try await tracer.flush()
        await tracer.shutdown()
        #expect(processor.calls == [.flush, .shutdown])
    }

    @Test func defaultTracerLifecycleIsAHarmlessNoOp() async throws {
        // OSLogTracer defines neither flush nor shutdown — it relies on the Tracer protocol defaults.
        let tracer = OSLogTracer()
        try await tracer.flush()
        await tracer.shutdown()
    }

    @Test func tokenCountsSurviveEvenWhenEachSpanAutoExports() async throws {
        // batchSize 1 ⇒ the root's end auto-exports before the gen exists; the gen's usage must
        // still land before its end finalizes (the ordering guarantee at the export boundary).
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 1)
        let root = tracer.startTrace(name: "turn", userId: nil, sessionId: nil, metadata: nil)
        let gen = root.generation("llm", model: "gpt", input: nil)
        gen.usage(promptTokens: 7, completionTokens: 3)
        gen.end(output: nil, error: nil)
        root.end(output: nil, error: nil)
        try await tracer.flush()

        let genEvent = try #require(await exporter.events.first { $0.kind == .generation })
        #expect(genEvent.promptTokens == 7)
        #expect(genEvent.completionTokens == 3)
    }

    @Test func shutdownIsIdempotentAndPostShutdownSpansAreDropped() async throws {
        let exporter = RecordingExporter()
        let tracer = ProcessingTracer(exporter: exporter, batchSize: 100)
        await tracer.shutdown()
        await tracer.shutdown()   // second shutdown must not hang or crash
        // A span opened after shutdown is silently dropped (channel terminated), not a crash.
        tracer.startTrace(name: "late", userId: nil, sessionId: nil, metadata: nil).end(output: nil, error: nil)
        try await tracer.flush()
        #expect(await exporter.events.isEmpty)
    }
}
