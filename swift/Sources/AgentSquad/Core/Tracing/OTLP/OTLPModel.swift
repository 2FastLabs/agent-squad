import Foundation
import CryptoKit

/// Maps `TraceEvent`s to an OTLP/HTTP JSON `ExportTraceServiceRequest`. One resource + one scope
/// holds every span (each carries its own `traceId`). Attribute keys follow the OTel GenAI
/// conventions so backends render model + token usage richly.
enum OTLPMapper {
    static let scopeName = "agent-squad"
    static let scopeVersion = AgentSquadInfo.version

    static func request(for batch: [TraceEvent], serviceName: String) -> OTLPExportRequest {
        OTLPExportRequest(resourceSpans: [
            OTLPResourceSpans(
                resource: OTLPResource(attributes: [.init(key: "service.name", value: .string(serviceName))]),
                scopeSpans: [OTLPScopeSpans(
                    scope: OTLPScope(name: scopeName, version: scopeVersion),
                    spans: batch.map(span(from:))
                )]
            )
        ])
    }

    static func span(from event: TraceEvent) -> OTLPSpan {
        OTLPSpan(
            traceId: hex(event.traceId, bytes: 16),   // 16-byte trace id → 32 hex
            spanId: hex(event.id, bytes: 8),           // 8-byte span id → 16 hex
            parentSpanId: event.parentId.map { hex($0, bytes: 8) },
            name: event.name,
            kind: event.kind == .generation ? 3 : 1,   // CLIENT for LLM calls, else INTERNAL
            startTimeUnixNano: nanos(event.startedAt),
            endTimeUnixNano: nanos(event.endedAt ?? event.startedAt),
            attributes: attributes(of: event),
            // OTel reserves OK(1) for an explicit application assertion; instrumentation leaves
            // non-error spans UNSET(0) so the backend's own heuristics still apply.
            status: OTLPStatus(code: event.status == .error ? 2 : 0, message: event.error)
        )
    }

    private static func attributes(of event: TraceEvent) -> [OTLPKeyValue] {
        var attributes: [OTLPKeyValue] = []
        if let model = event.model, !model.isEmpty { attributes.append(.init(key: "gen_ai.request.model", value: .string(model))) }
        if let prompt = event.promptTokens { attributes.append(.init(key: "gen_ai.usage.input_tokens", value: .int(prompt))) }
        if let completion = event.completionTokens { attributes.append(.init(key: "gen_ai.usage.output_tokens", value: .int(completion))) }
        if let userId = event.userId { attributes.append(.init(key: "enduser.id", value: .string(userId))) }
        if let sessionId = event.sessionId { attributes.append(.init(key: "session.id", value: .string(sessionId))) }
        if let input = event.input { attributes.append(.init(key: "gen_ai.prompt", value: .string(jsonString(input)))) }
        if let output = event.output { attributes.append(.init(key: "gen_ai.completion", value: .string(jsonString(output)))) }
        // Metadata's top-level keys become attributes verbatim. Sorted for deterministic output; a
        // key the mapper already emits is skipped so it can't ship twice — the reserved one wins.
        if case .object(let fields)? = event.metadata {
            for key in fields.keys.sorted() where !reservedAttributeKeys.contains(key) {
                if let value = fields[key].flatMap(anyValue) { attributes.append(.init(key: key, value: value)) }
            }
        }
        return attributes
    }

    /// Attribute keys the mapper emits from `TraceEvent`'s own fields — metadata must not shadow them.
    private static let reservedAttributeKeys: Set<String> = [
        "gen_ai.request.model", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens",
        "enduser.id", "session.id", "gen_ai.prompt", "gen_ai.completion",
    ]

    /// Map a metadata scalar onto an OTLP `AnyValue`; arrays/objects are JSON-stringified, `null` and
    /// non-finite doubles dropped (a `.nan`/`.inf` would throw in `JSONEncoder` and sink the batch).
    private static func anyValue(_ value: JSONValue) -> OTLPAnyValue? {
        switch value {
        case .null: return nil
        case .bool(let b): return .bool(b)
        case .int(let i): return .int(i)
        case .double(let d): return d.isFinite ? .double(d) : nil
        case .string(let s): return .string(s)
        case .array, .object: return .string(jsonString(value))
        }
    }

    /// OTLP/JSON wants lowercase hex of a fixed byte width. An already-hex id is used verbatim; any
    /// other id falls back to SHA-256 so the output is always valid hex of the right length. Span ids
    /// take the first 8 bytes — a 2^32 birthday bound, negligible for chat traces.
    static func hex(_ id: String, bytes: Int) -> String {
        let stripped = id.replacingOccurrences(of: "-", with: "").lowercased()
        let source = (stripped.count >= bytes * 2 && stripped.allSatisfy(\.isHexDigit))
            ? stripped
            : SHA256.hash(data: Data(id.utf8)).map { String(format: "%02x", $0) }.joined()
        return String(source.prefix(bytes * 2))
    }

    /// Nanoseconds since the epoch. Split seconds from sub-second so the full value never routes
    /// through `Double`, which would lose the low digits past 2^53.
    static func nanos(_ date: Date) -> String {
        let seconds = date.timeIntervalSince1970
        let whole = seconds.rounded(.down)
        let fraction = seconds - whole
        return String(UInt64(whole) * 1_000_000_000 + UInt64((fraction * 1_000_000_000).rounded()))
    }

    private static func jsonString(_ value: JSONValue) -> String {
        guard let data = try? JSONEncoder().encode(value) else { return "" }
        return String(decoding: data, as: UTF8.self)
    }
}

// MARK: - Wire DTOs (OTLP/JSON; proto3-JSON field names are lowerCamelCase, matching these)

struct OTLPExportRequest: Encodable {
    let resourceSpans: [OTLPResourceSpans]
}

struct OTLPResourceSpans: Encodable {
    let resource: OTLPResource
    let scopeSpans: [OTLPScopeSpans]
}

struct OTLPResource: Encodable {
    let attributes: [OTLPKeyValue]
}

struct OTLPScopeSpans: Encodable {
    let scope: OTLPScope
    let spans: [OTLPSpan]
}

struct OTLPScope: Encodable {
    let name: String
    let version: String
}

struct OTLPSpan: Encodable {
    let traceId: String
    let spanId: String
    let parentSpanId: String?
    let name: String
    let kind: Int
    let startTimeUnixNano: String
    let endTimeUnixNano: String
    let attributes: [OTLPKeyValue]
    let status: OTLPStatus

    enum CodingKeys: String, CodingKey {
        case traceId, spanId, parentSpanId, name, kind, startTimeUnixNano, endTimeUnixNano, attributes, status
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(traceId, forKey: .traceId)
        try container.encode(spanId, forKey: .spanId)
        try container.encodeIfPresent(parentSpanId, forKey: .parentSpanId)   // omit (not null) for roots
        try container.encode(name, forKey: .name)
        try container.encode(kind, forKey: .kind)
        try container.encode(startTimeUnixNano, forKey: .startTimeUnixNano)
        try container.encode(endTimeUnixNano, forKey: .endTimeUnixNano)
        try container.encode(attributes, forKey: .attributes)
        try container.encode(status, forKey: .status)
    }
}

struct OTLPStatus: Encodable {
    let code: Int          // 0 UNSET · 1 OK · 2 ERROR
    let message: String?

    enum CodingKeys: String, CodingKey { case code, message }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(code, forKey: .code)
        try container.encodeIfPresent(message, forKey: .message)
    }
}

struct OTLPKeyValue: Encodable {
    let key: String
    let value: OTLPAnyValue
}

/// OTLP `AnyValue`: a tagged union. int64 is serialized as a **string** in proto3-JSON; double and
/// bool serialize as a JSON number / bool.
enum OTLPAnyValue: Encodable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)

    enum CodingKeys: String, CodingKey { case stringValue, intValue, doubleValue, boolValue }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .string(let value): try container.encode(value, forKey: .stringValue)
        case .int(let value): try container.encode(String(value), forKey: .intValue)
        case .double(let value): try container.encode(value, forKey: .doubleValue)
        case .bool(let value): try container.encode(value, forKey: .boolValue)
        }
    }
}
