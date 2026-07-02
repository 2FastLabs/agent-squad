import Foundation

/// Schema-less JSON for tool args/results, trace payloads, MCP `structuredContent`/`_meta`.
///
/// Number gotchas: whole-number doubles decode to `.int` (`1.0` → `1`); ids beyond `Int` lose precision as `.double`; `.double` must be finite. Carry those as `.string`.
public enum JSONValue: Sendable, Hashable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])
}

// MARK: - Codable

extension JSONValue: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        // Order matters: bool before int before double, so `true` stays bool and whole numbers round-trip as `.int`.
        if container.decodeNil() {
            self = .null
        } else if let bool = try? container.decode(Bool.self) {
            self = .bool(bool)
        } else if let int = try? container.decode(Int.self) {
            self = .int(int)
        } else if let double = try? container.decode(Double.self) {
            self = .double(double)
        } else if let string = try? container.decode(String.self) {
            self = .string(string)
        } else if let array = try? container.decode([JSONValue].self) {
            self = .array(array)
        } else if let object = try? container.decode([String: JSONValue].self) {
            self = .object(object)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Value is not a valid JSON type"
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}

// MARK: - Literals
// Build values inline: ["odds": 1.26, "live": true, "tags": ["a"]]

extension JSONValue: ExpressibleByNilLiteral {
    public init(nilLiteral: ()) { self = .null }
}

extension JSONValue: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: Bool) { self = .bool(value) }
}

extension JSONValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) { self = .int(value) }
}

extension JSONValue: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) { self = .double(value) }
}

extension JSONValue: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) { self = .string(value) }
}

extension JSONValue: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
}

extension JSONValue: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self = .object(Dictionary(uniqueKeysWithValues: elements))
    }
}

// MARK: - Reading

extension JSONValue {
    /// Read a field from an object value: `arguments["city"]`. Returns `nil` for a non-object or a
    /// missing key — so a chain like `arguments["a"]?["b"]` is safe on any shape.
    public subscript(key: String) -> JSONValue? {
        if case .object(let object) = self { return object[key] }
        return nil
    }

    /// Read an element from an array value. Returns `nil` for a non-array or an out-of-range index.
    public subscript(index: Int) -> JSONValue? {
        guard case .array(let array) = self, array.indices.contains(index) else { return nil }
        return array[index]
    }

    /// The wrapped `String`, or `nil` if this isn't a `.string`.
    public var stringValue: String? { if case .string(let value) = self { return value } else { return nil } }
    /// The wrapped `Int`, or `nil` if this isn't an `.int`.
    public var intValue: Int? { if case .int(let value) = self { return value } else { return nil } }
    /// The wrapped `Bool`, or `nil` if this isn't a `.bool`.
    public var boolValue: Bool? { if case .bool(let value) = self { return value } else { return nil } }
    /// The wrapped `Double` — also returns a whole `Int` as a `Double`; `nil` otherwise.
    public var doubleValue: Double? {
        switch self {
        case .double(let value): return value
        case .int(let value): return Double(value)
        default: return nil
        }
    }

    /// Deep-merge `override` into `self`: objects merge key-by-key recursively, everything else
    /// (scalars, arrays, mismatched types) is replaced by the override.
    func deepMerging(_ override: JSONValue) -> JSONValue {
        guard case .object(let base) = self, case .object(let patch) = override else { return override }
        var merged = base
        for (key, value) in patch {
            merged[key] = base[key].map { $0.deepMerging(value) } ?? value
        }
        return .object(merged)
    }
}
