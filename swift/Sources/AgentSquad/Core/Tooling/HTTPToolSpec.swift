import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking   // URLSession lives here on non-Apple platforms
#endif

/// HTTP verb for an ``HTTPToolSpec``.
public enum HTTPMethod: String, Sendable, Hashable {
    case get = "GET"
    case post = "POST"
    case put = "PUT"
    case patch = "PATCH"
    case delete = "DELETE"
    case head = "HEAD"

    /// Verbs that carry their leftover arguments as a request body rather than as query items.
    var carriesBody: Bool { self == .post || self == .put || self == .patch }
}

/// How leftover arguments become a body, for body-carrying verbs. `GET`/`HEAD`/`DELETE` ignore this
/// and always map leftover arguments to query items.
public enum HTTPBody: Sendable, Hashable {
    /// JSON body for `POST`/`PUT`/`PATCH`, nothing otherwise. The default.
    case auto
    /// Always a JSON body of the leftover arguments.
    case json
    /// No body; drop leftover arguments.
    case empty
}

/// A raw HTTP response handed to a ``ResponseMapping``. `Sendable` so it crosses actor boundaries.
public struct HTTPToolResponse: Sendable {
    public let status: Int
    public let data: Data
    public let headers: [String: String]

    public init(status: Int, data: Data, headers: [String: String] = [:]) {
        self.status = status
        self.data = data
        self.headers = headers
    }

    public var isSuccess: Bool { (200..<300).contains(status) }
    public var bodyText: String { String(decoding: data, as: UTF8.self) }
    /// Decode the body as JSON. Throws if the body is not valid JSON.
    public func json() throws -> JSONValue { try JSONDecoder().decode(JSONValue.self, from: data) }
}

/// Turns an ``HTTPToolResponse`` into a `ToolResult`. Use ``standard``/``jsonEnvelopeError`` or
/// ``custom(_:)``. Prefer a `isError` result over `throws` so the model can recover.
public struct ResponseMapping: Sendable {
    let map: @Sendable (HTTPToolResponse) throws -> ToolResult

    public init(_ map: @escaping @Sendable (HTTPToolResponse) throws -> ToolResult) { self.map = map }

    public static func custom(_ map: @escaping @Sendable (HTTPToolResponse) throws -> ToolResult) -> ResponseMapping {
        ResponseMapping(map)
    }

    /// 2xx → body text as the model's content plus the parsed JSON as `structuredContent` (for the
    /// UI); non-2xx → a tool-level failure carrying the status and a body snippet.
    public static let standard = ResponseMapping { response in
        guard response.isSuccess else {
            let snippet = response.bodyText.prefix(500)
            return .failure("HTTP \(response.status)\(snippet.isEmpty ? "" : ": \(snippet)")")
        }
        let text = response.bodyText
        let structured = (try? response.json()) ?? .object([:])
        return ToolResult(
            content: text.isEmpty ? nil : [.text(text)],
            structuredContent: structured
        )
    }

    /// For APIs that always answer `200 OK` and signal failure with an error field in the body
    /// (e.g. `{"error_code": "..."}`). A present `errorKey` becomes a tool-level failure; otherwise
    /// behaves like ``standard``.
    public static func jsonEnvelope(errorKey: String = "error_code", messageKey: String = "error") -> ResponseMapping {
        ResponseMapping { response in
            let json = (try? response.json()) ?? .object([:])
            if let code = json[errorKey]?.stringValue {
                return .failure("\(code): \(json[messageKey]?.stringValue ?? "error")")
            }
            guard response.isSuccess else { return .failure("HTTP \(response.status)") }
            let text = response.bodyText
            return ToolResult(content: text.isEmpty ? nil : [.text(text)], structuredContent: json)
        }
    }

    /// `jsonEnvelope()` with the default `error_code` / `error` keys.
    public static let jsonEnvelopeError = jsonEnvelope()
}

/// Sends an HTTP request — the seam that keeps the HTTP stack out of the tool layer. Default is
/// ``URLSessionInvoker``; tests inject a mock.
public protocol HTTPInvoker: Sendable {
    func send(_ request: URLRequest) async throws -> HTTPToolResponse
}

/// ``HTTPInvoker`` backed by `URLSession`.
public struct URLSessionInvoker: HTTPInvoker {
    private let session: URLSession

    public init(session: URLSession = .shared) { self.session = session }

    public func send(_ request: URLRequest) async throws -> HTTPToolResponse {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw HTTPToolError.nonHTTPResponse }
        var headers: [String: String] = [:]
        for (key, value) in http.allHeaderFields {
            if let key = key as? String { headers[key] = "\(value)" }
        }
        return HTTPToolResponse(status: http.statusCode, data: data, headers: headers)
    }
}

/// Infrastructure-level failures from the HTTP tool layer (thrown, not returned as a tool error).
public enum HTTPToolError: Error, Sendable, Equatable {
    case invalidURL(String)
    case nonHTTPResponse
}

/// Declarative description of an HTTP API call. Argument mapping, by convention:
/// - `{token}` placeholders in `url` are filled from the matching argument (percent-encoded) and consumed.
/// - Leftover arguments become **query items** for `GET`/`HEAD`/`DELETE`, or a **JSON body** otherwise.
///
/// Keep credentials out of the model with `secrets`/`headers`; use `hostArguments` for host-supplied
/// values (e.g. `session_id`) injected per call and hidden from the schema.
public struct HTTPToolSpec: Sendable {
    public var method: HTTPMethod
    /// URL, optionally templated with `{token}` placeholders filled from arguments.
    public var url: String
    /// Static headers sent on every request.
    public var headers: [String: String]
    /// Auth headers merged into the request but kept out of `headers` (and traces). Never modelled.
    public var secrets: [String: String]
    /// Host-supplied values injected on every call and hidden from the model's schema.
    public var hostArguments: [String: JSONValue]
    public var body: HTTPBody
    public var response: ResponseMapping
    public var timeout: TimeInterval
    /// HTTP stack; defaults to `URLSession`, swap for a mock in tests.
    public var invoker: any HTTPInvoker

    public init(
        method: HTTPMethod,
        url: String,
        headers: [String: String] = [:],
        secrets: [String: String] = [:],
        hostArguments: [String: JSONValue] = [:],
        body: HTTPBody = .auto,
        response: ResponseMapping = .standard,
        timeout: TimeInterval = 30,
        invoker: any HTTPInvoker = URLSessionInvoker()
    ) {
        self.method = method
        self.url = url
        self.headers = headers
        self.secrets = secrets
        self.hostArguments = hostArguments
        self.body = body
        self.response = response
        self.timeout = timeout
        self.invoker = invoker
    }

    // MARK: - Request building

    /// Build the `URLRequest` for a call. `arguments` is the model's arguments already merged with
    /// `hostArguments`.
    func makeRequest(arguments: JSONValue) throws -> URLRequest {
        var args: [String: JSONValue] = { if case .object(let object) = arguments { return object } else { return [:] } }()

        // 1. Fill {token} path/query placeholders, consuming those keys.
        var urlString = url
        for key in Self.templateKeys(in: url) {
            let raw = args[key].map(Self.scalarString) ?? ""
            // Use the correct character set depending on whether the placeholder is in the query
            // string or the path. Path-component encoding forbids '/' (prevents segment injection);
            // query-value encoding forbids '&', '=', '+', '#' (prevents parameter injection).
            let charset: CharacterSet = Self.isInQueryPart(url, key: key)
                ? Self.urlQueryValueAllowed
                : Self.urlPathComponentAllowed
            let encoded = raw.addingPercentEncoding(withAllowedCharacters: charset) ?? raw
            urlString = urlString.replacingOccurrences(of: "{\(key)}", with: encoded)
            args.removeValue(forKey: key)
        }

        guard var components = URLComponents(string: urlString) else { throw HTTPToolError.invalidURL(urlString) }

        // 2. Leftover args → query (query-style verbs) or body (body-carrying verbs).
        if !method.carriesBody, !args.isEmpty {
            var items = components.queryItems ?? []
            for (key, value) in args.sorted(by: { $0.key < $1.key }) {
                items.append(URLQueryItem(name: key, value: Self.scalarString(value)))
            }
            components.queryItems = items
        }

        guard let finalURL = components.url else { throw HTTPToolError.invalidURL(urlString) }

        var request = URLRequest(url: finalURL, timeoutInterval: timeout)
        request.httpMethod = method.rawValue
        for (field, value) in headers { request.setValue(value, forHTTPHeaderField: field) }
        for (field, value) in secrets { request.setValue(value, forHTTPHeaderField: field) }

        if method.carriesBody, body != .empty, !args.isEmpty {
            request.httpBody = try JSONEncoder().encode(JSONValue.object(args))
            if request.value(forHTTPHeaderField: "Content-Type") == nil {
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            }
        }
        return request
    }

    // MARK: - Helpers (file-internal, unit-tested via @testable)

    /// The `{token}` names in a URL template, in order.
    static func templateKeys(in string: String) -> [String] {
        var keys: [String] = []
        var current = ""
        var inToken = false
        for character in string {
            switch character {
            case "{": inToken = true; current = ""
            case "}" where inToken: keys.append(current); inToken = false
            default: if inToken { current.append(character) }
            }
        }
        return keys
    }

    /// A scalar argument as a string for a path/query slot; non-scalars are JSON-encoded compactly.
    static func scalarString(_ value: JSONValue) -> String {
        switch value {
        case .string(let string): return string
        case .int(let int): return String(int)
        case .double(let double): return String(double)
        case .bool(let bool): return bool ? "true" : "false"
        case .null: return ""
        case .array, .object:
            guard let data = try? JSONEncoder().encode(value) else { return "" }
            return String(decoding: data, as: UTF8.self)
        }
    }

    /// Returns true when the `{key}` placeholder appears after the first `?` in the URL template,
    /// meaning its value will land in the query string rather than the path.
    static func isInQueryPart(_ url: String, key: String) -> Bool {
        let placeholder = "{\(key)}"
        guard let placeholderRange = url.range(of: placeholder),
              let questionMark = url.firstIndex(of: "?") else { return false }
        return placeholderRange.lowerBound > questionMark
    }

    /// Character set safe for percent-encoding a query-item value. Starts from `.urlQueryAllowed`
    /// and removes delimiter characters that would otherwise split or corrupt the query string.
    private static let urlQueryValueAllowed: CharacterSet = {
        var allowed = CharacterSet.urlQueryAllowed
        allowed.remove(charactersIn: "&=+#")
        return allowed
    }()

    /// Character set safe for percent-encoding a single path component. Starts from `.urlPathAllowed`
    /// and removes '/' so a value can't inject extra path segments. (Foundation has no built-in
    /// `urlPathComponentAllowed`.)
    private static let urlPathComponentAllowed: CharacterSet = {
        var allowed = CharacterSet.urlPathAllowed
        allowed.remove(charactersIn: "/")
        return allowed
    }()

    /// Merge host arguments into a call's arguments (host values win). Injects all of them, not just
    /// schema-declared keys (unlike MCP) — a host value may feed a `{token}` path slot or query item.
    static func merge(_ hostArguments: [String: JSONValue], into arguments: JSONValue) -> JSONValue {
        guard !hostArguments.isEmpty else { return arguments }
        var object: [String: JSONValue] = { if case .object(let object) = arguments { return object } else { return [:] } }()
        for (key, value) in hostArguments { object[key] = value }
        return .object(object)
    }

    /// Remove `keys` from a JSON-Schema object's `properties` and `required`, so a host-supplied
    /// argument isn't advertised to the model. Same shape as `MCPToolProvider.removingProperties`.
    static func removingProperties(_ keys: Set<String>, from schema: JSONValue) -> JSONValue {
        guard case .object(var root) = schema else { return schema }
        if case .object(var properties) = root["properties"] {
            for key in keys { properties.removeValue(forKey: key) }
            root["properties"] = .object(properties)
        }
        if case .array(let required) = root["required"] {
            root["required"] = .array(required.filter { if case .string(let name) = $0 { return !keys.contains(name) } else { return true } })
        }
        return .object(root)
    }
}

extension Tool {
    /// A tool backed by an HTTP API via an ``HTTPToolSpec`` — no handler code. The spec's
    /// `hostArguments` are hidden from the advertised `inputSchema`.
    public static func http(
        name: String,
        description: String,
        inputSchema: JSONValue = .object(["type": "object"]),
        ui: String? = nil,
        visibility: ToolVisibility = .all,
        outputSchema: JSONValue? = nil,
        spec: HTTPToolSpec
    ) -> Tool {
        let hidden = Set(spec.hostArguments.keys)
        let advertised = hidden.isEmpty ? inputSchema : HTTPToolSpec.removingProperties(hidden, from: inputSchema)
        let definition = AgentTool(
            name: name,
            description: description,
            inputSchema: advertised,
            ui: ui,
            visibility: visibility,
            outputSchema: outputSchema
        )
        return Tool(definition: definition) { arguments in
            let merged = HTTPToolSpec.merge(spec.hostArguments, into: arguments)
            let request = try spec.makeRequest(arguments: merged)
            let response = try await spec.invoker.send(request)
            return try spec.response.map(response)
        }
    }
}
