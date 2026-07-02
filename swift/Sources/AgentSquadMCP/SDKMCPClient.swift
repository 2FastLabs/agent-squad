import Foundation

import AgentSquad
import MCP

/// `MCPClient` backed by the official MCP Swift SDK (`Client` + `HTTPClientTransport`). The SDK is
/// confined to this type — nothing above the `MCPClient` seam sees it.
///
/// HTTP headers (e.g. `Authorization`, `X-Match-Id`) are passed via `headers` and forwarded
/// unchanged on every request. The MCP server is responsible for interpreting them.
public actor SDKMCPClient: MCPClient {
    private let clientName: String
    private let clientVersion: String
    private let endpoint: URL
    private let streaming: Bool
    private let headers: [String: String]
    private let configuration: URLSessionConfiguration

    private var client: Client?   // nil until connect()
    private var connectTask: Task<Void, any Error>?   // in-flight connect, so concurrent callers join it

    /// Default handshake identity, shared with `MCPServer`'s convenience initializers so the two
    /// construction paths never drift apart.
    public static let defaultClientName = "AgentSquad"
    public static let defaultClientVersion = AgentSquadInfo.version

    public init(
        endpoint: URL,
        clientName: String = SDKMCPClient.defaultClientName,
        clientVersion: String = SDKMCPClient.defaultClientVersion,
        streaming: Bool = true,
        headers: [String: String] = [:],
        configuration: URLSessionConfiguration = .default
    ) {
        self.endpoint = endpoint
        self.clientName = clientName
        self.clientVersion = clientVersion
        self.streaming = streaming
        self.headers = headers
        self.configuration = configuration
    }

    public func connect() async throws {
        if client != nil { return }                                   // already connected
        if let connectTask { return try await connectTask.value }     // join an in-flight connect
        let task = Task { try await self.performConnect() }
        connectTask = task
        try await task.value
    }

    private func performConnect() async throws {
        let headers = self.headers
        let transport = HTTPClientTransport(
            endpoint: endpoint,
            configuration: configuration,
            streaming: streaming,
            requestModifier: { request in
                var request = request
                for (field, value) in headers { request.setValue(value, forHTTPHeaderField: field) }
                return request
            }
        )
        let client = Client(name: clientName, version: clientVersion)
        do {
            try await client.connect(transport: transport)
        } catch {
            await client.disconnect()   // tear down a half-established transport before rethrowing
            connectTask = nil           // allow a fresh retry
            throw error
        }
        self.client = client
        connectTask = nil               // clear only after client is set, so late joiners see client != nil
    }

    public func listTools() async throws -> [MCPToolInfo] {
        let client = try requireClient()
        // Paginate to completion. Guard against a cycling server by checking each cursor
        // (incl. A→B→A cycles) before fetching it again.
        var tools: [MCP.Tool] = []
        var cursor: String?
        var seenCursors = Set<String>()
        while true {
            if let cursor {
                guard !seenCursors.contains(cursor) else { break }
                seenCursors.insert(cursor)
            }
            let page = try await client.listTools(cursor: cursor)
            tools += page.tools
            guard let next = page.nextCursor else { break }
            cursor = next
        }

        return tools.map { tool in
            MCPToolInfo(
                name: tool.name,
                description: tool.description ?? "",
                inputSchema: Bridging.toJSON(tool.inputSchema),
                ui: Bridging.uiResourceURI(tool._meta?.fields),
                visibility: Bridging.visibility(tool._meta?.fields)
            )
        }
    }

    public func callTool(name: String, arguments: JSONValue) async throws -> MCPCallResult {
        let client = try requireClient()
        // Use the full-result overload (the tuple overload drops structuredContent/_meta); the
        // explicit `RequestContext` type annotation forces that overload.
        let context: RequestContext<CallTool.Result> = try await client.callTool(
            name: name,
            arguments: Bridging.arguments(arguments)
        )
        let result = try await context.value
        return MCPCallResult(
            content: Bridging.content(result.content),
            structuredContent: result.structuredContent.map(Bridging.toJSON),
            meta: Bridging.metaToJSON(result._meta?.fields),
            isError: result.isError ?? false
        )
    }

    public func readResource(uri: String) async throws -> MCPResourceContents {
        let client = try requireClient()
        let contents = try await client.readResource(uri: uri)
        guard let first = contents.first else { throw MCPClientError.emptyResource(uri) }
        return MCPResourceContents(
            mimeType: first.mimeType ?? "application/octet-stream",
            text: first.text,
            blob: first.blob.flatMap { Data(base64Encoded: $0) },
            meta: nil   // resource _meta → UISecurity is the UI host's concern, not parsed here
        )
    }

    public func disconnect() async {
        await client?.disconnect()
        client = nil   // idempotent: a second disconnect is a no-op
    }

    // MARK: - Private

    private func requireClient() throws -> Client {
        guard let client else { throw MCPClientError.notConnected }
        return client
    }
}

/// Failures from `SDKMCPClient` that are infrastructure-level (transport/protocol/lifecycle) and so
/// are thrown, not returned as a tool-level `ToolResult(isError:)`.
public enum MCPClientError: Error, Sendable {
    case notConnected
    case emptyResource(String)
}
