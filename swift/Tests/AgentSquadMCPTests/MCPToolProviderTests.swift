import Foundation
import Testing

@testable import AgentSquad
@testable import AgentSquadMCP

@Suite struct MCPToolProviderTests {
    // Actor mock so we can count readResource calls (to assert template caching).
    private struct TransportError: Error {}

    private actor MockMCPClient: MCPClient {
        let tools: [MCPToolInfo]
        let callResults: [String: MCPCallResult]
        let resources: [String: MCPResourceContents]
        let throwOnCall: Bool
        private(set) var readResourceCalls = 0
        private(set) var connectCalls = 0
        private(set) var argsByTool: [String: JSONValue] = [:]   // last arguments seen per tool

        init(
            tools: [MCPToolInfo] = [],
            callResults: [String: MCPCallResult] = [:],
            resources: [String: MCPResourceContents] = [:],
            throwOnCall: Bool = false
        ) {
            self.tools = tools
            self.callResults = callResults
            self.resources = resources
            self.throwOnCall = throwOnCall
        }

        func connect() async throws { connectCalls += 1 }
        func listTools() async throws -> [MCPToolInfo] { tools }
        func callTool(name: String, arguments: JSONValue) async throws -> MCPCallResult {
            argsByTool[name] = arguments
            if throwOnCall { throw TransportError() }   // simulates a transport/protocol failure
            return callResults[name] ?? MCPCallResult()
        }
        func readResource(uri: String) async throws -> MCPResourceContents {
            readResourceCalls += 1
            return resources[uri] ?? MCPResourceContents(mimeType: "text/plain")
        }
        func disconnect() async {}
    }

    @Test func mcpServerConvenienceInitAndAlias() async throws {
        // The `url:` convenience builds an `SDKMCPClient` under the hood and constructs without
        // connecting — the one-liner the consumer wants instead of nesting `SDKMCPClient`. The string
        // form parses the URL internally.
        let viaString: any ToolProvider = MCPServer(url: "https://example.test/mcp",
                                                    hostArguments: ["session_id": .string("S1")])
        _ = viaString   // smoke: compiles + constructs the default-SDK provider from a string URL

        // `MCPServer` is `MCPToolProvider` — the custom-client seam works identically through the alias.
        let mock = MockMCPClient(tools: [MCPToolInfo(name: "matches", description: "", inputSchema: .object([:]))])
        let tools = try await MCPServer(client: mock).listTools()
        #expect(tools.map(\.name) == ["matches"])
    }

    @Test func mcpServerAcceptsCustomHeaders() {
        // Compile-time + construction smoke: both URL overloads accept headers and forward them.
        let viaString: any ToolProvider = MCPServer(
            url: "https://example.test/mcp",
            headers: ["Authorization": "Bearer tok", "X-Match-Id": "42"]
        )
        _ = viaString

        let viaURL: any ToolProvider = MCPServer(
            url: URL(string: "https://example.test/mcp")!,
            headers: ["Authorization": "Bearer tok", "X-Match-Id": "42"]
        )
        _ = viaURL
    }

    @Test func sdkMCPClientForwardsHeadersOnWire() async throws {
        // Verify that headers passed to SDKMCPClient actually reach the outgoing URLRequest.
        // Uses a URLProtocol stub to intercept the first POST without a real server.
        final class CapturingProtocol: URLProtocol, @unchecked Sendable {
            // Written from URLSession's queue, polled from the test — lock-guarded.
            private static let lock = NSLock()
            private nonisolated(unsafe) static var _capturedRequest: URLRequest?
            static var capturedRequest: URLRequest? {
                get { lock.withLock { _capturedRequest } }
                set { lock.withLock { _capturedRequest = newValue } }
            }
            override class func canInit(with request: URLRequest) -> Bool { true }
            override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
            override func startLoading() {
                CapturingProtocol.capturedRequest = request
                // Return a minimal HTTP 200 to satisfy URLSession; connect() will still fail the MCP
                // handshake, which is fine — we only need the outgoing request to be intercepted.
                let response = HTTPURLResponse(
                    url: request.url!,
                    statusCode: 200,
                    httpVersion: nil,
                    headerFields: ["Content-Type": "application/json"]
                )!
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: Data("{}".utf8))
                client?.urlProtocolDidFinishLoading(self)
            }
            override func stopLoading() {}
        }

        CapturingProtocol.capturedRequest = nil   // reset before use; static state persists across runs

        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [CapturingProtocol.self]

        let client = SDKMCPClient(
            endpoint: URL(string: "https://example.test/mcp")!,
            headers: ["X-Match-Id": "42", "Authorization": "Bearer tok"],
            configuration: config
        )

        // Don't await connect(): against the stub the SDK's handshake may never settle (no timeout),
        // and we only need the first POST intercepted. Poll for the capture, then cancel.
        let connect = Task { try? await client.connect() }
        defer { connect.cancel() }
        for _ in 0..<100 where CapturingProtocol.capturedRequest == nil {
            try await Task.sleep(for: .milliseconds(50))
        }

        let captured = try #require(CapturingProtocol.capturedRequest, "URLProtocol stub never intercepted a request")
        #expect(captured.value(forHTTPHeaderField: "X-Match-Id") == "42")
        #expect(captured.value(forHTTPHeaderField: "Authorization") == "Bearer tok")
    }

    @Test func listToolsMapsAndFiltersToModelVisible() async throws {
        let mock = MockMCPClient(tools: [
            MCPToolInfo(name: "get_odds", description: "1X2 odds", inputSchema: .object([:]), ui: "ui://sport/matches"),
            MCPToolInfo(name: "refresh", description: "", inputSchema: .object([:]), visibility: .app),  // app-only
        ])
        let tools = try await MCPToolProvider(client: mock).listTools()
        #expect(tools.map(\.name) == ["get_odds"])           // app-only tool hidden from the model
        #expect(tools.first?.ui == "ui://sport/matches")
    }

    @Test func callWithoutUIReturnsResultAndNoWidget() async throws {
        let mock = MockMCPClient(
            tools: [MCPToolInfo(name: "plain", description: "", inputSchema: .object([:]))],
            callResults: ["plain": MCPCallResult(content: [.text("done")], structuredContent: ["n": 1])]
        )
        let provider = MCPToolProvider(client: mock)
        _ = try await provider.listTools()
        let result = try await provider.call("plain", arguments: .object([:]))
        #expect(result.ui == nil)
        #expect(result.content == [.text("done")])
        #expect(result.structuredContent == ["n": 1])
    }

    @Test func callWithUILazilyFetchesAndAssemblesPayload() async throws {
        let mock = MockMCPClient(
            tools: [MCPToolInfo(name: "matches", description: "", inputSchema: .object([:]), ui: "ui://sport/matches")],
            callResults: ["matches": MCPCallResult(structuredContent: ["count": 3])],
            resources: ["ui://sport/matches": MCPResourceContents(mimeType: "text/html;profile=mcp-app", text: "<div>x</div>")]
        )
        let provider = MCPToolProvider(client: mock)
        _ = try await provider.listTools()

        let result = try await provider.call("matches", arguments: .object([:]))
        #expect(result.ui?.resourceURI == "ui://sport/matches")
        #expect(result.ui?.template == .html("<div>x</div>"))
        #expect(result.ui?.structuredContent == ["count": 3])   // data flows to the UI payload

        // A second call reuses the cached template (resource read once).
        _ = try await provider.call("matches", arguments: .object([:]))
        #expect(await mock.readResourceCalls == 1)
    }

    @Test func toolLevelErrorSurfacesAsErrorResult() async throws {
        let mock = MockMCPClient(
            tools: [MCPToolInfo(name: "boom", description: "", inputSchema: .object([:]))],
            callResults: ["boom": MCPCallResult(content: [.text("nope")], isError: true)]
        )
        let provider = MCPToolProvider(client: mock)
        _ = try await provider.listTools()
        let result = try await provider.call("boom", arguments: .object([:]))
        #expect(result.isError)
    }

    // call before listTools must still resolve the tool's UI (lazy listing).
    @Test func callBeforeListToolsStillAssemblesUI() async throws {
        let mock = MockMCPClient(
            tools: [MCPToolInfo(name: "matches", description: "", inputSchema: .object([:]), ui: "ui://m")],
            callResults: ["matches": MCPCallResult(structuredContent: ["count": 1])],
            resources: ["ui://m": MCPResourceContents(mimeType: "text/html;profile=mcp-app", text: "<div/>")]
        )
        let result = try await MCPToolProvider(client: mock).call("matches", arguments: .object([:]))
        #expect(result.ui?.resourceURI == "ui://m")
    }

    // Unknown tool name is a tool-level failure — an isError result, not a throw.
    @Test func unknownToolReturnsErrorResultNotThrow() async throws {
        let mock = MockMCPClient(tools: [MCPToolInfo(name: "known", description: "", inputSchema: .object([:]))])
        let result = try await MCPToolProvider(client: mock).call("missing", arguments: .object([:]))
        #expect(result.isError)
    }

    // A genuine transport failure propagates as a throw (not a tool-level error result).
    @Test func transportFailureThrows() async throws {
        let mock = MockMCPClient(
            tools: [MCPToolInfo(name: "t", description: "", inputSchema: .object([:]))],
            throwOnCall: true
        )
        let provider = MCPToolProvider(client: mock)
        await #expect(throws: TransportError.self) {
            try await provider.call("t", arguments: .object([:]))
        }
    }

    // App-only tools are hidden from listTools but remain callable (for UI-initiated calls).
    @Test func appOnlyToolIsCallableAndAssemblesUI() async throws {
        let mock = MockMCPClient(
            tools: [MCPToolInfo(name: "refresh", description: "", inputSchema: .object([:]), ui: "ui://r", visibility: .app)],
            callResults: ["refresh": MCPCallResult(structuredContent: .object([:]))],
            resources: ["ui://r": MCPResourceContents(mimeType: "text/html;profile=mcp-app", text: "<r/>")]
        )
        let provider = MCPToolProvider(client: mock)
        #expect(try await provider.listTools().isEmpty)              // hidden from the model
        let result = try await provider.call("refresh", arguments: .object([:]))
        #expect(result.ui?.template == .html("<r/>"))                // still callable for the UI
    }

    // The provider connects lazily and latches it: repeated operations reuse one handshake.
    @Test func connectsOnceThenReusesConnection() async throws {
        let mock = MockMCPClient(
            tools: [MCPToolInfo(name: "t", description: "", inputSchema: .object([:]))],
            callResults: ["t": MCPCallResult()]
        )
        let provider = MCPToolProvider(client: mock)
        _ = try await provider.listTools()
        _ = try await provider.call("t", arguments: .object([:]))
        _ = try await provider.listTools()
        #expect(await mock.connectCalls == 1)
    }

    // MARK: - Host arguments

    private func sessionSchema() -> JSONValue {
        .object([
            "type": .string("object"),
            "properties": .object([
                "session_id": .object(["type": .string("string")]),
                "q": .object(["type": .string("string")]),
            ]),
            "required": .array([.string("session_id"), .string("q")]),
            "additionalProperties": .bool(false),
        ])
    }

    // Host arguments are merged into a call only for tools whose schema declares the key.
    @Test func injectsHostArgumentsOnlyIntoDeclaringTools() async throws {
        let withoutSession: JSONValue = .object([
            "type": .string("object"),
            "properties": .object(["q": .object(["type": .string("string")])]),
        ])
        let mock = MockMCPClient(
            tools: [
                MCPToolInfo(name: "needs_session", description: "", inputSchema: sessionSchema()),
                MCPToolInfo(name: "no_session", description: "", inputSchema: withoutSession),
            ],
            callResults: ["needs_session": MCPCallResult(), "no_session": MCPCallResult()]
        )
        let provider = MCPToolProvider(client: mock, hostArguments: ["session_id": .string("S1")])
        _ = try await provider.listTools()

        _ = try await provider.call("needs_session", arguments: .object(["q": .string("hi")]))
        #expect(await mock.argsByTool["needs_session"] == .object(["q": .string("hi"), "session_id": .string("S1")]))

        _ = try await provider.call("no_session", arguments: .object(["q": .string("hi")]))
        #expect(await mock.argsByTool["no_session"] == .object(["q": .string("hi")]))   // undeclared → not injected
    }

    // A host argument is removed from the schema the model sees — from both `properties` and `required` —
    // while the rest of the schema (incl. `additionalProperties: false`) is left intact.
    @Test func hidesHostArgumentsFromAdvertisedSchema() async throws {
        let mock = MockMCPClient(tools: [MCPToolInfo(name: "t", description: "", inputSchema: sessionSchema())])
        let tools = try await MCPToolProvider(client: mock, hostArguments: ["session_id": .string("S1")]).listTools()
        #expect(tools.first?.inputSchema == .object([
            "type": .string("object"),
            "properties": .object(["q": .object(["type": .string("string")])]),
            "required": .array([.string("q")]),
            "additionalProperties": .bool(false),
        ]))
    }

    // Defensive: if the model somehow sends a host-owned key, the host value wins.
    @Test func hostArgumentOverridesModelSuppliedValue() async throws {
        let mock = MockMCPClient(
            tools: [MCPToolInfo(name: "needs_session", description: "", inputSchema: sessionSchema())],
            callResults: ["needs_session": MCPCallResult()]
        )
        let provider = MCPToolProvider(client: mock, hostArguments: ["session_id": .string("right")])
        _ = try await provider.listTools()
        _ = try await provider.call("needs_session", arguments: .object(["session_id": .string("wrong"), "q": .string("hi")]))
        #expect(await mock.argsByTool["needs_session"] == .object(["session_id": .string("right"), "q": .string("hi")]))
    }

    // When the host key is the only required field, `required` collapses to [] (still valid Schema).
    @Test func hidingTheOnlyRequiredFieldLeavesEmptyRequired() async throws {
        let schema: JSONValue = .object([
            "type": .string("object"),
            "properties": .object(["session_id": .object(["type": .string("string")])]),
            "required": .array([.string("session_id")]),
            "additionalProperties": .bool(false),
        ])
        let mock = MockMCPClient(tools: [MCPToolInfo(name: "t", description: "", inputSchema: schema)])
        let tools = try await MCPToolProvider(client: mock, hostArguments: ["session_id": .string("S1")]).listTools()
        #expect(tools.first?.inputSchema == .object([
            "type": .string("object"),
            "properties": .object([:]),
            "required": .array([]),
            "additionalProperties": .bool(false),
        ]))
    }

    // A call whose arguments aren't an object still gets the declared host key (coerced to an object).
    @Test func injectsHostArgumentsIntoNonObjectArguments() async throws {
        let mock = MockMCPClient(
            tools: [MCPToolInfo(name: "needs_session", description: "", inputSchema: sessionSchema())],
            callResults: ["needs_session": MCPCallResult()]
        )
        let provider = MCPToolProvider(client: mock, hostArguments: ["session_id": .string("S1")])
        _ = try await provider.listTools()
        _ = try await provider.call("needs_session", arguments: .null)
        #expect(await mock.argsByTool["needs_session"] == .object(["session_id": .string("S1")]))
    }

    // With no host arguments configured, schema and arguments pass through untouched.
    @Test func noHostArgumentsLeavesSchemaAndArgsUntouched() async throws {
        let mock = MockMCPClient(
            tools: [MCPToolInfo(name: "needs_session", description: "", inputSchema: sessionSchema())],
            callResults: ["needs_session": MCPCallResult()]
        )
        let provider = MCPToolProvider(client: mock)
        let tools = try await provider.listTools()
        #expect(tools.first?.inputSchema == sessionSchema())   // unchanged
        _ = try await provider.call("needs_session", arguments: .object(["q": .string("hi")]))
        #expect(await mock.argsByTool["needs_session"] == .object(["q": .string("hi")]))   // nothing injected
    }

    @Test func mapsUriListAndBlobTemplates() async throws {
        let mock = MockMCPClient(
            tools: [
                MCPToolInfo(name: "link", description: "", inputSchema: .object([:]), ui: "ui://link"),
                MCPToolInfo(name: "blobbed", description: "", inputSchema: .object([:]), ui: "ui://blob"),
            ],
            callResults: ["link": MCPCallResult(), "blobbed": MCPCallResult()],
            resources: [
                "ui://link": MCPResourceContents(mimeType: "text/uri-list", text: "https://example.com/widget"),
                "ui://blob": MCPResourceContents(mimeType: "text/html;profile=mcp-app", blob: Data("<b/>".utf8)),
            ]
        )
        let provider = MCPToolProvider(client: mock)
        _ = try await provider.listTools()
        #expect(try await provider.call("link", arguments: .object([:])).ui?.template == .url(URL(string: "https://example.com/widget")!))
        #expect(try await provider.call("blobbed", arguments: .object([:])).ui?.template == .html("<b/>"))
    }
}
