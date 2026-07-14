/**
 * Unit tests for MCPToolProvider.
 *
 * All MCP SDK imports are mocked so no real server is required.
 */

// ---------------------------------------------------------------------------
// Mock @modelcontextprotocol/sdk before any imports that might pull it in
// ---------------------------------------------------------------------------

const mockCallTool = jest.fn();
const mockListTools = jest.fn();
const mockConnect = jest.fn();
const mockClose = jest.fn();
const mockReadResource = jest.fn();

class MockClient {
  connect = mockConnect;
  listTools = mockListTools;
  callTool = mockCallTool;
  close = mockClose;
  readResource = mockReadResource;
}

class MockStdioTransport {
  constructor(public opts: any) {}
}

class MockSSETransport {
  constructor(public url: any, public opts: any) {}
}

jest.mock("@modelcontextprotocol/sdk/client/index.js", () => ({ Client: MockClient }), { virtual: true });
jest.mock("@modelcontextprotocol/sdk/client/stdio.js", () => ({ StdioClientTransport: MockStdioTransport }), { virtual: true });
jest.mock("@modelcontextprotocol/sdk/client/sse.js", () => ({ SSEClientTransport: MockSSETransport }), { virtual: true });

// ---------------------------------------------------------------------------
// Subject under test
// ---------------------------------------------------------------------------

import { MCPToolProvider, MCPServerConfig } from "../src/tools/mcpToolProvider";
import { AgentTools, AgentToolResult, ToolResult } from "../src/utils/tool";
import { GroundedAgent } from "../src/agents/groundedAgent";
import { Agent, AgentOptions } from "../src/agents/agent";
import { ConversationMessage, ParticipantRole } from "../src/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const weatherTool = {
  name: "get_weather",
  description: "Returns weather for a location",
  inputSchema: {
    type: "object",
    properties: {
      location: { type: "string", description: "City name" },
    },
    required: ["location"],
  },
};

const searchTool = {
  name: "search_web",
  description: "Search the web",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string" },
    },
    required: ["query"],
  },
};

function makeBedrockResponse(toolName: string, toolUseId: string, input: any) {
  return {
    role: "assistant",
    content: [
      {
        toolUse: { name: toolName, toolUseId, input },
      },
    ],
  };
}

// Bedrock extractor callbacks
const bedrockGetToolUseBlock = (block: any) => block.toolUse ?? null;
const bedrockGetToolName = (b: any) => b.name;
const bedrockGetToolId = (b: any) => b.toolUseId;
const bedrockGetInputData = (b: any) => b.input;

// A gatherer that drives an MCP tool the way a real agent does — through the provider's
// toolHandler (the old code bypassed each tool's func here, so GroundedAgent captured nothing).
class McpFakeGatherer extends Agent {
  constructor(options: AgentOptions, private readonly provider: MCPToolProvider) {
    super(options);
  }
  async processRequest(): Promise<ConversationMessage> {
    await this.provider.toolHandler(
      makeBedrockResponse("get_order", "1", { orderId: "42" }),
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );
    return { role: ParticipantRole.ASSISTANT, content: [{ text: "" }] };
  }
}

class McpFakePresenter extends Agent {
  receivedInput?: string;
  setSystemPrompt(_template?: string): void {}
  async processRequest(input: string): Promise<ConversationMessage> {
    this.receivedInput = input;
    return { role: ParticipantRole.ASSISTANT, content: [{ text: "presented" }] };
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("MCPToolProvider", () => {
  let provider: MCPToolProvider;

  beforeEach(() => {
    jest.clearAllMocks();

    mockConnect.mockResolvedValue(undefined);
    mockClose.mockResolvedValue(undefined);
    mockListTools.mockResolvedValue({ tools: [weatherTool] });
    mockCallTool.mockResolvedValue({
      isError: false,
      content: [{ type: "text", text: "Sunny, 25°C" }],
    });

    provider = new MCPToolProvider([
      { type: "stdio", command: "uvx", args: ["weather-server"] },
    ]);
  });

  // -------------------------------------------------------------------------
  // Inheritance
  // -------------------------------------------------------------------------

  it("should be an instance of AgentTools", () => {
    expect(provider).toBeInstanceOf(AgentTools);
  });

  // -------------------------------------------------------------------------
  // Lazy connection
  // -------------------------------------------------------------------------

  it("should not connect until ensureConnected is called", async () => {
    expect(mockConnect).not.toHaveBeenCalled();
    await provider.ensureConnected();
    expect(mockConnect).toHaveBeenCalledTimes(1);
  });

  it("should connect only once even with multiple ensureConnected calls", async () => {
    await provider.ensureConnected();
    await provider.ensureConnected();
    await provider.ensureConnected();
    expect(mockConnect).toHaveBeenCalledTimes(1);
  });

  it("should populate this.tools after ensureConnected", async () => {
    expect(provider.tools).toHaveLength(0);
    await provider.ensureConnected();
    expect(provider.tools).toHaveLength(1);
    expect(provider.tools[0].name).toBe("get_weather");
  });

  // -------------------------------------------------------------------------
  // toBedrockFormat
  // -------------------------------------------------------------------------

  it("toBedrockFormat returns tools in Bedrock toolSpec format", async () => {
    const formatted = await provider.toBedrockFormat();
    expect(formatted).toHaveLength(1);
    const spec = formatted[0].toolSpec;
    expect(spec.name).toBe("get_weather");
    expect(spec.description).toBe("Returns weather for a location");
    expect(spec.inputSchema.json).toEqual(weatherTool.inputSchema);
  });

  it("toBedrockFormat passes MCP inputSchema through unchanged", async () => {
    await provider.ensureConnected();
    const formatted = await provider.toBedrockFormat();
    // The JSON Schema from MCP should be passed through directly
    expect(formatted[0].toolSpec.inputSchema.json.properties.location).toEqual(
      weatherTool.inputSchema.properties.location
    );
  });

  // -------------------------------------------------------------------------
  // toAnthropicFormat
  // -------------------------------------------------------------------------

  it("toAnthropicFormat returns tools in Anthropic format", async () => {
    const formatted = await provider.toAnthropicFormat();
    expect(formatted).toHaveLength(1);
    expect(formatted[0].name).toBe("get_weather");
    expect(formatted[0].input_schema).toEqual(weatherTool.inputSchema);
  });

  // -------------------------------------------------------------------------
  // toOpenAIFormat
  // -------------------------------------------------------------------------

  it("toOpenAIFormat returns tools in OpenAI function-calling format", async () => {
    const formatted = await provider.toOpenAIFormat();
    expect(formatted).toHaveLength(1);
    expect(formatted[0].type).toBe("function");
    expect(formatted[0].function.name).toBe("get_weather");
    expect(formatted[0].function.parameters).toEqual(weatherTool.inputSchema);
  });

  // -------------------------------------------------------------------------
  // toolHandler — execution routing
  // -------------------------------------------------------------------------

  it("toolHandler calls the correct MCP tool and returns results", async () => {
    const response = makeBedrockResponse("get_weather", "call-001", {
      location: "Paris",
    });

    const results = await provider.toolHandler(
      response,
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );

    expect(mockCallTool).toHaveBeenCalledWith({
      name: "get_weather",
      arguments: { location: "Paris" },
    });

    expect(results).toHaveLength(1);
    expect(results[0]).toBeInstanceOf(AgentToolResult);
    expect(results[0].toolUseId).toBe("call-001");
    expect(results[0].content).toBe("Sunny, 25°C");
  });

  it("toolHandler triggers lazy connection if not yet connected", async () => {
    const response = makeBedrockResponse("get_weather", "call-002", {
      location: "London",
    });

    expect(mockConnect).not.toHaveBeenCalled();
    await provider.toolHandler(
      response,
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );
    expect(mockConnect).toHaveBeenCalledTimes(1);
  });

  it("toolHandler returns empty array when content has no tool-use blocks", async () => {
    const response = {
      role: "assistant",
      content: [{ text: "Hello!" }],
    };

    const results = await provider.toolHandler(
      response,
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );

    expect(results).toHaveLength(0);
  });

  it("toolHandler throws when response has no content", async () => {
    await expect(
      provider.toolHandler(
        {},
        bedrockGetToolUseBlock,
        bedrockGetToolName,
        bedrockGetToolId,
        bedrockGetInputData
      )
    ).rejects.toThrow("No content blocks in response");
  });

  // -------------------------------------------------------------------------
  // Error handling — isError from MCP
  // -------------------------------------------------------------------------

  it("toolHandler surfaces isError from MCP result as error string", async () => {
    mockCallTool.mockResolvedValueOnce({
      isError: true,
      content: [{ type: "text", text: "Service unavailable" }],
    });

    const response = makeBedrockResponse("get_weather", "call-err", {
      location: "Berlin",
    });

    const results = await provider.toolHandler(
      response,
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );

    expect(results[0].content).toContain("Error from MCP tool 'get_weather'");
    expect(results[0].content).toContain("Service unavailable");
  });

  it("toolHandler handles exception thrown by callTool", async () => {
    mockCallTool.mockRejectedValueOnce(new Error("Connection lost"));

    const response = makeBedrockResponse("get_weather", "call-exc", {
      location: "Tokyo",
    });

    const results = await provider.toolHandler(
      response,
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );

    expect(results[0].content).toContain("Error calling MCP tool");
    expect(results[0].content).toContain("Connection lost");
  });

  it("toolHandler returns not-found message for unknown tool name", async () => {
    await provider.ensureConnected();

    const response = makeBedrockResponse("nonexistent_tool", "call-404", {});

    const results = await provider.toolHandler(
      response,
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );

    expect(results[0].content).toContain("not found");
    expect(mockCallTool).not.toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // Multiple servers — tool routing
  // -------------------------------------------------------------------------

  it("routes tools to the correct server when multiple servers are configured", async () => {
    // Arrange: first listTools call returns weatherTool, second returns searchTool
    mockListTools
      .mockResolvedValueOnce({ tools: [weatherTool] })
      .mockResolvedValueOnce({ tools: [searchTool] });

    // callTool returns different results based on tool name
    mockCallTool.mockImplementation((args: any) => {
      if (args.name === "get_weather") {
        return Promise.resolve({ isError: false, content: [{ type: "text", text: "Sunny" }] });
      }
      return Promise.resolve({ isError: false, content: [{ type: "text", text: "Search result" }] });
    });

    const multiProvider = new MCPToolProvider([
      { type: "stdio", command: "uvx", args: ["weather-server"] },
      { type: "sse", url: "http://localhost:3000/sse" },
    ]);

    await multiProvider.ensureConnected();
    // Both servers contributed their tools
    expect(multiProvider.tools).toHaveLength(2);
    expect(multiProvider.tools.map((t) => t.name)).toContain("get_weather");
    expect(multiProvider.tools.map((t) => t.name)).toContain("search_web");

    // Call weather tool
    const weatherResponse = makeBedrockResponse("get_weather", "w-1", { location: "NYC" });
    const weatherResults = await multiProvider.toolHandler(
      weatherResponse,
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );
    expect(weatherResults[0].content).toBe("Sunny");

    // Call search tool
    const searchResponse = makeBedrockResponse("search_web", "s-1", { query: "MCP" });
    const searchResults = await multiProvider.toolHandler(
      searchResponse,
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );
    expect(searchResults[0].content).toBe("Search result");
  });

  // -------------------------------------------------------------------------
  // disconnect
  // -------------------------------------------------------------------------

  it("disconnect calls close on all clients and resets state", async () => {
    await provider.ensureConnected();
    expect(provider.tools).toHaveLength(1);

    await provider.disconnect();

    expect(mockClose).toHaveBeenCalledTimes(1);
    expect(provider.tools).toHaveLength(0);
  });

  // -------------------------------------------------------------------------
  // SSE transport
  // -------------------------------------------------------------------------

  it("creates SSEClientTransport for sse server config", async () => {
    const sseProvider = new MCPToolProvider([
      { type: "sse", url: "http://localhost:9000/sse", headers: { "x-api-key": "abc" } },
    ]);

    await sseProvider.ensureConnected();
    // If we got here without error, SSE transport was constructed correctly
    expect(mockConnect).toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // Config validation
  // -------------------------------------------------------------------------

  it("throws when stdio server config has no command", async () => {
    const badProvider = new MCPToolProvider([
      { type: "stdio" } as MCPServerConfig,
    ]);

    await expect(badProvider.ensureConnected()).rejects.toThrow(
      "requires a 'command' field"
    );
  });

  it("throws when sse server config has no url", async () => {
    const badProvider = new MCPToolProvider([
      { type: "sse" } as MCPServerConfig,
    ]);

    await expect(badProvider.ensureConnected()).rejects.toThrow(
      "requires a 'url' field"
    );
  });

  // -------------------------------------------------------------------------
  // Tool UI (widgets)
  // -------------------------------------------------------------------------

  const uiTool = (extraMeta: any = { ui: { resourceUri: "ui://shop/order-card" } }) => ({
    name: "get_order",
    description: "Order status",
    inputSchema: { type: "object", properties: {}, required: [] },
    _meta: extraMeta,
  });

  async function connectedProvider(): Promise<MCPToolProvider> {
    const p = new MCPToolProvider([{ type: "stdio", command: "x" }]);
    await p.ensureConnected();
    return p;
  }

  it("returns a ToolResult with a UIPayload for a tool advertising _meta.ui", async () => {
    mockListTools.mockResolvedValue({ tools: [uiTool()] });
    mockCallTool.mockResolvedValue({
      isError: false,
      content: [{ type: "text", text: "Order 42: shipped" }],
      structuredContent: { status: "shipped" },
      _meta: { a: 1 },
    });
    mockReadResource.mockResolvedValue({
      contents: [{ uri: "ui://shop/order-card", mimeType: "text/html;profile=mcp-app", text: "<div>card</div>" }],
    });

    const p = await connectedProvider();
    const result: any = await p.tools.find((t) => t.name === "get_order")!.func({ orderId: "42" });

    expect(result).toBeInstanceOf(ToolResult);
    expect(result.content).toBe("Order 42: shipped"); // only text goes to the model
    expect(result.structuredContent).toEqual({ status: "shipped" });
    expect(result.ui.resourceUri).toBe("ui://shop/order-card");
    expect(result.ui.mimeType).toBe("text/html;profile=mcp-app");
    expect(result.ui.template).toBe("<div>card</div>");
    expect(result.ui.structuredContent).toEqual({ status: "shipped" });
    expect(result.ui.meta).toEqual({ a: 1 }); // _meta forwarded to the UI, never the model
  });

  it("falls back to the mcp-app mime type when the resource omits mimeType", async () => {
    mockListTools.mockResolvedValue({ tools: [uiTool()] });
    mockCallTool.mockResolvedValue({ isError: false, content: [{ type: "text", text: "ok" }], structuredContent: {} });
    mockReadResource.mockResolvedValue({ contents: [{ text: "<div/>" }] }); // no mimeType

    const p = await connectedProvider();
    const result: any = await p.tools.find((t) => t.name === "get_order")!.func({});
    expect(result.ui.mimeType).toBe("text/html;profile=mcp-app");
  });

  it("forwards the full input even when it contains a 'messages' key", async () => {
    mockListTools.mockResolvedValue({
      tools: [{ name: "chat", inputSchema: { type: "object", properties: { messages: { type: "array" } } } }],
    });
    mockCallTool.mockResolvedValue({ isError: false, content: [{ type: "text", text: "ok" }] });

    const p = await connectedProvider();
    const resp = makeBedrockResponse("chat", "1", { messages: [{ role: "user" }], extra: 1 });
    await p.toolHandler(resp, bedrockGetToolUseBlock, bedrockGetToolName, bedrockGetToolId, bedrockGetInputData);

    // The whole object is sent to the server — not just inputData.messages.
    expect(mockCallTool).toHaveBeenCalledWith({
      name: "chat",
      arguments: { messages: [{ role: "user" }], extra: 1 },
    });
  });

  it("handles undefined tool input without throwing", async () => {
    mockListTools.mockResolvedValue({
      tools: [{ name: "ping", inputSchema: { type: "object", properties: {} } }],
    });
    mockCallTool.mockResolvedValue({ isError: false, content: [{ type: "text", text: "pong" }] });

    const p = await connectedProvider();
    const resp = makeBedrockResponse("ping", "1", undefined);
    const results = await p.toolHandler(
      resp,
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );
    expect(results[0].content).toBe("pong");
    expect(mockCallTool).toHaveBeenCalledWith({ name: "ping", arguments: {} });
  });

  it("reads the openai/outputTemplate alias", async () => {
    mockListTools.mockResolvedValue({ tools: [uiTool({ "openai/outputTemplate": "ui://alias" })] });
    mockCallTool.mockResolvedValue({ isError: false, content: [{ type: "text", text: "ok" }], structuredContent: {} });
    mockReadResource.mockResolvedValue({ contents: [{ mimeType: "text/html", text: "<b>x</b>" }] });

    const p = await connectedProvider();
    const result: any = await p.tools.find((t) => t.name === "get_order")!.func({});
    expect(result.ui.resourceUri).toBe("ui://alias");
  });

  it("decodes a base64 blob template", async () => {
    mockListTools.mockResolvedValue({ tools: [uiTool()] });
    mockCallTool.mockResolvedValue({ isError: false, content: [{ type: "text", text: "ok" }], structuredContent: {} });
    const blob = Buffer.from("<div>from blob</div>").toString("base64");
    mockReadResource.mockResolvedValue({ contents: [{ mimeType: "text/html", blob }] });

    const p = await connectedProvider();
    const result: any = await p.tools.find((t) => t.name === "get_order")!.func({});
    expect(result.ui.template).toBe("<div>from blob</div>");
  });

  it("degrades to text when the UI resource fetch fails", async () => {
    mockListTools.mockResolvedValue({ tools: [uiTool()] });
    mockCallTool.mockResolvedValue({
      isError: false,
      content: [{ type: "text", text: "Order: shipped" }],
      structuredContent: { s: 1 },
    });
    mockReadResource.mockRejectedValue(new Error("read failed"));

    const p = await connectedProvider();
    const result: any = await p.tools.find((t) => t.name === "get_order")!.func({});
    expect(result.ui).toBeUndefined();
    expect(result.content).toBe("Order: shipped");
    expect(result.structuredContent).toEqual({ s: 1 });
  });

  it("fetches the UI template once (cached per client)", async () => {
    mockListTools.mockResolvedValue({ tools: [uiTool()] });
    mockCallTool.mockResolvedValue({ isError: false, content: [{ type: "text", text: "ok" }], structuredContent: {} });
    mockReadResource.mockResolvedValue({ contents: [{ mimeType: "text/html", text: "<div/>" }] });

    const p = await connectedProvider();
    const tool = p.tools.find((t) => t.name === "get_order")!;
    await tool.func({});
    await tool.func({});
    expect(mockReadResource).toHaveBeenCalledTimes(1);
  });

  it("GroundedAgent captures an MCP tool result (regression: MCP grounding was silently skipped)", async () => {
    mockListTools.mockResolvedValue({ tools: [uiTool()] });
    mockCallTool.mockResolvedValue({
      isError: false,
      content: [{ type: "text", text: "Order 42: shipped" }],
      structuredContent: { status: "shipped" },
    });
    mockReadResource.mockResolvedValue({ contents: [{ mimeType: "text/html", text: "<div/>" }] });

    const provider = await connectedProvider();
    const gatherer = new McpFakeGatherer({ name: "g", description: "" }, provider);
    const presenter = new McpFakePresenter({ name: "p", description: "" });
    const agent = new GroundedAgent({ name: "Shop", description: "", gatherer, presenter, tools: provider });

    const result: any = await agent.processRequest("where is my order?", "u", "s", []);

    // The presenter ran on the captured MCP facts. Before the fix the tool call was never captured,
    // so GroundedAgent saw no tools, skipped the presenter, and returned the gatherer's empty draft.
    expect(presenter.receivedInput).toContain("get_order");
    expect(presenter.receivedInput).toContain("shipped");
    expect(result.content[0].text).toBe("presented");
  });

  it("hides app-only tools from the model but keeps them connected", async () => {
    const appOnly = {
      name: "refresh_order",
      inputSchema: { type: "object", properties: {} },
      _meta: { ui: { visibility: ["app"] } },
    };
    mockListTools.mockResolvedValue({ tools: [uiTool(), appOnly] });

    const p = await connectedProvider();
    // Only the model-visible tool is advertised...
    expect(p.tools.map((t) => t.name)).toEqual(["get_order"]);
    const bedrock = await p.toBedrockFormat();
    expect(bedrock.map((r: any) => r.toolSpec.name)).toEqual(["get_order"]);
  });

  it("rejects a model call to an app-only tool instead of executing it", async () => {
    const appOnly = {
      name: "refresh_order",
      inputSchema: { type: "object", properties: {} },
      _meta: { ui: { visibility: ["app"] } },
    };
    mockListTools.mockResolvedValue({ tools: [uiTool(), appOnly] });
    mockCallTool.mockResolvedValue({ isError: false, content: [{ type: "text", text: "should not run" }] });

    const p = await connectedProvider();
    const resp = makeBedrockResponse("refresh_order", "1", {});
    const results = await p.toolHandler(
      resp,
      bedrockGetToolUseBlock,
      bedrockGetToolName,
      bedrockGetToolId,
      bedrockGetInputData
    );

    expect(results[0].content).toContain("not found"); // the model can't tell it exists
    expect(mockCallTool).not.toHaveBeenCalled(); // and it was NOT executed via the model loop
  });
});
