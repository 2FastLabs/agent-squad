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

class MockClient {
  connect = mockConnect;
  listTools = mockListTools;
  callTool = mockCallTool;
  close = mockClose;
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
import { AgentTools, AgentToolResult } from "../src/utils/tool";

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
});
