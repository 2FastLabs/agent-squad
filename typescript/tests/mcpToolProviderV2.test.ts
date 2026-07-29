/**
 * Unit tests for MCPToolProvider on the v2 MCP SDK (@modelcontextprotocol/client).
 *
 * The v2 package is virtually mocked; when present it must be preferred over v1
 * and the client must be constructed with versionNegotiation mode "auto" so both
 * 2026-07-28 and legacy servers work. The v1 suite (mcpToolProvider.test.ts) does
 * NOT mock the v2 package, which proves the v1 fallback path stays intact.
 */

const mockCallTool = jest.fn();
const mockListTools = jest.fn();
const mockConnect = jest.fn();
const mockClose = jest.fn();
const mockReadResource = jest.fn();

const clientInstances: MockV2Client[] = [];
class MockV2Client {
  connect = mockConnect;
  listTools = mockListTools;
  callTool = mockCallTool;
  close = mockClose;
  readResource = mockReadResource;
  constructor(public info: any, public options: any) {
    clientInstances.push(this);
  }
}

class MockV2StdioTransport {
  constructor(public opts: any) {}
}

const sseInstances: MockV2SSETransport[] = [];
class MockV2SSETransport {
  constructor(public url: any, public opts: any) {
    sseInstances.push(this);
  }
}

const streamableInstances: MockV2StreamableTransport[] = [];
class MockV2StreamableTransport {
  constructor(public url: any, public opts: any) {
    streamableInstances.push(this);
  }
}

// v1 mock: a sentinel that must never be reached while v2 is installed
const v1ClientConstructed = jest.fn();
class MockV1Client {
  constructor() {
    v1ClientConstructed();
  }
}

jest.mock(
  "@modelcontextprotocol/client",
  () => ({
    Client: MockV2Client,
    SSEClientTransport: MockV2SSETransport,
    StreamableHTTPClientTransport: MockV2StreamableTransport,
  }),
  { virtual: true }
);
jest.mock(
  "@modelcontextprotocol/client/stdio",
  () => ({ StdioClientTransport: MockV2StdioTransport }),
  { virtual: true }
);
jest.mock(
  "@modelcontextprotocol/sdk/client/index.js",
  () => ({ Client: MockV1Client }),
  { virtual: true }
);

import { MCPToolProvider } from "../src/tools/mcpToolProvider";

const weatherTool = {
  name: "get_weather",
  description: "Returns weather for a location",
  inputSchema: {
    type: "object",
    properties: { location: { type: "string" } },
    required: ["location"],
  },
};

beforeEach(() => {
  jest.clearAllMocks();
  clientInstances.length = 0;
  sseInstances.length = 0;
  streamableInstances.length = 0;
  mockListTools.mockResolvedValue({ tools: [weatherTool] });
  mockConnect.mockResolvedValue(undefined);
});

describe("MCPToolProvider on the v2 SDK", () => {
  it("prefers v2 over v1 and never constructs the v1 client", async () => {
    const provider = new MCPToolProvider([
      { type: "stdio", command: "uvx", args: ["my-server"] },
    ]);

    await provider.ensureConnected();

    expect(clientInstances).toHaveLength(1);
    expect(v1ClientConstructed).not.toHaveBeenCalled();
  });

  it("constructs the v2 client with versionNegotiation mode auto", async () => {
    const provider = new MCPToolProvider([
      { type: "stdio", command: "uvx", args: ["my-server"] },
    ]);

    await provider.ensureConnected();

    expect(clientInstances[0].info).toEqual({
      name: "agent-squad-mcp-client",
      version: "1.0.0",
    });
    expect(clientInstances[0].options).toEqual({
      capabilities: {},
      versionNegotiation: { mode: "auto" },
    });
  });

  it("uses the v2 stdio transport from the /stdio subpath", async () => {
    const provider = new MCPToolProvider([
      {
        type: "stdio",
        command: "uvx",
        args: ["my-server"],
        env: { API_KEY: "k" },
      },
    ]);

    await provider.ensureConnected();

    expect(mockConnect).toHaveBeenCalledWith(expect.any(MockV2StdioTransport));
    const transport = mockConnect.mock.calls[0][0] as MockV2StdioTransport;
    expect(transport.opts).toEqual({
      command: "uvx",
      args: ["my-server"],
      env: { API_KEY: "k" },
    });
  });

  it("constructs the v2 streamable-http transport with requestInit headers", async () => {
    const provider = new MCPToolProvider([
      {
        type: "streamable-http",
        url: "http://localhost:9000/mcp",
        headers: { "x-api-key": "abc" },
      },
    ]);

    await provider.ensureConnected();

    expect(streamableInstances).toHaveLength(1);
    expect(streamableInstances[0].url.href).toBe("http://localhost:9000/mcp");
    expect(streamableInstances[0].opts).toEqual({
      requestInit: { headers: { "x-api-key": "abc" } },
    });
  });

  it("constructs the v2 sse transport with requestInit headers (no dead headers key)", async () => {
    const provider = new MCPToolProvider([
      {
        type: "sse",
        url: "http://localhost:9000/sse",
        headers: { "x-api-key": "abc" },
      },
    ]);

    await provider.ensureConnected();

    expect(sseInstances).toHaveLength(1);
    expect(sseInstances[0].opts).toEqual({
      requestInit: { headers: { "x-api-key": "abc" } },
    });
  });

  it("passes no options to the v2 sse transport when no headers are configured", async () => {
    const provider = new MCPToolProvider([
      { type: "sse", url: "http://localhost:9000/sse" },
    ]);

    await provider.ensureConnected();

    expect(sseInstances).toHaveLength(1);
    expect(sseInstances[0].opts).toBeUndefined();
  });

  it("lists and calls tools through the v2 client", async () => {
    mockCallTool.mockResolvedValue({
      content: [{ type: "text", text: "sunny" }],
      isError: false,
    });

    const provider = await MCPToolProvider.create([
      { type: "stdio", command: "uvx", args: ["my-server"] },
    ]);

    const formats = await provider.toBedrockFormat();
    expect(formats).toHaveLength(1);
    expect(formats[0].toolSpec.name).toBe("get_weather");

    const response = {
      role: "assistant",
      content: [
        { toolUse: { name: "get_weather", toolUseId: "1", input: { location: "Paris" } } },
      ],
    };
    await provider.toolHandler(
      response,
      (b: any) => b.toolUse ?? null,
      (b: any) => b.name,
      (b: any) => b.toolUseId,
      (b: any) => b.input
    );

    expect(mockCallTool).toHaveBeenCalledWith({
      name: "get_weather",
      arguments: { location: "Paris" },
    });
  });

  it("disconnect closes the v2 client", async () => {
    const provider = await MCPToolProvider.create([
      { type: "stdio", command: "uvx", args: ["my-server"] },
    ]);

    await provider.disconnect();

    expect(mockClose).toHaveBeenCalled();
  });
});
