/**
 * Edge case: the v2 root package resolves but its Node-only /stdio subpath does
 * not (e.g. a bundled/browser-ish environment). stdio configs must fail with an
 * error naming the v2 package, and HTTP transports must keep working.
 */

class MockV2Client {
  connect = jest.fn().mockResolvedValue(undefined);
  listTools = jest.fn().mockResolvedValue({ tools: [] });
  close = jest.fn();
}

class MockV2StreamableTransport {
  constructor(public url: any, public opts: any) {}
}

// Only the root subpath is mocked — "@modelcontextprotocol/client/stdio" stays
// unresolvable, so StdioClientTransport ends up null while usingV2 is true.
jest.mock(
  "@modelcontextprotocol/client",
  () => ({
    Client: MockV2Client,
    SSEClientTransport: class {},
    StreamableHTTPClientTransport: MockV2StreamableTransport,
  }),
  { virtual: true }
);

import { MCPToolProvider } from "../src/tools/mcpToolProvider";

describe("MCPToolProvider with v2 root but no /stdio subpath", () => {
  it("fails stdio configs with an error naming @modelcontextprotocol/client", async () => {
    const provider = new MCPToolProvider([
      { type: "stdio", command: "uvx", args: ["my-server"] },
    ]);

    await expect(provider.ensureConnected()).rejects.toThrow(
      "@modelcontextprotocol/client"
    );
  });

  it("still connects streamable-http configs", async () => {
    const provider = new MCPToolProvider([
      { type: "streamable-http", url: "http://localhost:9000/mcp" },
    ]);

    await provider.ensureConnected();
  });
});
