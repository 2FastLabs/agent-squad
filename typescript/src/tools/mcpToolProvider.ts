import { AgentTool, AgentToolCallbacks, AgentToolResult, AgentTools } from "../utils/tool";
import { ConversationMessage } from "../types";

/**
 * Configuration for a single MCP server connection.
 */
export interface MCPServerConfig {
  /** Transport type: stdio (spawn a local process) or sse (HTTP SSE endpoint) */
  type: "stdio" | "sse";
  /** stdio only: command to execute */
  command?: string;
  /** stdio only: arguments for the command */
  args?: string[];
  /** stdio only: environment variables for the child process */
  env?: Record<string, string>;
  /** sse only: full URL of the SSE endpoint */
  url?: string;
  /** sse only: extra HTTP headers */
  headers?: Record<string, string>;
}

/**
 * MCPToolProvider integrates one or more MCP (Model Context Protocol) servers
 * into the agent-squad tool system as a drop-in replacement for {@link AgentTools}.
 *
 * Use the async factory {@link MCPToolProvider.create} to build a provider.
 * It connects to all MCP servers before returning so that tool definitions are
 * available synchronously when the agent builds its API request:
 *
 * @example
 * ```typescript
 * const provider = await MCPToolProvider.create([
 *   { type: "stdio", command: "uvx", args: ["my-mcp-server"] },
 *   { type: "sse", url: "http://localhost:3000/sse" },
 * ]);
 *
 * const agent = new BedrockLLMAgent({
 *   name: "my-agent",
 *   description: "An agent with MCP tools",
 *   toolConfig: { tool: provider },
 * });
 *
 * // When done, clean up server connections:
 * await provider.disconnect();
 * ```
 *
 * The `@modelcontextprotocol/sdk` package must be installed separately:
 * ```
 * npm install @modelcontextprotocol/sdk
 * ```
 */
export class MCPToolProvider extends AgentTools {
  private servers: MCPServerConfig[];
  /** Connected MCP client instances, one per server */
  private clients: any[] = [];
  /** Map from MCP tool name → the client that owns it */
  private toolClientMap: Map<string, any> = new Map();
  private connected = false;

  constructor(servers: MCPServerConfig[], callbacks?: AgentToolCallbacks) {
    // Start with an empty tools array; populated lazily on first use
    super([], callbacks);
    this.servers = servers;
  }

  /**
   * Create a connected {@link MCPToolProvider}.
   *
   * Connects to all configured MCP servers and populates the internal tool list
   * before returning. Use this instead of `new MCPToolProvider(...)` so that
   * tool definitions are available when the agent builds its API request.
   *
   * @param servers - List of {@link MCPServerConfig} instances.
   * @param callbacks - Optional lifecycle hooks.
   * @returns A fully connected {@link MCPToolProvider}.
   */
  static async create(
    servers: MCPServerConfig[],
    callbacks?: AgentToolCallbacks
  ): Promise<MCPToolProvider> {
    const provider = new MCPToolProvider(servers, callbacks);
    await provider.ensureConnected();
    return provider;
  }

  // ---------------------------------------------------------------------------
  // Lazy connection
  // ---------------------------------------------------------------------------

  /**
   * Ensure all MCP servers are connected and the tool list is populated.
   * Safe to call multiple times — only runs once.
   */
  async ensureConnected(): Promise<void> {
    if (this.connected) return;

    let ClientClass: any;
    let StdioClientTransport: any;
    let SSEClientTransport: any;

    try {
      // @ts-ignore — optional peerDependency; not available in type-checking until installed
      const clientMod = await import("@modelcontextprotocol/sdk/client/index.js");
      ClientClass = clientMod.Client;
    } catch {
      throw new Error(
        "Install @modelcontextprotocol/sdk to use MCPToolProvider: npm install @modelcontextprotocol/sdk"
      );
    }

    try {
      // @ts-ignore — optional peerDependency
      const stdioMod = await import("@modelcontextprotocol/sdk/client/stdio.js");
      StdioClientTransport = stdioMod.StdioClientTransport;
    } catch {
      StdioClientTransport = null;
    }

    try {
      // @ts-ignore — optional peerDependency
      const sseMod = await import("@modelcontextprotocol/sdk/client/sse.js");
      SSEClientTransport = sseMod.SSEClientTransport;
    } catch {
      SSEClientTransport = null;
    }

    const allTools: AgentTool[] = [];

    for (const serverConfig of this.servers) {
      let transport: any;

      if (serverConfig.type === "stdio") {
        if (!StdioClientTransport) {
          throw new Error(
            "StdioClientTransport not available — check your @modelcontextprotocol/sdk installation"
          );
        }
        if (!serverConfig.command) {
          throw new Error(
            "MCPServerConfig with type 'stdio' requires a 'command' field"
          );
        }
        transport = new StdioClientTransport({
          command: serverConfig.command,
          args: serverConfig.args ?? [],
          env: serverConfig.env,
        });
      } else if (serverConfig.type === "sse") {
        if (!SSEClientTransport) {
          throw new Error(
            "SSEClientTransport not available — check your @modelcontextprotocol/sdk installation"
          );
        }
        if (!serverConfig.url) {
          throw new Error(
            "MCPServerConfig with type 'sse' requires a 'url' field"
          );
        }
        transport = new SSEClientTransport(new URL(serverConfig.url), {
          headers: serverConfig.headers ?? {},
        });
      } else {
        throw new Error(
          `Unsupported MCPServerConfig type: ${(serverConfig as any).type}`
        );
      }

      const client = new ClientClass(
        { name: "agent-squad-mcp-client", version: "1.0.0" },
        { capabilities: {} }
      );

      await client.connect(transport);
      this.clients.push(client);

      const { tools: mcpTools } = await client.listTools();

      for (const mcpTool of mcpTools) {
        this.toolClientMap.set(mcpTool.name, client);

        const schema = mcpTool.inputSchema ?? { type: "object", properties: {} };
        const properties: Record<string, any> = schema.properties ?? {};
        const required: string[] = schema.required ?? [];

        // Build an AgentTool with a dummy func — actual execution goes through
        // MCP via our overridden toolHandler.
        const agentTool = new AgentTool({
          name: mcpTool.name,
          description: mcpTool.description ?? `MCP tool: ${mcpTool.name}`,
          properties,
          required,
          func: async () => {
            // Placeholder — never called; MCPToolProvider.toolHandler handles
            // execution directly via the MCP client.
            return null;
          },
        });

        // Store the raw MCP input schema so format methods can pass it through
        // unchanged (it is already valid JSON Schema).
        (agentTool as any)._mcpInputSchema = schema;

        allTools.push(agentTool);
      }
    }

    this.tools = allTools;
    this.connected = true;
  }

  // ---------------------------------------------------------------------------
  // Format overrides — pass MCP inputSchema through directly
  // ---------------------------------------------------------------------------

  /**
   * Returns tool definitions in Amazon Bedrock Converse format.
   * Triggers lazy MCP connection on first call.
   */
  async toBedrockFormat(): Promise<any[]> {
    await this.ensureConnected();
    return this.tools.map((tool) => ({
      toolSpec: {
        name: tool.name,
        description: tool.description,
        inputSchema: {
          json: (tool as any)._mcpInputSchema ?? {
            type: "object",
            properties: tool.properties,
            required: tool.required,
          },
        },
      },
    }));
  }

  /**
   * Returns tool definitions in Anthropic Claude format.
   * Triggers lazy MCP connection on first call.
   */
  async toAnthropicFormat(): Promise<any[]> {
    await this.ensureConnected();
    return this.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: (tool as any)._mcpInputSchema ?? {
        type: "object",
        properties: tool.properties,
        required: tool.required,
      },
    }));
  }

  /**
   * Returns tool definitions in OpenAI Chat Completions format.
   * Triggers lazy MCP connection on first call.
   */
  async toOpenAIFormat(): Promise<any[]> {
    await this.ensureConnected();
    return this.tools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: (tool as any)._mcpInputSchema ?? {
          type: "object",
          properties: tool.properties,
          required: tool.required,
        },
      },
    }));
  }

  // ---------------------------------------------------------------------------
  // toolHandler override — routes to MCP instead of local funcs
  // ---------------------------------------------------------------------------

  /**
   * Handles tool-use blocks from the LLM response by calling the appropriate
   * MCP server. Mirrors the {@link AgentTools#toolHandler} signature exactly
   * so it works inside BedrockLLMAgent, AnthropicAgent, etc. without changes.
   */
  override async toolHandler(
    response: any,
    getToolUseBlock: (block: any) => any,
    getToolName: (toolUseBlock: any) => string,
    getToolId: (toolUseBlock: any) => string,
    getInputData: (toolUseBlock: any) => any
  ): Promise<AgentToolResult[]> {
    await this.ensureConnected();

    if (!response.content) {
      throw new Error("No content blocks in response");
    }

    const toolResults: AgentToolResult[] = [];

    for (const block of response.content) {
      const toolUseBlock = getToolUseBlock(block);
      if (!toolUseBlock) continue;

      const toolName = getToolName(toolUseBlock);
      const toolId = getToolId(toolUseBlock);
      const inputData = getInputData(toolUseBlock);

      const result = await this.callMCPTool(toolName, inputData);
      toolResults.push(new AgentToolResult(toolId, result));
    }

    return toolResults;
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  private async callMCPTool(toolName: string, inputData: any): Promise<string> {
    const client = this.toolClientMap.get(toolName);
    if (!client) {
      return `MCP tool '${toolName}' not found`;
    }

    try {
      const mcpResult = await client.callTool({
        name: toolName,
        arguments: inputData ?? {},
      });

      if (mcpResult.isError) {
        const errorText = this.extractTextFromContent(mcpResult.content);
        return `Error from MCP tool '${toolName}': ${errorText}`;
      }

      return this.extractTextFromContent(mcpResult.content);
    } catch (error: any) {
      return `Error calling MCP tool '${toolName}': ${error?.message ?? String(error)}`;
    }
  }

  private extractTextFromContent(content: any): string {
    if (!content) return "";
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .map((item: any) => {
          if (typeof item === "string") return item;
          if (item?.type === "text") return item.text ?? "";
          if (item?.text) return item.text;
          return JSON.stringify(item);
        })
        .join("\n");
    }
    if (typeof content === "object" && content.text) return content.text;
    return JSON.stringify(content);
  }

  /**
   * Disconnect all MCP clients. Call this when the agent is no longer needed
   * to cleanly shut down stdio child processes or SSE connections.
   */
  async disconnect(): Promise<void> {
    for (const client of this.clients) {
      try {
        await client.close();
      } catch {
        // ignore errors during cleanup
      }
    }
    this.clients = [];
    this.toolClientMap.clear();
    this.tools = [];
    this.connected = false;
  }
}

// Re-export for convenience
export type { ConversationMessage };
