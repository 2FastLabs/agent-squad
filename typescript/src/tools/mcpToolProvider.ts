import { AgentTool, AgentToolCallbacks, AgentToolResult, AgentTools, ToolResult } from "../utils/tool";
import { UIPayload } from "../utils/ui";
import { ConversationMessage } from "../types";

/** The advertised UI template URI: `_meta.ui.resourceUri`, or the OpenAI `openai/outputTemplate` alias. */
function uiResourceUri(meta: any): string | undefined {
  if (!meta || typeof meta !== "object") return undefined;
  const fromUi = meta.ui?.resourceUri;
  if (typeof fromUi === "string") return fromUi;
  const alias = meta["openai/outputTemplate"];
  return typeof alias === "string" ? alias : undefined;
}

/** Whether the model may be offered the tool. `_meta.ui.visibility` lists audiences; absent → both. */
function modelVisible(meta: any): boolean {
  const visibility = meta?.ui?.visibility;
  return Array.isArray(visibility) ? visibility.includes("model") : true;
}

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
  /** Map from MCP tool name → its advertised UI resource URI (if any) */
  private toolUiMap: Map<string, string> = new Map();
  /** Per-client cache of fetched UI templates, keyed by resource URI */
  private templateCache: WeakMap<object, Map<string, { mimeType: string; body: string }>> =
    new WeakMap();
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
        const toolName = mcpTool.name;
        this.toolClientMap.set(toolName, client);

        const meta = mcpTool._meta;
        const ui = uiResourceUri(meta);
        if (ui) this.toolUiMap.set(toolName, ui);

        const schema = mcpTool.inputSchema ?? { type: "object", properties: {} };
        const properties: Record<string, any> = schema.properties ?? {};
        const required: string[] = schema.required ?? [];

        // A real func: the gatherer invokes it through the base tool loop, which lets a UI-aware
        // consumer (GroundedAgent) capture the ToolResult it returns — including any widget.
        const agentTool = new AgentTool({
          name: toolName,
          description: mcpTool.description ?? `MCP tool: ${toolName}`,
          properties,
          required,
          func: async (input: any) => this.callMCPTool(toolName, input),
        });

        // Store the raw MCP input schema so format methods can pass it through
        // unchanged (it is already valid JSON Schema).
        (agentTool as any)._mcpInputSchema = schema;

        // App-only tools stay callable (in toolClientMap) but are never advertised to the model.
        if (modelVisible(meta)) allTools.push(agentTool);
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
   * Runs each tool-use block by invoking the tool's `func` (our MCP call, returning a
   * {@link ToolResult}) with the full input, then routes only its text to the model. Calling `func`
   * is what lets a UI-aware consumer such as `GroundedAgent` capture the widget-carrying result.
   *
   * We invoke `func` directly (rather than delegating to the base loop) so the tool receives its
   * full input — the base `processTool` would unwrap a `messages` key and drop the other arguments.
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

      // Model-visible tools carry a real func (captured by GroundedAgent); app-only / unknown tools
      // aren't advertised, so fall back to a direct MCP call.
      // Only model-visible tools (in this.tools) are executable via the model loop. An app-only or
      // unknown name is rejected here — an app-only MCP tool must never be invocable by the model.
      const tool = this.tools.find((t) => t.name === toolName);
      const result = tool
        ? await tool.func(inputData)
        : new ToolResult(`MCP tool '${toolName}' not found`);

      const content =
        result instanceof ToolResult
          ? result.content || JSON.stringify(result.structuredContent)
          : result;
      toolResults.push(new AgentToolResult(toolId, content));
    }

    return toolResults;
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  private async callMCPTool(toolName: string, inputData: any): Promise<ToolResult> {
    const client = this.toolClientMap.get(toolName);
    if (!client) {
      return new ToolResult(`MCP tool '${toolName}' not found`);
    }

    try {
      const mcpResult = await client.callTool({
        name: toolName,
        arguments: inputData ?? {},
      });

      const text = this.extractTextFromContent(mcpResult.content);
      if (mcpResult.isError) {
        return new ToolResult(`Error from MCP tool '${toolName}': ${text}`);
      }

      const structured = mcpResult.structuredContent ?? {};
      let ui: UIPayload | undefined;
      const resourceUri = this.toolUiMap.get(toolName);
      if (resourceUri) {
        const template = await this.readTemplate(client, resourceUri);
        if (template) {
          ui = {
            resourceUri,
            mimeType: template.mimeType,
            template: template.body,
            structuredContent: structured,
            meta: mcpResult._meta,
          };
        }
      }
      return new ToolResult(text, structured, ui);
    } catch (error: any) {
      return new ToolResult(
        `Error calling MCP tool '${toolName}': ${error?.message ?? String(error)}`
      );
    }
  }

  /** Fetch (and cache, per client) a UI template resource. A resource is text or a base64 blob. */
  private async readTemplate(
    client: any,
    resourceUri: string
  ): Promise<{ mimeType: string; body: string } | undefined> {
    const cached = this.templateCache.get(client);
    if (cached?.has(resourceUri)) return cached.get(resourceUri);

    let template: { mimeType: string; body: string } | undefined;
    try {
      const res = await client.readResource({ uri: resourceUri });
      const first = res?.contents?.[0];
      if (first) {
        const mimeType: string = first.mimeType ?? "text/html;profile=mcp-app";
        let body: string | undefined = typeof first.text === "string" ? first.text : undefined;
        if (body === undefined && typeof first.blob === "string") {
          body = Buffer.from(first.blob, "base64").toString("utf-8");
        }
        if (body !== undefined) template = { mimeType, body };
      }
    } catch {
      template = undefined;
    }

    if (template) {
      const byUri = cached ?? new Map<string, { mimeType: string; body: string }>();
      byUri.set(resourceUri, template);
      this.templateCache.set(client, byUri);
    }
    return template;
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
    this.toolUiMap.clear();
    this.templateCache = new WeakMap();
    this.tools = [];
    this.connected = false;
  }
}

// Re-export for convenience
export type { ConversationMessage };
