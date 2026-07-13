"""
MCPToolProvider — drop-in AgentTools subclass that exposes MCP server tools.

Use the async factory :meth:`MCPToolProvider.create` to build a provider.
This connects to all MCP servers upfront so that tool definitions are
available synchronously when the agent builds its API request::

    from agent_squad.tools import MCPToolProvider, MCPServerConfig
    from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions

    provider = await MCPToolProvider.create([
        MCPServerConfig(type="stdio", command="uvx", args=["my-mcp-server"]),
        MCPServerConfig(type="sse", url="http://localhost:3000/sse"),
    ])

    agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="my-agent",
        description="An agent with MCP tools",
        tool_config={"tool": provider}
    ))

    # When done, clean up server connections:
    await provider.disconnect()

Requires the ``mcp`` extra::

    pip install agent-squad[mcp]
"""

from __future__ import annotations

try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.sse import sse_client
except ImportError as exc:
    raise ImportError(
        "MCPToolProvider requires the 'mcp' package. "
        "Install it with: pip install agent-squad[mcp]"
    ) from exc

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_squad.types import AgentProviderType, ConversationMessage, ParticipantRole
from agent_squad.utils.tool import AgentTools, AgentToolCallbacks


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server.

    For stdio transport set ``type="stdio"`` and provide ``command`` / ``args`` / ``env``.
    For SSE/HTTP transport set ``type="sse"`` and provide ``url`` / ``headers``.

    Attributes:
        type: Transport type — ``"stdio"`` or ``"sse"``.
        command: Executable to launch (stdio only).
        args: Command-line arguments (stdio only).
        env: Environment variables to pass to the subprocess (stdio only).
        url: SSE endpoint URL (sse only).
        headers: HTTP headers to send with the SSE connection (sse only).
    """

    type: str  # "stdio" or "sse"
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: Optional[dict[str, str]] = None
    url: Optional[str] = None
    headers: Optional[dict[str, str]] = None


class MCPToolProvider(AgentTools):
    """AgentTools subclass that proxies tools from one or more MCP servers.

    Use the async class method :meth:`create` to build a provider.  It connects
    to all configured MCP servers and populates the tool list before returning,
    so tool definitions are available synchronously when the agent builds its
    API request::

        provider = await MCPToolProvider.create([
            MCPServerConfig(type="stdio", command="uvx", args=["my-server"]),
            MCPServerConfig(type="sse", url="http://localhost:3000/sse"),
        ])
        tool_config={"tool": provider}

    Call :meth:`disconnect` when the provider is no longer needed to cleanly
    shut down stdio child processes or SSE connections.

    Args:
        servers: List of :class:`MCPServerConfig` describing the MCP servers to
            connect to.
        callbacks: Optional :class:`~agent_squad.utils.tool.AgentToolCallbacks`
            instance for lifecycle hooks.
    """

    def __init__(
        self,
        servers: list[MCPServerConfig],
        callbacks: Optional[AgentToolCallbacks] = None,
    ) -> None:
        # Intentionally bypass AgentTools.__init__ — we have no static AgentTool list
        self.tools: list[Any] = []  # unused; tools come from MCP at runtime
        self.callbacks = callbacks or AgentToolCallbacks()

        self._servers = servers
        # Maps tool_name → (ClientSession, mcp_tool)
        self._tool_map: dict[str, tuple[Any, Any]] = {}
        self._connected = False
        # Keep hold of context-manager stacks so we can exit them on disconnect
        self._cm_stack: list[Any] = []
        self._sessions: list[Any] = []

    @classmethod
    async def create(
        cls,
        servers: list[MCPServerConfig],
        callbacks: Optional[AgentToolCallbacks] = None,
    ) -> "MCPToolProvider":
        """Create a connected :class:`MCPToolProvider`.

        Connects to all configured MCP servers and populates the internal tool
        map before returning.  Use this instead of constructing the class
        directly so that tool definitions are available when the agent builds
        its API request.

        Args:
            servers: List of :class:`MCPServerConfig` instances.
            callbacks: Optional lifecycle hooks.

        Returns:
            A fully connected :class:`MCPToolProvider` instance.
        """
        provider = cls(servers, callbacks)
        await provider._ensure_connected()
        return provider

    # ------------------------------------------------------------------
    # Internal connection management
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> None:
        """Lazily connect to all configured servers and cache their tools."""
        if self._connected:
            return

        for server_cfg in self._servers:
            if server_cfg.type == "stdio":
                if not server_cfg.command:
                    raise ValueError("MCPServerConfig with type='stdio' requires a 'command'")
                params = StdioServerParameters(
                    command=server_cfg.command,
                    args=server_cfg.args or [],
                    env=server_cfg.env,
                )
                cm = stdio_client(params)
            elif server_cfg.type == "sse":
                if not server_cfg.url:
                    raise ValueError("MCPServerConfig with type='sse' requires a 'url'")
                cm = sse_client(server_cfg.url, headers=server_cfg.headers or {})
            else:
                raise ValueError(
                    f"Unsupported MCPServerConfig type: '{server_cfg.type}'. "
                    "Use 'stdio' or 'sse'."
                )

            read, write = await cm.__aenter__()
            self._cm_stack.append(cm)

            session = ClientSession(read, write)
            await session.__aenter__()
            self._sessions.append(session)
            await session.initialize()

            tools_result = await session.list_tools()
            for mcp_tool in tools_result.tools:
                self._tool_map[mcp_tool.name] = (session, mcp_tool)

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from all MCP servers and release resources.

        Closes all client sessions and transport context managers.  After
        calling this method the provider must not be used again.
        """
        for session in self._sessions:
            try:
                await session.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        self._sessions = []

        for cm in self._cm_stack:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        self._cm_stack = []

        self._tool_map = {}
        self._connected = False

    # ------------------------------------------------------------------
    # AgentTools interface
    # ------------------------------------------------------------------

    async def tool_handler(
        self,
        provider_type: str,
        response: Any,
        _conversation: list[dict[str, Any]],
        agent_info: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Execute tool calls found in *response* and return results.

        Compatible with Bedrock, Anthropic, and OpenAI response shapes.
        """
        await self._ensure_connected()

        if not response.content:
            raise ValueError("No content blocks in response")

        tool_results = []

        for block in response.content:
            tool_use_block = self._get_tool_use_block(provider_type, block)
            if not tool_use_block:
                continue

            if provider_type == AgentProviderType.BEDROCK.value:
                tool_name = tool_use_block.get("name")
                tool_id = tool_use_block.get("toolUseId")
                input_data = tool_use_block.get("input", {})
            else:
                # Anthropic object-style block
                tool_name = tool_use_block.name
                tool_id = tool_use_block.id
                input_data = tool_use_block.input

            await self.callbacks.on_tool_start(
                tool_name, input_data, metadata={"agent_info": agent_info}
            )

            result = await self._call_mcp_tool(tool_name, input_data)

            await self.callbacks.on_tool_end(
                tool_name, input_data, result, metadata={"agent_info": agent_info}
            )

            if provider_type == AgentProviderType.BEDROCK.value:
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_id,
                            "content": [{"text": result}],
                        }
                    }
                )
            else:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result,
                    }
                )

        if provider_type == AgentProviderType.BEDROCK.value:
            return ConversationMessage(
                role=ParticipantRole.USER.value, content=tool_results
            )
        return {"role": ParticipantRole.USER.value, "content": tool_results}

    async def _call_mcp_tool(self, tool_name: str, input_data: dict) -> str:
        """Call a tool on the appropriate MCP server and return a string result."""
        if tool_name not in self._tool_map:
            return f"Tool '{tool_name}' not found in any connected MCP server"

        session, _ = self._tool_map[tool_name]
        try:
            call_result = await session.call_tool(tool_name, input_data)
        except Exception as exc:  # noqa: BLE001
            return f"Error calling tool '{tool_name}': {exc}"

        if getattr(call_result, "isError", False):
            # Surface the error text back to the model so it can react
            parts = [
                getattr(item, "text", str(item))
                for item in (call_result.content or [])
            ]
            return f"Tool error: {' '.join(parts)}" if parts else "Tool returned an error"

        # Collect text content from the result
        parts = [
            getattr(item, "text", str(item))
            for item in (call_result.content or [])
        ]
        return "\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Format conversion helpers
    # ------------------------------------------------------------------

    def to_bedrock_format(self) -> list[dict[str, Any]]:
        """Return MCP tools in the Bedrock ``toolSpec`` format.

        .. note::
            This is a *synchronous* method.  If the provider has not yet been
            connected you must call ``await provider._ensure_connected()``
            before using this method, or use the agent's async flow which will
            connect lazily via :meth:`tool_handler`.
        """
        result = []
        for tool_name, (_session, mcp_tool) in self._tool_map.items():
            input_schema = (
                mcp_tool.inputSchema
                if isinstance(mcp_tool.inputSchema, dict)
                else {"type": "object", "properties": {}}
            )
            result.append(
                {
                    "toolSpec": {
                        "name": tool_name,
                        "description": mcp_tool.description or "",
                        "inputSchema": {"json": input_schema},
                    }
                }
            )
        return result

    def to_claude_format(self) -> list[dict[str, Any]]:
        """Return MCP tools in the Anthropic / Claude ``input_schema`` format."""
        result = []
        for tool_name, (_session, mcp_tool) in self._tool_map.items():
            input_schema = (
                mcp_tool.inputSchema
                if isinstance(mcp_tool.inputSchema, dict)
                else {"type": "object", "properties": {}}
            )
            result.append(
                {
                    "name": tool_name,
                    "description": mcp_tool.description or "",
                    "input_schema": input_schema,
                }
            )
        return result

    def to_anthropic_format(self) -> list[dict[str, Any]]:
        """Alias for :meth:`to_claude_format` (same wire format)."""
        return self.to_claude_format()

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Return MCP tools in the OpenAI function-calling format."""
        result = []
        for tool_name, (_session, mcp_tool) in self._tool_map.items():
            input_schema = (
                mcp_tool.inputSchema
                if isinstance(mcp_tool.inputSchema, dict)
                else {"type": "object", "properties": {}}
            )
            # Ensure required field is present for strict mode compatibility
            parameters = {**input_schema}
            if "additionalProperties" not in parameters:
                parameters["additionalProperties"] = False
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": mcp_tool.description or "",
                        "parameters": parameters,
                    },
                }
            )
        return result
