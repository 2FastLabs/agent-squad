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

import base64
import json

try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.sse import sse_client
except ImportError as exc:
    raise ImportError(
        "MCPToolProvider requires the 'mcp' package. "
        "Install it with: pip install agent-squad[mcp]"
    ) from exc

from pydantic import AnyUrl  # mcp depends on pydantic, so it's available whenever the import above succeeds

from dataclasses import dataclass, field
from typing import Any, Optional

from agent_squad.types import AgentProviderType, ConversationMessage, ParticipantRole
from agent_squad.utils.tool import AgentTools, AgentToolCallbacks, ToolResult
from agent_squad.utils.ui import UIPayload


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


def _meta_dict(mcp_tool: Any) -> Optional[dict[str, Any]]:
    """The tool's ``_meta`` (MCP Apps UI metadata); the SDK exposes it as ``.meta``."""
    return getattr(mcp_tool, "meta", None) or getattr(mcp_tool, "_meta", None)


def _ui_resource_uri(meta: Optional[dict[str, Any]]) -> Optional[str]:
    """The advertised UI template URI: ``_meta.ui.resourceUri``, or the OpenAI
    ``openai/outputTemplate`` alias. Tolerant of absent/wrong-typed fields."""
    if not isinstance(meta, dict):
        return None
    ui = meta.get("ui")
    if isinstance(ui, dict) and isinstance(ui.get("resourceUri"), str):
        return ui["resourceUri"]
    alias = meta.get("openai/outputTemplate")
    return alias if isinstance(alias, str) else None


def _model_visible(meta: Optional[dict[str, Any]]) -> bool:
    """Whether the model may be offered the tool. ``_meta.ui.visibility`` lists audiences
    (``model`` / ``app``); absent means both (the MCP default)."""
    if not isinstance(meta, dict):
        return True
    ui = meta.get("ui")
    if not isinstance(ui, dict):
        return True
    visibility = ui.get("visibility")
    if not isinstance(visibility, list):
        return True
    return "model" in visibility


@dataclass
class _MCPToolEntry:
    """Per-tool state: owning session, the raw MCP tool, its advertised UI resource URI (if any),
    and whether the model may see it (app-only tools stay callable but unadvertised)."""

    session: Any
    tool: Any
    ui: Optional[str] = None
    model_visible: bool = True


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
        super().__init__(tools=[], callbacks=callbacks)

        self._servers = servers
        # Maps tool_name → _MCPToolEntry
        self._tool_map: dict[str, _MCPToolEntry] = {}
        # Caches fetched UI templates: (session id, resourceUri) → (mime_type, body).
        # Keyed by session too, so two servers advertising the same URI can't collide.
        self._template_cache: dict[tuple[int, str], tuple[str, str]] = {}
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
                meta = _meta_dict(mcp_tool)
                self._tool_map[mcp_tool.name] = _MCPToolEntry(
                    session=session,
                    tool=mcp_tool,
                    ui=_ui_resource_uri(meta),
                    model_visible=_model_visible(meta),
                )

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
        self._template_cache = {}
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

            # Only the text reaches the model; the structured data + widget ride on the ToolResult
            # for a UI-aware consumer (e.g. GroundedAgent), captured via on_tool_end above.
            model_text = result.content or json.dumps(result.structured_content, default=str)

            if provider_type == AgentProviderType.BEDROCK.value:
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_id,
                            "content": [{"text": model_text}],
                        }
                    }
                )
            else:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": model_text,
                    }
                )

        if provider_type == AgentProviderType.BEDROCK.value:
            return ConversationMessage(
                role=ParticipantRole.USER.value, content=tool_results
            )
        return {"role": ParticipantRole.USER.value, "content": tool_results}

    async def _call_mcp_tool(self, tool_name: str, input_data: dict) -> ToolResult:
        """Call a tool on the appropriate MCP server.

        Returns a :class:`~agent_squad.utils.tool.ToolResult` carrying the text (added to the model's
        context), the render-only ``structured_content``, and a ``UIPayload`` widget when the tool
        advertised one via its ``_meta.ui`` (fetched from the server as a resource)."""
        entry = self._tool_map.get(tool_name)
        if entry is None:
            return ToolResult(content=f"Tool '{tool_name}' not found in any connected MCP server")

        try:
            call_result = await entry.session.call_tool(tool_name, input_data)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"Error calling tool '{tool_name}': {exc}")

        parts = [
            getattr(item, "text", str(item))
            for item in (getattr(call_result, "content", None) or [])
        ]
        text = "\n".join(parts)

        if getattr(call_result, "isError", False):
            # Surface the error text back to the model so it can react.
            return ToolResult(content=f"Tool error: {text}" if text else "Tool returned an error")

        structured = getattr(call_result, "structuredContent", None) or {}

        ui: Optional[UIPayload] = None
        if entry.ui:
            template = await self._template_for(entry.session, entry.ui)
            if template is not None:
                mime_type, body = template
                ui = UIPayload(
                    resource_uri=entry.ui,
                    mime_type=mime_type,
                    template=body,
                    structured_content=structured,
                    meta=getattr(call_result, "meta", None),
                )

        return ToolResult(content=text, structured_content=structured, ui=ui)

    async def _template_for(self, session: Any, resource_uri: str) -> Optional[tuple[str, str]]:
        """Fetch (and cache) a UI template resource by URI. Returns ``(mime_type, body)`` or ``None``.

        A resource arrives as text or as a base64 ``blob``; the blob is decoded as UTF-8 markup."""
        cache_key = (id(session), resource_uri)
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]
        try:
            read_result = await session.read_resource(AnyUrl(resource_uri))
        except Exception:  # noqa: BLE001
            return None
        contents = getattr(read_result, "contents", None) or []
        if not contents:
            return None
        first = contents[0]
        mime_type = getattr(first, "mimeType", None) or "text/html;profile=mcp-app"
        body = getattr(first, "text", None)
        if body is None:
            blob = getattr(first, "blob", None)
            if isinstance(blob, str):
                try:
                    body = base64.b64decode(blob).decode("utf-8")
                except Exception:  # noqa: BLE001
                    body = None
        if body is None:
            return None
        self._template_cache[cache_key] = (mime_type, body)
        return (mime_type, body)

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
        for tool_name, entry in self._tool_map.items():
            if not entry.model_visible:
                continue  # app-only tool: callable by the UI, never advertised to the model
            mcp_tool = entry.tool
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
        for tool_name, entry in self._tool_map.items():
            if not entry.model_visible:
                continue  # app-only tool: callable by the UI, never advertised to the model
            mcp_tool = entry.tool
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
        for tool_name, entry in self._tool_map.items():
            if not entry.model_visible:
                continue  # app-only tool: callable by the UI, never advertised to the model
            mcp_tool = entry.tool
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
