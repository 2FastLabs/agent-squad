"""
MCPToolProvider — drop-in AgentTools subclass that exposes MCP server tools.

Use the async factory :meth:`MCPToolProvider.create` to build a provider.
This connects to all MCP servers upfront so that tool definitions are
available synchronously when the agent builds its API request::

    from agent_squad.tools import MCPToolProvider, MCPServerConfig
    from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions

    provider = await MCPToolProvider.create([
        MCPServerConfig(type="stdio", command="uvx", args=["my-mcp-server"]),
        MCPServerConfig(type="streamable-http", url="http://localhost:3000/mcp"),
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

# Guarded separately: the module only exists since mcp ~1.9, and older installs
# must keep working as long as they don't use the streamable-http transport.
try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    streamablehttp_client = None

# mcp 2.x detection: the first-class Client only exists there. When present, all
# connections go through it (mode="auto" probes server/discover for protocol
# 2026-07-28 and falls back to the legacy initialize handshake per server).
try:
    from mcp import Client as _V2Client
except ImportError:
    _V2Client = None

# v2 name of the streamable HTTP transport (also present in late 1.x).
try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    streamable_http_client = None

# Blessed httpx2/httpx client factory (headers, MCP-recommended timeouts).
try:
    from mcp.shared._httpx_utils import create_mcp_http_client
except ImportError:
    create_mcp_http_client = None

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
    For Streamable HTTP (the current standard HTTP transport) set
    ``type="streamable-http"`` and provide ``url`` / ``headers``.
    For the legacy HTTP+SSE transport set ``type="sse"`` and provide ``url`` / ``headers``.

    Attributes:
        type: Transport type — ``"stdio"``, ``"streamable-http"`` or ``"sse"``.
        command: Executable to launch (stdio only).
        args: Command-line arguments (stdio only).
        env: Environment variables to pass to the subprocess (stdio only).
        url: Server endpoint URL (streamable-http / sse).
        headers: HTTP headers to send with the connection (streamable-http / sse).
    """

    type: str  # "stdio", "streamable-http" or "sse"
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: Optional[dict[str, str]] = None
    url: Optional[str] = None
    headers: Optional[dict[str, str]] = None


_UNSET = object()


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first present attribute among ``names``.

    mcp 1.x exposes camelCase attributes (``inputSchema``, ``isError``) while
    2.x is strictly snake_case (``input_schema``, ``is_error``) with no alias
    attribute access — both spellings must be tried.
    """
    for name in names:
        value = getattr(obj, name, _UNSET)
        if value is not _UNSET:
            return value
    return default


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
            MCPServerConfig(type="streamable-http", url="http://localhost:3000/mcp"),
        ])
        tool_config={"tool": provider}

    Call :meth:`disconnect` when the provider is no longer needed to cleanly
    shut down stdio child processes or HTTP connections.

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
            if _V2Client is not None:
                session = await self._connect_v2(server_cfg)
            else:
                session = await self._connect_v1(server_cfg)

            tools = await self._list_all_tools(session)
            for mcp_tool in tools:
                meta = _meta_dict(mcp_tool)
                self._tool_map[mcp_tool.name] = _MCPToolEntry(
                    session=session,
                    tool=mcp_tool,
                    ui=_ui_resource_uri(meta),
                    model_visible=_model_visible(meta),
                )

        self._connected = True

    def _transport_cm(self, server_cfg: MCPServerConfig, v2: bool) -> Any:
        """The transport async context manager for a server config (shared by both majors)."""
        if server_cfg.type == "stdio":
            if not server_cfg.command:
                raise ValueError("MCPServerConfig with type='stdio' requires a 'command'")
            params = StdioServerParameters(
                command=server_cfg.command,
                args=server_cfg.args or [],
                env=server_cfg.env,
            )
            return stdio_client(params)
        if server_cfg.type == "sse":
            if not server_cfg.url:
                raise ValueError("MCPServerConfig with type='sse' requires a 'url'")
            return sse_client(server_cfg.url, headers=server_cfg.headers or {})
        if server_cfg.type == "streamable-http":
            if not server_cfg.url:
                raise ValueError("MCPServerConfig with type='streamable-http' requires a 'url'")
            if v2:
                # v2 renamed the function and moved headers onto an httpx client.
                if server_cfg.headers:
                    if create_mcp_http_client is not None:
                        http_client = create_mcp_http_client(headers=server_cfg.headers)
                    else:
                        # The factory lives in a private mcp module; fall back to a
                        # plain client (httpx2 is a hard dependency of mcp 2.x).
                        import httpx2

                        http_client = httpx2.AsyncClient(
                            headers=server_cfg.headers, follow_redirects=True
                        )
                    # v2 does not manage a caller-provided client; we own its lifecycle.
                    self._cm_stack.append(http_client)
                    return streamable_http_client(server_cfg.url, http_client=http_client)
                return streamable_http_client(server_cfg.url)
            if streamablehttp_client is None:
                raise ImportError(
                    "The streamable-http transport requires mcp>=1.9. "
                    "Upgrade it with: pip install -U mcp"
                )
            return streamablehttp_client(server_cfg.url, headers=server_cfg.headers or {})
        raise ValueError(
            f"Unsupported MCPServerConfig type: '{server_cfg.type}'. "
            "Use 'stdio', 'streamable-http' or 'sse'."
        )

    async def _connect_v1(self, server_cfg: MCPServerConfig) -> Any:
        """mcp 1.x: transport streams + ClientSession + legacy initialize handshake."""
        cm = self._transport_cm(server_cfg, v2=False)
        # stdio/sse yield (read, write); streamable-http yields (read, write, get_session_id)
        read, write, *_ = await cm.__aenter__()
        self._cm_stack.append(cm)

        session = ClientSession(read, write)
        await session.__aenter__()
        self._sessions.append(session)
        await session.initialize()
        return session

    async def _connect_v2(self, server_cfg: MCPServerConfig) -> Any:
        """mcp 2.x: one Client per server; mode="auto" negotiates the protocol era."""
        client = _V2Client(self._transport_cm(server_cfg, v2=True), mode="auto")
        await client.__aenter__()
        self._sessions.append(client)
        return client

    async def _list_all_tools(self, session: Any) -> list[Any]:
        """All tools from a session, following pagination cursors when present."""
        result = await session.list_tools()
        tools = list(result.tools)
        cursor = _field(result, "nextCursor", "next_cursor")
        # v1 deliberately keeps its pre-existing single-call behavior (no behavior
        # change for existing users); only the v2 path follows pagination cursors.
        while cursor and _V2Client is not None and isinstance(session, _V2Client):
            result = await session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = _field(result, "nextCursor", "next_cursor")
        return tools

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

        if _field(call_result, "isError", "is_error", default=False):
            # Surface the error text back to the model so it can react.
            return ToolResult(content=f"Tool error: {text}" if text else "Tool returned an error")

        structured = _field(call_result, "structuredContent", "structured_content") or {}

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
            # v2's read_resource takes a plain str; v1's ClientSession wants AnyUrl.
            if _V2Client is not None and isinstance(session, _V2Client):
                read_result = await session.read_resource(resource_uri)
            else:
                read_result = await session.read_resource(AnyUrl(resource_uri))
        except Exception:  # noqa: BLE001
            return None
        contents = getattr(read_result, "contents", None) or []
        if not contents:
            return None
        first = contents[0]
        mime_type = _field(first, "mimeType", "mime_type") or "text/html;profile=mcp-app"
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
            raw_schema = _field(mcp_tool, "inputSchema", "input_schema")
            input_schema = (
                raw_schema
                if isinstance(raw_schema, dict)
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
            raw_schema = _field(mcp_tool, "inputSchema", "input_schema")
            input_schema = (
                raw_schema
                if isinstance(raw_schema, dict)
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
            raw_schema = _field(mcp_tool, "inputSchema", "input_schema")
            input_schema = (
                raw_schema
                if isinstance(raw_schema, dict)
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
