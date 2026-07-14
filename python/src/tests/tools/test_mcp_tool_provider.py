"""Unit tests for MCPToolProvider.

All MCP transport and session interactions are mocked — no real server required.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from agent_squad.types import AgentProviderType, ConversationMessage, ParticipantRole
from agent_squad.utils.tool import ToolResult, AgentToolCallbacks


# ---------------------------------------------------------------------------
# Helpers to build mock MCP objects
# ---------------------------------------------------------------------------

def _make_mcp_tool(name: str, description: str = "", properties: dict | None = None,
                   required: list | None = None, meta: dict | None = None):
    """Return a mock object shaped like an mcp Tool (``meta`` == the SDK's ``_meta`` alias)."""
    input_schema = {
        "type": "object",
        "properties": properties or {"query": {"type": "string", "description": "search query"}},
        "required": required or ["query"],
    }
    return SimpleNamespace(name=name, description=description, inputSchema=input_schema, meta=meta)


def _make_call_result(text: str, is_error: bool = False, structured_content: dict | None = None,
                      meta: dict | None = None):
    """Return a mock object shaped like an mcp CallToolResult."""
    content_item = SimpleNamespace(text=text)
    return SimpleNamespace(isError=is_error, content=[content_item],
                           structuredContent=structured_content, meta=meta)


def _make_read_resource_result(text: str, mime_type: str = "text/html;profile=mcp-app"):
    """Return a mock object shaped like an mcp ReadResourceResult (text content)."""
    return SimpleNamespace(contents=[SimpleNamespace(uri="ui://x", mimeType=mime_type, text=text)])


def _make_read_resource_blob_result(blob_b64: str, mime_type: str = "text/html;profile=mcp-app"):
    """Return a ReadResourceResult whose single content is a base64 blob (no text)."""
    return SimpleNamespace(contents=[SimpleNamespace(uri="ui://x", mimeType=mime_type, blob=blob_b64)])


def _make_list_tools_result(tools):
    return SimpleNamespace(tools=tools)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_mcp_modules():
    """Patch mcp imports so tests don't need the real mcp package."""
    mock_client_session_cls = MagicMock()
    mock_stdio_client = MagicMock()
    mock_sse_client = MagicMock()
    mock_stdio_params_cls = MagicMock()

    with (
        patch("agent_squad.tools.mcp_tool_provider.ClientSession", mock_client_session_cls),
        patch("agent_squad.tools.mcp_tool_provider.stdio_client", mock_stdio_client),
        patch("agent_squad.tools.mcp_tool_provider.sse_client", mock_sse_client),
        patch("agent_squad.tools.mcp_tool_provider.StdioServerParameters", mock_stdio_params_cls),
    ):
        yield {
            "ClientSession": mock_client_session_cls,
            "stdio_client": mock_stdio_client,
            "sse_client": mock_sse_client,
            "StdioServerParameters": mock_stdio_params_cls,
        }


def _build_provider(mocks, tools: list, server_type: str = "stdio"):
    """Helper that returns an MCPToolProvider pre-wired with mock internals."""
    from agent_squad.tools.mcp_tool_provider import (
        MCPToolProvider, MCPServerConfig, _MCPToolEntry, _meta_dict, _ui_resource_uri, _model_visible,
    )

    if server_type == "stdio":
        cfg = MCPServerConfig(type="stdio", command="uvx", args=["my-server"])
    else:
        cfg = MCPServerConfig(type="sse", url="http://localhost:3000/sse")

    provider = MCPToolProvider([cfg])

    # Pre-populate the tool map so we skip the real async connect path (mirrors _ensure_connected).
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock()
    for t in tools:
        meta = _meta_dict(t)
        provider._tool_map[t.name] = _MCPToolEntry(
            session=mock_session, tool=t, ui=_ui_resource_uri(meta), model_visible=_model_visible(meta),
        )
    provider._connected = True

    return provider, mock_session


# ---------------------------------------------------------------------------
# to_bedrock_format
# ---------------------------------------------------------------------------

def test_to_bedrock_format(mock_mcp_modules):
    tool = _make_mcp_tool("search", "Search the web", {"q": {"type": "string"}}, ["q"])
    provider, _ = _build_provider(mock_mcp_modules, [tool])

    result = provider.to_bedrock_format()

    assert len(result) == 1
    spec = result[0]["toolSpec"]
    assert spec["name"] == "search"
    assert spec["description"] == "Search the web"
    assert spec["inputSchema"]["json"]["properties"] == {"q": {"type": "string"}}
    assert spec["inputSchema"]["json"]["required"] == ["q"]


# ---------------------------------------------------------------------------
# to_claude_format / to_anthropic_format
# ---------------------------------------------------------------------------

def test_to_claude_format(mock_mcp_modules):
    tool = _make_mcp_tool("calculator", "Do math")
    provider, _ = _build_provider(mock_mcp_modules, [tool])

    result = provider.to_claude_format()

    assert len(result) == 1
    assert result[0]["name"] == "calculator"
    assert result[0]["description"] == "Do math"
    assert "input_schema" in result[0]


def test_to_anthropic_format_alias(mock_mcp_modules):
    tool = _make_mcp_tool("calculator", "Do math")
    provider, _ = _build_provider(mock_mcp_modules, [tool])

    assert provider.to_anthropic_format() == provider.to_claude_format()


# ---------------------------------------------------------------------------
# to_openai_format
# ---------------------------------------------------------------------------

def test_to_openai_format(mock_mcp_modules):
    tool = _make_mcp_tool("fetch_url", "Fetch a URL", {"url": {"type": "string"}}, ["url"])
    provider, _ = _build_provider(mock_mcp_modules, [tool])

    result = provider.to_openai_format()

    assert len(result) == 1
    fn = result[0]
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "fetch_url"
    assert fn["function"]["description"] == "Fetch a URL"
    params = fn["function"]["parameters"]
    assert params["properties"] == {"url": {"type": "string"}}
    assert params["additionalProperties"] is False


# ---------------------------------------------------------------------------
# tool_handler — Bedrock provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_handler_bedrock(mock_mcp_modules):
    from agent_squad.tools.mcp_tool_provider import MCPToolProvider, MCPServerConfig

    tool = _make_mcp_tool("weather", "Get weather")
    provider, mock_session = _build_provider(mock_mcp_modules, [tool])

    mock_session.call_tool.return_value = _make_call_result("Sunny, 25°C")

    # Bedrock-style response content block
    bedrock_response = SimpleNamespace(
        content=[
            {
                "toolUse": {
                    "name": "weather",
                    "toolUseId": "tool-id-001",
                    "input": {"query": "London"},
                }
            }
        ]
    )

    result = await provider.tool_handler(
        AgentProviderType.BEDROCK.value, bedrock_response, []
    )

    assert isinstance(result, ConversationMessage)
    assert result.role == ParticipantRole.USER.value
    assert result.content[0]["toolResult"]["toolUseId"] == "tool-id-001"
    assert result.content[0]["toolResult"]["content"][0]["text"] == "Sunny, 25°C"
    mock_session.call_tool.assert_called_once_with("weather", {"query": "London"})


# ---------------------------------------------------------------------------
# tool_handler — Anthropic provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_handler_anthropic(mock_mcp_modules):
    tool = _make_mcp_tool("weather", "Get weather")
    provider, mock_session = _build_provider(mock_mcp_modules, [tool])

    mock_session.call_tool.return_value = _make_call_result("Rainy, 10°C")

    # Anthropic-style response content block (object with .type attribute)
    tool_use_block = SimpleNamespace(
        type="tool_use", name="weather", id="tool-id-002", input={"query": "Paris"}
    )
    anthropic_response = SimpleNamespace(content=[tool_use_block])

    result = await provider.tool_handler(
        AgentProviderType.ANTHROPIC.value, anthropic_response, []
    )

    assert isinstance(result, dict)
    assert result["role"] == ParticipantRole.USER.value
    tool_result = result["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "tool-id-002"
    assert tool_result["content"] == "Rainy, 10°C"


# ---------------------------------------------------------------------------
# Error handling — isError from MCP
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_handler_mcp_error(mock_mcp_modules):
    tool = _make_mcp_tool("broken_tool", "A broken tool")
    provider, mock_session = _build_provider(mock_mcp_modules, [tool])

    mock_session.call_tool.return_value = _make_call_result(
        "Something went wrong", is_error=True
    )

    bedrock_response = SimpleNamespace(
        content=[
            {
                "toolUse": {
                    "name": "broken_tool",
                    "toolUseId": "tool-id-err",
                    "input": {},
                }
            }
        ]
    )

    result = await provider.tool_handler(
        AgentProviderType.BEDROCK.value, bedrock_response, []
    )

    text_result = result.content[0]["toolResult"]["content"][0]["text"]
    assert "Tool error" in text_result
    assert "Something went wrong" in text_result


# ---------------------------------------------------------------------------
# Error handling — unknown tool name
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_handler_unknown_tool(mock_mcp_modules):
    provider, _ = _build_provider(mock_mcp_modules, [])  # no tools registered

    bedrock_response = SimpleNamespace(
        content=[
            {
                "toolUse": {
                    "name": "nonexistent",
                    "toolUseId": "tool-id-x",
                    "input": {},
                }
            }
        ]
    )

    result = await provider.tool_handler(
        AgentProviderType.BEDROCK.value, bedrock_response, []
    )

    text = result.content[0]["toolResult"]["content"][0]["text"]
    assert "not found" in text


# ---------------------------------------------------------------------------
# Error handling — call_tool raises exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_handler_call_exception(mock_mcp_modules):
    tool = _make_mcp_tool("flaky", "Flaky tool")
    provider, mock_session = _build_provider(mock_mcp_modules, [tool])

    mock_session.call_tool.side_effect = RuntimeError("connection lost")

    bedrock_response = SimpleNamespace(
        content=[
            {
                "toolUse": {
                    "name": "flaky",
                    "toolUseId": "tool-id-f",
                    "input": {},
                }
            }
        ]
    )

    result = await provider.tool_handler(
        AgentProviderType.BEDROCK.value, bedrock_response, []
    )

    text = result.content[0]["toolResult"]["content"][0]["text"]
    assert "Error calling tool" in text
    assert "connection lost" in text


# ---------------------------------------------------------------------------
# Lazy connection — _ensure_connected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lazy_connection_stdio(mock_mcp_modules):
    from agent_squad.tools.mcp_tool_provider import MCPToolProvider, MCPServerConfig

    tool = _make_mcp_tool("ping", "Ping")

    # Build a fake context manager that returns (read, write) streams
    read_mock = MagicMock()
    write_mock = MagicMock()
    fake_cm = AsyncMock()
    fake_cm.__aenter__ = AsyncMock(return_value=(read_mock, write_mock))
    fake_cm.__aexit__ = AsyncMock(return_value=False)

    mock_mcp_modules["stdio_client"].return_value = fake_cm

    # Build a fake ClientSession
    mock_session_instance = AsyncMock()
    mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_session_instance.__aexit__ = AsyncMock(return_value=False)
    mock_session_instance.initialize = AsyncMock()
    mock_session_instance.list_tools = AsyncMock(
        return_value=_make_list_tools_result([tool])
    )
    mock_mcp_modules["ClientSession"].return_value = mock_session_instance

    provider = MCPToolProvider(
        [MCPServerConfig(type="stdio", command="echo", args=["hello"])]
    )

    assert not provider._connected
    assert len(provider._tool_map) == 0

    await provider._ensure_connected()

    assert provider._connected
    assert "ping" in provider._tool_map
    mock_session_instance.initialize.assert_called_once()
    mock_session_instance.list_tools.assert_called_once()


@pytest.mark.asyncio
async def test_lazy_connection_sse(mock_mcp_modules):
    from agent_squad.tools.mcp_tool_provider import MCPToolProvider, MCPServerConfig

    tool = _make_mcp_tool("search", "Search")

    read_mock = MagicMock()
    write_mock = MagicMock()
    fake_cm = AsyncMock()
    fake_cm.__aenter__ = AsyncMock(return_value=(read_mock, write_mock))
    fake_cm.__aexit__ = AsyncMock(return_value=False)

    mock_mcp_modules["sse_client"].return_value = fake_cm

    mock_session_instance = AsyncMock()
    mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_session_instance.__aexit__ = AsyncMock(return_value=False)
    mock_session_instance.initialize = AsyncMock()
    mock_session_instance.list_tools = AsyncMock(
        return_value=_make_list_tools_result([tool])
    )
    mock_mcp_modules["ClientSession"].return_value = mock_session_instance

    provider = MCPToolProvider(
        [MCPServerConfig(type="sse", url="http://localhost:9000/sse")]
    )

    await provider._ensure_connected()

    assert provider._connected
    assert "search" in provider._tool_map
    mock_mcp_modules["sse_client"].assert_called_once_with(
        "http://localhost:9000/sse", headers={}
    )


@pytest.mark.asyncio
async def test_ensure_connected_idempotent(mock_mcp_modules):
    """Calling _ensure_connected twice should not reconnect."""
    from agent_squad.tools.mcp_tool_provider import MCPToolProvider, MCPServerConfig

    provider = MCPToolProvider(
        [MCPServerConfig(type="stdio", command="echo", args=[])]
    )
    provider._connected = True  # pretend already connected

    await provider._ensure_connected()

    # stdio_client should never have been called
    mock_mcp_modules["stdio_client"].assert_not_called()


# ---------------------------------------------------------------------------
# Config validation errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stdio_missing_command(mock_mcp_modules):
    from agent_squad.tools.mcp_tool_provider import MCPToolProvider, MCPServerConfig

    provider = MCPToolProvider([MCPServerConfig(type="stdio")])  # no command

    with pytest.raises(ValueError, match="command"):
        await provider._ensure_connected()


@pytest.mark.asyncio
async def test_sse_missing_url(mock_mcp_modules):
    from agent_squad.tools.mcp_tool_provider import MCPToolProvider, MCPServerConfig

    provider = MCPToolProvider([MCPServerConfig(type="sse")])  # no url

    with pytest.raises(ValueError, match="url"):
        await provider._ensure_connected()


@pytest.mark.asyncio
async def test_unknown_transport_type(mock_mcp_modules):
    from agent_squad.tools.mcp_tool_provider import MCPToolProvider, MCPServerConfig

    provider = MCPToolProvider([MCPServerConfig(type="grpc", url="grpc://localhost")])

    with pytest.raises(ValueError, match="Unsupported"):
        await provider._ensure_connected()


# ---------------------------------------------------------------------------
# Multiple tools from a single server
# ---------------------------------------------------------------------------

def test_multiple_tools_format(mock_mcp_modules):
    tools = [
        _make_mcp_tool("tool_a", "Tool A"),
        _make_mcp_tool("tool_b", "Tool B"),
        _make_mcp_tool("tool_c", "Tool C"),
    ]
    provider, _ = _build_provider(mock_mcp_modules, tools)

    bedrock_result = provider.to_bedrock_format()
    claude_result = provider.to_claude_format()
    openai_result = provider.to_openai_format()

    assert len(bedrock_result) == 3
    assert len(claude_result) == 3
    assert len(openai_result) == 3

    names_bedrock = {r["toolSpec"]["name"] for r in bedrock_result}
    assert names_bedrock == {"tool_a", "tool_b", "tool_c"}

    names_claude = {r["name"] for r in claude_result}
    assert names_claude == {"tool_a", "tool_b", "tool_c"}

    names_openai = {r["function"]["name"] for r in openai_result}
    assert names_openai == {"tool_a", "tool_b", "tool_c"}


# ---------------------------------------------------------------------------
# Tool UI (widget) passthrough
# ---------------------------------------------------------------------------

def test_ui_resource_uri_helper():
    from agent_squad.tools.mcp_tool_provider import _ui_resource_uri
    assert _ui_resource_uri({"ui": {"resourceUri": "ui://a"}}) == "ui://a"
    assert _ui_resource_uri({"openai/outputTemplate": "ui://b"}) == "ui://b"  # OpenAI alias
    assert _ui_resource_uri({}) is None
    assert _ui_resource_uri(None) is None


def test_model_visible_helper():
    from agent_squad.tools.mcp_tool_provider import _model_visible
    assert _model_visible(None) is True
    assert _model_visible({}) is True
    assert _model_visible({"ui": {"visibility": ["model", "app"]}}) is True
    assert _model_visible({"ui": {"visibility": ["model"]}}) is True
    assert _model_visible({"ui": {"visibility": ["app"]}}) is False


def test_app_only_tools_hidden_from_model_but_callable(mock_mcp_modules):
    visible = _make_mcp_tool("get_order", "visible")
    app_only = _make_mcp_tool("refresh_order", "app only", meta={"ui": {"visibility": ["app"]}})
    provider, _ = _build_provider(mock_mcp_modules, [visible, app_only])

    assert {r["toolSpec"]["name"] for r in provider.to_bedrock_format()} == {"get_order"}
    assert {r["name"] for r in provider.to_claude_format()} == {"get_order"}
    assert {r["function"]["name"] for r in provider.to_openai_format()} == {"get_order"}
    # app-only tool is not advertised to the model but stays callable
    assert "refresh_order" in provider._tool_map


@pytest.mark.asyncio
async def test_tool_handler_widget_passthrough(mock_mcp_modules):
    tool = _make_mcp_tool("get_order", "Order status", meta={"ui": {"resourceUri": "ui://shop/order-card"}})
    provider, mock_session = _build_provider(mock_mcp_modules, [tool])
    mock_session.call_tool.return_value = _make_call_result(
        "Order 42: shipped", structured_content={"status": "shipped"}, meta={"a": 1}
    )
    mock_session.read_resource = AsyncMock(return_value=_make_read_resource_result("<div>card</div>"))

    captured: dict = {}

    class _CB(AgentToolCallbacks):
        async def on_tool_end(self, tool_name, payload_input, output, *a, **k):
            captured["out"] = output

    provider.callbacks = _CB()

    bedrock_response = SimpleNamespace(
        content=[{"toolUse": {"name": "get_order", "toolUseId": "1", "input": {"orderId": "42"}}}]
    )
    result = await provider.tool_handler(AgentProviderType.BEDROCK.value, bedrock_response, [])

    # The model receives only the text.
    assert result.content[0]["toolResult"]["content"][0]["text"] == "Order 42: shipped"
    # The capture seam (what GroundedAgent reads) gets a ToolResult carrying the widget.
    out = captured["out"]
    assert isinstance(out, ToolResult)
    assert out.structured_content == {"status": "shipped"}
    assert out.ui is not None
    assert out.ui.resource_uri == "ui://shop/order-card"
    assert out.ui.mime_type == "text/html;profile=mcp-app"
    assert out.ui.template == "<div>card</div>"
    assert out.ui.structured_content == {"status": "shipped"}
    mock_session.read_resource.assert_called_once()


@pytest.mark.asyncio
async def test_tool_handler_no_ui_returns_plain_toolresult(mock_mcp_modules):
    tool = _make_mcp_tool("weather", "Get weather")  # no _meta.ui
    provider, mock_session = _build_provider(mock_mcp_modules, [tool])
    mock_session.call_tool.return_value = _make_call_result("Sunny", structured_content={"t": 25})
    mock_session.read_resource = AsyncMock()

    captured: dict = {}

    class _CB(AgentToolCallbacks):
        async def on_tool_end(self, tool_name, payload_input, output, *a, **k):
            captured["out"] = output

    provider.callbacks = _CB()

    bedrock_response = SimpleNamespace(
        content=[{"toolUse": {"name": "weather", "toolUseId": "1", "input": {}}}]
    )
    await provider.tool_handler(AgentProviderType.BEDROCK.value, bedrock_response, [])

    out = captured["out"]
    assert isinstance(out, ToolResult)
    assert out.ui is None
    mock_session.read_resource.assert_not_called()  # no resource fetch without an advertised UI


@pytest.mark.asyncio
async def test_template_cache_fetches_once(mock_mcp_modules):
    tool = _make_mcp_tool("get_order", "Order", meta={"ui": {"resourceUri": "ui://shop/order-card"}})
    provider, mock_session = _build_provider(mock_mcp_modules, [tool])
    mock_session.call_tool.return_value = _make_call_result("ok", structured_content={"x": 1})
    mock_session.read_resource = AsyncMock(return_value=_make_read_resource_result("<div>card</div>"))

    await provider._call_mcp_tool("get_order", {})
    await provider._call_mcp_tool("get_order", {})
    mock_session.read_resource.assert_called_once()  # cached after the first fetch


@pytest.mark.asyncio
async def test_widget_template_from_blob(mock_mcp_modules):
    import base64
    tool = _make_mcp_tool("get_order", "Order", meta={"ui": {"resourceUri": "ui://shop/order-card"}})
    provider, mock_session = _build_provider(mock_mcp_modules, [tool])
    mock_session.call_tool.return_value = _make_call_result("ok", structured_content={"x": 1})
    blob = base64.b64encode("<div>from blob</div>".encode()).decode()
    mock_session.read_resource = AsyncMock(return_value=_make_read_resource_blob_result(blob))

    result = await provider._call_mcp_tool("get_order", {})
    assert result.ui is not None
    assert result.ui.template == "<div>from blob</div>"  # base64 blob decoded as UTF-8


@pytest.mark.asyncio
async def test_widget_fetch_failure_degrades_to_text(mock_mcp_modules):
    tool = _make_mcp_tool("get_order", "Order", meta={"ui": {"resourceUri": "ui://shop/order-card"}})
    provider, mock_session = _build_provider(mock_mcp_modules, [tool])
    mock_session.call_tool.return_value = _make_call_result("Order 42: shipped", structured_content={"s": 1})
    mock_session.read_resource = AsyncMock(side_effect=RuntimeError("resource read failed"))

    result = await provider._call_mcp_tool("get_order", {})
    # Resource fetch failed → no widget, but the grounded text and structured data still come through.
    assert result.ui is None
    assert result.content == "Order 42: shipped"
    assert result.structured_content == {"s": 1}


@pytest.mark.asyncio
async def test_empty_content_falls_back_to_structured_json(mock_mcp_modules):
    tool = _make_mcp_tool("stats", "Stats")
    provider, mock_session = _build_provider(mock_mcp_modules, [tool])
    # No text content, but structured data present.
    mock_session.call_tool.return_value = SimpleNamespace(
        isError=False, content=[], structuredContent={"count": 3}, meta=None
    )
    bedrock_response = SimpleNamespace(
        content=[{"toolUse": {"name": "stats", "toolUseId": "1", "input": {}}}]
    )
    result = await provider.tool_handler(AgentProviderType.BEDROCK.value, bedrock_response, [])
    # Model receives the JSON of structured_content rather than an empty string.
    assert result.content[0]["toolResult"]["content"][0]["text"] == json.dumps({"count": 3}, default=str)


@pytest.mark.asyncio
async def test_app_only_tool_still_callable(mock_mcp_modules):
    app_only = _make_mcp_tool("refresh_order", "app only", meta={"ui": {"visibility": ["app"]}})
    provider, mock_session = _build_provider(mock_mcp_modules, [app_only])
    mock_session.call_tool.return_value = _make_call_result("refreshed", structured_content={"s": 2})

    # Not advertised to the model...
    assert provider.to_bedrock_format() == []
    # ...but the UI can still invoke it directly.
    result = await provider._call_mcp_tool("refresh_order", {})
    assert result.content == "refreshed"
    assert result.structured_content == {"s": 2}
