"""Optional tool providers for agent-squad."""

try:
    from .mcp_tool_provider import MCPToolProvider, MCPServerConfig
    __all__ = ["MCPToolProvider", "MCPServerConfig"]
except ImportError:
    __all__ = []
