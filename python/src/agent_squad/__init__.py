from .shared import user_agent

user_agent.inject_user_agent()

try:
    from .tools.mcp_tool_provider import MCPToolProvider, MCPServerConfig
    __all__ = ["MCPToolProvider", "MCPServerConfig"]
except ImportError:
    pass