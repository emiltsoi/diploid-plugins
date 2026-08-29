"""Self-management plugin: exposes an MCP server for plugin lifecycle."""

from __future__ import annotations

from diploid_agent.config import McpServerConfig
from diploid_agent.plugins.base import StatePlugin


class SelfManagementPlugin(StatePlugin):
    """No prompt block; only provides the diploid-self-management MCP server."""

    def mcp_server(self) -> McpServerConfig | None:
        return self.config.mcp_server
