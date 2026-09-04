"""Body state plugin for diploid-agent."""

from diploid_plugins.body.body import BodyPlugin
from diploid_plugins.body.body_mcp import BodyMcpServer, main

Plugin = BodyPlugin

__all__ = ["BodyMcpServer", "BodyPlugin", "Plugin", "main"]
