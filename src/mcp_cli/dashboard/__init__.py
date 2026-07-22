# mcp_cli/dashboard/__init__.py
"""Dashboard shell for mcp-cli.

Provides a browser-based tiled panel layout that displays MCP tool activity
and conversation state in real-time. Launched with ``mcp-cli chat --dashboard``.

Requires ``websockets`` — install with: pip install mcp-cli[dashboard]
"""

from __future__ import annotations

from mcp_cli.dashboard.bridge import DashboardBridge
from mcp_cli.dashboard.router import AgentDescriptor, AgentRouter
from mcp_cli.dashboard.server import DashboardServer

__all__ = ["AgentDescriptor", "AgentRouter", "DashboardBridge", "DashboardServer"]
