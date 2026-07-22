# mcp_cli/agents/config.py
"""Agent configuration for per-agent tool/server restrictions."""

from __future__ import annotations

from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration for a spawned agent.

    Controls identity, model selection, tool access, and parent relationship.
    """

    agent_id: str
    name: str = ""
    role: str = ""
    model: str | None = None
    provider: str | None = None
    system_prompt: str | None = None
    allowed_tools: list[str] | None = None  # None = all tools
    denied_tools: list[str] | None = None  # explicit blocklist
    allowed_servers: list[str] | None = None  # None = all servers
    tool_timeout_override: float | None = None
    auto_approve_tools: list[str] | None = None  # skip confirmation
    parent_agent_id: str | None = None
    initial_prompt: str = ""
