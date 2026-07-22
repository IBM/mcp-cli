# mcp_cli/agents — Multi-agent orchestration
"""Agent orchestration: config, lifecycle management, and supervisor tools."""

from mcp_cli.agents.config import AgentConfig
from mcp_cli.agents.manager import AgentManager

__all__ = ["AgentConfig", "AgentManager"]
