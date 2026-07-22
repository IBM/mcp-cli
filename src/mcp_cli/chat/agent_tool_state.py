"""Per-agent tool state isolation.

Wraps the global get_tool_state() singleton from chuk_ai_session_manager
with a per-agent registry.  The ``"default"`` agent_id delegates to the
global singleton; other agent_ids get independent ToolStateManager instances.
"""

from __future__ import annotations

import logging

from chuk_ai_session_manager.guards import get_tool_state
from chuk_ai_session_manager.guards.manager import ToolStateManager

logger = logging.getLogger(__name__)

_registry: dict[str, ToolStateManager] = {}


def get_agent_tool_state(agent_id: str = "default") -> ToolStateManager:
    """Return a ToolStateManager scoped to the given *agent_id*.

    * ``"default"`` → delegates to the upstream global singleton
    * anything else → per-agent instance (created on first access)
    """
    if agent_id == "default":
        return get_tool_state()
    if agent_id not in _registry:
        _registry[agent_id] = ToolStateManager()
        logger.debug("Created tool state for agent %s", agent_id)
    return _registry[agent_id]


def remove_agent_tool_state(agent_id: str) -> None:
    """Remove tool state for an agent (call when the agent is stopped)."""
    _registry.pop(agent_id, None)


def _reset_registry() -> None:
    """Clear the registry — for testing only."""
    _registry.clear()
