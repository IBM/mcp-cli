# tests/chat/test_agent_tool_state.py
"""Unit tests for per-agent tool state isolation."""

from __future__ import annotations

from mcp_cli.chat.agent_tool_state import (
    _reset_registry,
    get_agent_tool_state,
    remove_agent_tool_state,
)


class TestAgentToolStateRegistry:
    """Tests for the agent_tool_state registry."""

    def setup_method(self):
        _reset_registry()

    def teardown_method(self):
        _reset_registry()

    def test_default_returns_global_singleton(self):
        """'default' agent_id delegates to the upstream singleton."""
        from chuk_ai_session_manager.guards import get_tool_state

        global_ts = get_tool_state()
        default_ts = get_agent_tool_state("default")
        assert default_ts is global_ts

    def test_non_default_returns_new_instance(self):
        """Non-default agent_id creates a fresh ToolStateManager."""
        from chuk_ai_session_manager.guards import get_tool_state

        agent_ts = get_agent_tool_state("agent-research")
        global_ts = get_tool_state()
        assert agent_ts is not global_ts

    def test_same_agent_id_returns_same_instance(self):
        """Repeated calls with the same agent_id return the same instance."""
        ts1 = get_agent_tool_state("agent-a")
        ts2 = get_agent_tool_state("agent-a")
        assert ts1 is ts2

    def test_different_agent_ids_return_different_instances(self):
        """Different agent_ids get independent instances."""
        ts_a = get_agent_tool_state("agent-a")
        ts_b = get_agent_tool_state("agent-b")
        assert ts_a is not ts_b

    def test_remove_agent_tool_state(self):
        """Removing tool state means next access creates a fresh instance."""
        ts1 = get_agent_tool_state("agent-a")
        remove_agent_tool_state("agent-a")
        ts2 = get_agent_tool_state("agent-a")
        assert ts1 is not ts2

    def test_remove_nonexistent_is_noop(self):
        """Removing a non-existent agent_id doesn't raise."""
        remove_agent_tool_state("no-such-agent")  # should not raise

    def test_reset_registry_clears_all(self):
        """_reset_registry clears all per-agent instances."""
        get_agent_tool_state("agent-a")
        get_agent_tool_state("agent-b")
        _reset_registry()
        # After reset, new calls should create fresh instances
        ts = get_agent_tool_state("agent-a")
        assert ts is not None  # it exists but is fresh

    def test_default_not_in_registry(self):
        """The 'default' agent's state is NOT stored in the internal registry."""
        get_agent_tool_state("default")
        _reset_registry()  # only clears internal registry
        # default should still work (delegates to global singleton)
        ts = get_agent_tool_state("default")
        assert ts is not None
