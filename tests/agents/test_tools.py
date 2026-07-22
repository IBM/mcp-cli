# tests/agents/test_tools.py
"""Unit tests for agent orchestration tool definitions and handler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_cli.agents.tools import (
    _AGENT_TOOL_NAMES,
    get_agent_tools_as_dicts,
    handle_agent_tool,
)


# ---------------------------------------------------------------------------
# TestToolDefinitions
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    def test_tool_count(self):
        tools = get_agent_tools_as_dicts()
        assert len(tools) == 6

    def test_tool_names_match_frozenset(self):
        tools = get_agent_tools_as_dicts()
        names = {t["function"]["name"] for t in tools}
        assert names == _AGENT_TOOL_NAMES

    def test_all_have_function_key(self):
        tools = get_agent_tools_as_dicts()
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t["function"]
            assert "description" in t["function"]
            assert "parameters" in t["function"]

    def test_spawn_has_required_fields(self):
        tools = get_agent_tools_as_dicts()
        spawn = next(t for t in tools if t["function"]["name"] == "agent_spawn")
        required = spawn["function"]["parameters"]["required"]
        assert "name" in required
        assert "initial_prompt" in required


# ---------------------------------------------------------------------------
# TestHandleAgentTool
# ---------------------------------------------------------------------------


class TestHandleAgentTool:
    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        mgr = MagicMock()
        result = await handle_agent_tool("unknown_tool", {}, mgr)
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_agent_stop(self):
        mgr = MagicMock()
        mgr.stop_agent = AsyncMock(return_value=True)
        result = await handle_agent_tool("agent_stop", {"agent_id": "a1"}, mgr)
        data = json.loads(result)
        assert data["success"] is True
        mgr.stop_agent.assert_awaited_once_with("a1")

    @pytest.mark.asyncio
    async def test_agent_stop_missing_id(self):
        mgr = MagicMock()
        result = await handle_agent_tool("agent_stop", {}, mgr)
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_agent_message(self):
        mgr = MagicMock()
        mgr.send_message = AsyncMock(return_value=True)
        result = await handle_agent_tool(
            "agent_message",
            {"agent_id": "a1", "content": "hello"},
            mgr,
            caller_agent_id="supervisor",
        )
        data = json.loads(result)
        assert data["success"] is True
        mgr.send_message.assert_awaited_once_with("supervisor", "a1", "hello")

    @pytest.mark.asyncio
    async def test_agent_message_missing_fields(self):
        mgr = MagicMock()
        result = await handle_agent_tool("agent_message", {}, mgr)
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_agent_wait(self):
        mgr = MagicMock()
        mgr.wait_agent = AsyncMock(
            return_value={"agent_id": "a1", "status": "completed", "summary": "done"}
        )
        result = await handle_agent_tool("agent_wait", {"agent_id": "a1"}, mgr)
        data = json.loads(result)
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_agent_status(self):
        mgr = MagicMock()
        mgr.get_agent_status.return_value = {
            "agent_id": "a1",
            "status": "active",
            "name": "Test",
        }
        result = await handle_agent_tool("agent_status", {"agent_id": "a1"}, mgr)
        data = json.loads(result)
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_agent_status_unknown(self):
        mgr = MagicMock()
        mgr.get_agent_status.return_value = None
        result = await handle_agent_tool("agent_status", {"agent_id": "nope"}, mgr)
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_agent_list(self):
        mgr = MagicMock()
        mgr.list_agents.return_value = [
            {"agent_id": "a", "status": "active"},
            {"agent_id": "b", "status": "completed"},
        ]
        result = await handle_agent_tool("agent_list", {}, mgr)
        data = json.loads(result)
        assert len(data["agents"]) == 2

    @pytest.mark.asyncio
    async def test_agent_spawn(self):
        mgr = MagicMock()
        mgr.spawn_agent = AsyncMock(return_value="agent-research")
        result = await handle_agent_tool(
            "agent_spawn",
            {"name": "Research", "initial_prompt": "Find docs"},
            mgr,
            caller_agent_id="main",
        )
        data = json.loads(result)
        assert data["success"] is True
        assert data["agent_id"] == "agent-research"
        # Verify spawn was called with an AgentConfig
        mgr.spawn_agent.assert_awaited_once()
        config = mgr.spawn_agent.call_args[0][0]
        assert config.name == "Research"
        assert config.parent_agent_id == "main"

    @pytest.mark.asyncio
    async def test_handler_catches_exceptions(self):
        mgr = MagicMock()
        mgr.stop_agent = AsyncMock(side_effect=RuntimeError("boom"))
        result = await handle_agent_tool("agent_stop", {"agent_id": "a1"}, mgr)
        data = json.loads(result)
        assert "error" in data
        assert "boom" in data["error"]
