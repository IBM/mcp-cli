# tests/agents/test_manager.py
"""Unit tests for AgentManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_cli.agents.config import AgentConfig
from mcp_cli.agents.headless_ui import HeadlessUIManager
from mcp_cli.agents.manager import MAX_AGENTS, AgentManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router():
    from mcp_cli.dashboard.router import AgentRouter

    server = MagicMock()
    server.broadcast = AsyncMock()
    server.send_to_client = AsyncMock()
    server.has_clients = False
    server.on_browser_message = None
    server.on_client_connected = None
    server.on_client_disconnected = None
    return AgentRouter(server)


def _make_tool_manager():
    tm = MagicMock()
    tm.list_tools = MagicMock(return_value=[])
    tm.execute_tool = AsyncMock()
    tm.close = AsyncMock()
    return tm


def _make_model_manager():
    mm = MagicMock()
    mm.active_provider = "test"
    mm.active_model = "test-model"
    return mm


# ---------------------------------------------------------------------------
# TestHeadlessUIManager
# ---------------------------------------------------------------------------


class TestHeadlessUIManager:
    def test_defaults(self):
        ui = HeadlessUIManager(agent_id="a1")
        assert ui.agent_id == "a1"
        assert ui.verbose_mode is False
        assert ui.is_streaming_response is False

    @pytest.mark.asyncio
    async def test_auto_approve(self):
        ui = HeadlessUIManager()
        assert await ui.do_confirm_tool_execution("tool", {}) is True

    @pytest.mark.asyncio
    async def test_start_stop_streaming(self):
        ui = HeadlessUIManager()
        await ui.start_streaming_response()
        await ui.stop_streaming_response()
        assert ui.streaming_handler is None

    @pytest.mark.asyncio
    async def test_tool_lifecycle(self):
        ui = HeadlessUIManager()
        ui.print_tool_call("test_tool", {"a": 1})
        await ui.start_tool_execution("test_tool", {"a": 1})
        await ui.finish_tool_execution(result="ok", success=True)
        await ui.finish_tool_calls()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        ui = HeadlessUIManager()
        await ui.cleanup()


# ---------------------------------------------------------------------------
# TestAgentManagerBasic
# ---------------------------------------------------------------------------


class TestAgentManagerBasic:
    def test_init(self):
        tm = _make_tool_manager()
        router = _make_router()
        mgr = AgentManager(tm, router)
        assert mgr.list_agents() == []
        assert mgr.list_artifacts() == []

    @pytest.mark.asyncio
    async def test_spawn_duplicate_raises(self):
        tm = _make_tool_manager()
        router = _make_router()
        mgr = AgentManager(tm, router)

        # Mock spawn internals — patch at the source modules since
        # manager.py uses lazy imports inside spawn_agent()
        mock_ctx = MagicMock()
        mock_ctx.agent_id = "a"
        mock_ctx.initialize = AsyncMock(return_value=True)
        mock_ctx.openai_tools = []
        mock_ctx.conversation_history = []
        mock_ctx.exit_requested = True  # stop loop immediately
        mock_ctx.dashboard_bridge = None
        mock_ctx._system_prompt = ""
        mock_ctx._system_prompt_dirty = False

        with (
            patch(
                "mcp_cli.chat.chat_context.ChatContext.create",
                return_value=mock_ctx,
            ),
            patch("mcp_cli.dashboard.bridge.DashboardBridge") as MockBridge,
        ):
            mock_bridge = MagicMock()
            mock_bridge.set_context = MagicMock()
            mock_bridge.set_input_queue = MagicMock()
            MockBridge.return_value = mock_bridge

            await mgr.spawn_agent(AgentConfig(agent_id="a", name="A"))
            with pytest.raises(ValueError, match="already exists"):
                await mgr.spawn_agent(AgentConfig(agent_id="a", name="A2"))

            await mgr.stop_all()

    @pytest.mark.asyncio
    async def test_stop_nonexistent(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        assert await mgr.stop_agent("no-such") is False


# ---------------------------------------------------------------------------
# TestAgentManagerArtifacts
# ---------------------------------------------------------------------------


class TestAgentManagerArtifacts:
    def test_publish_and_get(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        mgr.publish_artifact("agent-a", "results", {"data": [1, 2, 3]})
        assert mgr.get_artifact("results") == {"data": [1, 2, 3]}

    def test_get_nonexistent(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        assert mgr.get_artifact("nope") is None

    def test_list_artifacts(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        mgr.publish_artifact("a", "art1", "content1")
        mgr.publish_artifact("b", "art2", "content2")
        arts = mgr.list_artifacts()
        assert len(arts) == 2
        ids = {a["artifact_id"] for a in arts}
        assert ids == {"art1", "art2"}


# ---------------------------------------------------------------------------
# TestAgentManagerMessaging
# ---------------------------------------------------------------------------


class TestAgentManagerMessaging:
    @pytest.mark.asyncio
    async def test_send_to_nonexistent_returns_false(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        assert await mgr.send_message("a", "b", "hello") is False

    @pytest.mark.asyncio
    async def test_get_messages_empty(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        assert await mgr.get_messages("nonexistent") == []


# ---------------------------------------------------------------------------
# TestAgentManagerStatus
# ---------------------------------------------------------------------------


class TestAgentManagerStatus:
    def test_get_status_nonexistent(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        assert mgr.get_agent_status("nope") is None


# ---------------------------------------------------------------------------
# TestAgentManagerMaxAgents
# ---------------------------------------------------------------------------


class TestAgentManagerMaxAgents:
    def test_max_agents_constant(self):
        assert MAX_AGENTS == 10


# ---------------------------------------------------------------------------
# Helper: spawn a mock agent into the manager
# ---------------------------------------------------------------------------


async def _spawn_mock_agent(mgr, agent_id="a", name="A", **kwargs):
    """Spawn a mock agent, returns the manager."""
    mock_ctx = MagicMock()
    mock_ctx.agent_id = agent_id
    mock_ctx.initialize = AsyncMock(return_value=True)
    mock_ctx.openai_tools = []
    mock_ctx.conversation_history = []
    mock_ctx.exit_requested = True  # stop loop immediately
    mock_ctx.dashboard_bridge = None
    mock_ctx._system_prompt = ""
    mock_ctx._system_prompt_dirty = False

    with (
        patch(
            "mcp_cli.chat.chat_context.ChatContext.create",
            return_value=mock_ctx,
        ),
        patch("mcp_cli.dashboard.bridge.DashboardBridge") as MockBridge,
    ):
        mock_bridge = MagicMock()
        mock_bridge.set_context = MagicMock()
        mock_bridge.set_input_queue = MagicMock()
        MockBridge.return_value = mock_bridge

        cfg = AgentConfig(agent_id=agent_id, name=name, **kwargs)
        await mgr.spawn_agent(cfg)

    return mgr


# ---------------------------------------------------------------------------
# TestAgentManagerLifecycle (with spawned agent)
# ---------------------------------------------------------------------------


class TestAgentManagerLifecycle:
    @pytest.mark.asyncio
    async def test_spawn_and_list(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        await _spawn_mock_agent(mgr, "x", "Agent X", role="worker")
        agents = mgr.list_agents()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == "x"
        assert agents[0]["name"] == "Agent X"
        await mgr.stop_all()

    @pytest.mark.asyncio
    async def test_stop_agent_returns_true(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        await _spawn_mock_agent(mgr)
        assert await mgr.stop_agent("a") is True
        assert mgr.list_agents() == []

    @pytest.mark.asyncio
    async def test_get_agent_snapshot(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        await _spawn_mock_agent(mgr, "s1", "Snap Agent")
        snap = mgr.get_agent_snapshot("s1")
        assert snap is not None
        assert snap["config"].agent_id == "s1"
        assert snap["context"] is not None
        await mgr.stop_all()

    @pytest.mark.asyncio
    async def test_get_agent_snapshot_nonexistent(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        assert mgr.get_agent_snapshot("nope") is None


# ---------------------------------------------------------------------------
# TestAgentManagerMessagingWithAgent
# ---------------------------------------------------------------------------


class TestAgentManagerMessagingWithAgent:
    @pytest.mark.asyncio
    async def test_send_message_injects_into_queue(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        await _spawn_mock_agent(mgr, "target", "Target")

        result = await mgr.send_message("sender", "target", "hello world")
        assert result is True

        # Verify the message is in the agent's input queue
        handle = mgr._agents["target"]
        msg = handle.input_queue.get_nowait()
        assert "[Message from sender]" in msg
        assert "hello world" in msg
        await mgr.stop_all()


# ---------------------------------------------------------------------------
# TestAgentManagerWait
# ---------------------------------------------------------------------------


class TestAgentManagerWait:
    @pytest.mark.asyncio
    async def test_wait_unknown_returns_error(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        result = await mgr.wait_agent("nonexistent")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        mgr = AgentManager(_make_tool_manager(), _make_router())
        await _spawn_mock_agent(mgr, "w1", "Waiter")
        # The done_event is not set by mock, so wait should timeout
        # But the agent loop exits immediately (exit_requested=True),
        # so let's just test with a very short timeout
        # The done_event might already be set by the loop finishing
        result = await mgr.wait_agent("w1", timeout=0.01)
        assert result["agent_id"] == "w1"
        assert result["status"] in ("completed", "timeout")
        await mgr.stop_all()
