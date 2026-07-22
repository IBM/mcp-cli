# tests/dashboard/test_router.py
"""Unit tests for AgentRouter."""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_cli.dashboard.router import AgentDescriptor, AgentRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server():
    """Return a mock DashboardServer."""
    from mcp_cli.dashboard.server import DashboardServer

    server = MagicMock(spec=DashboardServer)
    server.broadcast = AsyncMock()
    server.send_to_client = AsyncMock()
    server.has_clients = True
    server.on_browser_message = None
    server.on_client_connected = None
    server.on_client_disconnected = None
    return server


def _make_bridge(agent_id: str = "agent-1"):
    """Return a mock DashboardBridge."""
    bridge = MagicMock()
    bridge.agent_id = agent_id
    bridge._on_browser_message = AsyncMock()
    bridge._on_client_connected = AsyncMock()
    bridge.on_client_disconnected = AsyncMock()
    return bridge


# ---------------------------------------------------------------------------
# TestRouterSingleAgent
# ---------------------------------------------------------------------------


class TestRouterSingleAgent:
    """Single-bridge router behaves identically to direct wiring."""

    @pytest.mark.asyncio
    async def test_broadcast_from_agent_delegates_to_server(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("agent-1")
        router.register_agent("agent-1", bridge)

        msg = {"type": "TEST", "payload": {}}
        await router.broadcast_from_agent("agent-1", msg)
        server.broadcast.assert_awaited_once_with(msg)

    @pytest.mark.asyncio
    async def test_browser_message_routed_to_sole_bridge(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("agent-1")
        router.register_agent("agent-1", bridge)

        msg = {"type": "USER_MESSAGE", "content": "hi"}
        await router._on_browser_message(msg)
        bridge._on_browser_message.assert_awaited_once_with(msg)

    @pytest.mark.asyncio
    async def test_browser_message_with_agent_id_routes_correctly(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("agent-1")
        router.register_agent("agent-1", bridge)

        msg = {"type": "USER_MESSAGE", "content": "hi", "agent_id": "agent-1"}
        await router._on_browser_message(msg)
        bridge._on_browser_message.assert_awaited_once_with(msg)

    @pytest.mark.asyncio
    async def test_client_connected_delegates_to_bridge(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("agent-1")
        router.register_agent("agent-1", bridge)

        ws = AsyncMock()
        await router._on_client_connected(ws)
        bridge._on_client_connected.assert_awaited_once_with(ws)

    @pytest.mark.asyncio
    async def test_client_disconnected_delegates_to_bridge(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("agent-1")
        router.register_agent("agent-1", bridge)

        await router._on_client_disconnected()
        bridge.on_client_disconnected.assert_awaited_once()

    def test_has_clients_proxies_to_server(self):
        server = _make_server()
        server.has_clients = True
        router = AgentRouter(server)
        assert router.has_clients is True
        server.has_clients = False
        assert router.has_clients is False


# ---------------------------------------------------------------------------
# TestRouterRegistration
# ---------------------------------------------------------------------------


class TestRouterRegistration:
    def test_register_agent_adds_bridge(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("a")
        router.register_agent("a", bridge)
        assert "a" in router._bridges

    def test_unregister_agent_removes_bridge(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("a")
        router.register_agent("a", bridge)
        router.unregister_agent("a")
        assert "a" not in router._bridges

    def test_register_multiple_agents(self):
        server = _make_server()
        router = AgentRouter(server)
        b1 = _make_bridge("a")
        b2 = _make_bridge("b")
        router.register_agent("a", b1)
        router.register_agent("b", b2)
        assert len(router._bridges) == 2


# ---------------------------------------------------------------------------
# TestRouterTwoAgents
# ---------------------------------------------------------------------------


class TestRouterTwoAgents:
    @pytest.mark.asyncio
    async def test_message_with_unknown_agent_id_logs_debug(self):
        server = _make_server()
        router = AgentRouter(server)
        b1 = _make_bridge("a")
        b2 = _make_bridge("b")
        router.register_agent("a", b1)
        router.register_agent("b", b2)

        msg = {"type": "USER_MESSAGE", "agent_id": "unknown"}
        await router._on_browser_message(msg)
        b1._on_browser_message.assert_not_awaited()
        b2._on_browser_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_client_connected_replays_default_bridge_only(self):
        server = _make_server()
        router = AgentRouter(server)
        b1 = _make_bridge("a")
        b2 = _make_bridge("b")
        router.register_agent("a", b1)
        router.register_agent("b", b2)

        ws = AsyncMock()
        await router._on_client_connected(ws)
        # Only the default (first registered) agent is replayed
        b1._on_client_connected.assert_awaited_once_with(ws)
        b2._on_client_connected.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_client_disconnected_notifies_all_bridges(self):
        server = _make_server()
        router = AgentRouter(server)
        b1 = _make_bridge("a")
        b2 = _make_bridge("b")
        router.register_agent("a", b1)
        router.register_agent("b", b2)

        await router._on_client_disconnected()
        b1.on_client_disconnected.assert_awaited_once()
        b2.on_client_disconnected.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestAgentDescriptor
# ---------------------------------------------------------------------------


class TestAgentDescriptor:
    def test_defaults(self):
        desc = AgentDescriptor(agent_id="a1", name="Agent One")
        assert desc.agent_id == "a1"
        assert desc.name == "Agent One"
        assert desc.status == "active"
        assert desc.role == ""
        assert desc.model == ""
        assert desc.parent_agent_id is None
        assert desc.tool_count == 0
        assert desc.message_count == 0

    def test_asdict_roundtrip(self):
        desc = AgentDescriptor(agent_id="x", name="X", role="worker", model="gpt-4")
        d = asdict(desc)
        assert d["agent_id"] == "x"
        assert d["role"] == "worker"
        assert d["model"] == "gpt-4"


# ---------------------------------------------------------------------------
# TestRouterFocusTracking
# ---------------------------------------------------------------------------


class TestRouterFocusTracking:
    @pytest.mark.asyncio
    async def test_focus_agent_stores_focus(self):
        server = _make_server()
        router = AgentRouter(server)
        b1 = _make_bridge("a")
        b2 = _make_bridge("b")
        router.register_agent("a", b1)
        router.register_agent("b", b2)

        ws = AsyncMock()
        msg = {"type": "FOCUS_AGENT", "agent_id": "b"}
        await router._on_browser_message(msg, ws)
        assert router._client_focus[ws] == "b"

    @pytest.mark.asyncio
    async def test_focus_agent_replays_only_focused_bridge(self):
        server = _make_server()
        router = AgentRouter(server)
        b1 = _make_bridge("a")
        b2 = _make_bridge("b")
        router.register_agent("a", b1)
        router.register_agent("b", b2)

        ws = AsyncMock()
        msg = {"type": "FOCUS_AGENT", "agent_id": "b"}
        await router._on_browser_message(msg, ws)
        b2._on_client_connected.assert_awaited_once_with(ws)
        b1._on_client_connected.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_browser_message_uses_focus_when_no_agent_id(self):
        server = _make_server()
        router = AgentRouter(server)
        b1 = _make_bridge("a")
        b2 = _make_bridge("b")
        router.register_agent("a", b1)
        router.register_agent("b", b2)

        ws = AsyncMock()
        # Set focus to "b"
        router._client_focus[ws] = "b"
        msg = {"type": "USER_MESSAGE", "content": "hi"}
        await router._on_browser_message(msg, ws)
        b2._on_browser_message.assert_awaited_once_with(msg)
        b1._on_browser_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestRouterAgentList
# ---------------------------------------------------------------------------


class TestRouterAgentList:
    @pytest.mark.asyncio
    async def test_client_connected_sends_agent_list(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("a")
        router.register_agent("a", bridge)

        ws = AsyncMock()
        await router._on_client_connected(ws)
        # send_to_client is called with AGENT_LIST envelope
        server.send_to_client.assert_awaited()
        call_args = server.send_to_client.call_args_list[0]
        msg = call_args[0][1]  # second positional arg
        assert msg["type"] == "AGENT_LIST"
        assert len(msg["payload"]["agents"]) == 1
        assert msg["payload"]["agents"][0]["agent_id"] == "a"

    @pytest.mark.asyncio
    async def test_request_agent_list_not_forwarded_to_bridge(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("a")
        router.register_agent("a", bridge)

        ws = AsyncMock()
        msg = {"type": "REQUEST_AGENT_LIST"}
        await router._on_browser_message(msg, ws)
        bridge._on_browser_message.assert_not_awaited()
        server.send_to_client.assert_awaited()

    @pytest.mark.asyncio
    async def test_client_connected_sets_default_focus(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("a")
        router.register_agent("a", bridge)

        ws = AsyncMock()
        await router._on_client_connected(ws)
        assert router._client_focus[ws] == "a"


# ---------------------------------------------------------------------------
# TestRouterDescriptorBackwardCompat
# ---------------------------------------------------------------------------


class TestRouterDescriptorBackwardCompat:
    def test_register_without_descriptor_creates_default(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("test-agent")
        router.register_agent("test-agent", bridge)
        desc = router._agent_descriptors["test-agent"]
        assert desc.agent_id == "test-agent"
        assert desc.name == "test-agent"
        assert desc.status == "active"

    def test_register_with_custom_descriptor(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("a")
        desc = AgentDescriptor(agent_id="a", name="Custom", role="planner")
        router.register_agent("a", bridge, descriptor=desc)
        assert router._agent_descriptors["a"].name == "Custom"
        assert router._agent_descriptors["a"].role == "planner"


# ---------------------------------------------------------------------------
# TestBroadcastGlobal
# ---------------------------------------------------------------------------


class TestBroadcastGlobal:
    @pytest.mark.asyncio
    async def test_broadcast_global_delegates_to_server(self):
        server = _make_server()
        router = AgentRouter(server)
        msg = {"type": "AGENT_LIST", "payload": {}}
        await router.broadcast_global(msg)
        server.broadcast.assert_awaited_once_with(msg)


# ---------------------------------------------------------------------------
# TestUpdateAgentStatus
# ---------------------------------------------------------------------------


class TestUpdateAgentStatus:
    @pytest.mark.asyncio
    async def test_update_status_changes_descriptor_and_broadcasts(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("a")
        router.register_agent("a", bridge)

        await router.update_agent_status("a", "paused")
        assert router._agent_descriptors["a"].status == "paused"
        server.broadcast.assert_awaited_once()
        msg = server.broadcast.call_args[0][0]
        assert msg["type"] == "AGENT_STATUS"
        assert msg["payload"]["agent_id"] == "a"
        assert msg["payload"]["status"] == "paused"

    @pytest.mark.asyncio
    async def test_update_status_unknown_agent_does_nothing(self):
        server = _make_server()
        router = AgentRouter(server)
        await router.update_agent_status("nonexistent", "failed")
        server.broadcast.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestRouterAgentMessage
# ---------------------------------------------------------------------------


class TestRouterAgentMessage:
    @pytest.mark.asyncio
    async def test_agent_message_forwarded_to_target(self):
        server = _make_server()
        router = AgentRouter(server)
        b1 = _make_bridge("a")
        b2 = _make_bridge("b")
        router.register_agent("a", b1)
        router.register_agent("b", b2)

        msg = {
            "type": "AGENT_MESSAGE",
            "from_agent": "a",
            "to_agent": "b",
            "content": "hello",
        }
        await router._on_browser_message(msg)
        b2._on_browser_message.assert_awaited_once()
        call_msg = b2._on_browser_message.call_args[0][0]
        assert call_msg["type"] == "USER_MESSAGE"
        assert "hello" in call_msg["content"]
        b1._on_browser_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_agent_message_to_unknown_is_noop(self):
        server = _make_server()
        router = AgentRouter(server)
        b1 = _make_bridge("a")
        router.register_agent("a", b1)

        msg = {
            "type": "AGENT_MESSAGE",
            "from_agent": "a",
            "to_agent": "nonexistent",
            "content": "hello",
        }
        await router._on_browser_message(msg)
        b1._on_browser_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestRouterSubscriptions
# ---------------------------------------------------------------------------


class TestRouterSubscriptions:
    @pytest.mark.asyncio
    async def test_subscribe_stores_subscriptions(self):
        server = _make_server()
        router = AgentRouter(server)
        ws = AsyncMock()
        msg = {"type": "SUBSCRIBE", "agents": ["a", "b"], "global": False}
        await router._on_browser_message(msg, ws)
        assert router._client_subscriptions[ws] == {"a", "b"}

    @pytest.mark.asyncio
    async def test_subscribe_global_adds_wildcard(self):
        server = _make_server()
        router = AgentRouter(server)
        ws = AsyncMock()
        msg = {"type": "SUBSCRIBE", "agents": ["a"], "global": True}
        await router._on_browser_message(msg, ws)
        assert "*" in router._client_subscriptions[ws]
        assert "a" in router._client_subscriptions[ws]

    @pytest.mark.asyncio
    async def test_broadcast_from_agent_filters_by_subscription(self):
        server = _make_server()
        router = AgentRouter(server)
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        # ws1 subscribes to "a" only, ws2 subscribes to all
        router._client_subscriptions[ws1] = {"a"}
        router._client_subscriptions[ws2] = {"*"}

        msg = {"type": "TOOL_RESULT", "payload": {}}
        await router.broadcast_from_agent("b", msg)
        # ws1 is NOT subscribed to "b", so only ws2 should receive
        calls = server.send_to_client.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] is ws2

    @pytest.mark.asyncio
    async def test_broadcast_from_agent_no_subscriptions_broadcasts_all(self):
        server = _make_server()
        router = AgentRouter(server)
        # No subscriptions configured — should broadcast to all
        msg = {"type": "TOOL_RESULT", "payload": {}}
        await router.broadcast_from_agent("a", msg)
        server.broadcast.assert_awaited_once_with(msg)

    @pytest.mark.asyncio
    async def test_client_connected_sets_default_subscription(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("a")
        router.register_agent("a", bridge)

        ws = AsyncMock()
        await router._on_client_connected(ws)
        assert router._client_subscriptions[ws] == {"*"}

    @pytest.mark.asyncio
    async def test_client_disconnected_cleans_up_state(self):
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("a")
        router.register_agent("a", bridge)

        ws = AsyncMock()
        await router._on_client_connected(ws)
        assert ws in router._client_focus
        assert ws in router._client_subscriptions

        await router._on_client_disconnected(ws)
        assert ws not in router._client_focus
        assert ws not in router._client_subscriptions

    @pytest.mark.asyncio
    async def test_client_disconnected_without_ws_still_works(self):
        """Backward compat: calling with no ws arg doesn't crash."""
        server = _make_server()
        router = AgentRouter(server)
        bridge = _make_bridge("a")
        router.register_agent("a", bridge)

        await router._on_client_disconnected()
        bridge.on_client_disconnected.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_subscribe_replaces_previous(self):
        server = _make_server()
        router = AgentRouter(server)
        ws = AsyncMock()
        msg1 = {"type": "SUBSCRIBE", "agents": ["a", "b"], "global": False}
        await router._on_browser_message(msg1, ws)
        assert router._client_subscriptions[ws] == {"a", "b"}

        msg2 = {"type": "SUBSCRIBE", "agents": ["c"], "global": True}
        await router._on_browser_message(msg2, ws)
        assert router._client_subscriptions[ws] == {"c", "*"}


# ---------------------------------------------------------------------------
# TestRouterFocusEdgeCases
# ---------------------------------------------------------------------------


class TestRouterFocusEdgeCases:
    @pytest.mark.asyncio
    async def test_focus_nonexistent_agent_no_crash(self):
        """Focusing a nonexistent agent should not crash."""
        server = _make_server()
        router = AgentRouter(server)
        ws = MagicMock()
        # Focus an agent that doesn't exist
        await router._handle_focus_agent({"agent_id": "nonexistent"}, ws)
        # Should store focus but not crash
        assert router._client_focus.get(ws) == "nonexistent"

    @pytest.mark.asyncio
    async def test_focus_switch_multiple_times(self):
        """Client can switch focus multiple times."""
        server = _make_server()
        router = AgentRouter(server)
        bridge_a = _make_bridge("a")
        bridge_b = _make_bridge("b")
        router.register_agent("a", bridge_a)
        router.register_agent("b", bridge_b)
        ws = MagicMock()

        await router._handle_focus_agent({"agent_id": "a"}, ws)
        assert router._client_focus[ws] == "a"

        await router._handle_focus_agent({"agent_id": "b"}, ws)
        assert router._client_focus[ws] == "b"
