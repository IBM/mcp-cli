# tests/dashboard/test_bridge.py
"""Unit tests for DashboardBridge.

These tests mock DashboardServer so no real WebSocket is needed.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge():
    """Return a (bridge, mock_server) pair."""
    from mcp_cli.dashboard.bridge import DashboardBridge
    from mcp_cli.dashboard.server import DashboardServer

    server = MagicMock(spec=DashboardServer)
    server.broadcast = AsyncMock()
    server.on_browser_message = None
    server.on_client_connected = None
    bridge = DashboardBridge(server)
    return bridge, server


# ---------------------------------------------------------------------------
# _serialise
# ---------------------------------------------------------------------------


class TestSerialise:
    def test_primitives(self):
        from mcp_cli.dashboard.bridge import DashboardBridge

        assert DashboardBridge._serialise(None) is None
        assert DashboardBridge._serialise(True) is True
        assert DashboardBridge._serialise(42) == 42
        assert DashboardBridge._serialise(3.14) == 3.14
        assert DashboardBridge._serialise("hello") == "hello"

    def test_list(self):
        from mcp_cli.dashboard.bridge import DashboardBridge

        assert DashboardBridge._serialise([1, "two", None]) == [1, "two", None]

    def test_dict(self):
        from mcp_cli.dashboard.bridge import DashboardBridge

        result = DashboardBridge._serialise({"a": 1, "b": [2, 3]})
        assert result == {"a": 1, "b": [2, 3]}

    def test_object_with_to_dict(self):
        from mcp_cli.dashboard.bridge import DashboardBridge

        obj = MagicMock()
        obj.to_dict.return_value = {"key": "val"}
        result = DashboardBridge._serialise(obj)
        assert result == {"key": "val"}

    def test_object_with_model_dump(self):
        from mcp_cli.dashboard.bridge import DashboardBridge

        obj = MagicMock(spec=[])
        del obj.to_dict
        obj.model_dump = MagicMock(return_value={"x": 1})
        result = DashboardBridge._serialise(obj)
        assert result == {"x": 1}

    def test_unknown_object_stringified(self):
        from mcp_cli.dashboard.bridge import DashboardBridge

        class Custom:
            def __str__(self):
                return "custom-repr"

        result = DashboardBridge._serialise(Custom())
        assert result == "custom-repr"


# ---------------------------------------------------------------------------
# on_tool_result
# ---------------------------------------------------------------------------


class TestOnToolResult:
    @pytest.mark.asyncio
    async def test_broadcast_called(self):
        bridge, server = _make_bridge()
        await bridge.on_tool_result(
            tool_name="test_tool",
            server_name="test_server",
            result={"rows": 3},
            success=True,
        )
        server.broadcast.assert_awaited_once()
        msg = server.broadcast.call_args[0][0]
        assert msg["type"] == "TOOL_RESULT"
        assert msg["payload"]["tool_name"] == "test_tool"
        assert msg["payload"]["success"] is True

    @pytest.mark.asyncio
    async def test_error_result(self):
        bridge, server = _make_bridge()
        await bridge.on_tool_result(
            tool_name="bad_tool",
            server_name="srv",
            result=None,
            success=False,
            error="timeout",
        )
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["success"] is False
        assert msg["payload"]["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_meta_ui_included_in_payload(self):
        bridge, server = _make_bridge()
        meta_ui = {"view": "custom:stats"}
        await bridge.on_tool_result(
            tool_name="stats",
            server_name="srv",
            result=None,
            success=True,
            meta_ui=meta_ui,
        )
        msg = server.broadcast.call_args[0][0]
        assert "meta_ui" in msg["payload"]

    @pytest.mark.asyncio
    async def test_view_discovered_from_meta_ui(self):
        bridge, server = _make_bridge()
        meta_ui = {"view": "custom:stats", "name": "Stats Dashboard"}
        await bridge.on_tool_result(
            tool_name="get_stats",
            server_name="analytics",
            result=None,
            success=True,
            meta_ui=meta_ui,
        )
        # VIEW_REGISTRY broadcast should have happened (broadcast called twice total)
        calls = server.broadcast.call_args_list
        types = [c[0][0]["type"] for c in calls]
        assert "VIEW_REGISTRY" in types

    @pytest.mark.asyncio
    async def test_view_discovered_only_once(self):
        bridge, server = _make_bridge()
        meta_ui = {"view": "custom:stats"}
        await bridge.on_tool_result("t", "s", None, True, meta_ui=meta_ui)
        await bridge.on_tool_result("t", "s", None, True, meta_ui=meta_ui)
        registry_broadcasts = [
            c
            for c in server.broadcast.call_args_list
            if c[0][0].get("type") == "VIEW_REGISTRY"
        ]
        assert len(registry_broadcasts) == 1


# ---------------------------------------------------------------------------
# on_agent_state
# ---------------------------------------------------------------------------


class TestOnAgentState:
    @pytest.mark.asyncio
    async def test_thinking_state(self):
        bridge, server = _make_bridge()
        await bridge.on_agent_state("thinking", None, turn_number=1, tokens_used=500)
        msg = server.broadcast.call_args[0][0]
        assert msg["type"] == "AGENT_STATE"
        assert msg["payload"]["status"] == "thinking"
        assert msg["payload"]["turn_number"] == 1
        assert msg["payload"]["tokens_used"] == 500

    @pytest.mark.asyncio
    async def test_tool_calling_state(self):
        bridge, server = _make_bridge()
        await bridge.on_agent_state("tool_calling", "query_db")
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["current_tool"] == "query_db"


# ---------------------------------------------------------------------------
# on_message / on_token
# ---------------------------------------------------------------------------


class TestOnMessage:
    @pytest.mark.asyncio
    async def test_user_message(self):
        bridge, server = _make_bridge()
        await bridge.on_message("user", "hello world")
        msg = server.broadcast.call_args[0][0]
        assert msg["type"] == "CONVERSATION_MESSAGE"
        assert msg["payload"]["role"] == "user"
        assert msg["payload"]["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_assistant_message_streaming(self):
        bridge, server = _make_bridge()
        await bridge.on_message("assistant", "some text", streaming=True)
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["streaming"] is True

    @pytest.mark.asyncio
    async def test_token(self):
        bridge, server = _make_bridge()
        await bridge.on_token("Hello")
        msg = server.broadcast.call_args[0][0]
        assert msg["type"] == "CONVERSATION_TOKEN"
        assert msg["payload"]["token"] == "Hello"
        assert msg["payload"]["done"] is False

    @pytest.mark.asyncio
    async def test_token_done(self):
        bridge, server = _make_bridge()
        await bridge.on_token("", done=True)
        msg = server.broadcast.call_args[0][0]
        assert msg["payload"]["done"] is True


# ---------------------------------------------------------------------------
# Input queue / browser messages
# ---------------------------------------------------------------------------


class TestInputQueue:
    @pytest.mark.asyncio
    async def test_user_message_goes_to_queue(self):
        bridge, server = _make_bridge()
        q: asyncio.Queue[str] = asyncio.Queue()
        bridge.set_input_queue(q)
        await bridge._on_browser_message({"type": "USER_MESSAGE", "content": "ping"})
        assert q.get_nowait() == "ping"

    @pytest.mark.asyncio
    async def test_user_command_goes_to_queue(self):
        bridge, server = _make_bridge()
        q: asyncio.Queue[str] = asyncio.Queue()
        bridge.set_input_queue(q)
        await bridge._on_browser_message({"type": "USER_COMMAND", "command": "/tools"})
        assert q.get_nowait() == "/tools"

    @pytest.mark.asyncio
    async def test_no_queue_registered_logs_debug(self):
        bridge, server = _make_bridge()
        # Should not raise even without queue
        await bridge._on_browser_message({"type": "USER_MESSAGE", "content": "hi"})


# ---------------------------------------------------------------------------
# _on_client_connected
# ---------------------------------------------------------------------------


class TestOnClientConnected:
    @pytest.mark.asyncio
    async def test_sends_registry_when_views_exist(self):
        bridge, server = _make_bridge()
        bridge._view_registry = [{"id": "custom:stats", "name": "Stats"}]
        ws = AsyncMock()
        await bridge._on_client_connected(ws)
        ws.send.assert_awaited_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "VIEW_REGISTRY"
        assert len(sent["payload"]["views"]) == 1

    @pytest.mark.asyncio
    async def test_no_send_when_registry_empty(self):
        bridge, server = _make_bridge()
        ws = AsyncMock()
        await bridge._on_client_connected(ws)
        ws.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# REQUEST_TOOL
# ---------------------------------------------------------------------------


class TestRequestTool:
    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        bridge, server = _make_bridge()
        calls: list = []

        async def cb(tool_name, arguments):
            calls.append((tool_name, arguments))

        bridge.set_tool_call_callback(cb)
        await bridge._on_browser_message(
            {
                "type": "REQUEST_TOOL",
                "tool_name": "query_db",
                "arguments": {"sql": "SELECT 1"},
            }
        )
        assert calls == [("query_db", {"sql": "SELECT 1"})]

    @pytest.mark.asyncio
    async def test_no_callback_logs_debug(self):
        bridge, server = _make_bridge()
        # No callback registered — should not raise
        await bridge._on_browser_message(
            {"type": "REQUEST_TOOL", "tool_name": "query_db", "arguments": {}}
        )
        server.broadcast.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_tool_name_logs_debug(self):
        bridge, server = _make_bridge()
        calls: list = []

        async def cb(tool_name, arguments):  # pragma: no cover
            calls.append(tool_name)

        bridge.set_tool_call_callback(cb)
        await bridge._on_browser_message({"type": "REQUEST_TOOL"})
        assert calls == []

    @pytest.mark.asyncio
    async def test_callback_error_broadcasts_failure(self):
        bridge, server = _make_bridge()

        async def bad_cb(tool_name, arguments):
            raise RuntimeError("tool exploded")

        bridge.set_tool_call_callback(bad_cb)
        await bridge._on_browser_message(
            {"type": "REQUEST_TOOL", "tool_name": "bad_tool", "arguments": {}}
        )
        # Should have broadcast a TOOL_RESULT with success=False
        calls = [c[0][0] for c in server.broadcast.call_args_list]
        assert any(
            c.get("type") == "TOOL_RESULT" and not c["payload"]["success"]
            for c in calls
        )

    @pytest.mark.asyncio
    async def test_alt_field_names_accepted(self):
        """Browser may send 'tool' instead of 'tool_name', 'args' instead of 'arguments'."""
        bridge, server = _make_bridge()
        calls: list = []

        async def cb(tool_name, arguments):
            calls.append((tool_name, arguments))

        bridge.set_tool_call_callback(cb)
        await bridge._on_browser_message(
            {"type": "REQUEST_TOOL", "tool": "my_tool", "args": {"x": 1}}
        )
        assert calls == [("my_tool", {"x": 1})]


# ---------------------------------------------------------------------------
# USER_ACTION
# ---------------------------------------------------------------------------


class TestUserAction:
    @pytest.mark.asyncio
    async def test_action_name_queued_as_slash_command(self):
        bridge, server = _make_bridge()
        q: asyncio.Queue = asyncio.Queue()
        bridge.set_input_queue(q)
        await bridge._on_browser_message({"type": "USER_ACTION", "action": "clear"})
        assert q.get_nowait() == "/clear"

    @pytest.mark.asyncio
    async def test_content_preferred_over_action(self):
        bridge, server = _make_bridge()
        q: asyncio.Queue = asyncio.Queue()
        bridge.set_input_queue(q)
        await bridge._on_browser_message(
            {"type": "USER_ACTION", "action": "something", "content": "hello"}
        )
        assert q.get_nowait() == "hello"

    @pytest.mark.asyncio
    async def test_no_queue_does_not_raise(self):
        bridge, server = _make_bridge()
        # No queue registered — should not raise
        await bridge._on_browser_message({"type": "USER_ACTION", "action": "clear"})

    @pytest.mark.asyncio
    async def test_empty_action_and_content_not_queued(self):
        bridge, server = _make_bridge()
        q: asyncio.Queue = asyncio.Queue()
        bridge.set_input_queue(q)
        await bridge._on_browser_message({"type": "USER_ACTION"})
        assert q.empty()
