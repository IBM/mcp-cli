# tests/dashboard/test_integration.py
"""Integration tests for DashboardServer using a real WebSocket connection.

These tests start an actual local server and connect to it with a real
WebSocket client.  No mocks for the transport layer — this validates the
full server↔browser message path end-to-end.

Skipped automatically if the ``websockets`` package is not installed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

# Skip the whole module when websockets is absent (e.g. minimal install)
websockets = pytest.importorskip("websockets")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ws_connect(port: int):
    """Return an async context manager that connects to the WS endpoint."""
    from websockets.asyncio.client import connect

    return connect(f"ws://localhost:{port}/ws")


async def _recv(ws, timeout: float = 2.0):
    """Receive one message with a timeout."""
    return await asyncio.wait_for(ws.recv(), timeout=timeout)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def live_server():
    """Start a real DashboardServer on a random port. Stops on teardown."""
    from mcp_cli.dashboard.server import DashboardServer

    srv = DashboardServer()
    port = await srv.start(port=0)
    yield srv, port
    await srv.stop()


# ---------------------------------------------------------------------------
# Connection + broadcast
# ---------------------------------------------------------------------------


class TestServerConnect:
    async def test_ws_connects(self, live_server):
        _, port = live_server
        async with _ws_connect(port):
            pass  # no exception = success

    async def test_broadcast_received_by_client(self, live_server):
        srv, port = live_server
        async with _ws_connect(port) as ws:
            await asyncio.sleep(0.02)  # let server register client
            await srv.broadcast({"type": "PING", "val": 42})
            raw = await _recv(ws)
        assert json.loads(raw) == {"type": "PING", "val": 42}

    async def test_broadcast_reaches_two_clients(self, live_server):
        srv, port = live_server
        async with _ws_connect(port) as ws1:
            async with _ws_connect(port) as ws2:
                await asyncio.sleep(0.02)
                await srv.broadcast({"type": "MULTI"})
                r1 = await _recv(ws1)
                r2 = await _recv(ws2)
        assert json.loads(r1)["type"] == "MULTI"
        assert json.loads(r2)["type"] == "MULTI"

    async def test_broadcast_after_client_disconnects_does_not_raise(self, live_server):
        srv, port = live_server
        async with _ws_connect(port):
            pass  # connect then immediately disconnect
        await asyncio.sleep(0.05)
        # Server should have cleaned up; broadcasting to no clients is fine
        await srv.broadcast({"type": "AFTER_DISCONNECT"})

    async def test_stop_closes_server(self, live_server):
        srv, port = live_server
        await srv.stop()
        # Connecting after stop should fail
        with pytest.raises(Exception):
            async with _ws_connect(port):
                pass


# ---------------------------------------------------------------------------
# Browser → server message routing
# ---------------------------------------------------------------------------


class TestBrowserMessages:
    async def test_browser_message_triggers_callback(self, live_server):
        srv, port = live_server
        received: list[dict] = []

        async def handler(msg):
            received.append(msg)

        srv.on_browser_message = handler

        async with _ws_connect(port) as ws:
            await ws.send(json.dumps({"type": "USER_MESSAGE", "content": "hi"}))
            await asyncio.sleep(0.1)

        assert received == [{"type": "USER_MESSAGE", "content": "hi"}]

    async def test_invalid_json_ignored_server_stays_alive(self, live_server):
        srv, port = live_server

        async with _ws_connect(port) as ws:
            await ws.send("not valid json {{{{")
            await asyncio.sleep(0.05)

        # Server still running — new connection succeeds
        async with _ws_connect(port):
            pass


# ---------------------------------------------------------------------------
# on_client_connected callback
# ---------------------------------------------------------------------------


class TestOnClientConnectedCallback:
    async def test_callback_fired_on_connect(self, live_server):
        srv, port = live_server
        fired: list[int] = []

        async def cb(ws):
            fired.append(1)

        srv.on_client_connected = cb

        async with _ws_connect(port):
            await asyncio.sleep(0.05)

        assert fired == [1]

    async def test_bridge_sends_view_registry_to_new_client(self, live_server):
        """DashboardBridge wires on_client_connected; new clients get VIEW_REGISTRY."""
        from mcp_cli.dashboard.bridge import DashboardBridge

        srv, port = live_server
        bridge = DashboardBridge(srv)
        bridge._view_registry = [{"id": "stats:main", "name": "Stats"}]

        async with _ws_connect(port) as ws:
            raw = await _recv(ws)

        msg = json.loads(raw)
        assert msg["type"] == "VIEW_REGISTRY"
        assert msg["payload"]["views"][0]["id"] == "stats:main"


# ---------------------------------------------------------------------------
# DashboardBridge end-to-end message flow
# ---------------------------------------------------------------------------


class TestBridgeEndToEnd:
    async def test_tool_result_reaches_browser(self, live_server):
        from mcp_cli.dashboard.bridge import DashboardBridge

        srv, port = live_server
        bridge = DashboardBridge(srv)

        async with _ws_connect(port) as ws:
            await asyncio.sleep(0.02)
            await bridge.on_tool_result(
                tool_name="get_data",
                server_name="sqlite",
                result={"rows": 3},
                success=True,
            )
            raw = await _recv(ws)

        msg = json.loads(raw)
        assert msg["type"] == "TOOL_RESULT"
        assert msg["payload"]["tool_name"] == "get_data"
        assert msg["payload"]["success"] is True

    async def test_agent_state_reaches_browser(self, live_server):
        from mcp_cli.dashboard.bridge import DashboardBridge

        srv, port = live_server
        bridge = DashboardBridge(srv)

        async with _ws_connect(port) as ws:
            await asyncio.sleep(0.02)
            await bridge.on_agent_state("thinking", turn_number=3, tokens_used=1024)
            raw = await _recv(ws)

        msg = json.loads(raw)
        assert msg["type"] == "AGENT_STATE"
        assert msg["payload"]["status"] == "thinking"
        assert msg["payload"]["turn_number"] == 3

    async def test_conversation_token_reaches_browser(self, live_server):
        from mcp_cli.dashboard.bridge import DashboardBridge

        srv, port = live_server
        bridge = DashboardBridge(srv)

        async with _ws_connect(port) as ws:
            await asyncio.sleep(0.02)
            await bridge.on_token("Hello", done=False)
            raw = await _recv(ws)

        msg = json.loads(raw)
        assert msg["type"] == "CONVERSATION_TOKEN"
        assert msg["payload"]["token"] == "Hello"
        assert msg["payload"]["done"] is False

    async def test_request_tool_invokes_callback(self, live_server):
        from mcp_cli.dashboard.bridge import DashboardBridge

        srv, port = live_server
        bridge = DashboardBridge(srv)
        calls: list = []

        async def cb(name, args):
            calls.append((name, args))

        bridge.set_tool_call_callback(cb)

        async with _ws_connect(port) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "REQUEST_TOOL",
                        "tool_name": "query_db",
                        "arguments": {"sql": "SELECT 1"},
                    }
                )
            )
            await asyncio.sleep(0.1)

        assert calls == [("query_db", {"sql": "SELECT 1"})]

    async def test_user_action_queued_as_slash_command(self, live_server):
        from mcp_cli.dashboard.bridge import DashboardBridge

        srv, port = live_server
        bridge = DashboardBridge(srv)
        q: asyncio.Queue = asyncio.Queue()
        bridge.set_input_queue(q)

        async with _ws_connect(port) as ws:
            await ws.send(json.dumps({"type": "USER_ACTION", "action": "clear"}))
            await asyncio.sleep(0.1)

        assert q.get_nowait() == "/clear"

    async def test_user_action_content_queued_directly(self, live_server):
        from mcp_cli.dashboard.bridge import DashboardBridge

        srv, port = live_server
        bridge = DashboardBridge(srv)
        q: asyncio.Queue = asyncio.Queue()
        bridge.set_input_queue(q)

        async with _ws_connect(port) as ws:
            await ws.send(
                json.dumps({"type": "USER_ACTION", "content": "custom message"})
            )
            await asyncio.sleep(0.1)

        assert q.get_nowait() == "custom message"

    async def test_view_discovered_from_tool_result_meta_ui(self, live_server):
        from mcp_cli.dashboard.bridge import DashboardBridge

        srv, port = live_server
        bridge = DashboardBridge(srv)

        async with _ws_connect(port) as ws:
            await asyncio.sleep(0.02)
            await bridge.on_tool_result(
                tool_name="get_stats",
                server_name="analytics",
                result=None,
                success=True,
                meta_ui={"view": "stats:main", "name": "Stats Dashboard"},
            )
            # Should receive two messages: TOOL_RESULT and VIEW_REGISTRY
            msgs = []
            for _ in range(2):
                raw = await _recv(ws)
                msgs.append(json.loads(raw))

        types = {m["type"] for m in msgs}
        assert "TOOL_RESULT" in types
        assert "VIEW_REGISTRY" in types

        registry_msg = next(m for m in msgs if m["type"] == "VIEW_REGISTRY")
        assert any(v["id"] == "stats:main" for v in registry_msg["payload"]["views"])
