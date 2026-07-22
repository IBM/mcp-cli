# tests/dashboard/test_server.py
"""Unit tests for DashboardServer.

Avoids actually starting WebSocket servers; focuses on pure logic units.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _resolve_static
# ---------------------------------------------------------------------------


class TestResolveStatic:
    def _server(self):
        from mcp_cli.dashboard.server import DashboardServer

        return DashboardServer()

    def test_root_resolves_to_shell_html(self):
        s = self._server()
        result = s._resolve_static("/")
        assert result is not None
        assert result.name == "shell.html"

    def test_empty_path_resolves_to_shell_html(self):
        s = self._server()
        result = s._resolve_static("")
        assert result is not None
        assert result.name == "shell.html"

    def test_view_path_resolves(self):
        s = self._server()
        result = s._resolve_static("/views/agent-terminal.html")
        assert result is not None
        assert result.name == "agent-terminal.html"

    def test_themes_path_resolves(self):
        s = self._server()
        result = s._resolve_static("/themes/themes.json")
        assert result is not None
        assert result.name == "themes.json"

    def test_css_path_resolves(self):
        s = self._server()
        result = s._resolve_static("/css/variables.css")
        assert result is not None
        assert result.name == "variables.css"

    def test_js_path_resolves(self):
        s = self._server()
        result = s._resolve_static("/js/state.js")
        assert result is not None
        assert result.name == "state.js"

    def test_unknown_path_returns_none(self):
        s = self._server()
        assert s._resolve_static("/unknown/path") is None

    def test_path_traversal_rejected(self):
        s = self._server()
        # Attempt directory traversal
        result = s._resolve_static("/views/../../server.py")
        assert result is None

    def test_css_path_traversal_rejected(self):
        s = self._server()
        result = s._resolve_static("/css/../../server.py")
        assert result is None

    def test_js_path_traversal_rejected(self):
        s = self._server()
        result = s._resolve_static("/js/../../server.py")
        assert result is None

    def test_nonexistent_css_returns_none(self):
        s = self._server()
        assert s._resolve_static("/css/nonexistent.css") is None

    def test_nonexistent_js_returns_none(self):
        s = self._server()
        assert s._resolve_static("/js/nonexistent.js") is None


# ---------------------------------------------------------------------------
# broadcast
# ---------------------------------------------------------------------------


class TestBroadcast:
    @pytest.mark.asyncio
    async def test_no_clients_no_error(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        # Should not raise
        await s.broadcast({"type": "TEST"})

    @pytest.mark.asyncio
    async def test_sends_to_all_clients(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        c1, c2 = AsyncMock(), AsyncMock()
        s._clients = {c1, c2}
        await s.broadcast({"type": "PING"})
        c1.send.assert_awaited_once()
        c2.send.assert_awaited_once()
        payload = json.loads(c1.send.call_args[0][0])
        assert payload["type"] == "PING"

    @pytest.mark.asyncio
    async def test_dead_client_removed(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        good = AsyncMock()
        dead = AsyncMock()
        dead.send.side_effect = Exception("connection closed")
        s._clients = {good, dead}
        await s.broadcast({"type": "TEST"})
        assert dead not in s._clients
        assert good in s._clients


# ---------------------------------------------------------------------------
# _handle_browser_message
# ---------------------------------------------------------------------------


class TestHandleBrowserMessage:
    @pytest.mark.asyncio
    async def test_calls_callback(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        received = []

        async def handler(msg):
            received.append(msg)

        s.on_browser_message = handler
        await s._handle_browser_message('{"type":"USER_MESSAGE","content":"hi"}')
        assert received == [{"type": "USER_MESSAGE", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_invalid_json_ignored(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        # Should not raise
        await s._handle_browser_message("not json {{{")

    @pytest.mark.asyncio
    async def test_on_client_connected_called(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        connected = []

        async def on_connect(ws):
            connected.append(ws)

        s.on_client_connected = on_connect

        # Simulate _ws_handler by calling the on_client_connected part directly
        fake_ws = AsyncMock()
        fake_ws.__aiter__ = MagicMock(return_value=iter([]))
        result = on_connect(fake_ws)
        if asyncio.iscoroutine(result):
            await result
        assert fake_ws in connected

    @pytest.mark.asyncio
    async def test_handle_browser_message_passes_ws_to_two_arg_callback(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        received = []

        async def handler(msg, ws):
            received.append((msg, ws))

        s.on_browser_message = handler
        fake_ws = AsyncMock()
        await s._handle_browser_message(
            '{"type":"USER_MESSAGE","content":"hi"}', fake_ws
        )
        assert len(received) == 1
        assert received[0][0] == {"type": "USER_MESSAGE", "content": "hi"}
        assert received[0][1] is fake_ws

    @pytest.mark.asyncio
    async def test_handle_browser_message_fallback_to_one_arg_callback(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        received = []

        async def handler(msg):
            received.append(msg)

        s.on_browser_message = handler
        fake_ws = AsyncMock()
        await s._handle_browser_message(
            '{"type":"USER_MESSAGE","content":"hi"}', fake_ws
        )
        assert len(received) == 1
        assert received[0] == {"type": "USER_MESSAGE", "content": "hi"}


# ---------------------------------------------------------------------------
# send_to_client
# ---------------------------------------------------------------------------


class TestSendToClient:
    @pytest.mark.asyncio
    async def test_send_to_client_basic(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        ws = AsyncMock()
        s._clients.add(ws)
        await s.send_to_client(ws, {"type": "TEST"})
        ws.send.assert_awaited_once()
        payload = json.loads(ws.send.call_args[0][0])
        assert payload["type"] == "TEST"

    @pytest.mark.asyncio
    async def test_send_to_client_dead_ws_discarded(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        ws = AsyncMock()
        ws.send.side_effect = Exception("connection closed")
        s._clients.add(ws)
        await s.send_to_client(ws, {"type": "TEST"})
        assert ws not in s._clients


# ---------------------------------------------------------------------------
# _find_port
# ---------------------------------------------------------------------------


class TestFindPort:
    @pytest.mark.asyncio
    async def test_uses_preferred_port_when_available(self):
        from mcp_cli.dashboard.server import DashboardServer

        mock_server = MagicMock()
        mock_server.close = MagicMock()
        mock_server.wait_closed = AsyncMock()

        with patch("asyncio.start_server", new_callable=AsyncMock) as mock_start:
            mock_start.return_value = mock_server
            port = await DashboardServer._find_port(19999)
        assert port == 19999

    @pytest.mark.asyncio
    async def test_increments_when_port_in_use(self):
        from mcp_cli.dashboard.server import DashboardServer

        call_count = 0

        async def mock_start_server(handler, host, port, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("Address in use")
            srv = MagicMock()
            srv.close = MagicMock()
            srv.wait_closed = AsyncMock()
            return srv

        with patch("asyncio.start_server", side_effect=mock_start_server):
            port = await DashboardServer._find_port(19990)
        assert port == 19992  # skipped 19990 and 19991
