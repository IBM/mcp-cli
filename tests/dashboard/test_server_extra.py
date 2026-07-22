# tests/dashboard/test_server_extra.py
"""Additional unit tests for DashboardServer to push coverage above 90%.

Covers: _process_request (HTTP responses), _ws_handler edge cases,
_handle_browser_message sync/error paths, stop() error handler,
_resolve_static missing-file path, _find_port exhaustion.
"""

from __future__ import annotations

import http
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(path: str):
    r = MagicMock()
    r.path = path
    return r


class _FakeWS:
    """Minimal async-iterable fake WebSocket for _ws_handler unit tests."""

    def __init__(self, messages=()):
        self._messages = list(messages)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for m in self._messages:
            yield m


class _ConnectionClosedWS:
    """Fake WebSocket whose iterator raises ConnectionClosed immediately."""

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        from websockets.exceptions import ConnectionClosedOK
        from websockets.frames import Close

        raise ConnectionClosedOK(rcvd=Close(1000, ""), sent=None)
        yield  # noqa: F704 — dead yield makes this an async generator


# ---------------------------------------------------------------------------
# _process_request — HTTP paths
# ---------------------------------------------------------------------------


class TestProcessRequest:
    def test_ws_path_returns_none(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        assert s._process_request(MagicMock(), _req("/ws")) is None

    def test_ws_path_with_query_returns_none(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        # Query string should be stripped; /ws?foo=bar still routes to WS
        assert s._process_request(MagicMock(), _req("/ws?foo=bar")) is None

    def test_unknown_path_returns_404(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        resp = s._process_request(MagicMock(), _req("/secret.php"))
        assert resp.status_code == http.HTTPStatus.NOT_FOUND

    def test_root_returns_200(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        resp = s._process_request(MagicMock(), _req("/"))
        assert resp.status_code == http.HTTPStatus.OK

    def test_root_with_query_string_returns_200(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        resp = s._process_request(MagicMock(), _req("/?v=1"))
        assert resp.status_code == http.HTTPStatus.OK

    def test_view_returns_200(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        resp = s._process_request(MagicMock(), _req("/views/agent-terminal.html"))
        assert resp.status_code == http.HTTPStatus.OK

    def test_oserror_reading_file_returns_500(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        mock_path = MagicMock(spec=Path)
        mock_path.read_bytes.side_effect = OSError("permission denied")
        with patch.object(s, "_resolve_static", return_value=mock_path):
            resp = s._process_request(MagicMock(), _req("/"))
        assert resp.status_code == http.HTTPStatus.INTERNAL_SERVER_ERROR

    def test_content_type_header_set(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        resp = s._process_request(MagicMock(), _req("/"))
        assert resp.status_code == http.HTTPStatus.OK
        # shell.html is HTML so content-type should contain text/html
        ct = resp.headers.get("Content-Type", "")
        assert "text/html" in ct

    def test_binary_content_type(self):
        """A file with unknown MIME type gets application/octet-stream."""
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        mock_path = MagicMock(spec=Path)
        mock_path.read_bytes.return_value = b"\x00\x01\x02"
        mock_path.__str__ = MagicMock(return_value="file.unknown")
        with patch.object(s, "_resolve_static", return_value=mock_path):
            resp = s._process_request(MagicMock(), _req("/"))
        assert resp.status_code == http.HTTPStatus.OK
        assert "application/octet-stream" in resp.headers.get("Content-Type", "")


# ---------------------------------------------------------------------------
# _resolve_static — missing file paths
# ---------------------------------------------------------------------------


class TestResolveStaticMissing:
    def test_nonexistent_view_file_returns_none(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        assert s._resolve_static("/views/totally-nonexistent-xyz.html") is None

    def test_nonexistent_theme_file_returns_none(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        assert s._resolve_static("/themes/nonexistent-xyz.json") is None


# ---------------------------------------------------------------------------
# _ws_handler edge cases
# ---------------------------------------------------------------------------


class TestWsHandlerEdgeCases:
    @pytest.mark.asyncio
    async def test_sync_callback_returns_none_no_await(self):
        """Sync on_client_connected callback (non-coroutine) takes the False branch at 104."""
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        called = []

        def sync_cb(ws):
            called.append(ws)
            # Returns None (not a coroutine) — exercises the asyncio.iscoroutine False branch

        s.on_client_connected = sync_cb
        ws = _FakeWS()
        await s._ws_handler(ws)
        assert called == [ws]

    @pytest.mark.asyncio
    async def test_sync_callback_exception_does_not_crash(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()

        def bad_cb(ws):
            raise RuntimeError("oops")

        s.on_client_connected = bad_cb
        ws = _FakeWS()
        await s._ws_handler(ws)  # should not raise

    @pytest.mark.asyncio
    async def test_bytes_message_not_forwarded_to_handler(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        received: list = []

        async def handler(msg):
            received.append(msg)

        s.on_browser_message = handler
        ws = _FakeWS([b"binary frame", "text frame"])
        await s._ws_handler(ws)
        # bytes message is silently dropped (only str goes to _handle_browser_message)
        # "text frame" is str but invalid JSON, so handler is never called either
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_connection_closed_handled_gracefully(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        ws = _ConnectionClosedWS()
        await s._ws_handler(ws)  # ConnectionClosed should be caught, not raised

    @pytest.mark.asyncio
    async def test_client_removed_from_set_after_disconnect(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        ws = _FakeWS()
        await s._ws_handler(ws)
        assert ws not in s._clients


# ---------------------------------------------------------------------------
# _handle_browser_message — sync callback and error paths
# ---------------------------------------------------------------------------


class TestHandleBrowserMessageExtra:
    @pytest.mark.asyncio
    async def test_valid_json_no_callback_does_not_raise(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        # on_browser_message is None — should silently do nothing
        await s._handle_browser_message('{"type": "PING"}')

    @pytest.mark.asyncio
    async def test_sync_callback_called(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        received: list = []

        def sync_handler(msg):
            received.append(msg)

        s.on_browser_message = sync_handler
        await s._handle_browser_message('{"type": "SYNC_TEST"}')
        assert received == [{"type": "SYNC_TEST"}]

    @pytest.mark.asyncio
    async def test_callback_exception_logged_not_raised(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()

        def bad_handler(msg):
            raise RuntimeError("handler exploded")

        s.on_browser_message = bad_handler
        await s._handle_browser_message('{"type": "X"}')  # should not raise


# ---------------------------------------------------------------------------
# stop() error path
# ---------------------------------------------------------------------------


class TestStopEdgeCases:
    @pytest.mark.asyncio
    async def test_stop_suppresses_wait_closed_error(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        mock_srv = MagicMock()
        mock_srv.close = MagicMock()
        mock_srv.wait_closed = AsyncMock(side_effect=Exception("shutdown error"))
        s._server = mock_srv
        await s.stop()
        assert s._server is None

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self):
        from mcp_cli.dashboard.server import DashboardServer

        s = DashboardServer()
        await s.stop()  # _server is None — should not raise


# ---------------------------------------------------------------------------
# _find_port — all ports exhausted
# ---------------------------------------------------------------------------


class TestFindPortExhausted:
    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_all_ports_in_use(self):
        from mcp_cli.dashboard.server import DashboardServer

        async def always_fail(*args, **kwargs):
            raise OSError("Address in use")

        with patch("asyncio.start_server", side_effect=always_fail):
            with pytest.raises(RuntimeError, match="Could not find an available port"):
                await DashboardServer._find_port(19990)
