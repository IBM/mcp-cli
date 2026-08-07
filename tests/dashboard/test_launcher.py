# tests/dashboard/test_launcher.py
"""Unit tests for mcp_cli.dashboard.launcher."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mcp_cli.dashboard.router import AgentRouter


def _mock_server(port: int = 9120):
    """Return a mock DashboardServer whose start() resolves to the given port."""
    srv = AsyncMock()
    srv.start = AsyncMock(return_value=port)
    # AgentRouter needs to set these attributes
    srv.on_browser_message = None
    srv.on_client_connected = None
    srv.on_client_disconnected = None
    srv.has_clients = False
    return srv


class TestLaunchDashboard:
    @pytest.mark.asyncio
    @patch("mcp_cli.dashboard.launcher.webbrowser.open")
    async def test_returns_server_router_and_port(self, _mock_open):
        from mcp_cli.dashboard import launcher

        srv = _mock_server(9120)
        with patch.object(launcher, "DashboardServer", return_value=srv):
            server, router, port = await launcher.launch_dashboard()

        assert server is srv
        assert isinstance(router, AgentRouter)
        assert port == 9120

    @pytest.mark.asyncio
    @patch("mcp_cli.dashboard.launcher.webbrowser.open")
    async def test_returns_router(self, _mock_open):
        from mcp_cli.dashboard import launcher

        srv = _mock_server(9120)
        with patch.object(launcher, "DashboardServer", return_value=srv):
            _, router, _ = await launcher.launch_dashboard()

        assert isinstance(router, AgentRouter)
        assert router.server is srv

    @pytest.mark.asyncio
    async def test_opens_browser_when_no_browser_false(self):
        from mcp_cli.dashboard import launcher

        with patch.object(launcher, "DashboardServer", return_value=_mock_server(9120)):
            with patch("mcp_cli.dashboard.launcher.webbrowser.open") as mock_open:
                await launcher.launch_dashboard(no_browser=False)

        mock_open.assert_called_once_with("http://localhost:9120")

    @pytest.mark.asyncio
    async def test_skips_browser_when_no_browser_true(self):
        from mcp_cli.dashboard import launcher

        with patch.object(launcher, "DashboardServer", return_value=_mock_server(9120)):
            with patch("mcp_cli.dashboard.launcher.webbrowser.open") as mock_open:
                await launcher.launch_dashboard(no_browser=True)

        mock_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_webbrowser_exception_suppressed(self):
        from mcp_cli.dashboard import launcher

        with patch.object(launcher, "DashboardServer", return_value=_mock_server(9120)):
            # Patch where the name is looked up, not where it's defined
            with patch("mcp_cli.dashboard.launcher.webbrowser.open", side_effect=Exception("no display")):
                server, router, port = await launcher.launch_dashboard(no_browser=False)

        assert port == 9120  # function completed successfully

    @pytest.mark.asyncio
    async def test_preferred_port_passed_to_server(self):
        from mcp_cli.dashboard import launcher

        srv = _mock_server(8080)
        with patch.object(launcher, "DashboardServer", return_value=srv):
            with patch("mcp_cli.dashboard.launcher.webbrowser.open"):
                _, _, port = await launcher.launch_dashboard(port=8080)

        srv.start.assert_called_once_with(8080)
        assert port == 8080

    @pytest.mark.asyncio
    async def test_default_port_zero_passed_to_server(self):
        from mcp_cli.dashboard import launcher

        srv = _mock_server(9120)
        with patch.object(launcher, "DashboardServer", return_value=srv):
            with patch("mcp_cli.dashboard.launcher.webbrowser.open"):
                await launcher.launch_dashboard()

        srv.start.assert_called_once_with(0)
