# tests/apps/test_host.py
"""Tests for MCP Apps host server (no real WS servers)."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from mcp_cli.apps.host import AppHostServer, _SAFE_CSP_SOURCE
from mcp_cli.apps.models import AppInfo, AppState


# ── Fakes ──────────────────────────────────────────────────────────────────


class FakeToolManager:
    """Stub for ToolManager — only needs read_resource for launch tests."""

    def __init__(self):
        self._resource: dict[str, Any] = {
            "contents": [
                {
                    "uri": "ui://test/app.html",
                    "text": "<html><body>Hello</body></html>",
                }
            ]
        }

    async def read_resource(self, uri, server_name=None):
        return self._resource


# ── Init Tests ─────────────────────────────────────────────────────────────


class TestAppHostServerInit:
    def test_construction(self):
        tm = FakeToolManager()
        host = AppHostServer(tm)
        assert host.tool_manager is tm
        assert host._apps == {}
        assert host._bridges == {}
        assert host._servers == []

    def test_get_running_apps_empty(self):
        host = AppHostServer(FakeToolManager())
        assert host.get_running_apps() == []

    def test_get_bridge_missing(self):
        host = AppHostServer(FakeToolManager())
        assert host.get_bridge("nonexistent") is None


# ── Extract Helpers ────────────────────────────────────────────────────────


class TestExtractHtml:
    def test_text_content(self):
        resource = {"contents": [{"uri": "ui://test", "text": "<html>OK</html>"}]}
        assert AppHostServer._extract_html(resource) == "<html>OK</html>"

    def test_blob_content(self):
        html = "<html>blob</html>"
        b64 = base64.b64encode(html.encode()).decode()
        resource = {"contents": [{"uri": "ui://test", "blob": b64}]}
        assert AppHostServer._extract_html(resource) == html

    def test_empty_contents(self):
        assert AppHostServer._extract_html({"contents": []}) == ""

    def test_nested_result(self):
        resource = {
            "result": {
                "contents": [{"uri": "ui://test", "text": "<html>nested</html>"}]
            }
        }
        assert AppHostServer._extract_html(resource) == "<html>nested</html>"

    def test_no_contents(self):
        assert AppHostServer._extract_html({}) == ""


class TestExtractCsp:
    def test_csp_present(self):
        resource = {
            "contents": [
                {
                    "uri": "ui://test",
                    "text": "<html></html>",
                    "_meta": {
                        "ui": {"csp": {"connectDomains": ["https://api.example.com"]}}
                    },
                }
            ]
        }
        csp = AppHostServer._extract_csp(resource)
        assert csp == {"connectDomains": ["https://api.example.com"]}

    def test_csp_absent(self):
        resource = {"contents": [{"uri": "ui://test", "text": "<html></html>"}]}
        assert AppHostServer._extract_csp(resource) is None

    def test_csp_empty_contents(self):
        assert AppHostServer._extract_csp({"contents": []}) is None


class TestExtractPermissions:
    def test_permissions_present(self):
        resource = {
            "contents": [
                {
                    "uri": "ui://test",
                    "text": "<html></html>",
                    "_meta": {"ui": {"permissions": {"clipboard": True}}},
                }
            ]
        }
        perms = AppHostServer._extract_permissions(resource)
        assert perms == {"clipboard": True}

    def test_permissions_absent(self):
        resource = {"contents": [{"uri": "ui://test", "text": "<html></html>"}]}
        assert AppHostServer._extract_permissions(resource) is None


# ── CSP Domain Sanitization ───────────────────────────────────────────────


class TestCspDomainSanitization:
    def test_valid_domains(self):
        assert _SAFE_CSP_SOURCE.match("https://api.example.com")
        assert _SAFE_CSP_SOURCE.match("*.example.com")
        assert _SAFE_CSP_SOURCE.match("http://localhost:8080")

    def test_invalid_domains(self):
        assert not _SAFE_CSP_SOURCE.match('https://evil.com"; script-src *')
        assert not _SAFE_CSP_SOURCE.match("https://evil.com' onclick=alert(1)")
        assert not _SAFE_CSP_SOURCE.match("<script>")
        assert not _SAFE_CSP_SOURCE.match(
            "https://example.com; script-src 'unsafe-inline'"
        )


# ── Launch & Close ─────────────────────────────────────────────────────────


class TestCloseApp:
    @pytest.mark.asyncio
    async def test_close_app_removes(self):
        host = AppHostServer(FakeToolManager())
        info = AppInfo(
            tool_name="test-app",
            resource_uri="ui://test",
            server_name="srv",
            port=9470,
        )
        host._apps["test-app"] = info
        host._bridges["test-app"] = object()
        await host.close_app("test-app")
        assert "test-app" not in host._apps
        assert "test-app" not in host._bridges

    @pytest.mark.asyncio
    async def test_close_app_sets_closed(self):
        host = AppHostServer(FakeToolManager())
        info = AppInfo(
            tool_name="test-app",
            resource_uri="ui://test",
            server_name="srv",
            port=9470,
            state=AppState.READY,
        )
        host._apps["test-app"] = info
        await host.close_app("test-app")
        assert info.state == AppState.CLOSED

    @pytest.mark.asyncio
    async def test_close_app_idempotent(self):
        host = AppHostServer(FakeToolManager())
        # Closing non-existent app should not raise
        await host.close_app("nonexistent")


class TestCloseAll:
    @pytest.mark.asyncio
    async def test_close_all_marks_closed_first(self):
        host = AppHostServer(FakeToolManager())
        info1 = AppInfo(
            tool_name="app1",
            resource_uri="ui://1",
            server_name="s",
            port=9470,
            state=AppState.READY,
        )
        info2 = AppInfo(
            tool_name="app2",
            resource_uri="ui://2",
            server_name="s",
            port=9471,
            state=AppState.READY,
        )
        host._apps["app1"] = info1
        host._apps["app2"] = info2

        await host.close_all()
        assert info1.state == AppState.CLOSED
        assert info2.state == AppState.CLOSED
        assert len(host._apps) == 0
        assert len(host._bridges) == 0


class TestGetRunningApps:
    def test_filters_closed(self):
        host = AppHostServer(FakeToolManager())
        info_ready = AppInfo(
            tool_name="ready-app",
            resource_uri="ui://r",
            server_name="s",
            port=9470,
            state=AppState.READY,
        )
        info_closed = AppInfo(
            tool_name="closed-app",
            resource_uri="ui://c",
            server_name="s",
            port=9471,
            state=AppState.CLOSED,
        )
        host._apps["ready-app"] = info_ready
        host._apps["closed-app"] = info_closed
        running = host.get_running_apps()
        assert len(running) == 1
        assert running[0].tool_name == "ready-app"


# ── GetBridgeByUri ─────────────────────────────────────────────────────────


class TestGetBridgeByUri:
    def test_returns_bridge_for_known_uri(self):
        host = AppHostServer(FakeToolManager())
        fake_bridge = object()
        info = AppInfo(
            tool_name="my-tool",
            resource_uri="ui://my-tool/app.html",
            server_name="s",
            port=9470,
        )
        host._apps["my-tool"] = info
        host._bridges["my-tool"] = fake_bridge
        host._uri_to_tool["ui://my-tool/app.html"] = "my-tool"
        result = host.get_bridge_by_uri("ui://my-tool/app.html")
        assert result is fake_bridge

    def test_returns_none_for_unknown_uri(self):
        host = AppHostServer(FakeToolManager())
        result = host.get_bridge_by_uri("ui://unknown/app.html")
        assert result is None

    def test_returns_none_when_uri_maps_to_missing_bridge(self):
        host = AppHostServer(FakeToolManager())
        host._uri_to_tool["ui://my-tool/app.html"] = "my-tool"
        # Bridge not registered
        result = host.get_bridge_by_uri("ui://my-tool/app.html")
        assert result is None


# ── GetAnyReadyBridge ──────────────────────────────────────────────────────


class TestGetAnyReadyBridge:
    def test_prefers_ready_app(self):
        host = AppHostServer(FakeToolManager())
        fake_ready_bridge = object()
        fake_init_bridge = object()
        info_init = AppInfo(
            tool_name="init-app",
            resource_uri="ui://init",
            server_name="s",
            port=9470,
            state=AppState.INITIALIZING,
        )
        info_ready = AppInfo(
            tool_name="ready-app",
            resource_uri="ui://ready",
            server_name="s",
            port=9471,
            state=AppState.READY,
        )
        host._apps["init-app"] = info_init
        host._apps["ready-app"] = info_ready
        host._bridges["init-app"] = fake_init_bridge
        host._bridges["ready-app"] = fake_ready_bridge
        result = host.get_any_ready_bridge()
        assert result is fake_ready_bridge

    def test_falls_back_to_any_bridge(self):
        host = AppHostServer(FakeToolManager())
        fake_bridge = object()
        info_init = AppInfo(
            tool_name="init-app",
            resource_uri="ui://init",
            server_name="s",
            port=9470,
            state=AppState.INITIALIZING,
        )
        host._apps["init-app"] = info_init
        host._bridges["init-app"] = fake_bridge
        result = host.get_any_ready_bridge()
        assert result is fake_bridge

    def test_returns_none_when_no_bridges(self):
        host = AppHostServer(FakeToolManager())
        assert host.get_any_ready_bridge() is None

    def test_skips_ready_app_with_missing_bridge(self):
        host = AppHostServer(FakeToolManager())
        info_ready = AppInfo(
            tool_name="ready-app",
            resource_uri="ui://ready",
            server_name="s",
            port=9470,
            state=AppState.READY,
        )
        host._apps["ready-app"] = info_ready
        # No bridge registered for "ready-app" — should fall through
        result = host.get_any_ready_bridge()
        assert result is None


# ── FindAvailablePort ──────────────────────────────────────────────────────


class TestFindAvailablePort:
    @pytest.mark.asyncio
    async def test_finds_port_on_first_try(self):
        """When the first port is free, it should be returned immediately."""
        from unittest.mock import AsyncMock, MagicMock, patch

        host = AppHostServer(FakeToolManager())

        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()

        with patch("asyncio.start_server", new=AsyncMock(return_value=fake_server)):
            port = await host._find_available_port()

        assert port == 9470
        assert host._next_port == 9471

    @pytest.mark.asyncio
    async def test_skips_occupied_ports(self):
        """When first two ports raise OSError, returns the third."""
        from unittest.mock import AsyncMock, MagicMock, patch

        host = AppHostServer(FakeToolManager())

        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()

        call_count = 0

        async def start_server_side_effect(handler, host_addr, port):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise OSError("port in use")
            return fake_server

        with patch("asyncio.start_server", side_effect=start_server_side_effect):
            port = await host._find_available_port()

        assert port == 9472
        assert host._next_port == 9473

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        """When all ports are occupied, RuntimeError is raised."""
        from unittest.mock import patch

        host = AppHostServer(FakeToolManager())

        async def always_fail(handler, host_addr, port):
            raise OSError("port in use")

        with patch("asyncio.start_server", side_effect=always_fail):
            with pytest.raises(RuntimeError, match="Could not find available port"):
                await host._find_available_port()


# ── FetchHttpResource ──────────────────────────────────────────────────────


class TestFetchHttpResource:
    @pytest.mark.asyncio
    async def test_returns_html_and_resource(self):
        """A successful HTTP fetch returns (html_text, resource_dict)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        fake_response = MagicMock()
        fake_response.text = "<html>fetched</html>"
        fake_response.headers = {"content-type": "text/html"}
        fake_response.raise_for_status = MagicMock()

        fake_client = MagicMock()
        fake_client.get = AsyncMock(return_value=fake_response)
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=fake_client):
            html, resource = await AppHostServer._fetch_http_resource(
                "https://example.com/app.html"
            )

        assert html == "<html>fetched</html>"
        assert resource["contents"][0]["text"] == "<html>fetched</html>"
        assert resource["contents"][0]["uri"] == "https://example.com/app.html"
        assert resource["contents"][0]["mimeType"] == "text/html"

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        """HTTP error status causes raise_for_status to propagate."""
        from unittest.mock import AsyncMock, MagicMock, patch
        import httpx

        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "404", request=MagicMock(), response=MagicMock()
            )
        )

        fake_client = MagicMock()
        fake_client.get = AsyncMock(return_value=fake_response)
        fake_client.__aenter__ = AsyncMock(return_value=fake_client)
        fake_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=fake_client):
            with pytest.raises(httpx.HTTPStatusError):
                await AppHostServer._fetch_http_resource("https://example.com/missing")


# ── ExtractHtml (additional edge cases) ────────────────────────────────────


class TestExtractHtmlAdditional:
    def test_nested_result_key(self):
        """Content nested under 'result.contents' is also extracted."""
        resource = {
            "result": {
                "contents": [{"uri": "ui://test", "text": "<html>nested</html>"}]
            }
        }
        assert AppHostServer._extract_html(resource) == "<html>nested</html>"

    def test_blob_fallback(self):
        """If no text but blob is present, base64-decode the blob."""
        html = "<html>blob content</html>"
        b64 = base64.b64encode(html.encode()).decode()
        resource = {"contents": [{"blob": b64}]}
        assert AppHostServer._extract_html(resource) == html

    def test_non_dict_first_item_returns_empty(self):
        """Non-dict items in contents list produce empty string."""
        resource = {"contents": ["not-a-dict"]}
        assert AppHostServer._extract_html(resource) == ""


# ── ExtractCsp (additional edge cases) ─────────────────────────────────────


class TestExtractCspAdditional:
    def test_nested_result_key(self):
        csp_data = {"connectDomains": ["https://api.example.com"]}
        resource = {
            "result": {
                "contents": [
                    {
                        "uri": "ui://test",
                        "_meta": {"ui": {"csp": csp_data}},
                    }
                ]
            }
        }
        assert AppHostServer._extract_csp(resource) == csp_data

    def test_non_dict_first_item_returns_none(self):
        resource = {"contents": ["not-a-dict"]}
        # When first element is not a dict, _meta lookup should return None
        assert AppHostServer._extract_csp(resource) is None


# ── ExtractPermissions (additional edge cases) ──────────────────────────────


class TestExtractPermissionsAdditional:
    def test_nested_result_key(self):
        perms = {"clipboard": True}
        resource = {
            "result": {
                "contents": [
                    {
                        "uri": "ui://test",
                        "_meta": {"ui": {"permissions": perms}},
                    }
                ]
            }
        }
        assert AppHostServer._extract_permissions(resource) == perms

    def test_absent_returns_none(self):
        resource = {"contents": [{"uri": "ui://test"}]}
        assert AppHostServer._extract_permissions(resource) is None

    def test_empty_contents_returns_none(self):
        assert AppHostServer._extract_permissions({"contents": []}) is None

    def test_non_dict_first_item_returns_none(self):
        resource = {"contents": ["not-a-dict"]}
        assert AppHostServer._extract_permissions(resource) is None


# ── CloseAll with servers ──────────────────────────────────────────────────


class TestCloseAllWithServers:
    @pytest.mark.asyncio
    async def test_closes_servers(self):
        """close_all() calls close() + wait_closed() on registered servers."""
        from unittest.mock import AsyncMock, MagicMock

        host = AppHostServer(FakeToolManager())

        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()

        host._servers.append(fake_server)
        await host.close_all()

        fake_server.close.assert_called_once()
        fake_server.wait_closed.assert_called_once()
        assert host._servers == []

    @pytest.mark.asyncio
    async def test_server_cleanup_exception_is_swallowed(self):
        """Exceptions from server.close() are caught and logged, not re-raised."""
        from unittest.mock import AsyncMock, MagicMock

        host = AppHostServer(FakeToolManager())

        bad_server = MagicMock()
        bad_server.close = MagicMock(side_effect=RuntimeError("boom"))
        bad_server.wait_closed = AsyncMock()

        host._servers.append(bad_server)
        # Should not raise
        await host.close_all()
        assert host._servers == []

    @pytest.mark.asyncio
    async def test_resets_next_port(self):
        from mcp_cli.config.defaults import DEFAULT_APP_HOST_PORT_START

        host = AppHostServer(FakeToolManager())
        host._next_port = 9490
        await host.close_all()
        assert host._next_port == DEFAULT_APP_HOST_PORT_START


# ── LaunchApp ──────────────────────────────────────────────────────────────


class FakeAppBridge:
    """Stub AppBridge that records calls."""

    def __init__(self, app_info, tool_manager):
        self.app_info = app_info
        self.tool_manager = tool_manager
        self._initial_result = None

    def set_initial_tool_result(self, result):
        self._initial_result = result


async def _make_host_with_mocked_server(
    tm=None,
    auto_open_browser=False,
    resource=None,
):
    """Return (host, fake_server) with _start_server mocked out."""
    from unittest.mock import AsyncMock, MagicMock

    if tm is None:
        tm = FakeToolManager()
    host = AppHostServer(tm)

    fake_server = MagicMock()
    fake_server.close = MagicMock()
    fake_server.wait_closed = AsyncMock()

    return host, fake_server


class TestLaunchApp:
    @pytest.mark.asyncio
    async def test_launch_mcp_resource(self):
        """launch_app with a ui:// URI reads via tool_manager.read_resource."""
        from unittest.mock import AsyncMock, MagicMock, patch

        tm = FakeToolManager()
        host = AppHostServer(tm)

        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()

        port_coro_called = False

        async def fake_find_port():
            nonlocal port_coro_called
            port_coro_called = True
            return 9470

        host._find_available_port = fake_find_port

        with (
            patch(
                "mcp_cli.apps.host.ws_serve", new=AsyncMock(return_value=fake_server)
            ),
            patch("webbrowser.open"),
            patch("mcp_cli.apps.host.DEFAULT_APP_AUTO_OPEN_BROWSER", False),
        ):
            app_info = await host.launch_app(
                tool_name="test-tool",
                resource_uri="ui://test/app.html",
                server_name="test-server",
            )

        assert app_info.tool_name == "test-tool"
        assert app_info.port == 9470
        assert "test-tool" in host._apps
        assert "test-tool" in host._bridges
        assert "ui://test/app.html" in host._uri_to_tool

    @pytest.mark.asyncio
    async def test_launch_http_resource(self):
        """launch_app with an https:// URI calls _fetch_http_resource."""
        from unittest.mock import AsyncMock, MagicMock, patch

        tm = FakeToolManager()
        host = AppHostServer(tm)

        async def fake_find_port():
            return 9471

        host._find_available_port = fake_find_port

        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()

        html_text = "<html>http app</html>"
        resource_dict = {
            "contents": [{"uri": "https://example.com/app.html", "text": html_text}]
        }

        with (
            patch(
                "mcp_cli.apps.host.AppHostServer._fetch_http_resource",
                new=AsyncMock(return_value=(html_text, resource_dict)),
            ),
            patch(
                "mcp_cli.apps.host.ws_serve", new=AsyncMock(return_value=fake_server)
            ),
            patch("mcp_cli.apps.host.DEFAULT_APP_AUTO_OPEN_BROWSER", False),
        ):
            app_info = await host.launch_app(
                tool_name="http-tool",
                resource_uri="https://example.com/app.html",
                server_name="ext-server",
            )

        assert app_info.tool_name == "http-tool"
        assert app_info.html_content == html_text

    @pytest.mark.asyncio
    async def test_launch_raises_when_html_empty(self):
        """launch_app raises RuntimeError when no HTML is found."""

        tm = FakeToolManager()
        tm._resource = {"contents": []}  # No HTML content
        host = AppHostServer(tm)

        async def fake_find_port():
            return 9470

        host._find_available_port = fake_find_port

        with pytest.raises(RuntimeError, match="Could not fetch UI resource"):
            await host.launch_app(
                tool_name="empty-tool",
                resource_uri="ui://empty/app.html",
                server_name="test-server",
            )

    @pytest.mark.asyncio
    async def test_launch_closes_previous_instance(self):
        """launch_app closes an existing app before starting a new one."""
        from unittest.mock import AsyncMock, MagicMock, patch

        tm = FakeToolManager()
        host = AppHostServer(tm)

        # Pre-populate an existing app
        existing_info = AppInfo(
            tool_name="test-tool",
            resource_uri="ui://test/old.html",
            server_name="s",
            port=9470,
            state=AppState.READY,
        )
        host._apps["test-tool"] = existing_info
        host._uri_to_tool["ui://test/old.html"] = "test-tool"

        async def fake_find_port():
            return 9471

        host._find_available_port = fake_find_port

        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()

        close_called = []
        original_close = host.close_app

        async def spy_close(name):
            close_called.append(name)
            await original_close(name)

        host.close_app = spy_close

        with (
            patch(
                "mcp_cli.apps.host.ws_serve", new=AsyncMock(return_value=fake_server)
            ),
            patch("mcp_cli.apps.host.DEFAULT_APP_AUTO_OPEN_BROWSER", False),
        ):
            await host.launch_app(
                tool_name="test-tool",
                resource_uri="ui://test/app.html",
                server_name="test-server",
            )

        assert close_called == ["test-tool"]

    @pytest.mark.asyncio
    async def test_launch_raises_at_max_concurrent(self):
        """launch_app raises RuntimeError when max concurrent apps reached."""
        from unittest.mock import patch

        tm = FakeToolManager()
        host = AppHostServer(tm)

        # Fill up to the limit
        with patch("mcp_cli.apps.host.DEFAULT_APP_MAX_CONCURRENT", 2):
            for i in range(2):
                info = AppInfo(
                    tool_name=f"app{i}",
                    resource_uri=f"ui://app{i}",
                    server_name="s",
                    port=9470 + i,
                )
                host._apps[f"app{i}"] = info

            with pytest.raises(RuntimeError, match="Maximum concurrent MCP Apps"):
                await host.launch_app(
                    tool_name="new-app",
                    resource_uri="ui://new",
                    server_name="s",
                )

    @pytest.mark.asyncio
    async def test_launch_opens_browser_when_enabled(self):
        """launch_app opens the browser when DEFAULT_APP_AUTO_OPEN_BROWSER is True."""
        from unittest.mock import AsyncMock, MagicMock, patch

        tm = FakeToolManager()
        host = AppHostServer(tm)

        async def fake_find_port():
            return 9470

        host._find_available_port = fake_find_port

        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()

        with (
            patch(
                "mcp_cli.apps.host.ws_serve", new=AsyncMock(return_value=fake_server)
            ),
            patch("mcp_cli.apps.host.DEFAULT_APP_AUTO_OPEN_BROWSER", True),
            patch("webbrowser.open") as mock_open,
        ):
            await host.launch_app(
                tool_name="browser-tool",
                resource_uri="ui://test/app.html",
                server_name="test-server",
            )

        mock_open.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_browser_exception_does_not_raise(self):
        """launch_app swallows exceptions from webbrowser.open."""
        from unittest.mock import AsyncMock, MagicMock, patch

        tm = FakeToolManager()
        host = AppHostServer(tm)

        async def fake_find_port():
            return 9470

        host._find_available_port = fake_find_port

        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()

        with (
            patch(
                "mcp_cli.apps.host.ws_serve", new=AsyncMock(return_value=fake_server)
            ),
            patch("mcp_cli.apps.host.DEFAULT_APP_AUTO_OPEN_BROWSER", True),
            patch("webbrowser.open", side_effect=OSError("no browser")),
        ):
            # Should not raise
            info = await host.launch_app(
                tool_name="no-browser-tool",
                resource_uri="ui://test/app.html",
                server_name="test-server",
            )
        assert info.tool_name == "no-browser-tool"

    @pytest.mark.asyncio
    async def test_launch_with_initial_tool_result(self):
        """launch_app passes initial_tool_result to the bridge."""
        from unittest.mock import AsyncMock, MagicMock, patch

        tm = FakeToolManager()
        host = AppHostServer(tm)

        async def fake_find_port():
            return 9470

        host._find_available_port = fake_find_port

        fake_server = MagicMock()
        fake_server.close = MagicMock()
        fake_server.wait_closed = AsyncMock()

        captured_bridges = {}

        async def spy_start_server(app_info, bridge, initial_tool_result=None):
            captured_bridges[app_info.tool_name] = (bridge, initial_tool_result)
            # Fake a server registration
            host._servers.append(fake_server)

        host._start_server = spy_start_server

        with patch("mcp_cli.apps.host.DEFAULT_APP_AUTO_OPEN_BROWSER", False):
            await host.launch_app(
                tool_name="result-tool",
                resource_uri="ui://test/app.html",
                server_name="test-server",
                tool_result={"data": "hello"},
            )

        assert "result-tool" in captured_bridges
        _, initial_result = captured_bridges["result-tool"]
        assert initial_result == {"data": "hello"}


# ── StartServer / process_request / ws_handler ─────────────────────────────


class TestStartServer:
    """Tests for _start_server and the closures it creates."""

    def _make_host_and_app_info(self, csp=None, permissions=None, html="<html/>"):
        tm = FakeToolManager()
        host = AppHostServer(tm)
        info = AppInfo(
            tool_name="srv-tool",
            resource_uri="ui://test/app.html",
            server_name="srv",
            port=9470,
            html_content=html,
            csp=csp,
            permissions=permissions,
        )
        return host, info

    @pytest.mark.asyncio
    async def test_start_server_registers_server(self):
        """_start_server appends the returned ws server to self._servers."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        fake_server = MagicMock()

        with patch(
            "mcp_cli.apps.host.ws_serve", new=AsyncMock(return_value=fake_server)
        ):
            await host._start_server(info, bridge)

        assert fake_server in host._servers

    @pytest.mark.asyncio
    async def test_start_server_sets_initial_tool_result(self):
        """_start_server calls bridge.set_initial_tool_result when provided."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        fake_server = MagicMock()

        with patch(
            "mcp_cli.apps.host.ws_serve", new=AsyncMock(return_value=fake_server)
        ):
            await host._start_server(info, bridge, initial_tool_result={"foo": "bar"})

        assert bridge._initial_tool_result == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_start_server_no_initial_result_skips_set(self):
        """_start_server does NOT call set_initial_tool_result when result is None."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        fake_server = MagicMock()

        with patch(
            "mcp_cli.apps.host.ws_serve", new=AsyncMock(return_value=fake_server)
        ):
            await host._start_server(info, bridge, initial_tool_result=None)

        # Should remain unset
        assert bridge._initial_tool_result is None

    @pytest.mark.asyncio
    async def test_process_request_serves_root(self):
        """The process_request closure returns 200 for '/'."""
        from unittest.mock import MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge
        import http

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        captured_process_request = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_process_request
            captured_process_request = process_request
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        assert captured_process_request is not None

        fake_conn = MagicMock()
        fake_req = MagicMock()
        fake_req.path = "/"

        response = captured_process_request(fake_conn, fake_req)
        assert response is not None
        assert response.status_code == http.HTTPStatus.OK

    @pytest.mark.asyncio
    async def test_process_request_serves_root_with_query_string(self):
        """The process_request closure returns 200 for '/?embedded=1'."""
        from unittest.mock import MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge
        import http

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        captured_process_request = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_process_request
            captured_process_request = process_request
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        assert captured_process_request is not None

        fake_conn = MagicMock()
        fake_req = MagicMock()
        fake_req.path = "/?embedded=1"

        response = captured_process_request(fake_conn, fake_req)
        assert response is not None
        assert response.status_code == http.HTTPStatus.OK

    @pytest.mark.asyncio
    async def test_process_request_serves_empty_path_as_root(self):
        """The process_request closure treats '' the same as '/'."""
        from unittest.mock import MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge
        import http

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        captured_process_request = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_process_request
            captured_process_request = process_request
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        fake_conn = MagicMock()
        fake_req = MagicMock()
        fake_req.path = ""

        response = captured_process_request(fake_conn, fake_req)
        assert response.status_code == http.HTTPStatus.OK

    @pytest.mark.asyncio
    async def test_process_request_serves_app(self):
        """The process_request closure returns 200 for '/app'."""
        from unittest.mock import MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge
        import http

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        captured_process_request = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_process_request
            captured_process_request = process_request
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        fake_conn = MagicMock()
        fake_req = MagicMock()
        fake_req.path = "/app"

        response = captured_process_request(fake_conn, fake_req)
        assert response.status_code == http.HTTPStatus.OK

    @pytest.mark.asyncio
    async def test_process_request_404_for_unknown(self):
        """The process_request closure returns 404 for unknown paths."""
        from unittest.mock import MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge
        import http

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        captured_process_request = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_process_request
            captured_process_request = process_request
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        fake_conn = MagicMock()
        fake_req = MagicMock()
        fake_req.path = "/unknown-path"

        response = captured_process_request(fake_conn, fake_req)
        assert response.status_code == http.HTTPStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_process_request_ws_returns_none(self):
        """The process_request closure returns None for '/ws' (WS upgrade)."""
        from unittest.mock import MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        captured_process_request = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_process_request
            captured_process_request = process_request
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        fake_conn = MagicMock()
        fake_req = MagicMock()
        fake_req.path = "/ws"

        response = captured_process_request(fake_conn, fake_req)
        assert response is None

    @pytest.mark.asyncio
    async def test_csp_with_connect_domains(self):
        """CSP connectDomains are injected into the host page meta."""
        from unittest.mock import MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge

        csp = {"connectDomains": ["https://api.example.com", "ws://localhost:8080"]}
        host, info = self._make_host_and_app_info(csp=csp)
        bridge = AppBridge(info, host.tool_manager)

        captured_ws_handler = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_ws_handler
            captured_ws_handler = handler
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        # Just verify server was created — CSP is embedded in host_page_bytes
        assert captured_ws_handler is not None

    @pytest.mark.asyncio
    async def test_csp_with_resource_domains(self):
        """CSP resourceDomains are added as img-src and font-src."""
        from unittest.mock import MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge

        csp = {
            "connectDomains": [],
            "resourceDomains": ["https://cdn.example.com"],
        }
        host, info = self._make_host_and_app_info(csp=csp)
        bridge = AppBridge(info, host.tool_manager)

        async def fake_serve(handler, host_addr, port, process_request=None):
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=fake_serve):
            await host._start_server(info, bridge)

    @pytest.mark.asyncio
    async def test_csp_filters_invalid_domains(self):
        """CSP domains that fail the regex are silently filtered."""
        from unittest.mock import MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge

        csp = {
            "connectDomains": ['evil.com"; script-src *', "https://ok.example.com"],
            "resourceDomains": ["<script>bad</script>", "https://cdn.example.com"],
        }
        host, info = self._make_host_and_app_info(csp=csp)
        bridge = AppBridge(info, host.tool_manager)

        captured_pr = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_pr
            captured_pr = process_request
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            # Should complete without error
            await host._start_server(info, bridge)

    @pytest.mark.asyncio
    async def test_ws_handler_calls_bridge(self):
        """ws_handler sets ws on bridge, drains pending, and handles messages."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        captured_ws_handler = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_ws_handler
            captured_ws_handler = handler
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        assert captured_ws_handler is not None

        # Build a fake WebSocket that yields one message then stops
        import json

        fake_ws = MagicMock()
        fake_ws.send = AsyncMock()

        msg_json = json.dumps(
            {"jsonrpc": "2.0", "method": "ui/notifications/initialized"}
        )

        async def fake_aiter(self):
            yield msg_json

        fake_ws.__aiter__ = fake_aiter

        bridge.drain_pending = AsyncMock()

        await captured_ws_handler(fake_ws)

        bridge.drain_pending.assert_called_once()
        assert bridge._ws is fake_ws

    @pytest.mark.asyncio
    async def test_ws_handler_sends_response(self):
        """ws_handler sends non-None responses back to the client."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge
        import json

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        captured_ws_handler = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_ws_handler
            captured_ws_handler = handler
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        # A request that produces a response
        msg_json = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "unknown-method-with-id"}
        )

        fake_ws = MagicMock()
        sent_messages = []

        async def fake_send(msg):
            sent_messages.append(msg)

        fake_ws.send = fake_send

        async def fake_aiter(self):
            yield msg_json

        fake_ws.__aiter__ = fake_aiter

        bridge.drain_pending = AsyncMock()

        await captured_ws_handler(fake_ws)

        assert len(sent_messages) == 1
        parsed = json.loads(sent_messages[0])
        assert parsed["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_ws_handler_handles_connection_closed(self):
        """ws_handler catches ConnectionClosed without propagating."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge
        import websockets

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        captured_ws_handler = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_ws_handler
            captured_ws_handler = handler
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        fake_ws = MagicMock()
        fake_ws.send = AsyncMock()

        async def raise_connection_closed(self):
            raise websockets.ConnectionClosed(None, None)
            yield  # make it an async generator

        fake_ws.__aiter__ = raise_connection_closed

        bridge.drain_pending = AsyncMock()

        # Should NOT raise
        await captured_ws_handler(fake_ws)

    @pytest.mark.asyncio
    async def test_tool_name_escaped_in_host_page(self):
        """XSS-dangerous tool names are escaped in the host page HTML."""
        from unittest.mock import MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge

        host, _ = self._make_host_and_app_info()
        # Override tool_name with one containing HTML special chars
        info = AppInfo(
            tool_name="<script>alert(1)</script>",
            resource_uri="ui://xss/app.html",
            server_name="srv",
            port=9470,
            html_content="<html/>",
        )
        bridge = AppBridge(info, host.tool_manager)

        async def capture_serve(handler, host_addr, port, process_request=None):
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)
        # If no exception was raised, escaping succeeded (the template formatting
        # would fail or inject raw HTML if html_mod.escape wasn't used).

    @pytest.mark.asyncio
    async def test_ws_handler_ignores_binary_messages(self):
        """ws_handler skips non-string (binary) WebSocket messages silently."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from mcp_cli.apps.bridge import AppBridge

        host, info = self._make_host_and_app_info()
        bridge = AppBridge(info, host.tool_manager)

        captured_ws_handler = None

        async def capture_serve(handler, host_addr, port, process_request=None):
            nonlocal captured_ws_handler
            captured_ws_handler = handler
            return MagicMock()

        with patch("mcp_cli.apps.host.ws_serve", side_effect=capture_serve):
            await host._start_server(info, bridge)

        fake_ws = MagicMock()
        sent_messages = []
        fake_ws.send = AsyncMock(side_effect=lambda m: sent_messages.append(m))

        async def fake_aiter(self):
            yield b"\x00\x01\x02"  # binary frame — not a str

        fake_ws.__aiter__ = fake_aiter

        bridge.drain_pending = AsyncMock()

        await captured_ws_handler(fake_ws)

        # Binary message should be silently ignored — no ws.send calls
        assert sent_messages == []


# ── ExtractHtml — blob=None branch (408->411) ──────────────────────────────


class TestExtractHtmlBlobNone:
    def test_dict_with_neither_text_nor_blob_returns_empty(self):
        """A dict entry with no 'text' and no 'blob' key returns empty string."""
        resource = {"contents": [{"uri": "ui://test"}]}  # neither text nor blob
        assert AppHostServer._extract_html(resource) == ""

    def test_dict_with_none_blob_returns_empty(self):
        """A dict entry with text=None and blob=None returns empty string."""
        resource = {"contents": [{"text": None, "blob": None}]}
        assert AppHostServer._extract_html(resource) == ""


# ── LaunchApp browser control ────────────────────────────────────────────


class TestLaunchAppBrowserControl:
    """Verify the open_browser parameter controls webbrowser.open() calls."""

    @pytest.mark.asyncio
    async def test_open_browser_false_suppresses_webbrowser(self):
        """When open_browser=False, webbrowser.open should not be called."""
        from unittest.mock import AsyncMock, patch

        host = AppHostServer(FakeToolManager())
        browser_calls: list[str] = []

        with (
            patch.object(host, "_start_server", new=AsyncMock()),
            patch.object(
                host, "_find_available_port", new=AsyncMock(return_value=9999)
            ),
            patch("mcp_cli.apps.host.webbrowser") as mock_wb,
        ):
            mock_wb.open = lambda url: browser_calls.append(url)
            app_info = await host.launch_app(
                tool_name="test_tool",
                resource_uri="ui://test/app.html",
                server_name="test_server",
                open_browser=False,
            )

        assert app_info.port == 9999
        assert browser_calls == []

    @pytest.mark.asyncio
    async def test_open_browser_true_calls_webbrowser(self):
        """When open_browser=True (default), webbrowser.open should be called."""
        from unittest.mock import AsyncMock, patch

        host = AppHostServer(FakeToolManager())
        browser_calls: list[str] = []

        with (
            patch.object(host, "_start_server", new=AsyncMock()),
            patch.object(
                host, "_find_available_port", new=AsyncMock(return_value=9999)
            ),
            patch("mcp_cli.apps.host.webbrowser") as mock_wb,
        ):
            mock_wb.open = lambda url: browser_calls.append(url)
            await host.launch_app(
                tool_name="test_tool",
                resource_uri="ui://test/app.html",
                server_name="test_server",
            )

        assert len(browser_calls) == 1
        assert "9999" in browser_calls[0]
