# mcp_cli/apps/host.py
"""MCP Apps local host server.

Serves the host page and handles WebSocket communication between
the browser and the MCP server backend.

Uses the ``websockets`` library for both HTTP and WebSocket serving
on a single port.
"""

from __future__ import annotations

import asyncio
import base64
import html as html_mod
import http
import logging
import re
import webbrowser
from typing import Any, TYPE_CHECKING

try:
    import websockets
    from websockets.asyncio.server import serve as ws_serve, ServerConnection
    from websockets.http11 import Request, Response
except ImportError:
    raise ImportError(
        "MCP Apps support requires websockets. Install with:  pip install mcp-cli[apps]"
    )

from mcp_cli.apps.bridge import AppBridge
from mcp_cli.apps.host_page import HOST_PAGE_TEMPLATE
from mcp_cli.apps.models import AppInfo, AppState
from mcp_cli.config.defaults import (
    DEFAULT_APP_AUTO_OPEN_BROWSER,
    DEFAULT_APP_HOST_PORT_START,
    DEFAULT_APP_INIT_TIMEOUT,
    DEFAULT_APP_MAX_CONCURRENT,
    DEFAULT_HTTP_REQUEST_TIMEOUT,
)

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)

# Version injected into the host page
_MCP_CLI_VERSION = "0.13"

# Strict regex for CSP source values — reject anything that could break out
# of an HTML attribute or inject additional directives.
_SAFE_CSP_SOURCE = re.compile(r"^[a-zA-Z0-9\-.:/*]+$")


class AppHostServer:
    """Local web server for hosting MCP Apps in the user's browser."""

    def __init__(self, tool_manager: ToolManager) -> None:
        self.tool_manager = tool_manager
        self._apps: dict[str, AppInfo] = {}
        self._bridges: dict[str, AppBridge] = {}
        self._uri_to_tool: dict[str, str] = {}  # resourceUri → tool_name
        self._servers: list[Any] = []
        self._next_port = DEFAULT_APP_HOST_PORT_START

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    async def launch_app(
        self,
        tool_name: str,
        resource_uri: str,
        server_name: str,
        tool_result: Any = None,
        open_browser: bool = True,
        view_url: str | None = None,
    ) -> AppInfo:
        """Launch an MCP App in the browser.

        1. Fetch the UI resource HTML from the MCP server
        2. Start a local HTTP + WebSocket server
        3. Open the user's default browser
        4. Push the initial tool result once the WebSocket connects

        ``view_url`` is an optional direct HTTPS fallback used when
        ``resources/read`` for the ``resource_uri`` fails.
        """
        # Close any previous instance of this tool's app
        if tool_name in self._apps:
            logger.info("Closing previous instance of app %s", tool_name)
            await self.close_app(tool_name)

        if len(self._apps) >= DEFAULT_APP_MAX_CONCURRENT:
            raise RuntimeError(
                f"Maximum concurrent MCP Apps ({DEFAULT_APP_MAX_CONCURRENT}) reached"
            )

        # Fetch the UI resource
        html_content = ""
        resource: dict[str, Any] = {}

        if resource_uri.startswith(("http://", "https://")):
            # Direct HTTP fetch for HTTPS resource URIs
            html_content, resource = await self._fetch_http_resource(resource_uri)
        else:
            # MCP resources/read for ui:// and other schemes
            resource = await self.tool_manager.read_resource(
                resource_uri, server_name=server_name
            )
            html_content = self._extract_html(resource)

            # Fallback: retry without server filter (server may be registered
            # under a different transport name than the tool namespace).
            if not html_content and server_name:
                logger.debug(
                    "Retrying resource %s without server filter (was: %s)",
                    resource_uri,
                    server_name,
                )
                resource = await self.tool_manager.read_resource(resource_uri)
                html_content = self._extract_html(resource)

        # Last resort: use viewUrl (direct HTTPS) if resources/read failed.
        if not html_content and view_url:
            logger.info(
                "resources/read failed for %s, falling back to viewUrl %s",
                resource_uri,
                view_url,
            )
            html_content, resource = await self._fetch_http_resource(view_url)
            resource_uri = view_url  # update URI for app info

        if not html_content:
            raise RuntimeError(
                f"Could not fetch UI resource {resource_uri} from {server_name}"
            )

        csp = self._extract_csp(resource)
        permissions = self._extract_permissions(resource)

        # Allocate port
        port = await self._find_available_port()

        # Create app info
        app_info = AppInfo(
            tool_name=tool_name,
            resource_uri=resource_uri,
            server_name=server_name,
            state=AppState.PENDING,
            port=port,
            html_content=html_content,
            csp=csp,
            permissions=permissions,
        )
        self._apps[tool_name] = app_info
        self._uri_to_tool[resource_uri] = tool_name

        # Create bridge
        bridge = AppBridge(app_info, self.tool_manager)
        self._bridges[tool_name] = bridge

        # Start the local server
        await self._start_server(app_info, bridge, tool_result)

        # Open browser
        if open_browser and DEFAULT_APP_AUTO_OPEN_BROWSER:
            try:
                webbrowser.open(app_info.url)
                logger.info("Opened MCP App for %s at %s", tool_name, app_info.url)
            except Exception as e:
                logger.warning(
                    "Could not open browser for app %s at %s: %s",
                    tool_name,
                    app_info.url,
                    e,
                )

        return app_info

    async def close_app(self, tool_name: str) -> None:
        """Close a specific app and its server."""
        if tool_name in self._apps:
            uri = self._apps[tool_name].resource_uri
            self._uri_to_tool.pop(uri, None)
            self._apps[tool_name].state = AppState.CLOSED
            del self._apps[tool_name]
        self._bridges.pop(tool_name, None)

    async def close_all(self) -> None:
        """Shut down all app servers."""
        # Mark all apps CLOSED first so active handlers see the state change
        for app in self._apps.values():
            app.state = AppState.CLOSED
        servers = list(self._servers)
        self._servers.clear()
        for server in servers:
            try:
                server.close()
                await server.wait_closed()
            except Exception as e:
                logger.debug("Error cleaning up app server: %s", e)
        self._apps.clear()
        self._bridges.clear()
        self._uri_to_tool.clear()
        self._next_port = DEFAULT_APP_HOST_PORT_START

    def get_running_apps(self) -> list[AppInfo]:
        """Get list of currently running apps."""
        return [a for a in self._apps.values() if a.state != AppState.CLOSED]

    def get_bridge(self, tool_name: str) -> AppBridge | None:
        """Get the bridge for a running app by tool name."""
        return self._bridges.get(tool_name)

    def get_bridge_by_uri(self, resource_uri: str) -> AppBridge | None:
        """Get the bridge for a running app by its resource URI.

        Multiple tools can share the same resourceUri (e.g. show_video and
        play_video both point at the dashboard).  This lookup lets the host
        reuse the existing app instance instead of launching a new one.
        """
        tool_name = self._uri_to_tool.get(resource_uri)
        if tool_name:
            return self._bridges.get(tool_name)
        return None

    def get_any_ready_bridge(self) -> AppBridge | None:
        """Get a bridge for any running app (preferring READY state).

        Used to route ui_patch results from tools that don't carry a
        resourceUri themselves — the patch targets a panel inside an
        already-running dashboard.
        """
        # Prefer a READY app
        for tool_name, app in self._apps.items():
            if app.state == AppState.READY:
                bridge = self._bridges.get(tool_name)
                if bridge is not None:
                    return bridge
        # Fall back to any bridge (may still be INITIALIZING)
        for bridge in self._bridges.values():
            return bridge
        return None

    # ------------------------------------------------------------------ #
    #  Server setup                                                       #
    # ------------------------------------------------------------------ #

    async def _start_server(
        self,
        app_info: AppInfo,
        bridge: AppBridge,
        initial_tool_result: Any = None,
    ) -> None:
        """Start a websockets server for this app."""

        csp_attr = ""
        if app_info.csp:
            csp_parts = [
                "default-src 'none'",
                "script-src 'unsafe-inline'",
                "style-src 'unsafe-inline'",
            ]
            connect = [
                s
                for s in app_info.csp.get("connectDomains", [])
                if _SAFE_CSP_SOURCE.match(s)
            ]
            if connect:
                csp_parts.append("connect-src " + " ".join(connect))
            resource_domains = [
                s
                for s in app_info.csp.get("resourceDomains", [])
                if _SAFE_CSP_SOURCE.match(s)
            ]
            if resource_domains:
                domains = " ".join(resource_domains)
                csp_parts.append(f"img-src {domains} data:")
                csp_parts.append(f"font-src {domains}")
            csp_str = "; ".join(csp_parts)
            csp_attr = f'csp="{csp_str}"'

        # Escape tool_name to prevent XSS in host page HTML
        safe_tool_name = html_mod.escape(app_info.tool_name, quote=True)

        host_page = HOST_PAGE_TEMPLATE.format(
            tool_name=safe_tool_name,
            port=app_info.port,
            csp_attr=csp_attr,
            mcp_cli_version=_MCP_CLI_VERSION,
            init_timeout=DEFAULT_APP_INIT_TIMEOUT,
        )
        host_page_bytes = host_page.encode("utf-8")

        # Inject viewport-filling CSS into the app HTML.  MCP App views
        # are often designed for Claude.ai's inline display (fixed aspect
        # ratio).  When hosted inside an iframe panel, the root element
        # chain needs to fill 100% height so canvas-based apps (Chart.js,
        # Leaflet, D3) can use the available space.
        app_html = app_info.html_content
        _fill_css = (
            "<style>"
            "html,body{width:100%;height:100%;margin:0;overflow:auto}"
            "#root,#app,[data-reactroot]{width:100%;height:100%}"
            "#root>div,#app>div,[data-reactroot]>div"
            "{width:100%;height:100%;display:flex;flex-direction:column}"
            "canvas{max-width:100%!important;max-height:100%!important}"
            "</style>"
        )
        if "</head>" in app_html:
            app_html = app_html.replace("</head>", _fill_css + "</head>", 1)
        elif "<body" in app_html:
            app_html = app_html.replace("<body", _fill_css + "<body", 1)
        else:
            app_html = _fill_css + app_html
        app_html_bytes = app_html.encode("utf-8")

        # HTTP handler — serves the host page and app HTML
        def process_request(
            connection: ServerConnection, request: Request
        ) -> Response | None:
            # Strip query string for path matching (?embedded=1 etc.)
            path = request.path.split("?", 1)[0]

            if path == "/" or path == "":
                return Response(
                    http.HTTPStatus.OK,
                    "OK",
                    websockets.Headers(
                        {
                            "Content-Type": "text/html; charset=utf-8",
                            "Content-Length": str(len(host_page_bytes)),
                        }
                    ),
                    host_page_bytes,
                )
            if path == "/app":
                return Response(
                    http.HTTPStatus.OK,
                    "OK",
                    websockets.Headers(
                        {
                            "Content-Type": "text/html; charset=utf-8",
                            "Content-Length": str(len(app_html_bytes)),
                        }
                    ),
                    app_html_bytes,
                )
            if path != "/ws":
                body = b"Not Found"
                return Response(
                    http.HTTPStatus.NOT_FOUND,
                    "Not Found",
                    websockets.Headers({"Content-Length": str(len(body))}),
                    body,
                )
            # Return None to proceed with WebSocket upgrade for /ws
            return None

        # Store the initial tool result on the bridge so it is pushed
        # only after the app sends ui/notifications/initialized.
        if initial_tool_result is not None:
            bridge.set_initial_tool_result(initial_tool_result)

        # WebSocket handler
        async def ws_handler(ws: ServerConnection) -> None:
            bridge.set_ws(ws)
            logger.info("WebSocket connected for app %s", app_info.tool_name)

            # Drain any notifications that queued while WS was disconnected
            await bridge.drain_pending()

            try:
                async for message in ws:
                    if isinstance(message, str):
                        response = await bridge.handle_message(message)
                        if response:
                            await ws.send(response)
            except websockets.ConnectionClosed:
                pass

            logger.info("WebSocket closed for app %s", app_info.tool_name)

        server = await ws_serve(
            ws_handler,
            "localhost",
            app_info.port,
            process_request=process_request,
        )
        self._servers.append(server)

        logger.info(
            "MCP App server started for %s on port %d",
            app_info.tool_name,
            app_info.port,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    async def _find_available_port(self) -> int:
        """Find an available port starting from _next_port."""
        port = self._next_port
        max_attempts = 20
        for _ in range(max_attempts):
            try:
                server = await asyncio.start_server(
                    lambda r, w: None, "localhost", port
                )
                server.close()
                await server.wait_closed()
                self._next_port = port + 1
                return port
            except OSError:
                port += 1
        raise RuntimeError(
            f"Could not find available port after {max_attempts} attempts"
        )

    @staticmethod
    async def _fetch_http_resource(url: str) -> tuple[str, dict[str, Any]]:
        """Fetch HTML content directly from an HTTP/HTTPS URL."""
        import httpx

        async with httpx.AsyncClient(
            follow_redirects=True, timeout=DEFAULT_HTTP_REQUEST_TIMEOUT
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
            # Wrap in a resource-like structure for CSP/permissions extraction
            resource = {
                "contents": [
                    {
                        "uri": url,
                        "mimeType": resp.headers.get("content-type", "text/html"),
                        "text": html,
                    }
                ]
            }
            return html, resource

    @staticmethod
    def _extract_html(resource: dict[str, Any]) -> str:
        """Extract HTML content from a resources/read response."""
        contents = resource.get("contents", [])
        if not contents:
            contents = resource.get("result", {}).get("contents", [])

        if isinstance(contents, list) and contents:
            first = contents[0]
            if isinstance(first, dict):
                text = first.get("text")
                if text:
                    return str(text)
                blob = first.get("blob")
                if blob:
                    return base64.b64decode(blob).decode("utf-8")

        return ""

    @staticmethod
    def _extract_csp(resource: dict[str, Any]) -> dict[str, Any] | None:
        """Extract CSP configuration from resource _meta.ui.csp."""
        contents = resource.get(
            "contents", resource.get("result", {}).get("contents", [])
        )
        if isinstance(contents, list) and contents:
            first = contents[0] if isinstance(contents[0], dict) else {}
            meta = first.get("_meta", {})
            ui = meta.get("ui", {})
            csp: dict[str, Any] | None = ui.get("csp")
            return csp
        return None

    @staticmethod
    def _extract_permissions(resource: dict[str, Any]) -> dict[str, Any] | None:
        """Extract permissions from resource _meta.ui.permissions."""
        contents = resource.get(
            "contents", resource.get("result", {}).get("contents", [])
        )
        if isinstance(contents, list) and contents:
            first = contents[0] if isinstance(contents[0], dict) else {}
            meta = first.get("_meta", {})
            ui = meta.get("ui", {})
            perms: dict[str, Any] | None = ui.get("permissions")
            return perms
        return None
