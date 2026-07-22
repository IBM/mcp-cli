# mcp_cli/utils/loopback_origin.py
"""Origin validation shared by mcp-cli's local WebSocket servers.

Both the MCP Apps bridge (``mcp_cli.apps.host``) and the dashboard
(``mcp_cli.dashboard.server``) run an unauthenticated WebSocket server on
``localhost`` intended only for the host page they themselves serve. Real
browsers always attach an ``Origin`` header to WebSocket handshakes, but —
unlike fetch()/XHR — do not enforce same-origin policy on the connection
itself; that enforcement is the server's job. Without it, any web page the
browser loads (unrelated to mcp-cli) could attach to the bridge by simply
knowing (or scanning for) the port.
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Loopback hostnames the host page can legitimately be served from.
_ALLOWED_ORIGIN_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_allowed_origin(origin: str | None, port: int) -> bool:
    """Return True if *origin* is the http://localhost:<port> host page origin.

    Requires an exact match on scheme, loopback host, and port — this is
    what actually prevents an unrelated page from attaching to the bridge.
    """
    if not origin:
        return False
    try:
        parsed = urlsplit(origin)
        hostname = parsed.hostname
        origin_port = parsed.port
    except ValueError:
        # urlsplit()/.hostname/.port can each raise on malformed input
        # (bad IPv6 literal, non-numeric port, ...) depending on the
        # Python version, so all three are covered here.
        return False
    if parsed.scheme != "http":
        return False
    if hostname not in _ALLOWED_ORIGIN_HOSTS:
        return False
    return origin_port == port
