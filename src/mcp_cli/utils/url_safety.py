# mcp_cli/utils/url_safety.py
"""Outbound URL safety checks — guards against SSRF.

Used wherever mcp-cli fetches a URL supplied by a semi-trusted MCP server
(e.g. an MCP App's ``resource_uri`` or a tool result's ``viewUrl``). Without
this check, a malicious or compromised server could point mcp-cli at an
internal-only service or a cloud metadata endpoint (``169.254.169.254``)
and have the response rendered back to the user.

Known limitation: this validates the hostname's *current* DNS resolution
before connecting, so it does not defend against DNS rebinding (an
attacker-controlled domain with a very short TTL that resolves to a public
address at check time and a private one at connect time). Closing that
fully would require a custom transport that connects to the pinned,
validated IP rather than letting the HTTP client re-resolve the hostname.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _is_unsafe_address(addr: str) -> bool:
    """Return True if *addr* (an IP literal) is not a routable public address."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return True  # unparsable -> treat as unsafe
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_safe_fetch_url(url: str) -> bool:
    """Return True if *url* is http(s) and every resolved address is public.

    Checks all A/AAAA records for the hostname, not just the first, since a
    hostname can round-robin between multiple addresses.
    """
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
    except ValueError:
        # urlsplit()/`.hostname` can both raise on malformed input (e.g. a
        # broken IPv6 literal) - which one depends on the Python version,
        # so both are covered here rather than just the constructor call.
        return False
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    return all(not _is_unsafe_address(info[4][0]) for info in infos)
