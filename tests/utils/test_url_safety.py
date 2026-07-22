# tests/utils/test_url_safety.py
"""Tests for the outbound URL SSRF guard used by MCP Apps' resource fetch."""

from __future__ import annotations

import socket
from unittest.mock import patch

from mcp_cli.utils.url_safety import _is_unsafe_address, is_safe_fetch_url


def _fake_addrinfo(addr: str):
    return [(2, 1, 6, "", (addr, 443))]


class TestIsSafeFetchUrl:
    def test_public_ip_allowed(self):
        with patch(
            "mcp_cli.utils.url_safety.socket.getaddrinfo",
            return_value=_fake_addrinfo("93.184.216.34"),
        ):
            assert is_safe_fetch_url("https://example.com/app.html") is True

    def test_link_local_metadata_address_rejected(self):
        with patch(
            "mcp_cli.utils.url_safety.socket.getaddrinfo",
            return_value=_fake_addrinfo("169.254.169.254"),
        ):
            assert is_safe_fetch_url("http://metadata.example/latest/") is False

    def test_loopback_ip_literal_rejected(self):
        assert is_safe_fetch_url("http://127.0.0.1:8080/admin") is False

    def test_private_range_ip_literal_rejected(self):
        assert is_safe_fetch_url("http://10.0.0.5/internal") is False
        assert is_safe_fetch_url("http://192.168.1.1/internal") is False
        assert is_safe_fetch_url("http://172.16.0.1/internal") is False

    def test_ipv6_loopback_rejected(self):
        assert is_safe_fetch_url("http://[::1]:8080/") is False

    def test_non_http_scheme_rejected(self):
        assert is_safe_fetch_url("file:///etc/passwd") is False
        assert is_safe_fetch_url("gopher://example.com/") is False
        assert is_safe_fetch_url("javascript:alert(1)") is False

    def test_missing_hostname_rejected(self):
        assert is_safe_fetch_url("http://") is False

    def test_unresolvable_hostname_rejected(self):
        with patch(
            "mcp_cli.utils.url_safety.socket.getaddrinfo",
            side_effect=socket.gaierror("nodename nor servname provided"),
        ):
            assert is_safe_fetch_url("https://this-does-not-resolve.invalid/") is False

    def test_one_bad_address_among_many_rejects_whole_hostname(self):
        """If ANY resolved address is unsafe, the whole URL is rejected."""
        with patch(
            "mcp_cli.utils.url_safety.socket.getaddrinfo",
            return_value=[
                (2, 1, 6, "", ("93.184.216.34", 443)),
                (2, 1, 6, "", ("169.254.169.254", 443)),
            ],
        ):
            assert is_safe_fetch_url("https://round-robin.example/") is False

    def test_malformed_url_rejected(self):
        assert is_safe_fetch_url("not a url at all") is False

    def test_malformed_ipv6_literal_rejected(self):
        """Whether the raise happens in urlsplit() or on .hostname access is
        Python-version-dependent — both must be caught, not just one."""
        assert is_safe_fetch_url("http://[invalid::ipv6/path") is False

    def test_empty_addrinfo_rejected(self):
        """getaddrinfo() returning an empty list (no raise) must not be
        treated as vacuously safe."""
        with patch(
            "mcp_cli.utils.url_safety.socket.getaddrinfo",
            return_value=[],
        ):
            assert is_safe_fetch_url("https://no-records.example/") is False


class TestIsUnsafeAddress:
    def test_public_address_is_safe(self):
        assert _is_unsafe_address("93.184.216.34") is False

    def test_unparsable_address_treated_as_unsafe(self):
        assert _is_unsafe_address("not-an-ip-address") is True
