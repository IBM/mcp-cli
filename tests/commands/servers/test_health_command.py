# tests/commands/servers/test_health_command.py
"""Tests for the HealthCommand (servers/health.py)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.commands.servers.health import HealthCommand


# health.py imports get_context at module level, so patch the symbol there.
_GET_CONTEXT = "mcp_cli.commands.servers.health.get_context"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_manager(health_result=None):
    tm = MagicMock()
    tm.check_server_health = AsyncMock(return_value=health_result or {})
    return tm


def _healthy_result(name: str = "srv") -> dict:
    return {name: {"status": "healthy", "ping_success": True}}


def _timeout_result(name: str = "srv") -> dict:
    return {name: {"status": "timeout", "ping_success": False}}


def _error_result(name: str = "srv", error: str = "connection refused") -> dict:
    return {name: {"status": "error", "ping_success": False, "error": error}}


def _unhealthy_no_error(name: str = "srv") -> dict:
    return {name: {"status": "unhealthy", "ping_success": False}}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cmd():
    return HealthCommand()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestHealthCommandProperties:
    def test_name(self, cmd):
        assert cmd.name == "health"

    def test_aliases(self, cmd):
        assert cmd.aliases == []

    def test_description(self, cmd):
        assert "health" in cmd.description.lower()

    def test_help_text(self, cmd):
        text = cmd.help_text
        assert "/health" in text

    def test_parameters(self, cmd):
        names = {p.name for p in cmd.parameters}
        assert "server_name" in names


# ---------------------------------------------------------------------------
# execute() — no tool manager
# ---------------------------------------------------------------------------


class TestHealthNoToolManager:
    async def test_no_kwarg_and_no_context(self, cmd):
        with patch(_GET_CONTEXT, return_value=None):
            result = await cmd.execute()
        assert result.success is False
        assert "No active tool manager" in result.error

    async def test_no_kwarg_context_has_no_tool_manager(self, cmd):
        ctx = MagicMock()
        ctx.tool_manager = None
        with patch(_GET_CONTEXT, return_value=ctx):
            result = await cmd.execute()
        assert result.success is False
        assert "No active tool manager" in result.error

    async def test_tool_manager_from_context(self, cmd):
        """When no tool_manager kwarg, falls back to context.tool_manager."""
        tm = _make_tool_manager(_healthy_result("srv"))
        ctx = MagicMock()
        ctx.tool_manager = tm
        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("chuk_term.ui.output"):
                result = await cmd.execute()
        assert result.success is True
        tm.check_server_health.assert_called_once_with(None)

    async def test_get_context_raises_exception(self, cmd):
        """Exception in get_context is caught; falls to no-manager error."""
        with patch(_GET_CONTEXT, side_effect=RuntimeError("ctx error")):
            result = await cmd.execute()
        assert result.success is False
        assert "No active tool manager" in result.error


# ---------------------------------------------------------------------------
# execute() — server_name parsing
# ---------------------------------------------------------------------------


class TestHealthServerNameParsing:
    async def test_server_name_from_kwarg(self, cmd):
        tm = _make_tool_manager(_healthy_result("my-server"))
        with patch("chuk_term.ui.output"):
            await cmd.execute(tool_manager=tm, server_name="my-server")
        tm.check_server_health.assert_called_once_with("my-server")

    async def test_server_name_from_list_args(self, cmd):
        tm = _make_tool_manager(_healthy_result("srv-a"))
        with patch("chuk_term.ui.output"):
            await cmd.execute(tool_manager=tm, args=["srv-a"])
        tm.check_server_health.assert_called_once_with("srv-a")

    async def test_server_name_from_string_args(self, cmd):
        tm = _make_tool_manager(_healthy_result("srv-b"))
        with patch("chuk_term.ui.output"):
            await cmd.execute(tool_manager=tm, args="srv-b")
        tm.check_server_health.assert_called_once_with("srv-b")

    async def test_whitespace_string_args_ignored(self, cmd):
        """Whitespace-only string args should NOT set server_name."""
        tm = _make_tool_manager(_healthy_result("srv"))
        with patch("chuk_term.ui.output"):
            await cmd.execute(tool_manager=tm, args="   ")
        tm.check_server_health.assert_called_once_with(None)

    async def test_empty_list_args_ignored(self, cmd):
        tm = _make_tool_manager(_healthy_result("srv"))
        with patch("chuk_term.ui.output"):
            await cmd.execute(tool_manager=tm, args=[])
        tm.check_server_health.assert_called_once_with(None)


# ---------------------------------------------------------------------------
# execute() — empty results
# ---------------------------------------------------------------------------


class TestHealthEmptyResults:
    async def test_empty_results_without_server_name(self, cmd):
        tm = _make_tool_manager({})
        result = await cmd.execute(tool_manager=tm)
        assert result.success is False
        assert "No servers available" in result.error

    async def test_empty_results_with_server_name(self, cmd):
        tm = _make_tool_manager({})
        result = await cmd.execute(tool_manager=tm, server_name="ghost")
        assert result.success is False
        assert "Server not found" in result.error
        assert "ghost" in result.error


# ---------------------------------------------------------------------------
# execute() — result rendering
# ---------------------------------------------------------------------------


class TestHealthResultRendering:
    async def test_all_healthy_returns_success(self, cmd):
        tm = _make_tool_manager(_healthy_result("srv"))
        with patch("chuk_term.ui.output") as mock_out:
            result = await cmd.execute(tool_manager=tm)
        assert result.success is True
        assert result.data is not None
        mock_out.success.assert_called()

    async def test_timeout_server_marks_unhealthy(self, cmd):
        tm = _make_tool_manager(_timeout_result("srv"))
        with patch("chuk_term.ui.output") as mock_out:
            result = await cmd.execute(tool_manager=tm)
        assert result.success is False
        mock_out.warning.assert_called()

    async def test_error_server_with_detail(self, cmd):
        tm = _make_tool_manager(_error_result("srv", "connection refused"))
        with patch("chuk_term.ui.output") as mock_out:
            result = await cmd.execute(tool_manager=tm)
        assert result.success is False
        # error() should be called; arg should contain "connection refused"
        calls = mock_out.error.call_args_list
        assert any("connection refused" in str(c) for c in calls)

    async def test_unhealthy_without_error_detail(self, cmd):
        tm = _make_tool_manager(_unhealthy_no_error("srv"))
        with patch("chuk_term.ui.output") as mock_out:
            result = await cmd.execute(tool_manager=tm)
        assert result.success is False
        mock_out.error.assert_called()

    async def test_mixed_statuses(self, cmd):
        mixed = {
            "srv1": {"status": "healthy", "ping_success": True},
            "srv2": {"status": "timeout", "ping_success": False},
            "srv3": {"status": "error", "ping_success": False, "error": "refused"},
        }
        tm = _make_tool_manager(mixed)
        with patch("chuk_term.ui.output") as mock_out:
            result = await cmd.execute(tool_manager=tm)
        assert result.success is False
        assert result.data == mixed
        mock_out.success.assert_called()
        mock_out.warning.assert_called()
        mock_out.error.assert_called()

    async def test_none_info_value_treated_as_unknown(self, cmd):
        """A None value in the results dict should not raise."""
        results = {"srv": None}
        tm = _make_tool_manager(results)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(tool_manager=tm)
        # None info → status "unknown", ping_ok False → error path
        assert result.success is False

    async def test_output_rule_and_info_called(self, cmd):
        tm = _make_tool_manager(_healthy_result("s"))
        with patch("chuk_term.ui.output") as mock_out:
            await cmd.execute(tool_manager=tm)
        mock_out.rule.assert_called_once()
        mock_out.info.assert_called()

    async def test_returns_data_dict(self, cmd):
        results = _healthy_result("srv")
        tm = _make_tool_manager(results)
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(tool_manager=tm)
        assert result.data == results


# ---------------------------------------------------------------------------
# execute() — exception handling
# ---------------------------------------------------------------------------


class TestHealthExceptionHandling:
    async def test_check_server_health_raises(self, cmd):
        tm = MagicMock()
        tm.check_server_health = AsyncMock(side_effect=Exception("network error"))
        result = await cmd.execute(tool_manager=tm)
        assert result.success is False
        assert "Health check failed" in result.error
        assert "network error" in result.error
