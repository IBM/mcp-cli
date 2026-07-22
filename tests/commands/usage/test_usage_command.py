# tests/commands/usage/test_usage_command.py
"""Tests for the UsageCommand (commands/usage/usage.py)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from mcp_cli.commands.usage.usage import UsageCommand
from mcp_cli.commands.base import CommandMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(
    turn_count: int = 3, summary: str = "3 turns, 500 tokens"
) -> MagicMock:
    t = MagicMock()
    t.turn_count = turn_count
    t.format_summary.return_value = summary
    return t


def _make_chat_context(tracker=None) -> MagicMock:
    ctx = MagicMock()
    ctx.token_tracker = tracker
    return ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cmd():
    return UsageCommand()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestUsageCommandProperties:
    def test_name(self, cmd):
        assert cmd.name == "usage"

    def test_aliases(self, cmd):
        aliases = cmd.aliases
        assert "tokens" in aliases
        assert "cost" in aliases

    def test_description(self, cmd):
        assert "token" in cmd.description.lower() or "usage" in cmd.description.lower()

    def test_help_text(self, cmd):
        text = cmd.help_text
        assert "/usage" in text

    def test_modes(self, cmd):
        assert cmd.modes == CommandMode.CHAT

    def test_parameters_is_empty(self, cmd):
        assert cmd.parameters == []


# ---------------------------------------------------------------------------
# execute() — guard: no chat context
# ---------------------------------------------------------------------------


class TestUsageNoContext:
    async def test_no_chat_context_kwarg(self, cmd):
        result = await cmd.execute()
        assert result.success is False
        assert "No chat context" in result.error

    async def test_chat_context_none(self, cmd):
        result = await cmd.execute(chat_context=None)
        assert result.success is False
        assert "No chat context" in result.error


# ---------------------------------------------------------------------------
# execute() — no token tracker / no turns
# ---------------------------------------------------------------------------


class TestUsageNoTracker:
    async def test_no_tracker_attribute(self, cmd):
        """token_tracker is None on the context."""
        ctx = _make_chat_context(tracker=None)
        with patch("mcp_cli.commands.usage.usage.output") as mock_out:
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True
        assert "No usage data" in result.output
        mock_out.info.assert_called_once()

    async def test_tracker_with_zero_turns(self, cmd):
        tracker = _make_tracker(turn_count=0)
        ctx = _make_chat_context(tracker=tracker)
        with patch("mcp_cli.commands.usage.usage.output") as mock_out:
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True
        assert "No usage data" in result.output
        mock_out.info.assert_called_once()


# ---------------------------------------------------------------------------
# execute() — with tracker that has turns
# ---------------------------------------------------------------------------


class TestUsageWithTracker:
    async def test_returns_summary_string(self, cmd):
        tracker = _make_tracker(turn_count=5, summary="5 turns, 1000 tokens")
        ctx = _make_chat_context(tracker=tracker)
        with patch("mcp_cli.commands.usage.usage.output"):
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True
        assert result.output == "5 turns, 1000 tokens"

    async def test_calls_format_summary(self, cmd):
        tracker = _make_tracker(turn_count=2)
        ctx = _make_chat_context(tracker=tracker)
        with patch("mcp_cli.commands.usage.usage.output"):
            await cmd.execute(chat_context=ctx)
        tracker.format_summary.assert_called_once()

    async def test_calls_output_panel_with_summary(self, cmd):
        tracker = _make_tracker(turn_count=1, summary="1 turn, 200 tokens")
        ctx = _make_chat_context(tracker=tracker)
        with patch("mcp_cli.commands.usage.usage.output") as mock_out:
            await cmd.execute(chat_context=ctx)
        mock_out.panel.assert_called_once_with(
            "1 turn, 200 tokens", title="Token Usage"
        )

    async def test_success_true_when_tracker_has_data(self, cmd):
        tracker = _make_tracker(turn_count=10, summary="summary text")
        ctx = _make_chat_context(tracker=tracker)
        with patch("mcp_cli.commands.usage.usage.output"):
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True

    async def test_output_equals_format_summary_return(self, cmd):
        expected = "42 turns, 99999 tokens"
        tracker = _make_tracker(turn_count=42, summary=expected)
        ctx = _make_chat_context(tracker=tracker)
        with patch("mcp_cli.commands.usage.usage.output"):
            result = await cmd.execute(chat_context=ctx)
        assert result.output == expected
