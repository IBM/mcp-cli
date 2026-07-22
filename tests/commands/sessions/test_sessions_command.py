# tests/commands/sessions/test_sessions_command.py
"""Tests for the SessionsCommand."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from mcp_cli.commands.sessions.sessions import SessionsCommand

# SessionStore is imported lazily inside execute(), so patch the canonical location.
_SESSION_STORE = "mcp_cli.chat.session_store.SessionStore"
# output and format_table are imported at module top level in sessions.py.
_OUTPUT = "mcp_cli.commands.sessions.sessions.output"
_FORMAT_TABLE = "mcp_cli.commands.sessions.sessions.format_table"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_meta(
    session_id: str = "sess-001",
    updated_at: str = "2024-01-01T00:00:00+00:00",
    provider: str = "openai",
    model: str = "gpt-4",
    message_count: int = 5,
) -> MagicMock:
    m = MagicMock()
    m.session_id = session_id
    m.updated_at = updated_at
    m.provider = provider
    m.model = model
    m.message_count = message_count
    return m


def _make_store(sessions=None, delete_result: bool = True) -> MagicMock:
    store = MagicMock()
    store.list_sessions.return_value = sessions if sessions is not None else []
    store.delete.return_value = delete_result
    return store


def _make_chat_context(
    save_path: str | None = "/tmp/sess.json",
    load_result: bool = True,
    has_save: bool = True,
    has_load: bool = True,
) -> MagicMock:
    ctx = MagicMock()
    if has_save:
        ctx.save_session = MagicMock(return_value=save_path)
    if not has_save and hasattr(ctx, "save_session"):
        del ctx.save_session
    if has_load:
        ctx.load_session = MagicMock(return_value=load_result)
    if not has_load and hasattr(ctx, "load_session"):
        del ctx.load_session
    return ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cmd():
    return SessionsCommand()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestSessionsCommandProperties:
    def test_name(self, cmd):
        assert cmd.name == "sessions"

    def test_aliases(self, cmd):
        assert "session" in cmd.aliases

    def test_description(self, cmd):
        assert "session" in cmd.description.lower()

    def test_help_text(self, cmd):
        text = cmd.help_text
        assert "/sessions" in text
        assert "list" in text
        assert "save" in text
        assert "load" in text
        assert "delete" in text

    def test_parameters(self, cmd):
        names = {p.name for p in cmd.parameters}
        assert "action" in names
        assert "session_id" in names

    def test_modes(self, cmd):
        from mcp_cli.commands.base import CommandMode

        assert cmd.modes == CommandMode.CHAT


# ---------------------------------------------------------------------------
# LIST action
# ---------------------------------------------------------------------------


class TestSessionsListAction:
    async def test_no_sessions(self, cmd):
        store = _make_store(sessions=[])
        with patch(_SESSION_STORE, return_value=store):
            with patch(_OUTPUT) as mock_out:
                result = await cmd.execute(args="list")
        assert result.success is True
        assert "No saved sessions" in result.output
        mock_out.info.assert_called_once()

    async def test_list_with_sessions(self, cmd):
        metas = [
            _make_session_meta("s1", provider="openai", model="gpt-4", message_count=3),
            _make_session_meta(
                "s2", provider="anthropic", model="claude-3", message_count=7
            ),
        ]
        store = _make_store(sessions=metas)
        with patch(_SESSION_STORE, return_value=store):
            with patch(_OUTPUT) as mock_out:
                with patch(_FORMAT_TABLE, return_value=MagicMock()) as mock_fmt:
                    result = await cmd.execute(args="list")
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 2
        ids = [row["ID"] for row in result.data]
        assert "s1" in ids and "s2" in ids
        mock_fmt.assert_called_once()
        mock_out.print_table.assert_called_once()

    async def test_list_default_when_empty_args(self, cmd):
        """No args at all defaults to SessionAction.LIST."""
        store = _make_store(sessions=[])
        with patch(_SESSION_STORE, return_value=store):
            with patch(_OUTPUT):
                result = await cmd.execute()
        assert result.success is True

    async def test_provider_model_formatting(self, cmd):
        meta = _make_session_meta("s1", provider="openai", model="gpt-4")
        store = _make_store(sessions=[meta])
        with patch(_SESSION_STORE, return_value=store):
            with patch(_OUTPUT):
                with patch(_FORMAT_TABLE, return_value=MagicMock()):
                    result = await cmd.execute(args="list")
        assert result.data[0]["Provider/Model"] == "openai/gpt-4"

    async def test_message_count_stored_as_string(self, cmd):
        meta = _make_session_meta("s1", message_count=42)
        store = _make_store(sessions=[meta])
        with patch(_SESSION_STORE, return_value=store):
            with patch(_OUTPUT):
                with patch(_FORMAT_TABLE, return_value=MagicMock()):
                    result = await cmd.execute(args="list")
        assert result.data[0]["Messages"] == "42"

    async def test_updated_at_truncated_to_19_chars(self, cmd):
        meta = _make_session_meta("s1", updated_at="2024-01-15T10:30:45.123456+00:00")
        store = _make_store(sessions=[meta])
        with patch(_SESSION_STORE, return_value=store):
            with patch(_OUTPUT):
                with patch(_FORMAT_TABLE, return_value=MagicMock()):
                    result = await cmd.execute(args="list")
        assert len(result.data[0]["Updated"]) == 19


# ---------------------------------------------------------------------------
# SAVE action
# ---------------------------------------------------------------------------


class TestSessionsSaveAction:
    async def test_save_no_chat_context(self, cmd):
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            result = await cmd.execute(args="save")
        assert result.success is False
        assert "No chat context" in result.error

    async def test_save_successful(self, cmd):
        ctx = _make_chat_context(save_path="/tmp/session.json")
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            with patch(_OUTPUT) as mock_out:
                result = await cmd.execute(args="save", chat_context=ctx)
        assert result.success is True
        assert "/tmp/session.json" in result.output
        mock_out.success.assert_called_once()

    async def test_save_returns_none_path_gives_error(self, cmd):
        ctx = _make_chat_context(save_path=None)
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            result = await cmd.execute(args="save", chat_context=ctx)
        assert result.success is False
        assert "Failed to save session" in result.error

    async def test_save_context_without_save_session_attr(self, cmd):
        ctx = MagicMock(spec=[])  # no attributes at all
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            result = await cmd.execute(args="save", chat_context=ctx)
        assert result.success is False


# ---------------------------------------------------------------------------
# LOAD action
# ---------------------------------------------------------------------------


class TestSessionsLoadAction:
    async def test_load_no_session_id(self, cmd):
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            result = await cmd.execute(args="load")
        assert result.success is False
        assert "Session ID required" in result.error

    async def test_load_no_chat_context(self, cmd):
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            result = await cmd.execute(args="load abc123")
        assert result.success is False
        assert "No chat context" in result.error

    async def test_load_successful(self, cmd):
        ctx = _make_chat_context(load_result=True)
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            with patch(_OUTPUT) as mock_out:
                result = await cmd.execute(args="load abc123", chat_context=ctx)
        assert result.success is True
        assert "abc123" in result.output
        mock_out.success.assert_called_once()

    async def test_load_fails(self, cmd):
        ctx = _make_chat_context(load_result=False)
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            result = await cmd.execute(args="load abc123", chat_context=ctx)
        assert result.success is False
        assert "Failed to load session" in result.error

    async def test_load_context_without_load_session_attr(self, cmd):
        ctx = MagicMock(spec=[])  # no attributes
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            result = await cmd.execute(args="load abc123", chat_context=ctx)
        assert result.success is False


# ---------------------------------------------------------------------------
# DELETE action
# ---------------------------------------------------------------------------


class TestSessionsDeleteAction:
    async def test_delete_no_session_id(self, cmd):
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            result = await cmd.execute(args="delete")
        assert result.success is False
        assert "Session ID required" in result.error

    async def test_delete_found(self, cmd):
        store = _make_store(delete_result=True)
        with patch(_SESSION_STORE, return_value=store):
            with patch(_OUTPUT) as mock_out:
                result = await cmd.execute(args="delete sess-007")
        assert result.success is True
        assert "sess-007" in result.output
        store.delete.assert_called_once_with("sess-007")
        mock_out.success.assert_called_once()

    async def test_delete_not_found(self, cmd):
        store = _make_store(delete_result=False)
        with patch(_SESSION_STORE, return_value=store):
            result = await cmd.execute(args="delete sess-007")
        assert result.success is False
        assert "Session not found" in result.error


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------


class TestSessionsUnknownAction:
    async def test_unknown_action_returns_error(self, cmd):
        store = _make_store()
        with patch(_SESSION_STORE, return_value=store):
            result = await cmd.execute(args="bogusaction")
        assert result.success is False
        assert "Unknown action" in result.error
