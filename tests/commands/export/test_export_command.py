# tests/commands/export/test_export_command.py
"""Tests for the ExportCommand (commands/export/export.py)."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mcp_cli.commands.export.export import ExportCommand
from mcp_cli.commands.base import CommandMode

# Exporters are imported lazily inside execute():
#   from mcp_cli.chat.exporters import MarkdownExporter, JSONExporter
# so we patch them at their canonical source location.
_MARKDOWN_EXPORTER = "mcp_cli.chat.exporters.MarkdownExporter"
_JSON_EXPORTER = "mcp_cli.chat.exporters.JSONExporter"
# output is imported at module level in export.py.
_OUTPUT = "mcp_cli.commands.export.export.output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message_dict(role: str = "user", content: str = "hello") -> dict:
    return {"role": role, "content": content}


def _make_message_obj(role: str = "user", content: str = "hello") -> MagicMock:
    """A message object that has a to_dict() method."""
    m = MagicMock()
    m.to_dict.return_value = {"role": role, "content": content}
    return m


def _make_tracker(
    turn_count: int = 3, total_input: int = 100, total_output: int = 200
) -> MagicMock:
    t = MagicMock()
    t.turn_count = turn_count
    t.total_input = total_input
    t.total_output = total_output
    t.total_tokens = total_input + total_output
    return t


def _make_chat_context(
    messages=None,
    session_id: str = "test-session",
    provider: str = "openai",
    model: str = "gpt-4",
    tracker=None,
) -> MagicMock:
    ctx = MagicMock()
    ctx.get_conversation_history.return_value = messages or []
    ctx.session_id = session_id
    ctx.provider = provider
    ctx.model = model
    ctx.token_tracker = tracker
    return ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cmd():
    return ExportCommand()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestExportCommandProperties:
    def test_name(self, cmd):
        assert cmd.name == "export"

    def test_aliases(self, cmd):
        assert "save-chat" in cmd.aliases

    def test_description(self, cmd):
        desc = cmd.description.lower()
        assert "export" in desc or "markdown" in desc or "json" in desc

    def test_help_text(self, cmd):
        text = cmd.help_text
        assert "markdown" in text.lower()
        assert "json" in text.lower()

    def test_modes(self, cmd):
        assert cmd.modes == CommandMode.CHAT

    def test_parameters(self, cmd):
        names = {p.name for p in cmd.parameters}
        assert "format" in names
        assert "filename" in names


# ---------------------------------------------------------------------------
# execute() — guard: no chat context
# ---------------------------------------------------------------------------


class TestExportGuards:
    async def test_no_chat_context(self, cmd):
        result = await cmd.execute()
        assert result.success is False
        assert "No chat context" in result.error

    async def test_get_history_raises(self, cmd):
        ctx = MagicMock()
        ctx.get_conversation_history.side_effect = RuntimeError("db error")
        result = await cmd.execute(chat_context=ctx)
        assert result.success is False
        assert "Failed to get history" in result.error
        assert "db error" in result.error


# ---------------------------------------------------------------------------
# execute() — empty messages
# ---------------------------------------------------------------------------


class TestExportEmptyMessages:
    async def test_no_messages_returns_info(self, cmd):
        ctx = _make_chat_context(messages=[])
        with patch(_OUTPUT) as mock_out:
            result = await cmd.execute(chat_context=ctx)
        assert result.success is True
        assert "No messages" in result.output
        mock_out.info.assert_called_once()


# ---------------------------------------------------------------------------
# execute() — message normalisation (to_dict)
# ---------------------------------------------------------------------------


class TestExportMessageNormalisation:
    async def test_messages_with_to_dict_are_converted(self, cmd):
        """Objects implementing to_dict() are converted before export."""
        msg_obj = _make_message_obj("user", "hi")
        ctx = _make_chat_context(messages=[msg_obj])
        with patch(_MARKDOWN_EXPORTER + ".export", return_value="# Chat Export\n"):
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    result = await cmd.execute(chat_context=ctx)
        msg_obj.to_dict.assert_called_once()
        assert result.success is True

    async def test_plain_dict_messages_passed_as_is(self, cmd):
        raw_msg = _make_message_dict("user", "hello")
        ctx = _make_chat_context(messages=[raw_msg])
        with patch(_MARKDOWN_EXPORTER + ".export", return_value="# Chat") as mock_exp:
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    result = await cmd.execute(chat_context=ctx)
        assert result.success is True
        called_messages = mock_exp.call_args[0][0]
        assert called_messages[0] == raw_msg


# ---------------------------------------------------------------------------
# execute() — metadata & token usage
# ---------------------------------------------------------------------------


class TestExportMetadata:
    async def test_metadata_uses_context_attrs(self, cmd):
        ctx = _make_chat_context(
            messages=[_make_message_dict()],
            session_id="sess-abc",
            provider="anthropic",
            model="claude-3",
        )
        with patch(_MARKDOWN_EXPORTER + ".export", return_value="# Chat") as mock_exp:
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx)
        metadata = mock_exp.call_args[0][1]
        assert metadata["session_id"] == "sess-abc"
        assert metadata["provider"] == "anthropic"
        assert metadata["model"] == "claude-3"

    async def test_token_usage_included_when_tracker_has_turns(self, cmd):
        tracker = _make_tracker(turn_count=2, total_input=50, total_output=100)
        ctx = _make_chat_context(
            messages=[_make_message_dict()],
            tracker=tracker,
        )
        with patch(_JSON_EXPORTER + ".export", return_value="{}") as mock_exp:
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx, args="json")
        token_usage = mock_exp.call_args[0][2]
        assert token_usage is not None
        assert token_usage["total_input"] == 50
        assert token_usage["turn_count"] == 2

    async def test_no_token_usage_when_tracker_is_none(self, cmd):
        ctx = _make_chat_context(messages=[_make_message_dict()], tracker=None)
        ctx.token_tracker = None
        with patch(_JSON_EXPORTER + ".export", return_value="{}") as mock_exp:
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx, args="json")
        token_usage = mock_exp.call_args[0][2]
        assert token_usage is None

    async def test_no_token_usage_when_turn_count_is_zero(self, cmd):
        tracker = _make_tracker(turn_count=0)
        ctx = _make_chat_context(messages=[_make_message_dict()], tracker=tracker)
        with patch(_JSON_EXPORTER + ".export", return_value="{}") as mock_exp:
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx, args="json")
        token_usage = mock_exp.call_args[0][2]
        assert token_usage is None


# ---------------------------------------------------------------------------
# execute() — format selection
# ---------------------------------------------------------------------------


class TestExportFormatSelection:
    async def test_default_format_is_markdown(self, cmd):
        ctx = _make_chat_context(messages=[_make_message_dict()])
        with patch(_MARKDOWN_EXPORTER + ".export", return_value="# Chat") as mock_md:
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    result = await cmd.execute(chat_context=ctx)
        mock_md.assert_called_once()
        assert result.success is True

    async def test_json_format_selected(self, cmd):
        ctx = _make_chat_context(messages=[_make_message_dict()])
        with patch(_JSON_EXPORTER + ".export", return_value="{}") as mock_json:
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    result = await cmd.execute(chat_context=ctx, args="json")
        mock_json.assert_called_once()
        assert result.success is True

    async def test_json_prefix_matches(self, cmd):
        """'json' prefix check: e.g. 'json-pretty' still uses JSONExporter."""
        ctx = _make_chat_context(messages=[_make_message_dict()])
        with patch(_JSON_EXPORTER + ".export", return_value="{}") as mock_json:
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx, args="json-extra")
        mock_json.assert_called_once()

    async def test_markdown_explicit_arg(self, cmd):
        ctx = _make_chat_context(messages=[_make_message_dict()])
        with patch(_MARKDOWN_EXPORTER + ".export", return_value="# Chat") as mock_md:
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx, args="markdown")
        mock_md.assert_called_once()


# ---------------------------------------------------------------------------
# execute() — filename defaults
# ---------------------------------------------------------------------------


class TestExportFilenameDefaults:
    async def test_markdown_default_filename_uses_session_id(self, cmd):
        ctx = _make_chat_context(messages=[_make_message_dict()], session_id="sess-xyz")
        written_paths: list[str] = []

        def capture_write(self, content, encoding=None):
            written_paths.append(str(self))

        with patch(_MARKDOWN_EXPORTER + ".export", return_value="# Chat"):
            with patch.object(Path, "write_text", capture_write):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx)

        assert any("sess-xyz" in p and p.endswith(".md") for p in written_paths)

    async def test_json_default_filename_uses_session_id(self, cmd):
        ctx = _make_chat_context(messages=[_make_message_dict()], session_id="sess-abc")
        written_paths: list[str] = []

        def capture_write(self, content, encoding=None):
            written_paths.append(str(self))

        with patch(_JSON_EXPORTER + ".export", return_value="{}"):
            with patch.object(Path, "write_text", capture_write):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx, args="json")

        assert any("sess-abc" in p and p.endswith(".json") for p in written_paths)

    async def test_custom_filename_honoured(self, cmd):
        ctx = _make_chat_context(messages=[_make_message_dict()])
        written_paths: list[str] = []

        def capture_write(self, content, encoding=None):
            written_paths.append(str(self))

        with patch(_MARKDOWN_EXPORTER + ".export", return_value="# Chat"):
            with patch.object(Path, "write_text", capture_write):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx, args="markdown myfile.md")

        assert any("myfile.md" in p for p in written_paths)

    async def test_json_custom_filename_skips_default(self, cmd):
        """When a filename is provided with json format, default filename is NOT used."""
        ctx = _make_chat_context(
            messages=[_make_message_dict()], session_id="sess-ignored"
        )
        written_paths: list[str] = []

        def capture_write(self, content, encoding=None):
            written_paths.append(str(self))

        with patch(_JSON_EXPORTER + ".export", return_value="{}"):
            with patch.object(Path, "write_text", capture_write):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx, args="json custom.json")

        # Should write to custom.json, NOT chat-sess-ignored.json
        assert any("custom.json" in p for p in written_paths)
        assert not any("sess-ignored" in p for p in written_paths)


# ---------------------------------------------------------------------------
# execute() — file write success / failure
# ---------------------------------------------------------------------------


class TestExportFileWrite:
    async def test_successful_write_returns_success(self, cmd):
        ctx = _make_chat_context(messages=[_make_message_dict()])
        with patch(_MARKDOWN_EXPORTER + ".export", return_value="# Chat"):
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT) as mock_out:
                    result = await cmd.execute(chat_context=ctx)
        assert result.success is True
        mock_out.success.assert_called_once()

    async def test_write_error_returns_failure(self, cmd):
        ctx = _make_chat_context(messages=[_make_message_dict()])
        with patch(_MARKDOWN_EXPORTER + ".export", return_value="# Chat"):
            with patch.object(
                Path, "write_text", side_effect=PermissionError("access denied")
            ):
                result = await cmd.execute(chat_context=ctx)
        assert result.success is False
        assert "Failed to write file" in result.error
        assert "access denied" in result.error

    async def test_context_missing_attrs_uses_unknown_defaults(self, cmd):
        """A context object with no session_id/provider/model attrs gets 'unknown' defaults."""

        class MinimalContext:
            def get_conversation_history(self):
                return [_make_message_dict()]

            token_tracker = None

        ctx = MinimalContext()

        with patch(_MARKDOWN_EXPORTER + ".export", return_value="# Chat") as mock_exp:
            with patch.object(Path, "write_text"):
                with patch(_OUTPUT):
                    await cmd.execute(chat_context=ctx)

        metadata = mock_exp.call_args[0][1]
        assert metadata["session_id"] == "unknown"
        assert metadata["provider"] == "unknown"
        assert metadata["model"] == "unknown"
