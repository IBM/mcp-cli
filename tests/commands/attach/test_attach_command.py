# tests/commands/attach/test_attach_command.py
"""Unit tests for the /attach slash command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mcp_cli.chat.attachments import AttachmentStaging
from mcp_cli.commands.attach.attach import AttachCommand
from mcp_cli.commands.base import CommandMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cmd():
    return AttachCommand()


@pytest.fixture
def ctx():
    """Mock ChatContext with a real AttachmentStaging."""
    mock = MagicMock()
    mock.attachment_staging = AttachmentStaging()
    return mock


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestAttachCommandProperties:
    def test_name(self, cmd):
        assert cmd.name == "attach"

    def test_aliases(self, cmd):
        assert "file" in cmd.aliases
        assert "image" in cmd.aliases

    def test_modes(self, cmd):
        assert cmd.modes == CommandMode.CHAT

    def test_description(self, cmd):
        assert "attach" in cmd.description.lower() or "file" in cmd.description.lower()

    def test_help_text(self, cmd):
        assert "/attach" in cmd.help_text
        assert ".png" in cmd.help_text


# ---------------------------------------------------------------------------
# Execute: no context
# ---------------------------------------------------------------------------


class TestAttachNoContext:
    @pytest.mark.asyncio
    async def test_no_context(self, cmd):
        result = await cmd.execute()
        assert not result.success
        assert "context" in result.error.lower()


# ---------------------------------------------------------------------------
# Execute: staging files
# ---------------------------------------------------------------------------


class TestAttachStageFiles:
    @pytest.mark.asyncio
    async def test_stage_image(self, cmd, ctx, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args=str(img))

        assert result.success
        assert ctx.attachment_staging.count == 1

    @pytest.mark.asyncio
    async def test_stage_text_file(self, cmd, ctx, tmp_path):
        txt = tmp_path / "code.py"
        txt.write_text("print('hello')")

        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args=str(txt))

        assert result.success
        assert ctx.attachment_staging.count == 1

    @pytest.mark.asyncio
    async def test_stage_multiple_files(self, cmd, ctx, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("alpha")
        f2.write_text("beta")

        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args=f"{f1} {f2}")

        assert result.success
        assert ctx.attachment_staging.count == 2

    @pytest.mark.asyncio
    async def test_stage_missing_file(self, cmd, ctx):
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args="/nonexistent/file.png")

        assert not result.success
        assert ctx.attachment_staging.count == 0

    @pytest.mark.asyncio
    async def test_stage_partial_failure(self, cmd, ctx, tmp_path):
        good = tmp_path / "ok.txt"
        good.write_text("good file")

        with patch("chuk_term.ui.output"):
            result = await cmd.execute(
                chat_context=ctx, args=f"{good} /nonexistent/bad.png"
            )

        # Partially successful
        assert result.success
        assert ctx.attachment_staging.count == 1
        assert "1 failed" in result.output

    @pytest.mark.asyncio
    async def test_args_as_list(self, cmd, ctx, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2")

        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args=[str(f)])

        assert result.success
        assert ctx.attachment_staging.count == 1


# ---------------------------------------------------------------------------
# Execute: list subcommand
# ---------------------------------------------------------------------------


class TestAttachList:
    @pytest.mark.asyncio
    async def test_list_empty(self, cmd, ctx):
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args="list")

        assert result.success
        assert "no" in result.output.lower() or "0" in result.output

    @pytest.mark.asyncio
    async def test_list_with_staged(self, cmd, ctx, tmp_path):
        f = tmp_path / "photo.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        with patch("chuk_term.ui.output"):
            await cmd.execute(chat_context=ctx, args=str(f))
            result = await cmd.execute(chat_context=ctx, args="list")

        assert result.success
        assert "photo.png" in result.output


# ---------------------------------------------------------------------------
# Execute: clear subcommand
# ---------------------------------------------------------------------------


class TestAttachClear:
    @pytest.mark.asyncio
    async def test_clear_empty(self, cmd, ctx):
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args="clear")

        assert result.success
        assert "nothing" in result.output.lower() or "0" in result.output.lower()

    @pytest.mark.asyncio
    async def test_clear_with_staged(self, cmd, ctx, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")

        with patch("chuk_term.ui.output"):
            await cmd.execute(chat_context=ctx, args=str(f))
            assert ctx.attachment_staging.count == 1

            result = await cmd.execute(chat_context=ctx, args="clear")

        assert result.success
        assert ctx.attachment_staging.count == 0
        assert "1" in result.output


# ---------------------------------------------------------------------------
# Execute: no args (usage)
# ---------------------------------------------------------------------------


class TestAttachUsage:
    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self, cmd, ctx):
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args="")

        assert result.success

    @pytest.mark.asyncio
    async def test_none_args_shows_usage(self, cmd, ctx):
        with patch("chuk_term.ui.output"):
            result = await cmd.execute(chat_context=ctx, args=None)

        assert result.success
