# tests/chat/test_chat_handler_coverage.py
"""Tests for mcp_cli.chat.chat_handler achieving >90% coverage."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui(
    is_streaming=False,
    tools_running=False,
):
    """Build a mock ChatUIManager."""
    ui = MagicMock()
    ui.is_streaming_response = is_streaming
    ui.tools_running = tools_running
    ui.verbose_mode = False
    ui.streaming_handler = MagicMock()
    ui.streaming_handler.interrupt_streaming = MagicMock()

    ui.get_user_input = AsyncMock(return_value="exit")
    ui.handle_command = AsyncMock(return_value=True)
    ui.print_user_message = MagicMock()
    ui.interrupt_streaming = MagicMock()
    ui._interrupt_now = MagicMock()
    ui.stop_streaming_response_sync = MagicMock()
    ui.stop_tool_calls = MagicMock()
    ui.cleanup = MagicMock()
    return ui


def _make_ctx(exit_requested=False):
    """Build a mock ChatContext."""
    ctx = MagicMock()
    ctx.provider = "openai"
    ctx.model = "gpt-4"
    ctx.exit_requested = exit_requested
    ctx.add_user_message = AsyncMock()
    # Real staging so multi-modal processing works correctly (drain returns [])
    from mcp_cli.chat.attachments import AttachmentStaging

    ctx.attachment_staging = AttachmentStaging()
    return ctx


def _make_convo():
    """Build a mock ConversationProcessor."""
    convo = MagicMock()
    convo.process_conversation = AsyncMock()
    return convo


# ===========================================================================
# Tests for handle_chat_mode
# ===========================================================================


class TestHandleChatMode:
    """Tests for handle_chat_mode function."""

    @pytest.mark.asyncio
    async def test_happy_path(self):
        """Normal execution returns True."""
        tool_mgr = MagicMock()
        tool_mgr.close = AsyncMock()
        tool_mgr.get_tool_count = MagicMock(return_value=5)

        mock_ctx = _make_ctx()
        mock_ctx.initialize = AsyncMock(return_value=True)

        mock_app_ctx = MagicMock()
        mock_app_ctx.model_manager = MagicMock()
        mock_app_ctx.initialize = AsyncMock()

        ui = _make_ui()
        convo = _make_convo()

        with (
            patch("mcp_cli.chat.chat_handler.initialize_config"),
            patch(
                "mcp_cli.chat.chat_handler.initialize_context",
                return_value=mock_app_ctx,
            ),
            patch("mcp_cli.chat.chat_handler.output") as mock_output,
            patch("mcp_cli.chat.chat_handler.clear_screen"),
            patch("mcp_cli.chat.chat_handler.display_chat_banner"),
            patch("mcp_cli.chat.chat_handler.ChatContext") as MockCC,
            patch("mcp_cli.chat.chat_handler.ChatUIManager", return_value=ui),
            patch(
                "mcp_cli.chat.chat_handler.ConversationProcessor", return_value=convo
            ),
            patch(
                "mcp_cli.chat.chat_handler._run_enhanced_chat_loop",
                new_callable=AsyncMock,
            ),
            patch("mcp_cli.chat.chat_handler._safe_cleanup", new_callable=AsyncMock),
        ):
            mock_output.loading.return_value.__enter__ = MagicMock(return_value=None)
            mock_output.loading.return_value.__exit__ = MagicMock(return_value=False)
            MockCC.create.return_value = mock_ctx

            from mcp_cli.chat.chat_handler import handle_chat_mode

            result = await handle_chat_mode(tool_mgr, provider="openai", model="gpt-4")
            assert result is True

    @pytest.mark.asyncio
    async def test_ctx_init_fails(self):
        """Returns False when ctx.initialize returns False."""
        tool_mgr = MagicMock()
        tool_mgr.close = AsyncMock()

        mock_ctx = _make_ctx()
        mock_ctx.initialize = AsyncMock(return_value=False)

        mock_app_ctx = MagicMock()
        mock_app_ctx.model_manager = MagicMock()

        with (
            patch("mcp_cli.chat.chat_handler.initialize_config"),
            patch(
                "mcp_cli.chat.chat_handler.initialize_context",
                return_value=mock_app_ctx,
            ),
            patch("mcp_cli.chat.chat_handler.output") as mock_output,
            patch("mcp_cli.chat.chat_handler.clear_screen"),
            patch("mcp_cli.chat.chat_handler.ChatContext") as MockCC,
        ):
            mock_output.loading.return_value.__enter__ = MagicMock(return_value=None)
            mock_output.loading.return_value.__exit__ = MagicMock(return_value=False)
            mock_output.error = MagicMock()
            MockCC.create.return_value = mock_ctx

            from mcp_cli.chat.chat_handler import handle_chat_mode

            result = await handle_chat_mode(tool_mgr)
            assert result is False

    @pytest.mark.asyncio
    async def test_exception_returns_false(self):
        """Returns False on unexpected exception."""
        tool_mgr = MagicMock()
        tool_mgr.close = AsyncMock()

        with (
            patch(
                "mcp_cli.chat.chat_handler.initialize_config",
                side_effect=RuntimeError("boom"),
            ),
            patch("mcp_cli.chat.chat_handler.display_error_banner"),
        ):
            from mcp_cli.chat.chat_handler import handle_chat_mode

            result = await handle_chat_mode(tool_mgr)
            assert result is False

    @pytest.mark.asyncio
    async def test_tool_count_via_list_tools(self):
        """Tool count obtained via list_tools when get_tool_count absent."""
        tool_mgr = MagicMock(spec=[])
        tool_mgr.list_tools = MagicMock(return_value=["a", "b"])
        tool_mgr.close = AsyncMock()

        mock_ctx = _make_ctx()
        mock_ctx.initialize = AsyncMock(return_value=True)

        mock_app_ctx = MagicMock()
        mock_app_ctx.model_manager = MagicMock()
        mock_app_ctx.initialize = AsyncMock()

        with (
            patch("mcp_cli.chat.chat_handler.initialize_config"),
            patch(
                "mcp_cli.chat.chat_handler.initialize_context",
                return_value=mock_app_ctx,
            ),
            patch("mcp_cli.chat.chat_handler.output") as mock_output,
            patch("mcp_cli.chat.chat_handler.clear_screen"),
            patch("mcp_cli.chat.chat_handler.display_chat_banner"),
            patch("mcp_cli.chat.chat_handler.ChatContext") as MockCC,
            patch("mcp_cli.chat.chat_handler.ChatUIManager", return_value=_make_ui()),
            patch(
                "mcp_cli.chat.chat_handler.ConversationProcessor",
                return_value=_make_convo(),
            ),
            patch(
                "mcp_cli.chat.chat_handler._run_enhanced_chat_loop",
                new_callable=AsyncMock,
            ),
            patch("mcp_cli.chat.chat_handler._safe_cleanup", new_callable=AsyncMock),
        ):
            mock_output.loading.return_value.__enter__ = MagicMock(return_value=None)
            mock_output.loading.return_value.__exit__ = MagicMock(return_value=False)
            MockCC.create.return_value = mock_ctx

            from mcp_cli.chat.chat_handler import handle_chat_mode

            result = await handle_chat_mode(tool_mgr, api_base="http://localhost")
            assert result is True

    @pytest.mark.asyncio
    async def test_tool_count_via_private_tools(self):
        """Tool count obtained via _tools attribute when other methods absent."""
        tool_mgr = MagicMock(spec=[])
        tool_mgr._tools = ["x", "y", "z"]
        tool_mgr.close = AsyncMock()

        mock_ctx = _make_ctx()
        mock_ctx.initialize = AsyncMock(return_value=True)

        mock_app_ctx = MagicMock()
        mock_app_ctx.model_manager = MagicMock()
        mock_app_ctx.initialize = AsyncMock()

        with (
            patch("mcp_cli.chat.chat_handler.initialize_config"),
            patch(
                "mcp_cli.chat.chat_handler.initialize_context",
                return_value=mock_app_ctx,
            ),
            patch("mcp_cli.chat.chat_handler.output") as mock_output,
            patch("mcp_cli.chat.chat_handler.clear_screen"),
            patch("mcp_cli.chat.chat_handler.display_chat_banner"),
            patch("mcp_cli.chat.chat_handler.ChatContext") as MockCC,
            patch("mcp_cli.chat.chat_handler.ChatUIManager", return_value=_make_ui()),
            patch(
                "mcp_cli.chat.chat_handler.ConversationProcessor",
                return_value=_make_convo(),
            ),
            patch(
                "mcp_cli.chat.chat_handler._run_enhanced_chat_loop",
                new_callable=AsyncMock,
            ),
            patch("mcp_cli.chat.chat_handler._safe_cleanup", new_callable=AsyncMock),
        ):
            mock_output.loading.return_value.__enter__ = MagicMock(return_value=None)
            mock_output.loading.return_value.__exit__ = MagicMock(return_value=False)
            MockCC.create.return_value = mock_ctx

            from mcp_cli.chat.chat_handler import handle_chat_mode

            result = await handle_chat_mode(tool_mgr)
            assert result is True

    @pytest.mark.asyncio
    async def test_tool_count_fallback_available(self):
        """Tool count shows 'Available' when no known method exists."""
        tool_mgr = MagicMock(spec=[])
        tool_mgr.close = AsyncMock()
        # Remove all known tool-count attributes
        del tool_mgr.get_tool_count
        del tool_mgr.list_tools
        del tool_mgr._tools

        mock_ctx = _make_ctx()
        mock_ctx.initialize = AsyncMock(return_value=True)

        mock_app_ctx = MagicMock()
        mock_app_ctx.model_manager = MagicMock()
        mock_app_ctx.initialize = AsyncMock()

        with (
            patch("mcp_cli.chat.chat_handler.initialize_config"),
            patch(
                "mcp_cli.chat.chat_handler.initialize_context",
                return_value=mock_app_ctx,
            ),
            patch("mcp_cli.chat.chat_handler.output") as mock_output,
            patch("mcp_cli.chat.chat_handler.clear_screen"),
            patch("mcp_cli.chat.chat_handler.display_chat_banner"),
            patch("mcp_cli.chat.chat_handler.ChatContext") as MockCC,
            patch("mcp_cli.chat.chat_handler.ChatUIManager", return_value=_make_ui()),
            patch(
                "mcp_cli.chat.chat_handler.ConversationProcessor",
                return_value=_make_convo(),
            ),
            patch(
                "mcp_cli.chat.chat_handler._run_enhanced_chat_loop",
                new_callable=AsyncMock,
            ),
            patch("mcp_cli.chat.chat_handler._safe_cleanup", new_callable=AsyncMock),
        ):
            mock_output.loading.return_value.__enter__ = MagicMock(return_value=None)
            mock_output.loading.return_value.__exit__ = MagicMock(return_value=False)
            MockCC.create.return_value = mock_ctx

            from mcp_cli.chat.chat_handler import handle_chat_mode

            result = await handle_chat_mode(tool_mgr)
            assert result is True

    @pytest.mark.asyncio
    async def test_tool_manager_close_error_is_logged(self):
        """Error closing ToolManager is logged but does not crash."""
        tool_mgr = MagicMock()
        tool_mgr.close = AsyncMock(side_effect=RuntimeError("close failed"))

        with (
            patch(
                "mcp_cli.chat.chat_handler.initialize_config",
                side_effect=RuntimeError("x"),
            ),
            patch("mcp_cli.chat.chat_handler.display_error_banner"),
        ):
            from mcp_cli.chat.chat_handler import handle_chat_mode

            result = await handle_chat_mode(tool_mgr)
            assert result is False


# ===========================================================================
# Tests for handle_chat_mode_for_testing
# ===========================================================================


class TestHandleChatModeForTesting:
    """Tests for handle_chat_mode_for_testing."""

    @pytest.mark.asyncio
    async def test_happy_path(self):
        """Normal test-mode execution returns True."""
        stream_mgr = MagicMock()

        mock_ctx = _make_ctx()
        mock_ctx.initialize = AsyncMock(return_value=True)

        with (
            patch("mcp_cli.chat.chat_handler.output") as mock_output,
            patch("mcp_cli.chat.chat_handler.clear_screen"),
            patch("mcp_cli.chat.chat_handler.display_chat_banner"),
            patch("mcp_cli.chat.chat_handler.TestChatContext") as MockTC,
            patch("mcp_cli.chat.chat_handler.ChatUIManager", return_value=_make_ui()),
            patch(
                "mcp_cli.chat.chat_handler.ConversationProcessor",
                return_value=_make_convo(),
            ),
            patch(
                "mcp_cli.chat.chat_handler._run_enhanced_chat_loop",
                new_callable=AsyncMock,
            ),
            patch("mcp_cli.chat.chat_handler._safe_cleanup", new_callable=AsyncMock),
        ):
            mock_output.loading.return_value.__enter__ = MagicMock(return_value=None)
            mock_output.loading.return_value.__exit__ = MagicMock(return_value=False)
            MockTC.create_for_testing.return_value = mock_ctx

            from mcp_cli.chat.chat_handler import handle_chat_mode_for_testing

            result = await handle_chat_mode_for_testing(
                stream_mgr, provider="test", model="m"
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_init_fails(self):
        """Returns False when test context initialization fails."""
        stream_mgr = MagicMock()

        mock_ctx = _make_ctx()
        mock_ctx.initialize = AsyncMock(return_value=False)

        with (
            patch("mcp_cli.chat.chat_handler.output") as mock_output,
            patch("mcp_cli.chat.chat_handler.TestChatContext") as MockTC,
        ):
            mock_output.loading.return_value.__enter__ = MagicMock(return_value=None)
            mock_output.loading.return_value.__exit__ = MagicMock(return_value=False)
            mock_output.error = MagicMock()
            MockTC.create_for_testing.return_value = mock_ctx

            from mcp_cli.chat.chat_handler import handle_chat_mode_for_testing

            result = await handle_chat_mode_for_testing(stream_mgr)
            assert result is False

    @pytest.mark.asyncio
    async def test_exception_returns_false(self):
        """Returns False on unexpected exception."""
        with (
            patch("mcp_cli.chat.chat_handler.output") as mock_output,
            patch("mcp_cli.chat.chat_handler.TestChatContext") as MockTC,
            patch("mcp_cli.chat.chat_handler.display_error_banner"),
        ):
            mock_output.loading.return_value.__enter__ = MagicMock(return_value=None)
            mock_output.loading.return_value.__exit__ = MagicMock(return_value=False)
            MockTC.create_for_testing.side_effect = RuntimeError("boom")

            from mcp_cli.chat.chat_handler import handle_chat_mode_for_testing

            result = await handle_chat_mode_for_testing(MagicMock())
            assert result is False


# ===========================================================================
# Tests for _run_enhanced_chat_loop
# ===========================================================================


class TestRunEnhancedChatLoop:
    """Tests for _run_enhanced_chat_loop."""

    @pytest.mark.asyncio
    async def test_exit_command(self):
        """Loop exits on 'exit' command."""
        ui = _make_ui()
        ui.get_user_input = AsyncMock(return_value="exit")
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)

    @pytest.mark.asyncio
    async def test_quit_command(self):
        """Loop exits on 'quit' command."""
        ui = _make_ui()
        ui.get_user_input = AsyncMock(return_value="quit")
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)

    @pytest.mark.asyncio
    async def test_empty_message_skipped(self):
        """Empty messages are skipped."""
        ui = _make_ui()
        ui.get_user_input = AsyncMock(side_effect=["", "exit"])
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)
            # convo should not be called for empty message
            assert convo.process_conversation.call_count == 0

    @pytest.mark.asyncio
    async def test_slash_command_handled(self):
        """Slash commands are dispatched to ui.handle_command."""
        ui = _make_ui()
        ui.get_user_input = AsyncMock(side_effect=["/help", "exit"])
        ui.handle_command = AsyncMock(return_value=True)
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)
            ui.handle_command.assert_called_once_with("/help")

    @pytest.mark.asyncio
    async def test_slash_command_exit_requested(self):
        """Loop exits when ctx.exit_requested becomes True after command."""
        ui = _make_ui()
        ui.get_user_input = AsyncMock(return_value="/exit")
        ui.handle_command = AsyncMock(return_value=True)
        ctx = _make_ctx()
        ctx.exit_requested = True  # Exit is requested after the command
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)

    @pytest.mark.asyncio
    async def test_slash_command_not_handled(self):
        """When command is not handled, it falls through to conversation."""
        ui = _make_ui()
        ui.get_user_input = AsyncMock(side_effect=["/unknown", "exit"])
        ui.handle_command = AsyncMock(return_value=False)
        ui.verbose_mode = True
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)
            convo.process_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_interrupt_streaming(self):
        """Interrupt command when streaming interrupts streaming."""
        ui = _make_ui(is_streaming=True)
        ui.get_user_input = AsyncMock(side_effect=["/interrupt", "exit"])
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)
            ui.interrupt_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_interrupt_tools_running(self):
        """Interrupt command when tools running calls _interrupt_now."""
        ui = _make_ui(tools_running=True)
        ui.is_streaming_response = False
        ui.get_user_input = AsyncMock(side_effect=["/stop", "exit"])
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)
            ui._interrupt_now.assert_called_once()

    @pytest.mark.asyncio
    async def test_interrupt_nothing_running(self):
        """Interrupt command when nothing is running shows info."""
        ui = _make_ui()
        ui.is_streaming_response = False
        ui.tools_running = False
        ui.get_user_input = AsyncMock(side_effect=["/cancel", "exit"])
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output") as mock_output:
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)
            mock_output.info.assert_called()

    @pytest.mark.asyncio
    async def test_normal_message_with_verbose(self):
        """Normal message with verbose mode prints user message."""
        ui = _make_ui()
        ui.verbose_mode = True
        ui.get_user_input = AsyncMock(side_effect=["Hello", "exit"])
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)
            ui.print_user_message.assert_called_once_with("Hello")
            ctx.add_user_message.assert_called_once_with("Hello")
            convo.process_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_while_streaming(self):
        """KeyboardInterrupt during streaming is caught and loop continues."""
        ui = _make_ui(is_streaming=True)
        call_count = [0]

        async def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyboardInterrupt()
            return "exit"

        ui.get_user_input = side_effect
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_while_tools(self):
        """KeyboardInterrupt during tool execution is caught."""
        ui = _make_ui(tools_running=True)
        ui.is_streaming_response = False
        call_count = [0]

        async def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyboardInterrupt()
            return "exit"

        ui.get_user_input = side_effect
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_idle(self):
        """KeyboardInterrupt when idle is caught and loop continues."""
        ui = _make_ui()
        ui.is_streaming_response = False
        ui.tools_running = False
        call_count = [0]

        async def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyboardInterrupt()
            return "exit"

        ui.get_user_input = side_effect
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)

    @pytest.mark.asyncio
    async def test_cancelled_error(self):
        """CancelledError during streaming is caught and loop continues."""
        ui = _make_ui(is_streaming=True)
        call_count = [0]

        async def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise asyncio.CancelledError()
            return "exit"

        ui.get_user_input = side_effect
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)

    @pytest.mark.asyncio
    async def test_eof_error(self):
        """EOFError causes loop to exit."""
        ui = _make_ui()
        ui.get_user_input = AsyncMock(side_effect=EOFError())
        ctx = _make_ctx()
        convo = _make_convo()

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)

    @pytest.mark.asyncio
    async def test_generic_exception_continues(self):
        """Generic exception in processing logs error and continues."""
        ui = _make_ui()
        call_count = [0]

        async def side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return "hello"
            return "exit"

        ui.get_user_input = side_effect
        ctx = _make_ctx()
        convo = _make_convo()
        convo.process_conversation = AsyncMock(side_effect=[ValueError("oops"), None])

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _run_enhanced_chat_loop

            await _run_enhanced_chat_loop(ui, ctx, convo)


# ===========================================================================
# Tests for _safe_cleanup
# ===========================================================================


class TestSafeCleanup:
    """Tests for _safe_cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_streaming(self):
        """Cleans up streaming state."""
        ui = _make_ui(is_streaming=True)
        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _safe_cleanup

            await _safe_cleanup(ui)
            ui.interrupt_streaming.assert_called_once()
            ui.stop_streaming_response_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_tools_running(self):
        """Cleans up tools state."""
        ui = _make_ui(tools_running=True)
        ui.is_streaming_response = False
        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _safe_cleanup

            await _safe_cleanup(ui)
            ui.stop_tool_calls.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_normal(self):
        """Normal cleanup calls cleanup."""
        ui = _make_ui()
        ui.is_streaming_response = False
        ui.tools_running = False
        from mcp_cli.chat.chat_handler import _safe_cleanup

        await _safe_cleanup(ui)
        ui.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_exception_handled(self):
        """Exception during cleanup is caught."""
        ui = _make_ui()
        ui.is_streaming_response = False
        ui.tools_running = False
        ui.cleanup = MagicMock(side_effect=RuntimeError("cleanup boom"))

        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import _safe_cleanup

            await _safe_cleanup(ui)  # Should not raise


# ===========================================================================
# Tests for handle_interrupt_command
# ===========================================================================


class TestHandleInterruptCommand:
    """Tests for handle_interrupt_command."""

    @pytest.mark.asyncio
    async def test_interrupt_streaming(self):
        """Interrupts streaming when active."""
        ui = _make_ui(is_streaming=True)
        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import handle_interrupt_command

            result = await handle_interrupt_command(ui)
            assert result is True
            ui.interrupt_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_interrupt_tools(self):
        """Interrupts tools when running."""
        ui = _make_ui(tools_running=True)
        ui.is_streaming_response = False
        with patch("mcp_cli.chat.chat_handler.output"):
            from mcp_cli.chat.chat_handler import handle_interrupt_command

            result = await handle_interrupt_command(ui)
            assert result is True
            ui._interrupt_now.assert_called_once()

    @pytest.mark.asyncio
    async def test_interrupt_nothing(self):
        """No-op when nothing to interrupt."""
        ui = _make_ui()
        ui.is_streaming_response = False
        ui.tools_running = False
        with patch("mcp_cli.chat.chat_handler.output") as mock_output:
            from mcp_cli.chat.chat_handler import handle_interrupt_command

            result = await handle_interrupt_command(ui)
            assert result is True
            mock_output.info.assert_called()


# ===========================================================================
# Tests for multi-agent wiring
# ===========================================================================


class TestMultiAgentWiring:
    """Tests for multi_agent=True flag in handle_chat_mode."""

    @pytest.mark.asyncio
    async def test_multi_agent_creates_agent_manager(self):
        """multi_agent=True creates AgentManager and sets it on context."""
        tool_mgr = MagicMock()
        tool_mgr.close = AsyncMock()
        tool_mgr.get_tool_count = MagicMock(return_value=5)

        mock_ctx = _make_ctx()
        mock_ctx.initialize = AsyncMock(return_value=True)
        mock_ctx.agent_id = "default"
        mock_ctx.dashboard_bridge = None
        mock_ctx.conversation_history = []
        mock_ctx.save_session = MagicMock(return_value=None)

        mock_app_ctx = MagicMock()
        mock_app_ctx.model_manager = MagicMock()
        mock_app_ctx.initialize = AsyncMock()

        ui = _make_ui()
        convo = _make_convo()

        mock_bridge = MagicMock()
        mock_bridge.set_context = MagicMock()
        mock_bridge.set_tool_call_callback = MagicMock()
        mock_bridge.set_input_queue = MagicMock()
        mock_bridge.on_shutdown = AsyncMock()
        mock_bridge.server = MagicMock()
        mock_bridge.server.stop = AsyncMock()

        mock_launch = AsyncMock(return_value=(MagicMock(), MagicMock(), 9120))
        mock_agent_manager = MagicMock()
        mock_agent_manager.stop_all = AsyncMock()

        with (
            patch("mcp_cli.chat.chat_handler.initialize_config"),
            patch(
                "mcp_cli.chat.chat_handler.initialize_context",
                return_value=mock_app_ctx,
            ),
            patch("mcp_cli.chat.chat_handler.output"),
            patch("mcp_cli.chat.chat_handler.clear_screen"),
            patch("mcp_cli.chat.chat_handler.display_chat_banner"),
            patch("mcp_cli.chat.chat_handler.ChatContext") as MockCC,
            patch("mcp_cli.chat.chat_handler.ChatUIManager", return_value=ui),
            patch(
                "mcp_cli.chat.chat_handler.ConversationProcessor",
                return_value=convo,
            ),
            patch(
                "mcp_cli.chat.chat_handler._run_enhanced_chat_loop",
                new_callable=AsyncMock,
            ),
            patch(
                "mcp_cli.chat.chat_handler._safe_cleanup",
                new_callable=AsyncMock,
            ),
            patch(
                "mcp_cli.dashboard.launcher.launch_dashboard",
                mock_launch,
            ),
            patch(
                "mcp_cli.dashboard.bridge.DashboardBridge",
                return_value=mock_bridge,
            ),
            patch(
                "mcp_cli.agents.manager.AgentManager",
                return_value=mock_agent_manager,
            ) as MockAM,
        ):
            MockCC.create.return_value = mock_ctx

            from mcp_cli.chat.chat_handler import handle_chat_mode

            result = await handle_chat_mode(
                tool_mgr,
                provider="openai",
                model="gpt-4",
                multi_agent=True,
            )
            assert result is True
            # AgentManager was constructed
            MockAM.assert_called_once()
            # agent_manager set on context
            assert mock_ctx.agent_manager == mock_agent_manager
            # stop_all called during cleanup
            mock_agent_manager.stop_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_agent_implies_dashboard(self):
        """multi_agent=True forces dashboard=True even if not passed."""
        tool_mgr = MagicMock()
        tool_mgr.close = AsyncMock()
        tool_mgr.get_tool_count = MagicMock(return_value=0)

        mock_ctx = _make_ctx()
        mock_ctx.initialize = AsyncMock(return_value=True)
        mock_ctx.agent_id = "default"
        mock_ctx.dashboard_bridge = None
        mock_ctx.conversation_history = []
        mock_ctx.save_session = MagicMock(return_value=None)

        mock_app_ctx = MagicMock()
        mock_app_ctx.model_manager = MagicMock()
        mock_app_ctx.initialize = AsyncMock()

        ui = _make_ui()
        convo = _make_convo()

        mock_bridge = MagicMock()
        mock_bridge.set_context = MagicMock()
        mock_bridge.set_tool_call_callback = MagicMock()
        mock_bridge.set_input_queue = MagicMock()
        mock_bridge.on_shutdown = AsyncMock()
        mock_bridge.server = MagicMock()
        mock_bridge.server.stop = AsyncMock()

        mock_launch = AsyncMock(return_value=(MagicMock(), MagicMock(), 9120))

        with (
            patch("mcp_cli.chat.chat_handler.initialize_config"),
            patch(
                "mcp_cli.chat.chat_handler.initialize_context",
                return_value=mock_app_ctx,
            ),
            patch("mcp_cli.chat.chat_handler.output"),
            patch("mcp_cli.chat.chat_handler.clear_screen"),
            patch("mcp_cli.chat.chat_handler.display_chat_banner"),
            patch("mcp_cli.chat.chat_handler.ChatContext") as MockCC,
            patch("mcp_cli.chat.chat_handler.ChatUIManager", return_value=ui),
            patch(
                "mcp_cli.chat.chat_handler.ConversationProcessor",
                return_value=convo,
            ),
            patch(
                "mcp_cli.chat.chat_handler._run_enhanced_chat_loop",
                new_callable=AsyncMock,
            ),
            patch(
                "mcp_cli.chat.chat_handler._safe_cleanup",
                new_callable=AsyncMock,
            ),
            patch(
                "mcp_cli.dashboard.launcher.launch_dashboard",
                mock_launch,
            ),
            patch(
                "mcp_cli.dashboard.bridge.DashboardBridge",
                return_value=mock_bridge,
            ),
            patch(
                "mcp_cli.agents.manager.AgentManager",
                return_value=MagicMock(stop_all=AsyncMock()),
            ),
        ):
            MockCC.create.return_value = mock_ctx

            from mcp_cli.chat.chat_handler import handle_chat_mode

            # dashboard=False but multi_agent=True — dashboard should still launch
            result = await handle_chat_mode(
                tool_mgr,
                provider="openai",
                model="gpt-4",
                dashboard=False,
                multi_agent=True,
            )
            assert result is True
            # launch_dashboard was called (proving dashboard was forced on)
            mock_launch.assert_called_once()
