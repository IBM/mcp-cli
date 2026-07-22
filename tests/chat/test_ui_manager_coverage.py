# tests/chat/test_ui_manager_coverage.py
"""Tests for mcp_cli.chat.ui_manager.ChatUIManager achieving >90% coverage.

This file is separate from test_ui_manager.py (which tests the command completer).
"""

import asyncio
import signal
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers - mock all heavy dependencies at import time
# ---------------------------------------------------------------------------


def _make_context():
    """Build a minimal mock context for ChatUIManager."""
    ctx = MagicMock()
    ctx.provider = "openai"
    ctx.model = "gpt-4"
    ctx.exit_requested = False
    ctx.tool_manager = MagicMock()
    ctx.model_manager = MagicMock()
    ctx.to_dict = MagicMock(
        return_value={
            "conversation_history": [],
            "tools": [],
            "internal_tools": [],
            "client": MagicMock(),
            "provider": "openai",
            "model": "gpt-4",
            "model_manager": MagicMock(),
            "server_info": [],
            "openai_tools": [],
            "tool_name_mapping": {},
            "exit_requested": False,
            "tool_to_server_map": {},
            "tool_manager": MagicMock(),
            "session_id": "test-session",
        }
    )
    ctx.get_status_summary = MagicMock(
        return_value=MagicMock(
            provider="openai",
            model="gpt-4",
            message_count=5,
            tool_count=10,
            server_count=2,
            tool_execution_count=3,
        )
    )
    return ctx


@pytest.fixture
def ui_manager():
    """Create a ChatUIManager with mocked dependencies."""
    ctx = _make_context()

    with (
        patch("mcp_cli.chat.ui_manager.get_preference_manager") as mock_pref,
        patch("mcp_cli.chat.ui_manager.get_theme") as mock_theme,
        patch(
            "mcp_cli.chat.ui_manager.create_transparent_completion_style",
            return_value={},
        ),
        patch("mcp_cli.chat.ui_manager.Style"),
        patch("mcp_cli.chat.ui_manager.register_all_commands"),
        patch("mcp_cli.chat.ui_manager.ChatCommandCompleter"),
        patch("mcp_cli.chat.ui_manager.PromptSession"),
        patch("mcp_cli.chat.ui_manager.FileHistory"),
        patch("mcp_cli.chat.ui_manager.AutoSuggestFromHistory"),
        patch("mcp_cli.chat.ui_manager.StreamingDisplayManager") as MockDisplay,
    ):
        mock_pref.return_value.get_history_file.return_value = "/tmp/test_history"
        theme = MagicMock()
        theme.name = "dark"
        theme.colors = {}
        mock_theme.return_value = theme
        MockDisplay.return_value = MagicMock()
        MockDisplay.return_value.is_streaming = False
        MockDisplay.return_value.show_user_message = MagicMock()
        MockDisplay.return_value.start_streaming = AsyncMock()
        MockDisplay.return_value.stop_streaming = AsyncMock(return_value="")
        MockDisplay.return_value.start_tool_execution = AsyncMock()
        MockDisplay.return_value.stop_tool_execution = AsyncMock()

        from mcp_cli.chat.ui_manager import ChatUIManager

        ui = ChatUIManager(ctx)

    return ui


# ===========================================================================
# Initialization tests
# ===========================================================================


class TestChatUIManagerInit:
    """Tests for ChatUIManager initialization."""

    def test_basic_init(self, ui_manager):
        assert ui_manager.context is not None
        assert ui_manager.verbose_mode is False
        assert ui_manager.tool_calls == []
        assert ui_manager.tool_times == []
        assert ui_manager.tool_start_time is None
        assert ui_manager.current_tool_start_time is None
        assert ui_manager.streaming_handler is None
        assert ui_manager.tools_running is False
        assert ui_manager._interrupt_count == 0
        assert ui_manager._last_interrupt_time == 0.0

    def test_theme_light(self):
        """Light theme uses white bg."""
        ctx = _make_context()
        with (
            patch("mcp_cli.chat.ui_manager.get_preference_manager") as mock_pref,
            patch("mcp_cli.chat.ui_manager.get_theme") as mock_theme,
            patch(
                "mcp_cli.chat.ui_manager.create_transparent_completion_style",
                return_value={},
            ),
            patch("mcp_cli.chat.ui_manager.Style"),
            patch("mcp_cli.chat.ui_manager.register_all_commands"),
            patch("mcp_cli.chat.ui_manager.ChatCommandCompleter"),
            patch("mcp_cli.chat.ui_manager.PromptSession"),
            patch("mcp_cli.chat.ui_manager.FileHistory"),
            patch("mcp_cli.chat.ui_manager.AutoSuggestFromHistory"),
            patch("mcp_cli.chat.ui_manager.StreamingDisplayManager"),
        ):
            mock_pref.return_value.get_history_file.return_value = "/tmp/h"
            theme = MagicMock()
            theme.name = "light"
            theme.colors = {}
            mock_theme.return_value = theme

            from mcp_cli.chat.ui_manager import ChatUIManager

            ui = ChatUIManager(ctx)
            assert ui is not None

    def test_theme_minimal(self):
        """Minimal theme uses empty bg."""
        ctx = _make_context()
        with (
            patch("mcp_cli.chat.ui_manager.get_preference_manager") as mock_pref,
            patch("mcp_cli.chat.ui_manager.get_theme") as mock_theme,
            patch(
                "mcp_cli.chat.ui_manager.create_transparent_completion_style",
                return_value={},
            ),
            patch("mcp_cli.chat.ui_manager.Style"),
            patch("mcp_cli.chat.ui_manager.register_all_commands"),
            patch("mcp_cli.chat.ui_manager.ChatCommandCompleter"),
            patch("mcp_cli.chat.ui_manager.PromptSession"),
            patch("mcp_cli.chat.ui_manager.FileHistory"),
            patch("mcp_cli.chat.ui_manager.AutoSuggestFromHistory"),
            patch("mcp_cli.chat.ui_manager.StreamingDisplayManager"),
        ):
            mock_pref.return_value.get_history_file.return_value = "/tmp/h"
            theme = MagicMock()
            theme.name = "terminal"
            theme.colors = {}
            mock_theme.return_value = theme

            from mcp_cli.chat.ui_manager import ChatUIManager

            ui = ChatUIManager(ctx)
            assert ui is not None


# ===========================================================================
# User input tests
# ===========================================================================


class TestGetUserInput:
    """Tests for get_user_input."""

    @pytest.mark.asyncio
    async def test_normal_input(self, ui_manager):
        ui_manager.session.prompt = MagicMock(return_value="hello world")
        result = await ui_manager.get_user_input()
        assert result == "hello world"
        assert ui_manager.last_input == "hello world"

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_returns_exit(self, ui_manager):
        ui_manager.session.prompt = MagicMock(side_effect=KeyboardInterrupt())
        result = await ui_manager.get_user_input()
        assert result == "/exit"

    @pytest.mark.asyncio
    async def test_eof_returns_exit(self, ui_manager):
        ui_manager.session.prompt = MagicMock(side_effect=EOFError())
        result = await ui_manager.get_user_input()
        assert result == "/exit"

    @pytest.mark.asyncio
    async def test_custom_prompt(self, ui_manager):
        ui_manager.session.prompt = MagicMock(return_value="  test  ")
        result = await ui_manager.get_user_input(prompt="Custom")
        assert result == "test"


# ===========================================================================
# Message display tests
# ===========================================================================


class TestPrintUserMessage:
    """Tests for print_user_message."""

    def test_normal_message(self, ui_manager):
        ui_manager.tool_calls = [{"id": "1"}]
        ui_manager.print_user_message("Hello")
        ui_manager.display.show_user_message.assert_called_once_with("Hello")
        assert ui_manager.tool_calls == []

    def test_empty_message(self, ui_manager):
        ui_manager.print_user_message("")
        ui_manager.display.show_user_message.assert_called_once_with("[No Message]")

    def test_none_message(self, ui_manager):
        ui_manager.print_user_message(None)
        ui_manager.display.show_user_message.assert_called_once_with("[No Message]")


class TestPrintAssistantMessage:
    """Tests for print_assistant_message."""

    @pytest.mark.asyncio
    async def test_streaming_active(self, ui_manager):
        ui_manager.display.is_streaming = True
        await ui_manager.print_assistant_message("content", 1.5)
        ui_manager.display.stop_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_streaming(self, ui_manager):
        ui_manager.display.is_streaming = False
        with patch("mcp_cli.chat.ui_manager.output") as mock_output:
            await ui_manager.print_assistant_message("Hello!", 2.0)
            assert mock_output.print.call_count >= 1

    @pytest.mark.asyncio
    async def test_empty_content(self, ui_manager):
        ui_manager.display.is_streaming = False
        with patch("mcp_cli.chat.ui_manager.output") as mock_output:
            await ui_manager.print_assistant_message("", 0.5)
            # Should print "[No Response]"
            calls = [str(c) for c in mock_output.print.call_args_list]
            assert any("No Response" in c for c in calls)


# ===========================================================================
# Tool display tests
# ===========================================================================


class TestToolExecution:
    """Tests for tool execution display."""

    @pytest.mark.asyncio
    async def test_start_tool_execution(self, ui_manager):
        args = {"path": "/tmp", "data": {"key": "value"}, "items": [1, 2]}
        await ui_manager.start_tool_execution("read_file", args)
        ui_manager.display.start_tool_execution.assert_called_once()
        call_args = ui_manager.display.start_tool_execution.call_args
        assert call_args[0][0] == "read_file"
        # dict/list args should be JSON-stringified
        processed = call_args[0][1]
        assert isinstance(processed["data"], str)
        assert isinstance(processed["items"], str)
        assert processed["path"] == "/tmp"

    @pytest.mark.asyncio
    async def test_finish_tool_execution(self, ui_manager):
        await ui_manager.finish_tool_execution(result="OK", success=True)
        ui_manager.display.stop_tool_execution.assert_called_once_with("OK", True)

    @pytest.mark.asyncio
    async def test_finish_tool_execution_no_result(self, ui_manager):
        await ui_manager.finish_tool_execution()
        ui_manager.display.stop_tool_execution.assert_called_once_with("", True)

    def test_print_tool_call_no_op(self, ui_manager):
        """print_tool_call is a no-op (streaming display handles it)."""
        ui_manager.print_tool_call("fn", {"x": 1})
        # No crash is the test


# ===========================================================================
# Confirm tool execution tests
# ===========================================================================


class TestDoConfirmToolExecution:
    """Tests for do_confirm_tool_execution."""

    @pytest.mark.asyncio
    async def test_confirm_yes(self, ui_manager):
        with (
            patch("mcp_cli.chat.ui_manager.output"),
            patch("builtins.input", return_value="y"),
        ):
            result = await ui_manager.do_confirm_tool_execution("fn", '{"x": 1}')
            assert result is True

    @pytest.mark.asyncio
    async def test_confirm_empty_default_yes(self, ui_manager):
        with (
            patch("mcp_cli.chat.ui_manager.output"),
            patch("builtins.input", return_value=""),
        ):
            result = await ui_manager.do_confirm_tool_execution("fn", {"x": 1})
            assert result is True

    @pytest.mark.asyncio
    async def test_confirm_no(self, ui_manager):
        with (
            patch("mcp_cli.chat.ui_manager.output"),
            patch("builtins.input", return_value="n"),
        ):
            result = await ui_manager.do_confirm_tool_execution("fn", {"x": 1})
            assert result is False

    @pytest.mark.asyncio
    async def test_confirm_invalid_json_string(self, ui_manager):
        with (
            patch("mcp_cli.chat.ui_manager.output"),
            patch("builtins.input", return_value="yes"),
        ):
            result = await ui_manager.do_confirm_tool_execution("fn", "{not json")
            assert result is True

    @pytest.mark.asyncio
    async def test_confirm_none_args(self, ui_manager):
        with (
            patch("mcp_cli.chat.ui_manager.output"),
            patch("builtins.input", return_value="y"),
        ):
            result = await ui_manager.do_confirm_tool_execution("fn", None)
            assert result is True

    @pytest.mark.asyncio
    async def test_confirm_empty_string(self, ui_manager):
        with (
            patch("mcp_cli.chat.ui_manager.output"),
            patch("builtins.input", return_value="y"),
        ):
            result = await ui_manager.do_confirm_tool_execution("fn", "")
            assert result is True


# ===========================================================================
# Dashboard confirmation path tests
# ===========================================================================


class TestDoConfirmDashboardPath:
    """Tests for do_confirm_tool_execution routing to dashboard bridge."""

    @pytest.mark.asyncio
    async def test_routes_to_dashboard_when_clients_connected(self, ui_manager):
        """When dashboard bridge has clients, use bridge for approval."""
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(True)

        bridge = MagicMock()
        bridge.has_clients = True
        bridge.request_tool_approval = AsyncMock(return_value=fut)
        ui_manager.context.dashboard_bridge = bridge

        result = await ui_manager.do_confirm_tool_execution("test_tool", {"x": 1})
        assert result is True
        bridge.request_tool_approval.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_terminal_when_no_clients(self, ui_manager):
        """When dashboard has no clients, fall back to terminal input."""
        bridge = MagicMock()
        bridge.has_clients = False
        ui_manager.context.dashboard_bridge = bridge

        with (
            patch("mcp_cli.chat.ui_manager.output"),
            patch("builtins.input", return_value="y"),
        ):
            result = await ui_manager.do_confirm_tool_execution("test_tool", {"x": 1})
            assert result is True
        bridge.request_tool_approval.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_terminal_when_no_bridge(self, ui_manager):
        """When no dashboard bridge, fall back to terminal input."""
        ui_manager.context.dashboard_bridge = None

        with (
            patch("mcp_cli.chat.ui_manager.output"),
            patch("builtins.input", return_value="y"),
        ):
            result = await ui_manager.do_confirm_tool_execution("test_tool", {"x": 1})
            assert result is True

    @pytest.mark.asyncio
    async def test_dashboard_denial(self, ui_manager):
        """Dashboard user denies the tool execution."""
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(False)

        bridge = MagicMock()
        bridge.server = MagicMock()
        bridge.server.has_clients = True
        bridge.request_tool_approval = AsyncMock(return_value=fut)
        ui_manager.context.dashboard_bridge = bridge

        result = await ui_manager.do_confirm_tool_execution("test_tool", {"x": 1})
        assert result is False

    @pytest.mark.asyncio
    async def test_dashboard_timeout_returns_false(self, ui_manager):
        """If dashboard approval times out, return False."""
        # Create a future that never resolves
        fut = asyncio.get_event_loop().create_future()

        bridge = MagicMock()
        bridge.server = MagicMock()
        bridge.server.has_clients = True
        bridge.request_tool_approval = AsyncMock(return_value=fut)
        ui_manager.context.dashboard_bridge = bridge

        # Patch timeout to be very short for testing
        with patch(
            "mcp_cli.chat.ui_manager.asyncio.wait_for", side_effect=asyncio.TimeoutError
        ):
            result = await ui_manager.do_confirm_tool_execution("test_tool", {"x": 1})
            assert result is False
        # Clean up
        fut.cancel()

    @pytest.mark.asyncio
    async def test_dashboard_exception_falls_back_to_terminal(self, ui_manager):
        """If dashboard approval throws, fall back to terminal."""
        bridge = MagicMock()
        bridge.server = MagicMock()
        bridge.server.has_clients = True
        bridge.request_tool_approval = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )
        ui_manager.context.dashboard_bridge = bridge

        with (
            patch("mcp_cli.chat.ui_manager.output"),
            patch("builtins.input", return_value="y"),
        ):
            result = await ui_manager.do_confirm_tool_execution("test_tool", {"x": 1})
            assert result is True

    @pytest.mark.asyncio
    async def test_string_args_parsed_for_dashboard(self, ui_manager):
        """String arguments should be JSON-parsed before sending to dashboard."""
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(True)

        bridge = MagicMock()
        bridge.server = MagicMock()
        bridge.server.has_clients = True
        bridge.request_tool_approval = AsyncMock(return_value=fut)
        ui_manager.context.dashboard_bridge = bridge

        await ui_manager.do_confirm_tool_execution("test_tool", '{"key": "value"}')
        call_args = bridge.request_tool_approval.call_args
        # Arguments should have been parsed from JSON string to dict
        assert (
            call_args.kwargs.get("arguments") == {"key": "value"}
            or call_args[1].get("arguments") == {"key": "value"}
            or (len(call_args[0]) >= 2 and call_args[0][1] == {"key": "value"})
        )


# ===========================================================================
# Streaming support tests
# ===========================================================================


class TestStreamingSupport:
    """Tests for streaming-related methods."""

    def test_is_streaming_response_property(self, ui_manager):
        ui_manager.display.is_streaming = False
        assert ui_manager.is_streaming_response is False
        ui_manager.display.is_streaming = True
        assert ui_manager.is_streaming_response is True

    @pytest.mark.asyncio
    async def test_start_streaming_response(self, ui_manager):
        await ui_manager.start_streaming_response()
        # No-op, should not crash

    @pytest.mark.asyncio
    async def test_stop_streaming_response_when_streaming(self, ui_manager):
        ui_manager.display.is_streaming = True
        await ui_manager.stop_streaming_response()
        ui_manager.display.stop_streaming.assert_called_with(interrupted=True)

    @pytest.mark.asyncio
    async def test_stop_streaming_response_when_not_streaming(self, ui_manager):
        ui_manager.display.is_streaming = False
        await ui_manager.stop_streaming_response()
        ui_manager.display.stop_streaming.assert_not_called()

    def test_stop_streaming_response_sync(self, ui_manager):
        ui_manager.stop_streaming_response_sync()
        # No-op, should not crash

    def test_interrupt_streaming_with_handler(self, ui_manager):
        handler = MagicMock()
        handler.interrupt_streaming = MagicMock()
        ui_manager.streaming_handler = handler
        ui_manager.interrupt_streaming()
        handler.interrupt_streaming.assert_called_once()

    def test_interrupt_streaming_no_handler(self, ui_manager):
        ui_manager.streaming_handler = None
        ui_manager.interrupt_streaming()  # Should not crash


# ===========================================================================
# Signal handling tests
# ===========================================================================


class TestSignalHandling:
    """Tests for signal handling."""

    def test_setup_signal_handlers(self, ui_manager):
        with patch("mcp_cli.chat.ui_manager.signal.signal") as mock_signal:
            ui_manager.setup_signal_handlers()
            mock_signal.assert_called_once_with(
                signal.SIGINT, ui_manager._handle_sigint
            )

    def test_restore_signal_handlers(self, ui_manager):
        prev_handler = MagicMock()
        ui_manager._prev_sigint_handler = prev_handler
        with patch("mcp_cli.chat.ui_manager.signal.signal") as mock_signal:
            ui_manager.restore_signal_handlers()
            mock_signal.assert_called_once_with(signal.SIGINT, prev_handler)

    def test_restore_signal_handlers_none(self, ui_manager):
        ui_manager._prev_sigint_handler = None
        with patch("mcp_cli.chat.ui_manager.signal.signal") as mock_signal:
            ui_manager.restore_signal_handlers()
            mock_signal.assert_not_called()

    def test_handle_sigint_first_tap_streaming(self, ui_manager):
        ui_manager.display.is_streaming = True
        ui_manager._interrupt_count = 0
        ui_manager._last_interrupt_time = 0.0

        with patch("mcp_cli.chat.ui_manager.output"):
            ui_manager._handle_sigint(signal.SIGINT, None)

        assert ui_manager._interrupt_count == 1
        ui_manager.streaming_handler = MagicMock()  # Ensure handler is set
        # The method should call interrupt_streaming

    def test_handle_sigint_first_tap_not_streaming(self, ui_manager):
        ui_manager.display.is_streaming = False
        ui_manager._interrupt_count = 0
        ui_manager._last_interrupt_time = 0.0

        with patch("mcp_cli.chat.ui_manager.output"):
            ui_manager._handle_sigint(signal.SIGINT, None)

        assert ui_manager._interrupt_count == 1

    def test_handle_sigint_double_tap(self, ui_manager):
        """Double-tap raises KeyboardInterrupt."""
        ui_manager._interrupt_count = 1
        ui_manager._last_interrupt_time = time.time()

        with (
            patch("mcp_cli.chat.ui_manager.output"),
            pytest.raises(KeyboardInterrupt),
        ):
            ui_manager._handle_sigint(signal.SIGINT, None)

    def test_handle_sigint_resets_after_timeout(self, ui_manager):
        """Interrupt count resets after 2 seconds."""
        ui_manager._interrupt_count = 1
        ui_manager._last_interrupt_time = time.time() - 3.0  # 3 seconds ago

        with patch("mcp_cli.chat.ui_manager.output"):
            ui_manager._handle_sigint(signal.SIGINT, None)

        # Count should have been reset to 0, then incremented to 1
        assert ui_manager._interrupt_count == 1


# ===========================================================================
# Confirm tool execution (async version) tests
# ===========================================================================


class TestConfirmToolExecutionAsync:
    """Tests for async confirm_tool_execution."""

    @pytest.mark.asyncio
    async def test_confirmed(self, ui_manager):
        with patch("mcp_cli.chat.ui_manager.output"):
            with patch("mcp_cli.chat.ui_manager.prompts") as mock_prompts:
                mock_prompts.confirm = MagicMock(return_value=True)
                result = await ui_manager.confirm_tool_execution("fn", {"x": 1})
                assert result is True

    @pytest.mark.asyncio
    async def test_denied(self, ui_manager):
        with patch("mcp_cli.chat.ui_manager.output"):
            with patch("mcp_cli.chat.ui_manager.prompts") as mock_prompts:
                mock_prompts.confirm = MagicMock(return_value=False)
                result = await ui_manager.confirm_tool_execution("fn", {"x": 1})
                assert result is False


# ===========================================================================
# Status and help tests
# ===========================================================================


class TestStatusAndHelp:
    """Tests for show_status and show_help."""

    def test_show_status(self, ui_manager):
        with patch("mcp_cli.chat.ui_manager.output") as mock_output:
            ui_manager.show_status()
            ui_manager.context.get_status_summary.assert_called_once()
            assert mock_output.info.called
            assert mock_output.print.called

    def test_show_help(self, ui_manager):
        with patch("mcp_cli.chat.ui_manager.output") as mock_output:
            ui_manager.show_help()
            assert mock_output.info.called
            assert mock_output.print.called


# ===========================================================================
# Cleanup tests
# ===========================================================================


class TestCleanup:
    """Tests for cleanup method."""

    def test_cleanup(self, ui_manager):
        with patch.object(ui_manager, "restore_signal_handlers") as mock_restore:
            ui_manager.cleanup()
            mock_restore.assert_called_once()


# ===========================================================================
# Compatibility methods tests
# ===========================================================================


class TestCompatibilityMethods:
    """Tests for compatibility methods."""

    def test_interrupt_now(self, ui_manager):
        with patch.object(ui_manager, "interrupt_streaming") as mock_interrupt:
            ui_manager._interrupt_now()
            mock_interrupt.assert_called_once()

    def test_stop_tool_calls(self, ui_manager):
        ui_manager.tools_running = True
        ui_manager.stop_tool_calls()
        assert ui_manager.tools_running is False


# ===========================================================================
# Handle command tests
# ===========================================================================


class TestHandleCommand:
    """Tests for handle_command."""

    @pytest.mark.asyncio
    async def test_command_handled(self, ui_manager):
        with (
            patch("mcp_cli.chat.ui_manager.register_all_commands"),
            patch("mcp_cli.adapters.chat.ChatCommandAdapter") as MockAdapter,
        ):
            MockAdapter.handle_command = AsyncMock(return_value=True)
            result = await ui_manager.handle_command("/help")
            assert result is True

    @pytest.mark.asyncio
    async def test_command_not_handled(self, ui_manager):
        with (
            patch("mcp_cli.chat.ui_manager.register_all_commands"),
            patch("mcp_cli.adapters.chat.ChatCommandAdapter") as MockAdapter,
        ):
            MockAdapter.handle_command = AsyncMock(return_value=False)
            result = await ui_manager.handle_command("/unknown")
            assert result is False

    @pytest.mark.asyncio
    async def test_command_triggers_exit(self, ui_manager):
        ui_manager.context.exit_requested = True
        with (
            patch("mcp_cli.chat.ui_manager.register_all_commands"),
            patch("mcp_cli.adapters.chat.ChatCommandAdapter") as MockAdapter,
        ):
            MockAdapter.handle_command = AsyncMock(return_value=True)
            result = await ui_manager.handle_command("/exit")
            assert result is True

    @pytest.mark.asyncio
    async def test_command_exception(self, ui_manager):
        with (
            patch("mcp_cli.chat.ui_manager.register_all_commands"),
            patch("mcp_cli.adapters.chat.ChatCommandAdapter") as MockAdapter,
            patch("mcp_cli.chat.ui_manager.output"),
        ):
            MockAdapter.handle_command = AsyncMock(
                side_effect=RuntimeError("cmd error")
            )
            result = await ui_manager.handle_command("/broken")
            assert result is False
