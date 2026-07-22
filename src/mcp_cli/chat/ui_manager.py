"""Clean, async-native Chat UI Manager using unified display system.

This module provides UI management for chat mode with:
- StreamingDisplayManager for all display operations
- Async-native throughout
- No fallback display paths
- Clean integration with chuk-term
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from types import FrameType
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

from chuk_term.ui import output
from chuk_term.ui import prompts
from chuk_term.ui.theme import get_theme

from mcp_cli.display import StreamingDisplayManager, create_transparent_completion_style
from mcp_cli.chat.command_completer import ChatCommandCompleter
from mcp_cli.commands import register_all_commands
from mcp_cli.utils.preferences import get_preference_manager

logger = logging.getLogger(__name__)


class ChatUIManager:
    """Manages chat UI with unified async display system."""

    def __init__(self, context) -> None:
        """Initialize UI manager.

        Args:
            context: Chat context containing client, history, etc.
        """
        self.context = context
        self.verbose_mode = False

        # Tool tracking
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_times: list[float] = []
        self.tool_start_time: float | None = None
        self.current_tool_start_time: float | None = None

        # Streaming state
        self.streaming_handler: Any | None = None
        self.tools_running = False  # Compatibility

        # Unified display manager (async-native, chuk-term only)
        self.display = StreamingDisplayManager()

        # Signal handling
        self._prev_sigint_handler: (
            Callable[[int, FrameType | None, Any], int] | signal.Handlers | None
        ) = None
        self._interrupt_count = 0
        self._last_interrupt_time = 0.0

        # Initialize prompt session
        self._init_prompt_session()
        self.last_input: str | None = None

    def _init_prompt_session(self) -> None:
        """Initialize prompt_toolkit session with history."""
        pref_manager = get_preference_manager()
        history_path = pref_manager.get_history_file()

        theme = get_theme()

        # Determine background color based on theme
        if theme.name in ["light"]:
            bg_color = "white"
        elif theme.name in ["minimal", "terminal"]:
            bg_color = ""
        else:
            bg_color = "black"

        # Create completion style
        completion_style = create_transparent_completion_style(theme.colors, bg_color)

        # Create style from completion dict
        merged_style = Style.from_dict(completion_style)

        # Initialize command registry
        register_all_commands()

        # Create completer (uses context dict)
        completer = ChatCommandCompleter(self.context.to_dict())

        # Create key bindings for Tab completion behavior
        bindings = KeyBindings()

        @bindings.add(Keys.Tab)
        def handle_tab(event):
            """Handle Tab key: accept suggestion, complete command, or cycle completions."""
            buff = event.app.current_buffer

            # Priority 1: If completion menu is showing, cycle through completions
            if buff.complete_state:
                buff.complete_next()
                # If only one completion, apply it immediately
                if buff.complete_state and len(buff.complete_state.completions) == 1:
                    buff.complete_state = None
            # Priority 2: If there's an auto-suggestion (gray text from history), accept it
            elif buff.suggestion:
                buff.insert_text(buff.suggestion.text)
            # Priority 3: Start slash command completion if typing a command
            elif buff.text.startswith("/"):
                buff.start_completion(select_first=True)
            # Priority 4: For non-command text, try to find history match
            else:
                # Try to trigger auto-suggest and accept if found
                suggestion = buff.auto_suggest.get_suggestion(buff, buff.document)
                if suggestion:
                    buff.insert_text(suggestion.text)

        # Create session with all features
        # Note: enable_history_search conflicts with complete_while_typing
        # (up arrows browse completions vs history), so we disable history search
        # to allow slash command completion to work as you type
        self.session: PromptSession = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            enable_history_search=False,  # Disabled: conflicts with complete_while_typing
            completer=completer,
            complete_while_typing=True,  # Auto-trigger completions as you type
            complete_in_thread=False,  # Complete in main thread for responsiveness
            style=merged_style,
            complete_style=CompleteStyle.MULTI_COLUMN,  # Show completions in multi-column menu
            key_bindings=bindings,  # Custom Tab behavior
        )

        logger.debug("Prompt session initialized with history and commands")

    # ==================== USER INPUT ====================

    async def get_user_input(self, prompt: str = "You") -> str:
        """Get user input with async prompt.

        Args:
            prompt: Prompt text to display

        Returns:
            User input string
        """
        try:
            # Run prompt in executor since it's blocking
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(
                None, self.session.prompt, f"\n💬 {prompt}: "
            )

            self.last_input = user_input
            return str(user_input).strip()

        except (KeyboardInterrupt, EOFError):
            return "/exit"

    # ==================== MESSAGE DISPLAY ====================

    def print_user_message(self, message: str) -> None:
        """Display user message.

        Args:
            message: User message to display
        """
        self.display.show_user_message(message or "[No Message]")
        self.tool_calls.clear()

    async def print_assistant_message(self, content: str, elapsed: float = 0) -> None:
        """Display assistant message.

        Args:
            content: Assistant message content
            elapsed: Elapsed time for response
        """
        # If we were streaming, it's already displayed
        if self.display.is_streaming:
            await self.display.stop_streaming()
        else:
            # Not streaming, show message directly
            output.print(f"\n🤖 Assistant ({elapsed:.1f}s):")
            output.print(content or "[No Response]")

    # ==================== TOOL DISPLAY ====================

    async def start_tool_execution(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """Start tool execution display.

        Args:
            tool_name: Name of tool being executed
            arguments: Tool arguments
        """
        # Format arguments for display
        processed_args = {}
        for k, v in arguments.items():
            if isinstance(v, (dict, list)):
                processed_args[k] = json.dumps(v)
            else:
                processed_args[k] = str(v)

        await self.display.start_tool_execution(tool_name, processed_args)

    def print_tool_call(self, tool_name: str, raw_arguments: Any) -> None:
        """Print tool call notification before execution.

        Note: This is called but the output is immediately cleared by streaming display.
        The actual tool parameters are shown in the tool execution status line instead.

        Args:
            tool_name: Name of the tool being called
            raw_arguments: Raw arguments (JSON string or dict)
        """
        # Don't print here - streaming display will show it in the status line
        # The display manager shows tool name + arguments during execution
        pass

    async def do_confirm_tool_execution(self, tool_name: str, arguments: Any) -> bool:
        """Prompt user to confirm tool execution.

        Routes to dashboard if browser clients are connected,
        otherwise falls back to terminal prompt.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            True if user confirms, False otherwise
        """
        # Parse arguments for display
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                args = {"raw": arguments}
        else:
            args = arguments or {}

        # Route to dashboard if clients are connected
        bridge = getattr(self.context, "dashboard_bridge", None)
        if bridge is not None and bridge.has_clients:
            try:
                call_id = f"confirm-{id(arguments)}-{time.time_ns()}"
                fut = await bridge.request_tool_approval(
                    tool_name=tool_name,
                    arguments=args,
                    call_id=call_id,
                )
                # Wait for dashboard user to approve/deny (with timeout)
                return await asyncio.wait_for(fut, timeout=300)
            except asyncio.TimeoutError:
                logger.warning("Tool approval timed out for %s", tool_name)
                return False
            except Exception as exc:
                logger.warning("Dashboard tool approval error: %s", exc)
                # Fall through to terminal prompt

        # Terminal fallback
        output.warning(f"Tool confirmation required: {tool_name}")
        args_str = json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
        output.print(f"Parameters:\n{args_str}")

        # Use asyncio-friendly prompt
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: input("\nExecute this tool? [Y/n]: ").strip().lower()
        )
        return response in ("", "y", "yes")

    async def finish_tool_execution(
        self, result: str | None = None, success: bool = True
    ) -> None:
        """Finish tool execution display.

        Args:
            result: Tool execution result
            success: Whether execution succeeded
        """
        await self.display.stop_tool_execution(result or "", success)

    # ==================== STREAMING SUPPORT ====================

    @property
    def is_streaming_response(self) -> bool:
        """Whether currently streaming a response."""
        return self.display.is_streaming

    async def start_streaming_response(self) -> None:
        """Start streaming response (handled by display manager)."""
        # Display manager handles this via streaming_handler
        pass

    async def stop_streaming_response(self) -> None:
        """Stop streaming response."""
        if self.display.is_streaming:
            await self.display.stop_streaming(interrupted=True)

    def stop_streaming_response_sync(self) -> None:
        """Stop streaming response (sync version for cleanup)."""
        # Best-effort cleanup, don't await
        pass

    def interrupt_streaming(self) -> None:
        """Interrupt current streaming operation."""
        if self.streaming_handler and hasattr(
            self.streaming_handler, "interrupt_streaming"
        ):
            self.streaming_handler.interrupt_streaming()

    # ==================== SIGNAL HANDLING ====================

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful interruption."""
        self._prev_sigint_handler = signal.signal(  # type: ignore[assignment]
            signal.SIGINT, self._handle_sigint
        )

    def restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._prev_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._prev_sigint_handler)  # type: ignore[arg-type]

    def _handle_sigint(self, signum: int, frame: FrameType | None) -> None:
        """Handle SIGINT (Ctrl+C) gracefully."""
        current_time = time.time()

        # Track interrupt count for double-tap exit
        if current_time - self._last_interrupt_time > 2.0:
            self._interrupt_count = 0

        self._interrupt_count += 1
        self._last_interrupt_time = current_time

        if self._interrupt_count >= 2:
            # Double tap - force exit
            output.warning("\n\nForce exit requested")
            raise KeyboardInterrupt

        # Single tap - try graceful interrupt
        if self.display.is_streaming:
            self.interrupt_streaming()
            output.warning(
                "\n\n⚠️  Interrupting streaming... (Ctrl+C again to force exit)"
            )
        else:
            output.warning("\n\n⚠️  Interrupted (Ctrl+C again to exit)")

    # ==================== TOOL CONFIRMATIONS ====================

    async def confirm_tool_execution(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> bool:
        """Prompt user to confirm tool execution.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments

        Returns:
            True if confirmed, False otherwise
        """
        # Format arguments for display
        args_display = json.dumps(arguments, indent=2)

        # Show tool info
        output.info(f"\n🔧 Tool: {tool_name}")
        output.print(f"Arguments:\n{args_display}\n")

        # Get confirmation
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, prompts.confirm, "Execute this tool?")

        return bool(result)

    # ==================== STATUS & INFO ====================

    def show_status(self) -> None:
        """Show current chat status."""
        status = self.context.get_status_summary()

        output.info("📊 Chat Status:")
        output.print(f"  Provider: {status.provider}")
        output.print(f"  Model: {status.model}")
        output.print(f"  Messages: {status.message_count}")
        output.print(f"  Tools: {status.tool_count}")
        output.print(f"  Servers: {status.server_count}")
        output.print(f"  Tool Executions: {status.tool_execution_count}")

    def show_help(self) -> None:
        """Show help message."""
        output.info("💬 Chat Commands:")
        output.print("  /help       - Show this help")
        output.print("  /status     - Show status")
        output.print("  /clear      - Clear conversation")
        output.print("  /history    - Show conversation history")
        output.print("  /exit       - Exit chat")
        output.print("\n💡 Tip: Ctrl+C to interrupt streaming")

    def cleanup(self) -> None:
        """Cleanup UI manager resources."""
        self.restore_signal_handlers()
        logger.debug("UI manager cleaned up")

    # ==================== COMPATIBILITY METHODS ====================

    def _interrupt_now(self) -> None:
        """Immediate interrupt (compatibility method)."""
        self.interrupt_streaming()

    def stop_tool_calls(self) -> None:
        """Stop tool calls (compatibility method)."""
        self.tools_running = False

    async def handle_command(self, user_input: str) -> bool:
        """Handle slash command.

        Args:
            user_input: User input string

        Returns:
            True if handled as command, False otherwise
        """
        try:
            # Ensure commands are registered
            register_all_commands()

            # Build context for unified commands
            context = {
                "tool_manager": self.context.tool_manager,
                "model_manager": self.context.model_manager,
                "chat_handler": self,
                "chat_context": self.context,
                "ui_manager": self,
            }

            # Use the unified command adapter
            from mcp_cli.adapters.chat import ChatCommandAdapter

            handled = await ChatCommandAdapter.handle_command(user_input, context)

            # Check if context requested exit
            if self.context.exit_requested:
                return True

            return handled

        except Exception as exc:
            logger.exception("Error handling command")
            output.error(f"Error handling command: {exc}")
            return False
