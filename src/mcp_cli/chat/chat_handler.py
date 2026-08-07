# mcp_cli/chat/chat_handler.py
"""
Clean chat handler that uses ModelManager and ChatContext with streaming support.
"""

from __future__ import annotations

import asyncio
import gc
import logging

# UI imports — this module is the boundary between core and UI.
# It wires chuk_term UI components to core ChatContext/ConversationProcessor.
# Kept at module level for testability (tests patch these names).
from chuk_term.ui import (
    output,
    clear_screen,
    display_chat_banner,
    display_error_banner,
)

# Local imports
from mcp_cli.chat.chat_context import ChatContext
from mcp_cli.chat.testing import TestChatContext
from mcp_cli.chat.ui_manager import ChatUIManager
from mcp_cli.chat.conversation import ConversationProcessor
from mcp_cli.tools.manager import ToolManager
from mcp_cli.context import initialize_context
from mcp_cli.config import initialize_config
from mcp_cli.config.defaults import DEFAULT_PROVIDER, DEFAULT_MODEL, DEFAULT_CONFIG_FILENAME

# Set up logger
logger = logging.getLogger(__name__)


async def handle_chat_mode(
    tool_manager: ToolManager,
    provider: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    confirm_mode: str | None = None,
    max_turns: int = 100,
    model_manager=None,  # FIXED: Accept model_manager from caller
    runtime_config=None,  # RuntimeConfig | None
    max_history_messages: int = 0,
    enable_vm: bool = False,
    vm_mode: str = "passive",
    vm_budget: int = 128_000,
    health_interval: int = 0,
    enable_plan_tools: bool = False,
    dashboard: bool = False,
    no_browser: bool = False,
    dashboard_port: int = 0,
    agent_id: str = "default",
    multi_agent: bool = False,
    initial_attachments: list[str] | None = None,
    no_tools: bool = False,
) -> bool:
    """
    Launch the interactive chat loop with streaming support.

    Runtime uses adaptive policy: strict core with smooth wrapper.
    - Always enforces grounding rules (no ungrounded numeric calls)
    - Automatically attempts to repair blocked calls (rebind, symbolic fallback)
    - Only surfaces errors when all repair options exhausted

    Args:
        tool_manager: Initialized ToolManager instance
        provider: Provider to use (optional, uses ModelManager active if None)
        model: Model to use (optional, uses ModelManager active if None)
        api_base: API base URL override (optional)
        api_key: API key override (optional)
        confirm_mode: Tool confirmation mode override (optional)
        max_turns: Maximum conversation turns before forcing exit (default: 100)
        model_manager: Pre-configured ModelManager (optional, creates new if None)
        runtime_config: Runtime configuration with timeout overrides (optional)
        dashboard: Launch browser dashboard alongside chat (requires websockets).
        no_browser: If True, print dashboard URL but do not open the browser.
        dashboard_port: Preferred dashboard port (0 = auto-select).
        multi_agent: Enable multi-agent orchestration tools (implies dashboard).

    Returns:
        True if session ended normally, False on failure
    """
    # Multi-agent implies dashboard (needs the router for agent bridges)
    if multi_agent:
        dashboard = True

    ui: ChatUIManager | None = None
    ctx = None
    agent_manager = None

    try:
        # Initialize configuration manager
        from pathlib import Path

        # Use the project-wide constant instead of a hardcoded filename
        initialize_config(Path(DEFAULT_CONFIG_FILENAME))

        # Initialize global context manager for commands to work
        app_context = initialize_context(
            tool_manager=tool_manager,
            provider=provider or DEFAULT_PROVIDER,
            model=model or DEFAULT_MODEL,
            api_base=api_base,
            api_key=api_key,
            model_manager=model_manager,  # FIXED: Pass model_manager with runtime providers
        )

        # Create chat context with progress reporting
        def on_progress(msg: str) -> None:
            output.info(msg)

        # FIXED: Use the model_manager from app_context to ensure consistency
        ctx = ChatContext.create(
            tool_manager=tool_manager,
            provider=provider,
            model=model,
            api_base=api_base,
            api_key=api_key,
            model_manager=app_context.model_manager,  # Use the same instance
            max_history_messages=max_history_messages,
            enable_vm=enable_vm,
            vm_mode=vm_mode,
            vm_budget=vm_budget,
            health_interval=health_interval,
            enable_plan_tools=enable_plan_tools,
            agent_id=agent_id,
        )

        if not await ctx.initialize(on_progress=on_progress):
            output.error("Failed to initialize chat context.")
            return False

        ctx.no_tools = no_tools

        # Stage initial attachments from --attach CLI flag
        if initial_attachments:
            from mcp_cli.chat.attachments import process_local_file

            for path in initial_attachments:
                try:
                    att = process_local_file(path)
                    ctx.attachment_staging.stage(att)
                    logger.info("Staged initial attachment: %s", att.display_name)
                except (FileNotFoundError, ValueError) as exc:
                    output.error(f"Cannot attach {path}: {exc}")

        # Update global context with initialized data
        await app_context.initialize()

        # Start dashboard if requested
        if dashboard:
            try:
                from mcp_cli.dashboard.launcher import launch_dashboard
                from mcp_cli.dashboard.bridge import DashboardBridge

                _dash_server, _dash_router, _dash_port = await launch_dashboard(
                    dashboard_port, no_browser
                )
                output.info(f"Dashboard: http://localhost:{_dash_port}")
                ctx.dashboard_bridge = DashboardBridge(
                    _dash_router, agent_id=ctx.agent_id
                )
                ctx.dashboard_bridge.set_context(ctx)
                _dash_router.register_agent(ctx.agent_id, ctx.dashboard_bridge)

                # Wire REQUEST_TOOL from browser → tool_manager, result back to browser
                _bridge_ref = ctx.dashboard_bridge

                async def _dashboard_execute_tool(
                    tool_name: str, arguments: dict
                ) -> None:
                    result = await tool_manager.execute_tool(tool_name, arguments)
                    await _bridge_ref.on_tool_result(
                        tool_name=tool_name,
                        server_name="",
                        result=result.result,
                        success=result.success,
                        error=result.error,
                    )

                ctx.dashboard_bridge.set_tool_call_callback(_dashboard_execute_tool)

                # Multi-agent: create AgentManager and set on context
                if multi_agent:
                    from mcp_cli.agents.manager import AgentManager as _AM

                    agent_manager = _AM(
                        tool_manager=tool_manager,
                        router=_dash_router,
                        model_manager=app_context.model_manager,
                    )
                    ctx.agent_manager = agent_manager
                    logger.info("Multi-agent mode enabled (%s)", ctx.agent_id)

            except ImportError:
                output.warning(
                    "Dashboard requires 'websockets'. Install with: pip install mcp-cli[dashboard]"
                )

        # Welcome banner
        # Clear screen unless in debug mode
        if logger.level > logging.DEBUG:
            clear_screen()

        # NEW: Use the new banner function
        # Get tool count safely
        tool_count: int | str = 0
        if tool_manager:
            try:
                # Try to get tool count - ToolManager might have different ways to access this
                if hasattr(tool_manager, "get_tool_count"):
                    tool_count = tool_manager.get_tool_count()
                elif hasattr(tool_manager, "list_tools"):
                    tools = tool_manager.list_tools()
                    tool_count = len(tools) if tools else 0
                elif hasattr(tool_manager, "_tools"):
                    tool_count = len(tool_manager._tools)
                # Just show that we have a tool manager but don't know the count
                else:
                    tool_count = "Available"
            except Exception as e:
                logger.debug("Failed to get tool count: %s", e)
                tool_count = "Unknown"

        additional_info = {}
        if api_base:
            additional_info["API Base"] = api_base
        if tool_count != 0:
            additional_info["Tools"] = (
                str(tool_count) if isinstance(tool_count, int) else tool_count
            )

        display_chat_banner(
            provider=ctx.provider,
            model=ctx.model,
            additional_info=additional_info if additional_info else None,
        )

        # UI and conversation processor
        ui = ChatUIManager(ctx)
        convo = ConversationProcessor(ctx, ui, runtime_config)

        # Main chat loop with streaming support
        await _run_enhanced_chat_loop(ui, ctx, convo, max_turns)

        return True

    except Exception as exc:
        logger.exception("Error in chat mode")
        # NEW: Use error banner for better visibility
        display_error_banner(
            exc,
            context="During chat mode initialization",
            suggestions=[
                "Check your API credentials",
                "Verify network connectivity",
                "Try a different model or provider",
            ],
        )
        return False

    finally:
        # Auto-save session on exit
        if ctx is not None and ctx.conversation_history:
            try:
                saved = ctx.save_session()
                if saved:
                    logger.info("Session auto-saved: %s", saved)
            except Exception as exc:
                logger.warning("Failed to auto-save session: %s", exc)

        # Cleanup
        if ui:
            await _safe_cleanup(ui)

        # Stop all managed agents before tearing down dashboard
        if agent_manager is not None:
            try:
                await agent_manager.stop_all()
            except Exception as exc:
                logger.warning("Error stopping agents: %s", exc)

        # Stop dashboard server if running
        if ctx is not None and ctx.dashboard_bridge is not None:
            try:
                await ctx.dashboard_bridge.on_shutdown()
                await ctx.dashboard_bridge.server.stop()
            except Exception as exc:
                logger.warning("Error stopping dashboard server: %s", exc)

        # Close tool manager
        try:
            await tool_manager.close()
        except Exception as exc:
            logger.warning(f"Error closing ToolManager: {exc}")

        gc.collect()


async def handle_chat_mode_for_testing(
    stream_manager,
    provider: str | None = None,
    model: str | None = None,
    max_turns: int = 100,
    runtime_config=None,  # RuntimeConfig | None
) -> bool:
    """
    Launch chat mode for testing with stream_manager.

    Separated from main function to keep it clean.

    Args:
        stream_manager: Test stream manager
        provider: Provider for testing
        model: Model for testing
        max_turns: Maximum conversation turns before forcing exit (default: 100)
        runtime_config: Runtime configuration with timeout overrides (optional)

    Returns:
        True if session ended normally, False on failure
    """
    ui: ChatUIManager | None = None

    try:
        # Create test chat context
        with output.loading("Initializing test chat context..."):
            ctx = TestChatContext.create_for_testing(
                stream_manager=stream_manager, provider=provider, model=model
            )

            if not await ctx.initialize():
                output.error("Failed to initialize test chat context.")
                return False

        # Welcome banner
        clear_screen()
        display_chat_banner(
            provider=ctx.provider, model=ctx.model, additional_info={"Mode": "Testing"}
        )

        # UI and conversation processor
        ui = ChatUIManager(ctx)
        convo = ConversationProcessor(ctx, ui, runtime_config)

        # Main chat loop with streaming support
        await _run_enhanced_chat_loop(ui, ctx, convo, max_turns)

        return True

    except Exception as exc:
        logger.exception("Error in test chat mode")
        display_error_banner(
            exc,
            context="During test chat mode",
            suggestions=["Check test configuration", "Verify mock responses"],
        )
        return False

    finally:
        if ui:
            await _safe_cleanup(ui)
        gc.collect()


# Sentinel placed on the input queue when the reader catches a KeyboardInterrupt.
# Allows the main loop to handle Ctrl+C that originated inside get_user_input().
_INTERRUPT = object()


async def _terminal_reader(
    ui: ChatUIManager,
    queue: asyncio.Queue,
    ready: asyncio.Event | None = None,
) -> None:
    """Background task: reads terminal input and puts it on the shared queue.

    When *ready* is provided the reader waits for it before showing the prompt.
    This prevents the prompt from being overwritten by streaming / tool output.
    The reader clears the event immediately (one-shot) so it won't re-enter
    ``get_user_input`` until the main loop explicitly re-sets the event.
    """
    while True:
        try:
            # Wait until the main loop signals it's ready for new input.
            # Clear immediately so we only get ONE prompt per signal.
            if ready is not None:
                await ready.wait()
                ready.clear()

            msg = await ui.get_user_input()
            await queue.put(msg)
        except EOFError:
            await queue.put("exit")
            break
        except asyncio.CancelledError:
            # Re-raise only when this task itself was explicitly cancelled
            # (reader_task.cancel()). If CancelledError came from get_user_input()
            # directly (e.g. in tests), treat it as an interrupt signal instead.
            if asyncio.current_task().cancelling():  # type: ignore[union-attr]
                raise
            await queue.put(_INTERRUPT)
        except KeyboardInterrupt:
            # Forward the interrupt to the main loop via the sentinel so the
            # reader keeps running (allows subsequent input after Ctrl+C).
            await queue.put(_INTERRUPT)
        except Exception as exc:
            logger.debug("Terminal reader error: %s", exc)
        # Yield to the event loop each iteration so that task.cancel() can
        # deliver CancelledError even when get_user_input() resolves instantly
        # (e.g. in tests with AsyncMock).
        await asyncio.sleep(0)


async def _run_enhanced_chat_loop(
    ui: ChatUIManager,
    ctx: ChatContext,
    convo: ConversationProcessor,
    max_turns: int = 100,
) -> None:
    """
    Run the main chat loop with enhanced streaming support.

    Args:
        ui: UI manager with streaming coordination
        ctx: Chat context
        convo: Conversation processor with streaming support
        max_turns: Maximum conversation turns before forcing exit (default: 100)
    """
    # Shared queue: terminal reader task and browser WebSocket both put messages here.
    # This lets browser input arrive during the terminal prompt wait.
    # Type is Any because we also put _INTERRUPT sentinel objects on the queue.
    input_queue: asyncio.Queue = asyncio.Queue()

    # Wire dashboard bridge so browser USER_MESSAGE/USER_COMMAND go into the queue.
    if bridge := getattr(ctx, "dashboard_bridge", None):
        bridge.set_input_queue(input_queue)

    # Gate the prompt display: the reader waits for this event before showing
    # the "💬 You:" prompt, preventing streaming / tool output from overwriting it.
    prompt_ready = asyncio.Event()
    prompt_ready.set()  # Ready immediately for the first prompt

    # Background task: reads terminal input and forwards to the queue.
    reader_task = asyncio.create_task(
        _terminal_reader(ui, input_queue, ready=prompt_ready)
    )

    try:
        while True:
            try:
                user_msg = await input_queue.get()

                # Handle interrupt sentinel forwarded from _terminal_reader
                if user_msg is _INTERRUPT:
                    logger.info(
                        "Interrupt forwarded from reader — streaming=%s, tools_running=%s",
                        ui.is_streaming_response,
                        ui.tools_running,
                    )
                    if ui.is_streaming_response:
                        output.warning("\nStreaming interrupted - type 'exit' to quit.")
                        ui.interrupt_streaming()
                    elif ui.tools_running:
                        output.warning(
                            "\nTool execution interrupted - type 'exit' to quit."
                        )
                        ui._interrupt_now()
                    else:
                        output.warning("\nInterrupted - type 'exit' to quit.")
                    prompt_ready.set()
                    continue

                # Skip empty messages
                if not user_msg:
                    prompt_ready.set()
                    continue

                # Handle plain exit commands (without slash)
                if user_msg.lower() in ("exit", "quit"):
                    output.panel("Exiting chat mode.", style="red", title="Goodbye")
                    break

                # Handle slash commands
                if user_msg.startswith("/"):
                    # Special handling for interrupt command during streaming
                    if user_msg.lower() in ("/interrupt", "/stop", "/cancel"):
                        if ui.is_streaming_response:
                            ui.interrupt_streaming()
                            output.warning("Streaming response interrupted.")
                            prompt_ready.set()
                            continue
                        elif ui.tools_running:
                            ui._interrupt_now()
                            prompt_ready.set()
                            continue
                        else:
                            output.info("Nothing to interrupt.")
                            prompt_ready.set()
                            continue

                    handled = await ui.handle_command(user_msg)
                    if ctx.exit_requested:
                        break
                    if handled:
                        prompt_ready.set()
                        continue

                # Normal conversation turn with streaming support
                if ui.verbose_mode:
                    ui.print_user_message(user_msg)

                # Multi-modal processing: inline refs, staged attachments, image URLs
                from mcp_cli.chat.attachments import (
                    parse_inline_refs,
                    process_local_file,
                    detect_image_urls,
                    build_multimodal_content,
                )

                cleaned_text, inline_paths = parse_inline_refs(user_msg)

                # Process inline @file: references
                inline_atts = []
                for p in inline_paths:
                    try:
                        inline_atts.append(process_local_file(p))
                    except (FileNotFoundError, ValueError) as exc:
                        output.error(f"Cannot attach {p}: {exc}")

                # Drain staged attachments from /attach command
                staged = ctx.attachment_staging.drain()

                # Detect image URLs in message text
                image_urls = detect_image_urls(cleaned_text)

                # Build content (str when text-only, list[dict] when multi-modal)
                content = build_multimodal_content(
                    cleaned_text, staged + inline_atts, image_urls
                )

                await ctx.add_user_message(content)

                # Dashboard: broadcast user message with attachment metadata
                if _dash := getattr(ctx, "dashboard_bridge", None):
                    try:
                        att_descriptors = None
                        all_atts = staged + inline_atts
                        if all_atts or image_urls:
                            from mcp_cli.chat.attachments import (
                                attachment_descriptor,
                                process_url as _process_url,
                            )

                            att_descriptors = [
                                attachment_descriptor(a) for a in all_atts
                            ]
                            for _url in image_urls:
                                att_descriptors.append(
                                    attachment_descriptor(_process_url(_url))
                                )
                        await _dash.on_message(
                            "user", user_msg, attachments=att_descriptors
                        )
                    except Exception as _e:
                        logger.debug("Dashboard on_message(user) error: %s", _e)

                # Process the conversation. The reader already cleared
                # prompt_ready so no new prompt is shown during streaming.
                try:
                    await convo.process_conversation(max_turns=max_turns)
                finally:
                    prompt_ready.set()

            except (KeyboardInterrupt, asyncio.CancelledError):
                # Handle Ctrl+C gracefully
                logger.info(
                    f"Interrupt in chat loop - streaming={ui.is_streaming_response}, tools_running={ui.tools_running}"
                )
                if ui.is_streaming_response:
                    output.warning("\nStreaming interrupted - type 'exit' to quit.")
                    ui.interrupt_streaming()
                elif ui.tools_running:
                    output.warning(
                        "\nTool execution interrupted - type 'exit' to quit."
                    )
                    ui._interrupt_now()
                else:
                    output.warning("\nInterrupted - type 'exit' to quit.")
                # CRITICAL: Continue the loop instead of exiting
                logger.info("Continuing chat loop after interrupt...")
                prompt_ready.set()
                continue
            except EOFError:
                output.panel("EOF detected - exiting chat.", style="red", title="Exit")
                break
            except Exception as exc:
                logger.exception("Error processing message")
                output.error(f"Error processing message: {exc}")
                prompt_ready.set()
                continue
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass


async def _safe_cleanup(ui: ChatUIManager) -> None:
    """
    Safely cleanup UI manager with enhanced error handling.

    Args:
        ui: UI manager to cleanup
    """
    try:
        # Stop any streaming responses
        if ui.is_streaming_response:
            ui.interrupt_streaming()
            ui.stop_streaming_response_sync()

        # Stop any tool execution
        if ui.tools_running:
            ui.stop_tool_calls()

        # Standard cleanup
        ui.cleanup()
    except Exception as exc:
        logger.warning(f"Cleanup failed: {exc}")
        output.warning(f"Cleanup failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════════
# Enhanced interrupt command for chat mode
# ═══════════════════════════════════════════════════════════════════════════════════


async def handle_interrupt_command(ui: ChatUIManager) -> bool:
    """
    Handle the /interrupt command with streaming awareness.

    Args:
        ui: UI manager instance

    Returns:
        True if command was handled
    """
    if ui.is_streaming_response:
        ui.interrupt_streaming()
        output.success("Streaming response interrupted.")
    elif ui.tools_running:
        ui._interrupt_now()
        output.success("Tool execution interrupted.")
    else:
        output.info("Nothing currently running to interrupt.")

    return True


# ═══════════════════════════════════════════════════════════════════════════════════
# Usage examples:
# ═══════════════════════════════════════════════════════════════════════════════════

"""
# Production usage with streaming:
success = await handle_chat_mode(
    tool_manager,
    provider="anthropic",
    model="claude-3-sonnet",
    api_key="your-key"
)

# Test usage:
success = await handle_chat_mode_for_testing(
    stream_manager,
    provider="openai",
    model="gpt-4"
)

# Simple usage (uses ModelManager defaults):
success = await handle_chat_mode(tool_manager)
"""
