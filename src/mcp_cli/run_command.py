# mcp_cli/run_command.py
"""
Main entry-point helpers for all CLI sub-commands.

These helpers encapsulate

* construction / cleanup of the shared **ToolManager**
* hand-off to individual command modules
* a thin synchronous wrapper so `uv run mcp-cli …` works

ENHANCED: Now properly handles namespace selection for HTTP vs STDIO servers.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING, Any, Callable
from pathlib import Path

import typer
from rich.panel import Panel
from chuk_term.ui import output

from mcp_cli.config.defaults import DEFAULT_PROVIDER, DEFAULT_MODEL, DEFAULT_CONFIG_FILENAME
from mcp_cli.tools.manager import ToolManager
from mcp_cli.context import initialize_context

if TYPE_CHECKING:
    from mcp_cli.config.runtime import RuntimeConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# internal helpers / globals                                                  #
# --------------------------------------------------------------------------- #
_ALL_TM: list[ToolManager] = []  # referenced by the unit-tests

# Factory function for ToolManager - can be patched in tests
_tool_manager_factory: Callable[..., ToolManager] | None = None


def set_tool_manager_factory(factory: Callable[..., ToolManager] | None) -> None:
    """Set a custom ToolManager factory for testing.

    Args:
        factory: A callable that creates ToolManager instances, or None to reset.
    """
    global _tool_manager_factory
    _tool_manager_factory = factory


def _create_tool_manager(
    config_file: str,
    servers: list[str],
    server_names: dict[int, str | None] | None = None,
    initialization_timeout: float = 120.0,
    runtime_config: "RuntimeConfig | None" = None,
) -> ToolManager:
    """Create a ToolManager instance using factory or default constructor.

    This pattern allows tests to inject mock ToolManagers without dynamic getattr.
    """
    if _tool_manager_factory is not None:
        return _tool_manager_factory(
            config_file,
            servers,
            server_names,
            initialization_timeout=initialization_timeout,
            runtime_config=runtime_config,
        )
    return ToolManager(
        config_file,
        servers,
        server_names,
        initialization_timeout=initialization_timeout,
        runtime_config=runtime_config,
    )


# --------------------------------------------------------------------------- #
# Tool-manager helpers                                                        #
# --------------------------------------------------------------------------- #
async def _init_tool_manager(
    config_file: str,
    servers: list[str],
    server_names: dict[int, str | None] | None = None,
    initialization_timeout: float = 120.0,
    runtime_config: "RuntimeConfig | None" = None,
) -> ToolManager:
    """
    Create and initialize a ToolManager instance.

    Uses _create_tool_manager() which can be patched via set_tool_manager_factory()
    for testing purposes.

    ENHANCED: Automatically selects appropriate namespace based on server type.
    """
    tm = _create_tool_manager(
        config_file,
        servers,
        server_names,
        initialization_timeout=initialization_timeout,
        runtime_config=runtime_config,
    )

    # ENHANCED: Let ToolManager automatically select the namespace
    # It will use the server name for HTTP servers, "stdio" for STDIO servers
    ok = await tm.initialize()  # Remove the hardcoded namespace parameter

    # Clean up any loggers that were created during initialization
    from mcp_cli.config.logging import setup_logging
    import os

    log_level = os.environ.get("LOG_LEVEL", "WARNING")
    setup_logging(level=log_level, quiet=False, verbose=False)

    if not ok:
        # Check if this is just because there are no servers
        if not servers:
            logger.info("No servers configured - continuing with empty tool manager")
            # Still record and return the manager for chat without tools

            _ALL_TM.append(tm)
            return tm

        # record it for the tests
        _ALL_TM.append(tm)
        # ensure close() is still invoked
        try:
            await tm.close()
        finally:
            raise RuntimeError("Failed to initialise ToolManager")

    _ALL_TM.append(tm)
    return tm


async def _safe_close(tm) -> None:
    """Close the ToolManager, swallowing any exception during shutdown."""
    try:
        await tm.close()
    except Exception as e:  # noqa: BLE001
        logger.debug("Error during ToolManager close: %s", e)


# --------------------------------------------------------------------------- #
# command dispatch                                                            #
# --------------------------------------------------------------------------- #
async def run_command(
    async_command: Callable[..., Any],
    *,
    config_file: str,
    servers: list[str],
    extra_params: dict[str, Any | None],
) -> Any:
    """
    Initialise the ToolManager and context, then call *async_command(...)*.

    The *async_command* may itself be `async` **or** synchronous – both work.
    The ToolManager is always closed, even when the callable raises.
    """
    tm = None
    try:
        server_names = (extra_params or {}).get("server_names")
        init_timeout = (extra_params or {}).get("init_timeout", 120.0)

        # ------------------------------------------------------------------
        # build ToolManager  (patch-friendly, see helper)
        # ------------------------------------------------------------------
        tm = await _init_tool_manager(
            config_file, servers, server_names, init_timeout or 120.0
        )

        # ------------------------------------------------------------------
        # Initialize context with tool manager and other params
        # ------------------------------------------------------------------
        context = initialize_context(
            tool_manager=tm,
            config_path=Path(config_file),
            provider=(extra_params or {}).get("provider", DEFAULT_PROVIDER),
            model=(extra_params or {}).get("model", DEFAULT_MODEL),
            api_base=(extra_params or {}).get("api_base"),
            api_key=(extra_params or {}).get("api_key"),
        )

        # Initialize the context (load servers, tools, etc.)
        await context.initialize()

        # ------------------------------------------------------------------
        # special-case: interactive "app" object
        # ------------------------------------------------------------------
        name = getattr(async_command, "__name__", "")
        module = getattr(async_command, "__module__", "")
        if name == "app" and "interactive" in module:
            provider = context.provider
            model = context.model
            result = await _enter_interactive_mode(tm, provider=provider, model=model)
            return result

        # ------------------------------------------------------------------
        # normal pathway - commands no longer need context passed
        # ------------------------------------------------------------------
        # Remove tool_manager and context from call_kwargs since commands get it from context
        call_kwargs = {}

        # Pass all extra_params to the command
        if extra_params:
            call_kwargs.update(extra_params)

        maybe_coro = async_command(**call_kwargs)

        # support sync callables transparently
        if asyncio.iscoroutine(maybe_coro):
            return await maybe_coro
        else:
            loop = asyncio.get_running_loop()
            # run in default executor so we don't block the event-loop
            return await loop.run_in_executor(None, lambda: maybe_coro)
    finally:
        if tm:
            await _safe_close(tm)


def run_command_sync(
    async_command: Callable[..., Any],
    config_file: str,
    servers: list[str],
    *,
    extra_params: dict[str, Any | None],
) -> Any:
    """
    Synchronous convenience wrapper (used by `mcp-cli` entry-point and tests).

    Spins up its own event-loop when necessary.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        run_command(
            async_command,
            config_file=config_file,
            servers=servers,
            extra_params=extra_params,
        )
    )


# --------------------------------------------------------------------------- #
# specialised helpers (chat / interactive)                                    #
# --------------------------------------------------------------------------- #
async def _enter_chat_mode(
    tool_manager,
    *,
    provider: str,
    model: str,
) -> bool:
    """Kick off the full-screen chat UI (banner printed by the handler)."""
    from mcp_cli.chat.chat_handler import handle_chat_mode

    return await handle_chat_mode(
        tool_manager,
        provider=provider,
        model=model,
    )


async def _enter_interactive_mode(
    tool_manager,
    *,
    provider: str,
    model: str,
) -> bool:
    """
    Start the interactive-shell UI.

    We pass in the *StreamManager* plus the ToolManager for parity with
    production code.
    """
    from mcp_cli.commands.interactive import interactive_mode

    result: bool = await interactive_mode(
        stream_manager=tool_manager.stream_manager,
        tool_manager=tool_manager,
        provider=provider,
        model=model,
    )
    return result


# --------------------------------------------------------------------------- #
# CLI entry-point (used by the `mcp-cli` wrapper script)                      #
# --------------------------------------------------------------------------- #
app = typer.Typer(add_completion=False, help="Master control programme 🙂")


@app.command("run")
def cli_entry(
    mode: str = typer.Argument("chat", help="chat | interactive"),
    config_file: str = typer.Option(
        DEFAULT_CONFIG_FILENAME, "--config", "-c", help="Server config file"
    ),
    server: list[str] = typer.Option(
        ["sqlite"], "--server", "-s", help="Server(s) to connect"
    ),
    provider: str = typer.Option("ollama", help="LLM provider name"),
    model: str = typer.Option("gpt-oss", help="LLM model name"),
    init_timeout: float = typer.Option(
        120.0, "--init-timeout", help="Server initialization timeout in seconds"
    ),
) -> None:
    """
    Thin wrapper so `uv run mcp-cli chat` (or `interactive`) is minimal.
    """

    async def _inner() -> None:
        if mode not in {"chat", "interactive"}:
            raise typer.BadParameter("mode must be 'chat' or 'interactive'")

        tm = await _init_tool_manager(
            config_file, server, initialization_timeout=init_timeout
        )

        try:
            if mode == "chat":
                ok = await _enter_chat_mode(tm, provider=provider, model=model)
            else:
                ok = await _enter_interactive_mode(tm, provider=provider, model=model)

            if not ok:
                raise RuntimeError("Command returned non-zero status")

        finally:
            await _safe_close(tm)

    try:
        asyncio.run(_inner())
    except Exception as exc:  # noqa: BLE001 – show nicely then exit
        output.print(Panel(str(exc), title="Fatal Error", style="bold red"))
        sys.exit(1)
