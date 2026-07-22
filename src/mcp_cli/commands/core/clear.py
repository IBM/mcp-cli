# src/mcp_cli/commands/definitions/clear.py
"""
Unified clear command implementation.
"""

from __future__ import annotations

import logging

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandResult,
)

logger = logging.getLogger(__name__)


class ClearCommand(UnifiedCommand):
    """Clear the terminal screen."""

    @property
    def name(self) -> str:
        return "clear"

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def description(self) -> str:
        return "Clear the terminal screen"

    @property
    def help_text(self) -> str:
        return """
Clear the terminal screen.

Usage:
  /clear    - Clear screen (chat mode)
  clear     - Clear screen (interactive mode)
  
Aliases: cls
"""

    @property
    def modes(self) -> CommandMode:
        """Clear is only for chat and interactive modes."""
        return CommandMode.CHAT | CommandMode.INTERACTIVE

    @property
    def requires_context(self) -> bool:
        """Clear doesn't need tool manager context."""
        return False

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the clear command."""
        # Import needed modules
        from chuk_term.ui import clear_screen, display_chat_banner
        from mcp_cli.context import get_context
        from mcp_cli.model_management import ModelManager

        # Clear the screen first
        clear_screen()

        # Try to get context for banner information
        provider = None
        model = None
        additional_info = {}

        try:
            context = get_context()
            if context:
                # Get provider and model from context's model_manager
                if hasattr(context, "model_manager") and context.model_manager:
                    provider = context.model_manager.get_active_provider()
                    model = context.model_manager.get_active_model()

                # Get tool count if available
                if hasattr(context, "tool_manager") and context.tool_manager:
                    tool_manager = context.tool_manager
                    if hasattr(tool_manager, "get_tool_count"):
                        tool_count = tool_manager.get_tool_count()
                    elif hasattr(tool_manager, "list_tools"):
                        tools = tool_manager.list_tools()
                        tool_count = len(tools) if tools else 0
                    elif hasattr(tool_manager, "_tools"):
                        tool_count = len(tool_manager._tools)
                    else:
                        tool_count = None

                    if tool_count is not None and tool_count > 0:
                        additional_info["Tools"] = str(tool_count)
        except Exception as e:
            # Context not available, use ModelManager directly
            logger.debug("Context not available for banner: %s", e)
            try:
                model_manager = ModelManager()
                provider = model_manager.get_active_provider()
                model = model_manager.get_active_model()
            except Exception as e2:
                # Even ModelManager failed, use defaults
                logger.debug("ModelManager fallback also failed: %s", e2)

        # Display the welcome banner if we have provider and model
        if provider and model:
            display_chat_banner(
                provider=provider,
                model=model,
                additional_info=additional_info if additional_info else None,
            )

        return CommandResult(
            success=True,
            should_clear=False,  # Don't signal clear - we already did it
        )
