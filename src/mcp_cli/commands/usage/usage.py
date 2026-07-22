# src/mcp_cli/commands/usage/usage.py
"""Token usage command — shows per-turn and cumulative token stats."""

from __future__ import annotations

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandResult,
)
from chuk_term.ui import output


class UsageCommand(UnifiedCommand):
    """Display token usage statistics."""

    @property
    def name(self) -> str:
        return "usage"

    @property
    def aliases(self) -> list[str]:
        return ["tokens", "cost"]

    @property
    def description(self) -> str:
        return "Show token usage statistics"

    @property
    def help_text(self) -> str:
        return """
Show token usage statistics for the current session.

Usage:
  /usage  - Display per-turn and cumulative token counts

Aliases: /tokens, /cost
"""

    @property
    def modes(self) -> CommandMode:
        return CommandMode.CHAT

    @property
    def parameters(self) -> list:
        return []

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the usage command."""
        chat_context = kwargs.get("chat_context")
        if not chat_context:
            return CommandResult(success=False, error="No chat context available.")

        tracker = getattr(chat_context, "token_tracker", None)
        if tracker is None or tracker.turn_count == 0:
            output.info("No token usage recorded yet.")
            return CommandResult(success=True, output="No usage data.")

        # Format summary
        summary = tracker.format_summary()
        output.panel(summary, title="Token Usage")

        return CommandResult(success=True, output=summary)
