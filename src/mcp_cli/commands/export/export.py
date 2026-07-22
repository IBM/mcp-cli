# src/mcp_cli/commands/export/export.py
"""Export conversation to Markdown or JSON."""

from __future__ import annotations

from pathlib import Path

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from chuk_term.ui import output


class ExportCommand(UnifiedCommand):
    """Export conversation to file."""

    @property
    def name(self) -> str:
        return "export"

    @property
    def aliases(self) -> list[str]:
        return ["save-chat"]

    @property
    def description(self) -> str:
        return "Export conversation to markdown or JSON"

    @property
    def help_text(self) -> str:
        return """
Export the current conversation to a file.

Usage:
  /export markdown [filename]  - Export as formatted Markdown
  /export json [filename]      - Export as structured JSON

Default filenames: chat-<session_id>.md / .json
"""

    @property
    def modes(self) -> CommandMode:
        return CommandMode.CHAT

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="format",
                type=str,
                required=False,
                help="Export format: markdown or json (default: markdown)",
            ),
            CommandParameter(
                name="filename",
                type=str,
                required=False,
                help="Output filename (optional)",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the export command."""
        from mcp_cli.chat.exporters import MarkdownExporter, JSONExporter

        chat_context = kwargs.get("chat_context")
        if not chat_context:
            return CommandResult(success=False, error="No chat context available.")

        # Parse arguments
        args = kwargs.get("args", "").strip().split()
        export_format = args[0] if args else "markdown"
        filename = args[1] if len(args) > 1 else None

        # Get messages from context
        messages = []
        try:
            raw_messages = chat_context.get_conversation_history()
            messages = [
                m.to_dict() if hasattr(m, "to_dict") else m for m in raw_messages
            ]
        except Exception as e:
            return CommandResult(success=False, error=f"Failed to get history: {e}")

        if not messages:
            output.info("No messages to export.")
            return CommandResult(success=True, output="No messages to export.")

        # Build metadata
        metadata = {
            "session_id": getattr(chat_context, "session_id", "unknown"),
            "provider": getattr(chat_context, "provider", "unknown"),
            "model": getattr(chat_context, "model", "unknown"),
        }

        # Build token usage
        token_usage = None
        tracker = getattr(chat_context, "token_tracker", None)
        if tracker and tracker.turn_count > 0:
            token_usage = {
                "total_input": tracker.total_input,
                "total_output": tracker.total_output,
                "total_tokens": tracker.total_tokens,
                "turn_count": tracker.turn_count,
            }

        # Export
        session_id = getattr(chat_context, "session_id", "chat")
        if export_format.startswith("json"):
            content = JSONExporter.export(messages, metadata, token_usage)
            if not filename:
                filename = f"chat-{session_id}.json"
        else:
            content = MarkdownExporter.export(messages, metadata)
            if not filename:
                filename = f"chat-{session_id}.md"

        # Write to file
        try:
            path = Path(filename)
            path.write_text(content, encoding="utf-8")
            output.success(f"Exported to {path.resolve()}")
            return CommandResult(success=True, output=f"Exported to {path}")
        except Exception as e:
            return CommandResult(success=False, error=f"Failed to write file: {e}")
