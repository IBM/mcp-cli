# src/mcp_cli/commands/sessions/sessions.py
"""Session management command — list, save, load, delete sessions."""

from __future__ import annotations

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.config.enums import SessionAction
from chuk_term.ui import output, format_table


class SessionsCommand(UnifiedCommand):
    """Manage saved sessions."""

    @property
    def name(self) -> str:
        return "sessions"

    @property
    def aliases(self) -> list[str]:
        return ["session"]

    @property
    def description(self) -> str:
        return "Manage saved conversation sessions"

    @property
    def help_text(self) -> str:
        return """
Manage saved conversation sessions.

Usage:
  /sessions              - List all saved sessions
  /sessions list         - List all saved sessions
  /sessions save         - Save current session
  /sessions load <id>    - Load a saved session
  /sessions delete <id>  - Delete a saved session
"""

    @property
    def modes(self) -> CommandMode:
        return CommandMode.CHAT

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="action",
                type=str,
                required=False,
                help="Action: list, save, load, delete",
            ),
            CommandParameter(
                name="session_id",
                type=str,
                required=False,
                help="Session ID for load/delete",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the sessions command."""
        from mcp_cli.chat.session_store import SessionStore

        chat_context = kwargs.get("chat_context")

        # Parse args
        args = kwargs.get("args", "").strip().split()
        action = args[0] if args else SessionAction.LIST
        session_id = args[1] if len(args) > 1 else None

        store = SessionStore(
            agent_id=getattr(chat_context, "agent_id", "default")
            if chat_context
            else "default"
        )

        if action == SessionAction.LIST:
            sessions = store.list_sessions()
            if not sessions:
                output.info("No saved sessions.")
                return CommandResult(success=True, output="No saved sessions.")

            table_data = []
            for s in sessions:
                table_data.append(
                    {
                        "ID": s.session_id,
                        "Updated": s.updated_at[:19],
                        "Provider/Model": f"{s.provider}/{s.model}",
                        "Messages": str(s.message_count),
                    }
                )

            table = format_table(
                table_data,
                title="Saved Sessions",
                columns=["ID", "Updated", "Provider/Model", "Messages"],
            )
            output.print_table(table)
            return CommandResult(success=True, data=table_data)

        elif action == SessionAction.SAVE:
            if not chat_context:
                return CommandResult(success=False, error="No chat context.")
            if hasattr(chat_context, "save_session"):
                path = chat_context.save_session()
                if path:
                    output.success(f"Session saved: {path}")
                    return CommandResult(success=True, output=f"Saved to {path}")
            return CommandResult(success=False, error="Failed to save session.")

        elif action == SessionAction.LOAD:
            if not session_id:
                return CommandResult(
                    success=False,
                    error="Session ID required. Usage: /sessions load <id>",
                )
            if not chat_context:
                return CommandResult(success=False, error="No chat context.")
            if hasattr(chat_context, "load_session"):
                if chat_context.load_session(session_id):
                    output.success(f"Session loaded: {session_id}")
                    return CommandResult(success=True, output=f"Loaded {session_id}")
            return CommandResult(
                success=False, error=f"Failed to load session: {session_id}"
            )

        elif action == SessionAction.DELETE:
            if not session_id:
                return CommandResult(
                    success=False,
                    error="Session ID required. Usage: /sessions delete <id>",
                )
            if store.delete(session_id):
                output.success(f"Session deleted: {session_id}")
                return CommandResult(success=True, output=f"Deleted {session_id}")
            return CommandResult(
                success=False, error=f"Session not found: {session_id}"
            )

        else:
            return CommandResult(
                success=False,
                error=f"Unknown action: {action}. Use list, save, load, or delete.",
            )
