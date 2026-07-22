# src/mcp_cli/commands/sessions/new.py
"""New session command — start a fresh conversation."""

from __future__ import annotations

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from chuk_term.ui import output


class NewSessionCommand(UnifiedCommand):
    """Start a new chat session."""

    @property
    def name(self) -> str:
        return "new"

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def description(self) -> str:
        return "Start a new chat session (saves current session first)"

    @property
    def help_text(self) -> str:
        return """
Start a new chat session.

Usage:
  /new              - Save current session and start fresh
  /new <name>       - Save current session and start fresh with a description
"""

    @property
    def modes(self) -> CommandMode:
        return CommandMode.CHAT

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="description",
                type=str,
                required=False,
                help="Optional description for the new session",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the new session command."""
        chat_context = kwargs.get("chat_context")
        if not chat_context:
            return CommandResult(success=False, error="No chat context available.")

        args = kwargs.get("args", "").strip()
        description = args if args else ""

        # Save current session first (if there's history)
        if hasattr(chat_context, "save_session") and chat_context.conversation_history:
            try:
                path = chat_context.save_session()
                if path:
                    output.info(f"Previous session saved: {path}")
            except Exception as e:
                output.warning(f"Could not save previous session: {e}")

        # Clear history and start fresh
        await chat_context.clear_conversation_history(keep_system_prompt=True)

        # Broadcast to dashboard if connected
        bridge = getattr(chat_context, "dashboard_bridge", None)
        if bridge is not None:
            try:
                from mcp_cli.dashboard.bridge import _envelope

                _aid = bridge.agent_id
                await bridge.broadcast(
                    _envelope(
                        "CONVERSATION_HISTORY",
                        {"agent_id": _aid, "messages": []},
                    )
                )
                config = bridge._build_config_state()
                if config:
                    await bridge.broadcast(_envelope("CONFIG_STATE", config))
                await bridge.broadcast(
                    _envelope(
                        "SESSION_STATE",
                        {
                            "agent_id": _aid,
                            "session_id": chat_context.session_id,
                            "description": description,
                        },
                    )
                )
            except Exception:
                pass

        output.success(
            f"New session started: {chat_context.session_id}"
            + (f" ({description})" if description else "")
        )
        return CommandResult(
            success=True,
            output=f"New session: {chat_context.session_id}",
        )
