# mcp_cli/commands/attach/attach.py
"""Slash command for staging file attachments.

Usage:
    /attach <path> [path2 ...]  — stage files for next message
    /attach list                — show staged files
    /attach clear               — clear staging area
"""

from __future__ import annotations

from typing import Any

from chuk_term.ui import output

from mcp_cli.commands.base import (
    CommandMode,
    CommandParameter,
    CommandResult,
    UnifiedCommand,
)


class AttachCommand(UnifiedCommand):
    """Stage files for the next chat message."""

    @property
    def name(self) -> str:
        return "attach"

    @property
    def aliases(self) -> list[str]:
        return ["file", "image"]

    @property
    def description(self) -> str:
        return "Attach files (images, text, audio) to the next message"

    @property
    def help_text(self) -> str:
        return (
            "/attach <path> [path2 ...]  — stage files for next message\n"
            "/attach list               — show staged files\n"
            "/attach clear              — clear staging area\n"
            "\n"
            "Aliases: /file, /image\n"
            "\n"
            "Supported types:\n"
            "  Images: .png .jpg .jpeg .gif .webp\n"
            "  Audio:  .mp3 .wav\n"
            "  Text:   .py .js .ts .txt .md .csv .json .html .xml .yaml .yml\n"
            "          .sh .bash .rs .go .java .c .cpp .h .hpp .rb .swift\n"
            "          .kt .sql .toml .ini .cfg .env .log .jsx .tsx\n"
        )

    @property
    def modes(self) -> CommandMode:
        return CommandMode.CHAT

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="args",
                type=str,
                required=False,
                help="File path(s) or subcommand (list, clear)",
            ),
        ]

    async def execute(self, **kwargs: Any) -> CommandResult:
        """Execute the /attach command."""
        chat_context = kwargs.get("chat_context")
        if not chat_context:
            return CommandResult(
                success=False,
                error="Attach command requires chat context.",
            )

        # Parse arguments
        args_val = kwargs.get("args", "")
        if isinstance(args_val, list):
            args_str = " ".join(str(a) for a in args_val)
        else:
            args_str = str(args_val).strip() if args_val else ""

        if not args_str:
            return self._show_usage()

        parts = args_str.split()
        action = parts[0].lower()

        # Subcommands
        if action == "list":
            return self._list_staged(chat_context)
        if action == "clear":
            return self._clear_staged(chat_context)

        # Treat all parts as file paths to stage
        return self._stage_files(chat_context, parts)

    # ── Subcommand handlers ──────────────────────────────────────────────

    def _show_usage(self) -> CommandResult:
        output.info("Usage: /attach <path> [path2 ...] | list | clear")
        return CommandResult(success=True)

    def _list_staged(self, ctx: Any) -> CommandResult:
        staged = ctx.attachment_staging.peek()
        if not staged:
            output.info("No attachments staged.")
            return CommandResult(success=True, output="No attachments staged.")

        lines = []
        for att in staged:
            size_kb = att.size_bytes / 1024
            lines.append(f"  {att.display_name}  ({att.mime_type}, {size_kb:.1f} KB)")
        summary = f"{len(staged)} staged attachment(s):\n" + "\n".join(lines)
        output.info(summary)
        return CommandResult(success=True, output=summary)

    def _clear_staged(self, ctx: Any) -> CommandResult:
        count = ctx.attachment_staging.count
        ctx.attachment_staging.clear()
        msg = f"Cleared {count} attachment(s)." if count else "Nothing to clear."
        output.info(msg)
        return CommandResult(success=True, output=msg)

    def _stage_files(self, ctx: Any, paths: list[str]) -> CommandResult:
        from mcp_cli.chat.attachments import process_local_file

        staged_names: list[str] = []
        errors: list[str] = []

        for path in paths:
            try:
                att = process_local_file(path)
                ctx.attachment_staging.stage(att)
                staged_names.append(att.display_name)
                output.success(f"Staged: {att.display_name} ({att.mime_type})")
            except (FileNotFoundError, ValueError) as exc:
                errors.append(f"{path}: {exc}")
                output.error(f"Cannot attach {path}: {exc}")

        if errors and not staged_names:
            return CommandResult(success=False, error="; ".join(errors))

        total = ctx.attachment_staging.count
        msg = f"Staged {len(staged_names)} file(s). Total pending: {total}"
        if errors:
            msg += f" ({len(errors)} failed)"
        return CommandResult(success=True, output=msg)
