# src/mcp_cli/adapters/interactive.py
"""
Interactive mode adapter for unified commands.

Adapts unified commands to work with the interactive shell.
"""

from __future__ import annotations

import logging
import shlex
from typing import Any

from mcp_cli.commands.base import CommandMode
from mcp_cli.commands.registry import registry
from mcp_cli.context import get_context
from chuk_term.ui import output

logger = logging.getLogger(__name__)


class InteractiveExitException(Exception):
    """Custom exception for exiting interactive mode without interfering with pytest."""


class InteractiveCommandAdapter:
    """
    Adapts unified commands for use in interactive shell mode.

    Handles:
    - Command parsing without slash prefix
    - Shell-style argument parsing
    - Tab completion
    """

    @staticmethod
    async def handle_command(command_line: str) -> bool:
        """
        Handle an interactive command.

        Args:
            command_line: The full command line (e.g., "servers --detailed").

        Returns:
            True if command was handled, False otherwise.
        """
        if not command_line.strip():
            return False

        # Parse command line using shell-style parsing
        try:
            parts = shlex.split(command_line)
        except ValueError as e:
            output.error(f"Invalid command syntax: {e}")
            return False

        if not parts:
            return False

        command_name = parts[0]

        # Handle slash commands - strip the leading slash if present
        if command_name.startswith("/"):
            command_name = command_name[1:]

        args = parts[1:] if len(parts) > 1 else []

        logger.debug(f"Parsed command: {command_name}, args: {args}")

        # Look up command in registry
        command = registry.get(command_name, mode=CommandMode.INTERACTIVE)
        if not command:
            # Not a registered command, might be a shell command
            return False

        # Parse arguments into kwargs
        kwargs = InteractiveCommandAdapter._parse_arguments(command, args)

        # Add context if the command needs it
        if command.requires_context:
            context = get_context()
            if context:
                kwargs["tool_manager"] = context.tool_manager
                kwargs["model_manager"] = context.model_manager

        # Validate parameters
        error = command.validate_parameters(**kwargs)
        if error:
            output.error(error)
            return True  # Command was handled, just had an error

        try:
            # Execute command
            result = await command.execute(**kwargs)

            # Handle result
            if result.success:
                if result.output:
                    output.print(result.output)

                # Handle special actions
                if result.should_exit:
                    # Signal interactive mode to exit
                    raise InteractiveExitException()

                if result.should_clear:
                    # Clear the screen
                    from chuk_term.ui import clear_screen

                    clear_screen()
            else:
                if result.error:
                    output.error(result.error)
                else:
                    output.error(f"Command failed: {command_name}")

            return True

        except InteractiveExitException:
            # Re-raise exit exception
            raise
        except Exception as e:
            logger.exception(f"Error executing command: {command_name}")
            output.error(f"Command error: {str(e)}")
            return True

    @staticmethod
    def _parse_arguments(command: Any, args: list[str]) -> dict[str, Any]:
        """
        Parse shell-style arguments into kwargs.

        Handles:
        - Flags: --flag or -f
        - Options: --option value or --option=value
        - Positional arguments
        """
        kwargs: dict[str, Any] = {}
        i = 0
        positional: list[str] = []

        while i < len(args):
            arg = args[i]

            if arg.startswith("--"):
                # Long option
                if "=" in arg:
                    # --option=value format
                    option_name, value = arg[2:].split("=", 1)
                    kwargs[option_name] = value
                else:
                    option_name = arg[2:]

                    # Check if this is a flag
                    param = next(
                        (p for p in command.parameters if p.name == option_name), None
                    )

                    if param and param.is_flag:
                        kwargs[option_name] = True
                    elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                        # Has a value
                        kwargs[option_name] = args[i + 1]
                        i += 1
                    else:
                        # No value, treat as flag
                        kwargs[option_name] = True

            elif arg.startswith("-") and len(arg) > 1:
                # Short option(s)
                for c in arg[1:]:
                    # Map short option to long option if possible
                    # For now, just use the short option as-is
                    kwargs[c] = True

            else:
                # Positional argument
                positional.append(arg)

            i += 1

        # Add positional arguments (always as a list for consistency)
        if positional:
            kwargs["args"] = positional

        return kwargs

    @staticmethod
    def get_completions(partial_line: str, cursor_pos: int) -> list[str]:
        """
        Get tab completions for the current input.

        Args:
            partial_line: The current input line.
            cursor_pos: Cursor position in the line.

        Returns:
            List of possible completions.
        """
        # Get the text up to the cursor
        text = partial_line[:cursor_pos]

        # Parse what we have so far
        try:
            parts = shlex.split(text)
        except ValueError:
            # Incomplete quotes, etc.
            parts = text.split()

        if not parts or (len(parts) == 1 and not text.endswith(" ")):
            # Complete command names
            prefix = parts[0] if parts else ""
            completions = []

            for name in registry.get_command_names(
                mode=CommandMode.INTERACTIVE, include_aliases=True
            ):
                if name.startswith(prefix):
                    completions.append(name)

            return sorted(completions)

        else:
            # Complete command arguments
            command_name = parts[0]
            command = registry.get(command_name, mode=CommandMode.INTERACTIVE)

            if not command:
                return []

            # Get the current partial argument
            current_arg = parts[-1] if len(parts) > 1 and not text.endswith(" ") else ""

            completions = []

            # Complete parameter names
            for param in command.parameters:
                option = f"--{param.name}"
                if option.startswith(current_arg):
                    completions.append(option)

            # If parameter has choices, complete those
            if current_arg.startswith("--") and "=" in current_arg:
                option_name, partial_value = current_arg.split("=", 1)
                param_name = option_name[2:]

                found_param = next(
                    (p for p in command.parameters if p.name == param_name),
                    None,
                )

                if found_param and found_param.choices:
                    for choice in found_param.choices:
                        if str(choice).startswith(partial_value):
                            completions.append(f"{option_name}={choice}")

            return sorted(completions)

    @staticmethod
    def get_help() -> str:
        """
        Get help text for all available commands.

        Returns:
            Formatted help text.
        """
        lines = ["Available commands:", ""]

        for cmd in registry.list_commands(mode=CommandMode.INTERACTIVE):
            lines.append(f"  {cmd.name:<15} {cmd.description}")
            if cmd.aliases:
                lines.append(f"    Aliases: {', '.join(cmd.aliases)}")

        lines.append("")
        lines.append("Type 'help <command>' for detailed help on a specific command.")

        return "\n".join(lines)

    @staticmethod
    def get_command_help(command_name: str) -> str:
        """
        Get detailed help for a specific command.

        Args:
            command_name: Name of the command.

        Returns:
            Detailed help text.
        """
        command = registry.get(command_name, mode=CommandMode.INTERACTIVE)
        if not command:
            return f"Unknown command: {command_name}"

        return command.help_text or command.description
