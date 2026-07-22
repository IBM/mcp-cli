# src/mcp_cli/commands/__init__.py
"""
Unified command system for MCP CLI.

This module provides a single command implementation that works across:
- Chat mode (slash commands)
- CLI mode (typer subcommands)
- Interactive mode (shell commands)
"""

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandGroup,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.commands.registry import registry, UnifiedCommandRegistry
from mcp_cli.commands.decorators import validate_params, handle_errors
from mcp_cli.commands.exceptions import (
    CommandError,
    InvalidParameterError,
    CommandExecutionError,
    CommandNotFoundError,
    ValidationError,
)
from mcp_cli.commands import utils as command_utils  # noqa: F401
from mcp_cli.commands.models import (
    # Server models
    ServerActionParams,
    ServerStatusInfo,
    ServerPerformanceInfo,
    # Model models
    ModelActionParams,
    ModelInfo,
    # Provider models
    ProviderActionParams,
    ProviderInfo,
    # Token models
    TokenListParams,
    TokenSetParams,
    TokenDeleteParams,
    TokenClearParams,
    TokenProviderParams,
    # Tool models
    ToolActionParams,
    ToolCallParams,
    # Resource models
    ResourceActionParams,
    # Prompt models
    PromptActionParams,
    # Theme models
    ThemeActionParams,
    ThemeInfo,
    # Conversation models
    ConversationActionParams,
    ConversationInfo,
    # Response models
    ServerInfoResponse,
    ResourceInfoResponse,
    PromptInfoResponse,
    ToolInfoResponse,
)


def register_all_commands() -> None:
    """
    Register all built-in commands with the unified registry.

    This should be called once during application startup.
    """
    # Import all command implementations from grouped modules
    from mcp_cli.commands.core import (
        HelpCommand,
        ExitCommand,
        ClearCommand,
        VerboseCommand,
        InterruptCommand,
    )
    from mcp_cli.commands.tools import (
        ToolsCommand,
        ExecuteToolCommand,
        ToolHistoryCommand,
    )
    from mcp_cli.commands.servers import (
        ServersCommand,
        ServerSingularCommand,
        PingCommand,
        HealthCommand,
    )
    from mcp_cli.commands.providers import (
        ProviderCommand,
        ProviderSingularCommand,
        ModelCommand,
    )
    from mcp_cli.commands.resources import (
        ResourcesCommand,
        PromptsCommand,
    )
    from mcp_cli.commands.tokens import TokenCommand
    from mcp_cli.commands.theme import (
        ThemeSingularCommand,
        ThemesPluralCommand,
    )
    from mcp_cli.commands.conversation import ConversationCommand
    from mcp_cli.commands.usage import UsageCommand
    from mcp_cli.commands.export import ExportCommand
    from mcp_cli.commands.sessions import SessionsCommand, NewSessionCommand
    from mcp_cli.commands.apps import AppsCommand
    from mcp_cli.commands.memory import MemoryCommand
    from mcp_cli.commands.plan import PlanCommand
    from mcp_cli.commands.cmd import CmdCommand
    from mcp_cli.commands.attach import AttachCommand

    # Register basic commands
    registry.register(HelpCommand())
    registry.register(ExitCommand())
    registry.register(ClearCommand())
    registry.register(TokenCommand())

    # Register server commands (singular and plural)
    registry.register(ServerSingularCommand())  # /server <name> - show details
    registry.register(ServersCommand())  # /servers - list all

    registry.register(PingCommand())
    registry.register(HealthCommand())
    registry.register(ResourcesCommand())
    registry.register(PromptsCommand())

    # Register theme commands (singular and plural)
    registry.register(ThemeSingularCommand())  # /theme - show current
    registry.register(ThemesPluralCommand())  # /themes - list all
    # Note: Keep old ThemeCommand for backward compatibility if needed

    # Register provider commands (singular and plural)
    registry.register(ProviderSingularCommand())  # /provider - show current
    registry.register(ProviderCommand())  # /providers - list all

    # Register command groups
    registry.register(ToolsCommand())
    registry.register(ModelCommand())

    # Register chat-specific commands
    registry.register(ConversationCommand())
    registry.register(VerboseCommand())
    registry.register(InterruptCommand())
    registry.register(ToolHistoryCommand())
    registry.register(UsageCommand())
    registry.register(ExportCommand())
    registry.register(SessionsCommand())
    registry.register(NewSessionCommand())

    # Register tool execution command for interactive mode
    registry.register(ExecuteToolCommand())

    # Register MCP Apps command
    registry.register(AppsCommand())

    # Register VM visualization command (chat mode only)
    registry.register(MemoryCommand())

    # Register plan command
    registry.register(PlanCommand())

    # Register cmd command (CLI-only)
    registry.register(CmdCommand())
    # Register attach command (multi-modal file staging)
    registry.register(AttachCommand())

    # All commands have been migrated!
    # - tools (with subcommands: list, call, confirm)
    # - provider (with subcommands: list, set, show)
    # - model (with subcommands: list, set, show)
    # - resources
    # - prompts
    # - clear
    # - exit
    # - help
    # - theme
    # - ping
    # - conversation (chat mode only)
    # - verbose (chat mode only)


__all__ = [
    # Base classes
    "UnifiedCommand",
    "CommandGroup",
    "CommandMode",
    "CommandParameter",
    "CommandResult",
    # Registry
    "registry",
    "UnifiedCommandRegistry",
    "register_all_commands",
    # Decorators
    "validate_params",
    "handle_errors",
    # Exceptions
    "CommandError",
    "InvalidParameterError",
    "CommandExecutionError",
    "CommandNotFoundError",
    "ValidationError",
    # Server models
    "ServerActionParams",
    "ServerStatusInfo",
    "ServerPerformanceInfo",
    # Model models
    "ModelActionParams",
    "ModelInfo",
    # Provider models
    "ProviderActionParams",
    "ProviderInfo",
    # Token models
    "TokenListParams",
    "TokenSetParams",
    "TokenDeleteParams",
    "TokenClearParams",
    "TokenProviderParams",
    # Tool models
    "ToolActionParams",
    "ToolCallParams",
    # Resource models
    "ResourceActionParams",
    # Prompt models
    "PromptActionParams",
    # Theme models
    "ThemeActionParams",
    "ThemeInfo",
    # Conversation models
    "ConversationActionParams",
    "ConversationInfo",
    # Response models
    "ServerInfoResponse",
    "ResourceInfoResponse",
    "PromptInfoResponse",
    "ToolInfoResponse",
]
