# src/mcp_cli/commands/definitions/provider.py
"""
Unified provider command implementation.
Uses the existing enhanced provider commands from mcp_cli.commands.provider
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandGroup,
    CommandParameter,
    CommandResult,
)

if TYPE_CHECKING:
    from mcp_cli.commands.models.provider import ProviderStatus

logger = logging.getLogger(__name__)


class ProviderCommand(CommandGroup):
    """Provider command group."""

    def __init__(self):
        super().__init__()
        # Add subcommands
        self.add_subcommand(ProviderListCommand())
        self.add_subcommand(ProviderSetCommand())
        self.add_subcommand(ProviderShowCommand())

    @property
    def name(self) -> str:
        return "providers"

    @property
    def aliases(self) -> list[str]:
        return []  # Remove provider alias - it's now its own command

    @property
    def description(self) -> str:
        return "List available LLM providers"

    @property
    def help_text(self) -> str:
        return """
Manage LLM providers for the MCP CLI.

Subcommands:
  list    - List all available providers
  custom  - List custom OpenAI-compatible providers
  add     - Add a custom provider
  remove  - Remove a custom provider
  set     - Configure provider settings
  show    - Show current provider status

Custom Provider Management:
  /provider add <name> <api_base> [models...]
    Add a custom OpenAI-compatible provider (LocalAI, proxies, etc.)
    
  /provider remove <name>
    Remove a custom provider
    
  /provider custom
    List all custom providers

Usage:
  /provider              - Show current provider status
  /providers             - List all providers (preferred)
  /provider <name>       - Switch to a different provider
  /provider list         - List all providers
  
Examples:
  # Switch providers
  /provider ollama       - Switch to Ollama provider
  /provider openai       - Switch to OpenAI provider
  
  # Add custom providers
  /provider add localai http://localhost:8080/v1 gpt-4 gpt-3.5-turbo
  /provider add myproxy https://proxy.example.com/v1 custom-model
  
  # Use custom provider (after setting API key)
  export LOCALAI_API_KEY=your-key
  /provider localai
  
  # Remove custom provider
  /provider remove localai

Note: API keys are NEVER stored in config. Use environment variables:
  Pattern: {PROVIDER_NAME}_API_KEY
  Example: LOCALAI_API_KEY, MYPROXY_API_KEY
"""

    async def execute(self, subcommand: str | None = None, **kwargs) -> CommandResult:
        """Execute the provider command - handle direct provider switching."""
        from mcp_cli.context import get_context
        from chuk_term.ui import output

        args = kwargs.get("args", [])

        # No arguments - delegate to list subcommand to show providers with status
        if not args:
            list_cmd = self.subcommands.get("list")
            if list_cmd:
                return await list_cmd.execute(**kwargs)
            return CommandResult(success=False, error="List subcommand not available")

        first_arg = args[0] if isinstance(args, list) else str(args)

        # Known subcommands - let parent class handle routing
        if first_arg.lower() in [
            "list",
            "ls",
            "set",
            "use",
            "switch",
            "show",
            "current",
            "status",
        ]:
            return await super().execute(**kwargs)

        # Otherwise, treat it as a provider name to switch to
        try:
            context = get_context()
            if not context or not context.model_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            context.model_manager.switch_provider(first_arg)
            output.success(f"Switched to provider: {first_arg}")

            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(
                success=False, error=f"Failed to switch provider: {str(e)}"
            )


class ProviderListCommand(UnifiedCommand):
    """List available providers."""

    @property
    def name(self) -> str:
        return "list"

    @property
    def aliases(self) -> list[str]:
        return ["ls"]

    @property
    def description(self) -> str:
        return "List all available LLM providers"

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="detailed",
                type=bool,
                default=False,
                help="Show detailed provider information",
                is_flag=True,
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the provider list command."""
        from mcp_cli.context import get_context
        from chuk_term.ui import output, format_table
        from mcp_cli.commands.models.provider import ProviderData

        try:
            context = get_context()
            if not context or not context.model_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            current_provider = context.model_manager.get_active_provider()

            # Get rich provider info from chuk_llm
            providers: list[ProviderData] = []
            try:
                from chuk_llm.llm.client import list_available_providers

                providers_dict = list_available_providers()
                for name, info in providers_dict.items():
                    if isinstance(info, dict) and "error" not in info:
                        providers.append(
                            ProviderData(
                                name=name,
                                has_api_key=info.get("has_api_key", False),
                                models=info.get("models", []),
                                available_models=info.get("available_models", []),
                                default_model=info.get("default_model"),
                            )
                        )
            except Exception as e:
                logger.debug("Failed to list providers: %s", e)

            # Build table data with status info
            table_data = []
            for provider in providers:
                is_current = "✓" if provider.name == current_provider else ""
                status = await self._get_provider_status(provider)

                # Get default model for display
                default_model = provider.default_model or "-"
                if default_model and len(default_model) > 25:
                    default_model = default_model[:22] + "..."

                table_data.append(
                    {
                        "": is_current,
                        "Provider": provider.name,
                        "Default Model": default_model,
                        "Status": f"{status.icon} {status.text}",
                    }
                )

            # Display table
            table = format_table(
                table_data,
                title=f"{len(table_data)} Available Providers",
                columns=["", "Provider", "Default Model", "Status"],
            )
            output.print_table(table)

            return CommandResult(success=True, data={"command": "provider list"})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list providers: {str(e)}",
            )

    async def _get_provider_status(self, provider) -> "ProviderStatus":
        """Get the status for a provider."""
        import asyncio

        from mcp_cli.commands.models.provider import ProviderStatus
        from mcp_cli.config import PROVIDER_OLLAMA

        # Special handling for Ollama (no API key needed)
        if provider.name.lower() == PROVIDER_OLLAMA:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ollama",
                    "list",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
                if proc.returncode == 0:
                    lines = stdout.decode().strip().split("\n")
                    model_count = len([line for line in lines[1:] if line.strip()])
                    return ProviderStatus(
                        icon="✅",
                        text=f"Running ({model_count} models)",
                        reason="Ollama server is responding",
                    )
                return ProviderStatus(
                    icon="❌", text="Not running", reason="Ollama server not responding"
                )
            except Exception as e:
                logger.debug("Ollama status check failed: %s", e)
                return ProviderStatus(
                    icon="❌", text="Not available", reason="Ollama not installed"
                )

        # Standard providers
        if provider.has_api_key:
            model_count = provider.model_count
            if model_count > 0:
                return ProviderStatus(
                    icon="✅",
                    text=f"Configured ({model_count} models)",
                    reason="API key set and models available",
                )
            return ProviderStatus(
                icon="⚠️",
                text="API key set",
                reason="No models discovered yet",
            )

        return ProviderStatus(
            icon="❌",
            text="No API key",
            reason=f"Set {provider.name.upper()}_API_KEY environment variable",
        )


class ProviderSetCommand(UnifiedCommand):
    """Set the active provider."""

    @property
    def name(self) -> str:
        return "set"

    @property
    def aliases(self) -> list[str]:
        return ["use", "switch"]

    @property
    def description(self) -> str:
        return "Set the active LLM provider"

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="provider_name",
                type=str,
                required=True,
                help="Name of the provider to set",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the provider set command."""
        from mcp_cli.context import get_context
        from chuk_term.ui import output

        # Get provider name
        provider_name = kwargs.get("provider_name")
        if not provider_name and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list):
                provider_name = args_val[0] if args_val else None
            elif isinstance(args_val, str):
                provider_name = args_val

        if not provider_name:
            return CommandResult(
                success=False,
                error="Provider name is required. Usage: /provider set <name>",
            )

        try:
            context = get_context()
            if not context or not context.model_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            context.model_manager.switch_provider(provider_name)
            output.success(f"Switched to provider: {provider_name}")

            return CommandResult(success=True, data={"provider": provider_name})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to set provider: {str(e)}",
            )


class ProviderShowCommand(UnifiedCommand):
    """Show current provider."""

    @property
    def name(self) -> str:
        return "show"

    @property
    def aliases(self) -> list[str]:
        return ["current", "status"]

    @property
    def description(self) -> str:
        return "Show the current active provider"

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the provider show command."""
        from mcp_cli.context import get_context
        from chuk_term.ui import output

        try:
            context = get_context()
            if not context or not context.model_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            current_provider = context.model_manager.get_active_provider()
            current_model = context.model_manager.get_active_model()

            output.panel(
                f"Provider: {current_provider}\nModel: {current_model}",
                title="Current Provider",
            )

            return CommandResult(success=True, data={"command": "provider show"})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to get provider info: {str(e)}",
            )
