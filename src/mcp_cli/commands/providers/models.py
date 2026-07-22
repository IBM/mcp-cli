# src/mcp_cli/commands/definitions/model.py
"""
Unified model command implementation.
Uses the existing enhanced model commands from mcp_cli.commands.model
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
from mcp_cli.config.defaults import DEFAULT_PROVIDER_DISCOVERY_TIMEOUT

if TYPE_CHECKING:
    from mcp_cli.commands.models.model import ModelInfo

logger = logging.getLogger(__name__)


class ModelCommand(CommandGroup):
    """Model command group."""

    def __init__(self):
        super().__init__()
        # Add subcommands
        self.add_subcommand(ModelListCommand())
        self.add_subcommand(ModelSetCommand())
        self.add_subcommand(ModelShowCommand())

    @property
    def name(self) -> str:
        return "models"

    @property
    def aliases(self) -> list[str]:
        return ["model"]

    @property
    def description(self) -> str:
        return "Manage LLM models"

    @property
    def help_text(self) -> str:
        return """
Manage LLM models for the current provider.

Subcommands:
  list  - List available models
  set   - Set the active model
  show  - Show current model

Usage:
  /model               - Show current model and available models
  /models              - List all models (preferred)
  /model <name>        - Switch to a different model
  /model list          - List all models (alternative)
  /model refresh       - Refresh model discovery
  
Examples:
  /model gpt-4o-mini   - Switch to gpt-4o-mini
  /model set gpt-4     - Explicitly set to gpt-4
  /model show          - Show current model
  /model list          - List all available models
"""

    async def execute(self, subcommand: str | None = None, **kwargs) -> CommandResult:
        """Execute the model command - handle direct model switching."""
        from mcp_cli.context import get_context
        from chuk_term.ui import output

        args = kwargs.get("args", [])

        # No arguments - delegate to list subcommand to show all models
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

        # Otherwise, treat it as a model name to switch to
        try:
            context = get_context()
            if not context or not context.model_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            current_provider = context.model_manager.get_active_provider()
            context.model_manager.switch_model(current_provider, first_arg)
            output.success(f"Switched to model: {first_arg}")

            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(
                success=False, error=f"Failed to switch model: {str(e)}"
            )


class ModelListCommand(UnifiedCommand):
    """List available models."""

    @property
    def name(self) -> str:
        return "list"

    @property
    def aliases(self) -> list[str]:
        return ["ls"]

    @property
    def description(self) -> str:
        return "List available models for the current provider"

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="provider",
                type=str,
                required=False,
                help="Provider to list models for (uses current if not specified)",
            ),
            CommandParameter(
                name="detailed",
                type=bool,
                default=False,
                help="Show detailed model information",
                is_flag=True,
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the model list command."""
        from mcp_cli.context import get_context
        from chuk_term.ui import output, format_table

        try:
            context = get_context()
            if not context or not context.model_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            current_provider = context.model_manager.get_active_provider()
            current_model = context.model_manager.get_active_model()

            # Discover models for the current provider
            model_infos = await self._discover_models(current_provider, current_model)

            if not model_infos:
                output.warning(
                    f"No models discovered for {current_provider}. "
                    "Check API key configuration."
                )
                return CommandResult(success=True, data={"command": "model list"})

            # Build table data from Pydantic models
            table_data = []
            for model_info in model_infos:
                table_data.append(
                    {
                        "": "✓" if model_info.is_current else "",
                        "Model": model_info.name,
                    }
                )

            # Display table
            table = format_table(
                table_data,
                title=f"{len(model_infos)} Models for {current_provider}",
                columns=["", "Model"],
            )
            output.print_table(table)

            return CommandResult(success=True, data={"command": "model list"})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list models: {str(e)}",
            )

    async def _discover_models(
        self, provider: str, current_model: str
    ) -> list["ModelInfo"]:
        """Discover available models for a provider."""
        from mcp_cli.commands.models.model import ModelInfo
        from mcp_cli.config import PROVIDER_OLLAMA

        # For Ollama, get actual running models from CLI
        if provider.lower() == PROVIDER_OLLAMA:
            model_names = await self._get_ollama_models()
        else:
            # Get models from chuk_llm (already filters out placeholders)
            model_names = await self._get_provider_models(provider)

        # Convert to Pydantic ModelInfo objects
        return [
            ModelInfo(
                name=name,
                provider=provider,
                is_current=(name == current_model),
            )
            for name in model_names
        ]

    async def _get_ollama_models(self) -> list[str]:
        """Get models from Ollama CLI."""
        import asyncio

        try:
            proc = await asyncio.create_subprocess_exec(
                "ollama",
                "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=DEFAULT_PROVIDER_DISCOVERY_TIMEOUT
            )
            if proc.returncode == 0:
                lines = stdout.decode().strip().split("\n")
                models = []
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                return models
        except Exception as e:
            logger.debug("Ollama model discovery failed: %s", e)
        return []

    async def _get_provider_models(self, provider: str) -> list[str]:
        """Get models from chuk_llm for a provider.

        Tries multiple strategies:
        1. Get from chuk_llm's cached provider info
        2. If only "*" placeholder, call the provider's /models API endpoint
        3. Fall back to default_model if available
        """
        try:
            from chuk_llm.llm.client import list_available_providers

            providers_info = list_available_providers()
            provider_info = providers_info.get(provider, {})

            if isinstance(provider_info, dict):
                models = provider_info.get(
                    "models", provider_info.get("available_models", [])
                )
                model_list = list(models) if models else []

                # Filter out placeholder "*" values
                model_list = [m for m in model_list if m and m != "*"]

                # If only placeholder models, try calling the API
                if not model_list and provider_info.get("has_api_key"):
                    api_base = provider_info.get("api_base")
                    if api_base:
                        model_list = await self._fetch_models_from_api(
                            provider, api_base
                        )

                # Fall back to default_model if still empty
                if not model_list:
                    default_model = provider_info.get("default_model")
                    if default_model:
                        model_list = [default_model]

                return model_list
        except Exception as e:
            logger.debug("Provider model discovery failed for %s: %s", provider, e)
        return []

    async def _fetch_models_from_api(self, provider: str, api_base: str) -> list[str]:
        """Fetch models from provider's /models API endpoint.

        Works for OpenAI-compatible APIs (deepseek, openai, etc.)
        """
        import os

        try:
            import httpx

            # Get API key from environment
            api_key = os.environ.get(f"{provider.upper()}_API_KEY")
            if not api_key:
                return []

            # Ensure api_base ends properly for /models endpoint
            models_url = f"{api_base.rstrip('/')}/models"

            async with httpx.AsyncClient(
                timeout=DEFAULT_PROVIDER_DISCOVERY_TIMEOUT
            ) as client:
                resp = await client.get(
                    models_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                )

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "data" in data:
                    # OpenAI-compatible format: {"data": [{"id": "model-name", ...}]}
                    return [m.get("id") for m in data["data"] if m.get("id")]

        except Exception as e:
            logger.debug("API model fetch failed for %s: %s", provider, e)
        return []


class ModelSetCommand(UnifiedCommand):
    """Set the active model."""

    @property
    def name(self) -> str:
        return "set"

    @property
    def aliases(self) -> list[str]:
        return ["use", "switch"]

    @property
    def description(self) -> str:
        return "Set the active model"

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="model_name",
                type=str,
                required=True,
                help="Name of the model to set",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the model set command."""
        from mcp_cli.context import get_context
        from chuk_term.ui import output

        # Get model name
        model_name = kwargs.get("model_name")
        if not model_name and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list):
                model_name = args_val[0] if args_val else None
            elif isinstance(args_val, str):
                model_name = args_val

        if not model_name:
            return CommandResult(
                success=False,
                error="Model name is required. Usage: /model set <name>",
            )

        try:
            context = get_context()
            if not context or not context.model_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            current_provider = context.model_manager.get_active_provider()
            context.model_manager.switch_model(current_provider, model_name)
            output.success(f"Switched to model: {model_name}")

            return CommandResult(success=True, data={"model": model_name})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to set model: {str(e)}",
            )


class ModelShowCommand(UnifiedCommand):
    """Show current model."""

    @property
    def name(self) -> str:
        return "show"

    @property
    def aliases(self) -> list[str]:
        return ["current", "status"]

    @property
    def description(self) -> str:
        return "Show the current active model"

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the model show command."""
        from mcp_cli.context import get_context
        from chuk_term.ui import output

        try:
            context = get_context()
            if not context or not context.model_manager:
                return CommandResult(success=False, error="No LLM manager available.")

            current_model = context.model_manager.get_active_model()
            current_provider = context.model_manager.get_active_provider()

            output.panel(
                f"Provider: {current_provider}\nModel: {current_model}",
                title="Current Model",
            )

            return CommandResult(success=True, data={"command": "model show"})

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to get model info: {str(e)}",
            )
