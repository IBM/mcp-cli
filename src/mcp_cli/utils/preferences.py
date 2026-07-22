"""Centralized preference management for MCP CLI.

This module handles all user preferences including themes, provider settings,
model preferences, and other configuration options in a centralized way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from enum import Enum

from pydantic import BaseModel, Field

from mcp_cli.config.defaults import DEFAULT_CONFIG_DIR


class Theme(str, Enum):
    """Available UI themes from chuk-term."""

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    MINIMAL = "minimal"
    TERMINAL = "terminal"
    MONOKAI = "monokai"
    DRACULA = "dracula"
    SOLARIZED = "solarized"


class ConfirmationMode(str, Enum):
    """Global tool confirmation modes."""

    ALWAYS = "always"  # Always ask for confirmation
    NEVER = "never"  # Never ask for confirmation
    SMART = "smart"  # Smart mode based on risk level


class ToolRiskLevel(str, Enum):
    """Risk levels for tools."""

    SAFE = "safe"  # Read-only operations
    MODERATE = "moderate"  # Local modifications
    HIGH = "high"  # System-wide or destructive operations


class ToolPatternRule(BaseModel):
    """Pattern-based rule for tool confirmations - no dict goop!"""

    pattern: str = Field(description="Glob pattern for tool names")
    action: str = Field(description="Action: always/never or risk level")

    model_config = {"frozen": True}


class ToolConfirmationPreferences(BaseModel):
    """Tool confirmation preferences."""

    mode: ConfirmationMode = ConfirmationMode.SMART
    per_tool: dict[str, str] = Field(
        default_factory=dict, description="Per-tool overrides (always/never/ask)"
    )
    patterns: list[ToolPatternRule] = Field(
        default_factory=list, description="Pattern-based rules"
    )
    trusted_domains: list[str] = Field(
        default_factory=lambda: ["chukai.io"],
        description="Server domains that skip tool confirmation",
    )
    risk_thresholds: dict[ToolRiskLevel, bool] = Field(
        default_factory=lambda: {
            ToolRiskLevel.SAFE: False,
            ToolRiskLevel.MODERATE: False,
            ToolRiskLevel.HIGH: True,
        }
    )
    categories: dict[str, ToolRiskLevel] = Field(
        default_factory=lambda: {
            "read_*": ToolRiskLevel.SAFE,
            "list_*": ToolRiskLevel.SAFE,
            "get_*": ToolRiskLevel.SAFE,
            "describe_*": ToolRiskLevel.SAFE,
            "write_*": ToolRiskLevel.MODERATE,
            "create_*": ToolRiskLevel.MODERATE,
            "update_*": ToolRiskLevel.MODERATE,
            "delete_*": ToolRiskLevel.HIGH,
            "remove_*": ToolRiskLevel.HIGH,
            "execute_*": ToolRiskLevel.HIGH,
            "run_*": ToolRiskLevel.HIGH,
        }
    )


class UIPreferences(BaseModel):
    """UI-related preferences."""

    theme: Theme = Theme.DEFAULT
    verbose: bool = True
    confirm_tools: bool = True
    show_reasoning: bool = True
    tool_confirmation: ToolConfirmationPreferences = Field(
        default_factory=ToolConfirmationPreferences
    )


class CustomProvider(BaseModel):
    """Custom OpenAI-compatible provider configuration.

    API keys are stored in environment variables following the pattern:
    {PROVIDER_NAME_UPPER}_API_KEY

    For example, a provider named "myai" would use MYAI_API_KEY
    """

    name: str = Field(min_length=1, description="Provider name")
    api_base: str = Field(
        min_length=1, pattern="^https?://", description="API base URL"
    )
    default_model: str = Field(
        default="gpt-4", min_length=1, description="Default model"
    )
    models: list[str] = Field(
        default_factory=lambda: ["gpt-4", "gpt-3.5-turbo"],
        min_length=1,
        description="Available models",
    )
    env_var_name: str | None = None  # Optional custom env var name

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (no API key stored)."""
        return self.model_dump()  # type: ignore[no-any-return]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CustomProvider":
        """Create from dictionary."""
        return cls.model_validate(data)  # type: ignore[no-any-return]

    def get_env_var_name(self) -> str:
        """Get the environment variable name for this provider's API key."""
        if self.env_var_name:
            return self.env_var_name
        # Default pattern: PROVIDERNAME_API_KEY
        return f"{self.name.upper().replace('-', '_')}_API_KEY"


class ProviderPreferences(BaseModel):
    """Provider and model preferences."""

    active_provider: str | None = None
    active_model: str | None = None
    provider_settings: dict[str, Any] = Field(default_factory=dict)
    custom_providers: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Custom OpenAI-compatible providers"
    )


class ServerPreferences(BaseModel):
    """Server-related preferences."""

    disabled_servers: dict[str, bool] = Field(default_factory=dict)
    server_settings: dict[str, Any] = Field(default_factory=dict)
    runtime_servers: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="User-added servers"
    )


class MCPPreferences(BaseModel):
    """Complete MCP CLI preferences."""

    ui: UIPreferences = Field(default_factory=UIPreferences)
    provider: ProviderPreferences = Field(default_factory=ProviderPreferences)
    servers: ServerPreferences = Field(default_factory=ServerPreferences)
    last_servers: str | None = None
    config_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert preferences to dictionary."""
        return self.model_dump()  # type: ignore[no-any-return]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPPreferences":
        """Create preferences from dictionary."""
        # Pydantic will handle nested model validation automatically
        return cls.model_validate(data)  # type: ignore[no-any-return]


class PreferenceManager:
    """Manages MCP CLI preferences with file persistence."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize preference manager.

        Args:
            config_dir: Optional custom config directory, defaults to ~/.mcp-cli
        """
        self.config_dir = config_dir or Path(DEFAULT_CONFIG_DIR).expanduser()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.preferences_file = self.config_dir / "preferences.json"
        self.preferences = self.load_preferences()

    def load_preferences(self) -> MCPPreferences:
        """Load preferences from file or create defaults."""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, "r") as f:
                    data = json.load(f)
                    return MCPPreferences.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                # If preferences are corrupted or invalid, backup and create new
                backup_file = self.preferences_file.with_suffix(".json.backup")
                self.preferences_file.rename(backup_file)
                return MCPPreferences()
        return MCPPreferences()

    def save_preferences(self) -> None:
        """Save preferences to file."""
        with open(self.preferences_file, "w") as f:
            json.dump(self.preferences.to_dict(), f, indent=2)

    def get_theme(self) -> str:
        """Get current theme."""
        return self.preferences.ui.theme

    def set_theme(self, theme: str) -> None:
        """Set and persist theme.

        Args:
            theme: Theme name to set

        Raises:
            ValueError: If theme is not valid
        """
        try:
            theme_enum = Theme(theme)
        except ValueError:
            valid_themes = [t.value for t in Theme]
            raise ValueError(
                f"Invalid theme: {theme}. Valid themes are: {', '.join(valid_themes)}"
            )

        self.preferences.ui.theme = theme_enum
        self.save_preferences()

    def get_verbose(self) -> bool:
        """Get verbose setting."""
        return self.preferences.ui.verbose

    def set_verbose(self, verbose: bool) -> None:
        """Set verbose mode."""
        self.preferences.ui.verbose = verbose
        self.save_preferences()

    def get_confirm_tools(self) -> bool:
        """Get tool confirmation setting (legacy compatibility)."""
        return self.preferences.ui.tool_confirmation.mode != ConfirmationMode.NEVER

    def set_confirm_tools(self, confirm: bool) -> None:
        """Set tool confirmation mode (legacy compatibility)."""
        self.preferences.ui.confirm_tools = confirm
        self.preferences.ui.tool_confirmation.mode = (
            ConfirmationMode.SMART if confirm else ConfirmationMode.NEVER
        )
        self.save_preferences()

    def get_tool_confirmation_mode(self) -> ConfirmationMode:
        """Get the global tool confirmation mode."""
        return self.preferences.ui.tool_confirmation.mode

    def set_tool_confirmation_mode(self, mode: str) -> None:
        """Set the global tool confirmation mode.

        Args:
            mode: One of 'always', 'never', or 'smart'
        """
        try:
            confirmation_mode = ConfirmationMode(mode)
        except ValueError:
            raise ValueError(f"Invalid confirmation mode: {mode}")
        self.preferences.ui.tool_confirmation.mode = confirmation_mode
        self.preferences.ui.confirm_tools = confirmation_mode != ConfirmationMode.NEVER
        self.save_preferences()

    def get_tool_confirmation(self, tool_name: str) -> str | None:
        """Get confirmation setting for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            'always', 'never', 'ask', or None (use global default)
        """
        return self.preferences.ui.tool_confirmation.per_tool.get(tool_name)

    def set_tool_confirmation(self, tool_name: str, setting: str | None) -> None:
        """Set confirmation for a specific tool.

        Args:
            tool_name: Name of the tool
            setting: 'always', 'never', 'ask', or None (remove override)
        """
        if setting is None:
            self.preferences.ui.tool_confirmation.per_tool.pop(tool_name, None)
        else:
            if setting not in ["always", "never", "ask"]:
                raise ValueError(f"Invalid tool confirmation setting: {setting}")
            self.preferences.ui.tool_confirmation.per_tool[tool_name] = setting
        self.save_preferences()

    def get_all_tool_confirmations(self) -> dict[str, str]:
        """Get all per-tool confirmation settings."""
        return self.preferences.ui.tool_confirmation.per_tool.copy()

    def clear_tool_confirmations(self) -> None:
        """Clear all per-tool confirmation settings."""
        self.preferences.ui.tool_confirmation.per_tool.clear()
        self.save_preferences()

    def get_tool_risk_level(self, tool_name: str) -> ToolRiskLevel:
        """Determine the risk level of a tool based on patterns.

        Args:
            tool_name: Name of the tool

        Returns:
            Risk level enum value.
        """
        for pattern, risk in self.preferences.ui.tool_confirmation.categories.items():
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if tool_name.startswith(prefix):
                    return risk
            elif pattern.startswith("*"):
                suffix = pattern[1:]
                if tool_name.endswith(suffix):
                    return risk

        return ToolRiskLevel.MODERATE

    def should_confirm_tool(self, tool_name: str) -> bool:
        """Determine if a tool should be confirmed based on preferences.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool should be confirmed, False otherwise
        """
        # Check per-tool override first
        tool_setting = self.get_tool_confirmation(tool_name)
        if tool_setting == "always":
            return True
        elif tool_setting == "never":
            return False
        elif tool_setting == "ask":
            return True

        # Check global mode
        mode = self.preferences.ui.tool_confirmation.mode
        if mode == ConfirmationMode.ALWAYS:
            return True
        elif mode == ConfirmationMode.NEVER:
            return False
        elif mode == ConfirmationMode.SMART:
            risk_level = self.get_tool_risk_level(tool_name)
            return self.preferences.ui.tool_confirmation.risk_thresholds.get(
                risk_level, True
            )

        return True  # type: ignore[unreachable]  # safety fallback

    def is_trusted_domain(self, server_url: str | None) -> bool:
        """Check if a server URL belongs to a trusted domain.

        Args:
            server_url: The server URL to check (may be None for STDIO servers)

        Returns:
            True if the server's domain matches a trusted domain
        """
        if not server_url:
            return False

        from urllib.parse import urlparse

        try:
            parsed = urlparse(server_url)
            hostname = parsed.hostname or ""
        except Exception:
            return False

        trusted = self.preferences.ui.tool_confirmation.trusted_domains
        for domain in trusted:
            if hostname == domain or hostname.endswith(f".{domain}"):
                return True
        return False

    def get_trusted_domains(self) -> list[str]:
        """Get the list of trusted domains."""
        return list(self.preferences.ui.tool_confirmation.trusted_domains)

    def add_trusted_domain(self, domain: str) -> None:
        """Add a trusted domain.

        Args:
            domain: Domain to trust (e.g. 'example.com')
        """
        domains = self.preferences.ui.tool_confirmation.trusted_domains
        if domain not in domains:
            domains.append(domain)
            self.save_preferences()

    def remove_trusted_domain(self, domain: str) -> bool:
        """Remove a trusted domain.

        Args:
            domain: Domain to remove

        Returns:
            True if removed, False if not found
        """
        domains = self.preferences.ui.tool_confirmation.trusted_domains
        if domain in domains:
            domains.remove(domain)
            self.save_preferences()
            return True
        return False

    def add_tool_pattern(self, pattern: str, action: str) -> None:
        """Add a pattern-based rule for tool confirmations.

        Args:
            pattern: Glob pattern for tool names
            action: 'always', 'never', or risk level
        """
        rule = ToolPatternRule(pattern=pattern, action=action)
        self.preferences.ui.tool_confirmation.patterns.append(rule)
        self.save_preferences()

    def remove_tool_pattern(self, pattern: str) -> bool:
        """Remove a pattern-based rule.

        Args:
            pattern: The pattern to remove

        Returns:
            True if pattern was removed, False if not found
        """
        patterns = self.preferences.ui.tool_confirmation.patterns
        original_len = len(patterns)
        self.preferences.ui.tool_confirmation.patterns = [
            p for p in patterns if p.pattern != pattern
        ]
        if len(self.preferences.ui.tool_confirmation.patterns) < original_len:
            self.save_preferences()
            return True
        return False

    def set_risk_threshold(self, risk_level: str, should_confirm: bool) -> None:
        """Set whether to confirm tools at a specific risk level.

        Args:
            risk_level: 'safe', 'moderate', or 'high'
            should_confirm: Whether to confirm tools at this risk level
        """
        try:
            level = ToolRiskLevel(risk_level)
        except ValueError:
            raise ValueError(f"Invalid risk level: {risk_level}")
        self.preferences.ui.tool_confirmation.risk_thresholds[level] = should_confirm
        self.save_preferences()

    def get_active_provider(self) -> str | None:
        """Get active provider."""
        return self.preferences.provider.active_provider

    def set_active_provider(self, provider: str) -> None:
        """Set active provider."""
        self.preferences.provider.active_provider = provider
        self.save_preferences()

    def get_active_model(self) -> str | None:
        """Get active model."""
        return self.preferences.provider.active_model

    def set_active_model(self, model: str) -> None:
        """Set active model."""
        self.preferences.provider.active_model = model
        self.save_preferences()

    def get_last_servers(self) -> str | None:
        """Get last used servers."""
        return self.preferences.last_servers

    def set_last_servers(self, servers: str) -> None:
        """Set last used servers."""
        self.preferences.last_servers = servers
        self.save_preferences()

    def get_config_file(self) -> str | None:
        """Get default config file path."""
        return self.preferences.config_file

    def set_config_file(self, config_file: str) -> None:
        """Set default config file path."""
        self.preferences.config_file = config_file
        self.save_preferences()

    def reset_preferences(self) -> None:
        """Reset all preferences to defaults."""
        self.preferences = MCPPreferences()
        self.save_preferences()

    def get_history_file(self) -> Path:
        """Get path to chat history file."""
        return self.config_dir / "chat_history"

    def get_logs_dir(self) -> Path:
        """Get path to logs directory."""
        logs_dir = self.config_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    def is_server_disabled(self, server_name: str) -> bool:
        """Check if a server is disabled in preferences.

        Args:
            server_name: Name of the server to check

        Returns:
            True if server is disabled, False otherwise
        """
        return self.preferences.servers.disabled_servers.get(server_name, False)

    def set_server_disabled(self, server_name: str, disabled: bool = True) -> None:
        """Set server disabled state in preferences.

        Args:
            server_name: Name of the server
            disabled: Whether server should be disabled
        """
        if disabled:
            self.preferences.servers.disabled_servers[server_name] = True
        else:
            # Remove from disabled list if enabling
            self.preferences.servers.disabled_servers.pop(server_name, None)
        self.save_preferences()

    def enable_server(self, server_name: str) -> None:
        """Enable a server in preferences."""
        self.set_server_disabled(server_name, False)

    def disable_server(self, server_name: str) -> None:
        """Disable a server in preferences."""
        self.set_server_disabled(server_name, True)

    def get_disabled_servers(self) -> dict[str, bool]:
        """Get all disabled servers."""
        return self.preferences.servers.disabled_servers.copy()

    def clear_disabled_servers(self) -> None:
        """Clear all disabled server preferences."""
        self.preferences.servers.disabled_servers.clear()
        self.save_preferences()

    def add_runtime_server(self, name: str, config: dict[str, Any]) -> None:
        """Add a runtime server to preferences.

        Args:
            name: Server name
            config: Server configuration (command, args, env, transport, url, etc.)
        """
        self.preferences.servers.runtime_servers[name] = config
        self.save_preferences()

    def remove_runtime_server(self, name: str) -> bool:
        """Remove a runtime server from preferences.

        Args:
            name: Server name to remove

        Returns:
            True if server was removed, False if not found
        """
        if name in self.preferences.servers.runtime_servers:
            del self.preferences.servers.runtime_servers[name]
            self.save_preferences()
            return True
        return False

    def get_runtime_servers(self) -> dict[str, dict[str, Any]]:
        """Get all runtime servers."""
        return self.preferences.servers.runtime_servers.copy()

    def get_runtime_server(self, name: str) -> dict[str, Any] | None:
        """Get a specific runtime server configuration."""
        return self.preferences.servers.runtime_servers.get(name)

    def is_runtime_server(self, name: str) -> bool:
        """Check if a server is a runtime server."""
        return name in self.preferences.servers.runtime_servers

    def add_custom_provider(
        self,
        name: str,
        api_base: str,
        default_model: str = "gpt-4",
        models: list[str] | None = None,
        env_var_name: str | None = None,
    ) -> None:
        """Add a custom OpenAI-compatible provider.

        API keys should be set via environment variables.
        Default env var pattern: {PROVIDER_NAME}_API_KEY

        Args:
            name: Provider name (must be unique)
            api_base: API base URL (e.g., https://api.example.com)
            default_model: Default model to use
            models: List of available models
            env_var_name: Optional custom environment variable name for API key
        """
        if not models:
            models = ["gpt-4", "gpt-3.5-turbo"]

        provider = CustomProvider(
            name=name,
            api_base=api_base,
            default_model=default_model,
            models=models,
            env_var_name=env_var_name,
        )

        self.preferences.provider.custom_providers[name] = provider.to_dict()
        self.save_preferences()

    def remove_custom_provider(self, name: str) -> bool:
        """Remove a custom provider.

        Args:
            name: Provider name to remove

        Returns:
            True if provider was removed, False if not found
        """
        if name in self.preferences.provider.custom_providers:
            del self.preferences.provider.custom_providers[name]
            self.save_preferences()
            return True
        return False

    def get_custom_providers(self) -> dict[str, dict[str, Any]]:
        """Get all custom providers."""
        return self.preferences.provider.custom_providers.copy()

    def get_custom_provider(self, name: str) -> dict[str, Any] | None:
        """Get a specific custom provider configuration."""
        return self.preferences.provider.custom_providers.get(name)

    def is_custom_provider(self, name: str) -> bool:
        """Check if a provider is a custom provider."""
        return name in self.preferences.provider.custom_providers

    def update_custom_provider(
        self,
        name: str,
        api_base: str | None = None,
        default_model: str | None = None,
        models: list[str] | None = None,
        env_var_name: str | None = None,
    ) -> bool:
        """Update an existing custom provider.

        Args:
            name: Provider name to update
            api_base: New API base URL (optional)
            default_model: New default model (optional)
            models: New list of models (optional)
            env_var_name: New environment variable name (optional)

        Returns:
            True if provider was updated, False if not found
        """
        if name not in self.preferences.provider.custom_providers:
            return False

        provider_data = self.preferences.provider.custom_providers[name]

        if api_base is not None:
            provider_data["api_base"] = api_base
        if default_model is not None:
            provider_data["default_model"] = default_model
        if models is not None:
            provider_data["models"] = models
        if env_var_name is not None:
            provider_data["env_var_name"] = env_var_name

        self.save_preferences()
        return True

    def get_custom_provider_api_key(self, name: str) -> str | None:
        """Get the API key for a custom provider from environment variables.

        Args:
            name: Provider name

        Returns:
            API key from environment variable, or None if not set
        """
        import os

        provider_data = self.get_custom_provider(name)
        if not provider_data:
            return None

        # Parse provider data into model - no dict goop!
        provider = CustomProvider.from_dict(provider_data)
        env_var = provider.get_env_var_name()

        return os.environ.get(env_var)


# Global singleton instance
_preference_manager: PreferenceManager | None = None


__all__ = [
    "PreferenceManager",
    "get_preference_manager",
    "MCPPreferences",
    "UIPreferences",
    "ProviderPreferences",
    "ServerPreferences",
    "ToolConfirmationPreferences",
    "ToolPatternRule",
    "CustomProvider",
    "Theme",
    "ConfirmationMode",
    "ToolRiskLevel",
]


def get_preference_manager() -> PreferenceManager:
    """Get or create the global preference manager instance."""
    global _preference_manager
    if _preference_manager is None:
        _preference_manager = PreferenceManager()
    return _preference_manager
