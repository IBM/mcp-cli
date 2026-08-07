"""
Centralized configuration manager for MCP CLI.

This module provides a centralized way to manage configuration
instead of loading JSON files all over the place.

LEGACY: This module contains legacy MCPConfig that will be phased out.
Use mcp_cli.config.models.MCPConfig for new code.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mcp_cli.auth import OAuthConfig
from mcp_cli.config.defaults import DEFAULT_PROVIDER, DEFAULT_MODEL, DEFAULT_CONFIG_FILENAME
from mcp_cli.tools.models import ServerInfo, TransportType

# Import clean models from new config system
from mcp_cli.config.models import (
    MCPConfig as ModelsMCPConfig,
    TimeoutConfig as CleanTimeoutConfig,
    ToolConfig as CleanToolConfig,
)

logger = logging.getLogger(__name__)

# LEGACY: Use clean models for new code
# These are kept for backward compatibility with old code
TimeoutConfig = CleanTimeoutConfig
ToolConfig = CleanToolConfig


class ServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None  # For HTTP/SSE servers
    headers: dict[str, str] | None = None  # HTTP headers (e.g., Authorization)
    oauth: OAuthConfig | None = None  # OAuth configuration
    disabled: bool = False

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @property
    def transport(self) -> TransportType:
        """Determine transport type from config."""
        if self.url:
            return TransportType.HTTP
        elif self.command:
            return TransportType.STDIO
        else:
            # Return UNKNOWN when neither command nor url is provided
            return TransportType.UNKNOWN

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ServerConfig":
        """Create from dictionary format with environment variable handling.

        Uses ServerConfigInput Pydantic model for validation instead of manual .get() calls.
        """
        from mcp_cli.config.server_models import ServerConfigInput

        # Parse using Pydantic model (validates and provides defaults)
        input_model = ServerConfigInput.model_validate(data)

        # Get env and ensure PATH is inherited
        env = input_model.env.copy()
        if "PATH" not in env:
            env["PATH"] = os.environ.get("PATH", "")

        # Parse OAuth config if present
        oauth = None
        if input_model.oauth:
            if isinstance(input_model.oauth, dict):
                oauth = OAuthConfig.model_validate(input_model.oauth)
            else:
                oauth = input_model.oauth

        return cls(
            name=name,
            command=input_model.command,
            args=input_model.args,
            env=env,
            url=input_model.url,
            headers=input_model.headers,
            oauth=oauth,
            disabled=input_model.disabled,
        )

    def to_server_info(self, server_id: int = 0) -> ServerInfo:
        """Convert to ServerInfo model."""
        from mcp_cli.config.enums import ServerStatus

        return ServerInfo(
            id=server_id,
            name=self.name,
            status=ServerStatus.CONFIGURED.value,
            tool_count=0,
            namespace=self.name,
            enabled=not self.disabled,
            connected=False,
            transport=self.transport,
            capabilities={},
            command=self.command,
            args=self.args,
            env=self.env,
        )


class LegacyMCPConfig(BaseModel):
    """LEGACY: Complete MCP configuration with ServerConfig models.

    This class is kept for backward compatibility with code that uses
    ServerConfig models. New code should use CleanMCPConfig from models.py
    which stores servers as plain dicts.
    """

    servers: dict[str, ServerConfig] = Field(default_factory=dict)
    default_provider: str = DEFAULT_PROVIDER
    default_model: str = DEFAULT_MODEL
    theme: str = "default"
    verbose: bool = True
    confirm_tools: bool = True  # DEPRECATED: Use tools.confirm_tools instead

    # Centralized configurations (use clean models)
    timeouts: TimeoutConfig = Field(
        default_factory=TimeoutConfig, description="All timeout configurations"
    )
    tools: ToolConfig = Field(
        default_factory=ToolConfig, description="Tool filtering and behavior"
    )

    # Token storage configuration
    token_store_backend: str = (
        "auto"  # auto, keychain, windows, secretservice, vault, encrypted
    )
    token_store_password: str | None = None
    vault_url: str | None = None
    vault_token: str | None = None
    vault_mount_point: str = "secret"
    vault_path_prefix: str = "mcp-cli/oauth"
    vault_namespace: str | None = None

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @classmethod
    def load_from_file(cls, config_path: Path) -> "LegacyMCPConfig":
        """Load configuration from JSON file."""
        config = cls()

        if not config_path.exists():
            return config

        try:
            # Handle both regular files and package resources
            data_str = config_path.read_text()
            data = json.loads(data_str)

            # Load servers
            if "mcpServers" in data:
                for name, server_data in data["mcpServers"].items():
                    config.servers[name] = ServerConfig.from_dict(name, server_data)

            # Load other settings
            config.default_provider = data.get("defaultProvider", DEFAULT_PROVIDER)
            config.default_model = data.get("defaultModel", DEFAULT_MODEL)
            config.theme = data.get("theme", "default")
            config.verbose = data.get("verbose", True)
            config.confirm_tools = data.get("confirmTools", True)

            # Load token storage configuration
            token_storage = data.get("tokenStorage", {})
            config.token_store_backend = token_storage.get("backend", "auto")
            config.token_store_password = token_storage.get("password")
            config.vault_url = token_storage.get("vaultUrl")
            config.vault_token = token_storage.get("vaultToken")
            config.vault_mount_point = token_storage.get("vaultMountPoint", "secret")
            config.vault_path_prefix = token_storage.get(
                "vaultPathPrefix", "mcp-cli/oauth"
            )
            config.vault_namespace = token_storage.get("vaultNamespace")

            # Load timeout configuration (field names match clean TimeoutConfig)
            if "timeouts" in data:
                timeout_data = data["timeouts"]
                config.timeouts = TimeoutConfig(
                    streaming_chunk=timeout_data.get(
                        "streamingChunkTimeout",
                        config.timeouts.streaming_chunk,
                    ),
                    streaming_global=timeout_data.get(
                        "streamingGlobalTimeout",
                        config.timeouts.streaming_global,
                    ),
                    streaming_first_chunk=timeout_data.get(
                        "streamingFirstChunkTimeout",
                        config.timeouts.streaming_first_chunk,
                    ),
                    tool_execution=timeout_data.get(
                        "toolExecutionTimeout", config.timeouts.tool_execution
                    ),
                    server_init=timeout_data.get(
                        "serverInitTimeout", config.timeouts.server_init
                    ),
                    http_request=timeout_data.get(
                        "httpRequestTimeout", config.timeouts.http_request
                    ),
                    http_connect=timeout_data.get(
                        "httpConnectTimeout", config.timeouts.http_connect
                    ),
                )

            # NEW: Load tool configuration
            if "tools" in data:
                tool_data = data["tools"]
                config.tools = ToolConfig(
                    include_tools=tool_data.get("includeTools"),
                    exclude_tools=tool_data.get("excludeTools"),
                    dynamic_tools_enabled=tool_data.get(
                        "dynamicToolsEnabled", config.tools.dynamic_tools_enabled
                    ),
                    confirm_tools=tool_data.get(
                        "confirmTools", config.tools.confirm_tools
                    ),
                    max_concurrency=tool_data.get(
                        "maxConcurrency", config.tools.max_concurrency
                    ),
                )

        except Exception as e:
            # Log error but return empty config
            print(f"Error loading config: {e}")

        return config

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        data = {
            "mcpServers": {
                name: server.model_dump(exclude_none=True, exclude_defaults=True)
                for name, server in self.servers.items()
            },
            "defaultProvider": self.default_provider,
            "defaultModel": self.default_model,
            "theme": self.theme,
            "verbose": self.verbose,
            "confirmTools": self.confirm_tools,
        }

        # Add token storage configuration if non-default
        token_storage: dict[str, Any] = {}
        if self.token_store_backend != "auto":
            token_storage["backend"] = self.token_store_backend
        if self.token_store_password:
            token_storage["password"] = self.token_store_password
        if self.vault_url:
            token_storage["vaultUrl"] = self.vault_url
        if self.vault_token:
            token_storage["vaultToken"] = self.vault_token
        if self.vault_mount_point != "secret":
            token_storage["vaultMountPoint"] = self.vault_mount_point
        if self.vault_path_prefix != "mcp-cli/oauth":
            token_storage["vaultPathPrefix"] = self.vault_path_prefix
        if self.vault_namespace:
            token_storage["vaultNamespace"] = self.vault_namespace

        if token_storage:
            data["tokenStorage"] = token_storage

        # NEW: Add timeout configuration
        data["timeouts"] = {
            "streamingChunkTimeout": self.timeouts.streaming_chunk,
            "streamingGlobalTimeout": self.timeouts.streaming_global,
            "streamingFirstChunkTimeout": self.timeouts.streaming_first_chunk,
            "toolExecutionTimeout": self.timeouts.tool_execution,
            "serverInitTimeout": self.timeouts.server_init,
            "httpRequestTimeout": self.timeouts.http_request,
            "httpConnectTimeout": self.timeouts.http_connect,
        }

        # NEW: Add tool configuration
        data["tools"] = {
            "includeTools": self.tools.include_tools,
            "excludeTools": self.tools.exclude_tools,
            "dynamicToolsEnabled": self.tools.dynamic_tools_enabled,
            "confirmTools": self.tools.confirm_tools,
            "maxConcurrency": self.tools.max_concurrency,
        }

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_server(self, name: str) -> ServerConfig | None:
        """Get a server configuration by name."""
        return self.servers.get(name)

    def add_server(self, server: ServerConfig) -> None:
        """Add or update a server configuration."""
        self.servers[server.name] = server

    def remove_server(self, name: str) -> bool:
        """Remove a server configuration."""
        if name in self.servers:
            del self.servers[name]
            return True
        return False

    def list_servers(self) -> list[ServerConfig]:
        """Get list of all server configurations."""
        return list(self.servers.values())

    def list_enabled_servers(self) -> list[ServerConfig]:
        """Get list of enabled server configurations."""
        return [s for s in self.servers.values() if not s.disabled]


# Export alias for backward compatibility
# ConfigManager still uses LegacyMCPConfig internally
MCPConfig = LegacyMCPConfig


class ConfigManager:
    """
    Manager for application configuration.

    This provides a singleton-like pattern for managing configuration.
    """

    _instance: ConfigManager | None = None
    _config: MCPConfig | None = None
    _config_path: Path | None = None

    def __new__(cls) -> ConfigManager:
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, config_path: Path | None = None) -> MCPConfig:
        """
        Initialize or get the configuration.

        Priority order:
        1. Explicit config_path if provided
        2. server_config.json in current directory
        """
        if self._config is None:
            if config_path:
                self._config_path = Path(config_path)
            else:
                self._config_path = Path(DEFAULT_CONFIG_FILENAME)

            self._config = MCPConfig.load_from_file(self._config_path)
        return self._config

    def get_config(self) -> MCPConfig:
        """
        Get the current configuration.

        Raises:
            RuntimeError: If config hasn't been initialized
        """
        if self._config is None:
            raise RuntimeError("Config not initialized. Call initialize() first.")
        return self._config

    def save(self) -> None:
        """Save the current configuration to file."""
        if self._config and self._config_path:
            self._config.save_to_file(self._config_path)

    def reload(self) -> MCPConfig:
        """Reload configuration from file."""
        if self._config_path:
            self._config = MCPConfig.load_from_file(self._config_path)
            return self._config
        raise RuntimeError("No config path set")

    def reset(self) -> None:
        """Reset the configuration (useful for testing)."""
        self._config = None
        self._config_path = None


def get_config() -> MCPConfig:
    """
    Convenience function to get the current configuration.

    Returns:
        The current MCPConfig

    Raises:
        RuntimeError: If config hasn't been initialized
    """
    manager = ConfigManager()
    return manager.get_config()


def initialize_config(config_path: Path | None = None) -> MCPConfig:
    """
    Convenience function to initialize the configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        The initialized MCPConfig
    """
    manager = ConfigManager()
    return manager.initialize(config_path)


def detect_server_types(
    cfg: MCPConfig | ModelsMCPConfig, servers: list[str]
) -> tuple[list[dict], list[str]]:
    """
    Detect which servers are HTTP vs STDIO based on configuration.

    Args:
        cfg: MCPConfig instance
        servers: List of server names to detect

    Returns:
        Tuple of (http_servers_list, stdio_servers_list)
    """
    http_servers = []
    stdio_servers = []

    if not cfg or not cfg.servers:
        # No config, assume all are STDIO
        return [], servers

    for server in servers:
        server_config = cfg.servers.get(server)

        if not server_config:
            logger.warning(f"Server '{server}' not found in configuration")
            stdio_servers.append(server)
            continue

        # Handle both dict (new clean config) and ServerConfig (legacy)
        sc: Any = server_config
        if isinstance(sc, dict):
            url = sc.get("url")
            command = sc.get("command")
        else:
            # ServerConfig model
            url = sc.url
            command = sc.command

        if url:
            # HTTP server
            http_servers.append({"name": server, "url": url})
            logger.debug(f"Detected HTTP server: {server} -> {url}")
        elif command:
            # STDIO server
            stdio_servers.append(server)
            logger.debug(f"Detected STDIO server: {server}")
        else:
            logger.warning(
                f"Server '{server}' has unclear configuration, assuming STDIO"
            )
            stdio_servers.append(server)

    return http_servers, stdio_servers


def validate_server_config(
    cfg: MCPConfig | ModelsMCPConfig, servers: list[str]
) -> tuple[bool, list[str]]:
    """
    Validate server configuration and return status and errors.

    Args:
        cfg: MCPConfig instance
        servers: List of server names to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not cfg or not cfg.servers:
        errors.append("No servers found in configuration")
        return False, errors

    for server in servers:
        if server not in cfg.servers:
            errors.append(f"Server '{server}' not found in configuration")
            continue

        server_config = cfg.servers[server]

        # Handle both dict (new clean config) and ServerConfig (legacy)
        sc: Any = server_config
        if isinstance(sc, dict):
            has_url = sc.get("url") is not None
            has_command = sc.get("command") is not None
            url = sc.get("url")
            command = sc.get("command")
        else:
            # ServerConfig model
            has_url = server_config.url is not None
            has_command = server_config.command is not None
            url = server_config.url
            command = server_config.command

        # Check for valid configuration
        if not has_url and not has_command:
            errors.append(f"Server '{server}' missing both 'url' and 'command' fields")
        elif has_url and has_command:
            errors.append(
                f"Server '{server}' has both 'url' and 'command' fields (should have only one)"
            )
        elif has_url:
            # Validate URL format
            if url and not url.startswith(("http://", "https://")):
                errors.append(
                    f"Server '{server}' URL must start with http:// or https://"
                )
        elif has_command:
            # Validate command format
            if not isinstance(command, str) or not command.strip():
                errors.append(f"Server '{server}' command must be a non-empty string")

    return len(errors) == 0, errors


# ============================================================================
# Runtime Configuration Resolver
# ============================================================================


class RuntimeConfig:
    """
    Runtime configuration resolver with priority handling.

    Priority order (highest to lowest):
    1. CLI arguments (passed at initialization)
    2. Environment variables
    3. Config file (MCPConfig)
    4. Defaults (in TimeoutConfig/ToolConfig)

    This class provides a unified interface for accessing configuration
    values, automatically resolving from the appropriate source.
    """

    def __init__(
        self,
        mcp_config: MCPConfig | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ):
        """
        Initialize runtime config resolver.

        Args:
            mcp_config: The loaded MCPConfig (from file)
            cli_overrides: Dictionary of CLI argument overrides
        """
        self.mcp_config = mcp_config or MCPConfig()
        self.cli_overrides = cli_overrides or {}

    def get_timeout(self, timeout_name: str) -> float:
        """
        Get timeout value with priority resolution.

        Args:
            timeout_name: Name of timeout (e.g., "streaming_chunk", "tool_execution")

        Returns:
            Resolved timeout value in seconds

        Example:
            >>> config = RuntimeConfig(mcp_config)
            >>> config.get_timeout("streaming_chunk")  # Returns 45.0 by default
        """
        # 1. Check CLI overrides first
        cli_key = f"{timeout_name}_timeout"
        if cli_key in self.cli_overrides:
            return float(self.cli_overrides[cli_key])

        # 2. Check environment variables
        env_key = f"MCP_{timeout_name.upper()}_TIMEOUT"
        env_value = os.getenv(env_key)
        if env_value:
            try:
                return float(env_value)
            except ValueError:
                logger.warning(
                    f"Invalid timeout value in {env_key}={env_value}, using config/default"
                )

        # Special case: MCP_TOOL_TIMEOUT applies to multiple timeouts
        if timeout_name in ["streaming_chunk", "streaming_global", "tool_execution"]:
            tool_timeout_env = os.getenv("MCP_TOOL_TIMEOUT")
            if tool_timeout_env:
                try:
                    return float(tool_timeout_env)
                except ValueError:
                    pass

        # 3. Get from config file
        timeout_attr = f"{timeout_name}_timeout"
        if hasattr(self.mcp_config.timeouts, timeout_attr):
            return getattr(self.mcp_config.timeouts, timeout_attr)  # type: ignore[no-any-return]

        # 4. Fallback to default (should never reach here if TimeoutConfig has defaults)
        logger.warning(
            f"No timeout configuration found for '{timeout_name}', using 120.0"
        )
        return 120.0

    def get_tool_config_value(self, key: str) -> Any:
        """
        Get tool configuration value with priority resolution.

        Args:
            key: Configuration key (e.g., "include_tools", "confirm_tools")

        Returns:
            Resolved configuration value
        """
        # 1. Check CLI overrides
        if key in self.cli_overrides:
            return self.cli_overrides[key]

        # 2. Check environment variables
        env_key = f"MCP_CLI_{key.upper()}"
        env_value = os.getenv(env_key)

        if env_value is not None:
            # Handle different types
            if key in ["include_tools", "exclude_tools"]:
                # Comma-separated list
                return [t.strip() for t in env_value.split(",") if t.strip()]
            elif key == "dynamic_tools_enabled":
                return env_value.lower() in ["1", "true", "yes"]
            elif key == "confirm_tools":
                return env_value.lower() not in ["0", "false", "no"]
            elif key == "max_concurrency":
                try:
                    return int(env_value)
                except ValueError:
                    pass

        # 3. Get from config file
        if hasattr(self.mcp_config.tools, key):
            return getattr(self.mcp_config.tools, key)

        # 4. Return None if not found
        return None

    def get_all_timeouts(self) -> dict[str, float]:
        """Get all timeout values as a dictionary."""
        return {
            "streaming_chunk": self.get_timeout("streaming_chunk"),
            "streaming_global": self.get_timeout("streaming_global"),
            "streaming_first_chunk": self.get_timeout("streaming_first_chunk"),
            "tool_execution": self.get_timeout("tool_execution"),
            "server_init": self.get_timeout("server_init"),
            "http_request": self.get_timeout("http_request"),
            "http_connect": self.get_timeout("http_connect"),
        }

    def update_from_cli(self, **kwargs) -> None:
        """
        Update CLI overrides from keyword arguments.

        Args:
            **kwargs: CLI argument values to override
        """
        self.cli_overrides.update(kwargs)


def get_runtime_config(
    mcp_config: MCPConfig | None = None, cli_overrides: dict[str, Any] | None = None
) -> RuntimeConfig:
    """
    Convenience function to create a RuntimeConfig instance.

    Args:
        mcp_config: The loaded MCPConfig (defaults to getting from ConfigManager)
        cli_overrides: Dictionary of CLI argument overrides

    Returns:
        RuntimeConfig instance
    """
    if mcp_config is None:
        try:
            mcp_config = get_config()
        except RuntimeError:
            # Config not initialized, use defaults
            mcp_config = MCPConfig()

    return RuntimeConfig(mcp_config, cli_overrides)
