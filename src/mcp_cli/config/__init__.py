"""
Configuration management for MCP CLI.

Clean, async-native, Pydantic-based configuration system.
All constants, enums, and configuration are consolidated here.
"""

from importlib.metadata import PackageNotFoundError, version

# Enums (all type-safe enums)
from mcp_cli.config.enums import (
    ConfigSource,
    ConversationAction,
    OutputFormat,
    ServerAction,
    ServerStatus,
    ThemeAction,
    TimeoutType,
    TokenAction,
    TokenBackend,
    TokenNamespace,
    ToolAction,
)

# Environment variables
from mcp_cli.config.env_vars import (
    EnvVar,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_list,
    is_set,
    set_env,
    unset_env,
)

# Defaults and constants
from mcp_cli.config.defaults import (
    # Timeouts
    DEFAULT_HTTP_CONNECT_TIMEOUT,
    DEFAULT_HTTP_REQUEST_TIMEOUT,
    DEFAULT_SERVER_INIT_TIMEOUT,
    DEFAULT_STREAMING_CHUNK_TIMEOUT,
    DEFAULT_STREAMING_FIRST_CHUNK_TIMEOUT,
    DEFAULT_STREAMING_GLOBAL_TIMEOUT,
    DEFAULT_TOOL_EXECUTION_TIMEOUT,
    DISCOVERY_TIMEOUT,
    REFRESH_TIMEOUT,
    SHUTDOWN_TIMEOUT,
    # Tool config
    DEFAULT_CONFIRM_TOOLS,
    DEFAULT_DYNAMIC_TOOLS_ENABLED,
    DEFAULT_MAX_TOOL_CONCURRENCY,
    # Conversation
    DEFAULT_MAX_TURNS,
    DEFAULT_SYSTEM_PROMPT,
    # Provider/Model
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    # UI
    DEFAULT_THEME,
    DEFAULT_VERBOSE,
    # Token/Auth
    DEFAULT_TOKEN_BACKEND,
    # Paths
    DEFAULT_CONFIG_FILENAME,
    # Application
    APP_NAME,
    GENERIC_NAMESPACE,
    NAMESPACE,
    OAUTH_NAMESPACE,
    PROVIDER_NAMESPACE,
    # Platforms
    PLATFORM_DARWIN,
    PLATFORM_LINUX,
    PLATFORM_WINDOWS,
    # Providers
    PROVIDER_ANTHROPIC,
    PROVIDER_DEEPSEEK,
    PROVIDER_GROQ,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_XAI,
    SUPPORTED_PROVIDERS,
    # JSON Schema
    JSON_TYPE_ARRAY,
    JSON_TYPE_BOOLEAN,
    JSON_TYPE_INTEGER,
    JSON_TYPE_NULL,
    JSON_TYPE_NUMBER,
    JSON_TYPE_OBJECT,
    JSON_TYPE_STRING,
    JSON_TYPES,
)

# Config models
from mcp_cli.config.models import (
    ConfigOverride,
    MCPConfig,
    TimeoutConfig,
    ToolConfig,
    TokenStorageConfig,
    VaultConfig,
)
from mcp_cli.config.runtime import ResolvedValue, RuntimeConfig

# Get version from package metadata
try:
    APP_VERSION = version("mcp-cli")
except PackageNotFoundError:
    APP_VERSION = "0.0.0-dev"

# Legacy compatibility (during transition)
from mcp_cli.config.config_manager import (
    ConfigManager,
    ServerConfig,
    detect_server_types,
    get_config,
    initialize_config,
    validate_server_config,
)
from mcp_cli.config.discovery import (
    force_discovery_refresh,
    get_available_models_quick,
    get_discovery_status,
    setup_chuk_llm_environment,
    trigger_discovery_after_setup,
    validate_provider_exists,
)
from mcp_cli.config.cli_options import (
    extract_server_names,
    get_config_summary,
    load_config,
    process_options,
)
from mcp_cli.config.logging import (
    get_logger,
    setup_logging,
    setup_silent_mcp_environment,
)

__all__ = [
    # Config models
    "MCPConfig",
    "TimeoutConfig",
    "ToolConfig",
    "TokenStorageConfig",
    "VaultConfig",
    "RuntimeConfig",
    "ConfigOverride",
    "ResolvedValue",
    # Enums
    "ConfigSource",
    "ConversationAction",
    "OutputFormat",
    "ServerAction",
    "ServerStatus",
    "ThemeAction",
    "TimeoutType",
    "TokenAction",
    "TokenBackend",
    "TokenNamespace",
    "ToolAction",
    # Environment variables
    "EnvVar",
    "get_env",
    "get_env_bool",
    "get_env_float",
    "get_env_int",
    "get_env_list",
    "is_set",
    "set_env",
    "unset_env",
    # Timeouts
    "DEFAULT_HTTP_CONNECT_TIMEOUT",
    "DEFAULT_HTTP_REQUEST_TIMEOUT",
    "DEFAULT_SERVER_INIT_TIMEOUT",
    "DEFAULT_STREAMING_CHUNK_TIMEOUT",
    "DEFAULT_STREAMING_FIRST_CHUNK_TIMEOUT",
    "DEFAULT_STREAMING_GLOBAL_TIMEOUT",
    "DEFAULT_TOOL_EXECUTION_TIMEOUT",
    "DISCOVERY_TIMEOUT",
    "REFRESH_TIMEOUT",
    "SHUTDOWN_TIMEOUT",
    # Tool config defaults
    "DEFAULT_CONFIRM_TOOLS",
    "DEFAULT_DYNAMIC_TOOLS_ENABLED",
    "DEFAULT_MAX_TOOL_CONCURRENCY",
    # Conversation defaults
    "DEFAULT_MAX_TURNS",
    "DEFAULT_SYSTEM_PROMPT",
    # Provider/Model defaults
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    # UI defaults
    "DEFAULT_THEME",
    "DEFAULT_VERBOSE",
    # Token/Auth defaults
    "DEFAULT_TOKEN_BACKEND",
    # Path defaults
    "DEFAULT_CONFIG_FILENAME",
    # Application constants
    "APP_NAME",
    "APP_VERSION",
    "GENERIC_NAMESPACE",
    "NAMESPACE",
    "OAUTH_NAMESPACE",
    "PROVIDER_NAMESPACE",
    # Platform constants
    "PLATFORM_DARWIN",
    "PLATFORM_LINUX",
    "PLATFORM_WINDOWS",
    # Provider constants
    "PROVIDER_ANTHROPIC",
    "PROVIDER_DEEPSEEK",
    "PROVIDER_GROQ",
    "PROVIDER_OLLAMA",
    "PROVIDER_OPENAI",
    "PROVIDER_XAI",
    "SUPPORTED_PROVIDERS",
    # JSON Schema constants
    "JSON_TYPE_ARRAY",
    "JSON_TYPE_BOOLEAN",
    "JSON_TYPE_INTEGER",
    "JSON_TYPE_NULL",
    "JSON_TYPE_NUMBER",
    "JSON_TYPE_OBJECT",
    "JSON_TYPE_STRING",
    "JSON_TYPES",
    # Legacy (will be removed)
    "ServerConfig",
    "ConfigManager",
    "detect_server_types",
    "get_config",
    "initialize_config",
    "validate_server_config",
    # Discovery
    "setup_chuk_llm_environment",
    "trigger_discovery_after_setup",
    "get_available_models_quick",
    "validate_provider_exists",
    "get_discovery_status",
    "force_discovery_refresh",
    # CLI Options
    "load_config",
    "extract_server_names",
    "process_options",
    "get_config_summary",
    # Logging
    "get_logger",
    "setup_logging",
    "setup_silent_mcp_environment",
]


# Convenience function for loading config
def load_runtime_config(
    config_path: str | None = None,
    cli_overrides: ConfigOverride | None = None,
) -> RuntimeConfig:
    """Load runtime configuration with clean API.

    Args:
        config_path: Path to config file (default: server_config.json)
        cli_overrides: CLI argument overrides

    Returns:
        RuntimeConfig instance ready to use
    """
    from pathlib import Path

    path = Path(config_path or DEFAULT_CONFIG_FILENAME)
    file_config = MCPConfig.load_sync(path)
    return RuntimeConfig(file_config, cli_overrides)


async def load_runtime_config_async(
    config_path: str | None = None,
    cli_overrides: ConfigOverride | None = None,
) -> RuntimeConfig:
    """Async load runtime configuration.

    Args:
        config_path: Path to config file (default: server_config.json)
        cli_overrides: CLI argument overrides

    Returns:
        RuntimeConfig instance ready to use
    """
    from pathlib import Path

    path = Path(config_path or DEFAULT_CONFIG_FILENAME)
    file_config = await MCPConfig.load_async(path)
    return RuntimeConfig(file_config, cli_overrides)


__all__.extend(["load_runtime_config", "load_runtime_config_async"])
