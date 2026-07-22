"""Configuration enums - no magic strings!

All enums for type-safe configuration and commands.
"""

from __future__ import annotations

from enum import Enum


# ================================================================
# Configuration Enums
# ================================================================


class TimeoutType(str, Enum):
    """All timeout configuration types - type-safe timeout keys."""

    STREAMING_CHUNK = "streaming_chunk"
    STREAMING_GLOBAL = "streaming_global"
    STREAMING_FIRST_CHUNK = "streaming_first_chunk"
    TOOL_EXECUTION = "tool_execution"
    SERVER_INIT = "server_init"
    HTTP_REQUEST = "http_request"
    HTTP_CONNECT = "http_connect"


class TokenBackend(str, Enum):
    """Token storage backend types."""

    AUTO = "auto"
    KEYCHAIN = "keychain"
    WINDOWS = "windows"
    SECRET_SERVICE = "secretservice"
    ENCRYPTED = "encrypted"
    VAULT = "vault"


class ConfigSource(str, Enum):
    """Configuration value source for priority resolution."""

    CLI = "cli"
    ENV = "env"
    FILE = "file"
    DEFAULT = "default"


# NOTE: Theme names come from chuk-term.ui.theme.Theme
# Valid values: "default", "dark", "light", "minimal", "terminal"
# We don't duplicate them here - use strings and let chuk-term validate


# ================================================================
# Server/Status Enums
# ================================================================


class ServerStatus(str, Enum):
    """Server status values."""

    CONFIGURED = "configured"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


# ================================================================
# Command Action Enums
# ================================================================


class ConversationAction(str, Enum):
    """Actions for /conversation command."""

    SHOW = "show"
    CLEAR = "clear"
    SAVE = "save"
    LOAD = "load"


class TokenAction(str, Enum):
    """Actions for /token command."""

    LIST = "list"
    SET = "set"
    GET = "get"
    DELETE = "delete"
    CLEAR = "clear"
    BACKENDS = "backends"
    SET_PROVIDER = "set-provider"
    GET_PROVIDER = "get-provider"
    DELETE_PROVIDER = "delete-provider"


class OutputFormat(str, Enum):
    """Output format types for command results."""

    JSON = "json"
    TABLE = "table"
    TEXT = "text"
    TREE = "tree"


class TokenNamespace(str, Enum):
    """Token storage namespaces."""

    GENERIC = "generic"
    PROVIDER = "provider"
    BEARER = "bearer"
    API_KEY = "api-key"
    OAUTH = "oauth"


class SessionAction(str, Enum):
    """Actions for /sessions command."""

    LIST = "list"
    SAVE = "save"
    LOAD = "load"
    DELETE = "delete"


class PlanAction(str, Enum):
    """Actions for /plan command."""

    CREATE = "create"
    LIST = "list"
    SHOW = "show"
    RUN = "run"
    DELETE = "delete"
    RESUME = "resume"


class PlanStatus(str, Enum):
    """Plan execution status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ServerAction(str, Enum):
    """Actions for /server command."""

    ENABLE = "enable"
    DISABLE = "disable"
    STATUS = "status"
    INFO = "info"


class ToolAction(str, Enum):
    """Actions for /tools command."""

    LIST = "list"
    ENABLE = "enable"
    DISABLE = "disable"
    CONFIRM = "confirm"
    INFO = "info"


class ThemeAction(str, Enum):
    """Actions for /theme command."""

    SET = "set"
    LIST = "list"
    SHOW = "show"


__all__ = [
    # Configuration enums
    "TimeoutType",
    "TokenBackend",
    "ConfigSource",
    # Server enums
    "ServerStatus",
    # Command action enums
    "ConversationAction",
    "PlanAction",
    "PlanStatus",
    "SessionAction",
    "TokenAction",
    "ServerAction",
    "ToolAction",
    "ThemeAction",
    # Format/Namespace enums
    "OutputFormat",
    "TokenNamespace",
]
