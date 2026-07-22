"""Default configuration values - no more magic numbers!

All default values should be defined here, not hardcoded in the code.
"""

from __future__ import annotations


# ================================================================
# Timeout Defaults (in seconds)
# ================================================================

# Streaming timeouts
DEFAULT_STREAMING_CHUNK_TIMEOUT = 45.0
"""Default timeout for receiving each streaming chunk."""

DEFAULT_STREAMING_GLOBAL_TIMEOUT = 300.0
"""Default total streaming timeout."""

DEFAULT_STREAMING_FIRST_CHUNK_TIMEOUT = 60.0
"""Default timeout for first chunk (may need longer for complex queries)."""

DEFAULT_STREAMING_FIRST_CHUNK_AFTER_TOOLS_TIMEOUT = 180.0
"""Timeout for first chunk after tool calls (thinking models need extended processing time)."""

# Tool timeouts
DEFAULT_TOOL_EXECUTION_TIMEOUT = 120.0
"""Default timeout for tool execution."""

# Server timeouts
DEFAULT_SERVER_INIT_TIMEOUT = 120.0
"""Default timeout for server initialization."""

# HTTP timeouts
DEFAULT_HTTP_REQUEST_TIMEOUT = 30.0
"""Default timeout for HTTP requests."""

DEFAULT_HTTP_CONNECT_TIMEOUT = 10.0
"""Default timeout for HTTP connections."""

# Discovery/UI timeouts (moved from constants/timeouts.py)
DISCOVERY_TIMEOUT = 10.0
"""Provider discovery HTTP timeout."""

DEFAULT_PROVIDER_DISCOVERY_TIMEOUT = 5.0
"""Timeout for provider/model discovery HTTP requests and subprocess calls."""

REFRESH_TIMEOUT = 1.0
"""Display refresh timeout."""

SHUTDOWN_TIMEOUT = 0.5
"""Graceful shutdown timeout."""


# ================================================================
# Tool Configuration Defaults
# ================================================================

DEFAULT_MAX_TOOL_CONCURRENCY = 5
"""Default maximum concurrent tool executions."""

DEFAULT_CONFIRM_TOOLS = True
"""Default: require confirmation before executing tools."""

DEFAULT_DYNAMIC_TOOLS_ENABLED = False
"""Default: dynamic tool discovery disabled."""


# ================================================================
# Conversation Defaults
# ================================================================

DEFAULT_MAX_TURNS = 100
"""Default maximum conversation turns before exit."""

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant with access to tools."
"""Default system prompt."""


# ================================================================
# Context Management Defaults
# ================================================================

DEFAULT_MAX_TOOL_RESULT_CHARS = 100_000
"""Max chars for a single tool result in conversation history (~25K tokens). 0 = unlimited."""

DEFAULT_MAX_HISTORY_MESSAGES = 200
"""Max conversation history messages (sliding window). 0 = unlimited."""

DEFAULT_INFINITE_CONTEXT = False
"""Enable infinite context mode (auto-summarization via SessionManager)."""

DEFAULT_TOKEN_THRESHOLD = 4000
"""Token threshold for infinite context segmentation."""

DEFAULT_MAX_TURNS_PER_SEGMENT = 20
"""Max turns per segment before context packing triggers."""

DEFAULT_MAX_STREAMING_BUFFER_CHARS = 1_048_576
"""Max accumulated streaming content in chars (1 MB). 0 = unlimited."""

DEFAULT_MAX_STREAMING_CHUNKS = 50_000
"""Max streaming chunks before stall detection. 0 = unlimited."""


# ================================================================
# Virtual Memory Defaults (Experimental)
# ================================================================

DEFAULT_ENABLE_VM = False
"""Enable AI Virtual Memory subsystem (experimental)."""

DEFAULT_VM_MODE = "passive"
"""VM mode: strict, relaxed, or passive. Passive is safest for initial testing."""

DEFAULT_VM_BUDGET = 128_000
"""Token budget for conversation events in VM mode (on top of system prompt). Lower values force earlier eviction."""


# ================================================================
# Tier 2: Efficiency & Resilience Defaults
# ================================================================

DEFAULT_BATCH_TIMEOUT_MULTIPLIER = 2.0
"""Batch timeout = max(per_tool_timeout * multiplier, floor)."""

DEFAULT_BATCH_TIMEOUT_FLOOR = 60.0
"""Minimum batch timeout in seconds."""

DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD = 20
"""When server has more than this many tools, show summary instead of full list."""

DEFAULT_RECONNECT_ON_FAILURE = True
"""Attempt to reconnect when a tool execution fails with a connection error."""

DEFAULT_MAX_RECONNECT_ATTEMPTS = 3
"""Maximum number of reconnection attempts before giving up."""

DEFAULT_CONTEXT_NOTICES_ENABLED = True
"""Enable LLM-visible context management notices (truncation, eviction, stripping)."""

DEFAULT_MAX_CONSECUTIVE_DUPLICATES = 5
"""Abort conversation loop after this many consecutive duplicate tool calls."""

DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES = 3
"""Warn user after this many consecutive transport failures."""

DEFAULT_SYSTEM_PROMPT_TOOL_PREVIEW_COUNT = 5
"""Number of tool names to show before summarizing in system prompt."""

DYNAMIC_TOOL_PROXY_NAME = "call_tool"
"""Name of the dynamic tool proxy used in discovery mode."""

TRANSPORT_ERROR_PATTERNS = (
    "transport not initialized",
    "transport",
)
"""Patterns (lowercase) indicating transport/connection failure in error messages."""


# ================================================================
# Token Tracking Defaults
# ================================================================

DEFAULT_TOKEN_TRACKING_ENABLED = True
"""Track token usage per turn and cumulatively."""

DEFAULT_CHARS_PER_TOKEN_ESTIMATE = 4
"""Approximate characters per token for estimation fallback."""

DEFAULT_AUTO_SAVE_INTERVAL = 10
"""Auto-save session every N turns."""


# ================================================================
# Provider/Model Defaults
# ================================================================

DEFAULT_PROVIDER = "openai"
"""Default LLM provider."""

DEFAULT_MODEL = "gpt-4o-mini"
"""Default LLM model."""


# ================================================================
# UI Defaults
# ================================================================

DEFAULT_THEME = "default"
"""Default UI theme (from chuk-term)."""

DEFAULT_VERBOSE = True
"""Default verbosity level."""


# ================================================================
# Token/Auth Defaults
# ================================================================

DEFAULT_TOKEN_BACKEND = "auto"
"""Default token storage backend."""


# ================================================================
# Path Defaults
# ================================================================

DEFAULT_CONFIG_DIR = "~/.mcp-cli"
"""Default root config directory for mcp-cli."""

DEFAULT_SESSIONS_DIR = "~/.mcp-cli/sessions"
"""Default directory for saved conversation sessions."""

DEFAULT_DOWNLOADS_DIR = "~/.mcp-cli/downloads"
"""Default directory for downloaded files (e.g., VM page exports)."""

DEFAULT_CONFIG_FILENAME = "server_config.json"
"""Default configuration filename."""


# ================================================================
# Application Constants
# ================================================================

NAMESPACE = "mcp-cli"
"""Application namespace."""

OAUTH_NAMESPACE = NAMESPACE
"""OAuth namespace (same as app namespace)."""

PROVIDER_NAMESPACE = "provider"
"""Provider namespace for token storage."""

GENERIC_NAMESPACE = "generic"
"""Generic namespace."""

APP_NAME = "mcp-cli"
"""Application name."""


# ================================================================
# Platform Constants
# ================================================================

PLATFORM_WINDOWS = "win32"
"""Windows platform identifier (from sys.platform)."""

PLATFORM_DARWIN = "darwin"
"""macOS platform identifier (from sys.platform)."""

PLATFORM_LINUX = "linux"
"""Linux platform identifier (from sys.platform)."""


# ================================================================
# Provider Constants
# ================================================================

PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GROQ = "groq"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_XAI = "xai"

SUPPORTED_PROVIDERS = [
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GROQ,
    PROVIDER_DEEPSEEK,
    PROVIDER_XAI,
]
"""List of supported LLM providers."""


# ================================================================
# JSON Schema Type Constants
# ================================================================

JSON_TYPE_STRING = "string"
JSON_TYPE_NUMBER = "number"
JSON_TYPE_INTEGER = "integer"
JSON_TYPE_BOOLEAN = "boolean"
JSON_TYPE_ARRAY = "array"
JSON_TYPE_OBJECT = "object"
JSON_TYPE_NULL = "null"

JSON_TYPES = [
    JSON_TYPE_STRING,
    JSON_TYPE_NUMBER,
    JSON_TYPE_INTEGER,
    JSON_TYPE_BOOLEAN,
    JSON_TYPE_ARRAY,
    JSON_TYPE_OBJECT,
    JSON_TYPE_NULL,
]
"""All valid JSON Schema types."""


# ================================================================
# Middleware Configuration
# ================================================================
# Middleware (retry, circuit breaker, rate limiting) is provided by
# chuk-tool-processor. See chuk_tool_processor.mcp.MiddlewareConfig
# for configuration options.
DEFAULT_MIDDLEWARE_ENABLED = True
"""Enable CTP middleware by default."""


# ================================================================
# MCP Apps Defaults (SEP-1865)
# ================================================================

DEFAULT_APP_HOST_PORT_START = 9470
"""Starting port for local MCP App host servers."""

DEFAULT_APP_AUTO_OPEN_BROWSER = True
"""Automatically open browser when an MCP App is launched."""

DEFAULT_APP_MAX_CONCURRENT = 10
"""Maximum number of concurrent MCP Apps."""

DEFAULT_APP_TOOL_TIMEOUT = 120.0
"""Maximum seconds for a tool call from an MCP App."""

DEFAULT_APP_INIT_TIMEOUT = 30
"""Seconds before showing 'initialization timed out' in host page."""


# ================================================================
# Agent Identity Defaults
# ================================================================

DEFAULT_AGENT_ID = "default"
"""Default agent identifier for multi-agent support."""


# ================================================================
# Dashboard Defaults
# ================================================================

DEFAULT_DASHBOARD_PORT_START = 9120
"""Starting port for the dashboard HTTP+WebSocket server."""

DEFAULT_DASHBOARD_AUTO_OPEN_BROWSER = True
"""Automatically open browser when dashboard is launched."""

DEFAULT_DASHBOARD_RECONNECT_INTERVAL = 5.0
"""Seconds between WebSocket reconnect attempts."""

DEFAULT_DASHBOARD_LAYOUTS_FILE = "~/.config/mcp-cli/dashboard-layouts.json"
"""Path to user-saved dashboard layout configurations."""


# ================================================================
# Memory Scope Defaults (Tier 8)
# ================================================================

DEFAULT_MEMORY_BASE_DIR = "~/.mcp-cli/memory"
"""Default directory for persistent memory storage."""

DEFAULT_MEMORY_MAX_ENTRIES_PER_SCOPE = 100
"""Maximum number of memory entries per scope before oldest is evicted."""

DEFAULT_MEMORY_MAX_PROMPT_CHARS = 2000
"""Maximum characters for memory section in system prompt."""


# ================================================================
# Planning Defaults (Tier 6)
# ================================================================

DEFAULT_PLANS_DIR = "~/.mcp-cli/plans"
"""Default directory for plan persistence."""

DEFAULT_ENABLE_PLAN_TOOLS = False
"""Enable plan_create / plan_execute / plan_create_and_execute as LLM-callable tools."""

DEFAULT_PLAN_MAX_CONCURRENCY = 4
"""Maximum concurrent steps within a parallel batch."""

DEFAULT_PLAN_MAX_REPLANS = 2
"""Maximum number of re-plan attempts on step failure."""

DEFAULT_PLAN_MAX_STEP_RETRIES = 2
"""Maximum LLM retry attempts per plan step on tool failure."""

DEFAULT_PLAN_VARIABLE_SUMMARY_MAX_CHARS = 500
"""Maximum characters per variable in LLM variable summary."""

DEFAULT_PLAN_CHECKPOINT_MAX_CHARS = 1000
"""Maximum characters per variable in checkpoint serialization."""

DEFAULT_PLAN_ERROR_MESSAGE_MAX_CHARS = 200
"""Maximum characters for error messages in plan execution results."""

DEFAULT_PLAN_DAG_TITLE_MAX_CHARS = 35
"""Maximum characters for step titles in DAG visualization."""


# ================================================================
# Logging Defaults
# ================================================================

DEFAULT_LOG_DIR = "~/.mcp-cli/logs"
"""Default directory for log files."""

DEFAULT_LOG_MAX_BYTES = 10_485_760
"""Max size per log file (10 MB) before rotation."""

DEFAULT_LOG_BACKUP_COUNT = 3
"""Number of rotated log files to keep."""


# ================================================================
# Attachment Defaults
# ================================================================

DEFAULT_MAX_ATTACHMENT_SIZE_BYTES = 20_971_520
"""Maximum attachment file size (20 MB). Base64 encoding adds ~33%."""

DEFAULT_MAX_ATTACHMENTS_PER_MESSAGE = 10
"""Maximum attachments per user message (staged + inline combined)."""

DEFAULT_IMAGE_DETAIL_LEVEL = "auto"
"""Default detail level for image_url content blocks (auto, low, high)."""

DEFAULT_DASHBOARD_INLINE_IMAGE_THRESHOLD = 102_400
"""Max base64 size (bytes) for inline image previews in dashboard (100 KB)."""

DEFAULT_DASHBOARD_TEXT_PREVIEW_CHARS = 2000
"""Max chars of text file content to send as preview in dashboard."""
