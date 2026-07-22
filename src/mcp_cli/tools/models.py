# mcp_cli/tools/models.py
"""Data models used throughout MCP-CLI."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from mcp_cli.llm.content_models import ContentBlockType


# ──────────────────────────────────────────────────────────────────────────────
# Constants and Enums
# ──────────────────────────────────────────────────────────────────────────────
class TransportType(str, Enum):
    """MCP transport types."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    UNKNOWN = "unknown"


class ToolType(str, Enum):
    """Tool definition types for LLM APIs."""

    FUNCTION = "function"


# ──────────────────────────────────────────────────────────────────────────────
# Server Capabilities Model
# ──────────────────────────────────────────────────────────────────────────────
class ExperimentalCapabilities(BaseModel):
    """Experimental MCP server capabilities."""

    sampling: bool = Field(default=False, description="Sampling support")
    logging: bool = Field(default=False, description="Logging support")
    streaming: bool = Field(default=False, description="Streaming support")

    model_config = {"frozen": False, "extra": "allow"}


class ServerCapabilities(BaseModel):
    """MCP server capabilities."""

    tools: bool = Field(default=False, description="Tools support")
    prompts: bool = Field(default=False, description="Prompts support")
    resources: bool = Field(default=False, description="Resources support")
    experimental: ExperimentalCapabilities = Field(
        default_factory=ExperimentalCapabilities, description="Experimental features"
    )

    model_config = {"frozen": False, "extra": "allow"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServerCapabilities":
        """Create from dictionary."""
        # Handle nested experimental dict
        if "experimental" in data and isinstance(data["experimental"], dict):
            data = data.copy()
            data["experimental"] = ExperimentalCapabilities.model_validate(
                data["experimental"]
            )
        return cls.model_validate(data)  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")  # type: ignore[no-any-return]


# ──────────────────────────────────────────────────────────────────────────────
# MCP Apps metadata models (SEP-1865)
# ──────────────────────────────────────────────────────────────────────────────
class ToolUIMeta(BaseModel):
    """UI metadata for MCP Apps (SEP-1865).

    Indicates that a tool has an interactive HTML UI that can be
    rendered in a browser. The resourceUri points to a ui:// resource
    containing the HTML content.
    """

    resourceUri: str = Field(description="URI of the UI resource (ui:// scheme)")
    viewUrl: str | None = Field(
        default=None,
        description="Direct HTTPS URL for the UI (preferred over resourceUri)",
    )
    visibility: list[str] = Field(
        default_factory=lambda: ["model", "app"],
        description="Who can see/call this tool: 'model' and/or 'app'",
    )
    csp: dict[str, Any] | None = Field(
        default=None, description="Content Security Policy domains"
    )
    permissions: dict[str, Any] | None = Field(
        default=None,
        description="Requested iframe permissions (camera, microphone, etc.)",
    )

    model_config = {"frozen": False, "extra": "allow"}


class ToolMeta(BaseModel):
    """Tool _meta field from MCP spec.

    Contains optional metadata about the tool including UI information
    for MCP Apps support.
    """

    ui: ToolUIMeta | None = None

    model_config = {"frozen": False, "extra": "allow"}


# ──────────────────────────────────────────────────────────────────────────────
# Tool-related models (converted to Pydantic)
# ──────────────────────────────────────────────────────────────────────────────
class ToolInfo(BaseModel):
    """
    Information about a tool.

    Lightweight adapter over chuk-tool-processor's ToolMetadata for MCP-CLI UI/display purposes.
    """

    name: str
    namespace: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    is_async: bool = False
    tags: list[str] = Field(default_factory=list)
    supports_streaming: bool = False
    meta: ToolMeta | None = None

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @property
    def has_app_ui(self) -> bool:
        """Check if this tool has an MCP Apps interactive UI."""
        return self.meta is not None and self.meta.ui is not None

    @property
    def app_resource_uri(self) -> str | None:
        """Get the UI resource URI if available."""
        if self.meta and self.meta.ui:
            return self.meta.ui.resourceUri
        return None

    @property
    def app_view_url(self) -> str | None:
        """Get the direct view URL if available (fallback for resourceUri)."""
        if self.meta and self.meta.ui:
            return self.meta.ui.viewUrl
        return None

    @property
    def fully_qualified_name(self) -> str:
        """Get the fully qualified tool name (namespace.name)."""
        return f"{self.namespace}.{self.name}" if self.namespace else self.name

    @property
    def display_name(self) -> str:
        """Get a user-friendly display name."""
        return self.name

    @property
    def has_parameters(self) -> bool:
        """Check if the tool has parameters defined."""
        return bool(self.parameters and self.parameters.get("properties"))

    @property
    def required_parameters(self) -> list[str]:
        """Get list of required parameter names."""
        if not self.parameters:
            return []
        required = self.parameters.get("required", [])
        return required if isinstance(required, list) else []

    def to_llm_format(self) -> "LLMToolDefinition":
        """Convert to LLM function calling format (OpenAI/Anthropic compatible)."""
        return LLMToolDefinition(
            function=FunctionDefinition(
                name=self.name,
                description=self.description or "No description provided",
                parameters=self.parameters or {"type": "object", "properties": {}},
            )
        )


class ServerInfo(BaseModel):
    """Information about a connected server instance."""

    id: int
    name: str
    status: str
    tool_count: int
    namespace: str
    enabled: bool = True
    connected: bool = False
    transport: TransportType = TransportType.STDIO
    capabilities: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None  # From server metadata
    version: str | None = None  # Server version
    command: str | None = None  # Server command if known (for stdio)
    url: str | None = None  # Server URL (for http/sse)
    args: list[str] = Field(default_factory=list)  # Command arguments
    env: dict[str, str] = Field(default_factory=dict)  # Environment variables

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    @property
    def is_healthy(self) -> bool:
        """Check if server is healthy and ready."""
        return self.status == "healthy" and self.connected

    @property
    def display_status(self) -> str:
        """Get a user-friendly status string."""
        if not self.enabled:
            return "disabled"
        elif not self.connected:
            return "disconnected"
        else:
            return self.status

    @property
    def display_description(self) -> str:
        """Get description or a default based on name."""
        # Use server-provided description if available
        if self.description:
            return self.description
        # Otherwise just return a generic description
        return f"{self.name} MCP server"

    @property
    def has_tools(self) -> bool:
        """Check if server has any tools."""
        return self.tool_count > 0

    def get_capabilities_typed(self) -> ServerCapabilities:
        """Get capabilities as typed ServerCapabilities object."""
        return ServerCapabilities.from_dict(self.capabilities)


class ToolCallResult(BaseModel):
    """
    Outcome of a tool execution.

    ENHANCED: Now wraps chuk_tool_processor.models.tool_result.ToolResult
    for full tracking (start_time, end_time, machine, pid, cached, attempts).

    Provides simplified interface for backward compatibility while exposing
    chuk's rich result data via .chuk_result property.
    """

    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float | None = None

    # Rich chuk result data (optional, provides full tracking)
    chuk_result: Any | None = None

    model_config = {"frozen": False, "arbitrary_types_allowed": True, "extra": "allow"}

    @classmethod
    def from_chuk_result(cls, tool_result: Any) -> "ToolCallResult":
        """
        Create ToolCallResult from chuk's ToolResult.

        Args:
            tool_result: chuk_tool_processor.models.tool_result.ToolResult

        Returns:
            ToolCallResult with data mapped from chuk's ToolResult
        """
        # Calculate execution time from start/end
        execution_time = None
        if hasattr(tool_result, "start_time") and hasattr(tool_result, "end_time"):
            if tool_result.start_time and tool_result.end_time:
                delta = tool_result.end_time - tool_result.start_time
                execution_time = delta.total_seconds()

        return cls(
            tool_name=tool_result.tool,
            success=(tool_result.error is None),
            result=tool_result.result,
            error=tool_result.error,
            execution_time=execution_time,
            chuk_result=tool_result,
        )

    @property
    def is_cached(self) -> bool:
        """Check if result was cached."""
        if self.chuk_result and hasattr(self.chuk_result, "cached"):
            return bool(self.chuk_result.cached)
        return False

    @property
    def attempts(self) -> int:
        """Get number of execution attempts."""
        if self.chuk_result and hasattr(self.chuk_result, "attempts"):
            return int(self.chuk_result.attempts)
        return 1

    @property
    def machine(self) -> str | None:
        """Get machine where tool was executed."""
        if self.chuk_result and hasattr(self.chuk_result, "machine"):
            machine_val = self.chuk_result.machine
            return str(machine_val) if machine_val is not None else None
        return None

    def _extract_mcp_text_content(self, result: Any) -> str | None:
        """Extract text content from MCP SDK ToolResult structure."""
        if isinstance(result, dict):
            # Check for MCP response structure: {'isError': bool, 'content': ToolResult}
            if "content" in result and hasattr(result["content"], "content"):
                # Extract content array from MCP ToolResult
                tool_result_content = result["content"].content
                if isinstance(tool_result_content, list):
                    # Extract text from content blocks - use enum, no magic strings!
                    text_parts = []
                    for block in tool_result_content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == ContentBlockType.TEXT.value
                        ):
                            text_parts.append(block.get("text", ""))
                    if text_parts:
                        return "\n".join(text_parts)
        return None

    @property
    def display_result(self) -> str:
        """Get a display-friendly result string."""
        if not self.success:
            error_msg = self.error or "Unknown error"
            return f"Error: {error_msg}"

        # Try to extract MCP text content first
        mcp_text = self._extract_mcp_text_content(self.result)
        if mcp_text is not None:
            return mcp_text

        # Format result based on type
        import json

        if isinstance(self.result, dict):
            try:
                return json.dumps(self.result, indent=2)
            except (TypeError, ValueError):
                return str(self.result)
        elif isinstance(self.result, str):
            return self.result
        elif self.result is not None:
            return str(self.result)
        else:
            return ""

    @property
    def has_error(self) -> bool:
        """Check if the result contains an error."""
        return not self.success or self.error is not None

    def to_conversation_history(self) -> str:
        """Convert result to conversation history string."""
        if not self.success:
            error_msg = self.error or "Unknown error"
            return f"Tool execution failed: {error_msg}"

        # Try to extract MCP text content first
        mcp_text = self._extract_mcp_text_content(self.result)
        if mcp_text is not None:
            return mcp_text

        # Return result as string
        if isinstance(self.result, dict):
            import json

            try:
                return json.dumps(self.result, indent=2)
            except (TypeError, ValueError):
                return str(self.result)
        elif isinstance(self.result, str):
            return self.result
        elif self.result is not None:
            return str(self.result)
        else:
            return ""


class ValidationResult(BaseModel):
    """Result of tool schema validation."""

    is_valid: bool = Field(description="Whether the tool schema is valid")
    error_message: str | None = Field(
        default=None, description="Error message if validation failed"
    )
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings")

    model_config = {"frozen": False}

    @classmethod
    def success(cls) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(is_valid=True, error_message=None)

    @classmethod
    def failure(cls, error: str) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(is_valid=False, error_message=error)

    @classmethod
    def from_tuple(cls, result: tuple[bool, str | None]) -> "ValidationResult":
        """Create from legacy tuple format."""
        is_valid, error = result
        return cls(is_valid=is_valid, error_message=error)

    @property
    def display_result(self) -> str:
        """Get a display-friendly result string."""
        if not self.is_valid:
            return f"Error: {self.error_message or 'Unknown error'}"
        else:
            return "Validation successful"

    @property
    def has_error(self) -> bool:
        """Check if the result contains an error."""
        return not self.is_valid or self.error_message is not None


# ──────────────────────────────────────────────────────────────────────────────
# NEW - resource-related models (converted to Pydantic)
# ──────────────────────────────────────────────────────────────────────────────
class ResourceInfo(BaseModel):
    """
    Canonical representation of *one* resource entry as returned by
    ``resources.list``.

    The MCP spec does not prescribe a single shape, so we normalise the common
    fields we use in the UI.  **All additional keys** are preserved inside
    ``extra``.
    """

    # Common attributes we frequently need in the UI
    id: str | None = None
    name: str | None = None
    type: str | None = None

    # Anything else goes here …
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_raw(cls, raw: Any) -> "ResourceInfo":
        """
        Convert a raw list item (dict | str | int | …) into a ResourceInfo.

        If *raw* is not a mapping we treat it as an opaque scalar and store it
        in ``extra["value"]`` so it is never lost.
        """
        if isinstance(raw, dict):
            known = {k: raw.get(k) for k in ("id", "name", "type")}
            extra = {k: v for k, v in raw.items() if k not in known}
            return cls(**known, extra=extra)
        # primitive - wrap it
        return cls(extra={"value": raw})


# ──────────────────────────────────────────────────────────────────────────────
# Transport and Server Configuration Models
# ──────────────────────────────────────────────────────────────────────────────
class TransportServerConfig(BaseModel):
    """
    Configuration for a transport server (HTTP/SSE).

    This replaces dict-based server entries throughout the codebase.
    """

    name: str = Field(description="Server name/identifier")
    url: str = Field(description="Server URL endpoint")
    headers: dict[str, str] = Field(
        default_factory=dict, description="Custom HTTP headers"
    )
    api_key: str | None = Field(default=None, description="API key for authentication")
    timeout: float | None = Field(
        default=None, description="Server-specific timeout override"
    )
    max_retries: int | None = Field(
        default=None, description="Server-specific max retries override"
    )

    model_config = {"frozen": False}

    def to_stream_manager_config(self) -> dict[str, Any]:
        """Convert to format expected by StreamManager."""
        return self.model_dump(exclude_none=True)  # type: ignore[no-any-return]


# ──────────────────────────────────────────────────────────────────────────────
# Conversation Message Models
# ──────────────────────────────────────────────────────────────────────────────
class ToolCallMessage(BaseModel):
    """Tool call within a message (OpenAI format)."""

    id: str = Field(min_length=1, description="Tool call ID")
    type: str = Field(default="function", pattern="^function$", description="Call type")
    function: dict[str, Any] = Field(description="Function call details")

    model_config = {"frozen": False}


class ConversationMessage(BaseModel):
    """
    A single message in the conversation history.

    Compatible with OpenAI/Anthropic message format.
    """

    role: str = Field(
        pattern="^(user|assistant|system|tool)$",
        description="Message role: user, assistant, system, or tool",
    )
    content: str | None = Field(default=None, description="Message content")
    name: str | None = Field(default=None, description="Name for tool responses")
    tool_calls: list[ToolCallMessage] | None = Field(
        default=None, description="Tool calls made by assistant"
    )
    tool_call_id: str | None = Field(
        default=None, description="ID of tool call being responded to"
    )

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return self.model_dump(exclude_none=True, mode="json")  # type: ignore[no-any-return]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary."""
        return cls.model_validate(data)  # type: ignore[no-any-return]

    @classmethod
    def user_message(cls, content: str) -> "ConversationMessage":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant_message(
        cls, content: str | None = None, tool_calls: list[dict[str, Any]] | None = None
    ) -> "ConversationMessage":
        """Create an assistant message."""
        parsed_tool_calls = None
        if tool_calls:
            parsed_tool_calls = [
                ToolCallMessage.model_validate(tc) for tc in tool_calls
            ]
        return cls(role="assistant", content=content, tool_calls=parsed_tool_calls)

    @classmethod
    def system_message(cls, content: str) -> "ConversationMessage":
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def tool_message(
        cls, content: str, tool_call_id: str, name: str | None = None
    ) -> "ConversationMessage":
        """Create a tool response message."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


# ──────────────────────────────────────────────────────────────────────────────
# LLM Tool Definition Models (OpenAI/Anthropic Compatible)
# ──────────────────────────────────────────────────────────────────────────────
class FunctionDefinition(BaseModel):
    """Function definition for LLM tool calling."""

    name: str = Field(description="Function name")
    description: str = Field(description="Function description")
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for function parameters",
    )

    model_config = {"frozen": False}


class LLMToolDefinition(BaseModel):
    """
    LLM-compatible tool definition.

    Follows OpenAI function calling format, compatible with Anthropic and other providers.
    """

    type: ToolType = Field(
        default=ToolType.FUNCTION, description="Tool type (always 'function')"
    )
    function: FunctionDefinition = Field(description="Function definition")

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for LLM API calls."""
        return self.model_dump(mode="json")  # type: ignore[no-any-return]


# ──────────────────────────────────────────────────────────────────────────────
# Tool Definition Input Models (for parsing raw tool dicts)
# ──────────────────────────────────────────────────────────────────────────────
class ToolInputSchema(BaseModel):
    """Input schema for tool parameters (JSON Schema format)."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False

    model_config = {"frozen": True, "extra": "allow"}


class ToolDefinitionInput(BaseModel):
    """Input model for parsing tool definitions from dicts.

    Used to convert raw tool dicts from chuk_tool_processor into ToolInfo models.
    """

    name: str
    namespace: str = "default"
    description: str | None = None
    inputSchema: dict[str, Any] = Field(default_factory=dict)
    is_async: bool = False
    tags: list[str] = Field(default_factory=list)
    meta: ToolMeta | None = Field(default=None, alias="_meta")

    model_config = {"frozen": False, "extra": "ignore", "populate_by_name": True}
