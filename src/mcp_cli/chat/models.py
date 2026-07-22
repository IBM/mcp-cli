"""Chat-specific Pydantic models and protocols."""

from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager


class MessageRole(str, Enum):
    """Message roles in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageField(str, Enum):
    """Message field names for API serialization."""

    ROLE = "role"
    CONTENT = "content"
    NAME = "name"
    TOOL_CALLS = "tool_calls"
    TOOL_CALL_ID = "tool_call_id"
    REASONING_CONTENT = "reasoning_content"


class ToolCallField(str, Enum):
    """Tool call field names for API serialization."""

    ID = "id"
    TYPE = "type"
    FUNCTION = "function"
    INDEX = "index"
    NAME = "name"
    ARGUMENTS = "arguments"


class FunctionCallData(BaseModel):
    """Function call within a tool call (OpenAI format).

    Mutable counterpart to chuk_llm.core.models.FunctionCall (frozen).
    Must remain mutable because ToolCallData.merge_chunk() accumulates
    streaming chunks by mutating name/arguments in place.
    """

    name: str = Field(description="Function/tool name")
    arguments: str = Field(description="JSON string of arguments")

    model_config = {"frozen": False}

    def get_arguments_dict(self) -> dict[str, Any]:
        """Parse arguments JSON string to dict."""
        try:
            result = json.loads(self.arguments)
            return dict(result) if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            return {}

    @classmethod
    def from_dict_args(cls, name: str, arguments: dict[str, Any]) -> "FunctionCallData":
        """Create FunctionCallData from dict arguments."""
        return cls(name=name, arguments=json.dumps(arguments))


class ToolCallData(BaseModel):
    """Tool call data structure (OpenAI format)."""

    id: str = Field(description="Tool call ID")
    type: str = Field(default="function", description="Type of tool call")
    function: FunctionCallData = Field(description="Function call data")
    index: int = Field(default=0, description="Tool call index in batch")

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for API."""
        return {
            ToolCallField.ID: self.id,
            ToolCallField.TYPE: self.type,
            ToolCallField.FUNCTION: {
                ToolCallField.NAME: self.function.name,
                ToolCallField.ARGUMENTS: self.function.arguments,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCallData":
        """Create from dict."""
        return cls(
            id=data.get(ToolCallField.ID, ""),
            type=data.get(ToolCallField.TYPE, "function"),
            index=data.get(ToolCallField.INDEX, 0),
            function=FunctionCallData(
                name=data.get(ToolCallField.FUNCTION, {}).get(ToolCallField.NAME, ""),
                arguments=data.get(ToolCallField.FUNCTION, {}).get(
                    ToolCallField.ARGUMENTS, ""
                ),
            ),
        )

    def merge_chunk(self, chunk: "ToolCallData") -> None:
        """Merge data from a streaming chunk into this tool call.

        Args:
            chunk: New chunk data to merge
        """
        # Update function name if provided
        if chunk.function.name:
            self.function.name = chunk.function.name

        # Accumulate arguments (concatenate JSON strings)
        if chunk.function.arguments:
            self.function.arguments += chunk.function.arguments


class HistoryMessage(BaseModel):
    """A message in the conversation history (dict-based tool_calls for SessionManager compat).

    This is the LOCAL message model used by ChatContext/SessionManager.
    For chuk_llm's canonical Message with typed ToolCall objects,
    use mcp_cli.chat.response_models.Message instead.
    """

    role: MessageRole = Field(
        description="Message role (user, assistant, system, tool)"
    )
    content: str | list[dict[str, Any]] | None = Field(
        default=None,
        description="Message content (string, or list of content blocks for multimodal)",
    )
    name: str | None = Field(default=None, description="Name (for tool messages)")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls (for assistant messages with tools)"
    )
    tool_call_id: str | None = Field(
        default=None, description="Tool call ID (for tool response messages)"
    )
    reasoning_content: str | None = Field(
        default=None,
        description="Reasoning content (for models like DeepSeek reasoner)",
    )

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM API calls.

        Handles provider-specific requirements:
        - OpenAI: Requires 'content' field in assistant messages with tool_calls
        - DeepSeek Reasoner: Requires 'reasoning_content' field when model provided it
        """
        result = self.model_dump(exclude_none=True, mode="json")

        # CRITICAL FIX: OpenAI (especially newer models like gpt-5-mini) requires
        # the 'content' field to be present in assistant messages with tool_calls,
        # even if it's null. Without this, some models may hang or reject the request.
        if self.role == MessageRole.ASSISTANT and MessageField.TOOL_CALLS in result:
            if MessageField.CONTENT not in result:
                result[MessageField.CONTENT] = None

        # NOTE: reasoning_content is automatically included if set (not None)
        # because we're not explicitly excluding it. The exclude_none=True will
        # only exclude it if it's None. This is correct behavior per DeepSeek docs:
        # https://api-docs.deepseek.com/guides/thinking_mode#tool-calls
        # "the user needs to send the reasoning content back to the API"

        return result  # type: ignore[no-any-return]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistoryMessage":
        """Create from dict."""
        return cls.model_validate(data)  # type: ignore[no-any-return]

    def get_tool_calls_typed(self) -> list[ToolCallData]:
        """Get tool calls as typed ToolCallData objects."""
        if not self.tool_calls:
            return []
        return [ToolCallData.from_dict(tc) for tc in self.tool_calls]

    @classmethod
    def with_tool_calls(
        cls,
        role: MessageRole,
        tool_calls: list[ToolCallData],
        content: str | None = None,
    ) -> "HistoryMessage":
        """Create a message with typed tool calls."""
        return cls(
            role=role,
            content=content,
            tool_calls=[tc.to_dict() for tc in tool_calls],
        )


# Backward-compat alias — existing `from mcp_cli.chat.models import Message` still works
Message = HistoryMessage


class ToolExecutionRecord(BaseModel):
    """Record of a tool execution in the chat history."""

    tool_name: str = Field(description="Name of the tool that was executed")
    server: str | None = Field(
        default=None, description="Server that provided the tool"
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )
    result: Any | None = Field(default=None, description="Result from tool execution")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time: float | None = Field(
        default=None, description="Execution time in seconds"
    )
    timestamp: str | None = Field(
        default=None, description="ISO timestamp of execution"
    )

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return self.model_dump(exclude_none=True, mode="json")  # type: ignore[no-any-return]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolExecutionRecord:
        """Create from dict."""
        return cls.model_validate(data)  # type: ignore[no-any-return]


class ToolExecutionState(BaseModel):
    """State tracking for a tool currently being executed."""

    name: str = Field(description="Tool name")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )
    start_time: float = Field(description="Execution start time (timestamp)")
    result: str | None = Field(default=None, description="Tool execution result")
    success: bool = Field(default=True, description="Whether execution succeeded")
    elapsed: float | None = Field(default=None, description="Elapsed time in seconds")
    completed: bool = Field(default=False, description="Whether execution is complete")

    model_config = {"frozen": False}

    def elapsed_time(self, current_time: float) -> float:
        """Calculate elapsed time from start."""
        return current_time - self.start_time


class ChatStatus(BaseModel):
    """Status summary of the chat context."""

    provider: str = Field(description="Current LLM provider")
    model: str = Field(description="Current model")
    tool_count: int = Field(default=0, description="Number of available tools")
    internal_tool_count: int = Field(default=0, description="Number of internal tools")
    server_count: int = Field(default=0, description="Number of connected servers")
    message_count: int = Field(default=0, description="Messages in conversation")
    tool_execution_count: int = Field(
        default=0, description="Tools executed in this session"
    )

    model_config = {"frozen": False}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return self.model_dump(mode="json")  # type: ignore[no-any-return]


class ServerToolGroup(BaseModel):
    """Server-to-tools grouping for system prompt generation."""

    name: str = Field(description="Server name")
    description: str = Field(default="", description="Server description")
    tools: list[str] = Field(default_factory=list, description="Sorted tool names")

    model_config = {"frozen": True}


class ToolCallMetadata(BaseModel):
    """Metadata tracked per in-flight tool call during execution."""

    llm_tool_name: str = Field(description="Tool name as the LLM sees it")
    execution_tool_name: str = Field(description="Actual tool name for execution")
    display_name: str = Field(description="Display name for UI")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Resolved arguments"
    )
    raw_arguments: str = Field(default="", description="Raw arguments string from LLM")

    model_config = {"frozen": True}


# ──────────────────────────────────────────────────────────────────────────────
# Protocols - formalize interfaces for type safety
# ──────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class ToolProcessorContext(Protocol):
    """Protocol for context objects used by ToolProcessor.

    Formalizes the interface instead of using dynamic getattr/setattr.
    This ensures type safety and makes dependencies explicit.
    """

    # Required attributes
    tool_manager: "ToolManager"
    conversation_history: list[HistoryMessage]

    # Optional processor back-reference (set by ToolProcessor)
    tool_processor: Any  # Will be set to ToolProcessor instance

    # Optional planning context (lazy-created by _handle_plan_tool)
    _planning_context: Any

    def get_display_name_for_tool(self, tool_name: str) -> str:
        """Get display name for a tool (may be namespaced)."""
        ...

    def inject_tool_message(self, message: HistoryMessage) -> None:
        """Add a message directly to conversation history."""
        ...


@runtime_checkable
class UIManagerProtocol(Protocol):
    """Protocol for UI managers used by ToolProcessor.

    Defines the minimal interface required by ToolProcessor.
    The actual ChatUIManager has many more methods, but these are
    the core ones used during tool execution.

    Note: Uses Any return types where the implementation varies.
    """

    # Core attributes
    interrupt_requested: bool
    verbose_mode: bool
    console: Any  # Rich Console instance

    def print_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Print tool call info to console."""
        ...

    async def do_confirm_tool_execution(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        """Ask user to confirm tool execution.

        Returns True if user confirms, False otherwise.
        May route to dashboard if browser clients are connected.
        """
        ...

    async def start_tool_execution(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """Signal start of tool execution for UI updates."""
        ...

    async def finish_tool_execution(
        self, result: str | None = None, success: bool = True
    ) -> None:
        """Signal end of tool execution for UI updates."""
        ...

    def finish_tool_calls(self) -> None:
        """Clean up after all tool calls complete."""
        ...
