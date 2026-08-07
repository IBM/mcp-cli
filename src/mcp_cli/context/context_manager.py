"""
Centralized context manager for MCP CLI.

This module provides a centralized way to manage application context
instead of passing dictionaries around.
"""

from __future__ import annotations

from typing import Any, Annotated
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, SkipValidation

from mcp_cli.config.defaults import DEFAULT_PROVIDER, DEFAULT_MODEL, DEFAULT_CONFIG_FILENAME
from mcp_cli.tools.manager import ToolManager
from mcp_cli.model_management import ModelManager
from mcp_cli.tools.models import ServerInfo, ToolInfo, ConversationMessage


class ApplicationContext(BaseModel):
    """
    Centralized application context that holds all state and managers.

    This replaces the dictionary-based context that was being passed around.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow ToolManager, ModelManager, etc.
        validate_assignment=True,
    )

    # Core managers (skip validation to allow test mocks)
    tool_manager: Annotated[ToolManager | None, SkipValidation()] = None
    model_manager: Annotated[ModelManager | None, SkipValidation()] = None

    # Configuration
    config_path: Path = Field(default_factory=lambda: Path(DEFAULT_CONFIG_FILENAME))
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    api_base: str | None = None
    api_key: str | None = None

    # Server and tool state
    servers: list[ServerInfo] = Field(default_factory=list)
    tools: list[ToolInfo] = Field(default_factory=list)
    current_server: ServerInfo | None = None

    # UI state
    verbose_mode: bool = True
    confirm_tools: bool = True
    theme: str = "default"

    # Token storage
    token_backend: str | None = None

    # Session state
    session_id: str | None = None
    is_interactive: bool = False
    exit_requested: bool = False

    # Conversation state (for chat mode)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)

    # Additional context data (private attribute)
    _extra: dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Initialize managers if not provided."""
        if self.model_manager is None:
            from mcp_cli.model_management import ModelManager

            self.model_manager = ModelManager()

        # Set up model configuration
        if self.provider and self.model:
            self.model_manager.switch_model(self.provider, self.model)

    @classmethod
    def create(
        cls,
        tool_manager: ToolManager | None = None,
        config_path: Path | None = None,
        provider: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> ApplicationContext:
        """
        Factory method to create a context with common defaults.
        """
        context = cls(
            tool_manager=tool_manager,
            config_path=config_path or Path(DEFAULT_CONFIG_FILENAME),
            provider=provider or DEFAULT_PROVIDER,
            model=model or DEFAULT_MODEL,
            **kwargs,
        )
        return context

    async def initialize(self) -> None:
        """
        Initialize the context by loading servers, tools, etc.
        """
        if self.tool_manager:
            # Load servers
            self.servers = await self.tool_manager.get_server_info()

            # Load tools
            self.tools = await self.tool_manager.get_all_tools()

            # Set current server if only one
            if len(self.servers) == 1:
                self.current_server = self.servers[0]

    def get_current_server(self) -> ServerInfo | None:
        """Get the currently active server."""
        return self.current_server

    def set_current_server(self, server: ServerInfo) -> None:
        """Set the currently active server."""
        self.current_server = server

    def find_server(self, name: str) -> ServerInfo | None:
        """Find a server by name."""
        for server in self.servers:
            if server.name.lower() == name.lower():
                return server
        return None

    def find_tool(self, name: str) -> ToolInfo | None:
        """Find a tool by name."""
        for tool in self.tools:
            if tool.name == name or tool.fully_qualified_name == name:
                return tool
        return None

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from context (for compatibility with dict-based code).

        This method provides backwards compatibility for code expecting dict access.
        """
        # Check direct attributes first
        if hasattr(self, key):
            return getattr(self, key)

        # Check extra data
        return self._extra.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in context (for compatibility with dict-based code).
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self._extra[key] = value

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for backwards compatibility.

        This is a transitional method while migrating from dict-based context.
        """
        return {
            "tool_manager": self.tool_manager,
            "model_manager": self.model_manager,
            "config_path": str(self.config_path),
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "servers": self.servers,
            "tools": self.tools,
            "current_server": self.current_server,
            "verbose_mode": self.verbose_mode,
            "confirm_tools": self.confirm_tools,
            "theme": self.theme,
            "token_backend": self.token_backend,
            "session_id": self.session_id,
            "is_interactive": self.is_interactive,
            "exit_requested": self.exit_requested,
            "conversation_history": self.conversation_history,
            **self._extra,
        }

    def update_from_dict(self, data: dict[str, Any]) -> None:
        """
        Update context from a dictionary (for backwards compatibility).

        This is a transitional method while migrating from dict-based context.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._extra[key] = value

    def update(self, **kwargs) -> None:
        """
        Update context with keyword arguments.

        This method provides a simple way to update multiple context attributes.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._extra[key] = value

    # Conversation message helpers
    def add_message(self, message: ConversationMessage | dict[str, Any]) -> None:
        """Add a message to conversation history."""
        if isinstance(message, ConversationMessage):
            self.conversation_history.append(message.to_dict())
        else:
            self.conversation_history.append(message)

    def add_user_message(self, content: str) -> None:
        """Add a user message to conversation history."""
        self.add_message(ConversationMessage.user_message(content))

    def add_assistant_message(
        self, content: str | None = None, tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        """Add an assistant message to conversation history."""
        self.add_message(ConversationMessage.assistant_message(content, tool_calls))

    def add_system_message(self, content: str) -> None:
        """Add a system message to conversation history."""
        self.add_message(ConversationMessage.system_message(content))

    def add_tool_message(
        self, content: str, tool_call_id: str, name: str | None = None
    ) -> None:
        """Add a tool response message to conversation history."""
        self.add_message(ConversationMessage.tool_message(content, tool_call_id, name))

    def get_messages(self) -> list[ConversationMessage]:
        """Get conversation history as typed ConversationMessage objects."""
        return [ConversationMessage.from_dict(msg) for msg in self.conversation_history]

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()


class ContextManager:
    """
    Manager for application contexts.

    This provides a singleton-like pattern for managing the application context.
    """

    _instance: ContextManager | None = None
    _context: ApplicationContext | None = None

    def __new__(cls) -> ContextManager:
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(
        self,
        tool_manager: ToolManager | None = None,
        config_path: Path | None = None,
        **kwargs,
    ) -> ApplicationContext:
        """
        Initialize or get the application context.
        """
        if self._context is None:
            self._context = ApplicationContext.create(
                tool_manager=tool_manager, config_path=config_path, **kwargs
            )
        return self._context

    def get_context(self) -> ApplicationContext:
        """
        Get the current application context.

        Raises:
            RuntimeError: If context hasn't been initialized
        """
        if self._context is None:
            raise RuntimeError("Context not initialized. Call initialize() first.")
        return self._context

    def reset(self) -> None:
        """Reset the context (useful for testing)."""
        self._context = None


def get_context() -> ApplicationContext:
    """
    Convenience function to get the current application context.

    Returns:
        The current ApplicationContext

    Raises:
        RuntimeError: If context hasn't been initialized
    """
    manager = ContextManager()
    return manager.get_context()


def initialize_context(**kwargs) -> ApplicationContext:
    """
    Convenience function to initialize the application context.

    Args:
        **kwargs: Arguments to pass to ApplicationContext.create()

    Returns:
        The initialized ApplicationContext
    """
    manager = ContextManager()
    return manager.initialize(**kwargs)
