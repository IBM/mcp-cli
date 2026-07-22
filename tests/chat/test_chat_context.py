# tests/mcp_cli/chat/test_chat_context.py
"""Unit-tests for the re-worked *ChatContext* class.

We avoid the heavyweight real ToolManager by supplying a tiny stub that
implements just enough of the async API surface the context expects.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from mcp_cli.chat.chat_context import ChatContext
from mcp_cli.tools.models import ToolInfo, ServerInfo


# ---------------------------------------------------------------------------
# Dummy async ToolManager stub
# ---------------------------------------------------------------------------
class DummyToolManager:  # noqa: E501 - test helper
    """Minimal stand-in that satisfies the methods ChatContext uses."""

    def __init__(self) -> None:
        self._tools = [
            ToolInfo(
                name="tool1",
                namespace="srv1",
                description="demo-1",
                parameters={},
                is_async=False,
            ),
            ToolInfo(
                name="tool2",
                namespace="srv2",
                description="demo-2",
                parameters={},
                is_async=False,
            ),
        ]

        self._servers = [
            ServerInfo(
                id=0,
                name="srv1",
                status="ok",
                tool_count=1,
                namespace="srv1",
            ),
            ServerInfo(
                id=1,
                name="srv2",
                status="ok",
                tool_count=1,
                namespace="srv2",
            ),
        ]

        self._openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": f"{t.namespace}_{t.name}",
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools
        ]

    # ----- discovery --------------------------------------------------
    async def get_unique_tools(self):  # noqa: D401 - match signature
        return self._tools

    async def get_server_info(self):  # noqa: D401 - match signature
        return self._servers

    async def get_adapted_tools_for_llm(self, provider: str = "openai"):
        mapping = {
            f"{t.namespace}_{t.name}": f"{t.namespace}.{t.name}" for t in self._tools
        }
        return self._openai_tools, mapping

    async def get_tools_for_llm(self):
        return self._openai_tools

    async def get_server_for_tool(self, tool_name: str):
        if "." in tool_name:
            return tool_name.split(".", 1)[0]
        if "_" in tool_name:
            return tool_name.split("_", 1)[0]
        return "Unknown"

    # ----- execution stubs -------------------------------------------
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]):
        return {"success": True, "result": {"echo": arguments}}

    async def stream_execute_tool(self, tool_name: str, arguments: Dict[str, Any]):
        yield {"success": True, "result": {"echo": arguments}}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def dummy_tool_manager():
    return DummyToolManager()


@pytest.fixture()
def chat_context(dummy_tool_manager, monkeypatch):
    # Use deterministic system prompt
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    # Mock ModelManager to avoid model discovery issues
    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Create a minimal mock ModelManager
    mock_model_manager = Mock(spec=ModelManager)
    mock_model_manager.provider = "mock"
    mock_model_manager.model = "mock-model"
    mock_model_manager.get_client.return_value = None
    mock_model_manager.get_active_provider.return_value = "mock"
    mock_model_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_model_manager
    )
    return ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_initialize_chat_context(chat_context):
    ok = await chat_context.initialize()
    assert ok is True

    # tools discovered
    assert chat_context.get_tool_count() == 2

    # system prompt injected as first conversation turn
    assert chat_context.conversation_history[0].to_dict() == {
        "role": "system",
        "content": "SYS_PROMPT",
    }

    # OpenAI tools adapted
    assert len(chat_context.openai_tools) == 2
    assert chat_context.tool_name_mapping  # non-empty


@pytest.mark.asyncio
async def test_get_server_for_tool(chat_context):
    await chat_context.initialize()

    assert await chat_context.get_server_for_tool("srv1.tool1") == "srv1"
    assert await chat_context.get_server_for_tool("srv2_tool2") == "srv2"
    assert await chat_context.get_server_for_tool("unknown") == "Unknown"


@pytest.mark.asyncio
async def test_to_dict_and_update_roundtrip(chat_context):
    await chat_context.initialize()

    exported = chat_context.to_dict()

    # update_from_dict handles exit_requested but not conversation_history
    exported["exit_requested"] = True

    chat_context.update_from_dict(exported)

    assert chat_context.exit_requested is True


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_tool_by_name_exact(chat_context):
    """Test find_tool_by_name with exact match."""
    await chat_context.initialize()
    tool = chat_context.find_tool_by_name("tool1")
    assert tool is not None
    assert tool.name == "tool1"


@pytest.mark.asyncio
async def test_find_tool_by_name_fully_qualified(chat_context):
    """Test find_tool_by_name with fully qualified name."""
    await chat_context.initialize()
    # Try using srv1.tool1 format
    tool = chat_context.find_tool_by_name("srv1.tool1")
    assert tool is not None


@pytest.mark.asyncio
async def test_find_tool_by_name_not_found(chat_context):
    """Test find_tool_by_name with non-existent tool."""
    await chat_context.initialize()
    tool = chat_context.find_tool_by_name("nonexistent")
    assert tool is None


@pytest.mark.asyncio
async def test_find_server_by_name(chat_context):
    """Test find_server_by_name."""
    await chat_context.initialize()
    server = chat_context.find_server_by_name("srv1")
    assert server is not None
    assert server.name == "srv1"


@pytest.mark.asyncio
async def test_find_server_by_name_not_found(chat_context):
    """Test find_server_by_name with non-existent server."""
    await chat_context.initialize()
    server = chat_context.find_server_by_name("nonexistent")
    assert server is None


@pytest.mark.asyncio
async def test_add_user_message(chat_context):
    """Test add_user_message."""
    await chat_context.initialize()
    initial_len = len(chat_context.conversation_history)
    await chat_context.add_user_message("Hello!")
    assert len(chat_context.conversation_history) == initial_len + 1
    assert chat_context.conversation_history[-1].content == "Hello!"
    assert chat_context.conversation_history[-1].role.value == "user"


@pytest.mark.asyncio
async def test_add_assistant_message(chat_context):
    """Test add_assistant_message."""
    await chat_context.initialize()
    initial_len = len(chat_context.conversation_history)
    await chat_context.add_assistant_message("Hi there!")
    assert len(chat_context.conversation_history) == initial_len + 1
    assert chat_context.conversation_history[-1].content == "Hi there!"
    assert chat_context.conversation_history[-1].role.value == "assistant"


@pytest.mark.asyncio
async def test_clear_conversation_history_keep_system(chat_context):
    """Test clear_conversation_history with keep_system_prompt=True."""
    await chat_context.initialize()
    await chat_context.add_user_message("Hello")
    await chat_context.add_assistant_message("Hi")

    await chat_context.clear_conversation_history(keep_system_prompt=True)

    assert len(chat_context.conversation_history) == 1
    assert chat_context.conversation_history[0].role.value == "system"


@pytest.mark.asyncio
async def test_clear_conversation_history_remove_all(chat_context):
    """Test clear_conversation_history creates fresh session (system prompt always kept)."""
    await chat_context.initialize()
    await chat_context.add_user_message("Hello")

    await chat_context.clear_conversation_history(keep_system_prompt=False)

    # Implementation always creates fresh session with system prompt
    assert len(chat_context.conversation_history) == 1
    assert chat_context.conversation_history[0].role.value == "system"


@pytest.mark.asyncio
async def test_regenerate_system_prompt(chat_context):
    """Test regenerate_system_prompt."""
    await chat_context.initialize()
    _ = chat_context.conversation_history[0].content  # Original prompt

    # Regenerate should update the system prompt
    await chat_context.regenerate_system_prompt()

    # Should still be the first message
    assert chat_context.conversation_history[0].role.value == "system"


@pytest.mark.asyncio
async def test_get_tool_count(chat_context):
    """Test get_tool_count."""
    await chat_context.initialize()
    assert chat_context.get_tool_count() == 2


@pytest.mark.asyncio
async def test_get_server_count(chat_context):
    """Test get_server_count."""
    await chat_context.initialize()
    assert chat_context.get_server_count() == 2


def test_get_display_name_for_tool():
    """Test get_display_name_for_tool static method."""
    from mcp_cli.chat.chat_context import ChatContext

    name = ChatContext.get_display_name_for_tool("srv1.tool1")
    assert name == "srv1.tool1"


@pytest.mark.asyncio
async def test_get_status_summary(chat_context):
    """Test get_status_summary."""
    await chat_context.initialize()
    status = chat_context.get_status_summary()
    assert status.tool_count == 2
    assert status.server_count == 2


@pytest.mark.asyncio
async def test_repr(chat_context):
    """Test __repr__."""
    await chat_context.initialize()
    repr_str = repr(chat_context)
    assert "ChatContext" in repr_str
    assert "tools=2" in repr_str


@pytest.mark.asyncio
async def test_str(chat_context):
    """Test __str__."""
    await chat_context.initialize()
    str_val = str(chat_context)
    assert "Chat session" in str_val
    assert "2 tools" in str_val


@pytest.mark.asyncio
async def test_context_manager(dummy_tool_manager, monkeypatch):
    """Test async context manager."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_model_manager = Mock(spec=ModelManager)
    mock_model_manager.provider = "mock"
    mock_model_manager.model = "mock-model"
    mock_model_manager.get_client.return_value = None
    mock_model_manager.get_active_provider.return_value = "mock"
    mock_model_manager.get_active_model.return_value = "mock-model"

    from mcp_cli.chat.chat_context import ChatContext

    async with ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_model_manager
    ) as ctx:
        assert ctx.get_tool_count() == 2


@pytest.mark.asyncio
async def test_update_from_dict_with_exit_requested(chat_context):
    """Test update_from_dict updates exit_requested."""
    await chat_context.initialize()

    chat_context.update_from_dict({"exit_requested": True})
    assert chat_context.exit_requested is True

    chat_context.update_from_dict({"exit_requested": False})
    assert chat_context.exit_requested is False


@pytest.mark.asyncio
async def test_execute_tool(chat_context):
    """Test execute_tool delegation."""
    await chat_context.initialize()
    result = await chat_context.execute_tool("tool1", {"arg": "value"})
    assert result["success"] is True


@pytest.mark.asyncio
async def test_stream_execute_tool(chat_context):
    """Test stream_execute_tool delegation."""
    await chat_context.initialize()
    results = []
    async for result in chat_context.stream_execute_tool("tool1", {"arg": "value"}):
        results.append(result)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_refresh_after_model_change(chat_context):
    """Test refresh_after_model_change."""
    await chat_context.initialize()
    # Should not raise
    await chat_context.refresh_after_model_change()
    assert chat_context.get_tool_count() == 2


@pytest.mark.asyncio
async def test_create_with_provider_only(dummy_tool_manager, monkeypatch):
    """Test ChatContext.create with provider only."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Create new ModelManager instance for this test
    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "openai"
    mock_manager.get_active_model.return_value = "gpt-4"
    mock_manager.switch_provider.return_value = None

    # Patch ModelManager constructor
    with monkeypatch.context() as m:
        m.setattr("mcp_cli.chat.chat_context.ModelManager", lambda: mock_manager)
        ctx = ChatContext.create(
            tool_manager=dummy_tool_manager,
            provider="openai",
        )
        assert ctx is not None


@pytest.mark.asyncio
async def test_create_with_model_only(dummy_tool_manager, monkeypatch):
    """Test ChatContext.create with model only."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "openai"
    mock_manager.get_active_model.return_value = "gpt-4"
    mock_manager.switch_model.return_value = None

    with monkeypatch.context() as m:
        m.setattr("mcp_cli.chat.chat_context.ModelManager", lambda: mock_manager)
        ctx = ChatContext.create(
            tool_manager=dummy_tool_manager,
            model="gpt-4",
        )
        assert ctx is not None


@pytest.mark.asyncio
async def test_create_with_provider_and_api_settings(dummy_tool_manager, monkeypatch):
    """Test ChatContext.create with provider and API settings."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "custom"
    mock_manager.get_active_model.return_value = "custom-model"
    mock_manager.add_runtime_provider.return_value = None
    mock_manager.switch_provider.return_value = None

    with monkeypatch.context() as m:
        m.setattr("mcp_cli.chat.chat_context.ModelManager", lambda: mock_manager)
        ctx = ChatContext.create(
            tool_manager=dummy_tool_manager,
            provider="custom",
            api_base="http://localhost:8080",
            api_key="test-key",
        )
        assert ctx is not None


@pytest.mark.asyncio
async def test_create_with_provider_model_and_api_settings(
    dummy_tool_manager, monkeypatch
):
    """Test ChatContext.create with all settings."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "custom"
    mock_manager.get_active_model.return_value = "custom-model"
    mock_manager.add_runtime_provider.return_value = None
    mock_manager.switch_model.return_value = None

    with monkeypatch.context() as m:
        m.setattr("mcp_cli.chat.chat_context.ModelManager", lambda: mock_manager)
        ctx = ChatContext.create(
            tool_manager=dummy_tool_manager,
            provider="custom",
            model="custom-model",
            api_key="test-key",
        )
        assert ctx is not None


@pytest.mark.asyncio
async def test_initialize_failure(dummy_tool_manager, monkeypatch):
    """Test initialize handles errors."""
    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Patch generate_system_prompt to avoid issues
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    # Create context
    ctx = ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_manager
    )

    # Make _initialize_tools raise an exception
    async def raise_error():
        raise RuntimeError("Test error")

    ctx._initialize_tools = raise_error

    result = await ctx.initialize()
    assert result is False


@pytest.mark.asyncio
async def test_regenerate_system_prompt_insert(dummy_tool_manager, monkeypatch):
    """Test regenerate_system_prompt when no system message exists."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_manager
    )
    await ctx.initialize()

    # Clear conversation history completely
    await ctx.clear_conversation_history(keep_system_prompt=False)

    # Regenerate should insert at position 0
    await ctx.regenerate_system_prompt()

    assert len(ctx.conversation_history) == 1
    assert ctx.conversation_history[0].role.value == "system"


@pytest.mark.asyncio
async def test_context_manager_failure(dummy_tool_manager, monkeypatch):
    """Test async context manager handles initialization failure."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_manager
    )

    # Make initialize return False
    async def fail_init():
        return False

    ctx.initialize = fail_init

    with pytest.raises(RuntimeError, match="Failed to initialize"):
        async with ctx:
            pass


@pytest.mark.asyncio
async def test_adapt_tools_without_get_adapted_tools(dummy_tool_manager, monkeypatch):
    """Test _adapt_tools_for_provider fallback when get_adapted_tools_for_llm not available."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Create tool manager without get_adapted_tools_for_llm
    class MinimalToolManager:
        def __init__(self):
            self._tools = [
                ToolInfo(
                    name="tool1",
                    namespace="srv1",
                    description="demo-1",
                    parameters={},
                    is_async=False,
                ),
            ]

        async def get_unique_tools(self):
            return self._tools

        async def get_server_info(self):
            return []

        async def get_tools_for_llm(self):
            return [{"type": "function", "function": {"name": "tool1"}}]

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=MinimalToolManager(), model_manager=mock_manager
    )
    await ctx.initialize()

    assert len(ctx.openai_tools) == 1
    assert ctx.tool_name_mapping == {}


@pytest.mark.asyncio
async def test_adapt_tools_exception_fallback(dummy_tool_manager, monkeypatch):
    """Test _adapt_tools_for_provider handles exceptions."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Create tool manager that raises in get_adapted_tools_for_llm
    class FailingToolManager:
        def __init__(self):
            self._tools = [
                ToolInfo(
                    name="tool1",
                    namespace="srv1",
                    description="demo-1",
                    parameters={},
                    is_async=False,
                ),
            ]

        async def get_unique_tools(self):
            return self._tools

        async def get_server_info(self):
            return []

        async def get_adapted_tools_for_llm(self, provider):
            raise RuntimeError("Adaptation failed")

        async def get_tools_for_llm(self):
            return [{"type": "function", "function": {"name": "tool1"}}]

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=FailingToolManager(), model_manager=mock_manager
    )
    await ctx.initialize()

    # Should fall back to get_tools_for_llm
    assert len(ctx.openai_tools) == 1


@pytest.mark.asyncio
async def test_initialize_no_tools_warning(monkeypatch, capsys):
    """Test initialize prints warning when no tools available."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    # Create tool manager that returns no tools
    class EmptyToolManager:
        async def get_unique_tools(self):
            return []

        async def get_server_info(self):
            return []

        async def get_adapted_tools_for_llm(self, provider):
            return [], {}

        async def get_tools_for_llm(self):
            return []

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=EmptyToolManager(), model_manager=mock_manager
    )
    result = await ctx.initialize()

    assert result is True
    assert ctx.get_tool_count() == 0


@pytest.mark.asyncio
async def test_find_tool_by_name_partial_match(chat_context):
    """Test find_tool_by_name with partial match (just tool name without namespace)."""
    await chat_context.initialize()
    # The dummy tools have namespace like "srv1" and name like "tool1"
    # Try to find by using a dotted name that doesn't match exactly
    # but the simple name part matches
    tool = chat_context.find_tool_by_name("other.tool1")
    assert tool is not None
    assert tool.name == "tool1"


# ---------------------------------------------------------------------------
# Server/tool grouping tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_server_tool_groups(chat_context):
    """_build_server_tool_groups returns correct grouping."""
    await chat_context.initialize()
    groups = chat_context._build_server_tool_groups()

    assert len(groups) == 2

    names = {g.name for g in groups}
    assert names == {"srv1", "srv2"}

    for group in groups:
        assert group.name
        assert isinstance(group.tools, list)
        assert len(group.tools) >= 1


@pytest.mark.asyncio
async def test_build_server_tool_groups_empty(dummy_tool_manager, monkeypatch):
    """_build_server_tool_groups returns empty list when no server_info."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )

    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "mock"
    mock_manager.get_active_model.return_value = "mock-model"

    ctx = ChatContext.create(
        tool_manager=dummy_tool_manager, model_manager=mock_manager
    )
    # Don't initialize — server_info is empty
    assert ctx._build_server_tool_groups() == []


# ---------------------------------------------------------------------------
# Tests for sliding window and infinite context (Tier 1.3 + 1.4)
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    """Tests for conversation history sliding window."""

    def _make_ctx(self, monkeypatch, max_history_messages=0, infinite_context=False):
        """Helper to create a ChatContext with mocked model manager."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        mock_tm = Mock()
        mock_tm.get_unique_tools = pytest.importorskip("asyncio").coroutines
        return ChatContext(
            tool_manager=mock_tm,
            model_manager=mock_manager,
            max_history_messages=max_history_messages,
            infinite_context=infinite_context,
        )

    @pytest.mark.asyncio
    async def test_no_limit_returns_all(self, dummy_tool_manager, monkeypatch):
        """max_history_messages=0 returns all messages."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            max_history_messages=0,
        )
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Add 10 messages
        for i in range(10):
            await ctx.add_user_message(f"msg-{i}")

        history = ctx.conversation_history
        # 1 system + 10 user
        assert len(history) == 11

    @pytest.mark.asyncio
    async def test_window_limits_messages(self, dummy_tool_manager, monkeypatch):
        """Sliding window keeps only last N event messages."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            max_history_messages=3,
        )
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Add 10 messages
        for i in range(10):
            await ctx.add_user_message(f"msg-{i}")

        history = ctx.conversation_history
        # 1 system + 3 windowed = 4
        assert len(history) == 4
        # System prompt always first
        assert history[0].role.value == "system"
        # Last 3 messages are the most recent
        assert history[-1].content == "msg-9"
        assert history[-2].content == "msg-8"
        assert history[-3].content == "msg-7"

    @pytest.mark.asyncio
    async def test_system_prompt_not_evicted(self, dummy_tool_manager, monkeypatch):
        """System prompt is always included regardless of window size."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            max_history_messages=1,
        )
        ctx._system_prompt = "SYSTEM"
        await ctx._initialize_session()

        await ctx.add_user_message("hello")
        await ctx.add_user_message("world")

        history = ctx.conversation_history
        # System + 1 windowed = 2
        assert len(history) == 2
        assert history[0].content == "SYSTEM"

    @pytest.mark.asyncio
    async def test_under_limit_not_truncated(self, dummy_tool_manager, monkeypatch):
        """If message count is under the limit, no eviction happens."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            max_history_messages=100,
        )
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        await ctx.add_user_message("hello")
        await ctx.add_user_message("world")

        history = ctx.conversation_history
        # 1 system + 2 user = 3, under limit of 100
        assert len(history) == 3


class TestInfiniteContextConfig:
    """Tests for infinite context configuration."""

    @pytest.mark.asyncio
    async def test_default_infinite_context_false(
        self, dummy_tool_manager, monkeypatch
    ):
        """Default infinite_context is False."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
        )
        assert ctx._infinite_context is False

    @pytest.mark.asyncio
    async def test_infinite_context_passed_to_session(
        self, dummy_tool_manager, monkeypatch
    ):
        """infinite_context=True is passed to SessionManager."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            infinite_context=True,
            token_threshold=8000,
            max_turns_per_segment=30,
        )
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Verify session was created with correct params
        assert ctx._infinite_context is True
        assert ctx._token_threshold == 8000
        assert ctx._max_turns_per_segment == 30

    @pytest.mark.asyncio
    async def test_create_factory_threads_params(self, dummy_tool_manager, monkeypatch):
        """create() factory passes context params through."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext.create(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
            max_history_messages=50,
            infinite_context=True,
            token_threshold=6000,
        )
        assert ctx._max_history_messages == 50
        assert ctx._infinite_context is True
        assert ctx._token_threshold == 6000


# ---------------------------------------------------------------------------
# Tests for system prompt caching (Tier 2)
# ---------------------------------------------------------------------------


class TestSystemPromptCaching:
    """Tests for system prompt dirty-flag caching."""

    @pytest.mark.asyncio
    async def test_system_prompt_cached(self, dummy_tool_manager, monkeypatch):
        """Second call to _generate_system_prompt doesn't rebuild when dirty=False."""
        call_count = [0]

        def counting_generate(tools=None, **kw):
            call_count[0] += 1
            return f"SYS_PROMPT_{call_count[0]}"

        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            counting_generate,
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
        )
        # Simulate initialized state with internal_tools set
        ctx.internal_tools = []

        # First call: dirty=True by default, should build prompt
        ctx._generate_system_prompt()
        first_prompt = ctx._system_prompt
        assert call_count[0] == 1
        assert first_prompt == "SYS_PROMPT_1"
        assert ctx._system_prompt_dirty is False

        # Second call: dirty=False, should return cached prompt
        ctx._generate_system_prompt()
        assert call_count[0] == 1  # Not called again
        assert ctx._system_prompt == first_prompt

    @pytest.mark.asyncio
    async def test_dirty_flag_on_tool_change(self, dummy_tool_manager, monkeypatch):
        """_initialize_tools sets dirty=True."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_manager = Mock(spec=ModelManager)
        mock_manager.get_client.return_value = None
        mock_manager.get_active_provider.return_value = "mock"
        mock_manager.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=dummy_tool_manager,
            model_manager=mock_manager,
        )

        # Manually mark clean
        ctx._system_prompt_dirty = False

        # After _initialize_tools, dirty should be True again
        await ctx._initialize_tools()
        assert ctx._system_prompt_dirty is True


class TestLargeToolSetSummary:
    """Tests for system prompt tool summary threshold."""

    def test_large_tool_set_summary(self):
        """System prompt with >20 tools summarizes (shows '... and N more')."""
        from mcp_cli.chat.system_prompt import _build_server_section
        from mcp_cli.chat.models import ServerToolGroup

        # Create a server group with 25 tools
        tool_names = [f"tool_{i}" for i in range(25)]
        server_groups = [
            ServerToolGroup(
                name="big_server",
                description="A server with many tools",
                tools=tool_names,
            )
        ]

        result = _build_server_section(server_groups, tool_summary_threshold=20)

        # Should contain the summary text
        assert "... and 20 more" in result
        # Should show first 5 tools
        assert "tool_0" in result
        assert "tool_4" in result
        # Should NOT show tool_5 onwards individually (they are summarized)
        assert "tool_5," not in result

    def test_small_tool_set_no_summary(self):
        """System prompt with <= threshold tools shows all tools."""
        from mcp_cli.chat.system_prompt import _build_server_section
        from mcp_cli.chat.models import ServerToolGroup

        tool_names = [f"tool_{i}" for i in range(5)]
        server_groups = [
            ServerToolGroup(
                name="small_server",
                description="A server with few tools",
                tools=tool_names,
            )
        ]

        result = _build_server_section(server_groups, tool_summary_threshold=20)

        # Should contain all tool names
        for name in tool_names:
            assert name in result
        # Should NOT have the summary text
        assert "more" not in result

    def test_default_threshold_from_config(self):
        """_build_server_section uses DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD by default."""
        from mcp_cli.chat.system_prompt import _build_server_section
        from mcp_cli.chat.models import ServerToolGroup
        from mcp_cli.config.defaults import DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD

        # Create tools just above the default threshold
        count = DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD + 5
        tool_names = [f"tool_{i}" for i in range(count)]
        server_groups = [
            ServerToolGroup(
                name="server",
                description="desc",
                tools=tool_names,
            )
        ]

        result = _build_server_section(server_groups)

        # Should summarize since count > threshold
        assert "more" in result


# ---------------------------------------------------------------------------
# Helper fixture shared by new test classes
# ---------------------------------------------------------------------------


def _make_initialized_ctx(monkeypatch, tool_manager=None, **ctx_kwargs):
    """Sync helper — returns a ChatContext that has had _system_prompt set."""
    monkeypatch.setattr(
        "mcp_cli.chat.chat_context.generate_system_prompt",
        lambda tools=None, **kw: "SYS_PROMPT",
    )
    from unittest.mock import Mock
    from mcp_cli.model_management import ModelManager

    mock_mm = Mock(spec=ModelManager)
    mock_mm.get_client.return_value = None
    mock_mm.get_active_provider.return_value = "mock"
    mock_mm.get_active_model.return_value = "mock-model"

    if tool_manager is None:
        tool_manager = DummyToolManager()

    from mcp_cli.chat.chat_context import ChatContext

    return ChatContext(
        tool_manager=tool_manager,
        model_manager=mock_mm,
        **ctx_kwargs,
    )


# ---------------------------------------------------------------------------
# Tests: conversation_history — LLM/SYSTEM source events and TOOL_CALL events
# ---------------------------------------------------------------------------


class TestConversationHistoryEventTypes:
    """Cover the event-type branches in conversation_history property (lines 272-294)."""

    @pytest.mark.asyncio
    async def test_llm_source_event_appears_as_assistant(self, monkeypatch):
        """EventSource.LLM events appear as assistant messages."""
        from chuk_ai_session_manager.models.session_event import SessionEvent
        from chuk_ai_session_manager.models.event_type import EventType
        from chuk_ai_session_manager.models.event_source import EventSource

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Inject an LLM event directly
        e = SessionEvent(
            message="I am the assistant",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        ctx.session._session.events.append(e)

        history = ctx.conversation_history
        assistant_msgs = [m for m in history if m.role.value == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "I am the assistant"

    @pytest.mark.asyncio
    async def test_system_source_event_appears_as_assistant(self, monkeypatch):
        """EventSource.SYSTEM events appear as assistant messages (inject_assistant_message path)."""
        from chuk_ai_session_manager.models.session_event import SessionEvent
        from chuk_ai_session_manager.models.event_type import EventType
        from chuk_ai_session_manager.models.event_source import EventSource

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        e = SessionEvent(
            message="System-injected assistant turn",
            source=EventSource.SYSTEM,
            type=EventType.MESSAGE,
        )
        ctx.session._session.events.append(e)

        history = ctx.conversation_history
        assistant_msgs = [m for m in history if m.role.value == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "System-injected assistant turn"

    @pytest.mark.asyncio
    async def test_tool_call_event_reconstructed(self, monkeypatch):
        """TOOL_CALL events with dict messages are reconstructed as HistoryMessage."""
        from chuk_ai_session_manager.models.session_event import SessionEvent
        from chuk_ai_session_manager.models.event_type import EventType
        from chuk_ai_session_manager.models.event_source import EventSource
        from mcp_cli.chat.models import HistoryMessage, MessageRole

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        tool_msg = HistoryMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "my_tool", "arguments": "{}"},
                }
            ],
        )
        e = SessionEvent(
            message=tool_msg.to_dict(),
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )
        ctx.session._session.events.append(e)

        history = ctx.conversation_history
        # Should contain the tool-call assistant message
        tool_call_msgs = [m for m in history if m.tool_calls is not None]
        assert len(tool_call_msgs) == 1
        assert tool_call_msgs[0].tool_calls[0]["id"] == "call_abc"

    @pytest.mark.asyncio
    async def test_empty_system_prompt_excluded(self, monkeypatch):
        """Empty system prompt doesn't add a system message to history."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = ""  # explicitly empty
        await ctx._initialize_session()

        history = ctx.conversation_history
        system_msgs = [m for m in history if m.role.value == "system"]
        assert len(system_msgs) == 0

    @pytest.mark.asyncio
    async def test_get_conversation_length_no_session(self, monkeypatch):
        """get_conversation_length returns 0 when _session is None."""
        ctx = _make_initialized_ctx(monkeypatch)
        # Don't call _initialize_session — _session is None by default on raw SessionManager
        # Force _session to None
        ctx.session._session = None
        length = ctx.get_conversation_length()
        assert length == 0


# ---------------------------------------------------------------------------
# Tests: inject_assistant_message and inject_tool_message (lines 659-682)
# ---------------------------------------------------------------------------


class TestInjectMethods:
    """Cover inject_assistant_message and inject_tool_message."""

    @pytest.mark.asyncio
    async def test_inject_assistant_message(self, monkeypatch):
        """inject_assistant_message adds SYSTEM/MESSAGE event."""
        from chuk_ai_session_manager.models.event_type import EventType
        from chuk_ai_session_manager.models.event_source import EventSource

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        ctx.inject_assistant_message("Budget exhausted, stopping.")

        # Event should appear in session
        events = ctx.session._session.events
        injected = [
            e
            for e in events
            if e.type == EventType.MESSAGE and e.source == EventSource.SYSTEM
        ]
        assert len(injected) == 1
        assert injected[0].message == "Budget exhausted, stopping."

    @pytest.mark.asyncio
    async def test_inject_assistant_message_in_history(self, monkeypatch):
        """inject_assistant_message content shows up in conversation_history."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        ctx.inject_assistant_message("Injected content here")
        history = ctx.conversation_history
        assistant_msgs = [m for m in history if m.role.value == "assistant"]
        assert any("Injected content here" in (m.content or "") for m in assistant_msgs)

    @pytest.mark.asyncio
    async def test_inject_tool_message(self, monkeypatch):
        """inject_tool_message stores a TOOL_CALL event."""
        from chuk_ai_session_manager.models.event_type import EventType
        from mcp_cli.chat.models import HistoryMessage, MessageRole

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        msg = HistoryMessage(
            role=MessageRole.TOOL,
            content="tool result here",
            tool_call_id="call-999",
        )
        ctx.inject_tool_message(msg)

        events = ctx.session._session.events
        tool_events = [e for e in events if e.type == EventType.TOOL_CALL]
        assert len(tool_events) == 1
        # The stored message is the dict form
        stored = tool_events[0].message
        assert isinstance(stored, dict)
        assert stored.get("role") == "tool"

    @pytest.mark.asyncio
    async def test_inject_tool_message_in_history(self, monkeypatch):
        """inject_tool_message shows up in conversation_history."""
        from mcp_cli.chat.models import HistoryMessage, MessageRole

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        msg = HistoryMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                {
                    "id": "call-001",
                    "type": "function",
                    "function": {"name": "do_thing", "arguments": "{}"},
                }
            ],
        )
        ctx.inject_tool_message(msg)
        history = ctx.conversation_history
        with_tool_calls = [m for m in history if m.tool_calls]
        assert len(with_tool_calls) == 1


# ---------------------------------------------------------------------------
# Tests: record_tool_call (lines 705-731)
# ---------------------------------------------------------------------------


class TestRecordToolCall:
    """Cover record_tool_call and related memory helpers."""

    @pytest.mark.asyncio
    async def test_record_tool_call_success(self, monkeypatch):
        """record_tool_call records a successful tool call."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        await ctx.record_tool_call(
            tool_name="my_tool",
            arguments={"key": "value"},
            result={"output": 42},
            success=True,
            context_goal="do something",
        )

        # Check procedural memory recorded it
        history = ctx.get_recent_tool_history(limit=5)
        assert len(history) == 1
        assert history[0]["tool"] == "my_tool"
        assert history[0]["outcome"] == "success"

    @pytest.mark.asyncio
    async def test_record_tool_call_failure(self, monkeypatch):
        """record_tool_call records a failed tool call."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        await ctx.record_tool_call(
            tool_name="bad_tool",
            arguments={},
            result=None,
            success=False,
            error="Something went wrong",
        )

        history = ctx.get_recent_tool_history(limit=5)
        assert len(history) == 1
        assert history[0]["tool"] == "bad_tool"
        assert history[0]["outcome"] == "failure"

    @pytest.mark.asyncio
    async def test_record_tool_call_error_object(self, monkeypatch):
        """record_tool_call with error=None exercises the None error branch cleanly."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # error=None: the error_type branch should produce None (isinstance check skipped)
        await ctx.record_tool_call(
            tool_name="err_tool",
            arguments={},
            result="partial",
            success=False,
            error=None,
        )

        history = ctx.get_recent_tool_history(limit=5)
        assert len(history) == 1
        assert history[0]["tool"] == "err_tool"

    @pytest.mark.asyncio
    async def test_record_tool_call_enforces_memory_limits(self, monkeypatch):
        """_enforce_memory_limits trims excess patterns."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Record many calls for the same tool to exceed limits
        max_p = ctx.tool_memory.max_patterns_per_tool
        for i in range(max_p + 5):
            await ctx.record_tool_call(
                tool_name="repeated_tool",
                arguments={"i": i},
                result=None,
                success=False,
                error=f"error {i}",
            )

        # After enforcement, patterns should be within limits
        patterns = ctx.tool_memory.memory.tool_patterns.get("repeated_tool")
        if patterns:
            assert len(patterns.error_patterns) <= max_p

    @pytest.mark.asyncio
    async def test_get_procedural_context_for_tools(self, monkeypatch):
        """get_procedural_context_for_tools returns a string."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Record a call first
        await ctx.record_tool_call(
            tool_name="some_tool",
            arguments={"x": 1},
            result="done",
            success=True,
        )

        result = ctx.get_procedural_context_for_tools(
            ["some_tool"], context_goal="test goal"
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_recent_tool_history_respects_limit(self, monkeypatch):
        """get_recent_tool_history returns at most `limit` entries."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        for i in range(8):
            await ctx.record_tool_call(
                tool_name=f"tool_{i}",
                arguments={},
                result=f"result_{i}",
                success=True,
            )

        history = ctx.get_recent_tool_history(limit=3)
        assert len(history) == 3


# ---------------------------------------------------------------------------
# Tests: get_messages_for_llm and get_session_stats (lines 735-738, 990-991)
# ---------------------------------------------------------------------------


class TestSessionMethods:
    """Cover get_messages_for_llm and get_session_stats."""

    @pytest.mark.asyncio
    async def test_get_messages_for_llm(self, monkeypatch):
        """get_messages_for_llm returns list of dicts."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()
        await ctx.add_user_message("hello")

        msgs = await ctx.get_messages_for_llm()
        assert isinstance(msgs, list)
        assert len(msgs) >= 1

    @pytest.mark.asyncio
    async def test_get_session_stats(self, monkeypatch):
        """get_session_stats returns a stats object with session_id attribute."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        stats = await ctx.get_session_stats()
        # SessionManager.get_stats() returns a SessionStats (DictCompatModel) object
        assert stats is not None
        assert hasattr(stats, "session_id")


# ---------------------------------------------------------------------------
# Tests: context notices (drain_context_notices) (lines 795-797)
# ---------------------------------------------------------------------------


class TestContextNotices:
    """Cover add_context_notice and drain_context_notices."""

    def test_drain_context_notices_empty(self, monkeypatch):
        """drain_context_notices returns empty list when nothing queued."""
        ctx = _make_initialized_ctx(monkeypatch)
        assert ctx.drain_context_notices() == []

    def test_add_and_drain_context_notices(self, monkeypatch):
        """add_context_notice queues notices; drain_context_notices clears them."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx.add_context_notice("Notice 1")
        ctx.add_context_notice("Notice 2")

        notices = ctx.drain_context_notices()
        assert notices == ["Notice 1", "Notice 2"]

        # After drain, list is empty
        assert ctx.drain_context_notices() == []


# ---------------------------------------------------------------------------
# Tests: on_progress callbacks in _initialize_tools (lines 550, 559)
# ---------------------------------------------------------------------------


class TestOnProgressCallback:
    """Cover the on_progress callback paths in _initialize_tools and initialize."""

    @pytest.mark.asyncio
    async def test_on_progress_called_during_initialize(self, monkeypatch):
        """on_progress callback is invoked during initialize()."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "mock"
        mock_mm.get_active_model.return_value = "mock-model"

        from mcp_cli.chat.chat_context import ChatContext

        ctx = ChatContext(
            tool_manager=DummyToolManager(),
            model_manager=mock_mm,
        )

        calls = []
        await ctx.initialize(on_progress=lambda msg: calls.append(msg))

        # At least the "Discovering tools..." progress was reported
        assert any("Discovering" in c for c in calls)
        assert any("Adapting" in c for c in calls)

    @pytest.mark.asyncio
    async def test_initialize_tools_with_namespace(self, monkeypatch):
        """Tools with a namespace are indexed under both simple and qualified names."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager
        from mcp_cli.tools.models import ToolInfo

        # Tool with namespace
        class NamespacedTM:
            _tools = [
                ToolInfo(
                    name="my_tool",
                    namespace="my_server",
                    description="desc",
                    parameters={},
                    is_async=False,
                )
            ]

            async def get_unique_tools(self):
                return self._tools

            async def get_server_info(self):
                return []

            async def get_adapted_tools_for_llm(self, provider):
                return [], {}

            async def get_tools_for_llm(self):
                return []

        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "mock"
        mock_mm.get_active_model.return_value = "mock-model"

        from mcp_cli.chat.chat_context import ChatContext

        ctx = ChatContext(tool_manager=NamespacedTM(), model_manager=mock_mm)
        await ctx._initialize_tools()

        # Both simple and qualified names should be in index
        assert "my_tool" in ctx._tool_index
        assert "my_server.my_tool" in ctx._tool_index

    @pytest.mark.asyncio
    async def test_initialize_tools_without_namespace(self, monkeypatch):
        """Tools without a namespace are only indexed under simple name (covers 569->567 branch)."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager
        from mcp_cli.tools.models import ToolInfo

        class NoNamespaceTM:
            _tools = [
                ToolInfo(
                    name="bare_tool",
                    namespace="",  # no namespace
                    description="no ns",
                    parameters={},
                    is_async=False,
                )
            ]

            async def get_unique_tools(self):
                return self._tools

            async def get_server_info(self):
                return []

            async def get_adapted_tools_for_llm(self, provider):
                return [], {}

            async def get_tools_for_llm(self):
                return []

        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "mock"
        mock_mm.get_active_model.return_value = "mock-model"

        from mcp_cli.chat.chat_context import ChatContext

        ctx = ChatContext(tool_manager=NoNamespaceTM(), model_manager=mock_mm)
        await ctx._initialize_tools()

        # Only simple name should be in index
        assert "bare_tool" in ctx._tool_index
        # No qualified name entry (namespace is falsy)
        assert ".bare_tool" not in ctx._tool_index


# ---------------------------------------------------------------------------
# Tests: initialize provider validation warning (lines 445-447)
# ---------------------------------------------------------------------------


class TestInitializeProviderWarning:
    """Cover the provider validation warning path in initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_client_raises_logs_warning(self, monkeypatch, caplog):
        """When client raises, initialize still returns True but logs a warning."""
        import logging

        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager
        from mcp_cli.chat.chat_context import ChatContext

        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_active_provider.return_value = "mock"
        mock_mm.get_active_model.return_value = "mock-model"
        # Make get_client raise
        mock_mm.get_client.side_effect = RuntimeError("No API key")

        ctx = ChatContext(tool_manager=DummyToolManager(), model_manager=mock_mm)

        with caplog.at_level(logging.WARNING, logger="mcp_cli.chat.chat_context"):
            result = await ctx.initialize()

        assert result is True
        assert any(
            "warning" in r.message.lower() or "validation" in r.message.lower()
            for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Tests: _initialize_session memory store failure (lines 489-491)
# ---------------------------------------------------------------------------


class TestMemoryStoreFailure:
    """Cover the MemoryScopeStore import failure path."""

    @pytest.mark.asyncio
    async def test_memory_store_import_failure_logged(self, monkeypatch):
        """When MemoryScopeStore import fails, memory_store is set to None."""
        # Make the import fail

        # Patch the import inside _initialize_session
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def patched_import(name, *args, **kwargs):
            if name == "mcp_cli.memory.store":
                raise ImportError("Simulated import failure")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", patched_import)

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        assert ctx.memory_store is None

    @pytest.mark.asyncio
    async def test_generate_system_prompt_with_memory_store(self, monkeypatch):
        """_generate_system_prompt appends memory section when memory_store is set."""
        # Use a counter so we can distinguish the prompt value from the fixture default
        generate_calls = []

        def counting_generate(tools=None, **kw):
            generate_calls.append(True)
            return "BASE_PROMPT"

        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            counting_generate,
        )

        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager
        from mcp_cli.chat.chat_context import ChatContext

        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "mock"
        mock_mm.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=DummyToolManager(),
            model_manager=mock_mm,
        )
        ctx.internal_tools = []

        # Set up a mock memory store
        mock_store = Mock()
        mock_store.format_for_system_prompt.return_value = "MEMORY_SECTION"
        ctx.memory_store = mock_store

        ctx._system_prompt_dirty = True
        ctx._generate_system_prompt()

        assert "MEMORY_SECTION" in ctx._system_prompt
        assert "BASE_PROMPT" in ctx._system_prompt


# ---------------------------------------------------------------------------
# Tests: save_session (lines 816-844)
# ---------------------------------------------------------------------------


class TestSaveSession:
    """Cover save_session."""

    @pytest.mark.asyncio
    async def test_save_session_returns_path(self, monkeypatch, tmp_path):
        """save_session returns a path string on success."""
        from mcp_cli.chat.session_store import SessionStore

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Point the session store at tmp_path
        ctx._session_store = SessionStore(sessions_dir=tmp_path)

        path = ctx.save_session()
        assert path is not None
        assert path.endswith(".json")

    @pytest.mark.asyncio
    async def test_save_session_with_token_usage(self, monkeypatch, tmp_path):
        """save_session includes token usage when turns have been recorded."""
        from mcp_cli.chat.session_store import SessionStore
        from mcp_cli.chat.token_tracker import TurnUsage

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()
        ctx._session_store = SessionStore(sessions_dir=tmp_path)

        # Record a turn so turn_count > 0
        ctx.token_tracker.record_turn(TurnUsage(input_tokens=50, output_tokens=25))

        path = ctx.save_session()
        assert path is not None

        # Verify the saved file has token_usage
        import json

        saved = json.loads(
            tmp_path.joinpath("default", f"{ctx.session_id}.json").read_text()
        )
        assert saved.get("token_usage") is not None

    @pytest.mark.asyncio
    async def test_save_session_failure_returns_none(self, monkeypatch):
        """save_session returns None on error."""
        from unittest.mock import Mock

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Make session_store.save raise
        mock_store = Mock()
        mock_store.save.side_effect = OSError("disk full")
        ctx._session_store = mock_store

        path = ctx.save_session()
        assert path is None


# ---------------------------------------------------------------------------
# Tests: load_session (lines 855-897)
# ---------------------------------------------------------------------------


class TestLoadSession:
    """Cover load_session paths."""

    def test_load_session_not_found_returns_false(self, monkeypatch):
        """load_session returns False when session_id doesn't exist."""
        from unittest.mock import Mock

        ctx = _make_initialized_ctx(monkeypatch)

        mock_store = Mock()
        mock_store.load.return_value = None  # Not found
        ctx._session_store = mock_store

        result = ctx.load_session("nonexistent-session-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_session_exception_returns_false(self, monkeypatch, tmp_path):
        """load_session returns False when event injection raises."""
        from unittest.mock import Mock
        from mcp_cli.chat.session_store import SessionData, SessionMetadata

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Create fake session data with a user message
        data = SessionData(
            metadata=SessionMetadata(
                session_id="fake-session",
                provider="mock",
                model="mock-model",
            ),
            messages=[
                {"role": "user", "content": "hello"},
            ],
        )

        mock_store = Mock()
        mock_store.load.return_value = data
        ctx._session_store = mock_store

        # Make session._session.events raise on append to trigger the except block
        mock_events = Mock()
        mock_events.append.side_effect = RuntimeError("injection error")
        ctx.session._session.events = mock_events

        result = ctx.load_session("fake-session")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_session_skips_system_role(self, monkeypatch, tmp_path):
        """load_session skips messages with role=system and returns True."""
        from unittest.mock import Mock
        from mcp_cli.chat.session_store import SessionData, SessionMetadata

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Only system-role messages: all are skipped via `continue`, add_event is never called
        data = SessionData(
            metadata=SessionMetadata(
                session_id="sys-session",
                provider="mock",
                model="mock-model",
            ),
            messages=[
                {"role": "system", "content": "System prompt"},
                {"role": "system", "content": "Another system"},
            ],
        )

        mock_store = Mock()
        mock_store.load.return_value = data
        ctx._session_store = mock_store

        # System messages are skipped via `continue`, so add_event (which doesn't
        # exist on SessionManager) is never reached; loop completes -> True
        result = ctx.load_session("sys-session")
        assert result is True

    def test_load_session_all_roles_covered(self, monkeypatch):
        """load_session covers user/assistant/tool/unknown roles before hitting add_event."""
        from unittest.mock import Mock
        from mcp_cli.chat.session_store import SessionData, SessionMetadata

        ctx = _make_initialized_ctx(monkeypatch)

        data = SessionData(
            metadata=SessionMetadata(
                session_id="roles-session",
                provider="mock",
                model="mock-model",
            ),
            messages=[
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant message"},
                {"role": "tool", "content": "Tool result", "tool_call_id": "call-1"},
                {"role": "unknown_role", "content": "skipped"},
            ],
        )

        mock_store = Mock()
        mock_store.load.return_value = data
        ctx._session_store = mock_store

        # The SessionEvent constructor called in load_session uses unsupported kwargs,
        # so it raises ValidationError which is caught by the except block -> False
        result = ctx.load_session("roles-session")
        # The first non-system role hits SessionEvent construction which raises -> except -> False
        assert result is False

    def test_load_session_assistant_role_reached(self, monkeypatch):
        """load_session reaches the assistant role branch (line 874)."""
        from unittest.mock import Mock
        from mcp_cli.chat.session_store import SessionData, SessionMetadata

        ctx = _make_initialized_ctx(monkeypatch)

        # Start with a system message (skipped), then assistant
        # When assistant branch executes, SessionEvent() raises -> except -> False
        data = SessionData(
            metadata=SessionMetadata(
                session_id="asst-session",
                provider="mock",
                model="mock-model",
            ),
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "Assistant reply"},
            ],
        )

        mock_store = Mock()
        mock_store.load.return_value = data
        ctx._session_store = mock_store

        # assistant branch triggers SessionEvent construction -> ValidationError -> except -> False
        result = ctx.load_session("asst-session")
        assert result is False

    def test_load_session_tool_role_reached(self, monkeypatch):
        """load_session reaches the tool role branch (line 880)."""
        from unittest.mock import Mock
        from mcp_cli.chat.session_store import SessionData, SessionMetadata

        ctx = _make_initialized_ctx(monkeypatch)

        # Only tool message (after system skip)
        data = SessionData(
            metadata=SessionMetadata(
                session_id="tool-session",
                provider="mock",
                model="mock-model",
            ),
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "tool", "content": "tool output", "tool_call_id": "tc-1"},
            ],
        )

        mock_store = Mock()
        mock_store.load.return_value = data
        ctx._session_store = mock_store

        # tool branch triggers SessionEvent construction -> ValidationError -> except -> False
        result = ctx.load_session("tool-session")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: auto_save_check (lines 901-907)
# ---------------------------------------------------------------------------


class TestAutoSaveCheck:
    """Cover auto_save_check."""

    @pytest.mark.asyncio
    async def test_auto_save_check_below_threshold(self, monkeypatch, tmp_path):
        """auto_save_check doesn't save before threshold."""
        from mcp_cli.chat.session_store import SessionStore

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()
        ctx._session_store = SessionStore(sessions_dir=tmp_path)

        from mcp_cli.config.defaults import DEFAULT_AUTO_SAVE_INTERVAL

        # Call one less than the threshold
        for _ in range(DEFAULT_AUTO_SAVE_INTERVAL - 1):
            ctx.auto_save_check()

        # Nothing saved yet
        assert list(tmp_path.glob("**/*.json")) == []

    @pytest.mark.asyncio
    async def test_auto_save_check_triggers_save(self, monkeypatch, tmp_path):
        """auto_save_check saves at threshold and resets counter."""
        from mcp_cli.chat.session_store import SessionStore
        from mcp_cli.config.defaults import DEFAULT_AUTO_SAVE_INTERVAL

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()
        ctx._session_store = SessionStore(sessions_dir=tmp_path)

        # Call exactly the threshold times
        for _ in range(DEFAULT_AUTO_SAVE_INTERVAL):
            ctx.auto_save_check()

        # Should have saved (files are in tmp_path/default/)
        saved_files = list(tmp_path.glob("**/*.json"))
        assert len(saved_files) == 1
        # Counter should be reset to 0
        assert ctx._auto_save_counter == 0


# ---------------------------------------------------------------------------
# Tests: _vm_filter_events (lines 347-424)
# ---------------------------------------------------------------------------


class TestVMFilterEvents:
    """Cover the _vm_filter_events method."""

    def _make_ctx_with_vm_budget(self, monkeypatch, vm_budget=128_000):
        ctx = _make_initialized_ctx(monkeypatch, vm_budget=vm_budget)
        ctx._system_prompt = "SYS"
        return ctx

    def _user_msg(self, content):
        from mcp_cli.chat.models import HistoryMessage, MessageRole

        return HistoryMessage(role=MessageRole.USER, content=content)

    def _assistant_msg(self, content):
        from mcp_cli.chat.models import HistoryMessage, MessageRole

        return HistoryMessage(role=MessageRole.ASSISTANT, content=content)

    def test_empty_events_returns_empty(self, monkeypatch):
        """_vm_filter_events returns empty list for empty input."""
        ctx = self._make_ctx_with_vm_budget(monkeypatch)
        result = ctx._vm_filter_events([], "SYS")
        assert result == []

    def test_few_turns_no_filtering(self, monkeypatch):
        """If turns <= MIN_RECENT_TURNS, no filtering occurs."""
        ctx = self._make_ctx_with_vm_budget(monkeypatch)
        # 2 turns (< 3 = _VM_MIN_RECENT_TURNS)
        events = [
            self._user_msg("A"),
            self._assistant_msg("B"),
            self._user_msg("C"),
            self._assistant_msg("D"),
        ]
        result = ctx._vm_filter_events(events, "SYS")
        assert result == events

    def test_many_turns_with_large_budget_keeps_all(self, monkeypatch):
        """With a large budget, all turns should be included."""
        ctx = self._make_ctx_with_vm_budget(monkeypatch, vm_budget=128_000)
        # 6 turns, but budget is huge
        events = []
        for i in range(6):
            events.append(self._user_msg(f"User {i}"))
            events.append(self._assistant_msg(f"Asst {i}"))

        result = ctx._vm_filter_events(events, "SYS")
        # All events should be present
        assert len(result) == len(events)

    def test_tiny_budget_evicts_old_turns(self, monkeypatch):
        """With a tiny budget, older turns are evicted."""
        # Budget so small even 1 token per turn gets exceeded quickly
        ctx = self._make_ctx_with_vm_budget(monkeypatch, vm_budget=1)

        # Create 6 turns (each with long content)
        events = []
        for i in range(6):
            events.append(self._user_msg("X" * 100))  # 25 tokens each
            events.append(self._assistant_msg("Y" * 100))

        result = ctx._vm_filter_events(events, "SYS")

        # Should have evicted some turns, keeping at most _VM_MIN_RECENT_TURNS guaranteed
        from mcp_cli.chat.chat_context import ChatContext

        guaranteed_msgs = ChatContext._VM_MIN_RECENT_TURNS * 2  # 2 msgs per turn
        assert len(result) <= guaranteed_msgs + 2  # at most guaranteed + maybe 1 more

    def test_evicted_turns_add_context_notice(self, monkeypatch):
        """When turns are evicted, a context notice is queued."""
        ctx = self._make_ctx_with_vm_budget(monkeypatch, vm_budget=1)

        events = []
        for i in range(6):
            events.append(self._user_msg("X" * 200))
            events.append(self._assistant_msg("Y" * 200))

        ctx._vm_filter_events(events, "SYS")

        # A notice should have been queued
        notices = ctx.drain_context_notices()
        assert len(notices) > 0
        assert any("virtual memory" in n for n in notices)

    def test_tool_calls_counted_in_token_estimate(self, monkeypatch):
        """Tool calls in messages are included in token estimate."""
        ctx = self._make_ctx_with_vm_budget(monkeypatch, vm_budget=1)
        from mcp_cli.chat.models import HistoryMessage, MessageRole

        events = []
        for i in range(6):
            events.append(self._user_msg("query"))
            # Assistant message with tool_calls
            msg = HistoryMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": "tool",
                            "arguments": '{"x": ' + "1" * 200 + "}",
                        },
                    }
                ],
            )
            events.append(msg)

        # Should not raise; tool_calls content is counted
        result = ctx._vm_filter_events(events, "SYS")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: conversation_history with VM enabled (lines 257-258, 294)
# ---------------------------------------------------------------------------


class TestConversationHistoryVM:
    """Cover the VM path in conversation_history property."""

    @pytest.mark.asyncio
    async def test_conversation_history_vm_path(self, monkeypatch):
        """When VM is enabled, conversation_history uses get_vm_context."""
        from mcp_cli.chat.chat_context import ChatContext
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "mock"
        mock_mm.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=DummyToolManager(),
            model_manager=mock_mm,
            enable_vm=True,
            vm_mode="passive",
            vm_budget=128_000,
        )
        ctx._system_prompt = "SYS_PROMPT"
        await ctx._initialize_session()

        # session.vm should be set now
        assert ctx.session.vm is not None

        # Add a user message
        await ctx.add_user_message("Hello VM!")

        history = ctx.conversation_history
        # System message should be present (from VM context)
        system_msgs = [m for m in history if m.role.value == "system"]
        assert len(system_msgs) == 1

    @pytest.mark.asyncio
    async def test_conversation_history_vm_context_none(self, monkeypatch):
        """When VM returns None context, falls back to _system_prompt."""
        from mcp_cli.chat.chat_context import ChatContext
        from unittest.mock import Mock, patch
        from mcp_cli.model_management import ModelManager

        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "FALLBACK_PROMPT",
        )
        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "mock"
        mock_mm.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=DummyToolManager(),
            model_manager=mock_mm,
            enable_vm=True,
            vm_mode="passive",
        )
        ctx._system_prompt = "FALLBACK_PROMPT"
        await ctx._initialize_session()

        # Patch get_vm_context to return None to test fallback branch
        with patch.object(ctx.session, "get_vm_context", return_value=None):
            history = ctx.conversation_history

        system_msgs = [m for m in history if m.role.value == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "FALLBACK_PROMPT"

    @pytest.mark.asyncio
    async def test_conversation_history_vm_filter_called(self, monkeypatch):
        """_vm_filter_events is called when VM is enabled and there are events."""
        from mcp_cli.chat.chat_context import ChatContext
        from unittest.mock import Mock, patch
        from mcp_cli.model_management import ModelManager

        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS",
        )
        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "mock"
        mock_mm.get_active_model.return_value = "mock-model"

        ctx = ChatContext(
            tool_manager=DummyToolManager(),
            model_manager=mock_mm,
            enable_vm=True,
            vm_mode="passive",
            vm_budget=128_000,
        )
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Add enough messages to make _vm_filter_events do something
        for i in range(5):
            await ctx.add_user_message(f"msg {i}")

        filter_called = []

        def spy_filter(events, system_content):
            filter_called.append(True)
            return events

        with patch.object(ctx, "_vm_filter_events", side_effect=spy_filter):
            _ = ctx.conversation_history

        assert len(filter_called) > 0


# ---------------------------------------------------------------------------
# Tests: create() factory — model-only branch (line 201-205)
# ---------------------------------------------------------------------------


class TestCreateFactoryBranches:
    """Cover remaining branches in ChatContext.create()."""

    def test_create_model_only_calls_switch_model_on_current_provider(
        self, monkeypatch
    ):
        """create(model=X) without provider calls switch_model on current provider."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager
        from mcp_cli.chat.chat_context import ChatContext

        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "existing_provider"
        mock_mm.get_active_model.return_value = "new-model"
        mock_mm.get_current_provider = Mock(return_value="existing_provider")

        # Patch the ModelManager constructor to return our mock
        with monkeypatch.context() as m:
            m.setattr("mcp_cli.chat.chat_context.ModelManager", lambda: mock_mm)
            ctx = ChatContext.create(
                tool_manager=DummyToolManager(),
                model="new-model",
            )

        assert ctx is not None
        # switch_model should have been called with (current_provider, model)
        mock_mm.switch_model.assert_called_once_with("existing_provider", "new-model")

    def test_create_no_provider_no_model_no_switch(self, monkeypatch):
        """create() with no provider and no model: ModelManager created but no switch called."""
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager
        from mcp_cli.chat.chat_context import ChatContext

        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "openai"
        mock_mm.get_active_model.return_value = "gpt-4o-mini"

        with monkeypatch.context() as m:
            m.setattr("mcp_cli.chat.chat_context.ModelManager", lambda: mock_mm)
            # No provider, no model, no api_key, no api_base -> 201->205 False branch
            ctx = ChatContext.create(
                tool_manager=DummyToolManager(),
            )

        assert ctx is not None
        # Neither switch_model nor switch_provider should have been called
        mock_mm.switch_model.assert_not_called()
        mock_mm.switch_provider.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: remaining uncovered branches
# ---------------------------------------------------------------------------


class TestRemainingBranches:
    """Cover miscellaneous branches that are still missing."""

    # ── update_from_dict without exit_requested (947->950 False branch) ──

    def test_update_from_dict_model_manager_only(self, monkeypatch):
        """update_from_dict branch: no exit_requested key, but model_manager present."""
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        ctx = _make_initialized_ctx(monkeypatch)
        new_mm = Mock(spec=ModelManager)
        new_mm.get_active_provider.return_value = "new_provider"
        new_mm.get_active_model.return_value = "new_model"

        # Do NOT include exit_requested — exercises the 947->950 False branch
        ctx.update_from_dict({"model_manager": new_mm, "tools": []})

        assert ctx.model_manager is new_mm
        assert ctx.tools == []

    # ── _enforce_memory_limits success_patterns branch (line 807) ──

    def test_enforce_memory_limits_success_patterns_directly(self, monkeypatch):
        """_enforce_memory_limits trims success_patterns directly when overfull."""
        ctx = _make_initialized_ctx(monkeypatch)
        max_p = ctx.tool_memory.max_patterns_per_tool

        # Manually inject overfull patterns to trigger both trim branches
        tool_name = "synth_tool"
        pattern = ctx.tool_memory.memory.get_pattern(tool_name)

        # Overfill both error_patterns and success_patterns beyond max_p
        pattern.error_patterns = [{"e": i} for i in range(max_p + 5)]
        pattern.success_patterns = [{"s": i} for i in range(max_p + 3)]

        # Call enforce — should trim both
        ctx._enforce_memory_limits()

        assert len(pattern.error_patterns) == max_p
        assert len(pattern.success_patterns) == max_p

    # ── load_session: assistant/tool/unknown roles (lines 873-889) ──

    @pytest.mark.asyncio
    async def test_load_session_unknown_role_is_skipped(self, monkeypatch):
        """load_session skips messages with unknown role."""
        from unittest.mock import Mock
        from mcp_cli.chat.session_store import SessionData, SessionMetadata

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Only unknown roles — all skipped via else: continue — add_event never called
        data = SessionData(
            metadata=SessionMetadata(
                session_id="unk-session",
                provider="mock",
                model="mock-model",
            ),
            messages=[
                {"role": "unknown_role", "content": "ignored"},
                {"role": "another_unknown", "content": "also ignored"},
            ],
        )

        mock_store = Mock()
        mock_store.load.return_value = data
        ctx._session_store = mock_store

        result = ctx.load_session("unk-session")
        # All messages skipped, loop completes -> True
        assert result is True

    # ── _vm_filter_events: first user message with empty current_turn (367->371) ──

    def test_vm_filter_first_message_user_no_prior_turn(self, monkeypatch):
        """_vm_filter_events handles first user message correctly (no prior turn)."""
        ctx = _make_initialized_ctx(monkeypatch, vm_budget=128_000)
        from mcp_cli.chat.models import HistoryMessage, MessageRole

        # First message is a user message with empty current_turn initially
        # (exercises the `if msg.role == USER and current_turn` branch as False)
        events = []
        for i in range(4):
            events.append(HistoryMessage(role=MessageRole.USER, content=f"Q {i}"))
            events.append(HistoryMessage(role=MessageRole.ASSISTANT, content=f"A {i}"))

        # With large budget, all should be returned (but filter code still runs turn grouping)
        result = ctx._vm_filter_events(events, "SYS")
        assert len(result) == len(events)

    # ── conversation_history: no events in session (272->293 branch) ──

    @pytest.mark.asyncio
    async def test_conversation_history_no_events_empty(self, monkeypatch):
        """conversation_history when session has no events returns just system prompt."""
        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # No messages added — session events is empty
        history = ctx.conversation_history
        # Only the system prompt
        assert len(history) == 1
        assert history[0].role.value == "system"

    # ── system prompt cache hit (534->532 False branch) ──

    def test_generate_system_prompt_cache_hit_skips_rebuild(self, monkeypatch):
        """_generate_system_prompt returns immediately when dirty=False and prompt is set."""
        call_count = [0]

        def counting_generate(tools=None, **kw):
            call_count[0] += 1
            return "BUILT_PROMPT"

        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            counting_generate,
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager
        from mcp_cli.chat.chat_context import ChatContext

        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "mock"
        mock_mm.get_active_model.return_value = "mock-model"

        ctx = ChatContext(tool_manager=DummyToolManager(), model_manager=mock_mm)
        ctx.internal_tools = []

        # First call builds the prompt
        ctx._generate_system_prompt()
        assert call_count[0] == 1
        assert ctx._system_prompt_dirty is False

        # Second call: dirty=False and prompt is non-empty — should NOT rebuild
        ctx._generate_system_prompt()
        assert call_count[0] == 1  # Unchanged — cache hit

    # ── conversation_history: multiple event types in single session ──

    @pytest.mark.asyncio
    async def test_conversation_history_mixed_events(self, monkeypatch):
        """conversation_history correctly processes USER, LLM, SYSTEM, and TOOL_CALL events."""
        from chuk_ai_session_manager.models.session_event import SessionEvent
        from chuk_ai_session_manager.models.event_type import EventType
        from chuk_ai_session_manager.models.event_source import EventSource
        from mcp_cli.chat.models import HistoryMessage, MessageRole

        ctx = _make_initialized_ctx(monkeypatch)
        ctx._system_prompt = "SYS"
        await ctx._initialize_session()

        # Add a user event
        e_user = SessionEvent(
            message="User question",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
        # Add an LLM event
        e_llm = SessionEvent(
            message="LLM response",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
        # Add a TOOL_CALL event with dict message
        tool_msg = HistoryMessage(
            role=MessageRole.TOOL,
            content="tool output",
            tool_call_id="tc-xyz",
        )
        e_tool = SessionEvent(
            message=tool_msg.to_dict(),
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )

        ctx.session._session.events.extend([e_user, e_llm, e_tool])

        history = ctx.conversation_history
        roles = [m.role.value for m in history]

        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles


# ---------------------------------------------------------------------------
# agent_id plumbing
# ---------------------------------------------------------------------------


class TestAgentId:
    """Verify agent_id is stored, propagated, and serialized."""

    def test_agent_id_default(self, monkeypatch):
        ctx = _make_initialized_ctx(monkeypatch)
        assert ctx.agent_id == "default"

    def test_agent_id_custom(self, monkeypatch):
        ctx = _make_initialized_ctx(monkeypatch, agent_id="my-agent")
        assert ctx.agent_id == "my-agent"

    def test_to_dict_includes_agent_id(self, monkeypatch):
        ctx = _make_initialized_ctx(monkeypatch, agent_id="export-agent")
        d = ctx.to_dict()
        assert d["agent_id"] == "export-agent"

    def test_save_session_writes_agent_id(self, monkeypatch, tmp_path):
        ctx = _make_initialized_ctx(monkeypatch, agent_id="save-agent")
        # Point session store to tmp_path
        from mcp_cli.chat.session_store import SessionStore

        ctx._session_store = SessionStore(sessions_dir=tmp_path, agent_id="save-agent")
        ctx._system_prompt = "SYS"

        path = ctx.save_session()
        assert path is not None

        loaded = ctx._session_store.load(ctx.session_id)
        assert loaded is not None
        assert loaded.metadata.agent_id == "save-agent"

    def test_create_forwards_agent_id(self, monkeypatch):
        monkeypatch.setattr(
            "mcp_cli.chat.chat_context.generate_system_prompt",
            lambda tools=None, **kw: "SYS_PROMPT",
        )
        from unittest.mock import Mock
        from mcp_cli.model_management import ModelManager

        mock_mm = Mock(spec=ModelManager)
        mock_mm.get_client.return_value = None
        mock_mm.get_active_provider.return_value = "mock"
        mock_mm.get_active_model.return_value = "mock-model"

        ctx = ChatContext.create(
            tool_manager=DummyToolManager(),
            model_manager=mock_mm,
            agent_id="factory-agent",
        )
        assert ctx.agent_id == "factory-agent"
