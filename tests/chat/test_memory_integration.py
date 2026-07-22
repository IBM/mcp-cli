# tests/chat/test_memory_integration.py
"""Integration tests for memory scope tool interception and system prompt injection."""

import pytest
from pathlib import Path

import chuk_ai_session_manager.guards.manager as _guard_mgr
from chuk_ai_session_manager.guards import (
    reset_tool_state,
    RuntimeLimits,
    ToolStateManager,
)

from mcp_cli.chat.tool_processor import ToolProcessor
from mcp_cli.memory.tools import _MEMORY_TOOL_NAMES
from mcp_cli.chat.response_models import ToolCall, FunctionCall
from mcp_cli.memory.models import MemoryScope
from mcp_cli.memory.store import MemoryScopeStore
from mcp_cli.memory.tools import get_memory_tools_as_dicts


@pytest.fixture(autouse=True)
def _fresh_tool_state():
    """Reset the global tool state singleton before each test."""
    reset_tool_state()
    _guard_mgr._tool_state = ToolStateManager(
        limits=RuntimeLimits(
            per_tool_cap=100,
            tool_budget_total=100,
            discovery_budget=50,
            execution_budget=50,
        )
    )
    yield
    reset_tool_state()


class DummyUIManager:
    def __init__(self):
        self.printed_calls = []
        self.is_streaming_response = False

    def print_tool_call(self, tool_name, raw_arguments):
        self.printed_calls.append((tool_name, raw_arguments))

    async def finish_tool_execution(self, result=None, success=True):
        pass

    async def do_confirm_tool_execution(self, tool_name, arguments):
        return True

    async def start_tool_execution(self, tool_name, arguments):
        pass


class DummyToolManager:
    async def stream_execute_tools(self, calls, **kwargs):
        # Should never be called for memory tools
        raise AssertionError("Memory tools should not be routed to ToolManager")
        yield  # make it a generator


class DummyContext:
    def __init__(self, memory_store=None):
        self.conversation_history = []
        self.stream_manager = None
        self.tool_manager = DummyToolManager()
        self.memory_store = memory_store
        self._system_prompt_dirty = False

    def inject_tool_message(self, message):
        self.conversation_history.append(message)


class TestMemoryToolInterception:
    """Verify memory tools are intercepted before guard checks."""

    @pytest.mark.asyncio
    async def test_remember_intercepted(self, tmp_path: Path):
        store = MemoryScopeStore(base_dir=tmp_path, workspace_dir="/test")
        context = DummyContext(memory_store=store)
        ui = DummyUIManager()
        processor = ToolProcessor(context, ui)

        tool_call = ToolCall(
            id="call_rem1",
            type="function",
            function=FunctionCall(
                name="remember",
                arguments='{"scope": "global", "key": "lang", "content": "Python"}',
            ),
        )
        await processor.process_tool_calls([tool_call])

        # Should have stored the memory
        entries = store.list_entries(MemoryScope.GLOBAL)
        assert len(entries) == 1
        assert entries[0].key == "lang"
        assert entries[0].content == "Python"

        # Should have marked system prompt dirty
        assert context._system_prompt_dirty is True

        # Should have added to conversation history (assistant + tool result)
        assert len(context.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_recall_intercepted(self, tmp_path: Path):
        store = MemoryScopeStore(base_dir=tmp_path, workspace_dir="/test")
        store.remember(MemoryScope.GLOBAL, "framework", "pytest")

        context = DummyContext(memory_store=store)
        ui = DummyUIManager()
        processor = ToolProcessor(context, ui)

        tool_call = ToolCall(
            id="call_rec1",
            type="function",
            function=FunctionCall(name="recall", arguments="{}"),
        )
        await processor.process_tool_calls([tool_call])

        # Should have recall result in history
        assert len(context.conversation_history) == 2
        tool_msg = context.conversation_history[1]
        assert "framework" in str(tool_msg.content)

        # recall should NOT dirty the system prompt
        assert context._system_prompt_dirty is False

    @pytest.mark.asyncio
    async def test_forget_intercepted(self, tmp_path: Path):
        store = MemoryScopeStore(base_dir=tmp_path, workspace_dir="/test")
        store.remember(MemoryScope.WORKSPACE, "temp", "remove me")

        context = DummyContext(memory_store=store)
        ui = DummyUIManager()
        processor = ToolProcessor(context, ui)

        tool_call = ToolCall(
            id="call_fgt1",
            type="function",
            function=FunctionCall(
                name="forget",
                arguments='{"scope": "workspace", "key": "temp"}',
            ),
        )
        await processor.process_tool_calls([tool_call])

        # Should have removed the memory
        assert store.list_entries(MemoryScope.WORKSPACE) == []
        assert context._system_prompt_dirty is True

    @pytest.mark.asyncio
    async def test_no_store_returns_error(self):
        context = DummyContext(memory_store=None)
        ui = DummyUIManager()
        processor = ToolProcessor(context, ui)

        tool_call = ToolCall(
            id="call_rem2",
            type="function",
            function=FunctionCall(
                name="remember",
                arguments='{"scope": "global", "key": "k", "content": "v"}',
            ),
        )
        await processor.process_tool_calls([tool_call])

        # Should have error in history
        assert len(context.conversation_history) == 2
        tool_msg = context.conversation_history[1]
        assert "not available" in str(tool_msg.content).lower()


class TestMemoryToolNames:
    def test_names_constant(self):
        assert _MEMORY_TOOL_NAMES == {"remember", "recall", "forget"}


class TestMemoryToolsInjection:
    def test_tool_dicts_format(self):
        tools = get_memory_tools_as_dicts()
        assert len(tools) == 3

        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]
