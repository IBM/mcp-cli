# tests/memory/test_tools.py
"""Tests for memory scope tool definitions and handler."""

import pytest
from pathlib import Path

from mcp_cli.memory.models import MemoryScope
from mcp_cli.memory.store import MemoryScopeStore
from mcp_cli.memory.tools import (
    _MEMORY_TOOL_NAMES,
    get_memory_tools_as_dicts,
    handle_memory_tool,
)


@pytest.fixture
def store(tmp_path: Path) -> MemoryScopeStore:
    return MemoryScopeStore(base_dir=tmp_path, workspace_dir="/test")


class TestToolDefinitions:
    def test_tool_names_frozenset(self):
        assert _MEMORY_TOOL_NAMES == {"remember", "recall", "forget"}

    def test_get_memory_tools_as_dicts(self):
        tools = get_memory_tools_as_dicts()
        assert len(tools) == 3

        names = {t["function"]["name"] for t in tools}
        assert names == {"remember", "recall", "forget"}

        # All should be function type
        for tool in tools:
            assert tool["type"] == "function"
            assert "parameters" in tool["function"]

    def test_remember_tool_has_required_params(self):
        tools = get_memory_tools_as_dicts()
        remember = next(t for t in tools if t["function"]["name"] == "remember")
        params = remember["function"]["parameters"]
        assert set(params["required"]) == {"scope", "key", "content"}

    def test_recall_tool_no_required_params(self):
        tools = get_memory_tools_as_dicts()
        recall = next(t for t in tools if t["function"]["name"] == "recall")
        params = recall["function"]["parameters"]
        assert params["required"] == []

    def test_forget_tool_has_required_params(self):
        tools = get_memory_tools_as_dicts()
        forget = next(t for t in tools if t["function"]["name"] == "forget")
        params = forget["function"]["parameters"]
        assert set(params["required"]) == {"scope", "key"}


class TestHandleMemoryTool:
    @pytest.mark.asyncio
    async def test_remember(self, store: MemoryScopeStore):
        result = await handle_memory_tool(
            store, "remember", {"scope": "global", "key": "lang", "content": "Python"}
        )
        assert "Remembered 'lang'" in result
        assert "global" in result

    @pytest.mark.asyncio
    async def test_recall_all(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "a", "alpha")
        result = await handle_memory_tool(store, "recall", {})
        assert "[a]" in result
        assert "alpha" in result

    @pytest.mark.asyncio
    async def test_recall_empty(self, store: MemoryScopeStore):
        result = await handle_memory_tool(store, "recall", {})
        assert "No memories found" in result

    @pytest.mark.asyncio
    async def test_recall_by_key(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "target", "value")
        store.remember(MemoryScope.GLOBAL, "other", "nope")

        result = await handle_memory_tool(
            store, "recall", {"scope": "global", "key": "target"}
        )
        assert "target" in result
        assert "value" in result

    @pytest.mark.asyncio
    async def test_recall_by_query(self, store: MemoryScopeStore):
        store.remember(MemoryScope.WORKSPACE, "framework", "FastAPI is used here")

        result = await handle_memory_tool(
            store, "recall", {"scope": "workspace", "query": "fastapi"}
        )
        assert "framework" in result

    @pytest.mark.asyncio
    async def test_forget_existing(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "temp", "delete")
        result = await handle_memory_tool(
            store, "forget", {"scope": "global", "key": "temp"}
        )
        assert "Forgot 'temp'" in result

    @pytest.mark.asyncio
    async def test_forget_nonexistent(self, store: MemoryScopeStore):
        result = await handle_memory_tool(
            store, "forget", {"scope": "global", "key": "nope"}
        )
        assert "No memory with key" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self, store: MemoryScopeStore):
        result = await handle_memory_tool(store, "unknown_op", {})
        assert "Unknown memory tool" in result

    @pytest.mark.asyncio
    async def test_error_handling(self, store: MemoryScopeStore):
        # Missing required args
        result = await handle_memory_tool(store, "remember", {})
        assert "error" in result.lower()
