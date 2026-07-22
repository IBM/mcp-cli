# tests/utils/test_serialization.py
"""Tests for unwrap_tool_result and to_serializable in mcp_cli.utils.serialization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mcp_cli.utils.serialization import to_serializable, unwrap_tool_result


class TestUnwrapToolResult:
    """Tests for the MCP dict unwrapping path."""

    def test_success_returns_content(self):
        result = unwrap_tool_result({"isError": False, "content": "hello"})
        assert result == "hello"

    def test_error_raises_with_content_fallback(self):
        with pytest.raises(RuntimeError, match="boom"):
            unwrap_tool_result({"isError": True, "content": "boom"})

    def test_error_prefers_error_field_over_content(self):
        with pytest.raises(RuntimeError, match="specific error"):
            unwrap_tool_result(
                {"isError": True, "error": "specific error", "content": "fallback"}
            )

    def test_error_without_content_uses_default_message(self):
        with pytest.raises(RuntimeError, match="Tool returned an error"):
            unwrap_tool_result({"isError": True, "content": ""})

    def test_passthrough_plain_values(self):
        assert unwrap_tool_result("plain") == "plain"
        assert unwrap_tool_result(42) == 42
        assert unwrap_tool_result(None) is None

    def test_passthrough_dict_without_iserror(self):
        d = {"foo": "bar"}
        assert unwrap_tool_result(d) == {"foo": "bar"}


class TestUnwrapObjectWrapper:
    """Tests for the object-wrapper (ToolExecutionResult) unwrapping path."""

    def test_object_wrapper_success(self):
        wrapper = SimpleNamespace(success=True, result={"data": 42})
        assert unwrap_tool_result(wrapper) == {"data": 42}

    def test_object_wrapper_failure_raises(self):
        wrapper = SimpleNamespace(success=False, result=None, error="msg")
        with pytest.raises(RuntimeError, match="msg"):
            unwrap_tool_result(wrapper)

    def test_object_wrapper_failure_default_message(self):
        wrapper = SimpleNamespace(success=False, result=None)
        with pytest.raises(RuntimeError, match="Unknown tool error"):
            unwrap_tool_result(wrapper)

    def test_nested_object_wrappers(self):
        inner = SimpleNamespace(success=True, result="payload")
        outer = SimpleNamespace(success=True, result=inner)
        assert unwrap_tool_result(outer) == "payload"

    def test_max_depth_exceeded_raises(self):
        # Build 3 levels of nesting, but allow only 2
        level2 = SimpleNamespace(success=True, result="deep")
        level1 = SimpleNamespace(success=True, result=level2)
        level0 = SimpleNamespace(success=True, result=level1)
        with pytest.raises(RuntimeError, match="Exceeded max unwrap depth"):
            unwrap_tool_result(level0, max_depth=2)

    def test_iserror_list_content_stringified(self):
        """isError=True with list content should stringify the list."""
        obj = {"isError": True, "content": ["err1", "err2"]}
        with pytest.raises(RuntimeError, match=r"\['err1', 'err2'\]"):
            unwrap_tool_result(obj)


class TestToSerializable:
    """Tests for the to_serializable helper."""

    def test_primitives(self):
        assert to_serializable(None) is None
        assert to_serializable("hello") == "hello"
        assert to_serializable(42) == 42
        assert to_serializable(3.14) == 3.14
        assert to_serializable(True) is True

    def test_list_and_dict(self):
        assert to_serializable([1, "a", None]) == [1, "a", None]
        assert to_serializable({"k": [1, 2]}) == {"k": [1, 2]}

    def test_pydantic_model_dump(self):
        obj = MagicMock()
        obj.model_dump.return_value = {"field": "value"}
        del obj.dict
        # content attr is not a list, so it won't hit the MCP path first
        obj.content = None
        result = to_serializable(obj)
        assert result == {"field": "value"}

    def test_mcp_tool_result_single_text(self):
        item = SimpleNamespace(text="only line")
        obj = SimpleNamespace(content=[item])
        assert to_serializable(obj) == "only line"

    def test_mcp_tool_result_multiple_text(self):
        items = [SimpleNamespace(text="a"), SimpleNamespace(text="b")]
        obj = SimpleNamespace(content=items)
        assert to_serializable(obj) == ["a", "b"]

    def test_dict_content_items(self):
        obj = SimpleNamespace(content=[{"text": "from_dict"}])
        assert to_serializable(obj) == "from_dict"

    def test_fallback_to_str(self):
        class Custom:
            def __str__(self):
                return "custom_repr"

        assert to_serializable(Custom()) == "custom_repr"
