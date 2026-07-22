# tests/chat/test_tool_processor_extended.py
"""
Extended tests for mcp_cli/chat/tool_processor.py to achieve >90% coverage.

Covers: guard/rate limit checking, streaming tool execution, error paths,
tool display name resolution, internal tool handling, batch tool calls,
interrupt handling, value extraction, transport failure tracking, and more.
"""

from __future__ import annotations

import asyncio
import os
import platform
from datetime import datetime, UTC
from unittest.mock import MagicMock, patch

from mcp_cli.chat.models import ToolCallMetadata

import pytest
from chuk_tool_processor import ToolCall as CTPToolCall
from chuk_tool_processor import ToolResult as CTPToolResult
import chuk_ai_session_manager.guards.manager as _guard_mgr
from chuk_ai_session_manager.guards import (
    reset_tool_state,
    RuntimeLimits,
    ToolStateManager,
)

from mcp_cli.chat.tool_processor import ToolProcessor
from mcp_cli.chat.response_models import ToolCall, FunctionCall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_tool_state():
    """Reset the global tool state singleton before each test with permissive limits."""
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


# ---------------------------------------------------------------------------
# Dummy classes
# ---------------------------------------------------------------------------


class DummyUIManager:
    def __init__(self):
        self.printed_calls = []
        self.is_streaming_response = False
        self.interrupt_requested = False
        self.verbose_mode = False
        self.console = MagicMock()
        self._start_calls = []
        self._finish_calls = []

    def print_tool_call(self, tool_name, raw_arguments):
        self.printed_calls.append((tool_name, raw_arguments))

    async def finish_tool_execution(self, result=None, success=True):
        self._finish_calls.append((result, success))

    async def do_confirm_tool_execution(self, tool_name, arguments):
        return True

    async def start_tool_execution(self, tool_name, arguments):
        self._start_calls.append((tool_name, arguments))

    def finish_tool_calls(self):
        pass


class DummyUIManagerNoFinish:
    """UI manager without finish_tool_calls method."""

    def __init__(self):
        self.printed_calls = []
        self.is_streaming_response = False
        self.interrupt_requested = False
        self.verbose_mode = False
        self.console = MagicMock()

    def print_tool_call(self, tool_name, raw_arguments):
        self.printed_calls.append((tool_name, raw_arguments))

    async def finish_tool_execution(self, result=None, success=True):
        pass

    async def do_confirm_tool_execution(self, tool_name, arguments):
        return True

    async def start_tool_execution(self, tool_name, arguments):
        pass


class ConfirmDenyUIManager(DummyUIManager):
    """UI manager that denies tool confirmation."""

    async def do_confirm_tool_execution(self, tool_name, arguments):
        return False


class ErrorUIManager(DummyUIManager):
    """UI manager that raises on print_tool_call."""

    def print_tool_call(self, tool_name, raw_arguments):
        raise RuntimeError("UI explosion")


class AsyncFinishUIManager(DummyUIManager):
    """UI manager with async finish_tool_calls."""

    def __init__(self):
        super().__init__()
        self.finish_calls_invoked = False

    async def finish_tool_calls(self):
        self.finish_calls_invoked = True


class SyncFinishUIManager(DummyUIManager):
    """UI manager with sync finish_tool_calls."""

    def __init__(self):
        super().__init__()
        self.finish_calls_invoked = False

    def finish_tool_calls(self):
        self.finish_calls_invoked = True


class ErrorFinishUIManager(DummyUIManager):
    """UI manager whose finish_tool_calls raises."""

    def finish_tool_calls(self):
        raise RuntimeError("finish error")


class DummyToolManager:
    def __init__(self, return_result=None, raise_exception=False):
        self.return_result = return_result or {
            "isError": False,
            "content": "Tool executed successfully",
        }
        self.raise_exception = raise_exception
        self.executed_tool = None
        self.executed_args = None

    async def stream_execute_tools(
        self, calls, timeout=None, on_tool_start=None, max_concurrency=4
    ):
        for call in calls:
            self.executed_tool = call.tool
            self.executed_args = call.arguments

            if on_tool_start:
                await on_tool_start(call)

            now = datetime.now(UTC)
            if self.raise_exception:
                yield CTPToolResult(
                    id=call.id,
                    tool=call.tool,
                    result=None,
                    error="Simulated exception",
                    start_time=now,
                    end_time=now,
                    machine=platform.node(),
                    pid=os.getpid(),
                )
            elif self.return_result.get("isError"):
                yield CTPToolResult(
                    id=call.id,
                    tool=call.tool,
                    result=None,
                    error=self.return_result.get("error", "Error"),
                    start_time=now,
                    end_time=now,
                    machine=platform.node(),
                    pid=os.getpid(),
                )
            else:
                yield CTPToolResult(
                    id=call.id,
                    tool=call.tool,
                    result=self.return_result.get("content"),
                    error=None,
                    start_time=now,
                    end_time=now,
                    machine=platform.node(),
                    pid=os.getpid(),
                )


class CancellingToolManager:
    """Tool manager that sets cancelled during execution."""

    def __init__(self, processor):
        self.processor = processor

    async def stream_execute_tools(
        self, calls, timeout=None, on_tool_start=None, max_concurrency=4
    ):
        for call in calls:
            now = datetime.now(UTC)
            self.processor._cancelled = True
            yield CTPToolResult(
                id=call.id,
                tool=call.tool,
                result="cancelled result",
                error=None,
                start_time=now,
                end_time=now,
                machine=platform.node(),
                pid=os.getpid(),
            )


class DummyContext:
    def __init__(self, tool_manager=None):
        self.conversation_history = []
        self.tool_manager = tool_manager
        self.tool_processor = None

    def inject_tool_message(self, message):
        self.conversation_history.append(message)

    def get_display_name_for_tool(self, tool_name):
        return f"display:{tool_name}"


def make_tool_call(name="echo", args='{"msg": "hi"}', call_id="call_0"):
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments=args),
    )


def make_dict_tool_call(name="echo", args='{"msg": "hi"}', call_id="call_0"):
    return {
        "function": {"name": name, "arguments": args},
        "id": call_id,
    }


# ---------------------------------------------------------------------------
# Tests: ToolProcessor initialization
# ---------------------------------------------------------------------------


class TestToolProcessorInit:
    def test_init_sets_attributes(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui, max_concurrency=2)

        assert tp.tool_manager is tm
        assert tp.max_concurrency == 2
        assert tp._transport_failures == 0
        assert tp._cancelled is False
        assert ctx.tool_processor is tp

    def test_cancel_running_tasks(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        assert tp._cancelled is False
        tp.cancel_running_tasks()
        assert tp._cancelled is True


# ---------------------------------------------------------------------------
# Tests: process_tool_calls - empty and basic
# ---------------------------------------------------------------------------


class TestProcessToolCallsBasic:
    @pytest.mark.asyncio
    async def test_empty_list(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        await tp.process_tool_calls([])
        assert ctx.conversation_history == []

    @pytest.mark.asyncio
    async def test_none_name_mapping(self):
        """When name_mapping is None, should default to {}."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        tc = make_tool_call()
        await tp.process_tool_calls([tc], name_mapping=None)
        assert len(ctx.conversation_history) >= 2

    @pytest.mark.asyncio
    async def test_with_name_mapping(self):
        """Name mapping translates LLM tool name to execution name."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        tc = make_tool_call(name="llm_echo")
        await tp.process_tool_calls([tc], name_mapping={"llm_echo": "echo"})
        assert tm.executed_tool == "echo"

    @pytest.mark.asyncio
    async def test_with_reasoning_content(self):
        """Reasoning content is passed to assistant message."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        tc = make_tool_call()
        await tp.process_tool_calls([tc], reasoning_content="I need to call echo")
        # First message is assistant message
        assert ctx.conversation_history[0].reasoning_content == "I need to call echo"


# ---------------------------------------------------------------------------
# Tests: tool call info extraction
# ---------------------------------------------------------------------------


class TestExtractToolCallInfo:
    def test_from_tool_call_model(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tc = make_tool_call(name="test_tool", args='{"key": "val"}', call_id="c1")
        name, args, cid = tp._extract_tool_call_info(tc, 0)
        assert name == "test_tool"
        assert args == '{"key": "val"}'
        assert cid == "c1"

    def test_from_dict(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tc = make_dict_tool_call(name="dict_tool", args='{"a": 1}', call_id="c2")
        name, args, cid = tp._extract_tool_call_info(tc, 5)
        assert name == "dict_tool"
        assert args == '{"a": 1}'
        assert cid == "c2"

    def test_from_unknown_format(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        name, args, cid = tp._extract_tool_call_info("not a tool call", 3)
        assert name == "unknown_tool_3"
        assert args == {}
        assert cid == "call_3"

    def test_empty_name(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tc = {"function": {"name": "", "arguments": "{}"}, "id": "c0"}
        name, args, cid = tp._extract_tool_call_info(tc, 7)
        assert name == "unknown_tool_7"

    def test_missing_id_in_dict(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tc = {"function": {"name": "tool", "arguments": "{}"}}
        name, args, cid = tp._extract_tool_call_info(tc, 2)
        assert cid == "call_2"


# ---------------------------------------------------------------------------
# Tests: _parse_arguments
# ---------------------------------------------------------------------------


class TestParseArguments:
    def test_parse_json_string(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        result = tp._parse_arguments('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_parse_empty_string(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        result = tp._parse_arguments("")
        assert result == {}

    def test_parse_whitespace_string(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        result = tp._parse_arguments("   ")
        assert result == {}

    def test_parse_dict(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        result = tp._parse_arguments({"key": "val"})
        assert result == {"key": "val"}

    def test_parse_none(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        result = tp._parse_arguments(None)
        assert result == {}

    def test_parse_invalid_json(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        result = tp._parse_arguments("{invalid json")
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: _extract_result_value
# ---------------------------------------------------------------------------


class TestExtractResultValue:
    def _make_processor(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        return ToolProcessor(ctx, ui)

    def test_none(self):
        tp = self._make_processor()
        assert tp._extract_result_value(None) is None

    def test_string_none(self):
        tp = self._make_processor()
        assert tp._extract_result_value("None") is None

    def test_string_null(self):
        tp = self._make_processor()
        assert tp._extract_result_value("null") is None

    def test_direct_number(self):
        tp = self._make_processor()
        assert tp._extract_result_value(42) == 42
        assert tp._extract_result_value(3.14) == 3.14

    def test_numeric_string(self):
        tp = self._make_processor()
        assert tp._extract_result_value("42") == 42.0
        assert tp._extract_result_value("3.14") == 3.14

    def test_plain_string(self):
        tp = self._make_processor()
        assert tp._extract_result_value("hello world") == "hello world"

    def test_content_repr_string(self):
        tp = self._make_processor()
        result = tp._extract_result_value(
            "content=[{'type': 'text', 'text': '4.2426'}]"
        )
        assert result == 4.2426

    def test_content_repr_double_quotes(self):
        tp = self._make_processor()
        result = tp._extract_result_value('content=[{"type": "text", "text": "99.5"}]')
        assert result == 99.5

    def test_content_repr_no_match(self):
        tp = self._make_processor()
        result = tp._extract_result_value("content=[no match here]")
        assert result == "content=[no match here]"

    def test_dict_with_content_list(self):
        tp = self._make_processor()
        result = tp._extract_result_value({"content": [{"type": "text", "text": "42"}]})
        assert result == 42.0

    def test_dict_with_content_string(self):
        tp = self._make_processor()
        result = tp._extract_result_value({"content": "hello"})
        assert result == "hello"

    def test_dict_with_content_object_with_content_attr(self):
        tp = self._make_processor()
        inner = MagicMock()
        inner.content = [MagicMock(type="text", text="99")]
        result = tp._extract_result_value({"content": inner})
        assert result == 99.0

    def test_dict_success_result(self):
        tp = self._make_processor()
        result = tp._extract_result_value({"success": True, "result": "42"})
        assert result == 42.0

    def test_dict_success_result_none(self):
        tp = self._make_processor()
        result = tp._extract_result_value({"success": True, "result": None})
        assert result is None

    def test_dict_success_result_string_none(self):
        tp = self._make_processor()
        result = tp._extract_result_value({"success": True, "result": "None"})
        assert result is None

    def test_dict_is_error_false(self):
        tp = self._make_processor()
        result = tp._extract_result_value(
            {"isError": False, "content": [{"type": "text", "text": "ok"}]}
        )
        assert result == "ok"

    def test_dict_is_error_true(self):
        tp = self._make_processor()
        result = tp._extract_result_value({"isError": True, "error": "something broke"})
        assert result == "something broke"

    def test_dict_is_error_true_no_error_key(self):
        tp = self._make_processor()
        result = tp._extract_result_value({"isError": True, "content": "fallback"})
        assert result == "fallback"

    def test_dict_text_key(self):
        tp = self._make_processor()
        result = tp._extract_result_value({"text": "42"})
        assert result == 42.0

    def test_list_of_content_blocks(self):
        tp = self._make_processor()
        result = tp._extract_result_value(
            [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ]
        )
        assert result == "hello\nworld"

    def test_empty_list(self):
        tp = self._make_processor()
        assert tp._extract_result_value([]) is None

    def test_object_with_content_attr(self):
        tp = self._make_processor()
        obj = MagicMock()
        obj.content = [MagicMock(type="text", text="99")]
        result = tp._extract_result_value(obj)
        assert result == 99.0

    def test_content_list_with_no_text(self):
        tp = self._make_processor()
        result = tp._extract_result_value([{"type": "image", "data": "abc"}])
        assert result is None

    def test_content_block_dict_with_empty_text(self):
        tp = self._make_processor()
        result = tp._extract_result_value([{"type": "text", "text": ""}])
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _extract_from_content_list
# ---------------------------------------------------------------------------


class TestExtractFromContentList:
    def _make_processor(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        return ToolProcessor(ctx, ui)

    def test_empty(self):
        tp = self._make_processor()
        assert tp._extract_from_content_list([]) is None

    def test_text_content_objects(self):
        tp = self._make_processor()
        block = MagicMock()
        block.type = "text"
        block.text = "42"
        result = tp._extract_from_content_list([block])
        assert result == 42.0

    def test_non_text_block_type(self):
        tp = self._make_processor()
        block = MagicMock()
        block.type = "image"
        block.text = "ignored"
        result = tp._extract_from_content_list([block])
        assert result is None

    def test_dict_text_blocks(self):
        tp = self._make_processor()
        blocks = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        result = tp._extract_from_content_list(blocks)
        assert result == "first\nsecond"


# ---------------------------------------------------------------------------
# Tests: _try_parse_number
# ---------------------------------------------------------------------------


class TestTryParseNumber:
    def _make_processor(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        return ToolProcessor(ctx, ui)

    def test_integer_string(self):
        tp = self._make_processor()
        assert tp._try_parse_number("42") == 42.0

    def test_float_string(self):
        tp = self._make_processor()
        assert tp._try_parse_number("3.14") == 3.14

    def test_non_numeric(self):
        tp = self._make_processor()
        assert tp._try_parse_number("hello") == "hello"

    def test_none_input(self):
        tp = self._make_processor()
        assert tp._try_parse_number(None) is None

    def test_not_string(self):
        tp = self._make_processor()
        assert tp._try_parse_number(42) == 42  # not a string, returned as-is

    def test_string_none(self):
        tp = self._make_processor()
        assert tp._try_parse_number("None") is None

    def test_string_null(self):
        tp = self._make_processor()
        assert tp._try_parse_number("null") is None

    def test_whitespace(self):
        tp = self._make_processor()
        assert tp._try_parse_number("  42  ") == 42.0


# ---------------------------------------------------------------------------
# Tests: _format_tool_response
# ---------------------------------------------------------------------------


class TestFormatToolResponse:
    def _make_processor(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        return ToolProcessor(ctx, ui)

    def test_dict_with_mcp_content(self):
        tp = self._make_processor()
        inner = MagicMock()
        inner.content = [{"type": "text", "text": "hello"}]
        result = tp._format_tool_response({"content": inner})
        assert result == "hello"

    def test_dict_json_serializable(self):
        tp = self._make_processor()
        result = tp._format_tool_response({"key": "value"})
        assert '"key": "value"' in result

    def test_dict_not_serializable(self):
        tp = self._make_processor()
        # Create a dict with non-serializable content
        result = tp._format_tool_response({"key": object()})
        assert "key" in result

    def test_list_json_serializable(self):
        tp = self._make_processor()
        result = tp._format_tool_response([1, 2, 3])
        assert "[" in result

    def test_list_not_serializable(self):
        tp = self._make_processor()
        result = tp._format_tool_response([object()])
        assert "object" in result.lower() or "[" in result

    def test_string(self):
        tp = self._make_processor()
        result = tp._format_tool_response("hello")
        assert result == "hello"

    def test_number(self):
        tp = self._make_processor()
        result = tp._format_tool_response(42)
        assert result == "42"


# ---------------------------------------------------------------------------
# Tests: _track_transport_failures
# ---------------------------------------------------------------------------


class TestTrackTransportFailures:
    def _make_processor(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        return ToolProcessor(ctx, ui)

    def test_no_failure(self):
        tp = self._make_processor()
        tp._track_transport_failures(True, None)
        assert tp._consecutive_transport_failures == 0

    def test_non_transport_failure(self):
        tp = self._make_processor()
        tp._track_transport_failures(False, "Something went wrong")
        assert tp._consecutive_transport_failures == 0

    def test_transport_failure(self):
        tp = self._make_processor()
        tp._track_transport_failures(False, "Transport not initialized")
        assert tp._consecutive_transport_failures == 1
        assert tp._transport_failures == 1

    def test_transport_failure_lowercase(self):
        tp = self._make_processor()
        tp._track_transport_failures(False, "lost transport connection")
        assert tp._consecutive_transport_failures == 1

    def test_consecutive_transport_failures_warning(self):
        tp = self._make_processor()
        tp._track_transport_failures(False, "Transport not initialized")
        tp._track_transport_failures(False, "Transport not initialized")
        tp._track_transport_failures(False, "Transport not initialized")
        assert tp._consecutive_transport_failures == 3

    def test_transport_failure_reset_on_success(self):
        tp = self._make_processor()
        tp._track_transport_failures(False, "Transport not initialized")
        tp._track_transport_failures(False, "Transport not initialized")
        tp._track_transport_failures(True, None)
        assert tp._consecutive_transport_failures == 0
        assert tp._transport_failures == 2  # Total not reset

    def test_transport_failure_reset_on_non_transport_error(self):
        tp = self._make_processor()
        tp._track_transport_failures(False, "Transport not initialized")
        tp._track_transport_failures(False, "Some other error")
        assert tp._consecutive_transport_failures == 0


# ---------------------------------------------------------------------------
# Tests: _finish_tool_calls
# ---------------------------------------------------------------------------


class TestFinishToolCalls:
    @pytest.mark.asyncio
    async def test_async_finish(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = AsyncFinishUIManager()
        tp = ToolProcessor(ctx, ui)
        await tp._finish_tool_calls()
        assert ui.finish_calls_invoked is True

    @pytest.mark.asyncio
    async def test_sync_finish(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = SyncFinishUIManager()
        tp = ToolProcessor(ctx, ui)
        await tp._finish_tool_calls()
        assert ui.finish_calls_invoked is True

    @pytest.mark.asyncio
    async def test_error_in_finish(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = ErrorFinishUIManager()
        tp = ToolProcessor(ctx, ui)
        # Should not raise
        await tp._finish_tool_calls()

    @pytest.mark.asyncio
    async def test_no_finish_method(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManagerNoFinish()
        tp = ToolProcessor(ctx, ui)
        # Should not raise
        await tp._finish_tool_calls()


# ---------------------------------------------------------------------------
# Tests: _add_assistant_message_with_tool_calls
# ---------------------------------------------------------------------------


class TestAddAssistantMessage:
    def test_success(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        tc = make_tool_call()
        tp._add_assistant_message_with_tool_calls([tc])
        assert len(ctx.conversation_history) == 1
        assert ctx.conversation_history[0].role.value == "assistant"

    def test_with_reasoning(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        tc = make_tool_call()
        tp._add_assistant_message_with_tool_calls([tc], reasoning_content="thinking...")
        assert ctx.conversation_history[0].reasoning_content == "thinking..."

    def test_error_handling(self):
        ctx = MagicMock()
        ctx.tool_manager = DummyToolManager()
        ctx.inject_tool_message = MagicMock(side_effect=Exception("inject error"))
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        # Should not raise
        tp._add_assistant_message_with_tool_calls([make_tool_call()])


# ---------------------------------------------------------------------------
# Tests: _add_tool_result_to_history
# ---------------------------------------------------------------------------


class TestAddToolResult:
    def test_success(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        tp._add_tool_result_to_history("echo", "call_0", "result text")
        assert len(ctx.conversation_history) == 1
        assert ctx.conversation_history[0].role.value == "tool"
        assert ctx.conversation_history[0].content == "result text"

    def test_error_handling(self):
        ctx = MagicMock()
        ctx.tool_manager = DummyToolManager()
        ctx.inject_tool_message = MagicMock(side_effect=Exception("error"))
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        # Should not raise
        tp._add_tool_result_to_history("echo", "call_0", "result")


# ---------------------------------------------------------------------------
# Tests: _add_cancelled_tool_to_history
# ---------------------------------------------------------------------------


class TestAddCancelledTool:
    def test_success_with_dict_args(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        tp._add_cancelled_tool_to_history("echo", "call_0", {"msg": "hi"})
        assert len(ctx.conversation_history) == 3  # user, assistant, tool

    def test_success_with_string_args(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        tp._add_cancelled_tool_to_history("echo", "call_0", '{"msg": "hi"}')
        assert len(ctx.conversation_history) == 3

    def test_success_with_none_args(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        tp._add_cancelled_tool_to_history("echo", "call_0", None)
        assert len(ctx.conversation_history) == 3

    def test_error_handling(self):
        ctx = MagicMock()
        ctx.tool_manager = DummyToolManager()
        ctx.inject_tool_message = MagicMock(side_effect=Exception("error"))
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        # Should not raise
        tp._add_cancelled_tool_to_history("echo", "call_0", {})


# ---------------------------------------------------------------------------
# Tests: _should_confirm_tool
# ---------------------------------------------------------------------------


class TestShouldConfirmTool:
    def test_returns_prefs_result(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        mock_prefs = MagicMock()
        mock_prefs.should_confirm_tool.return_value = False
        with patch(
            "mcp_cli.chat.tool_processor.get_preference_manager",
            return_value=mock_prefs,
        ):
            assert tp._should_confirm_tool("echo") is False

    def test_returns_true_on_error(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        with patch(
            "mcp_cli.chat.tool_processor.get_preference_manager",
            side_effect=Exception("err"),
        ):
            assert tp._should_confirm_tool("echo") is True


# ---------------------------------------------------------------------------
# Tests: UI error during print_tool_call
# ---------------------------------------------------------------------------


class TestUIErrors:
    @pytest.mark.asyncio
    async def test_ui_error_non_fatal(self):
        """UI display error should not prevent tool execution."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = ErrorUIManager()
        tp = ToolProcessor(ctx, ui)
        tc = make_tool_call()
        with patch.object(tp, "_should_confirm_tool", return_value=False):
            await tp.process_tool_calls([tc])
        # Tool should still have executed
        assert tm.executed_tool == "echo"


# ---------------------------------------------------------------------------
# Tests: tool confirmation denial
# ---------------------------------------------------------------------------


class TestToolConfirmationDenial:
    @pytest.mark.asyncio
    async def test_denied_tool_adds_cancelled_history(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = ConfirmDenyUIManager()
        tp = ToolProcessor(ctx, ui)
        tc = make_tool_call()
        with patch.object(tp, "_should_confirm_tool", return_value=True):
            await tp.process_tool_calls([tc])
        # Should have the assistant tool call message + cancellation messages
        assert tp._cancelled is True
        # Check that interrupt_requested was set on UI
        assert ui.interrupt_requested is True


# ---------------------------------------------------------------------------
# Tests: interrupt during loop
# ---------------------------------------------------------------------------


class TestInterruptHandling:
    @pytest.mark.asyncio
    async def test_interrupt_requested_skips_remaining(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        ui.interrupt_requested = True  # Pre-set
        tp = ToolProcessor(ctx, ui)
        tc1 = make_tool_call(name="tool1", call_id="c1")
        tc2 = make_tool_call(name="tool2", call_id="c2")
        await tp.process_tool_calls([tc1, tc2])
        assert tp._cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_during_streaming(self):
        """Tool manager sets cancelled during streaming."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        cancelling_tm = CancellingToolManager(tp)
        tp.tool_manager = cancelling_tm

        tc = make_tool_call()
        with patch.object(tp, "_should_confirm_tool", return_value=False):
            await tp.process_tool_calls([tc])
        assert tp._cancelled is True


# ---------------------------------------------------------------------------
# Tests: no tool manager
# ---------------------------------------------------------------------------


class TestNoToolManager:
    @pytest.mark.asyncio
    async def test_raises_runtime_error(self):
        ctx = DummyContext(tool_manager=None)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        tc = make_tool_call()
        with pytest.raises(RuntimeError, match="No tool manager"):
            await tp.process_tool_calls([tc])


# ---------------------------------------------------------------------------
# Tests: None arguments rejected
# ---------------------------------------------------------------------------


class TestNoneArgumentsRejected:
    @pytest.mark.asyncio
    async def test_none_arg_values_rejected(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tc = make_tool_call(name="test_tool", args='{"param1": null, "param2": "ok"}')
        with patch.object(tp, "_should_confirm_tool", return_value=False):
            await tp.process_tool_calls([tc])

        # Should have blocked the call and added error to history
        tool_msgs = [m for m in ctx.conversation_history if m.role.value == "tool"]
        assert any("INVALID_ARGS" in m.content for m in tool_msgs)


# ---------------------------------------------------------------------------
# Tests: dynamic tools (call_tool)
# ---------------------------------------------------------------------------


class TestDynamicTools:
    @pytest.mark.asyncio
    async def test_call_tool_display_name(self):
        """When execution_tool_name is call_tool, display shows actual tool."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        # The LLM calls "call_tool" with tool_name in arguments
        tc = make_tool_call(
            name="call_tool",
            args='{"tool_name": "actual_tool", "param1": "value"}',
            call_id="c1",
        )
        with patch.object(tp, "_should_confirm_tool", return_value=False):
            await tp.process_tool_calls([tc], name_mapping={})

        # Check that displayed name includes the actual tool
        assert any("actual_tool" in str(call) for call in ui.printed_calls)

    @pytest.mark.asyncio
    async def test_call_tool_on_start_display(self):
        """_on_tool_start shows actual tool name for call_tool."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        call = CTPToolCall(
            id="c1",
            tool="call_tool",
            arguments={"tool_name": "real_tool", "x": 1},
        )
        tp._call_metadata["c1"] = ToolCallMetadata(
            llm_tool_name="call_tool",
            execution_tool_name="call_tool",
            display_name="call_tool",
            arguments={"tool_name": "real_tool", "x": 1},
        )
        await tp._on_tool_start(call)
        assert any("real_tool" in str(s) for s in ui._start_calls)


# ---------------------------------------------------------------------------
# Tests: display name resolution
# ---------------------------------------------------------------------------


class TestDisplayNameResolution:
    @pytest.mark.asyncio
    async def test_context_display_name(self):
        """get_display_name_for_tool is called for non-dynamic tools."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tc = make_tool_call(name="echo", args='{"msg": "hi"}')
        with patch.object(tp, "_should_confirm_tool", return_value=False):
            await tp.process_tool_calls([tc])

        # Display name should have been resolved via context
        assert any("display:echo" in str(call) for call in ui.printed_calls)


# ---------------------------------------------------------------------------
# Tests: _on_tool_result
# ---------------------------------------------------------------------------


class TestOnToolResult:
    @pytest.mark.asyncio
    async def test_successful_result_with_binding(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tp._call_metadata["c1"] = ToolCallMetadata(
            llm_tool_name="echo",
            execution_tool_name="echo",
            display_name="echo",
            arguments={"msg": "hi"},
            raw_arguments='{"msg": "hi"}',
        )

        now = datetime.now(UTC)
        result = CTPToolResult(
            id="c1",
            tool="echo",
            result="42",
            error=None,
            start_time=now,
            end_time=now,
            machine=platform.node(),
            pid=os.getpid(),
        )
        await tp._on_tool_result(result)

        # Should have added tool result to conversation history
        tool_msgs = [m for m in ctx.conversation_history if m.role.value == "tool"]
        assert len(tool_msgs) >= 1

    @pytest.mark.asyncio
    async def test_failed_result(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tp._call_metadata["c1"] = ToolCallMetadata(
            llm_tool_name="echo",
            execution_tool_name="echo",
            display_name="echo",
            arguments={},
            raw_arguments="{}",
        )

        now = datetime.now(UTC)
        result = CTPToolResult(
            id="c1",
            tool="echo",
            result=None,
            error="Tool failed",
            start_time=now,
            end_time=now,
            machine=platform.node(),
            pid=os.getpid(),
        )
        await tp._on_tool_result(result)

        tool_msgs = [m for m in ctx.conversation_history if m.role.value == "tool"]
        assert any("Error: Tool failed" in m.content for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_verbose_mode_display(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        ui.verbose_mode = True
        tp = ToolProcessor(ctx, ui)

        tp._call_metadata["c1"] = ToolCallMetadata(
            llm_tool_name="echo",
            execution_tool_name="echo",
            display_name="echo",
            arguments={},
            raw_arguments="{}",
        )

        now = datetime.now(UTC)
        result = CTPToolResult(
            id="c1",
            tool="echo",
            result="done",
            error=None,
            start_time=now,
            end_time=now,
            machine=platform.node(),
            pid=os.getpid(),
        )
        with patch("mcp_cli.display.display_tool_call_result") as mock_display:
            await tp._on_tool_result(result)
            mock_display.assert_called_once()

    @pytest.mark.asyncio
    async def test_result_missing_metadata(self):
        """Result with no metadata in _call_metadata uses defaults."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        now = datetime.now(UTC)
        result = CTPToolResult(
            id="unknown_id",
            tool="mystery_tool",
            result="data",
            error=None,
            start_time=now,
            end_time=now,
            machine=platform.node(),
            pid=os.getpid(),
        )
        await tp._on_tool_result(result)
        assert len(ctx.conversation_history) >= 1

    @pytest.mark.asyncio
    async def test_dynamic_tool_result(self):
        """call_tool results extract actual tool name from arguments."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tp._call_metadata["c1"] = ToolCallMetadata(
            llm_tool_name="call_tool",
            execution_tool_name="call_tool",
            display_name="call_tool",
            arguments={"tool_name": "real_tool", "x": 1},
            raw_arguments='{"tool_name": "real_tool", "x": 1}',
        )

        now = datetime.now(UTC)
        result = CTPToolResult(
            id="c1",
            tool="call_tool",
            result="done",
            error=None,
            start_time=now,
            end_time=now,
            machine=platform.node(),
            pid=os.getpid(),
        )
        await tp._on_tool_result(result)

        # Should have added tool result to conversation history
        tool_msgs = [m for m in ctx.conversation_history if m.role.value == "tool"]
        assert len(tool_msgs) >= 1


# ---------------------------------------------------------------------------
# Tests: CancelledError during streaming
# ---------------------------------------------------------------------------


class TestCancelledError:
    @pytest.mark.asyncio
    async def test_cancelled_error_handled(self):
        """asyncio.CancelledError during streaming is caught."""
        tm = MagicMock()

        async def raise_cancelled(*args, **kwargs):
            raise asyncio.CancelledError()

        # Make it async iterable that raises
        async def stream_gen(*args, **kwargs):
            raise asyncio.CancelledError()
            yield  # unreachable, but needed for async generator syntax

        tm.stream_execute_tools = stream_gen

        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tc = make_tool_call()
        with patch.object(tp, "_should_confirm_tool", return_value=False):
            await tp.process_tool_calls([tc])
        # Should complete without error


# ---------------------------------------------------------------------------
# Tests: multiple tool calls in batch
# ---------------------------------------------------------------------------


class TestBatchToolCalls:
    @pytest.mark.asyncio
    async def test_multiple_tools(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tc1 = make_tool_call(name="tool_a", args='{"x": 1}', call_id="c1")
        tc2 = make_tool_call(name="tool_b", args='{"y": 2}', call_id="c2")

        with patch.object(tp, "_should_confirm_tool", return_value=False):
            await tp.process_tool_calls([tc1, tc2])

        # Both tools should have been printed
        names = [call[0] for call in ui.printed_calls]
        assert "display:tool_a" in names
        assert "display:tool_b" in names


# ---------------------------------------------------------------------------
# Tests: _parse_content_repr
# ---------------------------------------------------------------------------


class TestParseContentRepr:
    def _make_processor(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        return ToolProcessor(ctx, ui)

    def test_single_quotes(self):
        tp = self._make_processor()
        result = tp._parse_content_repr("content=[{'type': 'text', 'text': '3.14'}]")
        assert result == 3.14

    def test_double_quotes(self):
        tp = self._make_processor()
        result = tp._parse_content_repr('content=[{"type": "text", "text": "99"}]')
        assert result == 99.0

    def test_no_match(self):
        tp = self._make_processor()
        result = tp._parse_content_repr("content=[something else]")
        assert result == "content=[something else]"

    def test_non_numeric_text(self):
        tp = self._make_processor()
        result = tp._parse_content_repr("content=[{'type': 'text', 'text': 'hello'}]")
        assert result == "hello"


# ---------------------------------------------------------------------------
# Tests: _on_tool_start callback
# ---------------------------------------------------------------------------


class TestOnToolStart:
    @pytest.mark.asyncio
    async def test_with_metadata(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tp._call_metadata["c1"] = ToolCallMetadata(
            llm_tool_name="my_tool",
            execution_tool_name="my_tool",
            display_name="my_tool",
            arguments={"key": "val"},
        )

        call = CTPToolCall(id="c1", tool="echo", arguments={"key": "val"})
        await tp._on_tool_start(call)
        assert ("my_tool", {"key": "val"}) in ui._start_calls

    @pytest.mark.asyncio
    async def test_without_metadata(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        call = CTPToolCall(id="unknown", tool="mystery", arguments={"a": 1})
        await tp._on_tool_start(call)
        assert ("mystery", {"a": 1}) in ui._start_calls


# ---------------------------------------------------------------------------
# NEW TESTS: Covering missing lines for >90% coverage
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lines 200-211: check_references returns invalid (missing $vN refs)
# ---------------------------------------------------------------------------


class TestCheckReferencesBlocking:
    """Cover lines 200-211: check_references().valid == False blocks tool."""

    @pytest.mark.asyncio
    async def test_missing_reference_blocks_tool(self):
        """When arguments contain $vN references that don't exist, the tool is blocked."""
        from chuk_ai_session_manager.guards.models import ReferenceCheckResult

        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        # Create a tool call with $v99 reference that doesn't exist
        tc = make_tool_call(
            name="compute_tool",
            args='{"value": "$v99"}',
            call_id="c_ref",
        )

        # Mock get_tool_state to return a mock with check_references returning invalid
        mock_tool_state = MagicMock()
        mock_tool_state.check_references.return_value = ReferenceCheckResult(
            valid=False,
            missing_refs=["$v99"],
            resolved_refs={},
            message="Missing references: $v99",
        )
        mock_tool_state.format_bindings_for_model.return_value = "No bindings"

        with (
            patch.object(tp, "_should_confirm_tool", return_value=False),
            patch(
                "mcp_cli.chat.tool_processor.get_agent_tool_state",
                return_value=mock_tool_state,
            ),
        ):
            await tp.process_tool_calls([tc])

        # The tool should NOT have been executed
        assert tm.executed_tool is None

        # Should have a tool message with "Blocked" indicating missing references
        tool_msgs = [m for m in ctx.conversation_history if m.role.value == "tool"]
        assert any("Blocked" in m.content for m in tool_msgs)


# ---------------------------------------------------------------------------
# Lines 241-250: Ungrounded call + not auto-rebound + precondition fails
# ---------------------------------------------------------------------------


class TestUngroundedPreconditionFail:
    """Cover lines 241-250: ungrounded tool with failed preconditions."""

    @pytest.mark.asyncio
    async def test_precondition_failure_blocks_tool(self):
        """When tool is ungrounded, not auto-rebound, and preconditions fail, it's blocked."""
        from chuk_ai_session_manager.guards.models import (
            ReferenceCheckResult,
            UngroundedCallResult,
        )

        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tc = make_tool_call(
            name="custom_tool",
            args='{"x": 42.0}',
            call_id="c_precond",
        )

        mock_ts = MagicMock()
        mock_ts.check_references.return_value = ReferenceCheckResult(
            valid=True,
            missing_refs=[],
            resolved_refs={},
            message="OK",
        )
        mock_ts.is_idempotent_math_tool.return_value = False
        mock_ts.is_discovery_tool.return_value = False
        mock_ts.check_ungrounded_call.return_value = UngroundedCallResult(
            is_ungrounded=True,
            numeric_args=["x=42.0"],
            has_bindings=False,
            message="Ungrounded numeric arguments",
        )
        mock_ts.should_auto_rebound.return_value = False
        mock_ts.check_tool_preconditions.return_value = (
            False,
            "Precondition: need computed values first",
        )

        with (
            patch.object(tp, "_should_confirm_tool", return_value=False),
            patch(
                "mcp_cli.chat.tool_processor.get_agent_tool_state", return_value=mock_ts
            ),
        ):
            await tp.process_tool_calls([tc])

        # Tool should not have been executed
        assert tm.executed_tool is None

        # Should have a blocked message in tool history
        tool_msgs = [m for m in ctx.conversation_history if m.role.value == "tool"]
        assert any("Blocked" in m.content for m in tool_msgs)
        assert any("Precondition" in m.content for m in tool_msgs)


# ---------------------------------------------------------------------------
# Lines 263-304: Soft block repair (3 paths)
# ---------------------------------------------------------------------------


class TestSoftBlockRepair:
    """Cover lines 263-304: try_soft_block_repair paths."""

    def _setup(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)
        return tm, ctx, ui, tp

    def _make_mock_tool_state(self, repair_return):
        """Create a mock tool state for soft block repair tests."""
        from chuk_ai_session_manager.guards.models import (
            ReferenceCheckResult,
            UngroundedCallResult,
        )
        from chuk_tool_processor.guards.base import GuardResult, GuardVerdict

        mock_ts = MagicMock()
        mock_ts.check_references.return_value = ReferenceCheckResult(
            valid=True,
            missing_refs=[],
            resolved_refs={},
            message="OK",
        )
        mock_ts.is_idempotent_math_tool.return_value = False
        mock_ts.is_discovery_tool.return_value = False
        mock_ts.check_ungrounded_call.return_value = UngroundedCallResult(
            is_ungrounded=True,
            numeric_args=["x=42.0"],
            has_bindings=True,
        )
        mock_ts.should_auto_rebound.return_value = True
        mock_ts.try_soft_block_repair.return_value = repair_return
        mock_ts.resolve_references.side_effect = lambda args: args
        mock_ts.check_per_tool_limit.return_value = GuardResult(
            verdict=GuardVerdict.ALLOW
        )
        mock_ts.limits = MagicMock()
        mock_ts.limits.per_tool_cap = 100
        mock_ts.format_bindings_for_model.return_value = "No bindings available"
        return mock_ts

    @pytest.mark.asyncio
    async def test_repair_succeeds_rebind(self):
        """Lines 271-280: Repair succeeds with rebound arguments."""
        tm, ctx, ui, tp = self._setup()

        tc = make_tool_call(
            name="some_tool",
            args='{"x": 42.0}',
            call_id="c_repair",
        )

        mock_ts = self._make_mock_tool_state((True, {"x": "$v1"}, None))

        with (
            patch.object(tp, "_should_confirm_tool", return_value=False),
            patch(
                "mcp_cli.chat.tool_processor.get_agent_tool_state", return_value=mock_ts
            ),
        ):
            await tp.process_tool_calls([tc])

        # Tool should have been executed with repaired args
        assert tm.executed_tool is not None

    @pytest.mark.asyncio
    async def test_repair_symbolic_fallback(self):
        """Lines 281-291: Repair returns symbolic fallback response."""
        tm, ctx, ui, tp = self._setup()

        tc = make_tool_call(
            name="some_tool",
            args='{"x": 42.0}',
            call_id="c_fallback",
        )

        fallback_msg = (
            "Cannot call some_tool with literal values. Please compute first."
        )
        mock_ts = self._make_mock_tool_state((False, None, fallback_msg))

        with (
            patch.object(tp, "_should_confirm_tool", return_value=False),
            patch(
                "mcp_cli.chat.tool_processor.get_agent_tool_state", return_value=mock_ts
            ),
        ):
            await tp.process_tool_calls([tc])

        # Tool should NOT have been executed
        assert tm.executed_tool is None

        # Should have the fallback message in tool history
        tool_msgs = [m for m in ctx.conversation_history if m.role.value == "tool"]
        assert any(fallback_msg in m.content for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_repair_all_failed(self):
        """Lines 292-304: All repairs failed - error in history."""
        tm, ctx, ui, tp = self._setup()

        tc = make_tool_call(
            name="some_tool",
            args='{"x": 42.0}',
            call_id="c_fail_repair",
        )

        mock_ts = self._make_mock_tool_state((False, None, None))

        with (
            patch.object(tp, "_should_confirm_tool", return_value=False),
            patch(
                "mcp_cli.chat.tool_processor.get_agent_tool_state", return_value=mock_ts
            ),
        ):
            await tp.process_tool_calls([tc])

        # Tool should NOT have been executed
        assert tm.executed_tool is None

        # Should have "Cannot proceed" in the tool message
        tool_msgs = [m for m in ctx.conversation_history if m.role.value == "tool"]
        assert any("Cannot proceed" in m.content for m in tool_msgs)


# ---------------------------------------------------------------------------
# Lines 310-319: Per-tool call limit blocking
# ---------------------------------------------------------------------------


class TestPerToolLimitBlocking:
    """Cover lines 310-319: per-tool limit blocks tool execution."""

    @pytest.mark.asyncio
    async def test_per_tool_limit_blocks(self):
        """When per_tool_cap > 0 and check_per_tool_limit returns blocked, tool is blocked."""
        from chuk_ai_session_manager.guards.models import (
            ReferenceCheckResult,
            UngroundedCallResult,
        )
        from chuk_tool_processor.guards.base import GuardResult, GuardVerdict

        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tc = make_tool_call(
            name="limited_tool",
            args='{"x": "hello"}',
            call_id="c_limit",
        )

        mock_ts = MagicMock()
        mock_ts.check_references.return_value = ReferenceCheckResult(
            valid=True,
            missing_refs=[],
            resolved_refs={},
            message="OK",
        )
        mock_ts.is_idempotent_math_tool.return_value = False
        mock_ts.is_discovery_tool.return_value = False
        # Must return a proper UngroundedCallResult (not truthy MagicMock)
        mock_ts.check_ungrounded_call.return_value = UngroundedCallResult(
            is_ungrounded=False,
        )
        mock_ts.limits = MagicMock()
        mock_ts.limits.per_tool_cap = 3
        mock_ts.check_per_tool_limit.return_value = GuardResult(
            verdict=GuardVerdict.BLOCK,
            reason="Tool limited_tool exceeded per-tool limit (3/3)",
        )

        with (
            patch.object(tp, "_should_confirm_tool", return_value=False),
            patch(
                "mcp_cli.chat.tool_processor.get_agent_tool_state", return_value=mock_ts
            ),
        ):
            await tp.process_tool_calls([tc])

        # Tool should NOT have been executed
        assert tm.executed_tool is None

        # Should have the limit message in tool history
        tool_msgs = [m for m in ctx.conversation_history if m.role.value == "tool"]
        assert any(
            "per-tool limit" in m.content.lower() or "exceeded" in m.content.lower()
            for m in tool_msgs
        )


# ---------------------------------------------------------------------------
# Line 447: requires_justification True
# ---------------------------------------------------------------------------


class TestRequiresJustification:
    """Cover line 447: per_tool_status.requires_justification is True."""

    @pytest.mark.asyncio
    async def test_requires_justification_logged(self):
        """When track_tool_call returns requires_justification=True, warning is logged."""
        from chuk_ai_session_manager.guards.models import PerToolCallStatus

        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tp._call_metadata["c1"] = ToolCallMetadata(
            llm_tool_name="heavy_tool",
            execution_tool_name="heavy_tool",
            display_name="heavy_tool",
            arguments={"msg": "hi"},
            raw_arguments='{"msg": "hi"}',
        )

        now = datetime.now(UTC)

        result = CTPToolResult(
            id="c1",
            tool="heavy_tool",
            result="done",
            error=None,
            start_time=now,
            end_time=now,
            machine=platform.node(),
            pid=os.getpid(),
        )

        mock_ts = MagicMock()
        mock_ts.is_discovery_tool.return_value = False
        mock_ts.track_tool_call.return_value = PerToolCallStatus(
            tool_name="heavy_tool",
            call_count=5,
            max_calls=3,
            requires_justification=True,
        )
        # Make cache_result and bind_value work
        mock_ts.cache_result.return_value = None
        mock_ts.bind_value.return_value = MagicMock(id="v1", typed_value="done")

        mock_search_engine = MagicMock()

        with (
            patch(
                "mcp_cli.chat.tool_processor.get_agent_tool_state", return_value=mock_ts
            ),
            patch(
                "mcp_cli.chat.tool_processor.get_search_engine",
                return_value=mock_search_engine,
            ),
        ):
            await tp._on_tool_result(result)

        # Verify result was still added to history (tool was not blocked, just warned)
        assert len(ctx.conversation_history) >= 1


# ---------------------------------------------------------------------------
# Lines 454-455: Discovery tool classify_by_result + _register_discovered_tools
# ---------------------------------------------------------------------------


class TestDiscoveryToolResult:
    """Cover lines 454-455: discovery tool triggers classify_by_result and _register_discovered_tools."""

    @pytest.mark.asyncio
    async def test_discovery_tool_result(self):
        """When result is from a discovery tool, classify_by_result and _register_discovered_tools are called."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        tp._call_metadata["c1"] = ToolCallMetadata(
            llm_tool_name="search_tools",
            execution_tool_name="search_tools",
            display_name="search_tools",
            arguments={"query": "math"},
            raw_arguments='{"query": "math"}',
        )

        now = datetime.now(UTC)
        discovery_result = [{"name": "sqrt_tool"}, {"name": "add_tool"}]

        result = CTPToolResult(
            id="c1",
            tool="search_tools",
            result=discovery_result,
            error=None,
            start_time=now,
            end_time=now,
            machine=platform.node(),
            pid=os.getpid(),
        )

        mock_ts = MagicMock()
        mock_ts.is_discovery_tool.return_value = True
        mock_ts.classify_by_result.return_value = None
        mock_ts.cache_result.return_value = None
        mock_ts.register_discovered_tool.return_value = None

        mock_search_engine = MagicMock()

        with (
            patch(
                "mcp_cli.chat.tool_processor.get_agent_tool_state", return_value=mock_ts
            ),
            patch(
                "mcp_cli.chat.tool_processor.get_search_engine",
                return_value=mock_search_engine,
            ),
        ):
            await tp._on_tool_result(result)

        # Verify it completed without error
        assert len(ctx.conversation_history) >= 1

        # Verify classify_by_result and _register_discovered_tools were triggered
        mock_ts.classify_by_result.assert_called_once_with(
            "search_tools", discovery_result
        )
        # _register_discovered_tools should have called register_discovered_tool
        assert mock_ts.register_discovered_tool.call_count >= 1


# ---------------------------------------------------------------------------
# Lines 585-587: Generic exception in _parse_arguments
# ---------------------------------------------------------------------------


class TestParseArgumentsGenericException:
    """Cover lines 585-587: non-JSONDecodeError exception in _parse_arguments."""

    def test_generic_exception(self):
        """When raw_arguments causes a non-JSON exception, returns empty dict."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        # Create an object whose __bool__ raises TypeError
        # This triggers an exception at `raw_arguments or {}` that is not JSONDecodeError
        class BadBool:
            def __bool__(self):
                raise TypeError("no bool")

        result = tp._parse_arguments(BadBool())
        assert result == {}


# ---------------------------------------------------------------------------
# Line 640: isError False path recursing into content
# ---------------------------------------------------------------------------


class TestExtractResultValueIsErrorFalse:
    """Cover line 640: isError=False recurses into content."""

    def test_is_error_false_with_content(self):
        """When isError=False, recursively extract from content."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        # isError False with content that's a string number
        result = tp._extract_result_value({"isError": False, "content": "42"})
        assert result == 42.0

    def test_is_error_false_with_nested_content(self):
        """When isError=False, recurse into nested content structure."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        result = tp._extract_result_value(
            {"isError": False, "content": [{"type": "text", "text": "hello"}]}
        )
        assert result == "hello"

    def test_is_error_false_with_none_content(self):
        """When isError=False with None content."""
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        tp = ToolProcessor(ctx, ui)

        result = tp._extract_result_value({"isError": False, "content": None})
        assert result is None


# ---------------------------------------------------------------------------
# Lines 862-915: _register_discovered_tools
# ---------------------------------------------------------------------------


class TestRegisterDiscoveredTools:
    """Cover lines 862-915: all branches of _register_discovered_tools."""

    def _make_processor(self):
        tm = DummyToolManager()
        ctx = DummyContext(tool_manager=tm)
        ui = DummyUIManager()
        return ToolProcessor(ctx, ui)

    def test_none_result(self):
        """Line 862-863: result is None, early return."""
        tp = self._make_processor()
        tool_state = MagicMock()
        tp._register_discovered_tools(tool_state, "search_tools", None)
        tool_state.register_discovered_tool.assert_not_called()

    def test_string_result_valid_json_list(self):
        """Lines 870-873: result is a JSON string that parses to a list."""
        import json

        tp = self._make_processor()
        tool_state = MagicMock()
        result_str = json.dumps([{"name": "tool_a"}, {"name": "tool_b"}])
        tp._register_discovered_tools(tool_state, "search_tools", result_str)
        assert tool_state.register_discovered_tool.call_count == 2
        tool_state.register_discovered_tool.assert_any_call("tool_a")
        tool_state.register_discovered_tool.assert_any_call("tool_b")

    def test_string_result_invalid_json(self):
        """Lines 873-874: result is a string that's not valid JSON."""
        tp = self._make_processor()
        tool_state = MagicMock()
        tp._register_discovered_tools(tool_state, "search_tools", "not json")
        tool_state.register_discovered_tool.assert_not_called()

    def test_list_with_dict_items_name_key(self):
        """Lines 877-884: list of dicts with 'name' key."""
        tp = self._make_processor()
        tool_state = MagicMock()
        result = [{"name": "sqrt"}, {"name": "add"}]
        tp._register_discovered_tools(tool_state, "search_tools", result)
        assert tool_state.register_discovered_tool.call_count == 2
        tool_state.register_discovered_tool.assert_any_call("sqrt")
        tool_state.register_discovered_tool.assert_any_call("add")

    def test_list_with_dict_items_tool_name_key(self):
        """Lines 881-883: list of dicts with 'tool_name' key."""
        tp = self._make_processor()
        tool_state = MagicMock()
        result = [{"tool_name": "my_tool"}]
        tp._register_discovered_tools(tool_state, "list_tools", result)
        tool_state.register_discovered_tool.assert_called_once_with("my_tool")

    def test_list_with_dict_items_tool_key(self):
        """Lines 881-883: list of dicts with 'tool' key."""
        tp = self._make_processor()
        tool_state = MagicMock()
        result = [{"tool": "other_tool"}]
        tp._register_discovered_tools(tool_state, "list_tools", result)
        tool_state.register_discovered_tool.assert_called_once_with("other_tool")

    def test_list_with_string_items(self):
        """Lines 885-886: list of strings."""
        tp = self._make_processor()
        tool_state = MagicMock()
        result = ["tool_x", "tool_y"]
        tp._register_discovered_tools(tool_state, "search_tools", result)
        assert tool_state.register_discovered_tool.call_count == 2
        tool_state.register_discovered_tool.assert_any_call("tool_x")
        tool_state.register_discovered_tool.assert_any_call("tool_y")

    def test_dict_with_name_key(self):
        """Lines 890-892: dict with 'name' key (single tool schema)."""
        tp = self._make_processor()
        tool_state = MagicMock()
        result = {"name": "single_tool", "description": "A tool"}
        tp._register_discovered_tools(tool_state, "get_tool_schema", result)
        tool_state.register_discovered_tool.assert_called_once_with("single_tool")

    def test_dict_with_tools_list_of_dicts(self):
        """Lines 894-897: dict with 'tools' list of dicts."""
        tp = self._make_processor()
        tool_state = MagicMock()
        result = {"tools": [{"name": "t1"}, {"name": "t2"}]}
        tp._register_discovered_tools(tool_state, "list_tools", result)
        assert tool_state.register_discovered_tool.call_count == 2

    def test_dict_with_tools_list_of_strings(self):
        """Lines 898-899: dict with 'tools' list of strings."""
        tp = self._make_processor()
        tool_state = MagicMock()
        result = {"tools": ["tool_a", "tool_b"]}
        tp._register_discovered_tools(tool_state, "list_tools", result)
        assert tool_state.register_discovered_tool.call_count == 2

    def test_dict_with_content_key_recurse(self):
        """Lines 901-906: dict with 'content' key recursively extracts."""
        tp = self._make_processor()
        tool_state = MagicMock()
        result = {"content": [{"name": "inner_tool"}]}
        tp._register_discovered_tools(tool_state, "search_tools", result)
        tool_state.register_discovered_tool.assert_called_once_with("inner_tool")

    def test_empty_tool_names_filtered(self):
        """Lines 909-911: empty tool names are filtered out."""
        tp = self._make_processor()
        tool_state = MagicMock()
        result = [{"name": ""}, {"name": "valid_tool"}, {"name": ""}]
        tp._register_discovered_tools(tool_state, "search_tools", result)
        # Only "valid_tool" should be registered (empty strings filtered)
        tool_state.register_discovered_tool.assert_called_once_with("valid_tool")

    def test_exception_handling(self):
        """Lines 914-915: exception during registration is caught."""
        tp = self._make_processor()
        tool_state = MagicMock()
        tool_state.register_discovered_tool.side_effect = Exception(
            "registration error"
        )
        result = [{"name": "failing_tool"}]
        # Should not raise
        tp._register_discovered_tools(tool_state, "search_tools", result)

    def test_list_with_mixed_items(self):
        """List with dict items missing expected keys and string items."""
        tp = self._make_processor()
        tool_state = MagicMock()
        result = [{"other_key": "val"}, "string_tool", {"name": "dict_tool"}]
        tp._register_discovered_tools(tool_state, "search_tools", result)
        # "string_tool" and "dict_tool" should be registered, the first dict is skipped
        assert tool_state.register_discovered_tool.call_count == 2

    def test_string_result_valid_json_dict(self):
        """String result that parses to a dict with 'name'."""
        import json

        tp = self._make_processor()
        tool_state = MagicMock()
        result_str = json.dumps({"name": "json_tool"})
        tp._register_discovered_tools(tool_state, "get_tool_schema", result_str)
        tool_state.register_discovered_tool.assert_called_once_with("json_tool")
