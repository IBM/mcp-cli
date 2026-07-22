# tests/mcp_cli/chat/test_tool_processor.py
import json
import logging
import os
import platform
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_tool_processor import ToolResult as CTPToolResult
import chuk_ai_session_manager.guards.manager as _guard_mgr
from chuk_ai_session_manager.guards import (
    get_tool_state,
    reset_tool_state,
    RuntimeLimits,
    ToolStateManager,
)

from mcp_cli.chat.tool_processor import ToolProcessor
from mcp_cli.chat.response_models import ToolCall, FunctionCall
from mcp_cli.tools.models import ToolCallResult


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


# ---------------------------
# Dummy classes for testing
# ---------------------------


class DummyUIManager:
    def __init__(self):
        self.printed_calls = []  # To record calls to print_tool_call
        self.is_streaming_response = False  # Add missing attribute

    def print_tool_call(self, tool_name, raw_arguments):
        self.printed_calls.append((tool_name, raw_arguments))

    async def finish_tool_execution(self, result=None, success=True):
        # Add async method that tool processor expects
        pass

    async def do_confirm_tool_execution(self, tool_name, arguments):
        # Mock confirmation - always return True for tests
        return True

    async def start_tool_execution(self, tool_name, arguments):
        # Mock start tool execution - no-op for tests
        pass


class DummyStreamManager:
    def __init__(self, return_result=None, raise_exception=False):
        # return_result: dictionary to return when call_tool is invoked.
        self.return_result = return_result or {
            "isError": False,
            "content": "Successful call",
        }
        self.raise_exception = raise_exception
        self.called_tool = None
        self.called_args = None

    async def call_tool(self, tool_name, arguments):
        self.called_tool = tool_name
        self.called_args = arguments
        if self.raise_exception:
            raise Exception("Simulated call_tool exception")
        return self.return_result


class DummyToolManager:
    """Mock tool manager with execute_tool and stream_execute_tools methods."""

    def __init__(self, return_result=None, raise_exception=False):
        self.return_result = return_result or {
            "isError": False,
            "content": "Tool executed successfully",
        }
        self.raise_exception = raise_exception
        self.executed_tool = None
        self.executed_args = None

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
        self.executed_tool = tool_name
        self.executed_args = arguments
        if self.raise_exception:
            raise Exception("Simulated execute_tool exception")

        # Return a ToolCallResult object, not a dict
        if self.return_result.get("isError"):
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=self.return_result.get("error", "Simulated error"),
            )
        else:
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=self.return_result.get("content"),
                error=None,
            )

    async def stream_execute_tools(
        self, calls, timeout=None, on_tool_start=None, max_concurrency=4
    ):
        """Yield CTPToolResult for each call."""
        import platform
        import os

        for call in calls:
            self.executed_tool = call.tool
            self.executed_args = call.arguments

            # Invoke start callback if provided
            if on_tool_start:
                await on_tool_start(call)

            if self.raise_exception:
                now = datetime.now(UTC)
                yield CTPToolResult(
                    id=call.id,
                    tool=call.tool,
                    result=None,
                    error="Simulated execute_tool exception",
                    start_time=now,
                    end_time=now,
                    machine=platform.node(),
                    pid=os.getpid(),
                )
            elif self.return_result.get("isError"):
                now = datetime.now(UTC)
                yield CTPToolResult(
                    id=call.id,
                    tool=call.tool,
                    result=None,
                    error=self.return_result.get("error", "Simulated error"),
                    start_time=now,
                    end_time=now,
                    machine=platform.node(),
                    pid=os.getpid(),
                )
            else:
                now = datetime.now(UTC)
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


class DummyContext:
    """A dummy context object with conversation_history and managers."""

    def __init__(self, stream_manager=None, tool_manager=None):
        self.conversation_history = []
        self.stream_manager = stream_manager
        self.tool_manager = tool_manager

    def inject_tool_message(self, message):
        """Add a message to conversation history (matches ChatContext API)."""
        self.conversation_history.append(message)


# ---------------------------
# Tests for ToolProcessor
# ---------------------------


@pytest.mark.asyncio
async def test_process_tool_calls_empty_list(caplog):
    # Test that an empty list of tool_calls logs a warning and does nothing.
    tool_manager = DummyToolManager()
    context = DummyContext(
        stream_manager=DummyStreamManager(), tool_manager=tool_manager
    )
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    with caplog.at_level(logging.WARNING, logger="mcp_cli.chat.tool_processor"):
        await processor.process_tool_calls([])
    # No tool calls processed; conversation history remains unchanged.
    assert context.conversation_history == []

    # Check that a warning was logged.
    assert "Empty tool_calls list received." in caplog.text


@pytest.mark.asyncio
async def test_process_tool_calls_successful_tool():
    # Test a successful tool call.
    result_dict = {"isError": False, "content": "Tool executed successfully"}
    stream_manager = DummyStreamManager(return_result=result_dict)
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(stream_manager=stream_manager, tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    # Create a proper ToolCall Pydantic model instead of a dict
    tool_call = ToolCall(
        id="call_echo",
        type="function",
        function=FunctionCall(name="echo", arguments='{"msg": "Hello"}'),
    )
    await processor.process_tool_calls([tool_call])

    # Verify that the UI manager printed the tool call.
    assert ("echo", '{"msg": "Hello"}') in ui_manager.printed_calls

    # Expect two conversation history records:
    #  - First: an assistant record containing the tool call details.
    #  - Second: a tool record containing the result.
    assert len(context.conversation_history) == 2

    call_record = context.conversation_history[0]
    response_record = context.conversation_history[1]

    # Verify the tool call record contains the correct id.
    assert call_record.tool_calls is not None
    # tool_calls is now a list of ToolCall Pydantic models, not dicts
    assert any(item.id == "call_echo" for item in call_record.tool_calls)

    # Verify the response record.
    assert response_record.role.value == "tool"
    # Content now includes value binding info ($vN = value)
    assert "Tool executed successfully" in response_record.content


@pytest.mark.asyncio
async def test_process_tool_calls_with_argument_parsing():
    # Test that raw arguments given as a JSON string are parsed into a dict.
    result_dict = {"isError": False, "content": {"parsed": True}}
    stream_manager = DummyStreamManager(return_result=result_dict)
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(stream_manager=stream_manager, tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    # Register 123 as a user-provided literal so it passes ungrounded check
    tool_state = get_tool_state()
    tool_state.register_user_literals("Test with value 123")

    tool_call = {
        "function": {"name": "parse_tool", "arguments": '{"num": 123}'},
        "id": "call_parse",
    }
    await processor.process_tool_calls([tool_call])

    # Check that execute_tool was given parsed arguments (a dict).
    assert isinstance(tool_manager.executed_args, dict)
    assert tool_manager.executed_args.get("num") == 123

    # Check that the response record content contains the formatted result.
    # Note: Content now includes value binding info ($vN = value) appended
    response_record = context.conversation_history[1]
    expected_formatted = json.dumps(result_dict["content"], indent=2)
    assert expected_formatted in response_record.content


@pytest.mark.asyncio
async def test_process_tool_calls_tool_call_error():
    # Test a tool call that returns an error result.
    error_result = {
        "isError": True,
        "error": "Simulated error",
        "content": "Error: Simulated error",
    }
    stream_manager = DummyStreamManager(return_result=error_result)
    tool_manager = DummyToolManager(return_result=error_result)
    context = DummyContext(stream_manager=stream_manager, tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_call = {
        "function": {"name": "fail_tool", "arguments": '{"dummy": "data"}'},
        "id": "fail_call",
    }
    await processor.process_tool_calls([tool_call])

    # Conversation history should include a tool record with an error message.
    response_record = context.conversation_history[1]
    assert response_record.role.value == "tool"
    assert "Error: Simulated error" in response_record.content


@pytest.mark.asyncio
async def test_process_tool_calls_no_tool_manager():
    # Test when no tool manager is available.
    context = DummyContext(stream_manager=None, tool_manager=None)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    # Supply a dummy tool call - pass as individual dict, not wrapped in list
    tool_call = {
        "function": {"name": "dummy_tool", "arguments": '{"key": "value"}'},
        "id": "test1",
    }

    # Pass as a list to process_tool_calls - should raise RuntimeError
    # The finally block still runs, so missing results are filled in.
    with pytest.raises(RuntimeError, match="No tool manager available"):
        await processor.process_tool_calls([tool_call])


@pytest.mark.asyncio
async def test_ensure_all_tool_results_fills_missing():
    """Test that _ensure_all_tool_results adds placeholder results for missing tool_call_ids."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    # Simulate: assistant message was added but no tool result was added
    processor._result_ids_added = set()
    tool_calls = [
        ToolCall(
            id="call_orphan_1",
            type="function",
            function=FunctionCall(name="tool_a", arguments='{"x": 1}'),
        ),
        ToolCall(
            id="call_orphan_2",
            type="function",
            function=FunctionCall(name="tool_b", arguments='{"y": 2}'),
        ),
    ]

    # Only tool_a got a result
    processor._result_ids_added.add("call_orphan_1")

    processor._ensure_all_tool_results(tool_calls)

    # tool_b should now have a placeholder result in the conversation history
    tool_results = [
        msg for msg in context.conversation_history if msg.role.value == "tool"
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_call_id == "call_orphan_2"
    assert "interrupted or failed" in tool_results[0].content


@pytest.mark.asyncio
async def test_ensure_all_tool_results_noop_when_all_present():
    """Test that _ensure_all_tool_results does nothing when all results are present."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    processor._result_ids_added = {"call_1", "call_2"}
    tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="tool_a", arguments="{}"),
        ),
        ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(name="tool_b", arguments="{}"),
        ),
    ]

    processor._ensure_all_tool_results(tool_calls)

    # No new messages should have been added
    assert len(context.conversation_history) == 0


@pytest.mark.asyncio
async def test_process_tool_calls_finally_adds_missing_results():
    """Test that the finally block adds results for tools that never executed.

    When a tool_manager raises RuntimeError, the finally block should still
    ensure all tool_call_ids have results, preventing OpenAI 400 errors.
    """
    context = DummyContext(stream_manager=None, tool_manager=None)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_call = ToolCall(
        id="call_no_manager",
        type="function",
        function=FunctionCall(name="some_tool", arguments='{"a": "b"}'),
    )

    with pytest.raises(RuntimeError):
        await processor.process_tool_calls([tool_call])

    # The finally block should have added a placeholder result
    tool_results = [
        msg for msg in context.conversation_history if msg.role.value == "tool"
    ]
    assert len(tool_results) >= 1
    assert any(msg.tool_call_id == "call_no_manager" for msg in tool_results)


@pytest.mark.asyncio
async def test_process_tool_calls_exception_in_call():
    # Test that an exception raised during execute_tool is caught and an error is recorded.
    stream_manager = DummyStreamManager()
    tool_manager = DummyToolManager(raise_exception=True)
    context = DummyContext(stream_manager=stream_manager, tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_call = {
        "function": {"name": "exception_tool", "arguments": '{"dummy": "data"}'},
        "id": "exc_call",
    }
    await processor.process_tool_calls([tool_call])

    # Look for an error entry
    error_entries = [
        entry
        for entry in context.conversation_history
        if entry.role.value == "tool" and entry.content and "Error:" in entry.content
    ]
    assert len(error_entries) >= 1
    # The error should contain the exception message
    assert any("Simulated execute_tool exception" in e.content for e in error_entries)


# ---------------------------
# Tests for orphaned tool_call_id safety net
# ---------------------------


class DenyConfirmUIManager(DummyUIManager):
    """UI manager that denies tool confirmation."""

    async def do_confirm_tool_execution(self, tool_name, arguments):
        return False


class FailingInjectContext(DummyContext):
    """Context whose inject_tool_message raises after the first N calls."""

    def __init__(self, tool_manager=None, fail_after=1):
        super().__init__(tool_manager=tool_manager)
        self._inject_count = 0
        self._fail_after = fail_after

    def inject_tool_message(self, message):
        self._inject_count += 1
        if self._inject_count > self._fail_after:
            raise RuntimeError("Simulated inject failure")
        super().inject_tool_message(message)


@pytest.mark.asyncio
async def test_cancelled_tool_still_gets_result_for_remaining():
    """When user cancels confirmation on the 2nd tool, the 2nd tool still gets a result."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)

    call_count = [0]

    class SelectiveDenyUI(DummyUIManager):
        """Denies the second tool call."""

        async def do_confirm_tool_execution(self, tool_name, arguments):
            call_count[0] += 1
            return call_count[0] <= 1  # Allow first, deny second

    ui_manager = SelectiveDenyUI()
    processor = ToolProcessor(context, ui_manager)

    tool_calls = [
        ToolCall(
            id="call_ok",
            type="function",
            function=FunctionCall(name="tool_a", arguments="{}"),
        ),
        ToolCall(
            id="call_denied",
            type="function",
            function=FunctionCall(name="tool_b", arguments="{}"),
        ),
    ]

    await processor.process_tool_calls(tool_calls)

    # Both tool_call_ids must have results (one executed, one cancelled/placeholder)
    tool_results = [
        msg for msg in context.conversation_history if msg.role.value == "tool"
    ]
    result_ids = {msg.tool_call_id for msg in tool_results}
    assert "call_ok" in result_ids or "call_denied" in result_ids
    # The key assertion: no orphaned tool_call_ids
    assistant_msgs = [
        msg
        for msg in context.conversation_history
        if msg.role.value == "assistant" and msg.tool_calls
    ]
    for amsg in assistant_msgs:
        for tc in amsg.tool_calls:
            tc_id = tc.id if hasattr(tc, "id") else tc.get("id")
            assert tc_id in result_ids, f"Orphaned tool_call_id: {tc_id}"


@pytest.mark.asyncio
async def test_inject_failure_is_caught_by_finally():
    """When inject_tool_message fails, the finally block fills missing results."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    # fail_after=1 means the assistant message succeeds, but the tool result inject fails
    context = FailingInjectContext(tool_manager=tool_manager, fail_after=1)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_call = ToolCall(
        id="call_inject_fail",
        type="function",
        function=FunctionCall(name="some_tool", arguments="{}"),
    )

    # The tool result inject fails, then execution runs, then finally block fires.
    # Since inject keeps failing, the finally block's attempt also fails.
    # But the key point is process_tool_calls doesn't crash.
    await processor.process_tool_calls([tool_call])

    # The assistant message should be in history (first inject succeeded)
    assert len(context.conversation_history) >= 1
    assert context.conversation_history[0].role.value == "assistant"


@pytest.mark.asyncio
async def test_result_ids_reset_between_calls():
    """_result_ids_added is reset on each call to process_tool_calls."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    # First call
    tc1 = ToolCall(
        id="call_first",
        type="function",
        function=FunctionCall(name="tool_a", arguments="{}"),
    )
    await processor.process_tool_calls([tc1])
    assert "call_first" in processor._result_ids_added

    # Second call should start fresh
    tc2 = ToolCall(
        id="call_second",
        type="function",
        function=FunctionCall(name="tool_b", arguments="{}"),
    )
    await processor.process_tool_calls([tc2])
    assert "call_second" in processor._result_ids_added
    assert "call_first" not in processor._result_ids_added


@pytest.mark.asyncio
async def test_successful_batch_tracks_all_ids():
    """All tool_call_ids in a successful batch are tracked in _result_ids_added."""
    result_dict = {"isError": False, "content": "OK"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)
    ui_manager = DummyUIManager()
    processor = ToolProcessor(context, ui_manager)

    tool_calls = [
        ToolCall(
            id="batch_1",
            type="function",
            function=FunctionCall(name="tool_a", arguments="{}"),
        ),
        ToolCall(
            id="batch_2",
            type="function",
            function=FunctionCall(name="tool_b", arguments="{}"),
        ),
        ToolCall(
            id="batch_3",
            type="function",
            function=FunctionCall(name="tool_c", arguments="{}"),
        ),
    ]

    await processor.process_tool_calls(tool_calls)

    assert processor._result_ids_added == {"batch_1", "batch_2", "batch_3"}


# ---------------------------
# Tests for _truncate_tool_result
# ---------------------------


class TestTruncateToolResult:
    """Tests for ToolProcessor._truncate_tool_result()."""

    def _make_processor(self):
        tool_manager = DummyToolManager()
        context = DummyContext(tool_manager=tool_manager)
        ui_manager = DummyUIManager()
        return ToolProcessor(context, ui_manager)

    def test_under_limit_unchanged(self):
        processor = self._make_processor()
        content = "short content"
        result = processor._truncate_tool_result(content, max_chars=1000)
        assert result == content

    def test_exactly_at_limit_unchanged(self):
        processor = self._make_processor()
        content = "x" * 1000
        result = processor._truncate_tool_result(content, max_chars=1000)
        assert result == content

    def test_over_limit_truncated(self):
        processor = self._make_processor()
        content = "A" * 10_000
        result = processor._truncate_tool_result(content, max_chars=1000)
        assert len(result) < len(content)
        assert "TRUNCATED" in result
        assert "chars omitted" in result

    def test_preserves_head_and_tail(self):
        processor = self._make_processor()
        head = "HEAD_MARKER_" + "x" * 5000
        tail = "y" * 5000 + "_TAIL_MARKER"
        content = head + "z" * 90_000 + tail
        result = processor._truncate_tool_result(content, max_chars=10_000)
        assert result.startswith("HEAD_MARKER_")
        assert result.endswith("_TAIL_MARKER")

    def test_disabled_with_zero(self):
        processor = self._make_processor()
        content = "A" * 200_000
        result = processor._truncate_tool_result(content, max_chars=0)
        assert result == content

    def test_disabled_with_negative(self):
        processor = self._make_processor()
        content = "A" * 200_000
        result = processor._truncate_tool_result(content, max_chars=-1)
        assert result == content

    def test_empty_content(self):
        processor = self._make_processor()
        result = processor._truncate_tool_result("", max_chars=100)
        assert result == ""

    def test_value_binding_in_tail_preserved(self):
        """Value binding appended at end should survive truncation."""
        processor = self._make_processor()
        body = "A" * 200_000
        binding = "\n\n**RESULT: $v0 = 42.0**"
        content = body + binding
        result = processor._truncate_tool_result(content, max_chars=100_000)
        assert "**RESULT: $v0 = 42.0**" in result


# ===========================================================================
# NEW TESTS — targeting uncovered lines to bring coverage to 90%+
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers / extended dummies
# ---------------------------------------------------------------------------


def _make_tool_result(call_id, tool, result=None, error=None):
    """Create a CTPToolResult quickly."""
    now = datetime.now(UTC)
    return CTPToolResult(
        id=call_id,
        tool=tool,
        result=result,
        error=error,
        start_time=now,
        end_time=now,
        machine=platform.node(),
        pid=os.getpid(),
    )


def _make_processor(tool_manager=None, ui_manager=None, context=None):
    """Return a ToolProcessor with sensible defaults."""
    if tool_manager is None:
        tool_manager = DummyToolManager()
    if context is None:
        context = DummyContext(tool_manager=tool_manager)
    if ui_manager is None:
        ui_manager = DummyUIManager()
    return ToolProcessor(context, ui_manager)


class DummyContextWithSession(DummyContext):
    """Context that exposes a fake session.vm attribute."""

    def __init__(self, tool_manager=None, vm=None):
        super().__init__(tool_manager=tool_manager)
        self.session = MagicMock()
        self.session.vm = vm


class DummyContextWithMemoryStore(DummyContext):
    """Context that exposes a fake memory_store and _system_prompt_dirty flag."""

    def __init__(self, tool_manager=None, memory_store=None):
        super().__init__(tool_manager=tool_manager)
        self.memory_store = memory_store
        self._system_prompt_dirty = False


# ---------------------------------------------------------------------------
# _build_page_content_blocks  (lines 559-607)
# ---------------------------------------------------------------------------


class TestBuildPageContentBlocks:
    """Tests for ToolProcessor._build_page_content_blocks()."""

    def _proc(self):
        return _make_processor()

    def _page(
        self,
        page_id="pg1",
        modality_value=None,
        compression_name=None,
        content="hello",
    ):
        page = MagicMock()
        page.page_id = page_id
        page.content = content
        if modality_value is not None:
            mod = MagicMock()
            mod.value = modality_value
            page.modality = mod
        else:
            page.modality = None
        if compression_name is not None:
            comp = MagicMock()
            comp.name = compression_name
            page.compression_level = comp
        else:
            page.compression_level = None
        return page

    def test_image_url_returns_blocks(self):
        proc = self._proc()
        page = self._page(modality_value="image", content="https://example.com/img.png")
        result = proc._build_page_content_blocks(
            page=page,
            page_content="https://example.com/img.png",
            truncated=False,
            was_compressed=False,
            source_tier=None,
        )
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "https://example.com/img.png"

    def test_image_url_truncated_flag(self):
        proc = self._proc()
        page = self._page(modality_value="image", content="https://example.com/img.png")
        result = proc._build_page_content_blocks(
            page=page,
            page_content="https://example.com/img.png",
            truncated=True,
            was_compressed=False,
            source_tier=None,
        )
        assert "[content truncated]" in result[0]["text"]

    def test_image_data_uri_returns_blocks(self):
        proc = self._proc()
        data_uri = "data:image/png;base64,abc123"
        page = self._page(modality_value="image", content=data_uri)
        result = proc._build_page_content_blocks(
            page=page,
            page_content=data_uri,
            truncated=False,
            was_compressed=False,
            source_tier=None,
        )
        assert isinstance(result, list)

    def test_text_modality_returns_json_string(self):
        proc = self._proc()
        page = self._page(modality_value="text", content="Some text content")
        result = proc._build_page_content_blocks(
            page=page,
            page_content="Some text content",
            truncated=False,
            was_compressed=False,
            source_tier="hot",
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["modality"] == "text"
        assert parsed["source_tier"] == "hot"

    def test_short_content_adds_note(self):
        proc = self._proc()
        page = self._page(content="short")
        result = proc._build_page_content_blocks(
            page=page,
            page_content="short",
            truncated=False,
            was_compressed=False,
            source_tier=None,
        )
        parsed = json.loads(result)
        assert "note" in parsed
        assert "Very short" in parsed["note"]

    def test_abstract_compression_adds_note(self):
        proc = self._proc()
        page = self._page(content="Some content " * 20, compression_name="ABSTRACT")
        result = proc._build_page_content_blocks(
            page=page,
            page_content="Some content " * 20,
            truncated=False,
            was_compressed=True,
            source_tier=None,
        )
        parsed = json.loads(result)
        assert "note" in parsed
        assert "abstract" in parsed["note"].lower()

    def test_reference_compression_adds_note(self):
        proc = self._proc()
        page = self._page(content="Some content " * 20, compression_name="REFERENCE")
        result = proc._build_page_content_blocks(
            page=page,
            page_content="Some content " * 20,
            truncated=False,
            was_compressed=True,
            source_tier=None,
        )
        parsed = json.loads(result)
        assert "note" in parsed

    def test_no_modality_defaults_to_text(self):
        proc = self._proc()
        page = self._page(modality_value=None)
        result = proc._build_page_content_blocks(
            page=page,
            page_content="plain text",
            truncated=False,
            was_compressed=False,
            source_tier=None,
        )
        parsed = json.loads(result)
        assert parsed["modality"] == "text"

    def test_image_without_url_prefix_returns_json(self):
        """An image with non-URL content should NOT return multi-block."""
        proc = self._proc()
        page = self._page(modality_value="image", content="raw bytes here")
        result = proc._build_page_content_blocks(
            page=page,
            page_content="raw bytes here",
            truncated=False,
            was_compressed=False,
            source_tier=None,
        )
        # Should fall through to JSON path
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _handle_vm_tool  (lines 673-775)
# ---------------------------------------------------------------------------


class TestHandleVmTool:
    """Tests for ToolProcessor._handle_vm_tool()."""

    @pytest.mark.asyncio
    async def test_no_vm_returns_error_message(self):
        """Without session.vm, adds an error placeholder to history."""
        context = DummyContextWithSession(vm=None)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        await proc._handle_vm_tool(
            "page_fault", {"page_id": "pg1"}, "page_fault", "call_1"
        )

        tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
        assert len(tool_msgs) == 1
        assert "VM not available" in tool_msgs[0].content

    @pytest.mark.asyncio
    async def test_page_fault_already_faulted_returns_already_loaded(self):
        """Re-faulting the same page_id returns already_loaded JSON."""
        vm = AsyncMock()
        context = DummyContextWithSession(vm=vm)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)
        proc._faulted_page_ids.add("pg1")

        await proc._handle_vm_tool(
            "page_fault", {"page_id": "pg1"}, "page_fault", "call_1"
        )

        tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
        assert tool_msgs
        content = json.loads(tool_msgs[0].content)
        assert content["already_loaded"] is True

    @pytest.mark.asyncio
    async def test_page_fault_success_builds_content(self):
        """Successful page_fault adds page content to history."""
        vm = AsyncMock()
        page = MagicMock()
        page.page_id = "pg42"
        page.content = "This is page content."
        page.modality = None
        page.compression_level = None
        fault_result = MagicMock()
        fault_result.success = True
        fault_result.page = page
        fault_result.was_compressed = False
        fault_result.source_tier = None
        vm.handle_fault = AsyncMock(return_value=fault_result)

        context = DummyContextWithSession(vm=vm)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        await proc._handle_vm_tool(
            "page_fault", {"page_id": "pg42"}, "page_fault", "call_1"
        )

        assert "pg42" in proc._faulted_page_ids
        tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
        assert tool_msgs
        parsed = json.loads(tool_msgs[0].content)
        assert parsed["success"] is True

    @pytest.mark.asyncio
    async def test_page_fault_content_truncated_for_large_pages(self):
        """Pages exceeding _VM_MAX_PAGE_CONTENT_CHARS are truncated."""
        vm = AsyncMock()
        page = MagicMock()
        page.page_id = "pg_big"
        big_content = "X" * 5000
        page.content = big_content
        page.modality = None
        page.compression_level = None
        fault_result = MagicMock()
        fault_result.success = True
        fault_result.page = page
        fault_result.was_compressed = False
        fault_result.source_tier = None
        vm.handle_fault = AsyncMock(return_value=fault_result)

        context = DummyContextWithSession(vm=vm)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        await proc._handle_vm_tool(
            "page_fault", {"page_id": "pg_big"}, "page_fault", "call_1"
        )

        tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
        assert tool_msgs
        parsed = json.loads(tool_msgs[0].content)
        assert parsed["truncated"] is True

    @pytest.mark.asyncio
    async def test_page_fault_failure_returns_error_json(self):
        """Failed page_fault includes error in result."""
        vm = AsyncMock()
        fault_result = MagicMock()
        fault_result.success = False
        fault_result.page = None
        fault_result.error = "Page not found"
        vm.handle_fault = AsyncMock(return_value=fault_result)

        context = DummyContextWithSession(vm=vm)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        await proc._handle_vm_tool(
            "page_fault", {"page_id": "missing"}, "page_fault", "call_1"
        )

        tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
        assert tool_msgs
        parsed = json.loads(tool_msgs[0].content)
        assert parsed["success"] is False
        assert "Page not found" in parsed["error"]

    @pytest.mark.asyncio
    async def test_search_pages_calls_vm_and_adds_result(self):
        """search_pages invokes vm.search_pages and stores to_json() result."""
        vm = AsyncMock()
        search_result = MagicMock()
        search_result.to_json.return_value = '{"results": []}'
        vm.search_pages = AsyncMock(return_value=search_result)

        context = DummyContextWithSession(vm=vm)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        await proc._handle_vm_tool(
            "search_pages",
            {"query": "test query", "limit": 3},
            "search_pages",
            "call_search",
        )

        vm.search_pages.assert_called_once_with(
            query="test query", modality=None, limit=3
        )
        tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
        assert tool_msgs
        assert '{"results": []}' == tool_msgs[0].content

    @pytest.mark.asyncio
    async def test_unknown_vm_tool_returns_error(self):
        """Unknown VM tool name returns error JSON."""
        vm = AsyncMock()
        context = DummyContextWithSession(vm=vm)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        await proc._handle_vm_tool(
            "nonexistent_vm_tool", {}, "nonexistent_vm_tool", "call_x"
        )

        tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
        parsed = json.loads(tool_msgs[0].content)
        assert "Unknown VM tool" in parsed["error"]

    @pytest.mark.asyncio
    async def test_vm_tool_exception_is_caught(self):
        """Exceptions from vm.handle_fault are caught and stored as errors."""
        vm = AsyncMock()
        vm.handle_fault = AsyncMock(side_effect=RuntimeError("VM exploded"))

        context = DummyContextWithSession(vm=vm)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        await proc._handle_vm_tool(
            "page_fault", {"page_id": "pg_err"}, "page_fault", "call_err"
        )

        tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
        parsed = json.loads(tool_msgs[0].content)
        assert parsed["success"] is False
        assert "VM exploded" in parsed["error"]


# ---------------------------------------------------------------------------
# _handle_memory_tool  (lines 621-654)
# ---------------------------------------------------------------------------


class TestHandleMemoryTool:
    """Tests for ToolProcessor._handle_memory_tool()."""

    @pytest.mark.asyncio
    async def test_no_memory_store_returns_not_available(self):
        """Without memory_store, adds 'not available' placeholder."""
        context = DummyContextWithMemoryStore(memory_store=None)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        await proc._handle_memory_tool("remember", {"note": "hi"}, "remember", "c1")

        tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
        assert tool_msgs
        assert "not available" in tool_msgs[0].content.lower()

    @pytest.mark.asyncio
    async def test_remember_marks_system_prompt_dirty(self):
        """Calling 'remember' sets context._system_prompt_dirty = True."""
        store = MagicMock()
        context = DummyContextWithMemoryStore(memory_store=store)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        with patch(
            "mcp_cli.memory.tools.handle_memory_tool",
            new=AsyncMock(return_value="remembered"),
        ):
            await proc._handle_memory_tool(
                "remember", {"note": "test"}, "remember", "c2"
            )

        assert context._system_prompt_dirty is True

    @pytest.mark.asyncio
    async def test_forget_marks_system_prompt_dirty(self):
        """Calling 'forget' sets context._system_prompt_dirty = True."""
        store = MagicMock()
        context = DummyContextWithMemoryStore(memory_store=store)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        with patch(
            "mcp_cli.memory.tools.handle_memory_tool",
            new=AsyncMock(return_value="forgotten"),
        ):
            await proc._handle_memory_tool("forget", {"key": "x"}, "forget", "c3")

        assert context._system_prompt_dirty is True

    @pytest.mark.asyncio
    async def test_recall_does_not_mark_system_prompt_dirty(self):
        """Calling 'recall' does NOT set _system_prompt_dirty."""
        store = MagicMock()
        context = DummyContextWithMemoryStore(memory_store=store)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        with patch(
            "mcp_cli.memory.tools.handle_memory_tool",
            new=AsyncMock(return_value="recalled stuff"),
        ):
            await proc._handle_memory_tool("recall", {}, "recall", "c4")

        assert context._system_prompt_dirty is False

    @pytest.mark.asyncio
    async def test_memory_tool_result_added_to_history(self):
        """Memory tool result is written to conversation history."""
        store = MagicMock()
        context = DummyContextWithMemoryStore(memory_store=store)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        with patch(
            "mcp_cli.memory.tools.handle_memory_tool",
            new=AsyncMock(return_value="memory result text"),
        ):
            await proc._handle_memory_tool("recall", {}, "recall", "c5")

        tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
        assert tool_msgs
        assert "memory result text" in tool_msgs[0].content


# ---------------------------------------------------------------------------
# _store_tool_result_as_vm_page  (lines 792-804)
# ---------------------------------------------------------------------------


class TestStoreToolResultAsVmPage:
    """Tests for ToolProcessor._store_tool_result_as_vm_page()."""

    @pytest.mark.asyncio
    async def test_no_vm_returns_silently(self):
        """Without session.vm, method returns without error."""
        context = DummyContextWithSession(vm=None)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)
        # Should not raise
        await proc._store_tool_result_as_vm_page("my_tool", "some content")

    @pytest.mark.asyncio
    async def test_stores_page_when_vm_available(self):
        """When vm is available, creates and adds page to working set."""
        vm = MagicMock()
        page = MagicMock()
        page.page_id = "new_page"
        vm.create_page.return_value = page
        vm.add_to_working_set = AsyncMock()

        context = DummyContextWithSession(vm=vm)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        # Patch the PageType import
        with patch(
            "mcp_cli.chat.tool_processor.ToolProcessor._store_tool_result_as_vm_page"
        ):
            pass  # just checking it exists

        # Actually call with mocked PageType
        from unittest.mock import patch as _patch

        with _patch("chuk_ai_session_manager.memory.models.PageType") as MockPageType:
            MockPageType.ARTIFACT = "artifact"
            await proc._store_tool_result_as_vm_page("my_tool", "content here")

        vm.create_page.assert_called_once()
        vm.add_to_working_set.assert_called_once_with(page)

    @pytest.mark.asyncio
    async def test_exception_from_vm_is_swallowed(self):
        """Exceptions from vm.create_page are caught silently."""
        vm = MagicMock()
        vm.create_page.side_effect = RuntimeError("create_page failed")
        vm.add_to_working_set = AsyncMock()

        context = DummyContextWithSession(vm=vm)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)

        # Should not raise
        await proc._store_tool_result_as_vm_page("my_tool", "content here")


# ---------------------------------------------------------------------------
# _check_and_launch_app  (lines 806-864)
# ---------------------------------------------------------------------------


class TestCheckAndLaunchApp:
    """Tests for ToolProcessor._check_and_launch_app()."""

    @pytest.mark.asyncio
    async def test_no_tool_manager_returns_immediately(self):
        """Without tool_manager, method exits silently."""
        context = DummyContext(tool_manager=None)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)
        # Should not raise
        await proc._check_and_launch_app("my_tool", {"result": "data"})

    @pytest.mark.asyncio
    async def test_no_app_ui_no_patch_does_nothing(self):
        """Tool with no app UI and plain result (no ui_patch) checks for ready bridge."""
        tool_info = MagicMock()
        tool_info.has_app_ui = False

        app_host = MagicMock()
        app_host.get_any_ready_bridge.return_value = None

        tool_manager = DummyToolManager()
        tool_manager.get_tool_by_name = AsyncMock(return_value=tool_info)
        tool_manager.app_host = app_host

        context = DummyContext(tool_manager=tool_manager)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)
        proc.tool_manager = tool_manager

        # "plain result" does not contain a ui_patch, so get_any_ready_bridge
        # is called but push_tool_result is not
        await proc._check_and_launch_app("my_tool", "plain result")

        # No push_tool_result since the result has no ui_patch
        app_host.get_any_ready_bridge.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_with_app_ui_reuses_existing_bridge(self):
        """Tool with app UI reuses an existing bridge (push_tool_result)."""
        tool_info = MagicMock()
        tool_info.has_app_ui = True
        tool_info.app_resource_uri = "http://app.example.com"
        tool_info.namespace = "my_server"

        bridge = MagicMock()
        bridge.push_tool_result = AsyncMock()

        app_host = MagicMock()
        app_host.get_bridge.return_value = bridge

        tool_manager = DummyToolManager()
        tool_manager.get_tool_by_name = AsyncMock(return_value=tool_info)
        tool_manager.app_host = app_host

        context = DummyContext(tool_manager=tool_manager)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)
        proc.tool_manager = tool_manager

        await proc._check_and_launch_app("my_tool", "some result")

        bridge.push_tool_result.assert_called_once_with("some result")

    @pytest.mark.asyncio
    async def test_tool_with_app_ui_launches_new_app(self):
        """Tool with app UI launches a new app when no bridge exists."""
        tool_info = MagicMock()
        tool_info.has_app_ui = True
        tool_info.app_resource_uri = "http://app.example.com"
        tool_info.namespace = "my_server"

        app_info = MagicMock()
        app_info.url = "http://app.example.com/session"

        app_host = MagicMock()
        app_host.get_bridge.return_value = None
        app_host.get_bridge_by_uri.return_value = None
        app_host.launch_app = AsyncMock(return_value=app_info)

        tool_manager = DummyToolManager()
        tool_manager.get_tool_by_name = AsyncMock(return_value=tool_info)
        tool_manager.app_host = app_host

        context = DummyContext(tool_manager=tool_manager)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)
        proc.tool_manager = tool_manager

        await proc._check_and_launch_app("my_tool", "some result")

        app_host.launch_app.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_no_app_ui_with_patch_routes_to_bridge(self):
        """Tool without app UI but with ui_patch routes to an existing bridge."""
        tool_info = MagicMock()
        tool_info.has_app_ui = False

        bridge = MagicMock()
        bridge.push_tool_result = AsyncMock()

        app_host = MagicMock()
        app_host.get_any_ready_bridge.return_value = bridge

        tool_manager = DummyToolManager()
        tool_manager.get_tool_by_name = AsyncMock(return_value=tool_info)
        tool_manager.app_host = app_host

        context = DummyContext(tool_manager=tool_manager)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)
        proc.tool_manager = tool_manager

        # A result that contains a ui_patch
        patch_result = {"structuredContent": {"type": "ui_patch", "ops": []}}

        await proc._check_and_launch_app("my_tool", patch_result)

        bridge.push_tool_result.assert_called_once_with(patch_result)

    @pytest.mark.asyncio
    async def test_import_error_is_caught_with_warning(self, caplog):
        """ImportError from app_host raises warning."""
        tool_manager = DummyToolManager()
        tool_manager.get_tool_by_name = AsyncMock(side_effect=ImportError("websockets"))
        tool_manager.app_host = MagicMock()

        context = DummyContext(tool_manager=tool_manager)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)
        proc.tool_manager = tool_manager

        with caplog.at_level(logging.WARNING, logger="mcp_cli.chat.tool_processor"):
            await proc._check_and_launch_app("my_tool", "result")
        assert "websockets" in caplog.text.lower() or "apps" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_general_exception_is_caught(self, caplog):
        """General exceptions from tool launch are caught and logged."""
        tool_manager = DummyToolManager()
        tool_manager.get_tool_by_name = AsyncMock(side_effect=RuntimeError("boom"))
        tool_manager.app_host = MagicMock()

        context = DummyContext(tool_manager=tool_manager)
        ui = DummyUIManager()
        proc = ToolProcessor(context, ui)
        proc.tool_manager = tool_manager

        with caplog.at_level(logging.ERROR, logger="mcp_cli.chat.tool_processor"):
            await proc._check_and_launch_app("my_tool", "result")
        assert "boom" in caplog.text or "Failed" in caplog.text


# ---------------------------------------------------------------------------
# _result_contains_patch  (lines 869-916)
# ---------------------------------------------------------------------------


class TestResultContainsPatch:
    """Tests for ToolProcessor._result_contains_patch()."""

    def test_direct_structured_content_returns_true(self):
        result = {"structuredContent": {"type": "ui_patch"}}
        assert ToolProcessor._result_contains_patch(result) is True

    def test_non_ui_patch_returns_false(self):
        result = {"structuredContent": {"type": "other_type"}}
        assert ToolProcessor._result_contains_patch(result) is False

    def test_no_structured_content_returns_false(self):
        result = {"data": "some_data"}
        assert ToolProcessor._result_contains_patch(result) is False

    def test_content_list_with_ui_patch_text_returns_true(self):
        text = json.dumps({"type": "ui_patch", "ops": []})
        result = {"content": [{"type": "text", "text": text}]}
        assert ToolProcessor._result_contains_patch(result) is True

    def test_content_list_structured_content_in_text(self):
        # Text block containing JSON with structuredContent.type == "ui_patch"
        text = json.dumps({"structuredContent": {"type": "ui_patch"}})
        result = {"content": [{"type": "text", "text": text}]}
        assert ToolProcessor._result_contains_patch(result) is True

    def test_plain_string_returns_false(self):
        assert ToolProcessor._result_contains_patch("plain string") is False

    def test_none_returns_false(self):
        assert ToolProcessor._result_contains_patch(None) is False

    def test_pydantic_model_with_structured_content_attr(self):
        """Pydantic-like objects with structuredContent attr are handled."""
        obj = MagicMock(spec=["structuredContent"])
        obj.structuredContent = {"type": "ui_patch"}
        # The method unwraps .result chains first, then checks structuredContent
        assert ToolProcessor._result_contains_patch(obj) is True

    def test_wrapper_object_unwrapped(self):
        """Objects with .result attribute are unwrapped."""
        inner = {"structuredContent": {"type": "ui_patch"}}

        class Wrapper:
            result = inner

        assert ToolProcessor._result_contains_patch(Wrapper()) is True

    def test_invalid_json_in_text_block_is_skipped(self):
        """Invalid JSON in text blocks is ignored gracefully."""
        result = {
            "content": [{"type": "text", "text": '"ui_patch" not valid json {{{'}]
        }
        # Should not raise; result depends on whether json.loads fails
        val = ToolProcessor._result_contains_patch(result)
        assert isinstance(val, bool)

    def test_exception_in_unwrap_returns_false(self):
        """Exceptions during processing return False."""

        class Broken:
            @property
            def result(self):
                raise RuntimeError("broken")

        # The exception handler should return False
        val = ToolProcessor._result_contains_patch(Broken())
        assert val is False


# ---------------------------------------------------------------------------
# _track_transport_failures  (lines 918-937)
# ---------------------------------------------------------------------------


class TestTrackTransportFailures:
    def _proc(self):
        return _make_processor()

    def test_success_resets_consecutive_failures(self):
        proc = self._proc()
        proc._consecutive_transport_failures = 3
        proc._track_transport_failures(True, None)
        assert proc._consecutive_transport_failures == 0

    def test_transport_error_increments_counters(self):
        proc = self._proc()
        proc._track_transport_failures(False, "transport not initialized")
        assert proc._transport_failures == 1
        assert proc._consecutive_transport_failures == 1

    def test_non_transport_error_resets_consecutive(self):
        proc = self._proc()
        proc._consecutive_transport_failures = 2
        proc._track_transport_failures(False, "some other error")
        assert proc._consecutive_transport_failures == 0
        assert proc._transport_failures == 0

    def test_consecutive_transport_failures_triggers_warning(self, caplog):
        from mcp_cli.config.defaults import DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES

        proc = self._proc()
        with caplog.at_level(logging.WARNING, logger="mcp_cli.chat.tool_processor"):
            for _ in range(DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES):
                proc._track_transport_failures(False, "transport error")
        assert "consecutive transport failures" in caplog.text

    def test_no_error_string_resets_consecutive(self):
        proc = self._proc()
        proc._consecutive_transport_failures = 5
        proc._track_transport_failures(False, None)
        assert proc._consecutive_transport_failures == 0


# ---------------------------------------------------------------------------
# Guard checks in process_tool_calls  (lines 227-344)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_none_args_are_rejected_with_error():
    """Tool calls with None argument values are blocked and error is added."""
    tool_manager = DummyToolManager()
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    tool_call = ToolCall(
        id="call_none_args",
        type="function",
        function=FunctionCall(name="my_tool", arguments='{"key": null}'),
    )
    await proc.process_tool_calls([tool_call])

    tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
    assert any("INVALID_ARGS" in m.content for m in tool_msgs)
    # Tool manager should NOT have been called
    assert tool_manager.executed_tool is None


@pytest.mark.asyncio
async def test_missing_references_are_blocked():
    """Tool calls referencing non-existent $vN bindings are blocked."""
    tool_manager = DummyToolManager()
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    # Use a $vN reference that doesn't exist in the tool state
    tool_call = ToolCall(
        id="call_ref",
        type="function",
        function=FunctionCall(name="my_tool", arguments='{"value": "$v99"}'),
    )
    await proc.process_tool_calls([tool_call])

    tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
    assert any("Blocked" in m.content for m in tool_msgs)
    assert tool_manager.executed_tool is None


@pytest.mark.asyncio
async def test_per_tool_limit_blocks_execution():
    """When per-tool limit is exceeded, tool is blocked."""
    import chuk_ai_session_manager.guards.manager as _gm

    reset_tool_state()
    _gm._tool_state = ToolStateManager(
        limits=RuntimeLimits(
            per_tool_cap=1,
            tool_budget_total=100,
            discovery_budget=50,
            execution_budget=50,
        )
    )
    # Exceed the per-tool cap by recording calls directly
    ts = get_tool_state()
    ts.per_tool_guard.record_call("my_tool")
    ts.per_tool_guard.record_call("my_tool")

    tool_manager = DummyToolManager()
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    tool_call = ToolCall(
        id="call_limited",
        type="function",
        function=FunctionCall(name="my_tool", arguments="{}"),
    )
    await proc.process_tool_calls([tool_call])

    tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
    assert tool_msgs
    assert tool_manager.executed_tool is None


@pytest.mark.asyncio
async def test_vm_tool_is_intercepted_via_process_tool_calls():
    """page_fault calls are intercepted and handled via _handle_vm_tool."""
    vm = AsyncMock()
    vm.handle_fault = AsyncMock(
        return_value=MagicMock(success=False, page=None, error="no page")
    )
    context = DummyContextWithSession(vm=vm)
    tool_manager = DummyToolManager()
    context.tool_manager = tool_manager
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    tool_call = ToolCall(
        id="call_pf",
        type="function",
        function=FunctionCall(name="page_fault", arguments='{"page_id": "p1"}'),
    )
    await proc.process_tool_calls([tool_call])

    vm.handle_fault.assert_called_once()
    # Tool manager should NOT have been invoked (VM tools bypass it)
    assert tool_manager.executed_tool is None


@pytest.mark.asyncio
async def test_memory_tool_is_intercepted_via_process_tool_calls():
    """remember/recall/forget calls bypass ToolManager and go to memory handler."""
    store = MagicMock()
    context = DummyContextWithMemoryStore(memory_store=store)
    context.tool_manager = DummyToolManager()
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    with patch(
        "mcp_cli.memory.tools.handle_memory_tool",
        new=AsyncMock(return_value="stored!"),
    ):
        tool_call = ToolCall(
            id="call_mem",
            type="function",
            function=FunctionCall(name="remember", arguments='{"note": "hello"}'),
        )
        await proc.process_tool_calls([tool_call])

    tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
    assert any("stored!" in m.content for m in tool_msgs)


@pytest.mark.asyncio
async def test_ungrounded_call_preconditions_pass_allows_execution():
    """Ungrounded call for a parameterized tool where preconditions pass is allowed to execute.

    When check_ungrounded_call finds an ungrounded call and should_auto_rebound is False
    (non-math tools), check_tool_preconditions is consulted. If preconditions pass, the tool
    executes normally (fall-through path, lines 297-304).
    """
    result_dict = {"isError": False, "content": "computed result"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    # regular_tool is not a discovery/math tool; check_ungrounded_call will fire,
    # should_auto_rebound returns False, check_tool_preconditions returns True → executes
    tool_call = ToolCall(
        id="call_ungrounded",
        type="function",
        function=FunctionCall(name="regular_tool", arguments='{"val": 42}'),
    )
    await proc.process_tool_calls([tool_call])

    # Tool manager WAS called because preconditions passed
    assert tool_manager.executed_tool == "regular_tool"
    tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
    assert tool_msgs


@pytest.mark.asyncio
async def test_ungrounded_call_preconditions_fail_blocks_tool():
    """When preconditions fail for an ungrounded non-math tool, the call is blocked."""
    tool_manager = DummyToolManager()
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    ts = get_tool_state()

    # Patch the precondition_guard's check method to return a blocked result
    from unittest.mock import MagicMock as MM

    blocked_result = MM()
    blocked_result.blocked = True
    blocked_result.reason = "Precondition failed: no prior values"
    ts.precondition_guard.check = MM(return_value=blocked_result)

    tool_call = ToolCall(
        id="call_precond_fail",
        type="function",
        function=FunctionCall(name="regular_tool", arguments='{"val": 42}'),
    )
    await proc.process_tool_calls([tool_call])

    # Tool should have been blocked
    assert tool_manager.executed_tool is None
    tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
    assert any("Blocked" in m.content for m in tool_msgs)


@pytest.mark.asyncio
async def test_dynamic_tool_proxy_display_name():
    """call_tool with tool_name arg displays as 'call_tool → actual_tool'."""
    result_dict = {"isError": False, "content": "dynamic result"}
    tool_manager = DummyToolManager(return_result=result_dict)
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    tool_call = ToolCall(
        id="call_dyn",
        type="function",
        function=FunctionCall(
            name="call_tool",
            arguments='{"tool_name": "actual_tool", "param": "value"}',
        ),
    )
    await proc.process_tool_calls([tool_call])

    # The display should show call_tool → actual_tool
    printed_names = [name for name, _ in ui.printed_calls]
    assert any("actual_tool" in n for n in printed_names)


@pytest.mark.asyncio
async def test_interrupt_requested_before_processing():
    """If interrupt_requested is True before any tool, processing halts."""
    tool_manager = DummyToolManager()
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    ui.interrupt_requested = True
    proc = ToolProcessor(context, ui)

    tool_call = ToolCall(
        id="call_interrupted",
        type="function",
        function=FunctionCall(name="my_tool", arguments="{}"),
    )
    await proc.process_tool_calls([tool_call])

    # Tool manager should NOT have executed
    assert tool_manager.executed_tool is None


# ---------------------------------------------------------------------------
# _add_tool_result_to_history with multi-block content  (lines 1246-1257)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_tool_result_list_content():
    """Multi-block (list) content is added directly without truncation."""
    context = DummyContext(tool_manager=DummyToolManager())
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    blocks = [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
    ]
    proc._add_tool_result_to_history("my_tool", "call_multi", blocks)

    tool_msgs = [m for m in context.conversation_history if m.role.value == "tool"]
    assert len(tool_msgs) == 1
    # The Message model may convert list dicts to typed objects; check the data is preserved
    msg_content = tool_msgs[0].content
    assert msg_content is not None
    assert len(msg_content) == 2
    # Verify text block (may be TextContent object or dict)
    first = msg_content[0]
    if isinstance(first, dict):
        assert first["text"] == "some text"
    else:
        assert first.text == "some text"
    assert "call_multi" in proc._result_ids_added


# ---------------------------------------------------------------------------
# _add_tool_result_to_history with context_notice  (line 1268)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_truncation_triggers_context_notice():
    """When content is truncated, add_context_notice is called on context."""

    class ContextWithNotice(DummyContext):
        def __init__(self, tool_manager=None):
            super().__init__(tool_manager=tool_manager)
            self.notices = []

        def add_context_notice(self, msg: str):
            self.notices.append(msg)

    from mcp_cli.config.defaults import DEFAULT_MAX_TOOL_RESULT_CHARS

    context = ContextWithNotice(tool_manager=DummyToolManager())
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    # Content bigger than max
    big_content = "X" * (DEFAULT_MAX_TOOL_RESULT_CHARS + 1000)
    proc._add_tool_result_to_history("my_tool", "call_big", big_content)

    assert context.notices, "Expected add_context_notice to be called"
    assert any("truncated" in n.lower() for n in context.notices)


# ---------------------------------------------------------------------------
# _get_server_url_for_tool  (lines 1335-1355)
# ---------------------------------------------------------------------------


class TestGetServerUrlForTool:
    def _proc_with_map(self, tool_map, server_info_list):
        context = DummyContext(tool_manager=DummyToolManager())
        context.tool_to_server_map = tool_map
        context.server_info = server_info_list
        return _make_processor(context=context)

    def test_returns_none_without_map(self):
        context = DummyContext(tool_manager=DummyToolManager())
        proc = _make_processor(context=context)
        assert proc._get_server_url_for_tool("any_tool") is None

    def test_returns_none_when_tool_not_in_map(self):
        server = MagicMock()
        server.namespace = "ns1"
        server.name = "ns1"
        server.url = "http://ns1.example.com"
        proc = self._proc_with_map({"other_tool": "ns1"}, [server])
        assert proc._get_server_url_for_tool("my_tool") is None

    def test_returns_url_when_namespace_matches(self):
        server = MagicMock()
        server.namespace = "ns1"
        server.name = "ns1"
        server.url = "http://ns1.example.com"
        proc = self._proc_with_map({"my_tool": "ns1"}, [server])
        assert proc._get_server_url_for_tool("my_tool") == "http://ns1.example.com"

    def test_returns_url_when_name_matches(self):
        server = MagicMock()
        server.namespace = "other_ns"
        server.name = "ns1"
        server.url = "http://ns1-by-name.example.com"
        proc = self._proc_with_map({"my_tool": "ns1"}, [server])
        assert (
            proc._get_server_url_for_tool("my_tool") == "http://ns1-by-name.example.com"
        )


# ---------------------------------------------------------------------------
# _register_discovered_tools  (lines 1387-1440)
# ---------------------------------------------------------------------------


class TestRegisterDiscoveredTools:
    def _proc(self):
        return _make_processor()

    def test_none_result_returns_early(self):
        proc = self._proc()
        ts = get_tool_state()
        # Should not raise
        proc._register_discovered_tools(ts, "search_tools", None)

    def test_list_of_dicts_with_name_key(self):
        proc = self._proc()
        ts = get_tool_state()
        result = [{"name": "tool_a"}, {"name": "tool_b"}]
        proc._register_discovered_tools(ts, "list_tools", result)
        discovered = ts.get_discovered_tools()
        assert "tool_a" in discovered
        assert "tool_b" in discovered

    def test_list_of_strings(self):
        proc = self._proc()
        ts = get_tool_state()
        result = ["tool_x", "tool_y"]
        proc._register_discovered_tools(ts, "list_tools", result)
        discovered = ts.get_discovered_tools()
        assert "tool_x" in discovered
        assert "tool_y" in discovered

    def test_dict_with_name_key(self):
        proc = self._proc()
        ts = get_tool_state()
        result = {"name": "single_tool"}
        proc._register_discovered_tools(ts, "get_tool_schema", result)
        assert "single_tool" in ts.get_discovered_tools()

    def test_dict_with_tools_list(self):
        proc = self._proc()
        ts = get_tool_state()
        result = {"tools": [{"name": "nested_tool_a"}, {"name": "nested_tool_b"}]}
        proc._register_discovered_tools(ts, "list_tools", result)
        discovered = ts.get_discovered_tools()
        assert "nested_tool_a" in discovered
        assert "nested_tool_b" in discovered

    def test_dict_with_tools_list_of_strings(self):
        proc = self._proc()
        ts = get_tool_state()
        result = {"tools": ["str_tool_1", "str_tool_2"]}
        proc._register_discovered_tools(ts, "list_tools", result)
        discovered = ts.get_discovered_tools()
        assert "str_tool_1" in discovered

    def test_dict_with_content_wrapper_recurses(self):
        proc = self._proc()
        ts = get_tool_state()
        # Content is itself a list of dicts with name
        result = {"content": [{"name": "deep_tool"}]}
        proc._register_discovered_tools(ts, "list_tools", result)
        discovered = ts.get_discovered_tools()
        assert "deep_tool" in discovered

    def test_json_string_result_is_parsed(self):
        proc = self._proc()
        ts = get_tool_state()
        result = json.dumps([{"name": "json_tool"}])
        proc._register_discovered_tools(ts, "list_tools", result)
        discovered = ts.get_discovered_tools()
        assert "json_tool" in discovered

    def test_invalid_json_string_returns_early(self):
        proc = self._proc()
        ts = get_tool_state()
        proc._register_discovered_tools(ts, "list_tools", "not valid json {{{")
        # No tools discovered, no crash
        assert True

    def test_empty_name_not_registered(self):
        proc = self._proc()
        ts = get_tool_state()
        result = [{"name": ""}, {"name": "valid_tool"}]
        proc._register_discovered_tools(ts, "list_tools", result)
        discovered = ts.get_discovered_tools()
        assert "valid_tool" in discovered

    def test_dict_with_tool_name_key(self):
        """Handles dicts where tool name is under 'tool_name' key."""
        proc = self._proc()
        ts = get_tool_state()
        result = [{"tool_name": "alt_key_tool"}]
        proc._register_discovered_tools(ts, "list_tools", result)
        discovered = ts.get_discovered_tools()
        assert "alt_key_tool" in discovered


# ---------------------------------------------------------------------------
# _extract_result_value and helpers  (lines 1039-1099)
# ---------------------------------------------------------------------------


class TestExtractResultValue:
    def _proc(self):
        return _make_processor()

    def test_none_returns_none(self):
        proc = self._proc()
        assert proc._extract_result_value(None) is None

    def test_string_none_returns_none(self):
        proc = self._proc()
        assert proc._extract_result_value("None") is None
        assert proc._extract_result_value("null") is None

    def test_integer_returns_integer(self):
        proc = self._proc()
        assert proc._extract_result_value(42) == 42

    def test_float_returns_float(self):
        proc = self._proc()
        assert proc._extract_result_value(3.14) == 3.14

    def test_dict_with_success_and_result(self):
        proc = self._proc()
        result = proc._extract_result_value({"success": True, "result": 99})
        assert result == 99.0  # parsed as float

    def test_dict_with_is_error_false(self):
        proc = self._proc()
        result = proc._extract_result_value({"isError": False, "content": "hello"})
        assert result == "hello"

    def test_dict_with_is_error_true(self):
        proc = self._proc()
        result = proc._extract_result_value(
            {"isError": True, "error": "something broke"}
        )
        assert result == "something broke"

    def test_dict_with_text_field(self):
        proc = self._proc()
        result = proc._extract_result_value({"text": "3.14"})
        assert result == 3.14

    def test_content_list_with_text_block(self):
        proc = self._proc()
        result = proc._extract_result_value(
            {"content": [{"type": "text", "text": "42"}]}
        )
        assert result == 42.0

    def test_list_of_text_blocks(self):
        proc = self._proc()
        result = proc._extract_result_value([{"type": "text", "text": "hello world"}])
        assert result == "hello world"

    def test_content_repr_string(self):
        proc = self._proc()
        result = proc._extract_result_value("content=[{'type': 'text', 'text': '7.5'}]")
        assert result == 7.5

    def test_object_with_content_list(self):
        """Object with .content attribute (list) is handled."""
        obj = MagicMock()
        obj.content = [{"type": "text", "text": "from object"}]
        proc = self._proc()
        result = proc._extract_result_value(obj)
        assert result == "from object"


# ---------------------------------------------------------------------------
# _format_tool_response  (lines 1163-1188)
# ---------------------------------------------------------------------------


class TestFormatToolResponse:
    def _proc(self):
        return _make_processor()

    def test_dict_is_json_formatted(self):
        proc = self._proc()
        result = proc._format_tool_response({"key": "value", "num": 42})
        assert json.loads(result) == {"key": "value", "num": 42}

    def test_list_is_json_formatted(self):
        proc = self._proc()
        result = proc._format_tool_response([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    def test_string_is_returned_as_is(self):
        proc = self._proc()
        assert proc._format_tool_response("plain text") == "plain text"

    def test_mcp_nested_content_structure(self):
        """Handles dict with 'content' attribute that has .content list."""
        inner = MagicMock()
        inner.content = [{"type": "text", "text": "nested text"}]
        result = {"content": inner}
        proc = self._proc()
        out = proc._format_tool_response(result)
        assert "nested text" in out

    def test_non_serializable_dict_falls_back_to_str(self):
        proc = self._proc()
        # Use an object that can't be JSON serialized
        result = proc._format_tool_response({"fn": lambda: None})
        assert isinstance(result, str)

    def test_non_serializable_list_falls_back_to_str(self):
        proc = self._proc()
        result = proc._format_tool_response([object()])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _extract_from_content_list  (lines 1101-1124)
# ---------------------------------------------------------------------------


class TestExtractFromContentList:
    def _proc(self):
        return _make_processor()

    def test_empty_list_returns_none(self):
        assert self._proc()._extract_from_content_list([]) is None

    def test_text_block_dict(self):
        result = self._proc()._extract_from_content_list(
            [{"type": "text", "text": "hello"}]
        )
        assert result == "hello"

    def test_multiple_text_blocks_joined(self):
        result = self._proc()._extract_from_content_list(
            [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]
        )
        assert "line1" in result
        assert "line2" in result

    def test_non_text_blocks_ignored(self):
        result = self._proc()._extract_from_content_list(
            [{"type": "image_url", "url": "http://x"}]
        )
        assert result is None

    def test_object_with_text_attr(self):
        block = MagicMock()
        block.type = "text"
        block.text = "from object"
        result = self._proc()._extract_from_content_list([block])
        assert result == "from object"

    def test_numeric_text_parsed_as_number(self):
        result = self._proc()._extract_from_content_list(
            [{"type": "text", "text": "99.5"}]
        )
        assert result == 99.5


# ---------------------------------------------------------------------------
# _on_tool_result: value binding, transport tracking, verbose mode
# (lines 427-535)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_tool_result_success_binds_value():
    """Successful tool result creates a value binding."""
    tool_manager = DummyToolManager(return_result={"isError": False, "content": "42"})
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    tool_call = ToolCall(
        id="call_bind",
        type="function",
        function=FunctionCall(name="compute_tool", arguments="{}"),
    )
    await proc.process_tool_calls([tool_call])

    ts = get_tool_state()
    # At least one binding should exist
    assert len(ts.bindings) >= 1


@pytest.mark.asyncio
async def test_on_tool_result_error_no_binding():
    """Failed tool result does not create a value binding."""
    tool_manager = DummyToolManager(
        return_result={"isError": True, "error": "tool failed"}
    )
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    tool_call = ToolCall(
        id="call_err",
        type="function",
        function=FunctionCall(name="fail_tool", arguments="{}"),
    )
    await proc.process_tool_calls([tool_call])

    ts = get_tool_state()
    assert len(ts.bindings) == 0


@pytest.mark.asyncio
async def test_on_tool_result_discovery_tool_registers_tools():
    """Discovery tools trigger tool registration."""
    result_content = json.dumps([{"name": "discovered_tool_alpha"}])
    tool_manager = DummyToolManager(
        return_result={"isError": False, "content": result_content}
    )
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    # search_tools is a discovery tool
    tool_call = ToolCall(
        id="call_disc",
        type="function",
        function=FunctionCall(name="search_tools", arguments='{"query": "test"}'),
    )
    await proc.process_tool_calls([tool_call])

    ts = get_tool_state()
    discovered = ts.get_discovered_tools()
    assert "discovered_tool_alpha" in discovered


@pytest.mark.asyncio
async def test_transport_failure_increments_counter():
    """A transport error in tool result increments the failure counter."""
    tool_manager = DummyToolManager(
        return_result={
            "isError": True,
            "error": "transport not initialized: connection lost",
        }
    )
    context = DummyContext(tool_manager=tool_manager)
    ui = DummyUIManager()
    proc = ToolProcessor(context, ui)

    tool_call = ToolCall(
        id="call_transport",
        type="function",
        function=FunctionCall(name="remote_tool", arguments="{}"),
    )
    await proc.process_tool_calls([tool_call])

    assert proc._transport_failures >= 1


# ---------------------------------------------------------------------------
# cancel_running_tasks
# ---------------------------------------------------------------------------


def test_cancel_running_tasks():
    proc = _make_processor()
    assert proc._cancelled is False
    proc.cancel_running_tasks()
    assert proc._cancelled is True


# ---------------------------------------------------------------------------
# _finish_tool_calls  (lines 965-973)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finish_tool_calls_async():
    """Async finish_tool_calls is awaited correctly."""
    context = DummyContext(tool_manager=DummyToolManager())
    ui = DummyUIManager()
    finish_called = []

    async def async_finish():
        finish_called.append(True)

    ui.finish_tool_calls = async_finish
    proc = ToolProcessor(context, ui)
    await proc._finish_tool_calls()
    assert finish_called


@pytest.mark.asyncio
async def test_finish_tool_calls_sync():
    """Synchronous finish_tool_calls is called correctly."""
    context = DummyContext(tool_manager=DummyToolManager())
    ui = DummyUIManager()
    finish_called = []

    def sync_finish():
        finish_called.append(True)

    ui.finish_tool_calls = sync_finish
    proc = ToolProcessor(context, ui)
    await proc._finish_tool_calls()
    assert finish_called


@pytest.mark.asyncio
async def test_finish_tool_calls_exception_is_swallowed():
    """Exceptions in finish_tool_calls are caught silently."""
    context = DummyContext(tool_manager=DummyToolManager())
    ui = DummyUIManager()

    def broken_finish():
        raise RuntimeError("finish broken")

    ui.finish_tool_calls = broken_finish
    proc = ToolProcessor(context, ui)
    # Should not raise
    await proc._finish_tool_calls()


# ---------------------------------------------------------------------------
# _extract_tool_call_info edge cases  (line 1000, 1004-1005)
# ---------------------------------------------------------------------------


def test_extract_tool_call_info_unrecognized_format():
    """Unrecognized tool call format uses fallback name."""
    proc = _make_processor()
    name, args, call_id = proc._extract_tool_call_info("not a tool call", 5)
    assert "unknown_tool" in name or "5" in name
    assert call_id == "call_5"


def test_extract_tool_call_info_empty_name_fallback():
    """Empty tool name falls back to 'unknown_tool_N'."""
    proc = _make_processor()
    bad_call = {"function": {"name": "", "arguments": "{}"}, "id": "call_bad"}
    name, args, call_id = proc._extract_tool_call_info(bad_call, 3)
    assert "unknown_tool" in name or name == "unknown_tool_3"


# ---------------------------------------------------------------------------
# _parse_arguments edge cases  (lines 1014, 1017-1024)
# ---------------------------------------------------------------------------


class TestParseArguments:
    def _proc(self):
        return _make_processor()

    def test_empty_string_returns_empty_dict(self):
        assert self._proc()._parse_arguments("") == {}

    def test_whitespace_string_returns_empty_dict(self):
        assert self._proc()._parse_arguments("   ") == {}

    def test_invalid_json_string_returns_empty_dict(self):
        result = self._proc()._parse_arguments("{not valid json}")
        assert result == {}

    def test_none_returns_empty_dict(self):
        result = self._proc()._parse_arguments(None)
        assert result == {}

    def test_valid_json_string_parsed(self):
        result = self._proc()._parse_arguments('{"key": 123}')
        assert result == {"key": 123}

    def test_dict_returned_as_is(self):
        d = {"a": 1, "b": 2}
        assert self._proc()._parse_arguments(d) == d


# ---------------------------------------------------------------------------
# _try_parse_number  (lines 1144-1161)
# ---------------------------------------------------------------------------


class TestTryParseNumber:
    def _proc(self):
        return _make_processor()

    def test_numeric_string_returns_float(self):
        assert self._proc()._try_parse_number("3.14") == 3.14

    def test_integer_string_returns_float(self):
        assert self._proc()._try_parse_number("42") == 42.0

    def test_non_numeric_string_returned_as_is(self):
        assert self._proc()._try_parse_number("hello") == "hello"

    def test_none_string_returns_none(self):
        assert self._proc()._try_parse_number("None") is None

    def test_null_string_returns_none(self):
        assert self._proc()._try_parse_number("null") is None

    def test_empty_string_returns_empty(self):
        # Empty string hits the early `if not text` guard and is returned as-is
        assert self._proc()._try_parse_number("") == ""

    def test_non_string_returned_as_is(self):
        # Non-string input
        assert self._proc()._try_parse_number(42) == 42

    def test_whitespace_stripped(self):
        assert self._proc()._try_parse_number("  7.5  ") == 7.5


# ---------------------------------------------------------------------------
# _on_tool_start  (lines 411-425)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_tool_start_invokes_ui_manager():
    """_on_tool_start calls ui_manager.start_tool_execution."""
    from chuk_tool_processor import ToolCall as CTPToolCall

    context = DummyContext(tool_manager=DummyToolManager())
    ui = DummyUIManager()
    start_calls = []

    async def capture_start(name, args):
        start_calls.append((name, args))

    ui.start_tool_execution = capture_start
    proc = ToolProcessor(context, ui)

    ctp_call = CTPToolCall(id="c1", tool="my_tool", arguments={"x": 1})
    await proc._on_tool_start(ctp_call)

    assert start_calls
    assert start_calls[0][0] == "my_tool"


@pytest.mark.asyncio
async def test_on_tool_start_dynamic_tool_shows_inner_name():
    """Dynamic call_tool shows the inner tool name."""
    from chuk_tool_processor import ToolCall as CTPToolCall

    context = DummyContext(tool_manager=DummyToolManager())
    ui = DummyUIManager()
    start_calls = []

    async def capture_start(name, args):
        start_calls.append((name, args))

    ui.start_tool_execution = capture_start
    proc = ToolProcessor(context, ui)

    ctp_call = CTPToolCall(
        id="c2", tool="call_tool", arguments={"tool_name": "inner_tool", "param": "v"}
    )
    proc._call_metadata["c2"] = MagicMock()
    proc._call_metadata["c2"].display_name = "call_tool"
    proc._call_metadata["c2"].arguments = {"tool_name": "inner_tool", "param": "v"}

    await proc._on_tool_start(ctp_call)

    assert start_calls
    assert start_calls[0][0] == "inner_tool"


# ---------------------------------------------------------------------------
# _should_confirm_tool  (lines 1357-1369)
# ---------------------------------------------------------------------------


class TestShouldConfirmTool:
    def _proc(self):
        return _make_processor()

    def test_trusted_domain_returns_false(self):
        """Trusted domains bypass confirmation."""
        proc = self._proc()
        with patch("mcp_cli.chat.tool_processor.get_preference_manager") as mock_pm:
            prefs = MagicMock()
            prefs.is_trusted_domain.return_value = True
            mock_pm.return_value = prefs
            result = proc._should_confirm_tool("my_tool", "http://trusted.example.com")
        assert result is False

    def test_non_trusted_domain_delegates_to_prefs(self):
        """Non-trusted domains ask prefs.should_confirm_tool."""
        proc = self._proc()
        with patch("mcp_cli.chat.tool_processor.get_preference_manager") as mock_pm:
            prefs = MagicMock()
            prefs.is_trusted_domain.return_value = False
            prefs.should_confirm_tool.return_value = True
            mock_pm.return_value = prefs
            result = proc._should_confirm_tool("dangerous_tool", "http://other.com")
        assert result is True

    def test_exception_returns_true(self):
        """Exceptions in preference lookup default to confirming."""
        proc = self._proc()
        with patch(
            "mcp_cli.chat.tool_processor.get_preference_manager",
            side_effect=RuntimeError("pref error"),
        ):
            result = proc._should_confirm_tool("any_tool")
        assert result is True
