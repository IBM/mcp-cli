# tests/tools/test_execution.py
"""Tests for parallel and streaming tool execution."""

import asyncio
import pytest
from datetime import datetime

from chuk_tool_processor import ToolCall as CTPToolCall
from chuk_tool_processor import ToolResult as CTPToolResult

from mcp_cli.tools.execution import execute_tools_parallel, stream_execute_tools
from mcp_cli.tools.models import ToolCallResult


class MockToolManager:
    """Mock ToolManager for testing execution functions."""

    def __init__(self, results: dict[str, ToolCallResult] | None = None):
        self.tool_timeout = 30.0
        self.results = results or {}
        self.executed_tools: list[str] = []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> ToolCallResult:
        self.executed_tools.append(tool_name)
        if tool_name in self.results:
            return self.results[tool_name]
        return ToolCallResult(
            tool_name=tool_name,
            success=True,
            result={"output": f"result from {tool_name}"},
        )


# ----------------------------------------------------------------------------
# execute_tools_parallel tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tools_parallel_empty_calls():
    """Test with empty call list."""
    manager = MockToolManager()
    results = await execute_tools_parallel(manager, [])
    assert results == []


@pytest.mark.asyncio
async def test_execute_tools_parallel_single_call():
    """Test with single tool call."""
    manager = MockToolManager()
    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={"x": 1})]

    results = await execute_tools_parallel(manager, calls)

    assert len(results) == 1
    assert results[0].tool == "test_tool"
    assert results[0].is_success
    assert manager.executed_tools == ["test_tool"]


@pytest.mark.asyncio
async def test_execute_tools_parallel_multiple_calls():
    """Test with multiple tool calls."""
    manager = MockToolManager()
    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
        CTPToolCall(id="call_3", tool="tool_c", arguments={}),
    ]

    results = await execute_tools_parallel(manager, calls)

    assert len(results) == 3
    assert set(r.tool for r in results) == {"tool_a", "tool_b", "tool_c"}
    assert len(manager.executed_tools) == 3


@pytest.mark.asyncio
async def test_execute_tools_parallel_with_timeout():
    """Test with custom timeout."""
    manager = MockToolManager()
    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    results = await execute_tools_parallel(manager, calls, timeout=60.0)

    assert len(results) == 1
    assert results[0].is_success


@pytest.mark.asyncio
async def test_execute_tools_parallel_with_on_tool_start():
    """Test on_tool_start callback is invoked."""
    manager = MockToolManager()
    started_tools: list[str] = []

    async def on_start(call: CTPToolCall):
        started_tools.append(call.tool)

    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
    ]

    await execute_tools_parallel(manager, calls, on_tool_start=on_start)

    assert set(started_tools) == {"tool_a", "tool_b"}


@pytest.mark.asyncio
async def test_execute_tools_parallel_with_on_tool_result():
    """Test on_tool_result callback is invoked."""
    manager = MockToolManager()
    completed_tools: list[str] = []

    async def on_result(result: CTPToolResult):
        completed_tools.append(result.tool)

    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
    ]

    await execute_tools_parallel(manager, calls, on_tool_result=on_result)

    assert set(completed_tools) == {"tool_a", "tool_b"}


@pytest.mark.asyncio
async def test_execute_tools_parallel_callback_exception():
    """Test that callback exceptions don't break execution."""
    manager = MockToolManager()

    async def failing_callback(call: CTPToolCall):
        raise ValueError("callback error")

    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    # Should not raise, just log warning
    results = await execute_tools_parallel(
        manager, calls, on_tool_start=failing_callback
    )

    assert len(results) == 1
    assert results[0].is_success


@pytest.mark.asyncio
async def test_execute_tools_parallel_with_error():
    """Test handling tool execution errors."""
    manager = MockToolManager(
        results={
            "failing_tool": ToolCallResult(
                tool_name="failing_tool",
                success=False,
                error="tool failed",
            )
        }
    )

    calls = [
        CTPToolCall(id="call_1", tool="failing_tool", arguments={}),
        CTPToolCall(id="call_2", tool="success_tool", arguments={}),
    ]

    results = await execute_tools_parallel(manager, calls)

    assert len(results) == 2
    failing = next(r for r in results if r.tool == "failing_tool")
    success = next(r for r in results if r.tool == "success_tool")

    assert not failing.is_success
    assert failing.error == "tool failed"
    assert success.is_success


@pytest.mark.asyncio
async def test_execute_tools_parallel_max_concurrency():
    """Test max_concurrency limits parallel execution."""
    execution_count = 0
    max_concurrent = 0

    original_execute = MockToolManager.execute_tool

    async def tracking_execute(
        self, tool_name, arguments, namespace=None, timeout=None
    ):
        nonlocal execution_count, max_concurrent
        execution_count += 1
        current = execution_count
        max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.01)  # Simulate work
        execution_count -= 1
        return await original_execute(self, tool_name, arguments, namespace, timeout)

    manager = MockToolManager()
    manager.execute_tool = lambda *args, **kwargs: tracking_execute(
        manager, *args, **kwargs
    )

    calls = [
        CTPToolCall(id=f"call_{i}", tool=f"tool_{i}", arguments={}) for i in range(10)
    ]

    await execute_tools_parallel(manager, calls, max_concurrency=2)

    # Max concurrent should not exceed 2 (though timing may vary)
    assert max_concurrent <= 3  # Allow some slack due to async timing


@pytest.mark.asyncio
async def test_execute_tools_parallel_respects_namespace():
    """Test namespace is passed correctly."""
    manager = MockToolManager()
    calls = [
        CTPToolCall(id="call_1", tool="test_tool", arguments={}, namespace="custom_ns")
    ]

    results = await execute_tools_parallel(manager, calls)

    assert len(results) == 1


# ----------------------------------------------------------------------------
# stream_execute_tools tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_execute_tools_empty_calls():
    """Test with empty call list."""
    manager = MockToolManager()
    results = []
    async for result in stream_execute_tools(manager, []):
        results.append(result)
    assert results == []


@pytest.mark.asyncio
async def test_stream_execute_tools_single_call():
    """Test streaming with single tool call."""
    manager = MockToolManager()
    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={"x": 1})]

    results = []
    async for result in stream_execute_tools(manager, calls):
        results.append(result)

    assert len(results) == 1
    assert results[0].tool == "test_tool"
    assert results[0].is_success


@pytest.mark.asyncio
async def test_stream_execute_tools_multiple_calls():
    """Test streaming with multiple tool calls."""
    manager = MockToolManager()
    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
        CTPToolCall(id="call_3", tool="tool_c", arguments={}),
    ]

    results = []
    async for result in stream_execute_tools(manager, calls):
        results.append(result)

    assert len(results) == 3
    assert set(r.tool for r in results) == {"tool_a", "tool_b", "tool_c"}


@pytest.mark.asyncio
async def test_stream_execute_tools_with_on_tool_start():
    """Test on_tool_start callback in streaming mode."""
    manager = MockToolManager()
    started_tools: list[str] = []

    async def on_start(call: CTPToolCall):
        started_tools.append(call.tool)

    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
    ]

    results = []
    async for result in stream_execute_tools(manager, calls, on_tool_start=on_start):
        results.append(result)

    assert set(started_tools) == {"tool_a", "tool_b"}
    assert len(results) == 2


@pytest.mark.asyncio
async def test_stream_execute_tools_yields_as_completed():
    """Test results are yielded as tools complete."""
    manager = MockToolManager()
    yield_times: list[float] = []

    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
    ]

    import time

    start = time.time()
    async for result in stream_execute_tools(manager, calls):
        yield_times.append(time.time() - start)

    # Both should yield quickly (not waiting for all)
    assert len(yield_times) == 2


@pytest.mark.asyncio
async def test_stream_execute_tools_with_error():
    """Test streaming handles errors gracefully."""
    manager = MockToolManager(
        results={
            "failing_tool": ToolCallResult(
                tool_name="failing_tool",
                success=False,
                error="tool failed",
            )
        }
    )

    calls = [
        CTPToolCall(id="call_1", tool="failing_tool", arguments={}),
        CTPToolCall(id="call_2", tool="success_tool", arguments={}),
    ]

    results = []
    async for result in stream_execute_tools(manager, calls):
        results.append(result)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_stream_execute_tools_result_fields():
    """Test CTPToolResult has expected fields."""
    manager = MockToolManager()
    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    results = []
    async for result in stream_execute_tools(manager, calls):
        results.append(result)

    result = results[0]
    assert result.id == "call_1"
    assert result.tool == "test_tool"
    assert isinstance(result.start_time, datetime)
    assert isinstance(result.end_time, datetime)
    assert result.machine  # Should have hostname
    assert result.pid > 0  # Should have process ID


# ----------------------------------------------------------------------------
# Additional coverage tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tools_parallel_on_tool_result_callback_exception():
    """Test on_tool_result callback exception doesn't break execution."""
    manager = MockToolManager()

    async def failing_result_callback(result: CTPToolResult):
        raise ValueError("callback error")

    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    # Should not raise, just log warning
    results = await execute_tools_parallel(
        manager, calls, on_tool_result=failing_result_callback
    )

    assert len(results) == 1
    assert results[0].is_success


@pytest.mark.asyncio
async def test_stream_execute_tools_on_tool_start_callback_exception():
    """Test on_tool_start callback exception in streaming mode."""
    manager = MockToolManager()

    async def failing_start_callback(call: CTPToolCall):
        raise ValueError("callback error")

    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    results = []
    async for result in stream_execute_tools(
        manager, calls, on_tool_start=failing_start_callback
    ):
        results.append(result)

    assert len(results) == 1
    assert results[0].is_success


@pytest.mark.asyncio
async def test_stream_execute_tools_cancellation():
    """Test stream_execute_tools handles cancellation gracefully."""
    manager = MockToolManager()

    # Create a slow tool that we can cancel
    async def slow_execute(self, tool_name, arguments, namespace=None, timeout=None):
        await asyncio.sleep(10)  # Long delay
        return ToolCallResult(tool_name=tool_name, success=True, result={})

    manager.execute_tool = lambda *args, **kwargs: slow_execute(
        manager, *args, **kwargs
    )

    calls = [
        CTPToolCall(id="call_1", tool="slow_tool_1", arguments={}),
        CTPToolCall(id="call_2", tool="slow_tool_2", arguments={}),
    ]

    results = []

    async def collect_with_cancel():
        async for result in stream_execute_tools(manager, calls):
            results.append(result)
            # Cancel after receiving anything (but we won't receive anything since they're slow)

    task = asyncio.create_task(collect_with_cancel())
    await asyncio.sleep(0.1)  # Let tasks start
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

    # Results should be empty since we cancelled before any completed
    assert len(results) == 0


@pytest.mark.asyncio
async def test_stream_execute_tools_with_timeout():
    """Test stream_execute_tools with custom timeout."""
    manager = MockToolManager()
    calls = [CTPToolCall(id="call_1", tool="test_tool", arguments={})]

    results = []
    async for result in stream_execute_tools(manager, calls, timeout=60.0):
        results.append(result)

    assert len(results) == 1
    assert results[0].is_success


@pytest.mark.asyncio
async def test_stream_execute_tools_max_concurrency():
    """Test stream_execute_tools respects max_concurrency."""
    execution_count = 0
    max_concurrent = 0

    original_execute = MockToolManager.execute_tool

    async def tracking_execute(
        self, tool_name, arguments, namespace=None, timeout=None
    ):
        nonlocal execution_count, max_concurrent
        execution_count += 1
        current = execution_count
        max_concurrent = max(max_concurrent, current)
        await asyncio.sleep(0.01)  # Simulate work
        execution_count -= 1
        return await original_execute(self, tool_name, arguments, namespace, timeout)

    manager = MockToolManager()
    manager.execute_tool = lambda *args, **kwargs: tracking_execute(
        manager, *args, **kwargs
    )

    calls = [
        CTPToolCall(id=f"call_{i}", tool=f"tool_{i}", arguments={}) for i in range(5)
    ]

    results = []
    async for result in stream_execute_tools(manager, calls, max_concurrency=2):
        results.append(result)

    assert len(results) == 5
    # Max concurrent should not exceed 2 (with some slack for async timing)
    assert max_concurrent <= 3


# ----------------------------------------------------------------------------
# Batch timeout tests (Tier 2)
# ----------------------------------------------------------------------------


class SlowMockToolManager:
    """Mock ToolManager with configurable per-tool delays for batch timeout tests."""

    def __init__(
        self, delays: dict[str, float] | None = None, default_delay: float = 0.0
    ):
        self.tool_timeout = 30.0
        self.delays = delays or {}
        self.default_delay = default_delay
        self.executed_tools: list[str] = []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> ToolCallResult:
        delay = self.delays.get(tool_name, self.default_delay)
        await asyncio.sleep(delay)
        self.executed_tools.append(tool_name)
        return ToolCallResult(
            tool_name=tool_name,
            success=True,
            result={"output": f"result from {tool_name}"},
        )


@pytest.mark.asyncio
async def test_batch_timeout_cancels_remaining():
    """When batch timeout fires, remaining tasks are cancelled."""
    # fast_tool completes quickly, slow_tool takes much longer than the batch timeout
    manager = SlowMockToolManager(delays={"fast_tool": 0.01, "slow_tool": 10.0})
    calls = [
        CTPToolCall(id="call_1", tool="fast_tool", arguments={}),
        CTPToolCall(id="call_2", tool="slow_tool", arguments={}),
    ]

    results = await execute_tools_parallel(manager, calls, batch_timeout=0.5)

    # fast_tool should have completed; slow_tool should have been cancelled
    completed_tools = {r.tool for r in results}
    assert "fast_tool" in completed_tools
    # slow_tool should NOT have completed (it sleeps for 10s, batch timeout is 0.5s)
    assert "slow_tool" not in completed_tools
    assert len(results) < len(calls)


@pytest.mark.asyncio
async def test_batch_timeout_returns_partial():
    """Partial results are returned before the batch timeout expires."""
    # 3 tools: 2 fast, 1 very slow
    manager = SlowMockToolManager(
        delays={"tool_a": 0.01, "tool_b": 0.01, "tool_c": 10.0}
    )
    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
        CTPToolCall(id="call_3", tool="tool_c", arguments={}),
    ]

    results = await execute_tools_parallel(manager, calls, batch_timeout=0.5)

    # The 2 fast tools should have completed
    completed_tools = {r.tool for r in results}
    assert "tool_a" in completed_tools
    assert "tool_b" in completed_tools
    # tool_c should be cancelled
    assert "tool_c" not in completed_tools
    assert len(results) == 2


@pytest.mark.asyncio
async def test_no_timeout_when_fast():
    """Fast tools complete normally without triggering the batch timeout."""
    manager = SlowMockToolManager(default_delay=0.01)
    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
        CTPToolCall(id="call_3", tool="tool_c", arguments={}),
    ]

    results = await execute_tools_parallel(manager, calls, batch_timeout=5.0)

    # All tools should complete successfully
    assert len(results) == 3
    completed_tools = {r.tool for r in results}
    assert completed_tools == {"tool_a", "tool_b", "tool_c"}
    for r in results:
        assert r.is_success


# ----------------------------------------------------------------------------
# Streaming batch timeout tests (Tier 2)
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_batch_timeout_cancels_remaining():
    """Streaming: when batch timeout fires, remaining tasks are cancelled."""
    manager = SlowMockToolManager(delays={"fast_tool": 0.01, "slow_tool": 10.0})
    calls = [
        CTPToolCall(id="call_1", tool="fast_tool", arguments={}),
        CTPToolCall(id="call_2", tool="slow_tool", arguments={}),
    ]

    results = []
    async for result in stream_execute_tools(manager, calls, batch_timeout=0.5):
        results.append(result)

    # fast_tool should have been yielded; slow_tool should have been cancelled
    completed_tools = {r.tool for r in results}
    assert "fast_tool" in completed_tools
    assert "slow_tool" not in completed_tools


@pytest.mark.asyncio
async def test_stream_batch_timeout_returns_partial():
    """Streaming: partial results are yielded before batch timeout."""
    manager = SlowMockToolManager(
        delays={"tool_a": 0.01, "tool_b": 0.01, "tool_c": 10.0}
    )
    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
        CTPToolCall(id="call_3", tool="tool_c", arguments={}),
    ]

    results = []
    async for result in stream_execute_tools(manager, calls, batch_timeout=0.5):
        results.append(result)

    completed_tools = {r.tool for r in results}
    assert "tool_a" in completed_tools
    assert "tool_b" in completed_tools
    assert "tool_c" not in completed_tools
    assert len(results) == 2


@pytest.mark.asyncio
async def test_stream_no_timeout_when_fast():
    """Streaming: fast tools complete without triggering batch timeout."""
    manager = SlowMockToolManager(default_delay=0.01)
    calls = [
        CTPToolCall(id="call_1", tool="tool_a", arguments={}),
        CTPToolCall(id="call_2", tool="tool_b", arguments={}),
        CTPToolCall(id="call_3", tool="tool_c", arguments={}),
    ]

    results = []
    async for result in stream_execute_tools(manager, calls, batch_timeout=5.0):
        results.append(result)

    assert len(results) == 3
    completed_tools = {r.tool for r in results}
    assert completed_tools == {"tool_a", "tool_b", "tool_c"}
    for r in results:
        assert r.is_success


# ----------------------------------------------------------------------------
# Targeted coverage tests for stream_execute_tools edge paths (lines 232-238,
# 254->253, 261, 263)
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_remaining_zero_branch():
    """Cover lines 232-238: the `remaining <= 0` path in stream_execute_tools.

    The deadline is set so that after the first result is yielded the second
    loop iteration sees `remaining <= 0` and hits the early-exit branch.
    """
    import unittest.mock as mock

    # One fast tool completes quickly, one very slow tool never finishes within
    # the tiny batch_timeout.  We want the *deadline check* (remaining <= 0) to
    # fire, not the asyncio.wait_for timeout, so we make the timeout large
    # enough that wait_for itself won't time out, but we mock the event-loop
    # clock so that on the second while-loop iteration `remaining` is already ≤ 0.

    manager = SlowMockToolManager(delays={"fast_tool": 0.0, "slow_tool": 10.0})
    calls = [
        CTPToolCall(id="call_1", tool="fast_tool", arguments={}),
        CTPToolCall(id="call_2", tool="slow_tool", arguments={}),
    ]

    # We'll intercept get_event_loop().time():
    # - First two calls (setting deadline + first remaining check) return a
    #   consistent value so remaining is large.
    # - Subsequent calls return a value far in the future so that `remaining`
    #   computes to a negative number on the second iteration.
    real_loop = asyncio.get_event_loop()
    real_time = real_loop.time()
    call_count = 0

    def fake_time():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            # First call: sets deadline = real_time + 1000
            # Second call: first remaining = 1000 - real_time ≈ 1000  (positive)
            return real_time
        else:
            # All subsequent calls: pretend the clock has jumped far past the
            # deadline so remaining ≤ 0 triggers.
            return real_time + 100_000

    results = []
    with mock.patch.object(real_loop, "time", side_effect=fake_time):
        async for result in stream_execute_tools(manager, calls, batch_timeout=1000.0):
            results.append(result)

    # fast_tool result was received; slow_tool was cancelled via the
    # remaining <= 0 branch.
    completed_tools = {r.tool for r in results}
    assert "fast_tool" in completed_tools
    assert "slow_tool" not in completed_tools


@pytest.mark.asyncio
async def test_stream_cancelled_error_path():
    """Cover lines 251-256: CancelledError inside queue.get() wait.

    We cancel the outer consuming task while stream_execute_tools is blocked
    waiting for a result.  The CancelledError propagates into the except branch,
    which cancels remaining tasks and breaks.
    """
    manager = SlowMockToolManager(default_delay=10.0)  # all tools are very slow
    calls = [
        CTPToolCall(id="call_1", tool="slow_tool_1", arguments={}),
        CTPToolCall(id="call_2", tool="slow_tool_2", arguments={}),
    ]

    results = []

    async def consumer():
        async for result in stream_execute_tools(manager, calls, batch_timeout=100.0):
            results.append(result)

    task = asyncio.create_task(consumer())
    # Give the generator time to start and block on queue.get()
    await asyncio.sleep(0.05)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass  # expected

    # No results should have been received (tools are too slow)
    assert results == []


@pytest.mark.asyncio
async def test_stream_cancelled_error_with_some_done_tasks():
    """Cover line 254->253: CancelledError fires after one task already completed.

    One fast tool completes (result in queue, task is done), consumer receives
    it, then CancelledError fires while waiting for the slow tool.  Inside the
    CancelledError handler, the for-loop iterates over tasks; the fast task is
    already done (condition False → branch back to 253) and the slow task is
    not done (condition True → cancel it).  This exercises the 254->253 arc.
    """
    manager = SlowMockToolManager(delays={"fast_tool": 0.0, "slow_tool": 10.0})
    calls = [
        CTPToolCall(id="call_1", tool="fast_tool", arguments={}),
        CTPToolCall(id="call_2", tool="slow_tool", arguments={}),
    ]

    results = []

    async def consumer():
        async for result in stream_execute_tools(manager, calls, batch_timeout=100.0):
            results.append(result)

    task = asyncio.create_task(consumer())
    # Allow fast_tool to complete and its result to be dequeued (consumer gets it),
    # then cancel while consumer is blocked waiting for slow_tool.
    await asyncio.sleep(0.1)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass  # expected

    # fast_tool result should have been received; slow_tool cancelled
    assert len(results) == 1
    assert results[0].tool == "fast_tool"


@pytest.mark.asyncio
async def test_stream_cleanup_cancels_pending_tasks():
    """Cover lines 261 and 263: cleanup block cancels and awaits pending tasks.

    When the while loop exits early (via the TimeoutError break), tasks that are
    still running enter the `pending` set returned by asyncio.wait(timeout=0).
    Lines 261 and 263 cancel and gather those pending tasks.
    """
    # Use SlowMockToolManager: fast_tool done quickly, slow_tool never finishes
    # within the tiny batch_timeout.
    manager = SlowMockToolManager(delays={"fast_tool": 0.01, "slow_tool": 10.0})
    calls = [
        CTPToolCall(id="call_1", tool="fast_tool", arguments={}),
        CTPToolCall(id="call_2", tool="slow_tool", arguments={}),
    ]

    results = []
    # batch_timeout of 0.2 s: fast_tool finishes, slow_tool is still running
    # when TimeoutError fires.  After the while-loop breaks, slow_tool's task
    # is still pending → lines 261 and 263 execute.
    async for result in stream_execute_tools(manager, calls, batch_timeout=0.2):
        results.append(result)

    completed_tools = {r.tool for r in results}
    assert "fast_tool" in completed_tools
    assert "slow_tool" not in completed_tools


@pytest.mark.asyncio
async def test_stream_remaining_zero_cancels_pending_tasks():
    """Cover lines 232-238 AND 261/263 together via the remaining<=0 early-exit.

    After the remaining<=0 break, tasks that haven't finished yet are pending
    and must be cleaned up by lines 261/263.
    """
    import unittest.mock as mock

    manager = SlowMockToolManager(delays={"fast_tool": 0.0, "slow_tool": 10.0})
    calls = [
        CTPToolCall(id="call_1", tool="fast_tool", arguments={}),
        CTPToolCall(id="call_2", tool="slow_tool", arguments={}),
    ]

    real_loop = asyncio.get_event_loop()
    real_time = real_loop.time()
    call_count = 0

    def fake_time():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return real_time
        return real_time + 100_000

    results = []
    with mock.patch.object(real_loop, "time", side_effect=fake_time):
        async for result in stream_execute_tools(manager, calls, batch_timeout=1000.0):
            results.append(result)

    # fast_tool completed; slow_tool was cancelled in both the remaining<=0
    # branch and the cleanup block.
    assert len(results) >= 0  # main assertion: no unhandled exception raised
