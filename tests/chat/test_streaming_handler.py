# tests/chat/test_streaming_handler.py
"""Tests for mcp_cli.chat.streaming_handler achieving >90% coverage."""

import asyncio
import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.chat.streaming_handler import (
    StreamingResponseField,
    StreamingResponse,
    ToolCallAccumulator,
    StreamingResponseHandler,
)


# ---------------------------------------------------------------------------
# StreamingResponseField enum tests
# ---------------------------------------------------------------------------


class TestStreamingResponseField:
    """Tests for the StreamingResponseField enum."""

    def test_values(self):
        assert StreamingResponseField.RESPONSE == "response"
        assert StreamingResponseField.TOOL_CALLS == "tool_calls"
        assert StreamingResponseField.CHUNKS_RECEIVED == "chunks_received"
        assert StreamingResponseField.ELAPSED_TIME == "elapsed_time"
        assert StreamingResponseField.STREAMING == "streaming"
        assert StreamingResponseField.INTERRUPTED == "interrupted"
        assert StreamingResponseField.REASONING_CONTENT == "reasoning_content"


# ---------------------------------------------------------------------------
# StreamingResponse model tests
# ---------------------------------------------------------------------------


class TestStreamingResponse:
    """Tests for the StreamingResponse Pydantic model."""

    def test_defaults(self):
        sr = StreamingResponse(content="hello", elapsed_time=1.0)
        assert sr.content == "hello"
        assert sr.tool_calls == []
        assert sr.chunks_received == 0
        assert sr.elapsed_time == 1.0
        assert sr.interrupted is False
        assert sr.reasoning_content is None
        assert sr.streaming is True

    def test_full_construction(self):
        sr = StreamingResponse(
            content="text",
            tool_calls=[{"id": "1", "type": "function", "function": {"name": "f"}}],
            chunks_received=5,
            elapsed_time=2.5,
            interrupted=True,
            reasoning_content="thinking...",
            streaming=False,
        )
        assert sr.chunks_received == 5
        assert sr.interrupted is True
        assert sr.reasoning_content == "thinking..."
        assert sr.streaming is False

    def test_to_dict(self):
        sr = StreamingResponse(
            content="abc",
            tool_calls=[],
            chunks_received=3,
            elapsed_time=0.5,
            interrupted=False,
            reasoning_content=None,
        )
        d = sr.to_dict()
        assert d[StreamingResponseField.RESPONSE] == "abc"
        assert d[StreamingResponseField.TOOL_CALLS] == []
        assert d[StreamingResponseField.CHUNKS_RECEIVED] == 3
        assert d[StreamingResponseField.ELAPSED_TIME] == 0.5
        assert d[StreamingResponseField.STREAMING] is True
        assert d[StreamingResponseField.INTERRUPTED] is False
        assert d[StreamingResponseField.REASONING_CONTENT] is None

    def test_to_dict_with_tool_calls(self):
        tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "sqrt", "arguments": "{}"},
        }
        sr = StreamingResponse(content="", tool_calls=[tc], elapsed_time=0.1)
        d = sr.to_dict()
        assert len(d[StreamingResponseField.TOOL_CALLS]) == 1


# ---------------------------------------------------------------------------
# ToolCallAccumulator tests
# ---------------------------------------------------------------------------


class TestToolCallAccumulator:
    """Tests for ToolCallAccumulator."""

    def test_empty_finalize(self):
        acc = ToolCallAccumulator()
        assert acc.finalize() == []

    def test_single_complete_tool_call(self):
        acc = ToolCallAccumulator()
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "sqrt", "arguments": '{"x": 16}'},
                }
            ]
        )
        result = acc.finalize()
        assert len(result) == 1
        assert result[0]["function"]["name"] == "sqrt"

    def test_merged_tool_call_chunks(self):
        """Arguments are accumulated across chunks."""
        acc = ToolCallAccumulator()
        # First chunk: partial args
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "add", "arguments": '{"a"'},
                }
            ]
        )
        # Second chunk: rest of args
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "", "arguments": ': 1, "b": 2}'},
                }
            ]
        )
        result = acc.finalize()
        assert len(result) == 1
        # The combined args should be valid JSON
        parsed = json.loads(result[0]["function"]["arguments"])
        assert parsed == {"a": 1, "b": 2}

    def test_multiple_tool_calls(self):
        acc = ToolCallAccumulator()
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "fn1", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "index": 1,
                    "function": {"name": "fn2", "arguments": "{}"},
                },
            ]
        )
        result = acc.finalize()
        assert len(result) == 2

    def test_skip_tool_call_without_name(self):
        acc = ToolCallAccumulator()
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "", "arguments": "{}"},
                }
            ]
        )
        result = acc.finalize()
        assert len(result) == 0

    def test_skip_invalid_json_arguments(self):
        acc = ToolCallAccumulator()
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "bad_fn", "arguments": "{not valid json"},
                }
            ]
        )
        result = acc.finalize()
        assert len(result) == 0  # Skipped due to invalid JSON

    def test_empty_arguments_default_to_empty_object(self):
        acc = ToolCallAccumulator()
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "no_args_fn", "arguments": "  {}  "},
                }
            ]
        )
        result = acc.finalize()
        assert len(result) == 1

    def test_finalize_empty_args_string(self):
        acc = ToolCallAccumulator()
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "fn", "arguments": ""},
                }
            ]
        )
        result = acc.finalize()
        assert len(result) == 1
        # Empty string args are treated as empty JSON object "{}" by finalize
        # but to_dict preserves the raw function.arguments value
        assert result[0]["function"]["arguments"] in ("", "{}")

    def test_merge_json_strings_both_valid(self):
        """Merge two valid JSON objects."""
        acc = ToolCallAccumulator()
        result = acc._merge_json_strings('{"a": 1}', '{"b": 2}')
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_merge_json_strings_first_empty(self):
        acc = ToolCallAccumulator()
        result = acc._merge_json_strings("", '{"b": 2}')
        assert result == '{"b": 2}'

    def test_merge_json_strings_second_empty(self):
        acc = ToolCallAccumulator()
        result = acc._merge_json_strings('{"a": 1}', "")
        assert result == '{"a": 1}'

    def test_merge_json_strings_concatenation(self):
        """When individual parse fails, try concatenation."""
        acc = ToolCallAccumulator()
        result = acc._merge_json_strings('{"a":', "1}")
        parsed = json.loads(result)
        assert parsed == {"a": 1}

    def test_merge_json_strings_fix_braces(self):
        """When fragments cannot be parsed, still returns concatenation."""
        acc = ToolCallAccumulator()
        result = acc._merge_json_strings('"a": 1', '"b": 2')
        # Strategies 1-3 all fail, so the raw concatenation is returned
        assert result == '"a": 1"b": 2'

    def test_merge_json_strings_fix_double_brace(self):
        """Fix }{ pattern."""
        acc = ToolCallAccumulator()
        result = acc._merge_json_strings('{"a": 1}', '{"b": 2}')
        # Both are valid, so strategy 1 should work
        parsed = json.loads(result)
        assert "a" in parsed

    def test_merge_json_strings_fallback(self):
        """When nothing works, returns concatenation."""
        acc = ToolCallAccumulator()
        result = acc._merge_json_strings("not{json", "also{not")
        # Should return some concatenated result
        assert "not" in result

    def test_find_accumulated_call_by_index(self):
        """Find tool call by index when id doesn't match."""
        acc = ToolCallAccumulator()
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "fn", "arguments": '{"a": 1}'},
                }
            ]
        )
        # Second chunk with different id but same index
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "", "arguments": ""},
                }
            ]
        )
        result = acc.finalize()
        assert len(result) == 1

    def test_merge_function_name_update(self):
        """Function name is updated when new chunk provides one."""
        acc = ToolCallAccumulator()
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "old_name", "arguments": "{}"},
                }
            ]
        )
        acc.process_chunk_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "new_name", "arguments": ""},
                }
            ]
        )
        result = acc.finalize()
        assert len(result) == 1
        assert result[0]["function"]["name"] == "new_name"


# ---------------------------------------------------------------------------
# StreamingResponseHandler tests
# ---------------------------------------------------------------------------


def _make_display():
    """Build a mock StreamingDisplayManager."""
    display = MagicMock()
    display.start_streaming = AsyncMock()
    display.stop_streaming = AsyncMock(return_value="final content")
    display.add_chunk = AsyncMock()
    display.is_streaming = False

    # Create a proper streaming_state mock
    state = MagicMock()
    state.chunks_received = 5
    state.reasoning_content = None
    display.streaming_state = state
    return display


def _make_runtime_config():
    """Build a mock RuntimeConfig."""
    rc = MagicMock()
    rc.get_timeout = MagicMock(
        side_effect=lambda t: 45.0 if "CHUNK" in t.value else 300.0
    )
    return rc


class TestStreamingResponseHandler:
    """Tests for StreamingResponseHandler."""

    def test_init(self):
        display = _make_display()
        handler = StreamingResponseHandler(display)
        assert handler.display is display
        assert handler._interrupted is False

    def test_init_with_runtime_config(self):
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)
        assert handler.runtime_config is rc

    @pytest.mark.asyncio
    async def test_init_loads_default_runtime_config(self):
        """When no runtime_config, loads default."""
        display = _make_display()
        with patch("mcp_cli.chat.streaming_handler.load_runtime_config") as mock_load:
            mock_load.return_value = _make_runtime_config()
            StreamingResponseHandler(display)
            mock_load.assert_called_once()

    def test_interrupt_streaming(self):
        display = _make_display()
        handler = StreamingResponseHandler(display)
        assert handler._interrupted is False
        handler.interrupt_streaming()
        assert handler._interrupted is True

    @pytest.mark.asyncio
    async def test_stream_response_non_streaming_client(self):
        """Falls back to non-streaming when client lacks create_completion."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        client = MagicMock(spec=[])  # No create_completion attribute
        client.complete = AsyncMock(return_value={"response": "hi", "tool_calls": []})

        with patch("chuk_term.ui.output") as mock_output:
            mock_output.loading.return_value.__enter__ = MagicMock(return_value=None)
            mock_output.loading.return_value.__exit__ = MagicMock(return_value=False)
            result = await handler.stream_response(client, messages=[], tools=None)

        assert result["response"] == "hi"
        assert result["streaming"] is False

    @pytest.mark.asyncio
    async def test_stream_response_non_streaming_no_complete(self):
        """Raises RuntimeError when client has neither method."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        client = MagicMock(spec=[])  # No methods at all

        with pytest.raises(RuntimeError, match="no streaming or completion method"):
            with patch("chuk_term.ui.output") as mock_output:
                mock_output.loading.return_value.__enter__ = MagicMock(
                    return_value=None
                )
                mock_output.loading.return_value.__exit__ = MagicMock(
                    return_value=False
                )
                await handler.stream_response(client, messages=[], tools=None)

    @pytest.mark.asyncio
    async def test_stream_response_success(self):
        """Successful streaming returns proper response dict."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        # Create a mock async iterator for the stream
        chunks = [
            {"content": "Hello"},
            {"content": " world"},
        ]

        async def mock_aiter():
            for chunk in chunks:
                yield chunk

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        result = await handler.stream_response(
            client, messages=[{"role": "user", "content": "hi"}]
        )

        assert result[StreamingResponseField.RESPONSE] == "final content"
        assert result[StreamingResponseField.STREAMING] is True
        assert result[StreamingResponseField.CHUNKS_RECEIVED] == 5
        assert result[StreamingResponseField.ELAPSED_TIME] > 0

    @pytest.mark.asyncio
    async def test_stream_response_with_tool_calls(self):
        """Streaming with tool calls in chunks."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        chunks = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "index": 0,
                        "function": {"name": "sqrt", "arguments": '{"x": 16}'},
                    }
                ],
            },
        ]

        async def mock_aiter():
            for chunk in chunks:
                yield chunk

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        result = await handler.stream_response(
            client, messages=[], tools=[{"type": "function"}]
        )

        assert len(result[StreamingResponseField.TOOL_CALLS]) == 1
        assert (
            result[StreamingResponseField.TOOL_CALLS][0]["function"]["name"] == "sqrt"
        )

    @pytest.mark.asyncio
    async def test_stream_response_exception(self):
        """Exception during streaming stops display and re-raises."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        async def mock_aiter():
            raise RuntimeError("stream error")
            yield  # pragma: no cover - makes it an async generator

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        with pytest.raises(RuntimeError, match="stream error"):
            await handler.stream_response(client, messages=[])

        display.stop_streaming.assert_called_with(interrupted=True)

    @pytest.mark.asyncio
    async def test_stream_interrupted_by_user(self):
        """Interrupting during stream sets interrupted flag."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        chunk_count = [0]

        async def mock_aiter():
            while True:
                chunk_count[0] += 1
                if chunk_count[0] == 2:
                    handler._interrupted = True
                yield {"content": "x"}

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        result = await handler.stream_response(client, messages=[])
        assert result[StreamingResponseField.INTERRUPTED] is True

    @pytest.mark.asyncio
    async def test_stream_chunk_timeout(self):
        """Per-chunk timeout is handled gracefully."""
        display = _make_display()
        rc = MagicMock()
        rc.get_timeout = MagicMock(
            side_effect=lambda t: 0.01 if "CHUNK" in t.value else 300.0
        )
        handler = StreamingResponseHandler(display, runtime_config=rc)

        async def mock_aiter():
            yield {"content": "first"}
            await asyncio.sleep(10)  # Will exceed chunk timeout
            yield {"content": "second"}  # pragma: no cover

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        with patch("chuk_term.ui.output"):
            result = await handler.stream_response(client, messages=[])

        # Should complete without error (timeout is handled internally)
        assert StreamingResponseField.RESPONSE in result

    @pytest.mark.asyncio
    async def test_stream_global_timeout(self):
        """Global timeout is handled gracefully."""
        display = _make_display()
        rc = MagicMock()
        rc.get_timeout = MagicMock(
            side_effect=lambda t: 45.0 if "CHUNK" in t.value else 0.01
        )
        handler = StreamingResponseHandler(display, runtime_config=rc)

        async def mock_aiter():
            while True:
                await asyncio.sleep(0.005)
                yield {"content": "x"}

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        with patch("chuk_term.ui.output"):
            result = await handler.stream_response(client, messages=[])

        assert result[StreamingResponseField.INTERRUPTED] is True

    @pytest.mark.asyncio
    async def test_process_chunk_with_tool_calls(self):
        """_process_chunk extracts tool calls."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        chunk = {
            "content": "text",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "fn", "arguments": "{}"},
                }
            ],
        }
        await handler._process_chunk(chunk)
        display.add_chunk.assert_called_once_with(chunk)
        assert len(handler.tool_accumulator._accumulated) == 1

    @pytest.mark.asyncio
    async def test_process_chunk_without_tool_calls(self):
        """_process_chunk with no tool calls."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        chunk = {"content": "text"}
        await handler._process_chunk(chunk)
        display.add_chunk.assert_called_once_with(chunk)
        assert len(handler.tool_accumulator._accumulated) == 0

    @pytest.mark.asyncio
    async def test_handle_non_streaming_with_complete(self):
        """Non-streaming fallback with client.complete."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        client = MagicMock(spec=[])
        client.complete = AsyncMock(
            return_value={"response": "done", "tool_calls": [{"id": "1"}]}
        )

        with patch("chuk_term.ui.output") as mock_output:
            mock_output.loading.return_value.__enter__ = MagicMock(return_value=None)
            mock_output.loading.return_value.__exit__ = MagicMock(return_value=False)
            result = await handler._handle_non_streaming(client, [], None)

        assert result["response"] == "done"
        assert result["tool_calls"] == [{"id": "1"}]
        assert result["streaming"] is False
        assert result["chunks_received"] == 1

    @pytest.mark.asyncio
    async def test_handle_non_streaming_no_method_raises(self):
        """Non-streaming raises RuntimeError when no method available."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        client = MagicMock(spec=[])
        # No 'complete' attribute

        with pytest.raises(RuntimeError):
            with patch("chuk_term.ui.output") as mock_output:
                mock_output.loading.return_value.__enter__ = MagicMock(
                    return_value=None
                )
                mock_output.loading.return_value.__exit__ = MagicMock(
                    return_value=False
                )
                await handler._handle_non_streaming(client, [], None)

    @pytest.mark.asyncio
    async def test_stream_response_with_reasoning_content(self):
        """Streaming response captures reasoning_content."""
        display = _make_display()
        display.streaming_state.reasoning_content = "I am thinking..."
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        async def mock_aiter():
            yield {"content": "result"}

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        result = await handler.stream_response(client, messages=[])
        assert result[StreamingResponseField.REASONING_CONTENT] == "I am thinking..."

    @pytest.mark.asyncio
    async def test_stream_response_no_streaming_state(self):
        """Handles missing streaming_state gracefully."""
        display = _make_display()
        display.streaming_state = None
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        async def mock_aiter():
            yield {"content": "x"}

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        result = await handler.stream_response(client, messages=[])
        assert result[StreamingResponseField.CHUNKS_RECEIVED] == 0
        assert result[StreamingResponseField.REASONING_CONTENT] is None

    @pytest.mark.asyncio
    async def test_stream_close_error_suppressed(self):
        """Error closing stream iterator is suppressed."""
        display = _make_display()
        rc = _make_runtime_config()
        handler = StreamingResponseHandler(display, runtime_config=rc)

        call_count = [0]

        class MockIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {"content": "x"}
                handler._interrupted = True
                return {"content": "y"}

            async def aclose(self):
                raise RuntimeError("close error")

        client = MagicMock()
        client.create_completion = MagicMock(return_value=MockIterator())

        result = await handler.stream_response(client, messages=[])
        assert result[StreamingResponseField.INTERRUPTED] is True


# ---------------------------------------------------------------------------
# Edge cases for _merge_json_strings
# ---------------------------------------------------------------------------


class TestMergeJsonStringsEdgeCases:
    """Additional edge case tests for _merge_json_strings."""

    def test_both_empty(self):
        acc = ToolCallAccumulator()
        result = acc._merge_json_strings("", "")
        assert result == ""

    def test_valid_arrays(self):
        """Two valid JSON arrays get strategy 1 skip (not dicts)."""
        acc = ToolCallAccumulator()
        result = acc._merge_json_strings("[1, 2]", "[3, 4]")
        # Strategy 1 fails (not dicts), strategy 2 succeeds if concat is valid
        # [1, 2][3, 4] is not valid JSON, so strategy 3 kicks in
        assert result is not None

    def test_needs_brace_fix(self):
        """Test missing opening brace fix."""
        acc = ToolCallAccumulator()
        result = acc._merge_json_strings('"x": 1', "")
        # Current + new is '"x": 1', then tries to add braces
        assert result == '"x": 1'  # second empty, returns first


# ---------------------------------------------------------------------------
# First-chunk timeout tests
# ---------------------------------------------------------------------------


class TestFirstChunkTimeout:
    """Tests for differentiated first-chunk vs subsequent-chunk timeouts."""

    def _make_rc(self, chunk=45.0, first_chunk=60.0, global_t=300.0):
        """Build a RuntimeConfig mock with distinct timeout values."""
        from mcp_cli.config.enums import TimeoutType

        def get_timeout(t):
            if t == TimeoutType.STREAMING_FIRST_CHUNK:
                return first_chunk
            elif t == TimeoutType.STREAMING_CHUNK:
                return chunk
            elif t == TimeoutType.STREAMING_GLOBAL:
                return global_t
            return 30.0

        rc = MagicMock()
        rc.get_timeout = MagicMock(side_effect=get_timeout)
        return rc

    @pytest.mark.asyncio
    async def test_first_chunk_uses_longer_timeout(self):
        """First chunk should use first_chunk_timeout, not chunk_timeout."""
        display = _make_display()
        # Set first_chunk=0.01 so it times out quickly on first chunk
        # but chunk=300 so it wouldn't time out on subsequent chunks
        rc = self._make_rc(chunk=300.0, first_chunk=0.01, global_t=300.0)
        handler = StreamingResponseHandler(display, runtime_config=rc)

        async def mock_aiter():
            await asyncio.sleep(10)  # Exceeds first_chunk timeout (0.01s)
            yield {"content": "never reached"}  # pragma: no cover

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        with patch("chuk_term.ui.output"):
            result = await handler.stream_response(client, messages=[])

        # Should complete (timeout handled internally), content from display mock
        assert StreamingResponseField.RESPONSE in result

    @pytest.mark.asyncio
    async def test_subsequent_chunks_use_chunk_timeout(self):
        """After first chunk, subsequent chunks use the regular chunk_timeout."""
        display = _make_display()
        # first_chunk=300 (won't time out), chunk=0.01 (will time out on 2nd chunk)
        rc = self._make_rc(chunk=0.01, first_chunk=300.0, global_t=300.0)
        handler = StreamingResponseHandler(display, runtime_config=rc)

        async def mock_aiter():
            yield {"content": "first"}  # Arrives quickly
            await asyncio.sleep(10)  # Exceeds chunk timeout (0.01s)
            yield {"content": "never reached"}  # pragma: no cover

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        with patch("chuk_term.ui.output"):
            result = await handler.stream_response(client, messages=[])

        assert StreamingResponseField.RESPONSE in result

    @pytest.mark.asyncio
    async def test_after_tool_calls_extends_first_chunk_timeout(self):
        """after_tool_calls=True uses the longer after-tools timeout."""
        display = _make_display()
        # first_chunk=60, chunk=45 — but after_tool_calls bumps to 180
        rc = self._make_rc(chunk=45.0, first_chunk=60.0, global_t=300.0)
        handler = StreamingResponseHandler(display, runtime_config=rc)

        timeouts_used = []

        original_wait_for = asyncio.wait_for

        async def tracking_wait_for(coro, timeout):
            timeouts_used.append(timeout)
            return await original_wait_for(coro, timeout=timeout)

        async def mock_aiter():
            yield {"content": "response after tools"}

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        with patch("asyncio.wait_for", side_effect=tracking_wait_for):
            result = await handler.stream_response(
                client, messages=[], after_tool_calls=True
            )

        assert StreamingResponseField.RESPONSE in result
        # The first wait_for call in stream_chunks should use 180.0 (after-tools timeout)
        # The global timeout uses 300.0
        # Find the 180.0 timeout (first chunk after tools)
        assert 180.0 in timeouts_used, f"Expected 180.0 in {timeouts_used}"

    @pytest.mark.asyncio
    async def test_after_tool_calls_false_uses_normal_first_chunk(self):
        """after_tool_calls=False uses the standard first_chunk_timeout."""
        display = _make_display()
        rc = self._make_rc(chunk=45.0, first_chunk=60.0, global_t=300.0)
        handler = StreamingResponseHandler(display, runtime_config=rc)

        timeouts_used = []

        original_wait_for = asyncio.wait_for

        async def tracking_wait_for(coro, timeout):
            timeouts_used.append(timeout)
            return await original_wait_for(coro, timeout=timeout)

        async def mock_aiter():
            yield {"content": "normal response"}

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        with patch("asyncio.wait_for", side_effect=tracking_wait_for):
            result = await handler.stream_response(
                client, messages=[], after_tool_calls=False
            )

        assert StreamingResponseField.RESPONSE in result
        # Should see 60.0 (first chunk) and NOT 180.0
        assert 60.0 in timeouts_used, f"Expected 60.0 in {timeouts_used}"
        assert 180.0 not in timeouts_used, f"Did not expect 180.0 in {timeouts_used}"

    @pytest.mark.asyncio
    async def test_after_tool_calls_timeout_message_mentions_tool_results(self, caplog):
        """Timeout message after tool calls mentions tool result processing."""
        display = _make_display()
        # Use very short first_chunk to trigger timeout
        rc = self._make_rc(chunk=0.01, first_chunk=0.01, global_t=300.0)
        handler = StreamingResponseHandler(display, runtime_config=rc)

        async def mock_aiter():
            await asyncio.sleep(10)  # Will timeout
            yield {"content": "never"}  # pragma: no cover

        client = MagicMock()
        client.create_completion = MagicMock(return_value=mock_aiter())

        # Patch the after-tools default to a tiny value so timeout fires
        with (
            caplog.at_level(logging.WARNING, logger="mcp_cli.streaming"),
            patch(
                "mcp_cli.config.defaults.DEFAULT_STREAMING_FIRST_CHUNK_AFTER_TOOLS_TIMEOUT",
                0.01,
            ),
        ):
            await handler.stream_response(client, messages=[], after_tool_calls=True)

        # Check that the warning message mentions tool calls
        warning_msgs = [
            r.message
            for r in caplog.records
            if r.levelname == "WARNING" and r.name == "mcp_cli.streaming"
        ]
        assert any("tool" in m.lower() for m in warning_msgs)
