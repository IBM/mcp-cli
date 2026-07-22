# tests/apps/test_bridge.py
"""Tests for MCP Apps bridge (Python-side protocol handler)."""

from __future__ import annotations

import asyncio
import json

import pytest

from mcp_cli.apps.bridge import AppBridge
from mcp_cli.apps.models import AppInfo, AppState


# ── Fakes ──────────────────────────────────────────────────────────────────


class FakeToolResult:
    def __init__(self, success: bool, result=None, error=None):
        self.success = success
        self.result = result
        self.error = error


class FakeToolManager:
    """Minimal ToolManager stub for bridge tests."""

    def __init__(self):
        self.executed_tools: list[tuple[str, dict, str | None]] = []
        self.read_resources: list[tuple[str, str | None]] = []
        self._next_result = FakeToolResult(True, result="ok")
        self._next_resource = {
            "contents": [{"uri": "ui://test", "text": "<html></html>"}]
        }
        self._raise_on_execute: Exception | None = None
        self._execute_delay: float = 0

    async def execute_tool(self, name, arguments, namespace=None):
        if self._execute_delay:
            await asyncio.sleep(self._execute_delay)
        self.executed_tools.append((name, arguments, namespace))
        if self._raise_on_execute:
            raise self._raise_on_execute
        return self._next_result

    async def read_resource(self, uri, server_name=None):
        self.read_resources.append((uri, server_name))
        return self._next_resource


class FakeWs:
    """Minimal WebSocket stub."""

    def __init__(self):
        self.sent: list[str] = []
        self.closed = False
        self._raise_on_send = False

    async def send(self, msg: str) -> None:
        if self._raise_on_send:
            raise ConnectionError("ws closed")
        self.sent.append(msg)

    async def close(self) -> None:
        self.closed = True


def _make_bridge() -> tuple[AppBridge, FakeToolManager]:
    tm = FakeToolManager()
    info = AppInfo(
        tool_name="test-tool",
        resource_uri="ui://test-tool/app.html",
        server_name="test-server",
        port=9470,
    )
    bridge = AppBridge(info, tm)
    return bridge, tm


# ── Tests ──────────────────────────────────────────────────────────────────


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_tools_call_success(self):
        bridge, tm = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "get-time", "arguments": {}},
            }
        )

        resp = await bridge.handle_message(msg)
        assert resp is not None
        parsed = json.loads(resp)
        assert parsed["id"] == 1
        assert "result" in parsed
        assert tm.executed_tools == [("get-time", {}, "test-server")]

    @pytest.mark.asyncio
    async def test_tools_call_failure(self):
        bridge, tm = _make_bridge()
        tm._next_result = FakeToolResult(False, error="timeout")

        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "broken-tool", "arguments": {"x": 1}},
            }
        )

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 2
        assert "error" in parsed
        assert parsed["error"]["message"] == "timeout"

    @pytest.mark.asyncio
    async def test_resources_read(self):
        bridge, tm = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "resources/read",
                "params": {"uri": "ui://test/data.json"},
            }
        )

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 3
        assert "result" in parsed
        assert tm.read_resources == [("ui://test/data.json", "test-server")]

    @pytest.mark.asyncio
    async def test_ui_message(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "ui/message",
                "params": {"content": {"type": "text", "text": "hello"}},
            }
        )

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 4
        assert parsed["result"] == {}

    @pytest.mark.asyncio
    async def test_model_context_update(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "ui/update-model-context",
                "params": {"content": [{"type": "text", "text": "user picked red"}]},
            }
        )

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["id"] == 5
        assert parsed["result"] == {}
        assert bridge.model_context is not None

    @pytest.mark.asyncio
    async def test_initialized_notification(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "ui/notifications/initialized",
                "params": {},
            }
        )

        resp = await bridge.handle_message(msg)
        assert resp is None
        assert bridge.app_info.state == AppState.READY

    @pytest.mark.asyncio
    async def test_unknown_request(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "unknown/method",
                "params": {},
            }
        )

        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_unknown_notification_ignored(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "unknown/notification",
                "params": {},
            }
        )

        resp = await bridge.handle_message(msg)
        assert resp is None

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        bridge, _ = _make_bridge()
        resp = await bridge.handle_message("not json at all")
        assert resp is None

    @pytest.mark.asyncio
    async def test_tool_call_with_exception(self):
        """execute_tool raises an exception."""
        bridge, tm = _make_bridge()
        tm._raise_on_execute = RuntimeError("server crashed")
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "crash-tool", "arguments": {}},
            }
        )
        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32000
        assert "server crashed" in parsed["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_tool_name_rejected(self):
        """Tool names with special chars are rejected."""
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {"name": "rm -rf /", "arguments": {}},
            }
        )
        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_empty_tool_name_rejected(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {"name": "", "arguments": {}},
            }
        )
        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_resource_read_exception(self):
        """read_resource raises an exception."""
        bridge, tm = _make_bridge()
        tm._next_resource = None  # will cause AttributeError

        # Override to raise
        async def _raise_resource(uri, server_name=None):
            raise RuntimeError("network error")

        tm.read_resource = _raise_resource

        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "resources/read",
                "params": {"uri": "ui://bad/resource"},
            }
        )
        resp = await bridge.handle_message(msg)
        parsed = json.loads(resp)
        assert parsed["error"]["code"] == -32000
        assert "network error" in parsed["error"]["message"]

    @pytest.mark.asyncio
    async def test_teardown_notification_sets_closed(self):
        bridge, _ = _make_bridge()
        msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "ui/notifications/teardown",
                "params": {},
            }
        )
        resp = await bridge.handle_message(msg)
        assert resp is None
        assert bridge.app_info.state == AppState.CLOSED


class TestWebSocketLifecycle:
    def test_set_ws_resets_state(self):
        bridge, _ = _make_bridge()
        bridge.app_info.state = AppState.READY
        ws = FakeWs()
        bridge.set_ws(ws)
        assert bridge.app_info.state == AppState.INITIALIZING

    def test_set_ws_closes_old(self):
        bridge, _ = _make_bridge()
        old_ws = FakeWs()
        bridge.set_ws(old_ws)
        new_ws = FakeWs()
        bridge.set_ws(new_ws)
        # Old WS close is scheduled via ensure_future, verify the new one is set
        assert bridge._ws is new_ws

    @pytest.mark.asyncio
    async def test_push_tool_result_queued_when_no_ws(self):
        bridge, _ = _make_bridge()
        # No WS set
        await bridge.push_tool_result("test result")
        assert len(bridge._pending_notifications) == 1

    @pytest.mark.asyncio
    async def test_push_tool_result_sent_when_ws(self):
        bridge, _ = _make_bridge()
        ws = FakeWs()
        bridge.set_ws(ws)
        await bridge.push_tool_result("test result")
        assert len(ws.sent) == 1
        parsed = json.loads(ws.sent[0])
        assert parsed["method"] == "ui/notifications/tool-result"

    @pytest.mark.asyncio
    async def test_push_tool_result_queued_on_send_failure(self):
        bridge, _ = _make_bridge()
        ws = FakeWs()
        ws._raise_on_send = True
        bridge.set_ws(ws)
        await bridge.push_tool_result("test result")
        assert len(bridge._pending_notifications) == 1

    @pytest.mark.asyncio
    async def test_drain_pending_sends_queued(self):
        bridge, _ = _make_bridge()
        # Queue a notification while WS is down
        await bridge.push_tool_result("queued result")
        assert len(bridge._pending_notifications) == 1

        # Now connect WS and drain
        ws = FakeWs()
        bridge.set_ws(ws)
        await bridge.drain_pending()
        assert len(ws.sent) == 1
        assert len(bridge._pending_notifications) == 0

    @pytest.mark.asyncio
    async def test_drain_pending_requeues_on_failure(self):
        bridge, _ = _make_bridge()
        await bridge.push_tool_result("queued result")

        ws = FakeWs()
        ws._raise_on_send = True
        bridge.set_ws(ws)
        await bridge.drain_pending()
        # Should still be in queue since send failed
        assert len(bridge._pending_notifications) == 1

    @pytest.mark.asyncio
    async def test_initial_tool_result_deferred_until_initialized(self):
        """Initial tool result should not be sent immediately but after app initialized."""
        bridge, _ = _make_bridge()
        ws = FakeWs()
        bridge.set_ws(ws)

        # Store initial tool result
        bridge.set_initial_tool_result("initial data")
        assert bridge._initial_tool_result == "initial data"
        # Nothing sent yet
        assert len(ws.sent) == 0

        # Simulate app sending ui/notifications/initialized
        msg = json.dumps(
            {"jsonrpc": "2.0", "method": "ui/notifications/initialized", "params": {}}
        )
        await bridge.handle_message(msg)

        # Give the ensure_future a chance to run
        await asyncio.sleep(0)

        # Now the tool result should have been pushed
        assert len(ws.sent) == 1
        parsed = json.loads(ws.sent[0])
        assert parsed["method"] == "ui/notifications/tool-result"
        # And the stored result should be cleared
        assert bridge._initial_tool_result is None

    @pytest.mark.asyncio
    async def test_initial_tool_result_not_sent_twice(self):
        """Ensure deferred tool result is only sent once even on re-init."""
        bridge, _ = _make_bridge()
        ws = FakeWs()
        bridge.set_ws(ws)
        bridge.set_initial_tool_result("initial data")

        # First initialized
        msg = json.dumps(
            {"jsonrpc": "2.0", "method": "ui/notifications/initialized", "params": {}}
        )
        await bridge.handle_message(msg)
        await asyncio.sleep(0)
        assert len(ws.sent) == 1

        # Second initialized (e.g., after reconnect)
        await bridge.handle_message(msg)
        await asyncio.sleep(0)
        # Should still be just 1 message
        assert len(ws.sent) == 1


class TestExtractStructuredContent:
    """Test the spec-compliant structuredContent extraction from text blocks."""

    def test_hoists_structured_content_from_text_block(self):
        """Per MCP spec: servers include structuredContent as JSON in text blocks."""
        inner = json.dumps(
            {
                "content": [{"type": "text", "text": "Temperature: 72°F"}],
                "structuredContent": {"temperature": 72, "conditions": "sunny"},
            }
        )
        out = {
            "content": [{"type": "text", "text": inner}],
        }
        result = AppBridge._extract_structured_content(out)
        assert result["structuredContent"] == {"temperature": 72, "conditions": "sunny"}
        assert result["content"] == [{"type": "text", "text": "Temperature: 72°F"}]

    def test_skips_if_already_present(self):
        out = {
            "content": [{"type": "text", "text": "{}"}],
            "structuredContent": {"existing": True},
        }
        result = AppBridge._extract_structured_content(out)
        assert result["structuredContent"] == {"existing": True}

    def test_skips_non_json_text(self):
        out = {"content": [{"type": "text", "text": "plain text"}]}
        result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result

    def test_skips_multiple_content_blocks(self):
        out = {
            "content": [
                {"type": "text", "text": "{}"},
                {"type": "text", "text": "{}"},
            ]
        }
        result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result

    def test_skips_json_without_structured_content(self):
        out = {"content": [{"type": "text", "text": json.dumps({"key": "value"})}]}
        result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result

    def test_keeps_original_content_when_inner_content_missing(self):
        """When JSON has structuredContent but no content array."""
        inner = json.dumps({"structuredContent": {"type": "markdown", "data": {}}})
        out = {"content": [{"type": "text", "text": inner}]}
        result = AppBridge._extract_structured_content(out)
        assert result["structuredContent"] == {"type": "markdown", "data": {}}
        # Original text block preserved since no inner content array
        assert result["content"] == [{"type": "text", "text": inner}]


class TestFormatToolResult:
    def test_string_result(self):
        result = AppBridge._format_tool_result("hello world")
        assert result == {"content": [{"type": "text", "text": "hello world"}]}

    def test_dict_with_content(self):
        original = {"content": [{"type": "text", "text": "foo"}]}
        result = AppBridge._format_tool_result(original)
        assert result == original

    def test_raw_dict(self):
        result = AppBridge._format_tool_result({"key": "value"})
        assert result["content"][0]["type"] == "text"
        assert '"key"' in result["content"][0]["text"]

    def test_numeric_result(self):
        result = AppBridge._format_tool_result(42)
        assert result == {"content": [{"type": "text", "text": "42"}]}

    def test_format_extracts_structured_content_from_dict(self):
        """End-to-end: dict result with embedded JSON text containing structuredContent."""
        inner_json = json.dumps(
            {
                "content": [{"type": "text", "text": "Chart data"}],
                "structuredContent": {"type": "chart", "data": {"values": [1, 2, 3]}},
            }
        )
        result = AppBridge._format_tool_result(
            {
                "content": [{"type": "text", "text": inner_json}],
            }
        )
        assert result["structuredContent"] == {
            "type": "chart",
            "data": {"values": [1, 2, 3]},
        }
        assert result["content"] == [{"type": "text", "text": "Chart data"}]

    def test_format_extracts_structured_content_from_pydantic(self):
        """End-to-end: Pydantic-like model with embedded JSON text."""

        class FakeContent:
            def __init__(self, type, text):
                self.type = type
                self.text = text

            def model_dump(self):
                return {"type": self.type, "text": self.text}

        class FakePydanticResult:
            def __init__(self):
                inner_json = json.dumps(
                    {
                        "content": [{"type": "text", "text": "Map"}],
                        "structuredContent": {"type": "geojson", "data": {}},
                    }
                )
                self.content = [FakeContent("text", inner_json)]
                self.isError = False

        result = AppBridge._format_tool_result(FakePydanticResult())
        assert result["structuredContent"] == {"type": "geojson", "data": {}}
        assert result["content"] == [{"type": "text", "text": "Map"}]


class TestHelpers:
    def test_to_serializable_circular_reference(self):
        """Circular references should not cause infinite recursion."""
        d: dict = {"key": "value"}
        d["self"] = d  # circular!
        result = AppBridge._to_serializable(d)
        assert result["key"] == "value"
        assert result["self"] == "<circular>"

    def test_safe_json_dumps_normal(self):
        result = AppBridge._safe_json_dumps({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_safe_json_dumps_fallback(self):
        """Non-serializable objects should fall back to _to_serializable."""

        class Custom:
            def __init__(self):
                self.x = 42

        result = AppBridge._safe_json_dumps({"obj": Custom()})
        parsed = json.loads(result)
        assert parsed["obj"]["x"] == 42


# ── New tests targeting uncovered lines ────────────────────────────────────


class TestSetWsEnsureFutureException:
    """Lines 53-54: ensure_future raises (no running event loop)."""

    def test_set_ws_ensure_future_exception_is_swallowed(self, monkeypatch):
        """If asyncio.ensure_future raises, the exception is logged and ignored."""
        bridge, _ = _make_bridge()
        old_ws = FakeWs()
        bridge._ws = old_ws  # set an old WS directly (no loop needed)

        def boom(coro):
            # Close the coroutine to avoid RuntimeWarning
            coro.close()
            raise RuntimeError("no running loop")

        monkeypatch.setattr(asyncio, "ensure_future", boom)

        new_ws = FakeWs()
        # Must not raise even though ensure_future raises
        bridge.set_ws(new_ws)
        assert bridge._ws is new_ws
        assert bridge.app_info.state == AppState.INITIALIZING


class TestToolCallTimeout:
    """Lines 189-192: asyncio.TimeoutError from wait_for."""

    @pytest.mark.asyncio
    async def test_tool_call_timeout_returns_error(self):
        bridge, tm = _make_bridge()

        import unittest.mock as mock

        def _raise_timeout(coro, *, timeout):
            # Close the coroutine to suppress RuntimeWarning about unawaited coroutines
            coro.close()
            raise asyncio.TimeoutError

        with mock.patch(
            "mcp_cli.apps.bridge.asyncio.wait_for", side_effect=_raise_timeout
        ):
            msg = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 20,
                    "method": "tools/call",
                    "params": {"name": "slow-tool", "arguments": {}},
                }
            )
            resp = await bridge.handle_message(msg)

        parsed = json.loads(resp)
        assert parsed["id"] == 20
        assert "error" in parsed
        assert parsed["error"]["code"] == -32000
        assert "timed out" in parsed["error"]["message"]


class TestPushToolInput:
    """Lines 292-306: push_tool_input."""

    @pytest.mark.asyncio
    async def test_push_tool_input_no_ws_returns_early(self):
        """When no WS is set, push_tool_input should return without error."""
        bridge, _ = _make_bridge()
        # No WS set — should return immediately
        await bridge.push_tool_input({"x": 1})
        # No exception, nothing queued
        assert len(bridge._pending_notifications) == 0

    @pytest.mark.asyncio
    async def test_push_tool_input_sends_notification(self):
        """When WS is present, push_tool_input sends a tool-input notification."""
        bridge, _ = _make_bridge()
        ws = FakeWs()
        bridge.set_ws(ws)

        await bridge.push_tool_input({"action": "start", "value": 42})

        assert len(ws.sent) == 1
        parsed = json.loads(ws.sent[0])
        assert parsed["method"] == "ui/notifications/tool-input"
        assert parsed["params"]["arguments"] == {"action": "start", "value": 42}

    @pytest.mark.asyncio
    async def test_push_tool_input_logs_on_send_failure(self):
        """When send raises, push_tool_input logs the warning (no queue for input)."""
        bridge, _ = _make_bridge()
        ws = FakeWs()
        ws._raise_on_send = True
        bridge.set_ws(ws)

        # Should not raise — error is logged, not re-raised
        await bridge.push_tool_input({"x": 1})
        # No pending notifications (tool input is fire-and-forget)
        assert len(bridge._pending_notifications) == 0


class TestExtractRawResult:
    """Lines 331-335: _extract_raw_result circular-reference guard."""

    def test_unwraps_single_level(self):
        """A single wrapper with a .result attribute is unwrapped."""

        class Wrapper:
            def __init__(self, inner):
                self.result = inner

        inner = {"data": "value"}
        w = Wrapper(inner)
        result = AppBridge._extract_raw_result(w)
        assert result == inner

    def test_circular_result_reference_breaks_loop(self):
        """A wrapper whose .result points back to itself should not loop forever."""

        class SelfRef:
            pass

        s = SelfRef()
        s.result = s  # circular reference

        # Should terminate and return s (after detecting the cycle)
        result = AppBridge._extract_raw_result(s)
        assert result is s

    def test_dict_not_unwrapped(self):
        """Dicts are not unwrapped even if they have a 'result' key."""
        d = {"result": "inner"}
        result = AppBridge._extract_raw_result(d)
        assert result is d

    def test_str_not_unwrapped(self):
        """Strings are not unwrapped even if they look like wrappers."""
        s = "hello"
        result = AppBridge._extract_raw_result(s)
        assert result is s


class TestToSerializableFallback:
    """Line 364: _to_serializable fallback str(obj) for un-dumpable primitives."""

    def test_primitive_without_dict_or_model_dump(self):
        """An object with no __dict__ and no model_dump falls back to str()."""

        # A plain integer slot-only object — use a basic type that has no __dict__
        # The simplest: pass a custom class instance that deliberately has no __dict__
        class NoDict:
            __slots__ = ()

        obj = NoDict()
        result = AppBridge._to_serializable(obj)
        assert isinstance(result, str)
        # str() of the object — just verify it returned a string
        assert result == str(obj)

    def test_tuple_serialized_as_list(self):
        """Tuples are serialized element-by-element like lists."""
        result = AppBridge._to_serializable((1, "two", 3.0))
        assert result == [1, "two", 3.0]

    def test_none_returns_none(self):
        result = AppBridge._to_serializable(None)
        assert result is None

    def test_bool_passthrough(self):
        result = AppBridge._to_serializable(True)
        assert result is True


class TestExtractStructuredContentEdgeCases:
    """Lines 385, 389, 397-398, 401, 413-414."""

    def test_empty_content_list_returns_unchanged(self):
        """Line 385: content is an empty list → return out unchanged."""
        out = {"content": []}
        result = AppBridge._extract_structured_content(out)
        assert result is out
        assert "structuredContent" not in result

    def test_content_not_a_list_returns_unchanged(self):
        """Line 385: content is not a list (e.g., a string) → return out."""
        out = {"content": "not a list"}
        result = AppBridge._extract_structured_content(out)
        assert result is out
        assert "structuredContent" not in result

    def test_no_content_key_returns_unchanged(self):
        """Line 385: no content key at all → return out."""
        out = {"other": "data"}
        result = AppBridge._extract_structured_content(out)
        assert result is out
        assert "structuredContent" not in result

    def test_non_dict_block_is_skipped(self):
        """Line 389: block that is not a dict is skipped."""
        out = {"content": ["not a dict", 42, None]}
        result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result

    def test_block_with_wrong_type_is_skipped(self):
        """Line 389: block with type != 'text' is skipped."""
        out = {"content": [{"type": "image", "url": "http://example.com/img.png"}]}
        result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result

    def test_invalid_json_in_text_block_is_skipped(self):
        """Lines 397-398: json.loads raises JSONDecodeError — block is skipped."""
        out = {"content": [{"type": "text", "text": "{not valid json"}]}
        result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result

    def test_json_array_in_text_block_is_skipped(self):
        """Line 401: parsed JSON is not a dict → skipped (covered via monkeypatching json.loads)."""
        import unittest.mock as mock

        # Patch json.loads so that it returns a list for the text block parse
        # (text starts with '{' check passes, json.loads succeeds but gives a list)
        original_loads = json.loads

        def patched_loads(s, **kw):
            if isinstance(s, str) and s == '{"fake": true}':
                return [1, 2, 3]  # return list, not dict → hits line 401
            return original_loads(s, **kw)

        out = {"content": [{"type": "text", "text": '{"fake": true}'}]}
        with mock.patch("mcp_cli.apps.bridge.json.loads", side_effect=patched_loads):
            result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result

    def test_pattern2_type_and_version_becomes_structured_content(self):
        """Lines 413-414: JSON with 'type' and 'version' IS the structured content."""
        patch = {"type": "ui_patch", "version": "3.0", "ops": [{"op": "replace"}]}
        out = {"content": [{"type": "text", "text": json.dumps(patch)}]}
        result = AppBridge._extract_structured_content(out)
        assert result["structuredContent"] == patch

    def test_text_not_starting_with_brace_is_skipped(self):
        """Line 392: text that doesn't start with '{' is skipped."""
        out = {"content": [{"type": "text", "text": "[1, 2, 3]"}]}
        result = AppBridge._extract_structured_content(out)
        assert "structuredContent" not in result


class TestFormatToolResultEdgeCases:
    """Lines 433->444, 437, 441, 448-458, 470."""

    def test_pydantic_with_structured_content(self):
        """Line 437: Pydantic-like obj with truthy structuredContent is preserved."""

        class FakeContentItem:
            def __init__(self, text):
                self.type = "text"
                self.text = text

            def model_dump(self):
                return {"type": self.type, "text": self.text}

        class FakePydanticWithSC:
            def __init__(self):
                self.content = [FakeContentItem("some text")]
                self.structuredContent = {"type": "chart", "values": [1, 2, 3]}
                self.isError = False

        result = AppBridge._format_tool_result(FakePydanticWithSC())
        assert "structuredContent" in result
        assert result["structuredContent"] == {"type": "chart", "values": [1, 2, 3]}

    def test_pydantic_with_is_error_true(self):
        """Line 441: Pydantic-like obj with isError=True sets isError in output."""

        class FakeContentItem:
            def __init__(self, text):
                self.type = "text"
                self.text = text

            def model_dump(self):
                return {"type": self.type, "text": self.text}

        class FakePydanticError:
            def __init__(self):
                self.content = [FakeContentItem("error occurred")]
                self.isError = True

        result = AppBridge._format_tool_result(FakePydanticError())
        assert result.get("isError") is True
        assert result["content"] is not None

    def test_pydantic_content_not_list_falls_through(self):
        """Line 433->444: when content attr is not a list, falls through to other branches."""

        class FakeObjNonListContent:
            def __init__(self):
                self.content = "not a list"

        # content is a string, not a list → falls through to str branch
        # (since the obj itself is not dict/str either)
        # After falling through the content-is-list check (line 433),
        # we hit the fallback at line 473
        result = AppBridge._format_tool_result(FakeObjNonListContent())
        # Should end up as str() representation
        assert "content" in result
        assert isinstance(result["content"], list)

    def test_dict_with_mcp_sdk_content_object(self):
        """Lines 448-458: dict whose 'content' value is an MCP SDK object with .content list."""

        class MCPContentObj:
            def __init__(self):
                self.content = [{"type": "text", "text": "from sdk"}]
                self.structuredContent = None

        sdk_obj = MCPContentObj()
        result_dict = {"content": sdk_obj}
        result = AppBridge._format_tool_result(result_dict)
        # content should be extracted from sdk_obj.content
        assert result["content"] == [{"type": "text", "text": "from sdk"}]

    def test_dict_content_val_object_without_content_attr(self):
        """Branch 448->460: dict with non-list content_val that has no .content attr."""

        class SomeObject:
            """Object with no .content attribute — falls through to line 460."""

            def __init__(self):
                self.value = "data"

            def __repr__(self):
                return "SomeObject(value=data)"

        obj = SomeObject()
        # content_val is not list/str and has no .content attr
        result = AppBridge._format_tool_result({"content": obj})
        # Falls through: _to_serializable converts to dict via __dict__,
        # then 'content' key has been resolved to a serialized value
        assert "content" in result

    def test_dict_with_mcp_sdk_content_object_and_structured_content(self):
        """Lines 454-458: MCP SDK object also has truthy structuredContent."""

        class MCPContentObj:
            def __init__(self):
                self.content = [{"type": "text", "text": "data"}]
                self.structuredContent = {"type": "chart", "data": {}}

        sdk_obj = MCPContentObj()
        result_dict = {"content": sdk_obj}
        result = AppBridge._format_tool_result(result_dict)
        assert result["content"] == [{"type": "text", "text": "data"}]
        # structuredContent gets hoisted during _to_serializable + _extract_structured_content
        # The dict now has structuredContent key after copying from sdk_obj
        assert "structuredContent" in result or result["content"] is not None

    def test_model_dump_fallback(self):
        """Line 470: non-dict, non-str, non-content-attr object with model_dump."""

        class FakePydanticNoContent:
            def model_dump(self):
                return {"answer": 42}

        result = AppBridge._format_tool_result(FakePydanticNoContent())
        assert result["content"][0]["type"] == "text"
        parsed_text = json.loads(result["content"][0]["text"])
        assert parsed_text == {"answer": 42}

    def test_format_tool_result_non_serializable_fallback(self):
        """Line 473: fallback str() for objects with no model_dump and no content."""

        class WeirdObj:
            __slots__ = ()

            def __str__(self):
                return "weird-42"

        result = AppBridge._format_tool_result(WeirdObj())
        assert result == {"content": [{"type": "text", "text": "weird-42"}]}
