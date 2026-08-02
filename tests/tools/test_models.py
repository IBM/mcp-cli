# tools/test_models.py
"""
Comprehensive tests for tools/models.py - Pydantic models.
Target: 90%+ coverage
"""

from datetime import datetime

import pytest

from mcp_cli.tools.models import (
    ConversationMessage,
    ExperimentalCapabilities,
    FunctionDefinition,
    LLMToolDefinition,
    PromptInfo,
    ResourceInfo,
    ServerCapabilities,
    ServerInfo,
    ToolCallMessage,
    ToolCallResult,
    ToolDefinitionInput,
    ToolInfo,
    ToolInputSchema,
    ToolType,
    TransportServerConfig,
    TransportType,
    ValidationResult,
)


class TestToolInfo:
    """Test ToolInfo Pydantic model."""

    def test_toolinfo_defaults_and_assignment(self):
        """Test ToolInfo with defaults and full assignment."""
        ti = ToolInfo(name="foo", namespace="bar")
        assert ti.name == "foo"
        assert ti.namespace == "bar"
        # defaults
        assert ti.description is None
        assert ti.parameters is None
        assert ti.is_async is False
        assert ti.tags == []
        assert ti.supports_streaming is False

        # with all fields
        ti2 = ToolInfo(
            name="x",
            namespace="y",
            description="desc",
            parameters={"p": 1},
            is_async=True,
            tags=["a", "b"],
            supports_streaming=True,
        )
        assert ti2.description == "desc"
        assert ti2.parameters == {"p": 1}
        assert ti2.is_async is True
        assert ti2.tags == ["a", "b"]
        assert ti2.supports_streaming is True

    def test_toolinfo_fully_qualified_name(self):
        """Test fully_qualified_name property."""
        ti = ToolInfo(name="test", namespace="server")
        assert ti.fully_qualified_name == "server.test"

        ti_no_namespace = ToolInfo(name="test", namespace="")
        assert ti_no_namespace.fully_qualified_name == "test"

    def test_toolinfo_display_name(self):
        """Test display_name property."""
        ti = ToolInfo(name="my_tool", namespace="server")
        assert ti.display_name == "my_tool"

    def test_toolinfo_has_parameters(self):
        """Test has_parameters property."""
        ti_no_params = ToolInfo(name="test", namespace="ns")
        assert ti_no_params.has_parameters is False

        ti_empty_params = ToolInfo(name="test", namespace="ns", parameters={})
        assert ti_empty_params.has_parameters is False

        ti_with_params = ToolInfo(
            name="test",
            namespace="ns",
            parameters={"properties": {"arg1": {"type": "string"}}},
        )
        assert ti_with_params.has_parameters is True

    def test_toolinfo_required_parameters(self):
        """Test required_parameters property."""
        ti_no_params = ToolInfo(name="test", namespace="ns")
        assert ti_no_params.required_parameters == []

        ti_with_required = ToolInfo(
            name="test",
            namespace="ns",
            parameters={"required": ["arg1", "arg2"]},
        )
        assert ti_with_required.required_parameters == ["arg1", "arg2"]

        ti_no_required = ToolInfo(
            name="test", namespace="ns", parameters={"properties": {}}
        )
        assert ti_no_required.required_parameters == []

    def test_toolinfo_to_openai_format(self):
        """Test to_openai_format method."""
        ti = ToolInfo(
            name="my_tool",
            namespace="server",
            description="Test tool",
            parameters={
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
                "required": ["arg1"],
            },
        )
        openai_format = ti.to_llm_format().to_dict()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "my_tool"
        assert openai_format["function"]["description"] == "Test tool"
        assert openai_format["function"]["parameters"]["required"] == ["arg1"]

    def test_toolinfo_to_llm_format_no_description(self):
        """Test to_llm_format with no description."""
        ti = ToolInfo(name="tool", namespace="ns")
        openai_format = ti.to_llm_format().to_dict()
        assert openai_format["function"]["description"] == "No description provided"


class TestServerInfo:
    """Test ServerInfo Pydantic model."""

    def test_serverinfo_fields(self):
        """Test ServerInfo basic fields."""
        si = ServerInfo(id=1, name="s1", status="Up", tool_count=5, namespace="ns")
        assert si.id == 1
        assert si.name == "s1"
        assert si.status == "Up"
        assert si.tool_count == 5
        assert si.namespace == "ns"

    def test_serverinfo_defaults(self):
        """Test ServerInfo default values."""
        si = ServerInfo(id=1, name="s1", status="ok", tool_count=0, namespace="ns")
        assert si.enabled is True
        assert si.connected is False
        assert si.transport == "stdio"
        assert si.capabilities == {}
        assert si.description is None
        assert si.version is None
        assert si.command is None
        assert si.args == []
        assert si.env == {}

    def test_serverinfo_is_healthy(self):
        """Test is_healthy property."""
        si_healthy = ServerInfo(
            id=1,
            name="s1",
            status="healthy",
            tool_count=5,
            namespace="ns",
            connected=True,
        )
        assert si_healthy.is_healthy is True

        si_not_connected = ServerInfo(
            id=1,
            name="s1",
            status="healthy",
            tool_count=5,
            namespace="ns",
            connected=False,
        )
        assert si_not_connected.is_healthy is False

        si_bad_status = ServerInfo(
            id=1,
            name="s1",
            status="error",
            tool_count=5,
            namespace="ns",
            connected=True,
        )
        assert si_bad_status.is_healthy is False

    def test_serverinfo_display_status(self):
        """Test display_status property."""
        si_enabled = ServerInfo(
            id=1,
            name="s1",
            status="running",
            tool_count=0,
            namespace="ns",
            enabled=True,
            connected=True,
        )
        assert si_enabled.display_status == "running"

        si_disabled = ServerInfo(
            id=1,
            name="s1",
            status="running",
            tool_count=0,
            namespace="ns",
            enabled=False,
        )
        assert si_disabled.display_status == "disabled"

        si_disconnected = ServerInfo(
            id=1,
            name="s1",
            status="running",
            tool_count=0,
            namespace="ns",
            enabled=True,
            connected=False,
        )
        assert si_disconnected.display_status == "disconnected"

    def test_serverinfo_display_description(self):
        """Test display_description property."""
        si_with_desc = ServerInfo(
            id=1,
            name="s1",
            status="ok",
            tool_count=0,
            namespace="ns",
            description="Custom description",
        )
        assert si_with_desc.display_description == "Custom description"

        si_no_desc = ServerInfo(
            id=1, name="s1", status="ok", tool_count=0, namespace="ns"
        )
        assert si_no_desc.display_description == "s1 MCP server"

    def test_serverinfo_has_tools(self):
        """Test has_tools property."""
        si_with_tools = ServerInfo(
            id=1, name="s1", status="ok", tool_count=5, namespace="ns"
        )
        assert si_with_tools.has_tools is True

        si_no_tools = ServerInfo(
            id=1, name="s1", status="ok", tool_count=0, namespace="ns"
        )
        assert si_no_tools.has_tools is False


class TestToolCallResult:
    """Test ToolCallResult Pydantic model."""

    def test_toolcallresult_defaults_and_assignment(self):
        """Test ToolCallResult with minimal and full fields."""
        # minimal
        tr = ToolCallResult(tool_name="t", success=True)
        assert tr.tool_name == "t"
        assert tr.success is True
        assert tr.result is None
        assert tr.error is None
        assert tr.execution_time is None

        # full
        tr2 = ToolCallResult(
            tool_name="u",
            success=False,
            result={"x": 1},
            error="oops",
            execution_time=0.123,
        )
        assert tr2.tool_name == "u"
        assert tr2.success is False
        assert tr2.result == {"x": 1}
        assert tr2.error == "oops"
        assert tr2.execution_time == pytest.approx(0.123)

    def test_toolcallresult_display_result(self):
        """Test display_result property."""
        tr_success = ToolCallResult(tool_name="t", success=True, result="output text")
        assert tr_success.display_result == "output text"

        tr_dict = ToolCallResult(tool_name="t", success=True, result={"key": "value"})
        assert '"key"' in tr_dict.display_result
        assert '"value"' in tr_dict.display_result

        tr_error = ToolCallResult(tool_name="t", success=False, error="Failed")
        assert tr_error.display_result == "Error: Failed"

        tr_unknown_error = ToolCallResult(tool_name="t", success=False)
        assert "Unknown error" in tr_unknown_error.display_result

    def test_toolcallresult_has_error(self):
        """Test has_error property."""
        tr_success = ToolCallResult(tool_name="t", success=True)
        assert tr_success.has_error is False

        tr_failed = ToolCallResult(tool_name="t", success=False)
        assert tr_failed.has_error is True

        tr_with_error = ToolCallResult(tool_name="t", success=True, error="warning")
        assert tr_with_error.has_error is True

    def test_toolcallresult_to_conversation_history(self):
        """Test to_conversation_history method."""
        tr_success = ToolCallResult(tool_name="t", success=True, result="output")
        assert tr_success.to_conversation_history() == "output"

        tr_failed = ToolCallResult(tool_name="t", success=False, error="Failed")
        assert "Tool execution failed: Failed" in tr_failed.to_conversation_history()


class TestResourceInfo:
    """Test ResourceInfo Pydantic model."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (
                {"id": "i1", "name": "n1", "type": "t1", "foo": 42},
                {"id": "i1", "name": "n1", "type": "t1", "extra": {"foo": 42}},
            ),
            ({}, {"id": None, "name": None, "type": None, "extra": {}}),
            # MCP wire shape: uri -> id, mimeType -> type (originals kept in extra)
            (
                {"uri": "debug://x", "name": "x", "mimeType": "text/plain"},
                {
                    "id": "debug://x",
                    "name": "x",
                    "type": "text/plain",
                    "extra": {"uri": "debug://x", "mimeType": "text/plain"},
                },
            ),
            # Explicit id/type win over uri/mimeType
            (
                {"id": "i", "uri": "u", "type": "t", "mimeType": "m", "name": "n"},
                {
                    "id": "i",
                    "name": "n",
                    "type": "t",
                    "extra": {"uri": "u", "mimeType": "m"},
                },
            ),
        ],
    )
    def test_resourceinfo_from_raw_dict(self, raw, expected):
        """Test ResourceInfo.from_raw with dict input."""
        ri = ResourceInfo.from_raw(raw)
        assert ri.id == expected["id"]
        assert ri.name == expected["name"]
        assert ri.type == expected["type"]
        assert ri.extra == expected["extra"]

    @pytest.mark.parametrize("primitive", ["just a string", 123, 4.56, True, None])
    def test_resourceinfo_from_raw_primitive(self, primitive):
        """Test ResourceInfo.from_raw with primitive input."""
        ri = ResourceInfo.from_raw(primitive)
        # id, name, type stay None
        assert ri.id is None and ri.name is None and ri.type is None
        # primitive ends up under extra["value"]
        assert ri.extra == {"value": primitive}

    def test_resourceinfo_direct_creation(self):
        """Test direct creation of ResourceInfo."""
        ri = ResourceInfo(
            id="res1",
            name="Resource 1",
            type="file",
            extra={"size": 1024, "mime": "text/plain"},
        )
        assert ri.id == "res1"
        assert ri.name == "Resource 1"
        assert ri.type == "file"
        assert ri.extra["size"] == 1024
        assert ri.extra["mime"] == "text/plain"


class TestPromptInfo:
    """Test PromptInfo Pydantic model."""

    def test_promptinfo_from_raw_dict(self):
        """from_raw normalizes the common fields and preserves the rest in extra."""
        pi = PromptInfo.from_raw(
            {
                "name": "greet",
                "description": "Greet someone",
                "arguments": [{"name": "who", "required": True}],
                "title": "Greeter",
            }
        )
        assert pi.name == "greet"
        assert pi.description == "Greet someone"
        assert pi.arguments == [{"name": "who", "required": True}]
        assert pi.extra == {"title": "Greeter"}

    def test_promptinfo_from_raw_defaults_arguments(self):
        """Missing arguments default to an empty list (never None)."""
        pi = PromptInfo.from_raw({"name": "p"})
        assert pi.name == "p"
        assert pi.description is None
        assert pi.arguments == []

    @pytest.mark.parametrize("primitive", ["a string", 7, None])
    def test_promptinfo_from_raw_primitive(self, primitive):
        """Primitives are wrapped under extra['value']."""
        pi = PromptInfo.from_raw(primitive)
        assert pi.name is None and pi.description is None
        assert pi.arguments == []
        assert pi.extra == {"value": primitive}


# ----------------------------------------------------------------------------
# Additional coverage tests for 90%+
# ----------------------------------------------------------------------------


class TestTransportType:
    """Test TransportType enum."""

    def test_transport_type_values(self):
        """Test all transport type values."""
        assert TransportType.STDIO.value == "stdio"
        assert TransportType.HTTP.value == "http"
        assert TransportType.SSE.value == "sse"
        assert TransportType.UNKNOWN.value == "unknown"


class TestToolType:
    """Test ToolType enum."""

    def test_tool_type_values(self):
        """Test tool type values."""
        assert ToolType.FUNCTION.value == "function"


class TestExperimentalCapabilities:
    """Test ExperimentalCapabilities model."""

    def test_experimental_capabilities_defaults(self):
        """Test default values."""
        ec = ExperimentalCapabilities()
        assert ec.sampling is False
        assert ec.logging is False
        assert ec.streaming is False

    def test_experimental_capabilities_with_values(self):
        """Test with custom values."""
        ec = ExperimentalCapabilities(sampling=True, logging=True, streaming=True)
        assert ec.sampling is True
        assert ec.logging is True
        assert ec.streaming is True


class TestServerCapabilities:
    """Test ServerCapabilities model."""

    def test_server_capabilities_defaults(self):
        """Test default values."""
        sc = ServerCapabilities()
        assert sc.tools is False
        assert sc.prompts is False
        assert sc.resources is False
        assert isinstance(sc.experimental, ExperimentalCapabilities)

    def test_server_capabilities_from_dict(self):
        """Test from_dict class method."""
        data = {
            "tools": True,
            "prompts": True,
            "resources": True,
            "experimental": {"sampling": True, "logging": False, "streaming": True},
        }
        sc = ServerCapabilities.from_dict(data)
        assert sc.tools is True
        assert sc.prompts is True
        assert sc.resources is True
        assert sc.experimental.sampling is True
        assert sc.experimental.streaming is True

    def test_server_capabilities_from_dict_no_experimental(self):
        """Test from_dict without experimental."""
        data = {"tools": True}
        sc = ServerCapabilities.from_dict(data)
        assert sc.tools is True
        assert isinstance(sc.experimental, ExperimentalCapabilities)

    def test_server_capabilities_to_dict(self):
        """Test to_dict method."""
        sc = ServerCapabilities(tools=True, prompts=True)
        result = sc.to_dict()
        assert result["tools"] is True
        assert result["prompts"] is True
        assert "experimental" in result


class TestServerInfoCapabilities:
    """Additional tests for ServerInfo capabilities methods."""

    def test_serverinfo_get_capabilities_typed(self):
        """Test get_capabilities_typed method."""
        si = ServerInfo(
            id=1,
            name="test",
            status="ok",
            tool_count=5,
            namespace="ns",
            capabilities={"tools": True, "prompts": True},
        )
        caps = si.get_capabilities_typed()
        assert isinstance(caps, ServerCapabilities)
        assert caps.tools is True
        assert caps.prompts is True


class TestToolCallResultChuk:
    """Test ToolCallResult chuk integration."""

    def test_from_chuk_result(self):
        """Test from_chuk_result class method."""

        class MockChukResult:
            tool = "test_tool"
            result = {"data": "value"}
            error = None
            start_time = datetime(2024, 1, 1, 12, 0, 0)
            end_time = datetime(2024, 1, 1, 12, 0, 5)

        chuk_result = MockChukResult()
        tcr = ToolCallResult.from_chuk_result(chuk_result)

        assert tcr.tool_name == "test_tool"
        assert tcr.success is True
        assert tcr.result == {"data": "value"}
        assert tcr.error is None
        assert tcr.execution_time == 5.0
        assert tcr.chuk_result is chuk_result

    def test_from_chuk_result_with_error(self):
        """Test from_chuk_result with error."""

        class MockChukResult:
            tool = "test_tool"
            result = None
            error = "Something failed"
            start_time = None
            end_time = None

        tcr = ToolCallResult.from_chuk_result(MockChukResult())

        assert tcr.tool_name == "test_tool"
        assert tcr.success is False
        assert tcr.error == "Something failed"
        assert tcr.execution_time is None

    def test_from_chuk_result_no_times(self):
        """Test from_chuk_result without start/end times."""

        class MockChukResult:
            tool = "test_tool"
            result = {"data": "value"}
            error = None
            start_time = datetime(2024, 1, 1, 12, 0, 0)
            end_time = None  # No end time

        tcr = ToolCallResult.from_chuk_result(MockChukResult())
        assert tcr.execution_time is None

    def test_is_cached_property(self):
        """Test is_cached property."""
        # No chuk_result
        tcr1 = ToolCallResult(tool_name="t", success=True)
        assert tcr1.is_cached is False

        # With chuk_result without cached attr
        class MockChukNoCached:
            pass

        tcr2 = ToolCallResult(
            tool_name="t", success=True, chuk_result=MockChukNoCached()
        )
        assert tcr2.is_cached is False

        # With cached=True
        class MockChukCached:
            cached = True

        tcr3 = ToolCallResult(tool_name="t", success=True, chuk_result=MockChukCached())
        assert tcr3.is_cached is True

    def test_attempts_property(self):
        """Test attempts property."""
        # No chuk_result
        tcr1 = ToolCallResult(tool_name="t", success=True)
        assert tcr1.attempts == 1

        # With attempts
        class MockChukAttempts:
            attempts = 3

        tcr2 = ToolCallResult(
            tool_name="t", success=True, chuk_result=MockChukAttempts()
        )
        assert tcr2.attempts == 3

    def test_machine_property(self):
        """Test machine property."""
        # No chuk_result
        tcr1 = ToolCallResult(tool_name="t", success=True)
        assert tcr1.machine is None

        # With machine
        class MockChukMachine:
            machine = "server1.local"

        tcr2 = ToolCallResult(
            tool_name="t", success=True, chuk_result=MockChukMachine()
        )
        assert tcr2.machine == "server1.local"

        # With None machine
        class MockChukNullMachine:
            machine = None

        tcr3 = ToolCallResult(
            tool_name="t", success=True, chuk_result=MockChukNullMachine()
        )
        assert tcr3.machine is None

    def test_extract_mcp_text_content(self):
        """Test _extract_mcp_text_content method."""
        tcr = ToolCallResult(tool_name="t", success=True)

        # Non-dict result
        assert tcr._extract_mcp_text_content("not a dict") is None

        # Dict without content key
        assert tcr._extract_mcp_text_content({"data": "value"}) is None

        # MCP structure
        class MockToolResultContent:
            content = [{"type": "text", "text": "Hello"}]

        result = {"content": MockToolResultContent()}
        assert tcr._extract_mcp_text_content(result) == "Hello"

        # Multiple text blocks
        class MockToolResultMultiple:
            content = [
                {"type": "text", "text": "Line 1"},
                {"type": "image", "data": "..."},
                {"type": "text", "text": "Line 2"},
            ]

        result2 = {"content": MockToolResultMultiple()}
        assert tcr._extract_mcp_text_content(result2) == "Line 1\nLine 2"

    def test_display_result_with_mcp_content(self):
        """Test display_result with MCP content structure."""

        class MockToolResultContent:
            content = [{"type": "text", "text": "MCP Output"}]

        tcr = ToolCallResult(
            tool_name="t", success=True, result={"content": MockToolResultContent()}
        )
        assert tcr.display_result == "MCP Output"

    def test_display_result_none_result(self):
        """Test display_result with None result."""
        tcr = ToolCallResult(tool_name="t", success=True, result=None)
        assert tcr.display_result == ""

    def test_display_result_non_serializable_dict(self):
        """Test display_result with non-serializable dict."""

        class NonSerializable:
            pass

        tcr = ToolCallResult(
            tool_name="t", success=True, result={"obj": NonSerializable()}
        )
        # Should fall back to str()
        assert "obj" in tcr.display_result

    def test_to_conversation_history_with_mcp_content(self):
        """Test to_conversation_history with MCP content."""

        class MockToolResultContent:
            content = [{"type": "text", "text": "MCP History"}]

        tcr = ToolCallResult(
            tool_name="t", success=True, result={"content": MockToolResultContent()}
        )
        assert tcr.to_conversation_history() == "MCP History"

    def test_to_conversation_history_none_result(self):
        """Test to_conversation_history with None result."""
        tcr = ToolCallResult(tool_name="t", success=True, result=None)
        assert tcr.to_conversation_history() == ""

    def test_to_conversation_history_dict_result(self):
        """Test to_conversation_history with dict result."""
        tcr = ToolCallResult(tool_name="t", success=True, result={"key": "value"})
        result = tcr.to_conversation_history()
        assert '"key"' in result
        assert '"value"' in result

    def test_to_conversation_history_non_serializable_dict(self):
        """Test to_conversation_history with non-serializable dict."""

        class NonSerializable:
            pass

        tcr = ToolCallResult(
            tool_name="t", success=True, result={"obj": NonSerializable()}
        )
        # Should fall back to str()
        result = tcr.to_conversation_history()
        assert "obj" in result


class TestValidationResult:
    """Test ValidationResult model."""

    def test_validation_result_success(self):
        """Test success factory method."""
        vr = ValidationResult.success()
        assert vr.is_valid is True
        assert vr.error_message is None
        assert vr.warnings == []

    def test_validation_result_failure(self):
        """Test failure factory method."""
        vr = ValidationResult.failure("Something went wrong")
        assert vr.is_valid is False
        assert vr.error_message == "Something went wrong"

    def test_validation_result_from_tuple(self):
        """Test from_tuple factory method."""
        vr1 = ValidationResult.from_tuple((True, None))
        assert vr1.is_valid is True

        vr2 = ValidationResult.from_tuple((False, "Error"))
        assert vr2.is_valid is False
        assert vr2.error_message == "Error"

    def test_validation_result_display_result(self):
        """Test display_result property."""
        vr_success = ValidationResult.success()
        assert vr_success.display_result == "Validation successful"

        vr_failure = ValidationResult.failure("Invalid schema")
        assert "Invalid schema" in vr_failure.display_result

        vr_unknown = ValidationResult(is_valid=False)
        assert "Unknown error" in vr_unknown.display_result

    def test_validation_result_has_error(self):
        """Test has_error property."""
        vr_success = ValidationResult.success()
        assert vr_success.has_error is False

        vr_failure = ValidationResult.failure("Error")
        assert vr_failure.has_error is True


class TestTransportServerConfig:
    """Test TransportServerConfig model."""

    def test_transport_server_config_defaults(self):
        """Test default values."""
        tsc = TransportServerConfig(name="server1", url="https://example.com")
        assert tsc.name == "server1"
        assert tsc.url == "https://example.com"
        assert tsc.headers == {}
        assert tsc.api_key is None
        assert tsc.timeout is None
        assert tsc.max_retries is None

    def test_transport_server_config_full(self):
        """Test with all values."""
        tsc = TransportServerConfig(
            name="server1",
            url="https://example.com",
            headers={"Authorization": "Bearer token"},
            api_key="api_key_123",
            timeout=30.0,
            max_retries=3,
        )
        assert tsc.headers["Authorization"] == "Bearer token"
        assert tsc.api_key == "api_key_123"
        assert tsc.timeout == 30.0
        assert tsc.max_retries == 3

    def test_transport_server_config_to_stream_manager(self):
        """Test to_stream_manager_config method."""
        tsc = TransportServerConfig(
            name="server1", url="https://example.com", timeout=30.0
        )
        config = tsc.to_stream_manager_config()
        assert config["name"] == "server1"
        assert config["url"] == "https://example.com"
        assert config["timeout"] == 30.0
        # None values should be excluded
        assert "api_key" not in config


class TestConversationMessage:
    """Test ConversationMessage model."""

    def test_conversation_message_user(self):
        """Test user_message factory."""
        msg = ConversationMessage.user_message("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_conversation_message_assistant(self):
        """Test assistant_message factory."""
        msg = ConversationMessage.assistant_message("Response")
        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.tool_calls is None

    def test_conversation_message_assistant_with_tools(self):
        """Test assistant_message with tool calls."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test", "arguments": "{}"},
            }
        ]
        msg = ConversationMessage.assistant_message(content=None, tool_calls=tool_calls)
        assert msg.role == "assistant"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_1"

    def test_conversation_message_system(self):
        """Test system_message factory."""
        msg = ConversationMessage.system_message("System prompt")
        assert msg.role == "system"
        assert msg.content == "System prompt"

    def test_conversation_message_tool(self):
        """Test tool_message factory."""
        msg = ConversationMessage.tool_message("Result", "call_1", name="my_tool")
        assert msg.role == "tool"
        assert msg.content == "Result"
        assert msg.tool_call_id == "call_1"
        assert msg.name == "my_tool"

    def test_conversation_message_to_dict(self):
        """Test to_dict method."""
        msg = ConversationMessage.user_message("Test")
        result = msg.to_dict()
        assert result["role"] == "user"
        assert result["content"] == "Test"
        # None values excluded
        assert "tool_calls" not in result

    def test_conversation_message_from_dict(self):
        """Test from_dict factory."""
        data = {"role": "user", "content": "Hello"}
        msg = ConversationMessage.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestToolCallMessage:
    """Test ToolCallMessage model."""

    def test_tool_call_message(self):
        """Test basic creation."""
        tcm = ToolCallMessage(
            id="call_1", type="function", function={"name": "test", "arguments": "{}"}
        )
        assert tcm.id == "call_1"
        assert tcm.type == "function"
        assert tcm.function["name"] == "test"


class TestFunctionDefinition:
    """Test FunctionDefinition model."""

    def test_function_definition_defaults(self):
        """Test default values."""
        fd = FunctionDefinition(name="test", description="A test function")
        assert fd.name == "test"
        assert fd.description == "A test function"
        assert fd.parameters == {"type": "object", "properties": {}}

    def test_function_definition_with_params(self):
        """Test with parameters."""
        fd = FunctionDefinition(
            name="add",
            description="Add numbers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        )
        assert fd.parameters["required"] == ["a", "b"]


class TestLLMToolDefinition:
    """Test LLMToolDefinition model."""

    def test_llm_tool_definition_defaults(self):
        """Test default type."""
        ltd = LLMToolDefinition(
            function=FunctionDefinition(name="test", description="Test")
        )
        assert ltd.type == ToolType.FUNCTION

    def test_llm_tool_definition_to_dict(self):
        """Test to_dict method."""
        ltd = LLMToolDefinition(
            function=FunctionDefinition(name="test", description="Test")
        )
        result = ltd.to_dict()
        assert result["type"] == "function"
        assert result["function"]["name"] == "test"


class TestToolInputSchema:
    """Test ToolInputSchema model."""

    def test_tool_input_schema_defaults(self):
        """Test default values."""
        tis = ToolInputSchema()
        assert tis.type == "object"
        assert tis.properties == {}
        assert tis.required == []
        assert tis.additionalProperties is False

    def test_tool_input_schema_with_values(self):
        """Test with values."""
        tis = ToolInputSchema(
            type="object", properties={"arg": {"type": "string"}}, required=["arg"]
        )
        assert tis.properties["arg"]["type"] == "string"
        assert tis.required == ["arg"]


class TestToolDefinitionInput:
    """Test ToolDefinitionInput model."""

    def test_tool_definition_input_defaults(self):
        """Test default values."""
        tdi = ToolDefinitionInput(name="test")
        assert tdi.name == "test"
        assert tdi.namespace == "default"
        assert tdi.description is None
        assert tdi.inputSchema == {}
        assert tdi.is_async is False
        assert tdi.tags == []

    def test_tool_definition_input_full(self):
        """Test with all values."""
        tdi = ToolDefinitionInput(
            name="test",
            namespace="server",
            description="Test tool",
            inputSchema={"type": "object"},
            is_async=True,
            tags=["tag1", "tag2"],
        )
        assert tdi.namespace == "server"
        assert tdi.description == "Test tool"
        assert tdi.is_async is True
        assert tdi.tags == ["tag1", "tag2"]


class TestToolInfoRequiredParametersEdgeCases:
    """Additional edge case tests for ToolInfo.required_parameters."""

    def test_required_parameters_non_list(self):
        """Test required_parameters with non-list value."""
        ti = ToolInfo(
            name="test", namespace="ns", parameters={"required": "not_a_list"}
        )
        assert ti.required_parameters == []


class TestToolCallResultOtherTypes:
    """Test ToolCallResult with other result types."""

    def test_display_result_with_list(self):
        """Test display_result with list result."""
        tcr = ToolCallResult(tool_name="t", success=True, result=[1, 2, 3])
        assert tcr.display_result == "[1, 2, 3]"

    def test_display_result_with_int(self):
        """Test display_result with int result."""
        tcr = ToolCallResult(tool_name="t", success=True, result=42)
        assert tcr.display_result == "42"

    def test_to_conversation_history_with_list(self):
        """Test to_conversation_history with list result."""
        tcr = ToolCallResult(tool_name="t", success=True, result=[1, 2, 3])
        assert tcr.to_conversation_history() == "[1, 2, 3]"

    def test_to_conversation_history_with_int(self):
        """Test to_conversation_history with int result."""
        tcr = ToolCallResult(tool_name="t", success=True, result=42)
        assert tcr.to_conversation_history() == "42"
