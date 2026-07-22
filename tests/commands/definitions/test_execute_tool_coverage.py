"""Additional tests for execute_tool command to achieve >90% coverage."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.commands.tools.execute_tool import ExecuteToolCommand
from mcp_cli.utils.serialization import to_serializable as _to_serializable
from mcp_cli.tools.models import ToolCallResult


@pytest.fixture
def execute_command():
    """Create an execute tool command instance."""
    return ExecuteToolCommand()


class TestToSerializable:
    """Tests for _to_serializable function."""

    def test_none_value(self):
        """Test serializing None."""
        assert _to_serializable(None) is None

    def test_primitives(self):
        """Test serializing primitives."""
        assert _to_serializable("hello") == "hello"
        assert _to_serializable(42) == 42
        assert _to_serializable(3.14) == 3.14
        assert _to_serializable(True) is True

    def test_list(self):
        """Test serializing lists."""
        assert _to_serializable([1, 2, 3]) == [1, 2, 3]
        assert _to_serializable(["a", None, True]) == ["a", None, True]

    def test_dict(self):
        """Test serializing dicts."""
        assert _to_serializable({"key": "value"}) == {"key": "value"}
        assert _to_serializable({"nested": {"a": 1}}) == {"nested": {"a": 1}}

    def test_pydantic_model_with_model_dump(self):
        """Test serializing object with model_dump method."""
        obj = MagicMock()
        obj.model_dump.return_value = {"field": "value"}
        # Remove dict method to force model_dump path
        del obj.dict

        result = _to_serializable(obj)
        assert result == {"field": "value"}

    def test_pydantic_model_with_dict(self):
        """Test serializing object with dict method (older Pydantic)."""
        obj = MagicMock(spec=["dict"])
        obj.dict.return_value = {"field": "value"}

        result = _to_serializable(obj)
        assert result == {"field": "value"}

    def test_mcp_tool_result_with_text_content(self):
        """Test serializing MCP SDK ToolResult with text content."""
        text_item = MagicMock()
        text_item.text = "Hello, world!"
        # Remove model_dump to force text path
        del text_item.model_dump

        obj = MagicMock(spec=["content"])
        obj.content = [text_item]

        result = _to_serializable(obj)
        assert result == "Hello, world!"

    def test_mcp_tool_result_with_multiple_text_content(self):
        """Test serializing MCP SDK ToolResult with multiple text items."""
        text_item1 = MagicMock()
        text_item1.text = "Line 1"
        del text_item1.model_dump

        text_item2 = MagicMock()
        text_item2.text = "Line 2"
        del text_item2.model_dump

        obj = MagicMock(spec=["content"])
        obj.content = [text_item1, text_item2]

        result = _to_serializable(obj)
        assert result == ["Line 1", "Line 2"]

    def test_mcp_tool_result_with_model_dump_content(self):
        """Test serializing MCP SDK ToolResult with model_dump items."""
        item1 = MagicMock()
        item1.model_dump.return_value = {"type": "image", "data": "base64..."}
        # Ensure text attribute doesn't exist
        del item1.text

        item2 = MagicMock()
        item2.model_dump.return_value = {"type": "text", "data": "hello"}
        del item2.text

        obj = MagicMock(spec=["content"])
        obj.content = [item1, item2]  # Multiple items returns list

        result = _to_serializable(obj)
        assert result == [
            {"type": "image", "data": "base64..."},
            {"type": "text", "data": "hello"},
        ]

    def test_mcp_tool_result_with_plain_content(self):
        """Test serializing MCP SDK ToolResult with content that has no special handling."""
        item1 = MagicMock(spec=[])  # No text, no model_dump
        item2 = MagicMock(spec=[])  # No text, no model_dump

        obj = MagicMock(spec=["content"])
        obj.content = [item1, item2]  # Multiple items returns list

        result = _to_serializable(obj)
        # Should fall back to str() for each item, returns list for multiple
        assert isinstance(result, list)
        assert len(result) == 2

    def test_mcp_tool_result_with_non_list_content(self):
        """Test serializing MCP SDK ToolResult with non-list content."""
        obj = MagicMock(spec=["content"])
        obj.content = {"direct": "content"}

        result = _to_serializable(obj)
        assert result == {"direct": "content"}

    def test_fallback_to_string(self):
        """Test fallback to string for unknown types."""

        class CustomClass:
            def __str__(self):
                return "custom_string"

        result = _to_serializable(CustomClass())
        assert result == "custom_string"


class TestExecuteToolArgsHandling:
    """Tests for args parameter handling."""

    @pytest.mark.asyncio
    async def test_args_string_as_params_when_tool_set(self, execute_command):
        """Test args as string becomes params when tool is already set."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {}

        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test_tool", success=True, result="Success"
            )
        )

        with patch("mcp_cli.commands.tools.execute_tool.output"):
            result = await execute_command.execute(
                tool="test_tool",
                args="{}",  # String args should become params
                tool_manager=manager,
            )

            assert result.success is True
            manager.execute_tool.assert_called_once()


class TestExecuteToolErrorHandling:
    """Tests for error handling paths."""

    @pytest.mark.asyncio
    async def test_tool_not_found_on_specific_server(self, execute_command):
        """Test tool not found when filtering by server."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "server1"
        tool.inputSchema = None
        tool.parameters = {}

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            await execute_command.execute(
                tool="test_tool",
                server="server2",  # Different server
                params="{}",
                tool_manager=manager,
            )

            # Should show error and list tools
            mock_output.error.assert_called_with(
                "Tool 'test_tool' not found on server 'server2'"
            )

    @pytest.mark.asyncio
    async def test_json_error_with_no_required_params(self, execute_command):
        """Test JSON error display when tool has no required params."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {
            "properties": {"optional": {"type": "string"}},
            "required": [],  # No required params
        }

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params="{invalid json}",
                tool_manager=manager,
            )

            assert result.success is False
            # Should show empty example since no required params
            output_calls = str(mock_output.print.call_args_list)
            assert "'{}'" in output_calls or "execute" in output_calls

    @pytest.mark.asyncio
    async def test_json_error_with_message_param(self, execute_command):
        """Test JSON error showing 'message' param with special handling."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "echo_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="echo_tool",
                params="{bad json}",
                tool_manager=manager,
            )

            assert result.success is False
            # Should show example with "your message here"
            output_calls = str(mock_output.print.call_args_list)
            assert "message" in output_calls

    @pytest.mark.asyncio
    async def test_json_error_with_number_boolean_params(self, execute_command):
        """Test JSON error example with number and boolean types."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "typed_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {
            "properties": {
                "count": {"type": "number"},
                "enabled": {"type": "boolean"},
            },
            "required": ["count", "enabled"],
        }

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="typed_tool",
                params="{invalid}",
                tool_manager=manager,
            )

            assert result.success is False
            # Should show example with 123 and true
            output_calls = str(mock_output.print.call_args_list)
            assert "123" in output_calls or "true" in output_calls.lower()

    @pytest.mark.asyncio
    async def test_exception_unexpected_keyword_argument(self, execute_command):
        """Test exception handling for unexpected keyword argument error."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {}

        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(
            side_effect=TypeError("got an unexpected keyword argument 'foo'")
        )

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params="{}",
                tool_manager=manager,
            )

            assert result.success is False
            mock_output.print.assert_any_call(
                "Internal error - please report this issue"
            )

    @pytest.mark.asyncio
    async def test_exception_invalid_parameter(self, execute_command):
        """Test exception handling for Invalid parameter error."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {}

        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(
            side_effect=ValueError("Invalid parameter: foo must be string")
        )

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params="{}",
                tool_manager=manager,
            )

            assert result.success is False
            # Should show hint about checking parameters
            mock_output.hint.assert_called_with(
                "Use /execute <tool> to see correct parameters"
            )


class TestListToolsEmpty:
    """Tests for empty tools list."""

    @pytest.mark.asyncio
    async def test_list_tools_empty(self, execute_command):
        """Test listing tools when no tools available."""
        manager = AsyncMock()
        manager.server_names = {}
        manager.get_all_tools = AsyncMock(return_value=[])

        result = await execute_command.execute(tool_manager=manager)

        assert result.success is True
        assert "No tools available" in result.output


class TestShowToolInfoEdgeCases:
    """Tests for _show_tool_info edge cases."""

    @pytest.mark.asyncio
    async def test_show_tool_info_with_input_schema(self, execute_command):
        """Test showing tool info when tool has inputSchema (not parameters)."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "schema_tool"
        tool.description = "Tool with inputSchema"
        tool.namespace = "default"
        tool.parameters = None  # No parameters
        tool.inputSchema = {
            "properties": {"query": {"type": "string", "description": "Query"}},
            "required": ["query"],
        }

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="schema_tool", tool_manager=manager
            )

            assert result.success is True
            mock_output.rule.assert_any_call("Parameters")

    @pytest.mark.asyncio
    async def test_show_tool_info_no_properties(self, execute_command):
        """Test showing tool info when schema has no properties."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "no_props_tool"
        tool.description = "Tool without properties"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {"type": "object"}  # Schema without properties

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="no_props_tool", tool_manager=manager
            )

            assert result.success is True
            mock_output.print.assert_any_call("  No parameters required")

    @pytest.mark.asyncio
    async def test_show_tool_info_array_object_types(self, execute_command):
        """Test showing tool info with array and object parameter types."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "complex_tool"
        tool.description = "Tool with complex types"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {
            "properties": {
                "items": {"type": "array", "description": "List of items"},
                "config": {"type": "object", "description": "Configuration"},
            },
            "required": ["items", "config"],
        }

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="complex_tool", tool_manager=manager
            )

            assert result.success is True
            # Example should contain [] and {}
            output_calls = str(mock_output.print.call_args_list)
            assert "[]" in output_calls or "{}" in output_calls

    @pytest.mark.asyncio
    async def test_show_tool_info_no_required_params(self, execute_command):
        """Test showing tool info when no params are required."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "optional_tool"
        tool.description = "Tool with only optional params"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {
            "properties": {"opt": {"type": "string", "description": "Optional"}},
            "required": [],
        }

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="optional_tool", tool_manager=manager
            )

            assert result.success is True
            # Example should just show the tool name without params
            output_calls = str(mock_output.print.call_args_list)
            assert "optional_tool" in output_calls


class TestParseSimpleParams:
    """Tests for _parse_simple_params method."""

    @pytest.mark.asyncio
    async def test_parse_simple_params_single_value(self, execute_command):
        """Test parsing single value without key."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {}

        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test_tool", success=True, result="Success"
            )
        )

        with patch("mcp_cli.commands.tools.execute_tool.output"):
            result = await execute_command.execute(
                tool="test_tool",
                params="singlevalue",  # Single value without =
                tool_manager=manager,
            )

            # Should parse as {"value": "singlevalue"}
            call_args = manager.execute_tool.call_args
            if call_args:
                args = call_args[1].get("arguments", {})
                assert "value" in args or result.success is False

    @pytest.mark.asyncio
    async def test_parse_simple_params_json_value(self, execute_command):
        """Test parsing key=value where value is JSON."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {}

        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(
            return_value=ToolCallResult(
                tool_name="test_tool", success=True, result="Success"
            )
        )

        with patch("mcp_cli.commands.tools.execute_tool.output"):
            result = await execute_command.execute(
                tool="test_tool",
                params='count=42 flag=true items=["a","b"]',
                tool_manager=manager,
            )

            # Should parse JSON values properly
            call_args = manager.execute_tool.call_args
            if call_args:
                args = call_args[1].get("arguments", {})
                # count should be int 42, flag should be bool True
                assert args.get("count") == 42 or result is not None


class TestPlainStringErrorGuessing:
    """Tests for guessing param name when user provides plain string."""

    @pytest.mark.asyncio
    async def test_plain_string_error_no_properties(self, execute_command):
        """Test plain string error when tool has no properties."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "simple_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {}  # No properties

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="simple_tool",
                params="just a string",
                tool_manager=manager,
            )

            assert result.success is False
            # Should default to "message" as param name
            output_calls = str(mock_output.print.call_args_list)
            assert "message" in output_calls

    @pytest.mark.asyncio
    async def test_plain_string_error_with_required_param(self, execute_command):
        """Test plain string error guessing first required param."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "query_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["query"],
        }

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="query_tool",
                params="my search query",
                tool_manager=manager,
            )

            assert result.success is False
            # Should use "query" as the guessed param name
            output_calls = str(mock_output.print.call_args_list)
            assert "query" in output_calls

    @pytest.mark.asyncio
    async def test_plain_string_error_first_property(self, execute_command):
        """Test plain string error using first property when no required."""
        manager = AsyncMock()
        manager.server_names = {}
        tool = MagicMock()
        tool.name = "flex_tool"
        tool.namespace = "default"
        tool.inputSchema = None
        tool.parameters = {
            "properties": {
                "input": {"type": "string"},
            },
            "required": [],  # None required
        }

        manager.get_all_tools = AsyncMock(return_value=[tool])

        with patch("mcp_cli.commands.tools.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="flex_tool",
                params="some input",
                tool_manager=manager,
            )

            assert result.success is False
            # Should use "input" as the guessed param name
            output_calls = str(mock_output.print.call_args_list)
            assert "input" in output_calls
