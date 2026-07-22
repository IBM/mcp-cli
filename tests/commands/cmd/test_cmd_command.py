# tests/commands/cmd/test_cmd_command.py
"""Tests for the CmdCommand (commands/cmd/cmd.py)."""

from __future__ import annotations

import io
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.commands.cmd.cmd import CmdCommand, _parse_tool_call
from mcp_cli.commands.base import CommandMode

# Patch targets (canonical import locations)
_GET_CONTEXT = "mcp_cli.context.get_context"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_manager(
    execute_result=None, tools_for_llm=None
) -> MagicMock:
    tm = MagicMock()
    if execute_result is None:
        result = MagicMock()
        result.success = True
        result.error = None
        result.result = {"status": "ok"}
        execute_result = result
    tm.execute_tool = AsyncMock(return_value=execute_result)
    tm.get_tools_for_llm = AsyncMock(return_value=tools_for_llm or [])
    return tm


def _make_context(provider="openai", model="gpt-4") -> MagicMock:
    ctx = MagicMock()
    ctx.provider = provider
    ctx.model = model
    ctx.model_manager = MagicMock()
    ctx.tool_manager = _make_tool_manager()
    return ctx


@pytest.fixture
def cmd():
    return CmdCommand()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestCmdCommandProperties:
    def test_name(self, cmd):
        assert cmd.name == "cmd"

    def test_description(self, cmd):
        assert "automation" in cmd.description.lower()

    def test_modes(self, cmd):
        assert cmd.modes == CommandMode.CLI

    def test_requires_context(self, cmd):
        assert cmd.requires_context is True

    def test_aliases_empty(self, cmd):
        assert cmd.aliases == []

    def test_parameters_defined(self, cmd):
        names = {p.name for p in cmd.parameters}
        assert "tool" in names
        assert "prompt" in names
        assert "input_file" in names
        assert "raw" in names


# ---------------------------------------------------------------------------
# Execute guards
# ---------------------------------------------------------------------------


class TestCmdExecuteGuards:
    @pytest.mark.asyncio
    async def test_no_args_returns_error(self, cmd):
        """Branch 3: no --tool, no --prompt, no --input → error."""
        result = await cmd.execute()
        assert result.success is False
        assert result.error is not None
        assert "No operation specified" in result.error

    @pytest.mark.asyncio
    async def test_tool_without_manager_returns_error(self, cmd):
        result = await cmd.execute(tool="some_tool", tool_manager=None)
        assert result.success is False
        assert "Tool manager not available" in result.error


# ---------------------------------------------------------------------------
# Tool direct execution
# ---------------------------------------------------------------------------


class TestCmdToolDirect:
    @pytest.mark.asyncio
    async def test_successful_tool_execution(self, cmd):
        tm = _make_tool_manager()
        with patch("builtins.print") as mock_print:
            result = await cmd.execute(tool="echo", tool_manager=tm)
        assert result.success is True
        assert result.data == {"status": "ok"}
        tm.execute_tool.assert_awaited_once_with("echo", {})
        mock_print.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_with_json_args(self, cmd):
        tm = _make_tool_manager()
        args_json = '{"query": "SELECT 1"}'
        with patch("builtins.print"):
            result = await cmd.execute(
                tool="run_query", tool_args=args_json, tool_manager=tm
            )
        assert result.success is True
        tm.execute_tool.assert_awaited_once_with(
            "run_query", {"query": "SELECT 1"}
        )

    @pytest.mark.asyncio
    async def test_tool_with_invalid_json_args(self, cmd):
        tm = _make_tool_manager()
        result = await cmd.execute(
            tool="echo", tool_args="not json", tool_manager=tm
        )
        assert result.success is False
        assert "Invalid JSON" in result.error

    @pytest.mark.asyncio
    async def test_tool_execution_failure(self, cmd):
        fail_result = MagicMock()
        fail_result.success = False
        fail_result.error = "tool crashed"
        fail_result.result = None
        tm = _make_tool_manager(execute_result=fail_result)
        with patch("builtins.print"):
            result = await cmd.execute(tool="bad_tool", tool_manager=tm)
        assert result.success is False
        assert "tool crashed" in result.error

    @pytest.mark.asyncio
    async def test_tool_output_to_file(self, cmd, tmp_path):
        tm = _make_tool_manager()
        out_file = str(tmp_path / "out.json")
        result = await cmd.execute(
            tool="echo", tool_manager=tm, output_file=out_file
        )
        assert result.success is True
        content = (tmp_path / "out.json").read_text()
        assert "ok" in content

    @pytest.mark.asyncio
    async def test_tool_raw_output(self, cmd):
        tm = _make_tool_manager()
        with patch("builtins.print") as mock_print:
            result = await cmd.execute(
                tool="echo", tool_manager=tm, raw=True
            )
        assert result.success is True
        # Raw mode: json.dumps without indent
        printed = mock_print.call_args[0][0]
        assert "\n" not in printed  # compact JSON


# ---------------------------------------------------------------------------
# Prompt mode
# ---------------------------------------------------------------------------


class TestCmdPromptMode:
    @pytest.mark.asyncio
    async def test_prompt_no_context_returns_error(self, cmd):
        with patch(_GET_CONTEXT, return_value=None):
            result = await cmd.execute(prompt="hello", model_manager=None)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_prompt_basic(self, cmd):
        ctx = _make_context()
        mock_client = MagicMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "Hello!", "tool_calls": []}
        )
        ctx.model_manager.get_client.return_value = mock_client

        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("builtins.print") as mock_print:
                result = await cmd.execute(
                    prompt="hi", model_manager=ctx.model_manager
                )
        assert result.success is True
        assert result.data == "Hello!"
        mock_print.assert_called_once_with("Hello!")

    @pytest.mark.asyncio
    async def test_stdin_input(self, cmd):
        """input_file='-' reads from sys.stdin and passes content to the LLM."""
        ctx = _make_context()
        mock_client = MagicMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "got it", "tool_calls": []}
        )
        ctx.model_manager.get_client.return_value = mock_client

        with patch(_GET_CONTEXT, return_value=ctx), \
             patch("mcp_cli.commands.cmd.cmd.sys.stdin", new=io.StringIO("stdin data\n")), \
             patch("builtins.print"):
            result = await cmd.execute(
                input_file="-", model_manager=ctx.model_manager
            )

        assert result.success is True
        call_kwargs = mock_client.create_completion.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert "stdin data" in messages[-1]["content"]

    @pytest.mark.asyncio
    async def test_input_file_not_found(self, cmd):
        ctx = _make_context()
        with patch(_GET_CONTEXT, return_value=ctx):
            result = await cmd.execute(
                input_file="/nonexistent/path.txt",
                model_manager=ctx.model_manager,
            )
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_input_file_read(self, cmd, tmp_path):
        input_file = tmp_path / "input.txt"
        input_file.write_text("test data")

        ctx = _make_context()
        mock_client = MagicMock()
        mock_client.create_completion = AsyncMock(
            return_value={"response": "analyzed", "tool_calls": []}
        )
        ctx.model_manager.get_client.return_value = mock_client

        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("builtins.print"):
                result = await cmd.execute(
                    input_file=str(input_file),
                    model_manager=ctx.model_manager,
                )
        assert result.success is True
        # Check that input text was used in the prompt
        call_kwargs = mock_client.create_completion.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert "test data" in messages[-1]["content"]


# ---------------------------------------------------------------------------
# _execute_tool_call_batch — JSON parse error handling (fix #2)
# ---------------------------------------------------------------------------


class TestCmdToolCallBatch:
    @pytest.mark.asyncio
    async def test_invalid_json_in_tool_args_adds_error_message(self, cmd):
        """When LLM returns invalid JSON arguments, error is appended to messages."""
        tm = _make_tool_manager()
        messages: list[dict] = []
        tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "echo", "arguments": "not valid json{"},
            }
        ]

        await cmd._execute_tool_call_batch(tm, tool_calls, messages, raw=True)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert "Invalid JSON" in messages[0]["content"]
        # Tool should NOT have been called
        tm.execute_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_valid_tool_call_succeeds(self, cmd):
        tm = _make_tool_manager()
        messages: list[dict] = []
        tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "echo", "arguments": '{"msg": "hi"}'},
            }
        ]

        await cmd._execute_tool_call_batch(tm, tool_calls, messages, raw=True)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        tm.execute_tool.assert_awaited_once_with("echo", {"msg": "hi"})


# ---------------------------------------------------------------------------
# _parse_tool_call (fix #3)
# ---------------------------------------------------------------------------


class TestParseToolCall:
    def test_dict_format(self):
        tc = {
            "id": "call_1",
            "function": {"name": "echo", "arguments": '{"x": 1}'},
        }
        name, args, call_id = _parse_tool_call(tc)
        assert name == "echo"
        assert args == '{"x": 1}'
        assert call_id == "call_1"

    def test_dict_missing_fields(self):
        name, args, call_id = _parse_tool_call({})
        assert name == ""
        assert args == "{}"
        assert call_id == ""

    def test_object_format(self):
        tc = MagicMock()
        tc.function.name = "search"
        tc.function.arguments = '{"q": "test"}'
        tc.id = "call_2"
        name, args, call_id = _parse_tool_call(tc)
        assert name == "search"
        assert args == '{"q": "test"}'
        assert call_id == "call_2"

    def test_invalid_object_returns_defaults(self):
        """Objects without .function attribute return safe defaults."""
        tc = object()  # no .function attr
        name, args, call_id = _parse_tool_call(tc)
        assert name == ""
        assert args == "{}"
        assert call_id == ""


# ---------------------------------------------------------------------------
# Max turns — final response (fix #4)
# ---------------------------------------------------------------------------


class TestCmdMaxTurns:
    @pytest.mark.asyncio
    async def test_max_turns_gets_final_response(self, cmd):
        """After hitting max_turns, a final create_completion is called without tools."""
        ctx = _make_context()
        tm = _make_tool_manager()

        # With max_turns=1: initial batch runs, while loop skips (1 < 1 = False),
        # then final completion is called.
        # Call sequence: initial from _execute_prompt_mode → final from _handle_tool_calls
        mock_client = MagicMock()
        mock_client.create_completion = AsyncMock(
            side_effect=[
                # Initial call from _execute_prompt_mode → returns tool_calls
                {
                    "response": "Let me check...",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "t", "arguments": "{}"}},
                    ],
                },
                # Final completion after max_turns (no tools param)
                {"response": "Final synthesized answer"},
            ]
        )
        ctx.model_manager.get_client.return_value = mock_client

        with patch(_GET_CONTEXT, return_value=ctx):
            with patch("builtins.print"):
                result = await cmd.execute(
                    prompt="test",
                    tool_manager=tm,
                    model_manager=ctx.model_manager,
                    max_turns=1,
                )

        assert result.success is True
        assert result.data == "Final synthesized answer"
