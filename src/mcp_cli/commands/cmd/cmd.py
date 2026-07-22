# src/mcp_cli/commands/cmd/cmd.py
"""Command mode for Unix-friendly automation and scripting."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from chuk_term.ui import output
from mcp_cli.commands.base import (
    CommandMode,
    CommandParameter,
    CommandResult,
    UnifiedCommand,
)
from mcp_cli.utils.serialization import to_serializable, unwrap_tool_result

logger = logging.getLogger(__name__)


class CmdCommand(UnifiedCommand):
    """Command mode for Unix-friendly automation and scripting."""

    @property
    def name(self) -> str:
        return "cmd"

    @property
    def description(self) -> str:
        return "Command mode for Unix-friendly automation and scripting"

    @property
    def modes(self) -> CommandMode:
        return CommandMode.CLI

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="input_file", type=str, required=False,
                help="Input file (use - for stdin)",
            ),
            CommandParameter(
                name="output_file", type=str, required=False,
                help="Output file (use - for stdout)",
            ),
            CommandParameter(
                name="prompt", type=str, required=False,
                help="Prompt text",
            ),
            CommandParameter(
                name="tool", type=str, required=False,
                help="Tool name to execute",
            ),
            CommandParameter(
                name="tool_args", type=str, required=False,
                help="Tool arguments as JSON",
            ),
            CommandParameter(
                name="system_prompt", type=str, required=False,
                help="Custom system prompt",
            ),
            CommandParameter(
                name="raw", type=bool, default=False, is_flag=True,
                help="Raw output without formatting",
            ),
            CommandParameter(
                name="single_turn", type=bool, default=False, is_flag=True,
                help="Disable multi-turn conversation",
            ),
            CommandParameter(
                name="max_turns", type=int, default=100,
                help="Maximum conversation turns",
            ),
        ]

    @property
    def help_text(self) -> str:
        return """
Command mode for Unix-friendly automation and scripting.

Usage:
  mcp-cli cmd --tool <name>                      Execute a tool directly
  mcp-cli cmd --tool <name> --tool-args '{...}'   Execute tool with arguments
  mcp-cli cmd --prompt "Summarize this"           Use LLM with a prompt
  mcp-cli cmd --input data.txt --prompt "..."     Combine file input with prompt
  echo "text" | mcp-cli cmd --input -             Read from stdin
"""

    @property
    def requires_context(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> CommandResult:
        """Execute the cmd command."""
        tool_manager = kwargs.get("tool_manager")

        tool = kwargs.get("tool")
        tool_args = kwargs.get("tool_args")
        input_file = kwargs.get("input_file")
        output_file = kwargs.get("output_file")
        prompt = kwargs.get("prompt")
        system_prompt = kwargs.get("system_prompt")
        raw = kwargs.get("raw", False)
        single_turn = kwargs.get("single_turn", False)
        max_turns = kwargs.get("max_turns", 100)

        # Branch 1: Tool direct execution
        if tool:
            return await self._execute_tool_direct(
                tool_manager=tool_manager,
                tool_name=tool,
                tool_args_json=tool_args,
                output_file=output_file,
                raw=raw,
            )

        # Branch 2: Prompt / LLM mode
        if prompt or input_file:
            return await self._execute_prompt_mode(
                tool_manager=tool_manager,
                model_manager=kwargs.get("model_manager"),
                input_file=input_file,
                output_file=output_file,
                prompt=prompt,
                system_prompt=system_prompt,
                raw=raw,
                single_turn=single_turn,
                max_turns=max_turns,
            )

        # Branch 3: No operation specified
        return CommandResult(
            success=False,
            error="No operation specified. Use --tool or --prompt/--input",
        )

    # ── tool direct execution ────────────────────────────────────────

    async def _execute_tool_direct(
        self,
        tool_manager: Any | None,
        tool_name: str,
        tool_args_json: str | None,
        output_file: str | None,
        raw: bool,
    ) -> CommandResult:
        """Execute a tool directly without LLM interaction."""
        if not tool_manager:
            return CommandResult(
                success=False,
                error="Tool manager not available. Are servers connected?",
            )

        # Parse tool arguments
        tool_args: dict[str, Any] = {}
        if tool_args_json:
            try:
                tool_args = json.loads(tool_args_json)
            except json.JSONDecodeError as e:
                return CommandResult(
                    success=False,
                    error=f"Invalid JSON in tool arguments: {e}",
                )

        try:
            if not raw:
                output.info(f"Executing tool: {tool_name}")

            tool_call_result = await tool_manager.execute_tool(tool_name, tool_args)

            if not tool_call_result.success or tool_call_result.error:
                return CommandResult(
                    success=False,
                    error=f"Tool execution failed: {tool_call_result.error}",
                )

            result_data = tool_call_result.result

            # Unwrap middleware ToolExecutionResult if present
            result_data = unwrap_tool_result(result_data)

            # Convert to JSON-serializable form
            result_data = to_serializable(result_data)

            # Format result
            result_str = (
                json.dumps(result_data, indent=None if raw else 2)
                if not isinstance(result_data, str)
                else result_data
            )

            # Write output
            if output_file and output_file != "-":
                Path(output_file).write_text(result_str)
                if not raw:
                    output.success(f"Output written to: {output_file}")
                return CommandResult(success=True, data=result_data)
            else:
                print(result_str)
                return CommandResult(success=True, data=result_data)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Tool execution failed: {e}",
            )

    # ── prompt / LLM mode ────────────────────────────────────────────

    async def _execute_prompt_mode(
        self,
        tool_manager: Any | None,
        model_manager: Any | None,
        input_file: str | None,
        output_file: str | None,
        prompt: str | None,
        system_prompt: str | None,
        raw: bool,
        single_turn: bool,
        max_turns: int,
    ) -> CommandResult:
        """Execute prompt mode with LLM interaction."""
        from mcp_cli.context import get_context

        context = get_context()

        # Read input
        input_text = ""
        if input_file:
            if input_file == "-":
                input_text = sys.stdin.read()
            else:
                try:
                    input_text = Path(input_file).read_text()
                except FileNotFoundError:
                    return CommandResult(
                        success=False,
                        error=f"Input file not found: {input_file}",
                    )

        # Build full prompt
        if prompt and input_text:
            full_prompt = f"{prompt}\n\nInput:\n{input_text}"
        elif prompt:
            full_prompt = prompt
        elif input_text:
            full_prompt = input_text
        else:
            return CommandResult(success=False, error="No prompt or input provided")

        # Get LLM client
        effective_mm = model_manager or (context.model_manager if context else None)
        if not effective_mm:
            return CommandResult(
                success=False, error="Model manager not available."
            )

        if not context:
            return CommandResult(
                success=False, error="Context not initialized."
            )

        try:
            client = effective_mm.get_client(
                provider=context.provider,
                model=context.model,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to initialize LLM client: {e}",
            )

        # Build messages
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": full_prompt})

        # Get tools for LLM if available
        tools = None
        if tool_manager and not single_turn:
            tools = await tool_manager.get_tools_for_llm()

        try:
            response = await client.create_completion(
                model=context.model,
                messages=messages,
                tools=tools,
                max_tokens=4096,
            )

            result_text = response.get("response", "")
            tool_calls = response.get("tool_calls", [])

            # Handle tool calls if present
            if tool_calls and not single_turn and tool_manager:
                result_text = await self._handle_tool_calls(
                    client=client,
                    model=context.model,
                    tool_manager=tool_manager,
                    messages=messages,
                    tool_calls=tool_calls,
                    response_text=result_text,
                    max_turns=max_turns,
                    raw=raw,
                )

            # Write output
            if output_file and output_file != "-":
                Path(output_file).write_text(result_text)
                if not raw:
                    output.success(f"Output written to: {output_file}")
                return CommandResult(success=True, data=result_text)
            else:
                print(result_text)
                return CommandResult(success=True, data=result_text)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"LLM execution failed: {e}",
            )

    # ── multi-turn tool call loop ────────────────────────────────────

    async def _handle_tool_calls(
        self,
        client: Any,
        model: str,
        tool_manager: Any,
        messages: list[dict[str, Any]],
        tool_calls: list[Any],
        response_text: str,
        max_turns: int,
        raw: bool,
    ) -> str:
        """Handle tool calls in multi-turn conversation."""
        messages.append({
            "role": "assistant",
            "content": response_text,
            "tool_calls": tool_calls,
        })

        await self._execute_tool_call_batch(
            tool_manager, tool_calls, messages, raw
        )

        turns = 1
        while turns < max_turns:
            tools = await tool_manager.get_tools_for_llm()
            response = await client.create_completion(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=4096,
            )

            response_text = response.get("response", "")
            new_tool_calls = response.get("tool_calls", [])

            if not new_tool_calls:
                return response_text

            messages.append({
                "role": "assistant",
                "content": response_text,
                "tool_calls": new_tool_calls,
            })

            await self._execute_tool_call_batch(
                tool_manager, new_tool_calls, messages, raw
            )

            turns += 1

        # Max turns reached — get a final synthesized response without tools
        if not raw:
            output.warning(f"Max turns ({max_turns}) reached")

        try:
            final_response = await client.create_completion(
                model=model,
                messages=messages,
                max_tokens=4096,
            )
            return final_response.get("response", response_text)
        except Exception:
            return response_text

    async def _execute_tool_call_batch(
        self,
        tool_manager: Any,
        tool_calls: list[Any],
        messages: list[dict[str, Any]],
        raw: bool,
    ) -> None:
        """Execute a batch of tool calls and append results to messages."""
        for tool_call in tool_calls:
            tool_name, tool_args_str, tool_call_id = _parse_tool_call(tool_call)

            try:
                tool_args = (
                    json.loads(tool_args_str)
                    if isinstance(tool_args_str, str)
                    else tool_args_str
                )
            except json.JSONDecodeError:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": f"Error: Invalid JSON in tool arguments: {tool_args_str}",
                })
                continue

            if not raw:
                output.info(f"Executing tool: {tool_name}")

            try:
                result = await tool_manager.execute_tool(tool_name, tool_args)
                if result.success:
                    result_data = to_serializable(
                        unwrap_tool_result(result.result)
                    )
                else:
                    result_data = f"Error: {result.error}"
                result_str = (
                    json.dumps(result_data)
                    if not isinstance(result_data, str)
                    else result_data
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result_str,
                })
            except Exception as e:
                output.error(f"Tool execution failed: {e}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": f"Error: {e}",
                })


def _parse_tool_call(tool_call: Any) -> tuple[str, str, str]:
    """Extract (tool_name, arguments_str, call_id) from a tool_call."""
    if isinstance(tool_call, dict):
        return (
            tool_call.get("function", {}).get("name", ""),
            tool_call.get("function", {}).get("arguments", "{}"),
            tool_call.get("id", ""),
        )
    try:
        return (
            tool_call.function.name,
            tool_call.function.arguments,
            tool_call.id,
        )
    except AttributeError:
        return ("", "{}", "")
