"""
Execute tool command for interactive mode.

Allows executing MCP tools directly with parameters.
"""

from __future__ import annotations

import json
from typing import Any

from mcp_cli.commands.base import (
    CommandMode,
    CommandParameter,
    CommandResult,
    UnifiedCommand,
)
from mcp_cli.config.defaults import (
    JSON_TYPE_ARRAY,
    JSON_TYPE_BOOLEAN,
    JSON_TYPE_NUMBER,
    JSON_TYPE_OBJECT,
    JSON_TYPE_STRING,
)
from mcp_cli.tools.manager import ToolManager
from mcp_cli.utils.serialization import to_serializable, unwrap_tool_result
from chuk_term.ui import output


class ExecuteToolCommand(UnifiedCommand):
    """Command to execute a tool with parameters."""

    def __init__(self) -> None:
        """Initialize the execute tool command."""
        super().__init__()
        self._name: str = "execute"
        self._description: str = "Execute a tool with parameters"
        self._modes: CommandMode = CommandMode.INTERACTIVE | CommandMode.CHAT
        self._aliases: list[str] = ["exec", "run"]
        self._parameters: list[CommandParameter] = [
            CommandParameter(
                name="tool",
                type=str,
                help="Name of the tool to execute",
                required=False,
            ),
            CommandParameter(
                name="params",
                type=str,
                help="Tool parameters as JSON string",
                required=False,
            ),
            CommandParameter(
                name="server",
                type=str,
                help="Server to use (if multiple available)",
                required=False,
            ),
        ]
        self._help_text: str = """
Execute a tool with JSON parameters.

Usage:
  execute                           # List all available tools
  execute <tool_name>               # Show tool parameters and format
  execute <tool_name> '<json>'      # Execute tool with parameters

IMPORTANT: Parameters must be in JSON format!

Examples:
  execute                           # See what tools are available
  execute echo_text                 # See what parameters echo_text needs
  execute echo_text '{"message": "Hello"}'   # Execute with JSON parameters
  execute list_tables '{}'          # Execute tool with no parameters

Common Mistakes to Avoid:
  ❌ execute echo_text "hello"      # Wrong: Not JSON format
  ✅ execute echo_text '{"message": "hello"}'  # Correct: Proper JSON

Tips:
  • Always use JSON format: '{"key": "value"}'
  • Use single quotes around JSON to avoid escaping
  • Use '{}' for tools with no required parameters
  • Check parameters first with: execute <tool_name>
"""
        self._requires_context: bool = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def modes(self) -> CommandMode:
        return self._modes

    @property
    def aliases(self) -> list[str]:
        return self._aliases

    @property
    def parameters(self) -> list[CommandParameter]:
        return self._parameters

    @property
    def help_text(self) -> str:
        return self._help_text

    @property
    def requires_context(self) -> bool:
        return self._requires_context

    async def execute(
        self,
        tool_manager: ToolManager | None = None,
        tool: str | None = None,
        params: str | None = None,
        server: str | None = None,
        args: Any | None = None,
        **kwargs,
    ) -> CommandResult:
        """Execute the tool command."""
        if not tool_manager:
            return CommandResult(
                success=False,
                error="Tool manager not available. Are servers connected?",
            )

        # Handle positional arguments
        if args:
            # Debug logging
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                f"Received args: {repr(args)}, tool={repr(tool)}, params={repr(params)}"
            )

            if isinstance(args, list):
                if len(args) > 0 and not tool:
                    tool = args[0]
                if len(args) > 1 and not params:
                    params = args[1]
            elif isinstance(args, str):
                # Could be either tool name or params
                if not tool:
                    tool = args
                elif not params:
                    params = args

        # If no tool specified, list available tools
        if not tool:
            return await self._list_tools(tool_manager)

        # Find the tool
        try:
            all_tools = await tool_manager.get_all_tools()
            matching_tools = [t for t in all_tools if t.name.lower() == tool.lower()]

            # Filter by server/namespace if specified
            if server:
                matching_tools = [
                    t for t in matching_tools if t.namespace.lower() == server.lower()
                ]
                if not matching_tools:
                    output.error(f"Tool '{tool}' not found on server '{server}'")
                    return await self._list_tools(tool_manager)

            if not matching_tools:
                output.error(f"Tool not found: {tool}")
                return await self._list_tools(tool_manager)

            tool_info = matching_tools[0]
        except Exception as e:
            return CommandResult(success=False, error=f"Failed to get tools: {e}")

        # If no params, show tool info
        if not params:
            return self._show_tool_info(tool_info)

        # Parse parameters
        try:
            # Debug: show what we received
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Received params: {repr(params)}")

            # Remove surrounding quotes if present (do this FIRST before checking format)
            if params.startswith("'") and params.endswith("'"):
                params = params[1:-1]
            elif params.startswith('"') and params.endswith('"'):
                params = params[1:-1]

            # Check if user provided a plain string (common mistake)
            if params and not params.startswith("{") and "=" not in params:
                # User likely provided a plain string instead of JSON
                output.error("❌ Invalid format: Parameters must be in JSON format")
                output.print(f"\nYou provided: {params}")
                output.print("This appears to be a plain string, not JSON.\n")

                # Show correct format
                output.rule("Correct Format")

                # Guess the most likely parameter name based on tool
                param_name = "message"  # Default for many tools
                if hasattr(tool_info, "parameters") and tool_info.parameters:
                    props = tool_info.parameters.get("properties", {})
                    if props:
                        # Use the first required parameter, or first parameter
                        required = tool_info.parameters.get("required", [])
                        if required:
                            param_name = required[0]
                        elif props:
                            param_name = list(props.keys())[0]

                output.success("✅ Use this format:")
                output.print(
                    f"   /exec {tool_info.name} '{{{json.dumps(param_name)}: {json.dumps(params)}}}'"
                )
                output.print("\n📝 Example:")
                output.print(
                    f"   /exec {tool_info.name} '{{{json.dumps(param_name)}: \"your text here\"}}'"
                )

                return CommandResult(success=False)

            if params.startswith("{"):
                # Already JSON
                tool_params = json.loads(params)
            else:
                # Try to parse as simple key=value pairs
                tool_params = self._parse_simple_params(params)
        except (json.JSONDecodeError, ValueError) as e:
            # Provide helpful error message with examples
            output.error("❌ Invalid parameter format")
            output.print(f"\nYou provided: {params}")
            output.print(f"Error: {e}")

            # Show correct format
            output.rule("Correct Format")

            # Get required parameters for this tool
            from mcp_cli.tools.models import ToolInfo

            if isinstance(tool_info, ToolInfo) and tool_info.parameters:
                schema = tool_info.parameters
            elif hasattr(tool_info, "parameters") and tool_info.parameters:
                schema = tool_info.parameters
            else:
                schema = {}

            # Build example
            example_params: dict[str, Any] = {}
            if "properties" in schema:
                from mcp_cli.config import (
                    JSON_TYPE_BOOLEAN,
                    JSON_TYPE_NUMBER,
                    JSON_TYPE_STRING,
                )

                for prop_name, prop_info in schema["properties"].items():
                    if prop_name in schema.get("required", []):
                        prop_type = prop_info.get("type", JSON_TYPE_STRING)
                        if prop_type == JSON_TYPE_STRING:
                            if prop_name == "message":
                                example_params[prop_name] = "your message here"
                            else:
                                example_params[prop_name] = f"<{prop_name}>"
                        elif prop_type == JSON_TYPE_NUMBER:
                            example_params[prop_name] = 123
                        elif prop_type == JSON_TYPE_BOOLEAN:
                            example_params[prop_name] = True

            output.success("✅ Use this format:")
            if example_params:
                output.print(
                    f"   /exec {tool_info.name} '{json.dumps(example_params)}'"
                )
            else:
                output.print(f"   /exec {tool_info.name} '{{}}'")

            output.print("\n📝 Remember:")
            output.print("   • Parameters must be in JSON format")
            output.print("   • Use single quotes around the JSON")
            output.print("   • Use /execute <tool> to see required parameters")

            return CommandResult(success=False)

        # Execute the tool
        try:
            output.info(f"Executing tool: {tool_info.name}")

            # Execute through tool manager
            result = await tool_manager.execute_tool(
                tool_name=tool_info.name, arguments=tool_params
            )

            # Display result
            if result:
                from mcp_cli.tools.models import ToolCallResult

                if isinstance(result, ToolCallResult):
                    if result.success and result.result is not None:
                        # Extract the actual result from ToolCallResult
                        # Unwrap middleware wrappers, then serialize
                        serializable_result = to_serializable(unwrap_tool_result(result.result))
                        output.success("✅ Tool executed successfully")
                        if isinstance(serializable_result, (dict, list)):
                            output.print(json.dumps(serializable_result, indent=2))
                        else:
                            output.print(str(serializable_result))
                    elif result.error:
                        # Handle error case cleanly without scary stack traces
                        output.error("❌ Tool execution failed")

                        # Parse the error to provide helpful feedback
                        error_msg = str(result.error)
                        if (
                            "Invalid parameter" in error_msg
                            and "expected string, got NoneType" in error_msg
                        ):
                            # Extract parameter name from error
                            import re

                            match = re.search(r"Invalid parameter '(\w+)'", error_msg)
                            if match:
                                param_name = match.group(1)
                                output.print(
                                    f"\nMissing required parameter: '{param_name}'"
                                )

                                # Show correct format
                                output.rule("Correct Format")

                                # Build example with actual parameter
                                example_params = {
                                    param_name: f"<your {param_name} here>"
                                }
                                output.success("✅ Use this format:")
                                output.print(
                                    f"   /exec {tool_info.name} '{json.dumps(example_params)}'"
                                )
                        else:
                            output.print(f"Error details: {error_msg}")
                    else:
                        output.warning("Tool returned no result")
                elif isinstance(result, dict):  # type: ignore[unreachable]
                    serializable_result = to_serializable(unwrap_tool_result(result))
                    output.success("✅ Tool executed successfully")
                    output.print(json.dumps(serializable_result, indent=2))
                else:
                    output.success("✅ Tool executed successfully")
                    output.print(str(result))
            else:
                output.warning("Tool returned no result")

            return CommandResult(success=True)

        except Exception as e:
            # Clean error message without stack trace
            output.error("❌ Tool execution error")
            error_str = str(e)

            # Check for common errors and provide helpful feedback
            if "got an unexpected keyword argument" in error_str:
                output.print("Internal error - please report this issue")
            elif "Invalid parameter" in error_str:
                output.print(f"Parameter validation failed: {error_str}")
                output.hint("Use /execute <tool> to see correct parameters")
            else:
                output.print(f"Error: {error_str}")

            return CommandResult(
                success=False, error=f"Failed to execute tool: {error_str}"
            )

    async def _list_tools(self, tool_manager: ToolManager) -> CommandResult:
        """List available tools."""
        try:
            # Get all tools from the tool manager
            tools = await tool_manager.get_all_tools()

            if not tools:
                return CommandResult(
                    success=True,
                    output="No tools available. Connect to a server first.",
                )

            output.rule("Available Tools")

            # Group tools by namespace/server
            tools_by_namespace: dict[str, list] = {}
            for tool in tools:
                namespace = tool.namespace or "default"
                if namespace not in tools_by_namespace:
                    tools_by_namespace[namespace] = []
                tools_by_namespace[namespace].append(tool)

            for namespace, namespace_tools in tools_by_namespace.items():
                # Try to get server name from tool_manager.server_names
                server_display = namespace
                if hasattr(tool_manager, "server_names"):
                    # Look for server name by index or use namespace
                    for idx, name in tool_manager.server_names.items():
                        if str(idx) in namespace or namespace == "default":
                            server_display = name if name else "unknown"
                            break

                output.rule(f"Server: {server_display}")
                for tool in namespace_tools:
                    output.print(f"  • {tool.name}: {tool.description}")

            output.hint("\nUse 'execute <tool_name>' to see tool parameters")
            output.hint("Use 'execute <tool_name> <params>' to run a tool")

            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(success=False, error=f"Failed to list tools: {e}")

    def _show_tool_info(self, tool: Any) -> CommandResult:
        """Show detailed tool information."""
        from mcp_cli.tools.models import ToolInfo

        output.rule(f"Tool: {tool.name}")
        output.print(f"Description: {tool.description or 'No description available'}")

        # Show parameters
        if isinstance(tool, ToolInfo) and tool.parameters:
            schema = tool.parameters
        elif hasattr(tool, "inputSchema") and tool.inputSchema:
            schema = tool.inputSchema
        elif hasattr(tool, "parameters") and tool.parameters:
            schema = tool.parameters
        else:
            schema = None

        if schema:
            output.rule("Parameters")

            if "properties" in schema:
                for prop_name, prop_info in schema["properties"].items():
                    required = prop_name in schema.get("required", [])
                    prop_type = prop_info.get("type", "any")
                    prop_desc = prop_info.get("description", "")

                    req_marker = "*" if required else ""
                    output.print(
                        f"  • {prop_name}{req_marker} ({prop_type}): {prop_desc}"
                    )

                if "required" in schema:
                    output.print("\n  * = required parameter")
            else:
                output.print("  No parameters required")

            # Show example
            output.rule("Example Usage")

            # Build example params
            example_params: dict[str, Any] = {}
            if "properties" in schema:
                for prop_name, prop_info in schema["properties"].items():
                    if prop_name in schema.get("required", []):
                        prop_type = prop_info.get("type", "string")
                        if prop_type == JSON_TYPE_STRING:
                            example_params[prop_name] = f"<{prop_name}>"
                        elif prop_type == JSON_TYPE_NUMBER:
                            example_params[prop_name] = 0
                        elif prop_type == JSON_TYPE_BOOLEAN:
                            example_params[prop_name] = True
                        elif prop_type == JSON_TYPE_ARRAY:
                            example_params[prop_name] = []
                        elif prop_type == JSON_TYPE_OBJECT:
                            example_params[prop_name] = {}

            if example_params:
                output.print(f"  execute {tool.name} '{json.dumps(example_params)}'")
            else:
                output.print(f"  execute {tool.name}")

        else:
            output.info("This tool has no parameters")
            output.print(f"\nUsage: execute {tool.name} '{{}}'")
            # Show example even for no-param tools
            output.rule("Example Usage")
            output.print(f"  execute {tool.name} '{{}}'   # Empty JSON object")

        return CommandResult(success=True)

    def _parse_simple_params(self, params: str) -> dict[str, Any]:
        """Parse simple key=value parameters."""
        result = {}

        # Split by spaces, respecting quotes
        import shlex

        parts = shlex.split(params)

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)

                # Try to parse value as JSON
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Keep as string
                    pass

                result[key] = value
            else:
                # Single value, use as "value" key
                result["value"] = part

        return result
