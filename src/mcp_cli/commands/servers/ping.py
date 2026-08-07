# src/mcp_cli/commands/definitions/ping.py
"""
Unified ping command implementation.
"""

from __future__ import annotations

import logging

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandParameter,
    CommandResult,
)
from mcp_cli.context import get_context

logger = logging.getLogger(__name__)


class PingCommand(UnifiedCommand):
    """Test connectivity to MCP servers."""

    @property
    def name(self) -> str:
        return "ping"

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def description(self) -> str:
        return "Test connectivity to MCP servers"

    @property
    def help_text(self) -> str:
        return """
Test connectivity to MCP servers.

Usage:
  /ping [server_index]    - Test server connectivity (chat mode)
  ping [server_index]     - Test server connectivity (interactive mode)
  mcp-cli ping            - Test all servers (CLI mode)
  
Options:
  --all     - Test all servers
  --timeout - Timeout in seconds (default: 5)

Examples:
  /ping           - Test all servers
  /ping 0         - Test first server
  ping --all      - Test all servers
"""

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="server_index",
                type=int,
                required=False,
                help="Index of server to ping (omit for all)",
            ),
            CommandParameter(
                name="all",
                type=bool,
                default=False,
                help="Test all servers",
                is_flag=True,
            ),
            CommandParameter(
                name="timeout",
                type=float,
                default=5.0,
                help="Timeout in seconds",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the ping command."""
        from chuk_term.ui import output

        # Get tool manager
        tool_manager = kwargs.get("tool_manager")
        if not tool_manager:
            try:
                context = get_context()
                if context:
                    tool_manager = context.tool_manager
            except Exception as e:
                logger.debug("Failed to get tool manager from context: %s", e)

        if not tool_manager:
            return CommandResult(
                success=False,
                error="No active tool manager. Please connect to a server first.",
            )

        # Get parameters
        server_index = kwargs.get("server_index")
        targets = []

        # Handle positional argument
        if server_index is None and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list) and args_val:
                # Could be server names or indices
                targets = args_val
            elif isinstance(args_val, str):
                targets = [args_val]
        elif server_index is not None:
            targets = [str(server_index)]

        try:
            # Get server information
            servers = await tool_manager.get_server_info()

            if not servers:
                return CommandResult(
                    success=False,
                    output="No servers available to ping.",
                )

            # Actually ping servers via transport-level health check.
            # This works for all transport types (stdio, SSE, HTTP).
            health = await tool_manager.check_server_health()

            output.info("Pinging servers...")
            success = True

            for idx, server in enumerate(servers):
                # Skip if filtering by targets and this server doesn't match
                if (
                    targets
                    and server.name not in targets
                    and str(idx) not in targets
                ):
                    continue

                try:
                    # Use live health check result from transport.send_ping()
                    server_health = health.get(server.name, {})
                    ping_ok = server_health.get("ping_success", False)

                    if ping_ok:
                        output.success(f"✓ {server.name}: Online")
                    else:
                        status = server_health.get("status", "unreachable")
                        output.error(f"✗ {server.name}: {status}")
                        success = False
                except Exception as e:
                    output.error(f"✗ {server.name}: Error - {str(e)}")
                    success = False

            online = sum(
                1 for s in servers
                if health.get(s.name, {}).get("ping_success", False)
            )
            output.info(f"{online}/{len(servers)} servers online")

            return CommandResult(success=success)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to ping servers: {str(e)}",
            )
