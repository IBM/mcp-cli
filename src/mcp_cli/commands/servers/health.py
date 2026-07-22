# src/mcp_cli/commands/servers/health.py
"""
Unified server health command — check MCP server connectivity and status.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandParameter,
    CommandResult,
)
from mcp_cli.context import get_context

logger = logging.getLogger(__name__)


class HealthCommand(UnifiedCommand):
    """Check health of MCP servers."""

    @property
    def name(self) -> str:
        return "health"

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def description(self) -> str:
        return "Check health of MCP servers"

    @property
    def help_text(self) -> str:
        return """
Check health of MCP servers via ping.

Usage:
  /health              - Check all servers
  /health <server>     - Check a specific server

Shows server status (healthy/unhealthy/timeout/error) and ping latency.
"""

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="server_name",
                type=str,
                required=False,
                help="Name of a specific server to check",
            ),
        ]

    async def execute(self, **kwargs: Any) -> CommandResult:
        """Execute the health command."""
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

        # Parse server name from args
        server_name = kwargs.get("server_name")
        if server_name is None:
            args_val = kwargs.get("args")
            if isinstance(args_val, list) and args_val:
                server_name = str(args_val[0])
            elif isinstance(args_val, str) and args_val.strip():
                server_name = args_val.strip()

        try:
            start = time.monotonic()
            results = await tool_manager.check_server_health(server_name)
            elapsed = time.monotonic() - start

            if not results:
                if server_name:
                    return CommandResult(
                        success=False,
                        error=f"Server not found: {server_name}",
                    )
                return CommandResult(
                    success=False,
                    error="No servers available.",
                )

            output.rule("[bold]Server Health[/bold]", style="primary")

            all_healthy = True
            for name, info in results.items():
                status = info.get("status", "unknown") if info else "unknown"
                ping_ok = info.get("ping_success", False) if info else False

                if status == "healthy" and ping_ok:
                    output.success(f"  {name}: healthy")
                elif status == "timeout":
                    output.warning(f"  {name}: timeout")
                    all_healthy = False
                else:
                    error_detail = info.get("error", "") if info else ""
                    detail = f" — {error_detail}" if error_detail else ""
                    output.error(f"  {name}: {status}{detail}")
                    all_healthy = False

            output.print()
            output.info(f"  Health check completed in {elapsed:.0f}ms")

            return CommandResult(success=all_healthy, data=results)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Health check failed: {e}",
            )
