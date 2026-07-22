"""Server management commands."""

from mcp_cli.commands.servers.servers import ServersCommand
from mcp_cli.commands.servers.server_singular import ServerSingularCommand
from mcp_cli.commands.servers.ping import PingCommand
from mcp_cli.commands.servers.health import HealthCommand

__all__ = [
    "ServersCommand",
    "ServerSingularCommand",
    "PingCommand",
    "HealthCommand",
]
