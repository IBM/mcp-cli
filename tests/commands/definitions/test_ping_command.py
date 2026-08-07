"""Tests for the ping command."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, PropertyMock
from mcp_cli.commands.servers.ping import PingCommand


class TestPingCommand:
    """Test the PingCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a PingCommand instance."""
        return PingCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "ping"
        assert command.aliases == []
        assert "Test connectivity to MCP servers" in command.description

        # Test help_text
        assert "ping" in command.help_text.lower()
        assert "server" in command.help_text.lower()

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "server_index" in params
        assert "all" in params
        assert "timeout" in params

    @pytest.mark.asyncio
    async def test_execute_all_servers(self, command):
        """Test pinging all servers."""
        from mcp_cli.tools.models import ServerInfo

        mock_tm = Mock()
        mock_server = ServerInfo(
            id=1,
            name="test-server",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])
        mock_tm.check_server_health = AsyncMock(return_value={
            "test-server": {"status": "healthy", "ping_success": True},
        })

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_specific_server(self, command):
        """Test pinging a specific server."""
        from mcp_cli.tools.models import ServerInfo

        mock_tm = Mock()
        mock_server = ServerInfo(
            id=1,
            name="test-server",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])
        mock_tm.check_server_health = AsyncMock(return_value={
            "test-server": {"status": "healthy", "ping_success": True},
        })

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm, server_index=0)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_no_tool_manager(self, command):
        """Test when no tool manager is available."""
        result = await command.execute()

        assert result.success is False
        assert "No active tool manager" in result.error

    @pytest.mark.asyncio
    async def test_execute_failed_ping(self, command):
        """Test when server ping fails (transport-level)."""
        from mcp_cli.tools.models import ServerInfo

        mock_tm = Mock()
        mock_server = ServerInfo(
            id=1,
            name="test-server",
            status="stopped",
            connected=False,
            tool_count=0,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])
        mock_tm.check_server_health = AsyncMock(return_value={
            "test-server": {"status": "unhealthy", "ping_success": False},
        })

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during ping."""
        mock_tm = Mock()
        mock_tm.get_server_info = AsyncMock(side_effect=Exception("Network error"))

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm)
            assert result.success is False
            assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_context_exception(self, command):
        """Test getting tool manager from context when it throws exception."""
        with patch("mcp_cli.context.get_context") as mock_ctx:
            # Make context throw exception
            mock_ctx.side_effect = Exception("Context error")

            result = await command.execute()

            # Should handle exception and report no tool manager
            assert result.success is False
            assert "No active tool manager" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_args_list(self, command):
        """Test executing with args as list."""
        from mcp_cli.tools.models import ServerInfo

        mock_tm = Mock()
        mock_server1 = ServerInfo(
            id=1,
            name="server1",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )
        mock_server2 = ServerInfo(
            id=2,
            name="server2",
            status="running",
            connected=True,
            tool_count=3,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server1, mock_server2])
        mock_tm.check_server_health = AsyncMock(return_value={
            "server1": {"status": "healthy", "ping_success": True},
            "server2": {"status": "healthy", "ping_success": True},
        })

        with patch("chuk_term.ui.output"):
            # Pass args as a list
            result = await command.execute(
                tool_manager=mock_tm, args=["server1", "server2"]
            )
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_args_string(self, command):
        """Test executing with args as string."""
        from mcp_cli.tools.models import ServerInfo

        mock_tm = Mock()
        mock_server = ServerInfo(
            id=1,
            name="server1",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])
        mock_tm.check_server_health = AsyncMock(return_value={
            "server1": {"status": "healthy", "ping_success": True},
        })

        with patch("chuk_term.ui.output"):
            # Pass args as a string
            result = await command.execute(tool_manager=mock_tm, args="server1")
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_no_servers(self, command):
        """Test when no servers are available."""
        mock_tm = Mock()
        mock_tm.get_server_info = AsyncMock(return_value=[])

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm)
            assert result.success is False
            assert "No servers available" in result.output

    @pytest.mark.asyncio
    async def test_execute_with_context_success(self, command):
        """Test getting tool manager from context successfully."""
        from mcp_cli.tools.models import ServerInfo

        mock_server = ServerInfo(
            id=1,
            name="test-server",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )

        mock_tm = Mock()
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])
        mock_tm.check_server_health = AsyncMock(return_value={
            "test-server": {"status": "healthy", "ping_success": True},
        })

        mock_ctx = Mock()
        mock_ctx.tool_manager = mock_tm

        with patch("mcp_cli.commands.servers.ping.get_context") as mock_get_ctx:
            mock_get_ctx.return_value = mock_ctx
            with patch("chuk_term.ui.output"):
                result = await command.execute()
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_filter_by_index(self, command):
        """Test filtering servers by index."""
        from mcp_cli.tools.models import ServerInfo

        mock_tm = Mock()
        mock_server1 = ServerInfo(
            id=1,
            name="server1",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )
        mock_server2 = ServerInfo(
            id=2,
            name="server2",
            status="running",
            connected=True,
            tool_count=3,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server1, mock_server2])
        mock_tm.check_server_health = AsyncMock(return_value={
            "server1": {"status": "healthy", "ping_success": True},
            "server2": {"status": "healthy", "ping_success": True},
        })

        with patch("chuk_term.ui.output"):
            # Filter by index "0" - should only match first server
            result = await command.execute(tool_manager=mock_tm, args=["0"])
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_context_returns_none(self, command):
        """Test when context returns None."""
        with patch("mcp_cli.commands.servers.ping.get_context") as mock_get_ctx:
            mock_get_ctx.return_value = None

            result = await command.execute()

            assert result.success is False
            assert "No active tool manager" in result.error

    @pytest.mark.asyncio
    async def test_execute_server_ping_exception(self, command):
        """Test when accessing health check raises an exception for a server."""
        from mcp_cli.tools.models import ServerInfo

        mock_tm = Mock()
        mock_server = ServerInfo(
            id=1,
            name="test-server",
            status="running",
            connected=True,
            tool_count=5,
            namespace="test",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])
        # Return empty health dict so .get() works but returns no ping_success
        mock_tm.check_server_health = AsyncMock(return_value={})

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm)
            # Should fail because the server didn't respond to ping
            assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_sse_server_ping(self, command):
        """Test that ping works for SSE servers (issue #203).

        SSE transports don't expose raw streams, so ping must use
        the transport-level health check instead.
        """
        from mcp_cli.tools.models import ServerInfo, TransportType

        mock_tm = Mock()
        mock_server = ServerInfo(
            id=0,
            name="sse-echo",
            status="running",
            connected=True,
            tool_count=3,
            namespace="sse-echo",
            transport=TransportType.SSE,
            url="http://localhost:8081/mcp",
        )
        mock_tm.get_server_info = AsyncMock(return_value=[mock_server])
        mock_tm.check_server_health = AsyncMock(return_value={
            "sse-echo": {"status": "healthy", "ping_success": True},
        })

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm)
            assert result.success is True
            # Verify check_server_health was called (not just server.connected)
            mock_tm.check_server_health.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_mixed_transport_servers(self, command):
        """Test ping with both stdio and SSE servers."""
        from mcp_cli.tools.models import ServerInfo, TransportType

        mock_tm = Mock()
        stdio_server = ServerInfo(
            id=0,
            name="sqlite",
            status="running",
            connected=True,
            tool_count=5,
            namespace="sqlite",
            transport=TransportType.STDIO,
        )
        sse_server = ServerInfo(
            id=1,
            name="sse-echo",
            status="running",
            connected=True,
            tool_count=3,
            namespace="sse-echo",
            transport=TransportType.SSE,
            url="http://localhost:8081/mcp",
        )
        mock_tm.get_server_info = AsyncMock(
            return_value=[stdio_server, sse_server]
        )
        mock_tm.check_server_health = AsyncMock(return_value={
            "sqlite": {"status": "healthy", "ping_success": True},
            "sse-echo": {"status": "healthy", "ping_success": True},
        })

        with patch("chuk_term.ui.output"):
            result = await command.execute(tool_manager=mock_tm)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_online_count_reported(self, command):
        """Test that online/total count is reported."""
        from mcp_cli.tools.models import ServerInfo

        mock_tm = Mock()
        servers = [
            ServerInfo(
                id=i,
                name=f"server{i}",
                status="running",
                connected=True,
                tool_count=1,
                namespace="test",
            )
            for i in range(3)
        ]
        mock_tm.get_server_info = AsyncMock(return_value=servers)
        mock_tm.check_server_health = AsyncMock(return_value={
            "server0": {"status": "healthy", "ping_success": True},
            "server1": {"status": "unhealthy", "ping_success": False},
            "server2": {"status": "healthy", "ping_success": True},
        })

        info_calls = []
        with patch("chuk_term.ui.output") as mock_output:
            mock_output.info = lambda msg: info_calls.append(msg)
            mock_output.success = lambda msg: None
            mock_output.error = lambda msg: None
            result = await command.execute(tool_manager=mock_tm)
            # One server failed, so overall success is False
            assert result.success is False
            # Check that the summary line was emitted
            assert any("2/3 servers online" in call for call in info_calls)
