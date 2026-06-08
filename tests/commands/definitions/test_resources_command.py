"""Tests for the resources command."""

import pytest
from unittest.mock import patch, AsyncMock
from mcp_cli.commands.resources.resources import ResourcesCommand


class TestResourcesCommand:
    """Test the ResourcesCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ResourcesCommand instance."""
        return ResourcesCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "resources"
        assert command.aliases == []  # No aliases in implementation
        assert "resources" in command.description.lower()

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "server" in params
        assert "raw" in params
        assert "uri" in params
        assert "read" in params

    @pytest.mark.asyncio
    async def test_execute_list_all(self, command):
        """Test listing all resources."""
        from mcp_cli.tools.models import ResourceInfo

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_resources = [
                ResourceInfo(
                    id="file:///database.db",
                    name="database.db",
                    type="application/x-sqlite3",
                ),
                ResourceInfo(
                    id="file:///config.json",
                    name="config.json",
                    type="application/json",
                ),
            ]
            mock_ctx.tool_manager.list_resources = AsyncMock(
                return_value=mock_resources
            )
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute()

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_by_server(self, command):
        """Test listing resources for a specific server."""
        from mcp_cli.tools.models import ResourceInfo

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_resources = [
                ResourceInfo(
                    id="file:///database.db",
                    name="database.db",
                    type="application/x-sqlite3",
                ),
            ]
            mock_ctx.tool_manager.list_resources = AsyncMock(
                return_value=mock_resources
            )
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute(
                        server=0
                    )  # server parameter is an index

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_detailed(self, command):
        """Test listing resources with detailed information."""
        from mcp_cli.tools.models import ResourceInfo

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_resources = [
                ResourceInfo(
                    id="file:///database.db",
                    name="database.db",
                    type="application/x-sqlite3",
                ),
            ]
            mock_ctx.tool_manager.list_resources = AsyncMock(
                return_value=mock_resources
            )
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute(raw=True)

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_no_resources(self, command):
        """Test when no resources are available."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.list_resources = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                result = await command.execute()

                assert result.success is True
                # Should indicate no resources available

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during execution."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.list_resources = AsyncMock(
                side_effect=Exception("Server not connected")
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute()

                assert result.success is False
                assert "Server not connected" in result.error

    @pytest.mark.asyncio
    async def test_execute_read_returns_contents(self, command):
        """--read <uri> prints the resource's text contents."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.read_resource = AsyncMock(
                return_value={
                    "contents": [
                        {
                            "uri": "debug://mail_log",
                            "mimeType": "text/plain",
                            "text": "line1\nline2",
                        }
                    ]
                }
            )
            # list_resources must NOT be consulted in read mode
            mock_ctx.tool_manager.list_resources = AsyncMock(
                side_effect=AssertionError(
                    "list_resources should not be called in read mode"
                )
            )
            result = await command.execute(read="debug://mail_log")

        assert result.success is True
        assert result.output == "line1\nline2"
        mock_ctx.tool_manager.read_resource.assert_awaited_once_with("debug://mail_log")

    @pytest.mark.asyncio
    async def test_execute_read_not_found(self, command):
        """--read on a missing/empty resource is a failure, not a crash."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.read_resource = AsyncMock(return_value={})
            result = await command.execute(read="debug://nope")

        assert result.success is False
        assert "debug://nope" in result.error
