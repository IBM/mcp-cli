"""Tests for the prompts command."""

import pytest
from unittest.mock import patch
from mcp_cli.commands.resources.prompts import PromptsCommand


class TestPromptsCommand:
    """Test the PromptsCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a PromptsCommand instance."""
        return PromptsCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "prompts"
        assert command.aliases == []  # No aliases in implementation
        assert "prompts" in command.description.lower()

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "server" in params
        assert "raw" in params
        assert "get" in params

    @pytest.mark.asyncio
    async def test_execute_list_all(self, command):
        """Test listing all prompts."""
        from unittest.mock import AsyncMock, MagicMock

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            # Create mock prompt objects
            mock_prompt1 = MagicMock()
            mock_prompt1.name = "summarize"
            mock_prompt1.description = "Summarize text"
            mock_prompt2 = MagicMock()
            mock_prompt2.name = "translate"
            mock_prompt2.description = "Translate text"
            mock_prompts = [mock_prompt1, mock_prompt2]
            mock_ctx.tool_manager.list_prompts = AsyncMock(return_value=mock_prompts)
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute()

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_by_server(self, command):
        """Test listing prompts for a specific server."""
        from unittest.mock import AsyncMock, MagicMock

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_prompt = MagicMock()
            mock_prompt.name = "summarize"
            mock_prompt.description = "Summarize text"
            mock_prompts = [mock_prompt]
            mock_ctx.tool_manager.list_prompts = AsyncMock(return_value=mock_prompts)
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute(
                        server=0
                    )  # server parameter is an index

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_detailed(self, command):
        """Test listing prompts with detailed information."""
        from unittest.mock import AsyncMock, MagicMock

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_prompt = MagicMock()
            mock_prompt.name = "summarize"
            mock_prompt.description = "Summarize text"
            mock_prompts = [mock_prompt]
            mock_ctx.tool_manager.list_prompts = AsyncMock(return_value=mock_prompts)
            with patch("chuk_term.ui.output"):
                with patch("chuk_term.ui.format_table"):
                    result = await command.execute(raw=True)

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_no_prompts(self, command):
        """Test when no prompts are available."""
        from unittest.mock import AsyncMock

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.list_prompts = AsyncMock(return_value=[])
            with patch("chuk_term.ui.output"):
                result = await command.execute()

                assert result.success is True
                # Should indicate no prompts available

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during execution."""
        from unittest.mock import AsyncMock

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.list_prompts = AsyncMock(
                side_effect=Exception("Server error")
            )
            with patch("chuk_term.ui.output"):
                result = await command.execute()

                assert result.success is False
                assert "Server error" in result.error

    @pytest.mark.asyncio
    async def test_execute_get_renders_messages(self, command):
        """--get <name> fetches a prompt and renders its messages."""
        from unittest.mock import AsyncMock

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_prompt = AsyncMock(
                return_value={
                    "description": "Greeter",
                    "messages": [
                        {"role": "user", "content": {"type": "text", "text": "Hello"}}
                    ],
                }
            )
            # list_prompts must NOT be consulted in get mode
            mock_ctx.tool_manager.list_prompts = AsyncMock(
                side_effect=AssertionError(
                    "list_prompts should not be called in get mode"
                )
            )
            result = await command.execute(get="greet")

        assert result.success is True
        assert "Greeter" in result.output
        assert "[user] Hello" in result.output
        mock_ctx.tool_manager.get_prompt.assert_awaited_once_with("greet")

    @pytest.mark.asyncio
    async def test_execute_get_not_found(self, command):
        """--get on a missing/empty prompt is a failure, not a crash."""
        from unittest.mock import AsyncMock

        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.tool_manager.get_prompt = AsyncMock(return_value={})
            result = await command.execute(get="nope")

        assert result.success is False
        assert "nope" in result.error
