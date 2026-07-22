# tests/agents/test_loop.py
"""Unit tests for the headless agent loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_cli.agents.loop import run_agent_loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(agent_id: str = "test-agent"):
    """Return a mock ChatContext."""
    ctx = MagicMock()
    ctx.agent_id = agent_id
    ctx.exit_requested = False
    ctx.conversation_history = []
    ctx.dashboard_bridge = None
    ctx.add_user_message = AsyncMock()
    return ctx


def _make_ui(agent_id: str = "test-agent"):
    from mcp_cli.agents.headless_ui import HeadlessUIManager

    return HeadlessUIManager(agent_id=agent_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunAgentLoop:
    @pytest.mark.asyncio
    async def test_exit_on_stop_signal(self):
        """Loop exits on '__stop__' message."""
        ctx = _make_context()
        ui = _make_ui()
        q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        await q.put("__stop__")

        with patch("mcp_cli.chat.conversation.ConversationProcessor"):
            result = await run_agent_loop(ctx, ui, q, done)

        assert done.is_set()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_exit_on_quit(self):
        """Loop exits on 'quit' message."""
        ctx = _make_context()
        ui = _make_ui()
        q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        await q.put("quit")

        with patch("mcp_cli.chat.conversation.ConversationProcessor"):
            await run_agent_loop(ctx, ui, q, done)

        assert done.is_set()

    @pytest.mark.asyncio
    async def test_skip_empty_and_none(self):
        """Loop skips None and empty messages."""
        ctx = _make_context()
        ui = _make_ui()
        q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        await q.put(None)
        await q.put("")
        await q.put("   ")
        await q.put("exit")

        with patch("mcp_cli.chat.conversation.ConversationProcessor"):
            await run_agent_loop(ctx, ui, q, done)

        assert done.is_set()
        ctx.add_user_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_processes_prompt(self):
        """Loop processes a normal prompt via ConversationProcessor."""
        ctx = _make_context()
        ui = _make_ui()
        q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        await q.put("Hello agent")
        await q.put("exit")

        mock_convo = MagicMock()
        mock_convo.process_conversation = AsyncMock()

        with patch(
            "mcp_cli.chat.conversation.ConversationProcessor",
            return_value=mock_convo,
        ):
            await run_agent_loop(ctx, ui, q, done)

        ctx.add_user_message.assert_called_once_with("Hello agent")
        mock_convo.process_conversation.assert_called_once()
        assert done.is_set()

    @pytest.mark.asyncio
    async def test_captures_last_response(self):
        """Loop captures the last assistant response from history."""
        ctx = _make_context()
        ctx.conversation_history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello there!"},
        ]
        ui = _make_ui()
        q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        await q.put("hi")
        await q.put("exit")

        mock_convo = MagicMock()
        mock_convo.process_conversation = AsyncMock()

        with patch(
            "mcp_cli.chat.conversation.ConversationProcessor",
            return_value=mock_convo,
        ):
            result = await run_agent_loop(ctx, ui, q, done)

        assert "Hello there!" in result

    @pytest.mark.asyncio
    async def test_handles_cancellation(self):
        """Loop handles CancelledError gracefully."""
        ctx = _make_context()
        ui = _make_ui()
        q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        mock_convo = MagicMock()
        mock_convo.process_conversation = AsyncMock(
            side_effect=asyncio.CancelledError()
        )

        await q.put("do something")

        with patch(
            "mcp_cli.chat.conversation.ConversationProcessor",
            return_value=mock_convo,
        ):
            await run_agent_loop(ctx, ui, q, done)

        assert done.is_set()

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        """Loop handles unexpected exceptions."""
        ctx = _make_context()
        ui = _make_ui()
        q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        mock_convo = MagicMock()
        mock_convo.process_conversation = AsyncMock(side_effect=RuntimeError("boom"))

        await q.put("do something")

        with patch(
            "mcp_cli.chat.conversation.ConversationProcessor",
            return_value=mock_convo,
        ):
            result = await run_agent_loop(ctx, ui, q, done)

        assert done.is_set()
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_exit_requested_stops_loop(self):
        """Loop exits when context.exit_requested is True."""
        ctx = _make_context()
        ctx.exit_requested = True
        ui = _make_ui()
        q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        with patch("mcp_cli.chat.conversation.ConversationProcessor"):
            result = await run_agent_loop(ctx, ui, q, done)

        assert done.is_set()
        assert result == "Agent completed."

    @pytest.mark.asyncio
    async def test_dashboard_bridge_wired(self):
        """Loop wires dashboard bridge input queue and broadcasts messages."""
        ctx = _make_context()
        bridge = MagicMock()
        bridge.set_input_queue = MagicMock()
        bridge.on_message = AsyncMock()
        ctx.dashboard_bridge = bridge

        ui = _make_ui()
        q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        await q.put("hello")
        await q.put("exit")

        mock_convo = MagicMock()
        mock_convo.process_conversation = AsyncMock()

        with patch(
            "mcp_cli.chat.conversation.ConversationProcessor",
            return_value=mock_convo,
        ):
            await run_agent_loop(ctx, ui, q, done)

        bridge.set_input_queue.assert_called_once_with(q)
        bridge.on_message.assert_called_once_with("user", "hello")

    @pytest.mark.asyncio
    async def test_max_response_length_truncated(self):
        """Loop truncates last_response to 500 chars."""
        ctx = _make_context()
        long_response = "x" * 1000
        ctx.conversation_history = [
            {"role": "assistant", "content": long_response},
        ]
        ui = _make_ui()
        q: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        await q.put("go")
        await q.put("exit")

        mock_convo = MagicMock()
        mock_convo.process_conversation = AsyncMock()

        with patch(
            "mcp_cli.chat.conversation.ConversationProcessor",
            return_value=mock_convo,
        ):
            result = await run_agent_loop(ctx, ui, q, done)

        assert len(result) == 500
