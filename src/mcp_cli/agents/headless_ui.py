# mcp_cli/agents/headless_ui.py
"""Headless UI manager for spawned agents.

Provides the same interface as ChatUIManager but with no terminal output.
All display operations are logged at DEBUG level instead.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HeadlessUIManager:
    """UIManager for headless agents — no terminal output, logs only.

    Implements the subset of ChatUIManager used by ConversationProcessor
    and ToolProcessor so spawned agents can run without a terminal.
    """

    def __init__(self, agent_id: str = "default") -> None:
        self.agent_id = agent_id
        self.verbose_mode = False
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_times: list[float] = []
        self.tool_start_time: float | None = None
        self.current_tool_start_time: float | None = None
        self.streaming_handler: Any = None
        self.tools_running = False
        self.display: Any = None  # no display manager
        self.console: Any = None  # no rich console

    # -- Streaming ----------------------------------------------------------

    @property
    def is_streaming_response(self) -> bool:
        return self.streaming_handler is not None

    async def start_streaming_response(self) -> None:
        logger.debug("[%s] streaming response started", self.agent_id)

    async def stop_streaming_response(self) -> None:
        logger.debug("[%s] streaming response stopped", self.agent_id)
        self.streaming_handler = None

    # -- Tool display -------------------------------------------------------

    def print_tool_call(self, tool_name: str, arguments: Any) -> None:
        logger.debug("[%s] tool call: %s(%s)", self.agent_id, tool_name, arguments)

    async def start_tool_execution(self, tool_name: str, arguments: Any) -> None:
        logger.debug("[%s] tool start: %s", self.agent_id, tool_name)

    async def finish_tool_execution(
        self, result: str = "", success: bool = True
    ) -> None:
        logger.debug("[%s] tool finish: success=%s", self.agent_id, success)

    async def finish_tool_calls(self) -> None:
        logger.debug("[%s] all tool calls finished", self.agent_id)

    # -- Confirmation -------------------------------------------------------

    async def do_confirm_tool_execution(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        server_name: str = "",
    ) -> bool:
        """Headless agents auto-approve all tool executions."""
        return True

    # -- Message display ----------------------------------------------------

    async def print_assistant_message(self, content: str) -> None:
        logger.debug("[%s] assistant: %.200s", self.agent_id, content)

    # -- Input (not used by headless agents) --------------------------------

    async def get_user_input(self) -> str | None:
        return None

    # -- Cleanup ------------------------------------------------------------

    async def cleanup(self) -> None:
        pass
