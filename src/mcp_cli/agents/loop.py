# mcp_cli/agents/loop.py
"""Headless agent loop for spawned agents.

Runs the ConversationProcessor in a background task, reading prompts from
an asyncio.Queue instead of stdin.  No terminal I/O, no signal handlers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def run_agent_loop(
    context: Any,
    ui_manager: Any,
    input_queue: asyncio.Queue,
    done_event: asyncio.Event,
    max_turns: int = 50,
) -> str:
    """Run a headless chat loop for a spawned agent.

    Similar to ``_run_enhanced_chat_loop`` in ``chat_handler.py`` but:

    * No terminal I/O — reads prompts from *input_queue*.
    * Uses ``HeadlessUIManager`` (no rich/curses).
    * Reports completion via *done_event*.
    * Returns a summary of the final assistant response.

    Parameters
    ----------
    context:
        ChatContext for this agent.
    ui_manager:
        HeadlessUIManager instance.
    input_queue:
        ``asyncio.Queue`` that receives user/supervisor prompts.
    done_event:
        Set when the loop finishes.
    max_turns:
        Maximum conversation turns per prompt.

    Returns
    -------
    str
        Summary of the agent's last response, or error description.
    """
    from mcp_cli.chat.conversation import ConversationProcessor

    convo = ConversationProcessor(context, ui_manager)
    last_response = ""

    # Wire dashboard bridge if present
    if bridge := getattr(context, "dashboard_bridge", None):
        bridge.set_input_queue(input_queue)

    try:
        while not context.exit_requested:
            try:
                # Wait for a prompt (with timeout so we can check exit)
                try:
                    user_msg = await asyncio.wait_for(input_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if user_msg is None:
                    continue

                # Exit signals
                msg_lower = str(user_msg).lower().strip()
                if msg_lower in ("exit", "quit", "__stop__"):
                    break

                # Skip empty
                if not str(user_msg).strip():
                    continue

                # Add to conversation and process
                await context.add_user_message(str(user_msg))

                # Broadcast to dashboard if wired
                if dash := getattr(context, "dashboard_bridge", None):
                    try:
                        await dash.on_message("user", str(user_msg))
                    except Exception as exc:
                        logger.debug("Dashboard on_message error: %s", exc)

                await convo.process_conversation(max_turns=max_turns)

                # Capture last assistant response for summary
                for msg in reversed(context.conversation_history):
                    if hasattr(msg, "role") and msg.role == "assistant":
                        last_response = getattr(msg, "content", "") or ""
                        break
                    elif isinstance(msg, dict) and msg.get("role") == "assistant":
                        last_response = msg.get("content", "") or ""
                        break

            except asyncio.CancelledError:
                logger.info("Agent loop cancelled: %s", context.agent_id)
                break
            except Exception as exc:
                logger.exception("Error in agent loop: %s", exc)
                last_response = f"Error: {exc}"
                break

    finally:
        done_event.set()

    return last_response[:500] if last_response else "Agent completed."
