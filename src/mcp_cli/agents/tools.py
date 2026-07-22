# mcp_cli/agents/tools.py
"""Supervisor tool definitions and handler for agent orchestration.

Provides internal tools that the LLM can call to spawn, stop, and
communicate with other agents.  These tools are intercepted in
tool_processor.py before MCP routing (same pattern as plan tools).

Tools:
- agent_spawn: Spawn a new agent with a role and initial prompt
- agent_stop: Stop a running agent
- agent_message: Send a message to another agent
- agent_wait: Wait for an agent to finish
- agent_status: Query an agent's current status
- agent_list: List all managed agents
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp_cli.agents.manager import AgentManager

logger = logging.getLogger(__name__)

_AGENT_TOOL_NAMES = frozenset(
    {
        "agent_spawn",
        "agent_stop",
        "agent_message",
        "agent_wait",
        "agent_status",
        "agent_list",
    }
)


def get_agent_tools_as_dicts() -> list[dict[str, Any]]:
    """Return OpenAI-format tool definitions for agent orchestration."""
    return [
        {
            "type": "function",
            "function": {
                "name": "agent_spawn",
                "description": (
                    "Spawn a new agent to work on a sub-task in parallel. "
                    "The agent runs independently and can be monitored or stopped."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Human-readable name for the agent.",
                        },
                        "role": {
                            "type": "string",
                            "description": (
                                "Role description (e.g. 'researcher', 'coder')."
                            ),
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use (optional, defaults to current).",
                        },
                        "provider": {
                            "type": "string",
                            "description": "Provider to use (optional, defaults to current).",
                        },
                        "initial_prompt": {
                            "type": "string",
                            "description": "The task/prompt to give the new agent.",
                        },
                    },
                    "required": ["name", "initial_prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "agent_stop",
                "description": "Stop a running agent by its agent_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "The ID of the agent to stop.",
                        },
                    },
                    "required": ["agent_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "agent_message",
                "description": (
                    "Send a message to another agent. The message is injected "
                    "into that agent's input as a prompt."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "The target agent's ID.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Message content to send.",
                        },
                    },
                    "required": ["agent_id", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "agent_wait",
                "description": (
                    "Wait for an agent to finish its task. Returns the agent's "
                    "completion summary."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "The agent to wait for.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Max seconds to wait (default: 300).",
                        },
                    },
                    "required": ["agent_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "agent_status",
                "description": "Get the current status of a specific agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "The agent to query.",
                        },
                    },
                    "required": ["agent_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "agent_list",
                "description": "List all currently managed agents and their statuses.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]


async def handle_agent_tool(
    tool_name: str,
    arguments: dict[str, Any],
    agent_manager: AgentManager,
    caller_agent_id: str = "default",
) -> str:
    """Execute an agent orchestration tool.

    Parameters
    ----------
    tool_name:
        One of the ``_AGENT_TOOL_NAMES``.
    arguments:
        Tool arguments from the LLM.
    agent_manager:
        The AgentManager instance.
    caller_agent_id:
        The agent_id of the caller (for message attribution).

    Returns
    -------
    str
        JSON string with the result.
    """
    try:
        if tool_name == "agent_spawn":
            return await _handle_spawn(arguments, agent_manager, caller_agent_id)
        if tool_name == "agent_stop":
            return await _handle_stop(arguments, agent_manager)
        if tool_name == "agent_message":
            return await _handle_message(arguments, agent_manager, caller_agent_id)
        if tool_name == "agent_wait":
            return await _handle_wait(arguments, agent_manager)
        if tool_name == "agent_status":
            return _handle_status(arguments, agent_manager)
        if tool_name == "agent_list":
            return _handle_list(agent_manager)
        return json.dumps({"error": f"Unknown agent tool: {tool_name}"})
    except Exception as exc:
        logger.error("Agent tool %s failed: %s", tool_name, exc)
        return json.dumps({"error": str(exc)})


async def _handle_spawn(
    arguments: dict[str, Any],
    manager: AgentManager,
    caller_id: str,
) -> str:
    from mcp_cli.agents.config import AgentConfig

    name = arguments.get("name", "unnamed")
    # Generate agent_id from name
    agent_id = f"agent-{name.lower().replace(' ', '-')}"

    config = AgentConfig(
        agent_id=agent_id,
        name=name,
        role=arguments.get("role", ""),
        model=arguments.get("model"),
        provider=arguments.get("provider"),
        parent_agent_id=caller_id,
        initial_prompt=arguments.get("initial_prompt", ""),
    )

    result_id = await manager.spawn_agent(config)
    return json.dumps({"success": True, "agent_id": result_id, "name": name})


async def _handle_stop(arguments: dict[str, Any], manager: AgentManager) -> str:
    agent_id = arguments.get("agent_id", "")
    if not agent_id:
        return json.dumps({"error": "agent_id is required"})
    stopped = await manager.stop_agent(agent_id)
    return json.dumps({"success": stopped, "agent_id": agent_id})


async def _handle_message(
    arguments: dict[str, Any],
    manager: AgentManager,
    caller_id: str,
) -> str:
    agent_id = arguments.get("agent_id", "")
    content = arguments.get("content", "")
    if not agent_id or not content:
        return json.dumps({"error": "agent_id and content are required"})
    sent = await manager.send_message(caller_id, agent_id, content)
    return json.dumps({"success": sent, "to": agent_id})


async def _handle_wait(arguments: dict[str, Any], manager: AgentManager) -> str:
    agent_id = arguments.get("agent_id", "")
    if not agent_id:
        return json.dumps({"error": "agent_id is required"})
    timeout = float(arguments.get("timeout", 300))
    result = await manager.wait_agent(agent_id, timeout=timeout)
    return json.dumps(result, default=str)


def _handle_status(arguments: dict[str, Any], manager: AgentManager) -> str:
    agent_id = arguments.get("agent_id", "")
    if not agent_id:
        return json.dumps({"error": "agent_id is required"})
    status = manager.get_agent_status(agent_id)
    if status is None:
        return json.dumps({"error": f"Unknown agent: {agent_id}"})
    return json.dumps(status, default=str)


def _handle_list(manager: AgentManager) -> str:
    agents = manager.list_agents()
    return json.dumps({"agents": agents}, default=str)
