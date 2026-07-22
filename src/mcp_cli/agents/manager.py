# mcp_cli/agents/manager.py
"""AgentManager — orchestration hub for multi-agent scenarios.

Manages agent lifecycle (spawn, stop, wait), inter-agent messaging,
and shared artifacts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mcp_cli.agents.config import AgentConfig
from mcp_cli.chat.agent_tool_state import remove_agent_tool_state
from mcp_cli.dashboard.router import AgentDescriptor

if TYPE_CHECKING:
    from mcp_cli.dashboard.router import AgentRouter
    from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)

MAX_AGENTS = 10


@dataclass
class _AgentHandle:
    """Internal bookkeeping for a spawned agent."""

    agent_id: str
    config: AgentConfig
    context: Any  # ChatContext
    bridge: Any  # DashboardBridge | None
    task: asyncio.Task
    input_queue: asyncio.Queue
    done_event: asyncio.Event
    result_summary: str = ""
    created_at: float = field(default_factory=time.time)


class AgentManager:
    """Manages spawning, stopping, and communicating with agents."""

    def __init__(
        self,
        tool_manager: ToolManager,
        router: AgentRouter,
        model_manager: Any = None,
    ) -> None:
        self._tool_manager = tool_manager
        self._router = router
        self._model_manager = model_manager
        self._agents: dict[str, _AgentHandle] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}  # id → {agent_id, content, ...}
        self._message_queues: dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    async def spawn_agent(self, config: AgentConfig) -> str:
        """Spawn a new agent from *config*. Returns the agent_id."""
        if len(self._agents) >= MAX_AGENTS:
            raise RuntimeError(f"Maximum of {MAX_AGENTS} concurrent agents reached.")
        if config.agent_id in self._agents:
            raise ValueError(f"Agent {config.agent_id!r} already exists.")

        agent_id = config.agent_id
        registered = False  # track router registration for rollback

        # Lazy imports to avoid circular deps
        from mcp_cli.agents.headless_ui import HeadlessUIManager
        from mcp_cli.agents.loop import run_agent_loop
        from mcp_cli.chat.chat_context import ChatContext
        from mcp_cli.dashboard.bridge import DashboardBridge

        try:
            # Create agent's ChatContext
            ctx = ChatContext.create(
                tool_manager=self._tool_manager,
                provider=config.provider,
                model=config.model,
                model_manager=self._model_manager,
                agent_id=agent_id,
            )
            if not await ctx.initialize():
                raise RuntimeError(f"Failed to initialize context for {agent_id}")

            # Apply system prompt override
            if config.system_prompt:
                ctx._system_prompt = config.system_prompt
                ctx._system_prompt_dirty = True

            # Apply tool filtering
            if config.allowed_tools is not None:
                allowed = set(config.allowed_tools)
                ctx.openai_tools = [
                    t
                    for t in ctx.openai_tools
                    if t.get("function", {}).get("name", "") in allowed
                ]
            if config.denied_tools is not None:
                denied = set(config.denied_tools)
                ctx.openai_tools = [
                    t
                    for t in ctx.openai_tools
                    if t.get("function", {}).get("name", "") not in denied
                ]

            # Create bridge and register with router
            bridge = DashboardBridge(self._router, agent_id=agent_id)
            bridge.set_context(ctx)
            ctx.dashboard_bridge = bridge

            descriptor = AgentDescriptor(
                agent_id=agent_id,
                name=config.name or agent_id,
                role=config.role,
                model=config.model or "",
                parent_agent_id=config.parent_agent_id,
                tool_count=len(ctx.openai_tools),
            )
            self._router.register_agent(agent_id, bridge, descriptor=descriptor)
            registered = True

            # Create headless UI and input queue
            ui = HeadlessUIManager(agent_id=agent_id)
            input_queue: asyncio.Queue = asyncio.Queue()
            done_event = asyncio.Event()

            # Message queue for inter-agent messaging
            self._message_queues[agent_id] = asyncio.Queue()

            # Launch the agent loop
            task = asyncio.create_task(
                run_agent_loop(ctx, ui, input_queue, done_event),
                name=f"agent-loop-{agent_id}",
            )

            handle = _AgentHandle(
                agent_id=agent_id,
                config=config,
                context=ctx,
                bridge=bridge,
                task=task,
                input_queue=input_queue,
                done_event=done_event,
            )
            self._agents[agent_id] = handle

            # Inject initial prompt
            if config.initial_prompt:
                await input_queue.put(config.initial_prompt)

            logger.info("Spawned agent %s (role=%s)", agent_id, config.role)
            return agent_id

        except BaseException:
            # Rollback partial state so the system stays consistent
            if registered:
                self._router.unregister_agent(agent_id)
            self._message_queues.pop(agent_id, None)
            remove_agent_tool_state(agent_id)
            logger.error("Failed to spawn agent %s — rolled back", agent_id)
            raise

    async def stop_agent(self, agent_id: str) -> bool:
        """Stop an agent. Returns True if it existed."""
        handle = self._agents.pop(agent_id, None)
        if handle is None:
            return False

        # Cancel the task
        handle.task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(handle.task), timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        # Update status and unregister
        await self._router.update_agent_status(agent_id, "completed")
        self._router.unregister_agent(agent_id)

        # Cleanup tool state
        remove_agent_tool_state(agent_id)
        self._message_queues.pop(agent_id, None)

        logger.info("Stopped agent %s", agent_id)
        return True

    async def wait_agent(self, agent_id: str, timeout: float = 300) -> dict[str, Any]:
        """Wait for an agent to finish. Returns status dict."""
        handle = self._agents.get(agent_id)
        if handle is None:
            return {"error": f"Unknown agent: {agent_id}"}

        try:
            await asyncio.wait_for(handle.done_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return {
                "agent_id": agent_id,
                "status": "timeout",
                "summary": f"Agent did not finish within {timeout}s",
            }

        # Capture summary
        try:
            handle.result_summary = handle.task.result()
        except (asyncio.CancelledError, Exception) as exc:
            handle.result_summary = f"Error: {exc}"

        return {
            "agent_id": agent_id,
            "status": "completed",
            "summary": handle.result_summary,
        }

    # ------------------------------------------------------------------ #
    #  Messaging                                                           #
    # ------------------------------------------------------------------ #

    async def send_message(self, from_id: str, to_id: str, content: str) -> bool:
        """Send a message from one agent to another.

        The message is injected into the target agent's input queue
        as a system-annotated prompt.
        """
        handle = self._agents.get(to_id)
        if handle is None:
            return False

        annotated = f"[Message from {from_id}]: {content}"
        await handle.input_queue.put(annotated)
        logger.debug("Message from %s → %s: %.100s", from_id, to_id, content)
        return True

    async def get_messages(self, agent_id: str) -> list[dict[str, Any]]:
        """Drain pending inter-agent messages for *agent_id*.

        Returns list of ``{from_agent, content}`` dicts.
        """
        queue = self._message_queues.get(agent_id)
        if queue is None:
            return []
        messages = []
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    # ------------------------------------------------------------------ #
    #  Artifacts                                                           #
    # ------------------------------------------------------------------ #

    def publish_artifact(self, agent_id: str, artifact_id: str, content: Any) -> None:
        """Publish an artifact to the shared store."""
        self._artifacts[artifact_id] = {
            "agent_id": agent_id,
            "content": content,
            "created_at": time.time(),
        }
        logger.debug("Artifact published: %s by %s", artifact_id, agent_id)

    def get_artifact(self, artifact_id: str) -> Any | None:
        """Retrieve an artifact by ID, or None."""
        entry = self._artifacts.get(artifact_id)
        return entry["content"] if entry else None

    def list_artifacts(self) -> list[dict[str, Any]]:
        """List all published artifacts."""
        return [
            {"artifact_id": k, "agent_id": v["agent_id"]}
            for k, v in self._artifacts.items()
        ]

    # ------------------------------------------------------------------ #
    #  Status                                                              #
    # ------------------------------------------------------------------ #

    def get_agent_status(self, agent_id: str) -> dict[str, Any] | None:
        """Return status dict for a single agent, or None."""
        handle = self._agents.get(agent_id)
        if handle is None:
            return None
        return {
            "agent_id": handle.agent_id,
            "name": handle.config.name,
            "role": handle.config.role,
            "model": handle.config.model,
            "provider": handle.config.provider,
            "status": "active" if not handle.done_event.is_set() else "completed",
            "parent_agent_id": handle.config.parent_agent_id,
        }

    def list_agents(self) -> list[dict[str, Any]]:
        """Return status dicts for all managed agents."""
        return [self.get_agent_status(aid) for aid in self._agents]  # type: ignore[misc]

    def get_agent_snapshot(self, agent_id: str) -> dict[str, Any] | None:
        """Return config and context for an agent (used by group_store).

        Returns ``{config: AgentConfig, context: ChatContext}`` or None.
        """
        handle = self._agents.get(agent_id)
        if handle is None:
            return None
        return {"config": handle.config, "context": handle.context}

    # ------------------------------------------------------------------ #
    #  Cleanup                                                             #
    # ------------------------------------------------------------------ #

    async def stop_all(self) -> None:
        """Stop all managed agents."""
        agent_ids = list(self._agents.keys())
        for agent_id in agent_ids:
            await self.stop_agent(agent_id)
        logger.info("All agents stopped.")
