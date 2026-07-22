# mcp_cli/dashboard/router.py
"""AgentRouter — routes messages between browser clients and DashboardBridge instances.

In single-agent mode the router has exactly one bridge and behaves identically
to the legacy direct-wiring path.  In multi-agent mode it tracks per-client
focus, sends AGENT_LIST on connect, and handles FOCUS_AGENT switching.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from mcp_cli.dashboard.server import DashboardServer

if TYPE_CHECKING:
    from mcp_cli.dashboard.bridge import DashboardBridge

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Protocol helpers (mirror bridge.py)                                 #
# ------------------------------------------------------------------ #

_PROTOCOL = "mcp-dashboard"
_VERSION = 2


def _envelope(msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": _PROTOCOL,
        "version": _VERSION,
        "type": msg_type,
        "payload": payload,
    }


# ------------------------------------------------------------------ #
#  AgentDescriptor                                                     #
# ------------------------------------------------------------------ #


@dataclass
class AgentDescriptor:
    """Lightweight descriptor for an agent visible to the dashboard."""

    agent_id: str
    name: str
    role: str = ""
    status: Literal["active", "paused", "completed", "failed"] = "active"
    model: str = ""
    provider: str = ""
    session_id: str = ""
    parent_agent_id: str | None = None
    tool_count: int = 0
    message_count: int = 0
    created_at: str = ""


# ------------------------------------------------------------------ #
#  AgentRouter                                                         #
# ------------------------------------------------------------------ #


class AgentRouter:
    """Routes dashboard messages between the server and one or more bridges."""

    def __init__(self, server: DashboardServer) -> None:
        self.server = server
        self._bridges: dict[str, DashboardBridge] = {}
        self._agent_descriptors: dict[str, AgentDescriptor] = {}
        # Per-client focus: ws → agent_id
        self._client_focus: dict[Any, str] = {}
        # Per-client subscriptions: ws → set of agent_ids ("*" = all)
        self._client_subscriptions: dict[Any, set[str]] = {}

        # Own the three server callbacks
        server.on_browser_message = self._on_browser_message
        server.on_client_connected = self._on_client_connected
        server.on_client_disconnected = self._on_client_disconnected

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def has_clients(self) -> bool:
        """Proxy to server.has_clients."""
        return self.server.has_clients

    @property
    def _default_agent_id(self) -> str:
        """First registered bridge id, fallback 'default'."""
        if self._bridges:
            return next(iter(self._bridges))
        return "default"

    # ------------------------------------------------------------------ #
    #  Bridge registration                                                #
    # ------------------------------------------------------------------ #

    def register_agent(
        self,
        agent_id: str,
        bridge: DashboardBridge,
        descriptor: AgentDescriptor | None = None,
    ) -> None:
        """Register a bridge for the given agent_id."""
        self._bridges[agent_id] = bridge
        if descriptor is None:
            descriptor = AgentDescriptor(agent_id=agent_id, name=agent_id)
        self._agent_descriptors[agent_id] = descriptor
        logger.debug("AgentRouter: registered agent %s", agent_id)
        # Notify connected browsers
        if self.server.has_clients:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._broadcast_agent_registered(descriptor))
            except RuntimeError:
                pass  # no running loop yet

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister a bridge for the given agent_id."""
        self._bridges.pop(agent_id, None)
        self._agent_descriptors.pop(agent_id, None)
        # Re-focus clients that were focused on the removed agent
        default_id = self._default_agent_id
        for ws, focused_id in list(self._client_focus.items()):
            if focused_id == agent_id:
                self._client_focus[ws] = default_id
        logger.debug("AgentRouter: unregistered agent %s", agent_id)
        # Notify connected browsers
        if self.server.has_clients:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._broadcast_agent_unregistered(agent_id))
            except RuntimeError:
                pass

    # ------------------------------------------------------------------ #
    #  Agent status                                                       #
    # ------------------------------------------------------------------ #

    async def update_agent_status(self, agent_id: str, status: str) -> None:
        """Update an agent's status and broadcast AGENT_STATUS."""
        desc = self._agent_descriptors.get(agent_id)
        if desc:
            desc.status = status  # type: ignore[assignment]
            await self.broadcast_global(
                _envelope("AGENT_STATUS", {"agent_id": agent_id, "status": status})
            )

    # ------------------------------------------------------------------ #
    #  Outbound (bridge → browser)                                        #
    # ------------------------------------------------------------------ #

    async def broadcast_from_agent(
        self, agent_id: str, message: dict[str, Any]
    ) -> None:
        """Broadcast a message from a specific agent.

        If no clients have subscriptions configured, broadcasts to all
        (backward compat). Otherwise, only sends to clients subscribed
        to this agent_id (or to ``"*"``).
        """
        if not self._client_subscriptions:
            # No subscriptions configured — broadcast to all (default)
            await self.server.broadcast(message)
            return

        for ws, subs in list(self._client_subscriptions.items()):
            if "*" in subs or agent_id in subs:
                try:
                    await self.server.send_to_client(ws, message)
                except Exception:
                    pass  # client may have disconnected

    async def broadcast_global(self, message: dict[str, Any]) -> None:
        """Broadcast a message not scoped to any agent (e.g. AGENT_LIST)."""
        await self.server.broadcast(message)

    async def send_to_client(self, ws: Any, message: dict[str, Any]) -> None:
        """Send a message to a specific WebSocket client."""
        await self.server.send_to_client(ws, message)

    # ------------------------------------------------------------------ #
    #  Agent list helpers                                                  #
    # ------------------------------------------------------------------ #

    async def _send_agent_list(self, ws: Any) -> None:
        """Send AGENT_LIST to a specific client."""
        agents = [asdict(d) for d in self._agent_descriptors.values()]
        await self.server.send_to_client(
            ws, _envelope("AGENT_LIST", {"agents": agents})
        )

    async def _broadcast_agent_registered(self, desc: AgentDescriptor) -> None:
        await self.broadcast_global(_envelope("AGENT_REGISTERED", asdict(desc)))

    async def _broadcast_agent_unregistered(self, agent_id: str) -> None:
        await self.broadcast_global(
            _envelope("AGENT_UNREGISTERED", {"agent_id": agent_id})
        )

    # ------------------------------------------------------------------ #
    #  Focus management                                                    #
    # ------------------------------------------------------------------ #

    async def _handle_focus_agent(self, msg: dict[str, Any], ws: Any) -> None:
        """Browser client changed which agent it's viewing."""
        agent_id = msg.get("agent_id") or self._default_agent_id
        if ws is None:
            return
        self._client_focus[ws] = agent_id
        logger.debug("AgentRouter: client focused on %s", agent_id)
        # Replay focused agent's state to this client only
        bridge = self._bridges.get(agent_id)
        if bridge:
            await bridge._on_client_connected(ws)

    def _handle_subscribe(self, msg: dict[str, Any], ws: Any) -> None:
        """Handle a SUBSCRIBE message from a browser client."""
        if ws is None:
            return
        agents = msg.get("agents", [])
        is_global = msg.get("global", False)
        subs: set[str] = set(agents)
        if is_global:
            subs.add("*")
        self._client_subscriptions[ws] = subs
        logger.debug("AgentRouter: client subscribed to %s", subs)

    async def _handle_agent_message(self, msg: dict[str, Any]) -> None:
        """Route an AGENT_MESSAGE from browser to target agent's bridge."""
        to_agent = msg.get("to_agent")
        content = msg.get("content", "")
        from_agent = msg.get("from_agent", "browser")
        if not to_agent:
            return
        bridge = self._bridges.get(to_agent)
        if bridge:
            # Inject as a user message annotated with source
            await bridge._on_browser_message(
                {"type": "USER_MESSAGE", "content": f"[From {from_agent}]: {content}"}
            )

    # ------------------------------------------------------------------ #
    #  Inbound callbacks (browser → bridge)                               #
    # ------------------------------------------------------------------ #

    async def _on_browser_message(self, msg: dict[str, Any], ws: Any = None) -> None:
        """Route an inbound browser message to the appropriate bridge."""
        msg_type = msg.get("type")

        # Handle router-level message types
        if msg_type == "FOCUS_AGENT":
            await self._handle_focus_agent(msg, ws)
            return
        if msg_type == "REQUEST_AGENT_LIST":
            if ws is not None:
                await self._send_agent_list(ws)
            return
        if msg_type == "AGENT_MESSAGE":
            await self._handle_agent_message(msg)
            return
        if msg_type == "SUBSCRIBE":
            self._handle_subscribe(msg, ws)
            return

        # Route to bridge: explicit agent_id > client focus > sole bridge
        target_id = msg.get("agent_id")
        bridge: DashboardBridge | None = None

        if target_id and target_id in self._bridges:
            bridge = self._bridges[target_id]
        elif ws is not None and ws in self._client_focus:
            target_id = self._client_focus[ws]
            bridge = self._bridges.get(target_id)
        elif len(self._bridges) == 1:
            bridge = next(iter(self._bridges.values()))
        elif target_id:
            logger.debug(
                "AgentRouter: unknown agent_id %r in browser message", target_id
            )
            return
        else:
            logger.debug(
                "AgentRouter: no agent_id in message and %d bridges registered",
                len(self._bridges),
            )
            return

        if bridge is not None:
            await bridge._on_browser_message(msg)

    async def _on_client_connected(self, ws: Any) -> None:
        """Send agent list and replay focused agent state to a new client."""
        # Send AGENT_LIST first
        await self._send_agent_list(ws)
        # Default focus: first registered agent
        default_id = self._default_agent_id
        self._client_focus[ws] = default_id
        # Default subscription: all agents
        self._client_subscriptions[ws] = {"*"}
        # Replay only the focused agent's state
        bridge = self._bridges.get(default_id)
        if bridge:
            await bridge._on_client_connected(ws)

    async def _on_client_disconnected(self, ws: Any = None) -> None:
        """Notify all registered bridges and clean up per-client state."""
        for bridge in self._bridges.values():
            await bridge.on_client_disconnected()
        # Clean up per-client tracking for the disconnected ws
        if ws is not None:
            self._client_focus.pop(ws, None)
            self._client_subscriptions.pop(ws, None)
