# mcp_cli/dashboard/bridge.py
"""Dashboard bridge — the integration layer between mcp-cli's chat engine
and the browser dashboard.

The bridge is stored on ``ChatContext.dashboard_bridge``. All hooks are no-ops
when the bridge is ``None`` (i.e., ``--dashboard`` was not set), so there is
zero performance impact on normal usage.

Hook call sites:
  tool_processor._on_tool_result()    → bridge.on_tool_result()
  conversation.process_user_input()   → bridge.on_agent_state(), bridge.on_message()
  streaming_handler._process_chunk()  → bridge.on_token()
"""

from __future__ import annotations

import asyncio
import datetime
import json as _json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from mcp_cli.config.defaults import DEFAULT_AGENT_ID
from mcp_cli.dashboard.server import DashboardServer

if TYPE_CHECKING:
    from mcp_cli.chat.chat_context import ChatContext
    from mcp_cli.dashboard.router import AgentRouter

logger = logging.getLogger(__name__)

_PROTOCOL = "mcp-dashboard"
_VERSION = 2


def _envelope(msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": _PROTOCOL,
        "version": _VERSION,
        "type": msg_type,
        "payload": payload,
    }


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class DashboardBridge:
    """Routes chat-engine events to connected browser dashboard clients."""

    def __init__(
        self, server: DashboardServer | AgentRouter, agent_id: str = DEFAULT_AGENT_ID
    ) -> None:
        from mcp_cli.dashboard.router import AgentRouter

        self.agent_id = agent_id
        self._turn_number: int = 0
        self._ctx: ChatContext | None = None
        # Set this to inject user messages from the browser back into the chat engine
        self._user_message_callback: Callable[[str], None] | None = None
        # Queue for injecting browser messages into the chat loop
        self._input_queue: asyncio.Queue[str] | None = None
        # Callback to execute a tool requested by the browser (REQUEST_TOOL)
        self._tool_call_callback: (
            Callable[[str, dict[str, Any]], Awaitable[Any]] | None
        ) = None
        # View registry discovered from _meta.ui fields in tool results
        self._view_registry: list[dict[str, Any]] = []
        self._seen_view_ids: set[str] = set()
        # Pending tool approval futures keyed by call_id
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        # Running MCP Apps (for replay on client reconnect)
        self._running_apps: dict[str, dict[str, Any]] = {}

        # Dual-mode: router-managed vs direct-wiring
        if isinstance(server, AgentRouter):
            self._router: AgentRouter | None = server
            self.server: DashboardServer = server.server
            # Router owns the server callbacks — do NOT wire them here
        else:
            self._router = None
            self.server = server
            # Legacy direct-wiring path
            server.on_browser_message = self._on_browser_message
            server.on_client_connected = self._on_client_connected
            server.on_client_disconnected = self.on_client_disconnected

    def set_context(self, ctx: ChatContext) -> None:
        """Store a back-reference to ChatContext for history/config queries."""
        self._ctx = ctx

    async def on_shutdown(self) -> None:
        """Cancel all pending approval futures. Call before stopping server."""
        for call_id, fut in list(self._pending_approvals.items()):
            if not fut.done():
                fut.set_result(False)
        self._pending_approvals.clear()

    @property
    def has_clients(self) -> bool:
        """Whether any browser clients are connected."""
        if self._router is not None:
            return self._router.has_clients
        return self.server.has_clients

    async def _broadcast(self, envelope: dict[str, Any]) -> None:
        """Dispatch a broadcast through the router or directly to the server."""
        if self._router is not None:
            await self._router.broadcast_from_agent(self.agent_id, envelope)
        else:
            await self.server.broadcast(envelope)

    async def broadcast(self, envelope: dict[str, Any]) -> None:
        """Public broadcast method for external callers (e.g. commands)."""
        await self._broadcast(envelope)

    async def on_client_disconnected(self) -> None:
        """Called when a browser client disconnects.

        If no clients remain, cancel all pending approval futures so the
        tool processor doesn't hang waiting for a response that will never come.
        """
        if not self.has_clients:
            for call_id, fut in list(self._pending_approvals.items()):
                if not fut.done():
                    fut.set_result(False)
            self._pending_approvals.clear()

    # ------------------------------------------------------------------ #
    #  Outbound hooks (chat engine → browser)                            #
    # ------------------------------------------------------------------ #

    async def on_tool_result(
        self,
        tool_name: str,
        server_name: str,
        result: Any,
        success: bool,
        error: str | None = None,
        duration_ms: int | None = None,
        meta_ui: Any = None,
        call_id: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Called after every tool execution completes."""
        payload: dict[str, Any] = {
            "tool_name": tool_name,
            "server_name": server_name,
            "agent_id": self.agent_id,
            "call_id": call_id or "",
            "timestamp": _now(),
            "duration_ms": duration_ms,
            "result": self._serialise(result),
            "error": error,
            "success": success,
            "arguments": self._serialise(arguments) if arguments else None,
        }
        if meta_ui is not None:
            payload["meta_ui"] = self._serialise(meta_ui)
            # Discover new views declared in _meta.ui before broadcasting
            if isinstance(meta_ui, dict) and meta_ui.get("view"):
                await self._discover_view(meta_ui, server_name)
        await self._broadcast(_envelope("TOOL_RESULT", payload))

    async def on_agent_state(
        self,
        status: Literal["thinking", "tool_calling", "idle"],
        current_tool: str | None = None,
        turn_number: int | None = None,
        tokens_used: int = 0,
    ) -> None:
        """Called when the agent's status changes."""
        if turn_number is not None:
            self._turn_number = turn_number
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "status": status,
            "current_tool": current_tool,
            "turn_number": self._turn_number,
            "tokens_used": tokens_used,
            "budget_remaining": None,
        }
        await self._broadcast(_envelope("AGENT_STATE", payload))

    async def on_message(
        self,
        role: Literal["user", "assistant", "tool"],
        content: str,
        streaming: bool = False,
        reasoning: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Called when a complete conversation message is emitted."""
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "role": role,
            "content": content,
            "streaming": streaming,
            "tool_calls": tool_calls,
            "reasoning": reasoning,
            "attachments": attachments,
        }
        await self._broadcast(_envelope("CONVERSATION_MESSAGE", payload))

    async def on_token(self, token: str, done: bool = False) -> None:
        """Called for each streamed LLM token (high-volume — only used by agent terminal)."""
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "token": token,
            "done": done,
        }
        await self._broadcast(_envelope("CONVERSATION_TOKEN", payload))

    async def on_view_registry_update(self, views: list[dict[str, Any]]) -> None:
        """Called when the set of available views changes (server connect/disconnect)."""
        await self._broadcast(
            _envelope("VIEW_REGISTRY", {"agent_id": self.agent_id, "views": views})
        )

    async def on_app_launched(self, app_info: Any) -> None:
        """Notify dashboard that an MCP App launched — embed as panel."""
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "tool_name": app_info.tool_name,
            "url": app_info.url,
            "port": app_info.port,
            "server_name": app_info.server_name,
            "resource_uri": app_info.resource_uri,
            "state": (
                app_info.state.value
                if hasattr(app_info.state, "value")
                else str(app_info.state)
            ),
            "timestamp": _now(),
        }
        # Key by resource_uri when available so different tools sharing
        # the same resource replace each other rather than accumulating.
        app_key = app_info.resource_uri or app_info.tool_name
        self._running_apps[app_key] = payload
        await self._broadcast(_envelope("APP_LAUNCHED", payload))

    async def on_app_closed(self, tool_name: str) -> None:
        """Notify dashboard that an MCP App closed."""
        # Try removing by tool_name first, then scan by resource_uri
        if tool_name not in self._running_apps:
            to_remove = [
                k
                for k, v in self._running_apps.items()
                if v.get("tool_name") == tool_name
            ]
            for k in to_remove:
                self._running_apps.pop(k, None)
        else:
            self._running_apps.pop(tool_name, None)
        await self._broadcast(
            _envelope(
                "APP_CLOSED",
                {
                    "agent_id": self.agent_id,
                    "tool_name": tool_name,
                    "timestamp": _now(),
                },
            )
        )

    async def _discover_view(self, meta_ui: dict[str, Any], server_name: str) -> None:
        """Register a new view from a _meta.ui block and broadcast VIEW_REGISTRY."""
        view_id: str = meta_ui["view"]
        if view_id in self._seen_view_ids:
            return
        self._seen_view_ids.add(view_id)
        entry: dict[str, Any] = {
            "id": view_id,
            "name": meta_ui.get("name") or view_id.replace(":", " ").title(),
            "icon": meta_ui.get("icon") or "◻",
            "source": server_name,
            "type": meta_ui.get("type") or "tool",
            "url": meta_ui.get("url") or f"/views/{view_id}.html",
        }
        self._view_registry.append(entry)
        logger.debug(
            "Dashboard: discovered new view %s from server %s", view_id, server_name
        )
        await self.on_view_registry_update(self._view_registry)

    async def _on_client_connected(self, ws: Any) -> None:
        """Send current state to a newly connected browser client."""
        try:
            # VIEW_REGISTRY
            if self._view_registry:
                await ws.send(
                    _json.dumps(
                        _envelope(
                            "VIEW_REGISTRY",
                            {"agent_id": self.agent_id, "views": self._view_registry},
                        )
                    )
                )
            # CONFIG_STATE (model, provider, servers, system prompt preview)
            config = self._build_config_state()
            if config:
                await ws.send(_json.dumps(_envelope("CONFIG_STATE", config)))
            # TOOL_REGISTRY
            tools = await self._build_tool_registry()
            if tools is not None:
                await ws.send(_json.dumps(_envelope("TOOL_REGISTRY", {"tools": tools})))
            # CONVERSATION_HISTORY replay (chat view)
            history = self._build_conversation_history()
            if history:
                await ws.send(
                    _json.dumps(
                        _envelope("CONVERSATION_HISTORY", {"messages": history})
                    )
                )
            # ACTIVITY_HISTORY replay (activity stream)
            activity = self._build_activity_history()
            if activity:
                await ws.send(
                    _json.dumps(_envelope("ACTIVITY_HISTORY", {"events": activity}))
                )
            # APP replay for reconnecting clients
            for app_payload in self._running_apps.values():
                await ws.send(_json.dumps(_envelope("APP_LAUNCHED", app_payload)))
        except Exception as exc:
            logger.debug("Error sending initial state to new client: %s", exc)

    # ------------------------------------------------------------------ #
    #  Inbound messages (browser → chat engine)                          #
    # ------------------------------------------------------------------ #

    def set_input_queue(self, queue: asyncio.Queue[str]) -> None:
        """Register the asyncio.Queue that the chat loop reads from.

        Browser messages (USER_MESSAGE / USER_COMMAND) are put on this queue
        so the chat loop picks them up alongside terminal input.
        """
        self._input_queue = queue

    def set_tool_call_callback(
        self, fn: Callable[[str, dict[str, Any]], Awaitable[Any]]
    ) -> None:
        """Register a callback to execute a browser-requested tool call.

        The callback receives (tool_name, arguments) and should execute the
        tool.  It is responsible for broadcasting the result back (e.g. by
        calling ``on_tool_result()`` internally) or returning the result.
        """
        self._tool_call_callback = fn

    async def _on_browser_message(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")
        if msg_type in ("USER_MESSAGE", "USER_COMMAND"):
            content = msg.get("content") or msg.get("command", "")
            files = msg.get("files")  # list of {name, data, mime_type}

            # Stage browser-uploaded files on ChatContext
            if files and self._ctx is not None:
                from mcp_cli.chat.attachments import process_browser_file

                for f in files:
                    try:
                        att = process_browser_file(f["name"], f["data"], f["mime_type"])
                        self._ctx.attachment_staging.stage(att)
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "Bad browser file %s: %s", f.get("name", "?"), exc
                        )

            # If files attached but no text, queue a space so the chat loop iterates
            if not content and files:
                content = " "

            if content and self._input_queue is not None:
                try:
                    await self._input_queue.put(content)
                except Exception as exc:
                    logger.warning("Error queuing browser message: %s", exc)
            elif content:
                logger.debug(
                    "Dashboard received %s but no input queue registered", msg_type
                )
        elif msg_type == "REQUEST_TOOL":
            tool_name = msg.get("tool_name") or msg.get("tool") or ""
            arguments = msg.get("arguments") or msg.get("args") or {}
            call_id = msg.get("call_id") or ""
            if tool_name and self._tool_call_callback is not None:
                try:
                    await self._tool_call_callback(tool_name, arguments)
                except Exception as exc:
                    logger.warning(
                        "Error executing REQUEST_TOOL %s: %s", tool_name, exc
                    )
                    await self.on_tool_result(
                        tool_name=tool_name,
                        server_name="",
                        result=None,
                        success=False,
                        error=str(exc),
                        call_id=call_id,
                    )
            else:
                logger.debug(
                    "Dashboard REQUEST_TOOL: %s (no callback registered)",
                    tool_name or "(missing tool_name)",
                )
        elif msg_type == "USER_ACTION":
            action = msg.get("action") or ""
            content = msg.get("content") or ""
            text = content or (f"/{action}" if action else "")
            if text and self._input_queue is not None:
                try:
                    await self._input_queue.put(text)
                except Exception as exc:
                    logger.warning("Error queuing USER_ACTION: %s", exc)
            else:
                logger.debug("Dashboard USER_ACTION not routed: %s", msg)
        elif msg_type == "REQUEST_CONFIG":
            await self._handle_request_config()
        elif msg_type == "REQUEST_TOOLS":
            await self._handle_request_tools()
        elif msg_type == "SWITCH_MODEL":
            await self._handle_switch_model(msg)
        elif msg_type == "UPDATE_SYSTEM_PROMPT":
            await self._handle_update_system_prompt(msg)
        elif msg_type == "TOOL_APPROVAL_RESPONSE":
            await self._handle_tool_approval_response(msg)
        elif msg_type == "CLEAR_HISTORY":
            await self._handle_clear_history()
        elif msg_type == "NEW_SESSION":
            await self._handle_new_session(msg)
        elif msg_type == "REQUEST_SESSIONS":
            await self._handle_request_sessions()
        elif msg_type == "SWITCH_SESSION":
            await self._handle_switch_session(msg)
        elif msg_type == "DELETE_SESSION":
            await self._handle_delete_session(msg)
        elif msg_type == "RENAME_SESSION":
            await self._handle_rename_session(msg)
        elif msg_type == "REQUEST_APP_LIST":
            for app_payload in self._running_apps.values():
                await self._broadcast(_envelope("APP_LAUNCHED", app_payload))
        else:
            logger.debug("Dashboard received unknown message type: %s", msg_type)

    # ------------------------------------------------------------------ #
    #  Config / Model / System-prompt handlers                            #
    # ------------------------------------------------------------------ #

    async def _handle_request_config(self) -> None:
        """Browser requested current config — broadcast CONFIG_STATE."""
        config = self._build_config_state()
        if config:
            await self._broadcast(_envelope("CONFIG_STATE", config))

    async def _handle_switch_model(self, msg: dict[str, Any]) -> None:
        """Browser requested a model switch."""
        ctx = self._ctx
        if not ctx:
            logger.debug("SWITCH_MODEL: no context set")
            return
        provider = msg.get("provider") or ""
        model = msg.get("model") or ""
        if not provider or not model:
            logger.debug("SWITCH_MODEL: missing provider or model")
            return
        try:
            ctx.model_manager.switch_model(provider, model)
            await ctx.refresh_after_model_change()
            logger.info("Dashboard: switched to %s/%s", provider, model)
        except Exception as exc:
            logger.warning("Dashboard SWITCH_MODEL failed: %s", exc)
        # Broadcast updated state
        config = self._build_config_state()
        if config:
            await self._broadcast(_envelope("CONFIG_STATE", config))

    async def _handle_update_system_prompt(self, msg: dict[str, Any]) -> None:
        """Browser updated the system prompt."""
        ctx = self._ctx
        if not ctx:
            logger.debug("UPDATE_SYSTEM_PROMPT: no context set")
            return
        new_prompt = msg.get("system_prompt")
        if new_prompt is None:
            return
        try:
            if new_prompt == "":
                # Empty string → regenerate default prompt
                await ctx.regenerate_system_prompt()
            else:
                ctx._system_prompt = new_prompt
                await ctx.session.update_system_prompt(new_prompt)
            logger.info(
                "Dashboard: system prompt updated (%d chars)", len(ctx._system_prompt)
            )
        except Exception as exc:
            logger.warning("Dashboard UPDATE_SYSTEM_PROMPT failed: %s", exc)
        config = self._build_config_state()
        if config:
            await self._broadcast(_envelope("CONFIG_STATE", config))

    # ------------------------------------------------------------------ #
    #  Clear history handler                                              #
    # ------------------------------------------------------------------ #

    async def _handle_clear_history(self) -> None:
        """Browser requested conversation history clear."""
        ctx = self._ctx
        if not ctx:
            logger.debug("CLEAR_HISTORY: no context set")
            return
        try:
            await ctx.clear_conversation_history(keep_system_prompt=True)
            logger.info("Dashboard: conversation history cleared")
        except Exception as exc:
            logger.warning("Dashboard CLEAR_HISTORY failed: %s", exc)
        # Broadcast empty history + clear activity stream
        await self._broadcast(
            _envelope(
                "CONVERSATION_HISTORY", {"agent_id": self.agent_id, "messages": []}
            )
        )
        await self._broadcast(
            _envelope("ACTIVITY_HISTORY", {"agent_id": self.agent_id, "events": []})
        )
        config = self._build_config_state()
        if config:
            await self._broadcast(_envelope("CONFIG_STATE", config))

    # ------------------------------------------------------------------ #
    #  New session handler                                                #
    # ------------------------------------------------------------------ #

    async def _handle_new_session(self, msg: dict[str, Any]) -> None:
        """Browser requested a new session (save current + start fresh)."""
        ctx = self._ctx
        if not ctx:
            logger.debug("NEW_SESSION: no context set")
            return
        description = msg.get("description", "")

        # Save current session first (if there's history)
        if hasattr(ctx, "save_session") and ctx.conversation_history:
            try:
                path = ctx.save_session()
                if path:
                    logger.info("Dashboard: previous session saved: %s", path)
            except Exception as exc:
                logger.warning("Dashboard: could not save previous session: %s", exc)

        # Clear history and start fresh
        try:
            await ctx.clear_conversation_history(keep_system_prompt=True)
            logger.info("Dashboard: new session started: %s", ctx.session_id)
        except Exception as exc:
            logger.warning("Dashboard NEW_SESSION failed: %s", exc)
            return

        # Broadcast fresh state to all clients + clear activity stream
        await self._broadcast(
            _envelope(
                "CONVERSATION_HISTORY", {"agent_id": self.agent_id, "messages": []}
            )
        )
        await self._broadcast(
            _envelope("ACTIVITY_HISTORY", {"agent_id": self.agent_id, "events": []})
        )
        config = self._build_config_state()
        if config:
            await self._broadcast(_envelope("CONFIG_STATE", config))
        await self._broadcast(
            _envelope(
                "SESSION_STATE",
                {
                    "agent_id": self.agent_id,
                    "session_id": ctx.session_id,
                    "description": description,
                },
            )
        )

    # ------------------------------------------------------------------ #
    #  Session management handlers                                         #
    # ------------------------------------------------------------------ #

    async def _handle_request_sessions(self) -> None:
        """Browser requested list of saved sessions."""
        ctx = self._ctx
        if not ctx:
            return
        try:
            store = getattr(ctx, "_session_store", None)
            if not store:
                return
            sessions = store.list_sessions()
            payload = {
                "agent_id": self.agent_id,
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "created_at": s.created_at,
                        "updated_at": s.updated_at,
                        "provider": s.provider,
                        "model": s.model,
                        "message_count": s.message_count,
                        "description": s.description,
                        "is_current": s.session_id == ctx.session_id,
                    }
                    for s in sessions
                ],
                "current_session_id": ctx.session_id,
            }
            await self._broadcast(_envelope("SESSION_LIST", payload))
        except Exception as exc:
            logger.warning("Error building session list: %s", exc)

    async def _handle_switch_session(self, msg: dict[str, Any]) -> None:
        """Browser requested to switch to a different session."""
        ctx = self._ctx
        if not ctx:
            return
        session_id = msg.get("session_id", "")
        if not session_id:
            return

        # Save current session first
        if ctx.conversation_history:
            try:
                ctx.save_session()
            except Exception as exc:
                logger.warning("Failed to save current session before switch: %s", exc)

        # Clear and load the target session
        try:
            await ctx.clear_conversation_history(keep_system_prompt=True)
            loaded = ctx.load_session(session_id)
            if not loaded:
                logger.warning("Failed to load session %s", session_id)
                return
            # Override session_id to match loaded session
            ctx.session_id = session_id
            logger.info("Dashboard: switched to session %s", session_id)
        except Exception as exc:
            logger.warning("Dashboard SWITCH_SESSION failed: %s", exc)
            return

        # Broadcast updated state
        history = self._build_conversation_history()
        await self._broadcast(
            _envelope("CONVERSATION_HISTORY", {"messages": history or []})
        )
        # Activity stream replay for loaded session
        activity = self._build_activity_history()
        await self._broadcast(_envelope("ACTIVITY_HISTORY", {"events": activity or []}))
        config = self._build_config_state()
        if config:
            await self._broadcast(_envelope("CONFIG_STATE", config))
        await self._broadcast(
            _envelope(
                "SESSION_STATE",
                {"agent_id": self.agent_id, "session_id": session_id},
            )
        )
        # Refresh session list
        await self._handle_request_sessions()

    async def _handle_delete_session(self, msg: dict[str, Any]) -> None:
        """Browser requested to delete a saved session."""
        ctx = self._ctx
        if not ctx:
            return
        session_id = msg.get("session_id", "")
        if not session_id:
            return
        # Don't allow deleting the current session
        if session_id == ctx.session_id:
            logger.debug("Cannot delete the current active session")
            return
        try:
            store = getattr(ctx, "_session_store", None)
            if store:
                store.delete(session_id)
                logger.info("Dashboard: deleted session %s", session_id)
        except Exception as exc:
            logger.warning("Dashboard DELETE_SESSION failed: %s", exc)
        # Refresh session list
        await self._handle_request_sessions()

    async def _handle_rename_session(self, msg: dict[str, Any]) -> None:
        """Browser requested to rename/describe a session."""
        ctx = self._ctx
        if not ctx:
            return
        session_id = msg.get("session_id", "")
        description = msg.get("description", "")
        if not session_id:
            return
        try:
            store = getattr(ctx, "_session_store", None)
            if not store:
                return
            data = store.load(session_id)
            if data:
                data.metadata.description = description
                store.save(data)
                logger.info(
                    "Dashboard: renamed session %s → %s", session_id, description
                )
        except Exception as exc:
            logger.warning("Dashboard RENAME_SESSION failed: %s", exc)
        # Refresh session list
        await self._handle_request_sessions()

    # ------------------------------------------------------------------ #
    #  Tool registry handler                                              #
    # ------------------------------------------------------------------ #

    async def _handle_request_tools(self) -> None:
        """Browser requested current tool list — broadcast TOOL_REGISTRY."""
        tools = await self._build_tool_registry()
        if tools is not None:
            await self._broadcast(_envelope("TOOL_REGISTRY", {"tools": tools}))

    async def _build_tool_registry(self) -> list[dict[str, Any]] | None:
        """Build tool registry payload from ChatContext's tool_manager."""
        ctx = self._ctx
        if not ctx:
            return None
        try:
            tm = getattr(ctx, "tool_manager", None)
            if not tm:
                return None
            all_tools = await tm.get_all_tools()
            return [
                {
                    "name": t.name,
                    "namespace": t.namespace,
                    "description": t.description,
                    "parameters": t.parameters,
                    "is_async": t.is_async,
                    "tags": t.tags,
                    "supports_streaming": t.supports_streaming,
                }
                for t in all_tools
            ]
        except Exception as exc:
            logger.debug("Error building tool registry: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    #  Tool approval (dashboard ↔ tool_processor)                         #
    # ------------------------------------------------------------------ #

    async def request_tool_approval(
        self,
        tool_name: str,
        arguments: Any,
        call_id: str,
    ) -> asyncio.Future[bool]:
        """Send a tool approval request to the dashboard.

        The browser will show an approve/deny dialog and respond with
        TOOL_APPROVAL_RESPONSE.  Returns a Future that resolves to True/False.
        """
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "tool_name": tool_name,
            "arguments": self._serialise(arguments),
            "call_id": call_id,
            "timestamp": _now(),
        }
        # Create a future that the tool processor can await
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending_approvals[call_id] = fut
        await self._broadcast(_envelope("TOOL_APPROVAL_REQUEST", payload))

        # Auto-deny after 5 minutes if no response (prevents hanging forever)
        def _timeout() -> None:
            if not fut.done():
                logger.warning("Tool approval for %s timed out after 5m", tool_name)
                fut.set_result(False)
                self._pending_approvals.pop(call_id, None)

        loop.call_later(300, _timeout)
        return fut

    async def _handle_tool_approval_response(self, msg: dict[str, Any]) -> None:
        """Handle approval/denial response from the dashboard."""
        call_id = msg.get("call_id", "")
        approved = msg.get("approved", False)
        fut = self._pending_approvals.pop(call_id, None)
        if fut and not fut.done():
            fut.set_result(approved)

    # ------------------------------------------------------------------ #
    #  Plan updates (chat engine → browser)                               #
    # ------------------------------------------------------------------ #

    async def on_plan_update(
        self,
        plan_id: str,
        title: str,
        steps: list[dict[str, Any]],
        status: str = "running",
        current_step: int | None = None,
        error: str | None = None,
    ) -> None:
        """Broadcast a plan update to the dashboard."""
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "plan_id": plan_id,
            "title": title,
            "steps": steps,
            "status": status,
            "current_step": current_step,
            "error": error,
            "timestamp": _now(),
        }
        await self._broadcast(_envelope("PLAN_UPDATE", payload))

    # ------------------------------------------------------------------ #
    #  State builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_config_state(self) -> dict[str, Any] | None:
        """Build CONFIG_STATE payload from current ChatContext."""
        ctx = self._ctx
        if not ctx:
            return None
        try:
            provider = ctx.provider
            model = ctx.model
            mm = ctx.model_manager

            # Build provider → models map
            available_providers: list[dict[str, Any]] = []
            for pname in mm.get_available_providers():
                try:
                    models = mm.get_available_models(pname)
                except Exception:
                    models = []
                available_providers.append({"name": pname, "models": models})

            # Server info
            servers: list[dict[str, Any]] = []
            for si in getattr(ctx, "server_info", []) or []:
                servers.append(
                    {
                        "id": si.id,
                        "name": si.name,
                        "namespace": si.namespace,
                        "status": si.status,
                        "tool_count": si.tool_count,
                        "connected": si.connected,
                        "transport": str(si.transport),
                    }
                )

            # System prompt preview (first 500 chars)
            sys_prompt = getattr(ctx, "_system_prompt", "") or ""

            return {
                "agent_id": self.agent_id,
                "provider": provider,
                "model": model,
                "available_providers": available_providers,
                "servers": servers,
                "system_prompt": sys_prompt,
            }
        except Exception as exc:
            logger.debug("Error building CONFIG_STATE: %s", exc)
            return None

    @staticmethod
    def _content_block_to_descriptor(block: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw content block dict into a dashboard-safe descriptor.

        Used during conversation history replay when we have raw content
        blocks but not ``Attachment`` objects.
        """
        from mcp_cli.config.defaults import (
            DEFAULT_DASHBOARD_INLINE_IMAGE_THRESHOLD,
            DEFAULT_DASHBOARD_TEXT_PREVIEW_CHARS,
        )

        btype = block.get("type", "")
        desc: dict[str, Any] = {"kind": "unknown", "display_name": "", "size_bytes": 0}

        if btype == "image_url":
            url = block.get("image_url", {}).get("url", "")
            desc["kind"] = "image"
            desc["mime_type"] = "image/unknown"
            if url.startswith("http"):
                desc["preview_url"] = url
                desc["display_name"] = url.rsplit("/", 1)[-1][:60]
            elif url.startswith("data:"):
                # Estimate raw size from data URI length (base64 ≈ 4/3 of raw)
                data_part = url.split(",", 1)[-1] if "," in url else ""
                est_size = len(data_part) * 3 // 4
                if est_size <= DEFAULT_DASHBOARD_INLINE_IMAGE_THRESHOLD:
                    desc["preview_url"] = url
                else:
                    desc["preview_url"] = None
                desc["size_bytes"] = est_size
                desc["display_name"] = "image"
            else:
                desc["preview_url"] = None

        elif btype == "text":
            text = block.get("text", "")
            desc["kind"] = "text"
            desc["mime_type"] = "text/plain"
            desc["text_preview"] = text[:DEFAULT_DASHBOARD_TEXT_PREVIEW_CHARS]
            desc["text_truncated"] = len(text) > DEFAULT_DASHBOARD_TEXT_PREVIEW_CHARS
            # Try to extract filename from "--- filename ---" wrapper
            if text.startswith("--- ") and " ---\n" in text:
                name = text[4 : text.index(" ---\n")]
                desc["display_name"] = name

        elif btype == "input_audio":
            audio = block.get("input_audio", {})
            fmt = audio.get("format", "mp3")
            data = audio.get("data", "")
            est_size = len(data) * 3 // 4
            desc["kind"] = "audio"
            desc["mime_type"] = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
            desc["size_bytes"] = est_size
            desc["display_name"] = f"audio.{fmt}"
            if est_size <= DEFAULT_DASHBOARD_INLINE_IMAGE_THRESHOLD:
                mime = desc["mime_type"]
                desc["audio_data_uri"] = f"data:{mime};base64,{data}"
            else:
                desc["audio_data_uri"] = None

        return desc

    def _build_conversation_history(self) -> list[dict[str, Any]] | None:
        """Build conversation history payload from ChatContext.

        Returns only user/assistant messages — tool-role messages are excluded
        from the chat view and instead served via ``_build_activity_history()``.
        """
        ctx = self._ctx
        if not ctx:
            return None
        try:
            messages: list[dict[str, Any]] = []
            for msg in ctx.conversation_history:
                d = msg.to_dict()
                role = d.get("role", "")
                # Skip system and tool messages — system isn't shown,
                # tool results belong in the activity stream, not chat
                if role in ("system", "tool"):
                    continue

                raw_content = d.get("content")
                attachments = None

                # Handle multimodal content blocks
                if isinstance(raw_content, list):
                    text_parts: list[str] = []
                    att_descriptors: list[dict[str, Any]] = []
                    for block in raw_content:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        else:
                            att_descriptors.append(
                                self._content_block_to_descriptor(block)
                            )
                    content_str = "\n".join(text_parts)
                    if att_descriptors:
                        attachments = att_descriptors
                else:
                    content_str = raw_content or ""

                messages.append(
                    {
                        "role": role,
                        "content": content_str,
                        "tool_calls": d.get("tool_calls"),
                        "reasoning": d.get("reasoning_content"),
                        "attachments": attachments,
                    }
                )
            return messages if messages else None
        except Exception as exc:
            logger.debug("Error building conversation history: %s", exc)
            return None

    def _build_activity_history(self) -> list[dict[str, Any]] | None:
        """Build activity stream events from conversation history.

        Pairs assistant ``tool_calls`` with their corresponding ``tool``-role
        result messages to synthesize TOOL_RESULT-like payloads for the
        activity stream.  Also includes assistant messages that carry
        reasoning or tool_calls (for the "Calling …" cards).
        """
        ctx = self._ctx
        if not ctx:
            return None
        try:
            raw = [m.to_dict() for m in ctx.conversation_history]

            # Index tool-role results by tool_call_id for fast lookup
            tool_results: dict[str, dict[str, Any]] = {}
            for d in raw:
                if d.get("role") == "tool" and d.get("tool_call_id"):
                    tool_results[d["tool_call_id"]] = d

            events: list[dict[str, Any]] = []
            for d in raw:
                role = d.get("role", "")

                # User messages with attachments (multimodal content)
                if role == "user":
                    raw_content = d.get("content")
                    if isinstance(raw_content, list):
                        text_parts: list[str] = []
                        att_descs: list[dict[str, Any]] = []
                        for block in raw_content:
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            else:
                                att_descs.append(
                                    self._content_block_to_descriptor(block)
                                )
                        if att_descs:
                            events.append(
                                {
                                    "type": "CONVERSATION_MESSAGE",
                                    "payload": {
                                        "role": "user",
                                        "content": "\n".join(text_parts),
                                        "attachments": att_descs,
                                    },
                                }
                            )

                if role == "assistant":
                    tool_calls = d.get("tool_calls") or []
                    reasoning = d.get("reasoning_content")

                    # Emit a CONVERSATION_MESSAGE event for reasoning / tool_calls
                    if reasoning or tool_calls:
                        events.append(
                            {
                                "type": "CONVERSATION_MESSAGE",
                                "payload": {
                                    "role": "assistant",
                                    "content": d.get("content") or "",
                                    "tool_calls": tool_calls or None,
                                    "reasoning": reasoning,
                                },
                            }
                        )

                    # For each tool_call, try to pair with its result
                    for tc in tool_calls:
                        call_id = tc.get("id", "")
                        fn = tc.get("function") or {}
                        tool_name = fn.get("name", "")
                        args_str = fn.get("arguments", "")
                        try:
                            arguments = (
                                _json.loads(args_str)
                                if isinstance(args_str, str) and args_str
                                else args_str
                            )
                        except _json.JSONDecodeError:
                            arguments = args_str

                        # Look up the matching tool result
                        tr = tool_results.get(call_id)
                        result_content = (tr.get("content") or "") if tr else None

                        events.append(
                            {
                                "type": "TOOL_RESULT",
                                "payload": {
                                    "tool_name": tool_name,
                                    "server_name": "",
                                    "agent_id": self.agent_id,
                                    "call_id": call_id,
                                    "timestamp": None,
                                    "duration_ms": None,
                                    "result": result_content,
                                    "error": None,
                                    "success": True,
                                    "arguments": (
                                        self._serialise(arguments)
                                        if arguments
                                        else None
                                    ),
                                },
                            }
                        )

            return events if events else None
        except Exception as exc:
            logger.debug("Error building activity history: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    _SERIALISE_MAX_DEPTH = 20

    @staticmethod
    def _serialise(value: Any, _depth: int = 0) -> Any:
        """Convert a value to a JSON-safe representation."""
        if _depth > DashboardBridge._SERIALISE_MAX_DEPTH:
            return "<max depth exceeded>"
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [DashboardBridge._serialise(v, _depth + 1) for v in value]
        if isinstance(value, dict):
            return {
                k: DashboardBridge._serialise(v, _depth + 1) for k, v in value.items()
            }
        # Objects with a to_dict / model_dump / __dict__
        if hasattr(value, "to_dict"):
            try:
                return DashboardBridge._serialise(value.to_dict(), _depth + 1)
            except Exception as exc:
                logger.debug(
                    "_serialise to_dict() failed for %s: %s", type(value).__name__, exc
                )
        if hasattr(value, "model_dump"):
            try:
                return DashboardBridge._serialise(value.model_dump(), _depth + 1)
            except Exception as exc:
                logger.debug(
                    "_serialise model_dump() failed for %s: %s",
                    type(value).__name__,
                    exc,
                )
        return str(value)
