# mcp_cli/apps/bridge.py
"""MCP Apps bridge — handles protocol between browser WebSocket and MCP servers.

This is the Python-side protocol handler.  It receives JSON-RPC messages
from the browser host page via WebSocket and routes them to the
appropriate MCP server via ToolManager.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from typing import Any, TYPE_CHECKING

from mcp_cli.apps.models import AppInfo, AppState
from mcp_cli.config.defaults import DEFAULT_APP_TOOL_TIMEOUT

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)

# MCP spec: tool names use A-Z, a-z, 0-9, underscore, hyphen, dot.
_VALID_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_\-./]+$")


class AppBridge:
    """Bridges WebSocket messages from browser to MCP server tool calls."""

    def __init__(self, app_info: AppInfo, tool_manager: ToolManager) -> None:
        self.app_info = app_info
        self.tool_manager = tool_manager
        self._ws: Any = None
        self._model_context: dict[str, Any] | None = None
        self._pending_notifications: deque[str] = deque(maxlen=50)
        self._initial_tool_result: Any = None

    # ------------------------------------------------------------------ #
    #  WebSocket lifecycle                                                #
    # ------------------------------------------------------------------ #

    def set_ws(self, ws: Any) -> None:
        """Attach the active WebSocket connection, closing any previous one."""
        old = self._ws
        self._ws = ws
        self.app_info.state = AppState.INITIALIZING
        if old is not None and old is not ws:
            try:
                asyncio.ensure_future(old.close())
            except Exception as e:
                logger.debug("Failed to close old WebSocket: %s", e)
        logger.info(
            "WebSocket set for app %s (state -> INITIALIZING)", self.app_info.tool_name
        )

    async def drain_pending(self) -> None:
        """Send queued notifications that accumulated while WS was down."""
        while self._pending_notifications and self._ws:
            msg = self._pending_notifications.popleft()
            try:
                await self._ws.send(msg)
            except Exception:
                self._pending_notifications.appendleft(msg)
                break

    # ------------------------------------------------------------------ #
    #  Inbound: browser -> Python                                        #
    # ------------------------------------------------------------------ #

    async def handle_message(self, raw: str) -> str | None:
        """Handle a JSON-RPC message from the browser.

        Returns a JSON-RPC response string, or *None* for notifications.
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from browser: %s", raw[:200])
            return None

        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "tools/call":
            return await self._handle_tool_call(msg_id, params)

        if method == "resources/read":
            return await self._handle_resource_read(msg_id, params)

        if method == "ui/message":
            return self._handle_ui_message(msg_id, params)

        if method == "ui/update-model-context":
            return self._handle_model_context_update(msg_id, params)

        if method == "ui/notifications/initialized":
            self.app_info.state = AppState.READY
            logger.info("App %s initialized", self.app_info.tool_name)
            # Push deferred initial tool result now that the app is ready
            if self._initial_tool_result is not None:
                pending = self._initial_tool_result
                self._initial_tool_result = None
                asyncio.ensure_future(self.push_tool_result(pending))
            return None

        if method == "ui/notifications/teardown":
            self.app_info.state = AppState.CLOSED
            logger.info("App %s teardown", self.app_info.tool_name)
            return None

        # Unknown notification — ignore silently
        if msg_id is None:
            return None

        # Unknown request — return error
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }
        )

    # ------------------------------------------------------------------ #
    #  Handler: tools/call                                                #
    # ------------------------------------------------------------------ #

    async def _handle_tool_call(self, msg_id: Any, params: dict[str, Any]) -> str:
        """Proxy a tool call from the app to the MCP server."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if not tool_name or not _VALID_TOOL_NAME.match(tool_name):
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": f"Invalid tool name: {tool_name!r}",
                    },
                }
            )

        logger.debug(
            "App %s calling tool %s with %s",
            self.app_info.tool_name,
            tool_name,
            arguments,
        )

        try:
            result = await asyncio.wait_for(
                self.tool_manager.execute_tool(
                    tool_name,
                    arguments,
                    namespace=self.app_info.server_name,
                ),
                timeout=DEFAULT_APP_TOOL_TIMEOUT,
            )

            if result.success:
                # Format result in MCP content structure
                result_content = self._format_tool_result(result.result)
                return json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": result_content,
                    }
                )
            else:
                return json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32000,
                            "message": result.error or "Tool execution failed",
                        },
                    }
                )

        except asyncio.TimeoutError:
            logger.error(
                "Tool call timed out after %ss: %s", DEFAULT_APP_TOOL_TIMEOUT, tool_name
            )
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32000,
                        "message": f"Tool call timed out after {DEFAULT_APP_TOOL_TIMEOUT}s",
                    },
                }
            )

        except Exception as e:
            logger.error("Tool call failed: %s", e)
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32000, "message": str(e)},
                }
            )

    # ------------------------------------------------------------------ #
    #  Handler: resources/read                                            #
    # ------------------------------------------------------------------ #

    async def _handle_resource_read(self, msg_id: Any, params: dict[str, Any]) -> str:
        """Proxy a resource read from the app to the MCP server."""
        uri = params.get("uri", "")

        try:
            result = await self.tool_manager.read_resource(
                uri, server_name=self.app_info.server_name
            )
            return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result})
        except Exception as e:
            logger.error("Resource read failed: %s", e)
            return json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32000, "message": str(e)},
                }
            )

    # ------------------------------------------------------------------ #
    #  Handler: ui/message                                                #
    # ------------------------------------------------------------------ #

    def _handle_ui_message(self, msg_id: Any, params: dict[str, Any]) -> str:
        """Handle a message from the app to be added to conversation."""
        content = params.get("content", {})
        logger.info("App %s sent message: %s", self.app_info.tool_name, content)
        return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": {}})

    # ------------------------------------------------------------------ #
    #  Handler: ui/update-model-context                                   #
    # ------------------------------------------------------------------ #

    def _handle_model_context_update(self, msg_id: Any, params: dict[str, Any]) -> str:
        """Store updated model context from the app."""
        self._model_context = params
        logger.info("App %s updated model context", self.app_info.tool_name)
        return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": {}})

    def set_initial_tool_result(self, result: Any) -> None:
        """Store the initial tool result to be pushed when the app is ready.

        Rather than pushing immediately (which risks the app not having
        its message listener set up yet), this defers delivery until the
        app sends ``ui/notifications/initialized``.
        """
        self._initial_tool_result = result

    # ------------------------------------------------------------------ #
    #  Outbound: Python -> browser                                        #
    # ------------------------------------------------------------------ #

    async def push_tool_result(self, result: Any) -> None:
        """Push a tool result notification to the app."""
        notification = self._safe_json_dumps(
            {
                "jsonrpc": "2.0",
                "method": "ui/notifications/tool-result",
                "params": self._format_tool_result(result),
            }
        )

        if not self._ws:
            self._pending_notifications.append(notification)
            logger.debug("Queued tool-result notification (ws not connected)")
            return

        try:
            await self._ws.send(notification)
        except Exception as e:
            self._pending_notifications.append(notification)
            logger.warning("Failed to push tool result, queued: %s", e)

    async def push_tool_input(self, arguments: dict[str, Any]) -> None:
        """Push tool input to the app (sent after initialization)."""
        if not self._ws:
            return

        notification = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "ui/notifications/tool-input",
                "params": {"arguments": arguments},
            }
        )

        try:
            await self._ws.send(notification)
        except Exception as e:
            logger.warning("Failed to push tool input: %s", e)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @property
    def model_context(self) -> dict[str, Any] | None:
        """Get the latest model context from the app (if any)."""
        return self._model_context

    @staticmethod
    def _safe_json_dumps(obj: Any) -> str:
        """Serialise *obj* to JSON, falling back to _to_serializable on error."""
        try:
            return json.dumps(obj)
        except (TypeError, ValueError):
            return json.dumps(AppBridge._to_serializable(obj))

    @staticmethod
    def _extract_raw_result(result: Any) -> Any:
        """Unwrap middleware/ToolCallResult wrappers to get the raw MCP result."""
        # Unwrap objects that have a .result attribute (ToolExecutionResult, etc.)
        seen: set[int] = set()
        while hasattr(result, "result") and not isinstance(result, (dict, str)):
            rid = id(result)
            if rid in seen:
                break
            seen.add(rid)
            result = result.result
        return result

    @staticmethod
    def _to_serializable(obj: Any, _seen: set[int] | None = None) -> Any:
        """Convert an object to a JSON-serializable form."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        # Circular reference protection for mutable containers
        if _seen is None:
            _seen = set()
        oid = id(obj)
        if oid in _seen:
            return "<circular>"
        _seen.add(oid)
        if isinstance(obj, dict):
            return {k: AppBridge._to_serializable(v, _seen) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [AppBridge._to_serializable(v, _seen) for v in obj]
        # Pydantic models
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        # Dataclass / namedtuple fallback
        if hasattr(obj, "__dict__"):
            return {
                k: AppBridge._to_serializable(v, _seen)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        return str(obj)

    @staticmethod
    def _extract_structured_content(out: dict[str, Any]) -> dict[str, Any]:
        """Extract structuredContent from text blocks per MCP spec.

        The MCP spec says servers SHOULD include structuredContent serialised
        as JSON inside a text content block for backwards compatibility.
        When the upstream transport loses the top-level structuredContent
        (e.g. CTP normalisation), we recover it from that text block.

        Scans all text blocks — servers commonly return a human-readable
        text block alongside a JSON text block containing the structured
        content (e.g. ``play_video`` returns "Video playback started."
        plus the ``ui_patch`` JSON).
        """
        if "structuredContent" in out:
            return out  # already present

        content = out.get("content")
        if not isinstance(content, list) or not content:
            return out

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue

            text = block.get("text", "")
            if not isinstance(text, str) or not text.startswith("{"):
                continue

            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(parsed, dict):
                continue

            # Pattern 1: wrapper dict with a structuredContent key
            if "structuredContent" in parsed:
                out["structuredContent"] = parsed["structuredContent"]
                if "content" in parsed and isinstance(parsed["content"], list):
                    out["content"] = parsed["content"]
                return out

            # Pattern 2: the JSON IS the structured content (has type+version)
            # e.g. {"type": "ui_patch", "version": "3.0", "ops": [...]}
            if "type" in parsed and "version" in parsed:
                out["structuredContent"] = parsed
                return out

        return out

    @staticmethod
    def _format_tool_result(result: Any) -> dict[str, Any]:
        """Normalise a tool result into MCP CallToolResult structure.

        Returns ``{"content": [...], "structuredContent": {...}}`` matching
        the MCP spec's CallToolResult schema.
        """
        # Unwrap any middleware/result wrappers
        result = AppBridge._extract_raw_result(result)

        out: dict[str, Any]

        # Pydantic model with content attr — extract the content list directly
        if not isinstance(result, (dict, str)) and hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                out = {"content": AppBridge._to_serializable(content)}
                # Preserve structuredContent / isError if present
                if hasattr(result, "structuredContent") and result.structuredContent:
                    out["structuredContent"] = AppBridge._to_serializable(
                        result.structuredContent
                    )
                if hasattr(result, "isError") and result.isError:
                    out["isError"] = True
                return AppBridge._extract_structured_content(out)

        if isinstance(result, dict):
            # If content value is an MCP SDK object, extract its content list
            content_val = result.get("content")
            if content_val is not None and not isinstance(content_val, (list, str)):
                if hasattr(content_val, "content") and isinstance(
                    content_val.content, list
                ):
                    result = dict(result)
                    result["content"] = content_val.content
                    # Copy structuredContent if present
                    if (
                        hasattr(content_val, "structuredContent")
                        and content_val.structuredContent
                    ):
                        result["structuredContent"] = content_val.structuredContent
            # Make all nested values JSON-serializable
            result = AppBridge._to_serializable(result)
            if "content" in result:
                return AppBridge._extract_structured_content(result)
            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        if isinstance(result, str):
            return {"content": [{"type": "text", "text": result}]}

        # Fallback
        if hasattr(result, "model_dump"):
            return {
                "content": [{"type": "text", "text": json.dumps(result.model_dump())}]
            }
        return {"content": [{"type": "text", "text": str(result)}]}
