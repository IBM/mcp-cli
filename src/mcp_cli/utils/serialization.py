# src/mcp_cli/utils/serialization.py
"""Shared helpers for unwrapping and serializing MCP tool results."""

from __future__ import annotations

from typing import Any

_UNWRAP_MAX_DEPTH = 10


def unwrap_tool_result(obj: Any, *, max_depth: int = _UNWRAP_MAX_DEPTH) -> Any:
    """Unwrap middleware ``ToolExecutionResult`` wrappers and MCP result dicts.

    When middleware is enabled, ``ToolManager`` returns the result wrapped in
    a ``ToolExecutionResult`` object (from ``chuk_tool_processor.mcp.middleware``).
    The inner payload is typically ``{"isError": bool, "content": ToolResult}``.
    This peels off those layers to get the actual content.

    Raises ``RuntimeError`` if any wrapper layer reports failure.
    """
    depth = 0
    while (
        hasattr(obj, "success")
        and hasattr(obj, "result")
        and not isinstance(obj, dict)
    ):
        if depth >= max_depth:
            raise RuntimeError(f"Exceeded max unwrap depth ({max_depth})")
        if not obj.success:
            error = getattr(obj, "error", None) or "Unknown tool error"
            raise RuntimeError(error)
        obj = obj.result
        depth += 1

    # Unwrap MCP call_tool dict pattern: {"isError": ..., "content": ...}
    if isinstance(obj, dict) and "content" in obj and "isError" in obj:
        if obj["isError"]:
            error_msg = obj.get("error") or obj.get("content") or "Tool returned an error"
            if not isinstance(error_msg, str):
                error_msg = str(error_msg)
            raise RuntimeError(error_msg)
        obj = obj["content"]

    return obj


def to_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable form.

    Handles MCP SDK ``ToolResult``, Pydantic models, and other
    non-serializable types.  MCP ``ToolResult`` objects (identified by
    a ``content`` list attribute) are checked *before* generic Pydantic
    ``model_dump()`` so that text content is extracted directly.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, list):
        return [to_serializable(item) for item in obj]

    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    # MCP SDK ToolResult (has content *list* of TextContent / ImageContent).
    # Must be checked BEFORE generic Pydantic model_dump to extract text.
    content = getattr(obj, "content", None)
    if isinstance(content, list):
        parts: list[Any] = []
        for item in content:
            if hasattr(item, "text"):
                parts.append(item.text)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "model_dump"):
                parts.append(to_serializable(item.model_dump()))
            else:
                parts.append(str(item))
        return parts[0] if len(parts) == 1 else parts

    # Pydantic models (generic fallback)
    if hasattr(obj, "model_dump"):
        return to_serializable(obj.model_dump())
    if hasattr(obj, "dict"):
        return to_serializable(obj.dict())

    # Non-list .content attribute on non-Pydantic objects
    if content is not None:
        return to_serializable(content)

    return str(obj)
