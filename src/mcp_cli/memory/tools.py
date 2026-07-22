"""Memory scope tool definitions and handler for LLM tool interception."""

from __future__ import annotations

import logging

from mcp_cli.memory.models import MemoryScope
from mcp_cli.memory.store import MemoryScopeStore

logger = logging.getLogger(__name__)

_MEMORY_TOOL_NAMES = frozenset({"remember", "recall", "forget"})


def get_memory_tools_as_dicts() -> list[dict]:
    """Return OpenAI-format tool definitions for memory scope tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": (
                    "Store a persistent memory that survives across sessions. "
                    "Use 'workspace' scope for project-specific knowledge, "
                    "'global' scope for personal preferences."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": ["workspace", "global"],
                            "description": "Memory scope: 'workspace' (this project) or 'global' (all projects).",
                        },
                        "key": {
                            "type": "string",
                            "description": "Short identifier for this memory (e.g., 'test_framework', 'db_type').",
                        },
                        "content": {
                            "type": "string",
                            "description": "The memory content to store.",
                        },
                    },
                    "required": ["scope", "key", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": (
                    "Retrieve persistent memories. "
                    "Call with no arguments to list all memories across scopes. "
                    "Use 'key' for exact lookup or 'query' for search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": ["workspace", "global"],
                            "description": "Limit to a specific scope. Omit to search both.",
                        },
                        "key": {
                            "type": "string",
                            "description": "Exact key to look up.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search term (matches key and content, case-insensitive).",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forget",
                "description": "Remove a persistent memory by key.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": ["workspace", "global"],
                            "description": "Memory scope: 'workspace' or 'global'.",
                        },
                        "key": {
                            "type": "string",
                            "description": "Key of the memory to remove.",
                        },
                    },
                    "required": ["scope", "key"],
                },
            },
        },
    ]


async def handle_memory_tool(
    store: MemoryScopeStore, tool_name: str, arguments: dict
) -> str:
    """Execute a memory tool and return the result as a string."""
    try:
        if tool_name == "remember":
            scope = MemoryScope(arguments["scope"])
            entry = store.remember(scope, arguments["key"], arguments["content"])
            return f"Remembered '{entry.key}' in {scope.value} scope."

        if tool_name == "recall":
            scope_str = arguments.get("scope")
            recall_scope: MemoryScope | None = (
                MemoryScope(scope_str) if scope_str else None
            )
            key = arguments.get("key")
            query = arguments.get("query")

            entries = store.recall(scope=recall_scope, key=key, query=query)
            if not entries:
                return "No memories found."

            lines = []
            for e in entries:
                lines.append(f"- [{e.key}]: {e.content}")
            return "\n".join(lines)

        if tool_name == "forget":
            scope = MemoryScope(arguments["scope"])
            removed = store.forget(scope, arguments["key"])
            if removed:
                return f"Forgot '{arguments['key']}' from {scope.value} scope."
            return (
                f"No memory with key '{arguments['key']}' found in {scope.value} scope."
            )

        return f"Unknown memory tool: {tool_name}"

    except Exception as exc:
        logger.warning("Memory tool %s failed: %s", tool_name, exc)
        return f"Memory tool error: {exc}"
