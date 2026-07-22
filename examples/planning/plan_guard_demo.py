#!/usr/bin/env python
"""
Guard integration demo — shows how plans respect budget and per-tool limits.

Demonstrates:
1. McpToolBackend with guard checks enabled
2. Pre-execution guard blocking (budget exhausted, per-tool cap)
3. Post-execution result recording (value binding, tool count tracking)
4. Guards disabled mode (bypass all checks)
5. Result extraction from MCP content blocks

No API key, MCP server, or chuk_ai_session_manager needed — uses mocks.

Usage:
    uv run python examples/planning/plan_guard_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_ai_planner.execution.models import ToolExecutionRequest
from mcp_cli.planning.backends import McpToolBackend, _extract_result


# ── Mock ToolManager ────────────────────────────────────────────────────────


@dataclass
class FakeToolCallResult:
    tool_name: str
    success: bool = True
    result: Any = None
    error: str | None = None


class MockToolManager:
    """Minimal ToolManager for demos."""

    def __init__(self, results: dict[str, Any] | None = None):
        self._results = results or {}
        self.calls: list[tuple[str, dict]] = []

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
        self.calls.append((tool_name, arguments))
        result = self._results.get(tool_name, f"result from {tool_name}")
        return FakeToolCallResult(tool_name=tool_name, result=result)


# ── Demo Runner ─────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


async def main() -> None:
    print()
    print("=" * 60)
    print("  Guard Integration Demo")
    print("  Budget, per-tool limits, and result recording")
    print("=" * 60)

    # ── 1. Basic Execution (Guards Disabled) ──
    section("1. Basic Execution (Guards Disabled)")

    tm = MockToolManager({"read_file": "Hello, World!"})
    backend = McpToolBackend(tm, enable_guards=False)

    request = ToolExecutionRequest(
        tool_name="read_file",
        args={"path": "/tmp/test.txt"},
        step_id="step-1",
    )
    result = await backend.execute_tool(request)

    print(f"  Tool:     {result.tool_name}")
    print(f"  Success:  {result.success}")
    print(f"  Result:   {result.result}")
    print(f"  Duration: {result.duration:.4f}s")
    print(f"  Error:    {result.error}")

    # ── 2. Guard Blocks Execution ──
    section("2. Guard Blocks Execution")
    print("  Simulating budget exhausted guard block:\n")

    tm2 = MockToolManager({"write_file": "should not see this"})
    backend2 = McpToolBackend(tm2, enable_guards=True)

    with patch(
        "mcp_cli.planning.backends._check_guards",
        return_value="Budget exhausted: $12.50 of $10.00 limit used",
    ):
        request2 = ToolExecutionRequest(
            tool_name="write_file",
            args={"path": "/tmp/output.txt", "content": "data"},
            step_id="step-2",
        )
        result2 = await backend2.execute_tool(request2)

    print(f"  Tool:     {result2.tool_name}")
    print(f"  Success:  {result2.success}")
    print(f"  Error:    {result2.error}")
    print(f"  Result:   {result2.result}")
    print(f"  Tool was called: {len(tm2.calls) > 0}")
    print("\n  The tool was never executed — guard blocked it pre-flight.")

    # ── 3. Guard Allows + Records ──
    section("3. Guard Allows + Records Result")
    print("  Simulating guard allowing execution and recording result:\n")

    tm3 = MockToolManager({"search_code": "Found 5 matches"})
    backend3 = McpToolBackend(tm3, enable_guards=True)

    record_calls = []

    with (
        patch(
            "mcp_cli.planning.backends._check_guards",
            return_value=None,  # Guard allows
        ),
        patch(
            "mcp_cli.planning.backends._record_result",
            side_effect=lambda *args: record_calls.append(args),
        ),
    ):
        request3 = ToolExecutionRequest(
            tool_name="search_code",
            args={"query": "def main"},
            step_id="step-3",
        )
        result3 = await backend3.execute_tool(request3)

    print(f"  Tool:          {result3.tool_name}")
    print(f"  Success:       {result3.success}")
    print(f"  Result:        {result3.result}")
    print(f"  Record called: {len(record_calls) > 0}")
    if record_calls:
        tool, args, res = record_calls[0]
        print(f"  Recorded:      tool={tool}, args={args}, result={res}")

    # ── 4. Namespace Prefix ──
    section("4. Namespace Prefix")
    print("  When a namespace is set, tools are called with a prefix:\n")

    tm4 = MockToolManager({"filesystem__read_file": "file contents"})
    backend4 = McpToolBackend(tm4, namespace="filesystem", enable_guards=False)

    request4 = ToolExecutionRequest(
        tool_name="read_file",
        args={"path": "/tmp/x"},
        step_id="step-4",
    )
    result4 = await backend4.execute_tool(request4)

    print("  Request tool:    read_file")
    print(f"  Actual call:     {tm4.calls[0][0]}")
    print(f"  Result tool:     {result4.tool_name}")
    print(f"  Success:         {result4.success}")

    # ── 5. Error Handling ──
    section("5. Tool Error Handling")
    print("  Backend catches exceptions from ToolManager:\n")

    class ExplodingToolManager:
        async def execute_tool(self, *args, **kwargs):
            raise ConnectionError("MCP server connection refused")

    backend5 = McpToolBackend(ExplodingToolManager(), enable_guards=False)

    request5 = ToolExecutionRequest(
        tool_name="ping",
        args={},
        step_id="step-5",
    )
    result5 = await backend5.execute_tool(request5)

    print(f"  Success:  {result5.success}")
    print(f"  Error:    {result5.error}")
    print(f"  Duration: {result5.duration:.4f}s")
    print("\n  Exception caught and wrapped — no crash.")

    # ── 6. Result Extraction ──
    section("6. MCP Content Block Extraction")
    print("  _extract_result normalizes MCP-style content blocks:\n")

    examples = [
        ("None", None),
        ("String", "hello world"),
        ("Dict", {"key": "value"}),
        ("Single text block", [{"type": "text", "text": "result data"}]),
        (
            "Multiple text blocks",
            [
                {"type": "text", "text": "line 1"},
                {"type": "text", "text": "line 2"},
            ],
        ),
        (
            "Mixed blocks (image + text)",
            [
                {"type": "image", "url": "http://example.com/img.png"},
                {"type": "text", "text": "caption"},
            ],
        ),
        ("List of strings", ["a", "b", "c"]),
    ]

    for label, raw in examples:
        extracted = _extract_result(raw)
        display = repr(extracted)
        if len(display) > 50:
            display = display[:50] + "..."
        print(f"    {label:<30} -> {display}")

    print(f"\n{'=' * 60}")
    print("  Demo complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
