#!/usr/bin/env python
"""
Parallel execution demo — shows how independent steps run concurrently.

Demonstrates:
1. Topological batching: steps grouped by dependency structure
2. Concurrent execution within batches via asyncio
3. Diamond, fan-out, and pipeline DAG patterns
4. Timing evidence that parallel steps run concurrently
5. DAG visualization with parallel markers

No API key or MCP server needed — runs entirely with mocks.

Usage:
    uv run python examples/planning/plan_parallel_demo.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_cli.planning.context import PlanningContext
from mcp_cli.planning.executor import (
    PlanRunner,
    render_plan_dag,
    _compute_batches,
)


# ── Slow Mock ToolManager ──────────────────────────────────────────────────


@dataclass
class FakeToolCallResult:
    tool_name: str
    success: bool = True
    result: Any = None
    error: str | None = None


class SlowToolManager:
    """ToolManager that takes 200ms per tool to demonstrate parallelism."""

    DELAY = 0.2  # 200ms per tool call

    def __init__(self):
        self.call_times: list[tuple[str, float, float]] = []

    @dataclass
    class ToolInfo:
        name: str

    def get_all_tools(self):
        tools = [
            "fetch",
            "parse",
            "validate",
            "transform",
            "aggregate",
            "store",
            "notify",
        ]
        return [self.ToolInfo(name=n) for n in tools]

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
        start = time.perf_counter()
        await asyncio.sleep(self.DELAY)
        end = time.perf_counter()
        self.call_times.append((tool_name, start, end))
        return FakeToolCallResult(
            tool_name=tool_name,
            result=f"{tool_name} result",
        )


# ── Demo Plans ──────────────────────────────────────────────────────────────

# Pattern 1: Fan-out (1 → many)
FANOUT_PLAN = {
    "id": "fanout-demo",
    "title": "Fan-Out Pattern (1 root, 5 parallel leaves)",
    "steps": [
        {
            "index": "1",
            "title": "Fetch data source",
            "tool": "fetch",
            "args": {},
            "depends_on": [],
            "result_variable": "data",
        },
        {
            "index": "2",
            "title": "Parse section A",
            "tool": "parse",
            "args": {"section": "A"},
            "depends_on": ["1"],
            "result_variable": "section_a",
        },
        {
            "index": "3",
            "title": "Parse section B",
            "tool": "parse",
            "args": {"section": "B"},
            "depends_on": ["1"],
            "result_variable": "section_b",
        },
        {
            "index": "4",
            "title": "Parse section C",
            "tool": "parse",
            "args": {"section": "C"},
            "depends_on": ["1"],
            "result_variable": "section_c",
        },
        {
            "index": "5",
            "title": "Parse section D",
            "tool": "parse",
            "args": {"section": "D"},
            "depends_on": ["1"],
            "result_variable": "section_d",
        },
        {
            "index": "6",
            "title": "Parse section E",
            "tool": "parse",
            "args": {"section": "E"},
            "depends_on": ["1"],
            "result_variable": "section_e",
        },
    ],
}

# Pattern 2: Diamond (1 → 2 → 1)
DIAMOND_PLAN = {
    "id": "diamond-demo",
    "title": "Diamond Pattern (fork and join)",
    "steps": [
        {
            "index": "1",
            "title": "Fetch raw data",
            "tool": "fetch",
            "args": {},
            "depends_on": [],
            "result_variable": "raw",
        },
        {
            "index": "2",
            "title": "Validate schema",
            "tool": "validate",
            "args": {},
            "depends_on": ["1"],
            "result_variable": "schema_ok",
        },
        {
            "index": "3",
            "title": "Transform format",
            "tool": "transform",
            "args": {},
            "depends_on": ["1"],
            "result_variable": "transformed",
        },
        {
            "index": "4",
            "title": "Aggregate results",
            "tool": "aggregate",
            "args": {},
            "depends_on": ["2", "3"],
            "result_variable": "final",
        },
    ],
}

# Pattern 3: Wide pipeline (3 independent → 3 independent → 1 join)
WIDE_PIPELINE = {
    "id": "wide-pipeline",
    "title": "Wide Pipeline (3 sources, 3 processors, 1 merge)",
    "steps": [
        {
            "index": "1",
            "title": "Fetch API A",
            "tool": "fetch",
            "args": {"source": "A"},
            "depends_on": [],
            "result_variable": "api_a",
        },
        {
            "index": "2",
            "title": "Fetch API B",
            "tool": "fetch",
            "args": {"source": "B"},
            "depends_on": [],
            "result_variable": "api_b",
        },
        {
            "index": "3",
            "title": "Fetch API C",
            "tool": "fetch",
            "args": {"source": "C"},
            "depends_on": [],
            "result_variable": "api_c",
        },
        {
            "index": "4",
            "title": "Process A",
            "tool": "transform",
            "args": {"data": "${api_a}"},
            "depends_on": ["1"],
            "result_variable": "proc_a",
        },
        {
            "index": "5",
            "title": "Process B",
            "tool": "transform",
            "args": {"data": "${api_b}"},
            "depends_on": ["2"],
            "result_variable": "proc_b",
        },
        {
            "index": "6",
            "title": "Process C",
            "tool": "transform",
            "args": {"data": "${api_c}"},
            "depends_on": ["3"],
            "result_variable": "proc_c",
        },
        {
            "index": "7",
            "title": "Merge all results",
            "tool": "aggregate",
            "args": {},
            "depends_on": ["4", "5", "6"],
            "result_variable": "merged",
        },
    ],
}


# ── Demo Runner ─────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def show_batches(steps):
    """Show the computed batch structure."""
    batches = _compute_batches(steps)
    for i, batch in enumerate(batches, 1):
        indices = [s.get("index", "?") for s in batch]
        titles = [s.get("title", "?") for s in batch]
        parallel = " (PARALLEL)" if len(batch) > 1 else ""
        print(f"    Batch {i}{parallel}:")
        for idx, title in zip(indices, titles):
            print(f"      Step {idx}: {title}")


def show_timing(call_times: list[tuple[str, float, float]], plan_start: float):
    """Visualize execution timing as a timeline."""
    if not call_times:
        return

    print("    Timeline (relative to plan start):\n")
    for name, start, end in sorted(call_times, key=lambda x: x[1]):
        offset = start - plan_start
        duration = end - start
        bar_start = int(offset * 20)  # 20 chars per second
        bar_len = max(1, int(duration * 20))
        bar = " " * bar_start + "\u2588" * bar_len
        print(f"      {name:<15} [{bar}] {offset:.2f}s - {offset + duration:.2f}s")


async def run_and_time(tm, ctx, plan):
    """Execute a plan and return (result, plan_start_time)."""
    plan_start = time.perf_counter()

    runner = PlanRunner(ctx, enable_guards=False)
    result = await runner.execute_plan(plan, checkpoint=False)

    return result, plan_start


async def main() -> None:
    print()
    print("=" * 60)
    print("  Parallel Execution Demo")
    print("  Topological batching & concurrent step execution")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Pattern 1: Fan-Out ──
        section("1. Fan-Out Pattern")
        print("  DAG:")
        print(render_plan_dag(FANOUT_PLAN))
        print()

        print("  Batch structure:")
        show_batches(FANOUT_PLAN["steps"])
        print()

        tm1 = SlowToolManager()
        ctx1 = PlanningContext(tm1, plans_dir=Path(tmpdir) / "p1")
        result, t0 = await run_and_time(tm1, ctx1, FANOUT_PLAN)

        serial_time = len(FANOUT_PLAN["steps"]) * SlowToolManager.DELAY
        print(
            f"  Execution: {result.total_duration:.2f}s "
            f"(vs {serial_time:.2f}s serial = "
            f"{serial_time / max(0.01, result.total_duration):.1f}x speedup)"
        )
        print()
        show_timing(tm1.call_times, t0)

        # ── Pattern 2: Diamond ──
        section("2. Diamond Pattern")
        print("  DAG:")
        print(render_plan_dag(DIAMOND_PLAN))
        print()

        print("  Batch structure:")
        show_batches(DIAMOND_PLAN["steps"])
        print()

        tm2 = SlowToolManager()
        ctx2 = PlanningContext(tm2, plans_dir=Path(tmpdir) / "p2")
        result, t0 = await run_and_time(tm2, ctx2, DIAMOND_PLAN)

        serial_time = len(DIAMOND_PLAN["steps"]) * SlowToolManager.DELAY
        print(
            f"  Execution: {result.total_duration:.2f}s "
            f"(vs {serial_time:.2f}s serial = "
            f"{serial_time / max(0.01, result.total_duration):.1f}x speedup)"
        )
        print()
        show_timing(tm2.call_times, t0)

        # ── Pattern 3: Wide Pipeline ──
        section("3. Wide Pipeline Pattern")
        print("  DAG:")
        print(render_plan_dag(WIDE_PIPELINE))
        print()

        print("  Batch structure:")
        show_batches(WIDE_PIPELINE["steps"])
        print()

        tm3 = SlowToolManager()
        ctx3 = PlanningContext(tm3, plans_dir=Path(tmpdir) / "p3")
        result, t0 = await run_and_time(tm3, ctx3, WIDE_PIPELINE)

        serial_time = len(WIDE_PIPELINE["steps"]) * SlowToolManager.DELAY
        print(
            f"  Execution: {result.total_duration:.2f}s "
            f"(vs {serial_time:.2f}s serial = "
            f"{serial_time / max(0.01, result.total_duration):.1f}x speedup)"
        )
        print()
        show_timing(tm3.call_times, t0)

        # ── Summary ──
        section("Summary")
        print("  Pattern       | Steps | Batches | Serial  | Parallel | Speedup")
        print("  " + "-" * 56)

        for name, plan, tm_obj in [
            ("Fan-Out", FANOUT_PLAN, tm1),
            ("Diamond", DIAMOND_PLAN, tm2),
            ("Wide Pipeline", WIDE_PIPELINE, tm3),
        ]:
            n_steps = len(plan["steps"])
            batches = _compute_batches(plan["steps"])
            serial = n_steps * SlowToolManager.DELAY
            # Total wall time from first start to last end
            if tm_obj.call_times:
                all_starts = [s for _, s, _ in tm_obj.call_times]
                all_ends = [e for _, _, e in tm_obj.call_times]
                wall = max(all_ends) - min(all_starts)
            else:
                wall = 0
            speedup = serial / max(0.01, wall)
            print(
                f"  {name:<15} | {n_steps:>5} | {len(batches):>7} | {serial:>6.2f}s | {wall:>7.2f}s | {speedup:>5.1f}x"
            )

    print(f"\n{'=' * 60}")
    print("  Demo complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
