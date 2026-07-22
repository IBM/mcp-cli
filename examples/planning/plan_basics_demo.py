#!/usr/bin/env python
"""
Plan basics demo — create, inspect, save, load, and delete plans.

Demonstrates:
1. PlanningContext initialization and plan persistence
2. Building a plan from a dict (both 'tool' and 'tool_calls' formats)
3. PlanRegistry round-trip: save to disk, load from disk, fresh context reload
4. DAG visualization with render_plan_dag()
5. Plan CRUD: list, get, delete

No API key or MCP server needed — runs entirely with mocks.

Usage:
    uv run python examples/planning/plan_basics_demo.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_cli.planning.context import PlanningContext
from mcp_cli.planning.executor import render_plan_dag


# ── Mock ToolManager ────────────────────────────────────────────────────────


@dataclass
class FakeToolInfo:
    name: str


class MockToolManager:
    """Minimal ToolManager stub for demos."""

    TOOLS = ["read_file", "write_file", "search_code", "list_files", "run_tests"]

    async def get_all_tools(self):
        return [FakeToolInfo(name=n) for n in self.TOOLS]

    async def get_adapted_tools_for_llm(self, provider: str) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": n, "description": f"Tool: {n}"}}
            for n in self.TOOLS
        ]


# ── Demo Plans ──────────────────────────────────────────────────────────────

REFACTOR_PLAN = {
    "title": "Refactor Auth Module",
    "description": "Read the auth module, find all usages, then refactor",
    "tags": ["refactor", "auth"],
    "variables": {"module_path": "src/auth/handler.py"},
    "steps": [
        {
            "index": "1",
            "title": "Read auth module",
            "tool": "read_file",
            "args": {"path": "${module_path}"},
            "depends_on": [],
            "result_variable": "auth_code",
        },
        {
            "index": "2",
            "title": "Find all auth usages",
            "tool": "search_code",
            "args": {"query": "from auth.handler import"},
            "depends_on": [],
            "result_variable": "usages",
        },
        {
            "index": "3",
            "title": "List test files",
            "tool": "list_files",
            "args": {"pattern": "tests/auth/*.py"},
            "depends_on": [],
            "result_variable": "test_files",
        },
        {
            "index": "4",
            "title": "Write refactored module",
            "tool": "write_file",
            "args": {"path": "${module_path}", "content": "refactored code"},
            "depends_on": ["1", "2"],
            "result_variable": "write_result",
        },
        {
            "index": "5",
            "title": "Run auth tests",
            "tool": "run_tests",
            "args": {"path": "tests/auth/"},
            "depends_on": ["3", "4"],
            "result_variable": "test_results",
        },
    ],
}

API_PLAN = {
    "title": "Deploy API Endpoint",
    "steps": [
        {
            "title": "Read API spec",
            "tool_calls": [
                {"id": "tc-1", "name": "read_file", "args": {"path": "api/spec.yaml"}}
            ],
            "depends_on": [],
            "result_variable": "spec",
        },
        {
            "title": "Generate handler code",
            "tool_calls": [
                {
                    "id": "tc-2",
                    "name": "write_file",
                    "args": {"path": "api/handler.py", "content": "..."},
                }
            ],
            "depends_on": ["1"],
            "result_variable": "handler",
        },
        {
            "title": "Run integration tests",
            "tool_calls": [
                {"id": "tc-3", "name": "run_tests", "args": {"path": "tests/api/"}}
            ],
            "depends_on": ["2"],
            "result_variable": "test_result",
        },
    ],
}


# ── Demo Runner ─────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


async def main() -> None:
    print()
    print("=" * 60)
    print("  Plan Basics Demo")
    print("  Create, inspect, save, load, and delete plans")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        plans_dir = Path(tmpdir) / "plans"
        tm = MockToolManager()
        ctx = PlanningContext(tm, plans_dir=plans_dir)

        # ── 1. Tool Catalog ──
        section("1. Tool Catalog")
        names = await ctx.get_tool_names()
        print(f"  Available tools ({len(names)}):")
        for name in names:
            print(f"    - {name}")

        # ── 2. DAG Visualization ──
        section("2. DAG Visualization — Refactor Plan")
        print(f"  Plan: {REFACTOR_PLAN['title']}")
        print(f"  Tags: {REFACTOR_PLAN['tags']}")
        print(f"  Variables: {REFACTOR_PLAN['variables']}")
        print()
        dag = render_plan_dag(REFACTOR_PLAN)
        print(dag)
        print()
        print("  Note: Steps 1-3 have no dependencies on each other")
        print("  and will execute in parallel (same batch).")

        # ── 3. Save Plans ──
        section("3. Save Plans to Registry")
        plan_id_1 = await ctx.save_plan_from_dict(REFACTOR_PLAN)
        print(f"  Saved refactor plan: {plan_id_1[:12]}...")

        plan_id_2 = await ctx.save_plan_from_dict(API_PLAN)
        print(f"  Saved API plan:      {plan_id_2[:12]}...")

        # ── 4. List Plans ──
        section("4. List All Plans")
        plans = await ctx.list_plans()
        for p in plans:
            step_count = len(p.get("steps", []))
            print(f"  [{p['id'][:12]}...]  {p['title']:<30}  ({step_count} steps)")

        # ── 5. Load Plan ──
        section("5. Load Plan by ID")
        loaded = await ctx.get_plan(plan_id_1)
        print(f"  Title:       {loaded['title']}")
        print(f"  Description: {loaded.get('description', 'N/A')}")
        print(f"  Steps:       {len(loaded['steps'])}")
        print(f"  Variables:   {loaded.get('variables', {})}")

        # ── 6. Persistence — Fresh Context ──
        section("6. Persistence — Fresh Context Loads from Disk")
        ctx2 = PlanningContext(tm, plans_dir=plans_dir)
        reloaded = await ctx2.get_plan(plan_id_1)
        print(f"  Fresh context found plan: {reloaded is not None}")
        print(
            f"  Title matches:            {reloaded['title'] == REFACTOR_PLAN['title']}"
        )

        # ── 7. DAG with Status Indicators ──
        section("7. DAG with Status Indicators")
        status_plan = {
            "steps": [
                {
                    "index": "1",
                    "title": "Read source",
                    "tool": "read_file",
                    "_status": "completed",
                },
                {
                    "index": "2",
                    "title": "Search usages",
                    "tool": "search_code",
                    "_status": "completed",
                },
                {
                    "index": "3",
                    "title": "Refactor code",
                    "tool": "write_file",
                    "_status": "running",
                    "depends_on": ["1", "2"],
                },
                {
                    "index": "4",
                    "title": "Run tests",
                    "tool": "run_tests",
                    "_status": "pending",
                    "depends_on": ["3"],
                },
            ]
        }
        dag = render_plan_dag(status_plan)
        print(dag)
        print()
        print(
            "  Legend: \u25cf completed  \u25c9 running  \u25cb pending  \u2717 failed"
        )

        # ── 8. Delete Plan ──
        section("8. Delete Plan")
        deleted = await ctx.delete_plan(plan_id_2)
        print(f"  Deleted API plan: {deleted}")
        remaining = await ctx.list_plans()
        print(f"  Plans remaining:  {len(remaining)}")

        # ── 9. tool_calls Format ──
        section("9. tool_calls Format (from PlanRegistry)")
        dag = render_plan_dag(API_PLAN)
        print(f"  Plan: {API_PLAN['title']}")
        print()
        print(dag)

    print(f"\n{'=' * 60}")
    print("  Demo complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
