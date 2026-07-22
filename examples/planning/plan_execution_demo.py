#!/usr/bin/env python
"""
Plan execution demo — run plans with parallel batches, checkpoints, and dry-run.

Demonstrates:
1. Dry-run mode: trace execution without side effects
2. Live execution with mock tools and progress callbacks
3. Parallel batch execution: independent steps run concurrently
4. Variable resolution: ${var}, ${var.field}, template strings
5. Execution checkpointing and resume
6. Step failure handling

No API key or MCP server needed — runs entirely with mocks.

Usage:
    uv run python examples/planning/plan_execution_demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_cli.planning.context import PlanningContext
from mcp_cli.planning.executor import PlanRunner, render_plan_dag


# ── Mock ToolManager ────────────────────────────────────────────────────────


@dataclass
class FakeToolCallResult:
    tool_name: str
    success: bool = True
    result: Any = None
    error: str | None = None


class MockToolManager:
    """ToolManager stub that simulates tool execution with realistic results."""

    TOOLS = {
        "read_file": "def handle_auth(request):\n    token = request.headers.get('Authorization')\n    return verify(token)",
        "search_code": "Found 12 usages across 5 files:\n  - api/routes.py:14\n  - api/middleware.py:8\n  - tests/test_auth.py:23\n  - tests/test_routes.py:41\n  - utils/decorators.py:7",
        "list_files": "tests/auth/test_handler.py\ntests/auth/test_middleware.py\ntests/auth/conftest.py",
        "write_file": "Written 42 lines to file",
        "run_tests": "5 tests passed, 0 failed",
        "fetch_url": '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}',
        "process_data": "Processed 2 records",
    }

    def __init__(self, *, fail_tools: set[str] | None = None, delay: float = 0.05):
        self._fail_tools = fail_tools or set()
        self._delay = delay
        self.call_log: list[tuple[str, dict]] = []

    @dataclass
    class ToolInfo:
        name: str

    def get_all_tools(self):
        return [self.ToolInfo(name=n) for n in self.TOOLS]

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
        await asyncio.sleep(self._delay)  # Simulate network latency
        self.call_log.append((tool_name, arguments))

        if tool_name in self._fail_tools:
            return FakeToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Connection refused: {tool_name} server is down",
            )

        result = self.TOOLS.get(tool_name, f"Result from {tool_name}")
        return FakeToolCallResult(tool_name=tool_name, result=result)


# ── Demo Plans ──────────────────────────────────────────────────────────────

REFACTOR_PLAN = {
    "id": "refactor-auth-001",
    "title": "Refactor Auth Module",
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

API_PIPELINE = {
    "id": "api-pipeline-001",
    "title": "API Data Pipeline",
    "variables": {
        "api": {"host": "api.example.com", "version": "v2"},
    },
    "steps": [
        {
            "index": "1",
            "title": "Fetch users",
            "tool": "fetch_url",
            "args": {"url": "https://${api.host}/${api.version}/users"},
            "depends_on": [],
            "result_variable": "users",
        },
        {
            "index": "2",
            "title": "Process user data",
            "tool": "process_data",
            "args": {"data": "${users}"},
            "depends_on": ["1"],
            "result_variable": "processed",
        },
    ],
}


# ── Demo Runner ─────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def make_callbacks():
    """Create progress callbacks that print step progress."""

    def on_start(index, title, tool_name):
        print(f"    [{index}] {title} [{tool_name}]...")

    def on_complete(step_result):
        if step_result.success:
            result_preview = str(step_result.result)[:60]
            print(f"         -> OK ({step_result.duration:.2f}s): {result_preview}")
        else:
            print(
                f"         -> FAIL ({step_result.duration:.2f}s): {step_result.error}"
            )

    return on_start, on_complete


async def main() -> None:
    print()
    print("=" * 60)
    print("  Plan Execution Demo")
    print("  Parallel batches, checkpoints, dry-run, variable resolution")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        plans_dir = Path(tmpdir) / "plans"

        # ── 1. Dry Run ──
        section("1. Dry-Run Mode")
        print("  Trace plan execution without running any tools:\n")

        tm = MockToolManager()
        ctx = PlanningContext(tm, plans_dir=plans_dir)
        on_start, on_complete = make_callbacks()

        runner = PlanRunner(
            ctx,
            on_step_start=on_start,
            on_step_complete=on_complete,
            enable_guards=False,
        )

        result = await runner.execute_plan(
            REFACTOR_PLAN, dry_run=True, checkpoint=False
        )

        print(f"\n  Result: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"  Simulated variables: {list(result.variables.keys())}")
        print(f"  Tools called: {len(tm.call_log)} (should be 0 in dry-run)")

        # ── 2. DAG View ──
        section("2. Execution DAG")
        print(f"  Plan: {REFACTOR_PLAN['title']}")
        print("  Batch structure:\n")
        dag = render_plan_dag(REFACTOR_PLAN)
        print(dag)
        print()
        print("  Batch 1: Steps 1, 2, 3 (parallel - no deps)")
        print("  Batch 2: Step 4 (depends on 1, 2)")
        print("  Batch 3: Step 5 (depends on 3, 4)")

        # ── 3. Live Execution ──
        section("3. Live Execution with Parallel Batches")
        print("  Executing refactor plan with mock tools:\n")

        tm2 = MockToolManager()
        ctx2 = PlanningContext(tm2, plans_dir=plans_dir)
        on_start2, on_complete2 = make_callbacks()

        runner2 = PlanRunner(
            ctx2,
            on_step_start=on_start2,
            on_step_complete=on_complete2,
            enable_guards=False,
        )

        result = await runner2.execute_plan(REFACTOR_PLAN, checkpoint=True)

        print(f"\n  Result:   {'SUCCESS' if result.success else 'FAILED'}")
        print(f"  Steps:    {len(result.steps)}")
        print(f"  Duration: {result.total_duration:.2f}s")
        print(f"  Tools called: {len(tm2.call_log)}")
        print()
        print("  Variable bindings:")
        for key, value in result.variables.items():
            preview = str(value)[:50]
            print(f"    ${key} = {preview}")

        # ── 4. Variable Resolution ──
        section("4. Variable Resolution (Nested + Templates)")
        print(f"  Plan: {API_PIPELINE['title']}")
        print(f"  Variables: {json.dumps(API_PIPELINE['variables'], indent=4)}")
        print()
        print('  Step 1 args: url = "https://${{api.host}}/${{api.version}}/users"')
        print('  Step 2 args: data = "${{users}}"')
        print()

        tm3 = MockToolManager()
        ctx3 = PlanningContext(tm3, plans_dir=plans_dir)
        runner3 = PlanRunner(ctx3, enable_guards=False)

        result = await runner3.execute_plan(API_PIPELINE, checkpoint=False)

        print("  Executed. Tool calls:")
        for tool_name, args in tm3.call_log:
            print(f"    {tool_name}({json.dumps(args, default=str)[:60]})")

        # ── 5. Checkpoint & Resume ──
        section("5. Execution Checkpointing")

        checkpoint_path = plans_dir / "refactor-auth-001_state.json"
        if checkpoint_path.exists():
            data = json.loads(checkpoint_path.read_text())
            print(f"  Checkpoint file: {checkpoint_path.name}")
            print(f"  Status:          {data['status']}")
            print(f"  Completed steps: {data['completed_steps']}")
            print(f"  Variables saved:  {list(data['variables'].keys())}")
        else:
            print("  No checkpoint found (checkpoint=False was used)")

        # ── 6. Step Failure ──
        section("6. Step Failure Handling")
        print("  Executing with 'write_file' tool failing:\n")

        tm4 = MockToolManager(fail_tools={"write_file"})
        ctx4 = PlanningContext(tm4, plans_dir=Path(tmpdir) / "plans2")
        on_start4, on_complete4 = make_callbacks()

        runner4 = PlanRunner(
            ctx4,
            on_step_start=on_start4,
            on_step_complete=on_complete4,
            enable_guards=False,
        )

        result = await runner4.execute_plan(REFACTOR_PLAN, checkpoint=True)

        print(f"\n  Result: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"  Error:  {result.error}")
        print(f"  Steps completed: {sum(1 for s in result.steps if s.success)}")
        print(f"  Steps failed:    {sum(1 for s in result.steps if not s.success)}")

        # Check failure checkpoint
        fail_ckpt = Path(tmpdir) / "plans2" / "refactor-auth-001_state.json"
        if fail_ckpt.exists():
            data = json.loads(fail_ckpt.read_text())
            print("\n  Failure checkpoint saved:")
            print(f"    Status:          {data['status']}")
            print(f"    Completed steps: {data['completed_steps']}")
            print("    (Resume with: /plan resume refactor-auth-001)")

    print(f"\n{'=' * 60}")
    print("  Demo complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
