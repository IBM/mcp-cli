#!/usr/bin/env python
"""
LLM-integrated planning demo — generate plans from natural language.

Demonstrates the full planning pipeline:
1. PlanAgent generates a structured plan from a natural language description
2. Plan validation against available tool catalog
3. DAG visualization of the generated plan
4. Live execution with mock tools and progress callbacks
5. Retry loop: if the LLM produces invalid steps, PlanAgent auto-corrects

Requires an OpenAI API key (OPENAI_API_KEY environment variable or .env file).
Uses gpt-4o-mini by default for fast, cheap plan generation.

Usage:
    uv run python examples/planning/plan_llm_demo.py
    uv run python examples/planning/plan_llm_demo.py --model gpt-4o
    uv run python examples/planning/plan_llm_demo.py --prompt "fetch weather for 3 cities and compare"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Load .env if available (for OPENAI_API_KEY)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add src to path for local development
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
    """ToolManager with a realistic tool catalog for LLM plan generation.

    The LLM sees the tool names and generates plans using them.
    Execution returns mock results so no real MCP server is needed.
    """

    TOOL_CATALOG = {
        "read_file": "Read a file and return its contents",
        "write_file": "Write content to a file",
        "list_files": "List files matching a glob pattern",
        "search_code": "Search codebase for a pattern",
        "run_tests": "Run test suite and return results",
        "fetch_url": "Fetch data from a URL",
        "execute_query": "Execute a database query",
        "send_notification": "Send a notification message",
    }

    MOCK_RESULTS = {
        "read_file": "def handle_request(req):\n    return process(req.data)",
        "write_file": "Written 35 lines to file",
        "list_files": "src/auth.py\nsrc/routes.py\nsrc/middleware.py",
        "search_code": "Found 8 matches:\n  src/auth.py:12\n  src/routes.py:45\n  tests/test_auth.py:7",
        "run_tests": "12 passed, 0 failed, 0 skipped",
        "fetch_url": '{"status": "ok", "data": [{"id": 1, "name": "Alice"}]}',
        "execute_query": "3 rows returned",
        "send_notification": "Notification sent successfully",
    }

    def __init__(self, *, delay: float = 0.05):
        self._delay = delay
        self.call_log: list[tuple[str, dict]] = []

    @dataclass
    class ToolInfo:
        name: str

    def get_all_tools(self):
        return [self.ToolInfo(name=n) for n in self.TOOL_CATALOG]

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
        await asyncio.sleep(self._delay)
        self.call_log.append((tool_name, arguments))
        result = self.MOCK_RESULTS.get(tool_name, f"Result from {tool_name}")
        return FakeToolCallResult(tool_name=tool_name, result=result)


# ── Plan Generation ─────────────────────────────────────────────────────────


def build_system_prompt(tool_names: list[str]) -> str:
    """Build the system prompt that tells the LLM what tools are available."""
    tools_list = "\n".join(f"  - {name}" for name in tool_names)
    return f"""You are a planning assistant. Given a task description, create a structured execution plan.

Available tools:
{tools_list}

Output a JSON object with this exact structure:
{{
  "title": "Short plan title",
  "steps": [
    {{
      "title": "What this step does",
      "tool": "tool_name",
      "args": {{"arg1": "value1"}},
      "depends_on": [],
      "result_variable": "optional_var_name"
    }}
  ]
}}

Rules:
- Only use tools from the available tools list above
- depends_on is a list of step indices (1-based) that must complete first
- result_variable stores the output for use in later steps as ${{var_name}}
- Keep plans focused — prefer fewer, targeted steps over many small ones
- Each step should have exactly one tool call
- Steps with no dependencies can run in parallel"""


def validate_step(step: dict, tool_names: list[str]) -> tuple[bool, str]:
    """Validate a single plan step against the tool catalog."""
    tool = step.get("tool", "")
    if tool not in tool_names:
        return False, f"Unknown tool: {tool}. Available: {', '.join(tool_names)}"
    if not step.get("title"):
        return False, "Step must have a title"
    return True, ""


# ── Demo Runner ─────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="LLM plan generation demo")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for plan generation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom task description (overrides built-in demos)",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print()
        print("Set it in your shell:")
        print("  export OPENAI_API_KEY=sk-...")
        print()
        print("Or create a .env file in the project root:")
        print("  OPENAI_API_KEY=sk-...")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  LLM Plan Generation Demo")
    print(f"  Model: {args.model}")
    print("=" * 60)

    # Set up mock tool manager with realistic tool catalog
    tm = MockToolManager()
    tool_names = [t.name for t in tm.get_all_tools()]

    print(f"\n  Available tools ({len(tool_names)}):")
    for name in tool_names:
        desc = MockToolManager.TOOL_CATALOG[name]
        print(f"    - {name}: {desc}")

    # Import PlanAgent
    from chuk_ai_planner.agents.plan_agent import PlanAgent

    # Build prompts to demo
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "Read the auth module, find all places that import it, and then run the tests",
            "Fetch user data from the API, save it to a file, and send a notification",
        ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, user_prompt in enumerate(prompts, 1):
            section(f"Demo {i}: LLM Plan Generation")
            print(f'  Prompt: "{user_prompt}"\n')

            # ── Step 1: Generate plan with PlanAgent ──
            print("  [1/4] Generating plan with LLM...\n")

            system_prompt = build_system_prompt(tool_names)
            agent = PlanAgent(
                system_prompt=system_prompt,
                validate_step=lambda step: validate_step(step, tool_names),
                model=args.model,
                max_retries=3,
            )

            try:
                plan_dict = await agent.plan(user_prompt)
            except RuntimeError as e:
                print(f"  FAILED: {e}")
                continue

            # Show retry history
            if len(agent.history) > 1:
                print(
                    f"  (PlanAgent needed {len(agent.history)} attempts "
                    f"to produce a valid plan)\n"
                )

            print(f"  Title: {plan_dict.get('title', 'Untitled')}")
            print(f"  Steps: {len(plan_dict.get('steps', []))}")
            print()

            # ── Step 2: Show generated plan ──
            print("  [2/4] Generated plan:\n")
            for step in plan_dict.get("steps", []):
                idx = step.get("index", "?")
                title = step.get("title", "Untitled")
                tool = step.get("tool", "?")
                deps = step.get("depends_on", [])
                args_preview = json.dumps(step.get("args", {}), default=str)
                if len(args_preview) > 60:
                    args_preview = args_preview[:57] + "..."
                dep_str = (
                    f"  (after: {', '.join(str(d) for d in deps)})" if deps else ""
                )
                print(f"    Step {idx}: {title}")
                print(f"      tool: {tool}({args_preview}){dep_str}")

            # ── Step 3: DAG visualization ──
            print("\n  [3/4] Execution DAG:\n")

            # Ensure steps have index fields for DAG rendering
            for j, step in enumerate(plan_dict.get("steps", []), 1):
                if "index" not in step:
                    step["index"] = str(j)

            dag = render_plan_dag(plan_dict)
            print(dag)

            # ── Step 4: Execute the plan ──
            print("\n  [4/4] Executing plan with mock tools:\n")

            ctx = PlanningContext(tm, plans_dir=Path(tmpdir) / f"plans_{i}")

            def on_start(index, title, tool_name):
                print(f"    [{index}] {title} [{tool_name}]...")

            def on_complete(step_result):
                if step_result.success:
                    preview = str(step_result.result)[:50]
                    print(f"         -> OK ({step_result.duration:.2f}s): {preview}")
                else:
                    print(f"         -> FAIL: {step_result.error}")

            runner = PlanRunner(
                ctx,
                on_step_start=on_start,
                on_step_complete=on_complete,
                enable_guards=False,
            )

            result = await runner.execute_plan(plan_dict, checkpoint=False)

            print(f"\n  Result:   {'SUCCESS' if result.success else 'FAILED'}")
            print(f"  Steps:    {len(result.steps)}")
            print(f"  Duration: {result.total_duration:.2f}s")

            if result.variables:
                print("  Variables:")
                for key, value in result.variables.items():
                    preview = str(value)[:50]
                    print(f"    ${key} = {preview}")

            # Show raw LLM output for transparency
            print("\n  LLM generation history:")
            for record in agent.history:
                attempt = record["attempt"]
                errors = record.get("errors", [])
                status = "valid" if not errors else f"errors: {errors}"
                print(f"    Attempt {attempt}: {status}")

    print(f"\n{'=' * 60}")
    print("  Demo complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
