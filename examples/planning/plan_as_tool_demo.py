#!/usr/bin/env python
"""
Plan-as-a-Tool demo — the LLM autonomously creates and executes plans.

Demonstrates Tier 6.8: Model-Driven Planning. Instead of the user
explicitly requesting a plan, the model itself decides when a task
requires multi-step coordination and calls plan_create_and_execute.

Flow:
1. User asks a complex question that needs multiple tool calls
2. LLM sees plan_create_and_execute alongside regular tools
3. LLM decides to create a plan and invokes the tool
4. planning/tools.py generates the plan via PlanAgent, then executes it
5. Results are returned to the LLM, which summarizes them for the user

Requires OPENAI_API_KEY (uses gpt-4o-mini by default).

Usage:
    uv run python examples/planning/plan_as_tool_demo.py
    uv run python examples/planning/plan_as_tool_demo.py --model gpt-4o
    uv run python examples/planning/plan_as_tool_demo.py --prompt "your task here"
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

# Load .env if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_llm.llm.client import get_client
from mcp_cli.planning.context import PlanningContext
from mcp_cli.planning.tools import get_plan_tools_as_dicts, handle_plan_tool


# ── Mock ToolManager ────────────────────────────────────────────────────────


@dataclass
class FakeToolCallResult:
    tool_name: str
    success: bool = True
    result: Any = None
    error: str | None = None


class MockToolManager:
    """ToolManager with realistic tools for demonstration.

    The LLM uses these tools inside its generated plans.
    Plan tools (plan_create_and_execute, etc.) are added separately
    and intercepted before they reach this manager.
    """

    TOOL_CATALOG = {
        "read_file": {
            "description": "Read a file and return its contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
        "write_file": {
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write to"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
        "search_code": {
            "description": "Search codebase for a pattern and return matching files/lines",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search pattern or regex",
                    },
                },
                "required": ["query"],
            },
        },
        "run_tests": {
            "description": "Run test suite for a given path and return pass/fail results",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Test path (directory or file)",
                    },
                },
                "required": ["path"],
            },
        },
        "fetch_url": {
            "description": "Fetch data from a URL and return the response body",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        },
    }

    MOCK_RESULTS = {
        "read_file": "def handle_auth(request):\n    token = request.headers.get('Authorization')\n    if not token:\n        raise AuthError('Missing token')\n    return verify_jwt(token)",
        "write_file": "Written 28 lines to file successfully",
        "search_code": "Found 6 matches:\n  src/auth.py:12 - from auth.handler import handle_auth\n  src/routes.py:3 - from auth.handler import handle_auth\n  src/middleware.py:8 - import auth.handler\n  tests/test_auth.py:5 - from auth.handler import handle_auth\n  tests/test_routes.py:11 - from auth.handler import handle_auth\n  utils/decorators.py:3 - from auth.handler import verify_jwt",
        "run_tests": "8 passed, 0 failed, 2 skipped in 1.34s",
        "fetch_url": '{"users": [{"id": 1, "name": "Alice", "role": "admin"}, {"id": 2, "name": "Bob", "role": "user"}], "total": 2}',
    }

    def __init__(self, *, delay: float = 0.05):
        self._delay = delay
        self.call_log: list[tuple[str, dict]] = []

    @dataclass
    class ToolInfo:
        name: str

    def get_all_tools(self):
        return [self.ToolInfo(name=n) for n in self.TOOL_CATALOG]

    async def get_adapted_tools_for_llm(
        self, provider: str = "openai"
    ) -> tuple[list[dict[str, Any]], dict]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"],
                },
            }
            for name, info in self.TOOL_CATALOG.items()
        ]
        return tools, {}

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
        await asyncio.sleep(self._delay)
        self.call_log.append((tool_name, arguments))
        result = self.MOCK_RESULTS.get(tool_name, f"Result from {tool_name}")
        return FakeToolCallResult(tool_name=tool_name, result=result)


# ── Helpers ──────────────────────────────────────────────────────────────────


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def build_conversation_tools(tm: MockToolManager) -> list[dict[str, Any]]:
    """Build the full tool list the LLM sees: regular tools + plan tools."""
    regular_tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"],
            },
        }
        for name, info in tm.TOOL_CATALOG.items()
    ]

    plan_tools = get_plan_tools_as_dicts()

    return regular_tools + plan_tools


# ── Demo ─────────────────────────────────────────────────────────────────────


async def run_conversation(
    model: str,
    user_message: str,
    tm: MockToolManager,
    plans_dir: Path,
) -> None:
    """Run a single conversation turn where the LLM may use plan tools."""
    print(f'  User: "{user_message}"')
    print()

    # Build the full tool list
    all_tools = build_conversation_tools(tm)

    print(f"  Tools available to LLM: {len(all_tools)}")
    for t in all_tools:
        name = t["function"]["name"]
        desc = t["function"]["description"][:50]
        print(f"    - {name}: {desc}...")
    print()

    # Create LLM client
    client = get_client(provider="openai", model=model)

    system_prompt = (
        "You are a helpful assistant with access to tools. "
        "When a task requires multiple coordinated steps (e.g., read a file, "
        "then search for usages, then run tests), use plan_create_and_execute "
        "to handle it efficiently. For simple single-tool tasks, call the tool "
        "directly."
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # ── Turn 1: LLM decides what to do ──
    print("  [1] Sending to LLM...")
    completion = await client.create_completion(
        messages=messages,
        tools=all_tools,
        tool_choice="auto",
    )

    assistant_content = completion.get("content", "")
    tool_calls = completion.get("tool_calls", [])

    if not tool_calls:
        print(f"  LLM responded directly: {assistant_content}")
        return

    # Add assistant message with tool calls
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_content or "",
    }
    assistant_msg["tool_calls"] = [
        {
            "id": tc.get("id", f"call_{i}"),
            "type": "function",
            "function": {
                "name": tc.get("function", {}).get("name", tc.get("name", "")),
                "arguments": tc.get("function", {}).get(
                    "arguments", json.dumps(tc.get("arguments", {}))
                ),
            },
        }
        for i, tc in enumerate(tool_calls)
    ]
    messages.append(assistant_msg)

    print(f"  LLM chose {len(tool_calls)} tool call(s):")

    # ── Execute tool calls ──
    for tc in tool_calls:
        func = tc.get("function", {})
        tool_name = func.get("name", tc.get("name", ""))
        raw_args = func.get("arguments", tc.get("arguments", "{}"))
        call_id = tc.get("id", "call_0")

        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {}
        else:
            arguments = raw_args

        print(f"\n    Tool: {tool_name}")
        print(f"    Args: {json.dumps(arguments, default=str)[:80]}")

        # Check if this is a plan tool (intercepted)
        from mcp_cli.planning.tools import _PLAN_TOOL_NAMES

        if tool_name in _PLAN_TOOL_NAMES:
            print("    -> Intercepted as plan tool!")
            print()

            # Create PlanningContext
            ctx = PlanningContext(tm, plans_dir=plans_dir)

            # Execute the plan tool
            print("  [2] Executing plan tool...")
            result_json = await handle_plan_tool(
                tool_name, arguments, ctx, model_manager=None
            )

            result_data = json.loads(result_json)

            if result_data.get("success"):
                print(f"  Plan: {result_data.get('title', 'Untitled')}")
                print(
                    f"  Steps completed: {result_data.get('steps_completed', 0)}/{result_data.get('steps_total', 0)}"
                )
                print(f"  Duration: {result_data.get('duration', 0)}s")

                if result_data.get("steps"):
                    print("\n  Step results:")
                    for step in result_data["steps"]:
                        status = "OK" if step.get("success") else "FAIL"
                        print(
                            f"    [{step.get('index', '?')}] {step.get('title', '')} [{step.get('tool', '')}] -> {status}"
                        )

                if result_data.get("results"):
                    print("\n  Variable results:")
                    for key, value in result_data["results"].items():
                        preview = str(value)[:60]
                        print(f"    ${key} = {preview}")
            else:
                print(f"  ERROR: {result_data.get('error', 'Unknown error')}")

            # Add tool result to messages
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_json,
                }
            )

        else:
            # Regular tool — execute directly via mock
            result = await tm.execute_tool(tool_name, arguments)
            result_text = (
                str(result.result) if result.success else f"Error: {result.error}"
            )
            print(f"    -> {result_text[:60]}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_text,
                }
            )

    # ── Turn 2: LLM summarizes results ──
    print("\n  [3] LLM summarizing results...")
    final = await client.create_completion(messages=messages)

    # Handle different response shapes from chuk_llm
    final_content = final.get("content", "") or ""
    if not final_content and "choices" in final:
        choices = final["choices"]
        if choices:
            final_content = choices[0].get("message", {}).get("content", "")

    if final_content:
        print(f"\n  Assistant: {final_content}")
    else:
        print(
            "\n  (LLM returned empty summary — tool results above speak for themselves)"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Plan-as-a-Tool demo (Tier 6.8)")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom user message (overrides built-in demos)",
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
    print("  Plan-as-a-Tool Demo (Tier 6.8)")
    print(f"  Model: {args.model}")
    print("  The LLM decides WHEN to plan — not the user")
    print("=" * 60)

    tm = MockToolManager()

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            # Multi-step task — should trigger plan_create_and_execute
            "Read the auth module at src/auth/handler.py, find all files that import it, and then run the tests to make sure everything passes.",
            # Single-step task — should use the tool directly
            "What's in the file src/auth/handler.py?",
        ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, prompt in enumerate(prompts, 1):
            section(f"Demo {i}")
            await run_conversation(
                model=args.model,
                user_message=prompt,
                tm=tm,
                plans_dir=Path(tmpdir) / f"plans_{i}",
            )

    print(f"\n{'=' * 60}")
    print("  Demo complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
