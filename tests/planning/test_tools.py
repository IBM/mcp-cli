# tests/planning/test_tools.py
"""Tests for planning/tools.py — plan tool definitions and handler."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from mcp_cli.planning.tools import (
    _PLAN_TOOL_NAMES,
    _validate_step,
    get_plan_tools_as_dicts,
    handle_plan_tool,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class FakeToolInfo:
    name: str


class FakeToolManager:
    """Minimal ToolManager stub."""

    def __init__(self, tool_names: list[str] | None = None):
        self._tool_names = tool_names or ["read_file", "write_file"]

    async def get_all_tools(self) -> list[FakeToolInfo]:
        return [FakeToolInfo(name=n) for n in self._tool_names]

    async def get_adapted_tools_for_llm(self, provider: str) -> tuple[list[dict], dict]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": n,
                    "description": f"Tool: {n}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"}
                        },
                        "required": ["path"],
                    },
                },
            }
            for n in self._tool_names
        ]
        return tools, {}


class FakePlanningContext:
    """Minimal PlanningContext stub for tool tests."""

    def __init__(self, tool_names: list[str] | None = None):
        self.tool_manager = FakeToolManager(tool_names)
        self._saved_plans: dict[str, dict] = {}

    async def get_tool_catalog(self) -> list[dict[str, Any]]:
        tools, _ = await self.tool_manager.get_adapted_tools_for_llm("openai")
        return tools

    async def save_plan_from_dict(self, plan_dict: dict[str, Any]) -> str:
        plan_id = f"plan-{len(self._saved_plans) + 1}"
        self._saved_plans[plan_id] = plan_dict
        return plan_id

    async def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        return self._saved_plans.get(plan_id)


# ── Tests: tool definitions ──────────────────────────────────────────────────


class TestGetPlanToolsAsDicts:
    """Tests for get_plan_tools_as_dicts()."""

    def test_returns_three_tools(self):
        tools = get_plan_tools_as_dicts()
        assert len(tools) == 3

    def test_tool_names_match_frozenset(self):
        tools = get_plan_tools_as_dicts()
        names = {t["function"]["name"] for t in tools}
        assert names == _PLAN_TOOL_NAMES

    def test_all_have_function_type(self):
        tools = get_plan_tools_as_dicts()
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_plan_create_requires_goal(self):
        tools = get_plan_tools_as_dicts()
        create = next(t for t in tools if t["function"]["name"] == "plan_create")
        assert "goal" in create["function"]["parameters"]["properties"]
        assert "goal" in create["function"]["parameters"]["required"]

    def test_plan_execute_requires_plan_id(self):
        tools = get_plan_tools_as_dicts()
        execute = next(t for t in tools if t["function"]["name"] == "plan_execute")
        assert "plan_id" in execute["function"]["parameters"]["properties"]
        assert "plan_id" in execute["function"]["parameters"]["required"]

    def test_plan_create_and_execute_requires_goal(self):
        tools = get_plan_tools_as_dicts()
        combined = next(
            t for t in tools if t["function"]["name"] == "plan_create_and_execute"
        )
        assert "goal" in combined["function"]["parameters"]["properties"]
        assert "goal" in combined["function"]["parameters"]["required"]


class TestPlanToolNames:
    """Tests for the _PLAN_TOOL_NAMES frozenset."""

    def test_contains_expected_names(self):
        assert "plan_create" in _PLAN_TOOL_NAMES
        assert "plan_execute" in _PLAN_TOOL_NAMES
        assert "plan_create_and_execute" in _PLAN_TOOL_NAMES

    def test_is_frozenset(self):
        assert isinstance(_PLAN_TOOL_NAMES, frozenset)

    def test_has_exactly_three_entries(self):
        assert len(_PLAN_TOOL_NAMES) == 3


# ── Tests: _validate_step ────────────────────────────────────────────────────


class TestValidateStep:
    """Tests for the step validation function."""

    def test_valid_step(self):
        ok, msg = _validate_step(
            {"tool": "read_file", "title": "Read"}, ["read_file", "write_file"]
        )
        assert ok is True
        assert msg == ""

    def test_unknown_tool(self):
        ok, msg = _validate_step(
            {"tool": "hack_server", "title": "Hack"}, ["read_file"]
        )
        assert ok is False
        assert "Unknown tool" in msg

    def test_missing_title(self):
        ok, msg = _validate_step({"tool": "read_file", "title": ""}, ["read_file"])
        assert ok is False
        assert "title" in msg.lower()

    def test_missing_tool_key(self):
        ok, msg = _validate_step({"title": "Something"}, ["read_file"])
        assert ok is False
        assert "Unknown tool" in msg


# ── Tests: handle_plan_tool ──────────────────────────────────────────────────


class TestHandlePlanTool:
    """Tests for the main handle_plan_tool dispatch function."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        ctx = FakePlanningContext()
        result = await handle_plan_tool("plan_unknown", {}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Unknown plan tool" in parsed["error"]

    @pytest.mark.asyncio
    async def test_plan_create_missing_goal(self):
        ctx = FakePlanningContext()
        result = await handle_plan_tool("plan_create", {}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Goal" in parsed["error"] or "required" in parsed["error"].lower()

    @pytest.mark.asyncio
    async def test_plan_create_empty_goal(self):
        ctx = FakePlanningContext()
        result = await handle_plan_tool("plan_create", {"goal": ""}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_plan_execute_missing_plan_id(self):
        ctx = FakePlanningContext()
        result = await handle_plan_tool("plan_execute", {}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "plan_id" in parsed["error"]

    @pytest.mark.asyncio
    async def test_plan_execute_unknown_plan(self):
        ctx = FakePlanningContext()
        result = await handle_plan_tool("plan_execute", {"plan_id": "nonexistent"}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower()

    @pytest.mark.asyncio
    async def test_plan_create_and_execute_missing_goal(self):
        ctx = FakePlanningContext()
        result = await handle_plan_tool("plan_create_and_execute", {}, ctx)
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_plan_create_success(self):
        """Test plan_create with mocked PlanAgent."""
        ctx = FakePlanningContext(["read_file", "write_file"])

        fake_plan = {
            "title": "Test Plan",
            "steps": [
                {"title": "Read", "tool": "read_file", "args": {"path": "/tmp/test"}},
            ],
        }

        with patch("chuk_ai_planner.agents.plan_agent.PlanAgent") as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.plan = AsyncMock(return_value=fake_plan)

            result = await handle_plan_tool("plan_create", {"goal": "Read a file"}, ctx)

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert "plan_id" in parsed
        assert parsed["title"] == "Test Plan"
        assert len(parsed["steps"]) == 1

    @pytest.mark.asyncio
    async def test_plan_create_agent_returns_empty(self):
        """Test plan_create when PlanAgent returns no steps."""
        ctx = FakePlanningContext()

        with patch("chuk_ai_planner.agents.plan_agent.PlanAgent") as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.plan = AsyncMock(return_value={"title": "Empty", "steps": []})

            result = await handle_plan_tool("plan_create", {"goal": "Do nothing"}, ctx)

        parsed = json.loads(result)
        assert "error" in parsed
        assert "valid plan" in parsed["error"].lower()

    @pytest.mark.asyncio
    async def test_plan_create_agent_exception(self):
        """Test plan_create when PlanAgent raises."""
        ctx = FakePlanningContext()

        with patch("chuk_ai_planner.agents.plan_agent.PlanAgent") as MockAgent:
            mock_agent = MockAgent.return_value
            mock_agent.plan = AsyncMock(side_effect=RuntimeError("LLM is down"))

            result = await handle_plan_tool(
                "plan_create", {"goal": "Break things"}, ctx
            )

        parsed = json.loads(result)
        assert "error" in parsed
        assert "failed" in parsed["error"].lower()

    @pytest.mark.asyncio
    async def test_plan_create_and_execute_success(self):
        """Test plan_create_and_execute with mocked PlanAgent + PlanRunner."""
        ctx = FakePlanningContext(["read_file"])

        fake_plan = {
            "title": "Read Plan",
            "steps": [
                {"title": "Read", "tool": "read_file", "args": {"path": "/tmp/test"}},
            ],
        }

        @dataclass
        class FakeStepResult:
            step_index: int = 1
            step_title: str = "Read"
            tool_name: str = "read_file"
            success: bool = True
            error: str | None = None

        @dataclass
        class FakeExecResult:
            success: bool = True
            plan_id: str = "plan-1"
            plan_title: str = "Read Plan"
            total_duration: float = 0.5
            steps: list = None
            error: str | None = None
            variables: dict = None

            def __post_init__(self):
                if self.steps is None:
                    self.steps = [FakeStepResult()]
                if self.variables is None:
                    self.variables = {"step_1_result": "file contents"}

        with (
            patch("chuk_ai_planner.agents.plan_agent.PlanAgent") as MockAgent,
            patch("mcp_cli.planning.executor.PlanRunner") as MockRunner,
        ):
            mock_agent = MockAgent.return_value
            mock_agent.plan = AsyncMock(return_value=fake_plan)

            mock_runner = MockRunner.return_value
            mock_runner.execute_plan = AsyncMock(return_value=FakeExecResult())

            result = await handle_plan_tool(
                "plan_create_and_execute", {"goal": "Read a file"}, ctx
            )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["steps_completed"] == 1
        assert "results" in parsed

    @pytest.mark.asyncio
    async def test_plan_execute_success(self):
        """Test plan_execute with a saved plan and mocked PlanRunner."""
        ctx = FakePlanningContext(["read_file"])

        # Pre-save a plan
        plan_data = {
            "title": "Saved Plan",
            "steps": [{"title": "Read", "tool": "read_file", "args": {"path": "/tmp"}}],
        }
        plan_id = await ctx.save_plan_from_dict(plan_data)

        @dataclass
        class FakeStepResult:
            step_index: int = 1
            step_title: str = "Read"
            tool_name: str = "read_file"
            success: bool = True
            error: str | None = None

        @dataclass
        class FakeExecResult:
            success: bool = True
            plan_id: str = "plan-1"
            plan_title: str = "Saved Plan"
            total_duration: float = 0.3
            steps: list = None
            error: str | None = None
            variables: dict = None

            def __post_init__(self):
                if self.steps is None:
                    self.steps = [FakeStepResult()]
                if self.variables is None:
                    self.variables = {}

        with patch("mcp_cli.planning.executor.PlanRunner") as MockRunner:
            mock_runner = MockRunner.return_value
            mock_runner.execute_plan = AsyncMock(return_value=FakeExecResult())

            result = await handle_plan_tool("plan_execute", {"plan_id": plan_id}, ctx)

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["steps_total"] == 1
