# tests/planning/test_executor.py
"""Tests for PlanRunner — plan execution with parallel batches, guards, DAG viz,
checkpoints, variable resolution, and agentic LLM-driven execution."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_cli.planning.executor import (
    PlanRunner,
    PlanExecutionResult,
    ModelManagerProtocol,
    render_plan_dag,
    _maybe_await,
    _serialize_variables,
    _summarize_variables,
    _compute_batches,
    _resolve_variables,
    _resolve_value,
    _extract_tool_call,
    _parse_tool_call_entry,
)
from mcp_cli.planning.context import PlanningContext


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class FakeToolCallResult:
    tool_name: str
    success: bool = True
    result: Any = "mock result"
    error: str | None = None


class FakeToolInfo:
    def __init__(self, name):
        self.name = name


class FakeToolManager:
    """Minimal ToolManager stub."""

    def __init__(self, results: dict[str, Any] | None = None, *, delay: float = 0):
        self._results = results or {}
        self._delay = delay
        self.calls: list[tuple[str, dict]] = []

    def get_all_tools(self):
        return [FakeToolInfo(n) for n in self._results.keys()] if self._results else []

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        self.calls.append((tool_name, arguments))
        result = self._results.get(tool_name, "default result")
        if isinstance(result, Exception):
            return FakeToolCallResult(
                tool_name=tool_name, success=False, error=str(result)
            )
        return FakeToolCallResult(tool_name=tool_name, result=result)


class FailingToolManager(FakeToolManager):
    """ToolManager that fails on specific tools."""

    def __init__(self, fail_tools: set[str], results: dict[str, Any] | None = None):
        super().__init__(results or {})
        self._fail_tools = fail_tools

    async def execute_tool(self, tool_name, arguments, namespace=None, timeout=None):
        self.calls.append((tool_name, arguments))
        if tool_name in self._fail_tools:
            return FakeToolCallResult(
                tool_name=tool_name, success=False, error=f"{tool_name} failed"
            )
        result = self._results.get(tool_name, "default result")
        return FakeToolCallResult(tool_name=tool_name, result=result)


SAMPLE_PLAN = {
    "id": "test-plan-001",
    "title": "Test Plan",
    "steps": [
        {
            "index": "1",
            "title": "Read file",
            "tool_calls": [
                {"id": "tc-1", "name": "read_file", "args": {"path": "test.py"}}
            ],
            "depends_on": [],
            "result_variable": "file_content",
        },
        {
            "index": "2",
            "title": "Search code",
            "tool_calls": [
                {"id": "tc-2", "name": "search_code", "args": {"query": "def main"}}
            ],
            "depends_on": ["1"],
            "result_variable": "search_results",
        },
    ],
    "variables": {},
}

PARALLEL_PLAN = {
    "id": "test-plan-parallel",
    "title": "Parallel Plan",
    "steps": [
        {
            "index": "1",
            "title": "Read file A",
            "tool_calls": [
                {"id": "tc-1", "name": "read_file", "args": {"path": "a.py"}}
            ],
            "depends_on": [],
            "result_variable": "file_a",
        },
        {
            "index": "2",
            "title": "Read file B",
            "tool_calls": [
                {"id": "tc-2", "name": "read_file", "args": {"path": "b.py"}}
            ],
            "depends_on": [],
            "result_variable": "file_b",
        },
        {
            "index": "3",
            "title": "Merge results",
            "tool_calls": [{"id": "tc-3", "name": "merge", "args": {}}],
            "depends_on": ["1", "2"],
            "result_variable": "merged",
        },
    ],
    "variables": {},
}

DIAMOND_PLAN = {
    "id": "test-plan-diamond",
    "title": "Diamond Plan",
    "steps": [
        {
            "index": "1",
            "title": "Init",
            "tool_calls": [{"id": "tc-1", "name": "init", "args": {}}],
            "depends_on": [],
            "result_variable": "init_result",
        },
        {
            "index": "2",
            "title": "Branch A",
            "tool_calls": [{"id": "tc-2", "name": "branch_a", "args": {}}],
            "depends_on": ["1"],
            "result_variable": "branch_a_result",
        },
        {
            "index": "3",
            "title": "Branch B",
            "tool_calls": [{"id": "tc-3", "name": "branch_b", "args": {}}],
            "depends_on": ["1"],
            "result_variable": "branch_b_result",
        },
        {
            "index": "4",
            "title": "Branch C",
            "tool_calls": [{"id": "tc-4", "name": "branch_c", "args": {}}],
            "depends_on": ["1"],
            "result_variable": "branch_c_result",
        },
        {
            "index": "5",
            "title": "Join",
            "tool_calls": [{"id": "tc-5", "name": "join", "args": {}}],
            "depends_on": ["2", "3", "4"],
            "result_variable": "join_result",
        },
    ],
    "variables": {},
}

VARS_PLAN = {
    "id": "test-plan-vars",
    "title": "Variable Plan",
    "variables": {"base_url": "http://localhost:8080"},
    "steps": [
        {
            "index": "1",
            "title": "Fetch users",
            "tool_calls": [
                {"id": "tc-1", "name": "fetch", "args": {"url": "${base_url}/users"}}
            ],
            "depends_on": [],
            "result_variable": "users",
        },
        {
            "index": "2",
            "title": "Process users",
            "tool_calls": [
                {"id": "tc-2", "name": "process", "args": {"data": "${users}"}}
            ],
            "depends_on": ["1"],
            "result_variable": "processed",
        },
    ],
}


# ── Tests: Dry Run ───────────────────────────────────────────────────────────


class TestPlanRunnerDryRun:
    """Test dry-run mode — trace without executing."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_all_steps(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, dry_run=True)

        assert result.success
        assert result.plan_id == "test-plan-001"
        assert result.plan_title == "Test Plan"
        assert len(result.steps) == 2
        assert result.steps[0].step_title == "Read file"
        assert result.steps[0].tool_name == "read_file"
        assert result.steps[1].step_title == "Search code"

    @pytest.mark.asyncio
    async def test_dry_run_marks_not_executed(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, dry_run=True)

        for step in result.steps:
            assert step.result == "[dry-run: not executed]"

    @pytest.mark.asyncio
    async def test_dry_run_simulates_variables(self, tmp_path):
        """Dry run should simulate variable binding."""
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, dry_run=True)

        assert "file_content" in result.variables
        assert "search_results" in result.variables

    @pytest.mark.asyncio
    async def test_dry_run_callbacks(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        started = []
        completed = []

        runner = PlanRunner(
            context,
            on_step_start=lambda i, t, tn: started.append((i, t, tn)),
            on_step_complete=lambda sr: completed.append(sr.step_title),
            enable_guards=False,
        )

        await runner.execute_plan(SAMPLE_PLAN, dry_run=True)

        assert len(started) == 2
        assert started[0] == ("1", "Read file", "read_file")
        assert len(completed) == 2


# ── Tests: Live Execution ──────────────────────────────────────────────────


class TestPlanRunnerExecution:
    """Test live plan execution with the new parallel batch engine."""

    @pytest.mark.asyncio
    async def test_linear_execution(self, tmp_path):
        """Sequential plan executes all steps in order."""
        tm = FakeToolManager({"read_file": "file data", "search_code": "found main"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert result.success
        assert len(result.steps) == 2
        assert result.steps[0].success
        assert result.steps[0].tool_name == "read_file"
        assert result.steps[1].success
        assert result.steps[1].tool_name == "search_code"
        assert result.total_duration > 0

    @pytest.mark.asyncio
    async def test_variable_binding(self, tmp_path):
        """Result variables are stored and available to later steps."""
        tm = FakeToolManager(
            {"read_file": "file contents", "search_code": "search hits"}
        )
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert result.variables.get("file_content") == "file contents"
        assert result.variables.get("search_results") == "search hits"

    @pytest.mark.asyncio
    async def test_parallel_execution(self, tmp_path):
        """Independent steps run in the same batch."""
        tm = FakeToolManager(
            {"read_file": "data", "merge": "merged"},
            delay=0.01,  # Small delay to verify concurrency
        )
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(PARALLEL_PLAN, checkpoint=False)

        assert result.success
        assert len(result.steps) == 3
        # Steps 1 and 2 should have been in the same batch
        assert result.steps[0].success
        assert result.steps[1].success
        assert result.steps[2].success

    @pytest.mark.asyncio
    async def test_diamond_execution(self, tmp_path):
        """Diamond DAG (1 → 2,3,4 → 5) executes correctly."""
        tm = FakeToolManager(
            {
                "init": "initialized",
                "branch_a": "A",
                "branch_b": "B",
                "branch_c": "C",
                "join": "joined",
            }
        )
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(DIAMOND_PLAN, checkpoint=False)

        assert result.success
        assert len(result.steps) == 5
        # All branches should have completed
        assert result.variables.get("init_result") == "initialized"
        assert result.variables.get("branch_a_result") == "A"
        assert result.variables.get("branch_b_result") == "B"
        assert result.variables.get("branch_c_result") == "C"
        assert result.variables.get("join_result") == "joined"

    @pytest.mark.asyncio
    async def test_execution_with_variables(self, tmp_path):
        """Variable overrides are passed through."""
        tm = FakeToolManager({"fetch": "[user1, user2]", "process": "done"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(
            VARS_PLAN,
            variables={"base_url": "http://api.example.com"},
            checkpoint=False,
        )

        assert result.success
        # The overridden base_url should have been used
        assert result.variables.get("base_url") == "http://api.example.com"

    @pytest.mark.asyncio
    async def test_step_failure_stops_execution(self, tmp_path):
        """When a step fails, execution stops and error is reported."""
        tm = FailingToolManager({"search_code"}, {"read_file": "data"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert not result.success
        assert "search_code failed" in result.error

    @pytest.mark.asyncio
    async def test_empty_plan(self, tmp_path):
        """Empty plan succeeds immediately."""
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(
            {"id": "empty", "title": "Empty", "steps": []},
            checkpoint=False,
        )

        assert result.success
        assert len(result.steps) == 0

    @pytest.mark.asyncio
    async def test_execution_callbacks(self, tmp_path):
        """Callbacks fire for each step during live execution."""
        tm = FakeToolManager({"read_file": "data", "search_code": "found"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        started = []
        completed = []

        runner = PlanRunner(
            context,
            on_step_start=lambda i, t, tn: started.append((i, t, tn)),
            on_step_complete=lambda sr: completed.append(sr.step_title),
            enable_guards=False,
        )

        await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert len(started) == 2
        assert len(completed) == 2
        assert started[0] == ("1", "Read file", "read_file")

    @pytest.mark.asyncio
    async def test_checkpoint_after_execution(self, tmp_path):
        """Execution checkpoints are saved after each batch."""
        tm = FakeToolManager({"read_file": "data", "search_code": "found"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=True)

        assert result.success
        checkpoint_path = context.plans_dir / "test-plan-001_state.json"
        assert checkpoint_path.exists()

        data = json.loads(checkpoint_path.read_text())
        assert data["status"] == "completed"
        assert "1" in data["completed_steps"]
        assert "2" in data["completed_steps"]

    @pytest.mark.asyncio
    async def test_tool_field_fallback(self, tmp_path):
        """Steps with 'tool' field (not 'tool_calls') work correctly."""
        tm = FakeToolManager({"my_tool": "result"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        plan = {
            "id": "tool-field",
            "title": "Tool Field Plan",
            "steps": [
                {
                    "index": "1",
                    "title": "Do thing",
                    "tool": "my_tool",
                    "args": {"x": 1},
                },
            ],
        }

        result = await runner.execute_plan(plan, checkpoint=False)
        assert result.success
        assert result.steps[0].tool_name == "my_tool"


# ── Tests: Topological Batching ────────────────────────────────────────────


class TestComputeBatches:
    """Test _compute_batches topological sort."""

    def test_empty_steps(self):
        assert _compute_batches([]) == []

    def test_single_step(self):
        steps = [{"index": "1", "title": "A"}]
        batches = _compute_batches(steps)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_linear_chain(self):
        """A → B → C produces 3 batches of 1."""
        steps = [
            {"index": "1", "title": "A"},
            {"index": "2", "title": "B", "depends_on": ["1"]},
            {"index": "3", "title": "C", "depends_on": ["2"]},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 3
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1
        assert len(batches[2]) == 1

    def test_parallel_steps(self):
        """Two independent steps produce 1 batch of 2."""
        steps = [
            {"index": "1", "title": "A"},
            {"index": "2", "title": "B"},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_diamond_dag(self):
        """Diamond: 1 → (2,3) → 4 produces 3 batches."""
        steps = [
            {"index": "1", "title": "Root"},
            {"index": "2", "title": "Left", "depends_on": ["1"]},
            {"index": "3", "title": "Right", "depends_on": ["1"]},
            {"index": "4", "title": "Join", "depends_on": ["2", "3"]},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 3
        assert len(batches[0]) == 1  # Root
        assert len(batches[1]) == 2  # Left, Right (parallel)
        assert len(batches[2]) == 1  # Join

    def test_wide_dag(self):
        """5 independent roots + 1 join = 2 batches."""
        steps = [
            {"index": "1", "title": "A"},
            {"index": "2", "title": "B"},
            {"index": "3", "title": "C"},
            {"index": "4", "title": "D"},
            {"index": "5", "title": "E"},
            {"index": "6", "title": "Join", "depends_on": ["1", "2", "3", "4", "5"]},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 2
        assert len(batches[0]) == 5  # All roots parallel
        assert len(batches[1]) == 1  # Join

    def test_missing_dependency_ignored(self):
        """Dependencies on non-existent steps are ignored."""
        steps = [
            {"index": "1", "title": "A", "depends_on": ["99"]},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 1

    def test_auto_assigns_index(self):
        """Steps without explicit index get auto-assigned."""
        steps = [
            {"title": "A"},
            {"title": "B"},
        ]
        batches = _compute_batches(steps)
        assert len(batches) == 1
        assert len(batches[0]) == 2


# ── Tests: Variable Resolution ─────────────────────────────────────────────


class TestVariableResolution:
    """Test ${var} resolution in tool arguments."""

    def test_no_variables(self):
        result = _resolve_variables({"path": "/tmp/test.py"}, {})
        assert result == {"path": "/tmp/test.py"}

    def test_simple_variable(self):
        result = _resolve_variables(
            {"path": "${target_path}"},
            {"target_path": "/tmp/test.py"},
        )
        assert result == {"path": "/tmp/test.py"}

    def test_template_string(self):
        result = _resolve_variables(
            {"url": "${base}/api/${version}"},
            {"base": "http://localhost", "version": "v2"},
        )
        assert result == {"url": "http://localhost/api/v2"}

    def test_nested_path(self):
        result = _resolve_variables(
            {"port": "${config.server.port}"},
            {"config": {"server": {"port": 8080}}},
        )
        assert result == {"port": 8080}

    def test_unresolved_variable_preserved(self):
        result = _resolve_variables(
            {"x": "${missing}"},
            {},
        )
        assert result == {"x": "${missing}"}

    def test_preserves_type(self):
        """Single ${var} reference preserves the original type (not stringified)."""
        result = _resolve_variables(
            {"count": "${n}"},
            {"n": 42},
        )
        assert result == {"count": 42}

    def test_list_values(self):
        result = _resolve_variables(
            {"items": ["${a}", "${b}"]},
            {"a": "alpha", "b": "beta"},
        )
        assert result == {"items": ["alpha", "beta"]}

    def test_nested_dict(self):
        result = _resolve_variables(
            {"opts": {"key": "${val}"}},
            {"val": "resolved"},
        )
        assert result == {"opts": {"key": "resolved"}}

    def test_non_string_passthrough(self):
        result = _resolve_value(42, {"x": 1})
        assert result == 42

    def test_none_passthrough(self):
        result = _resolve_value(None, {})
        assert result is None

    def test_bool_passthrough(self):
        result = _resolve_value(True, {})
        assert result is True


# ── Tests: Checkpointing ────────────────────────────────────────────────────


class TestPlanRunnerCheckpoint:
    """Test execution checkpointing."""

    @pytest.mark.asyncio
    async def test_checkpoint_saved(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        runner._save_checkpoint(
            "test-plan-001",
            completed_steps=["1", "2"],
            variables={"file_content": "hello"},
            status="completed",
        )

        checkpoint_path = context.plans_dir / "test-plan-001_state.json"
        assert checkpoint_path.exists()

        data = json.loads(checkpoint_path.read_text())
        assert data["plan_id"] == "test-plan-001"
        assert data["status"] == "completed"
        assert data["completed_steps"] == ["1", "2"]

    @pytest.mark.asyncio
    async def test_checkpoint_loaded(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        runner._save_checkpoint(
            "test-plan-002",
            completed_steps=["1"],
            variables={"x": "y"},
            status="running",
        )

        checkpoint = runner.load_checkpoint("test-plan-002")
        assert checkpoint is not None
        assert checkpoint["status"] == "running"
        assert checkpoint["completed_steps"] == ["1"]

    @pytest.mark.asyncio
    async def test_checkpoint_not_found(self, tmp_path):
        tm = FakeToolManager()
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        assert runner.load_checkpoint("nonexistent") is None

    @pytest.mark.asyncio
    async def test_failed_execution_checkpoints(self, tmp_path):
        """Failed execution should save a checkpoint with 'failed' status."""
        tm = FailingToolManager({"search_code"}, {"read_file": "data"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=True)

        assert not result.success
        checkpoint_path = context.plans_dir / "test-plan-001_state.json"
        assert checkpoint_path.exists()

        data = json.loads(checkpoint_path.read_text())
        assert data["status"] == "failed"
        assert "1" in data["completed_steps"]


# ── Tests: DAG Visualization ────────────────────────────────────────────────


class TestRenderPlanDag:
    """Test ASCII DAG rendering."""

    def test_empty_plan(self):
        result = render_plan_dag({"steps": []})
        assert "empty plan" in result

    def test_linear_plan(self):
        dag = render_plan_dag(SAMPLE_PLAN)
        assert "Read file" in dag
        assert "Search code" in dag
        assert "read_file" in dag
        assert "search_code" in dag
        assert "after: 1" in dag

    def test_parallel_plan(self):
        dag = render_plan_dag(PARALLEL_PLAN)
        assert "Read file A" in dag
        assert "Read file B" in dag
        assert "Merge results" in dag
        assert "after: 1, 2" in dag

    def test_parallel_marker(self):
        """Parallel steps should have ∥ marker."""
        dag = render_plan_dag(PARALLEL_PLAN)
        # Steps 1 and 2 are parallel — should have ∥ marker
        assert "∥" in dag

    def test_status_indicators(self):
        plan = {
            "steps": [
                {
                    "index": "1",
                    "title": "Done step",
                    "tool_calls": [{"name": "tool_a"}],
                    "_status": "completed",
                },
                {
                    "index": "2",
                    "title": "Running step",
                    "tool_calls": [{"name": "tool_b"}],
                    "_status": "running",
                    "depends_on": ["1"],
                },
                {
                    "index": "3",
                    "title": "Pending step",
                    "tool_calls": [{"name": "tool_c"}],
                    "_status": "pending",
                    "depends_on": ["2"],
                },
                {
                    "index": "4",
                    "title": "Failed step",
                    "tool_calls": [{"name": "tool_d"}],
                    "_status": "failed",
                },
            ]
        }
        dag = render_plan_dag(plan)
        assert "●" in dag  # completed
        assert "◉" in dag  # running
        assert "○" in dag  # pending
        assert "✗" in dag  # failed

    def test_tool_field_fallback(self):
        """Handles steps with 'tool' field instead of 'tool_calls'."""
        plan = {
            "steps": [
                {"index": "1", "title": "Step A", "tool": "my_tool"},
            ]
        }
        dag = render_plan_dag(plan)
        assert "my_tool" in dag


# ── Tests: LLM-Driven Execution ───────────────────────────────────────────


class TestLLMDrivenExecution:
    """Test agentic LLM-driven step execution.

    The agentic loop feeds ALL tool results (success and failure) back to the
    LLM, letting it evaluate results and decide whether to retry, try
    different parameters, or signal step completion with a text response.
    """

    def _make_model_manager(self, responses: list[dict]) -> MagicMock:
        """Create a mock ModelManager with a sequence of LLM responses.

        Each response can have 'tool_calls' (LLM wants to call a tool)
        or just 'content' (LLM signals step complete).
        """
        client = AsyncMock()
        client.create_completion = AsyncMock(side_effect=responses)

        mm = MagicMock(spec=ModelManagerProtocol)
        mm.get_client.return_value = client
        return mm

    def _tool_call_response(self, name: str, args: dict) -> dict:
        """Build a mock LLM response containing a tool call.

        Uses the chuk_llm native format (top-level tool_calls key).
        """
        return {
            "response": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 15},
        }

    def _text_response(self, text: str = "Step complete.") -> dict:
        """Build a mock LLM response with text only (no tool call).

        Uses the chuk_llm native format (top-level response key).
        """
        return {
            "response": text,
            "tool_calls": None,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    @pytest.mark.asyncio
    async def test_no_model_manager_uses_static(self, tmp_path):
        """Without model_manager, falls back to static arg execution."""
        tm = FakeToolManager(results={"read_file": "file content"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        plan = {
            "id": "test",
            "title": "Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Read",
                    "tool": "read_file",
                    "args": {"path": "/tmp/x"},
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        assert result.success
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_failure_without_llm_stops_plan(self, tmp_path):
        """Without LLM, a failed step stops the plan."""
        tm = FailingToolManager({"search_code"}, {"read_file": "data"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)

        result = await runner.execute_plan(SAMPLE_PLAN, checkpoint=False)

        assert not result.success

    @pytest.mark.asyncio
    async def test_agentic_loop_success_then_text(self, tmp_path):
        """LLM calls tool, sees result, then responds with text = step done."""
        tm = FakeToolManager(results={"geocode_location": {"lat": 52.0, "lon": 0.85}})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        mm = self._make_model_manager(
            [
                # Turn 1: LLM calls the tool
                self._tool_call_response("geocode_location", {"name": "Leavenheath"}),
                # Turn 2: LLM sees the successful result, responds with text
                self._text_response("Geocoded: lat=52.0, lon=0.85"),
            ]
        )

        runner = PlanRunner(context, model_manager=mm, enable_guards=False)

        plan = {
            "id": "test",
            "title": "Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Geocode location",
                    "tool": "geocode_location",
                    "args": {"name": "Leavenheath"},
                    "result_variable": "geo_result",
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        assert result.success
        assert result.steps[0].success
        assert result.variables["geo_result"] == {"lat": 52.0, "lon": 0.85}

    @pytest.mark.asyncio
    async def test_agentic_loop_retry_on_failure(self, tmp_path):
        """LLM calls tool with wrong args, sees error, retries with correct args."""
        call_count = 0

        async def smart_execute(tool_name, arguments, namespace=None, timeout=None):
            nonlocal call_count
            call_count += 1
            if arguments.get("latitude") and isinstance(arguments["latitude"], str):
                return FakeToolCallResult(
                    tool_name=tool_name,
                    success=False,
                    error="expected number, got str",
                )
            return FakeToolCallResult(
                tool_name=tool_name,
                result={"temperature": 15.5},
            )

        tm = FakeToolManager()
        tm.execute_tool = smart_execute
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        mm = self._make_model_manager(
            [
                # Turn 1: LLM passes string lat/lon
                self._tool_call_response(
                    "get_weather", {"latitude": "52.0", "longitude": "0.85"}
                ),
                # Turn 2: LLM sees error, retries with numbers
                self._tool_call_response(
                    "get_weather", {"latitude": 52.0, "longitude": 0.85}
                ),
                # Turn 3: LLM sees success, signals completion
                self._text_response("Weather: 15.5°C"),
            ]
        )

        runner = PlanRunner(context, model_manager=mm, enable_guards=False)

        plan = {
            "id": "test",
            "title": "Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Get weather",
                    "tool": "get_weather",
                    "args": {"latitude": 52.0, "longitude": 0.85},
                    "result_variable": "weather",
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        assert result.success
        assert result.variables["weather"] == {"temperature": 15.5}
        assert call_count == 2  # First call failed, second succeeded

    @pytest.mark.asyncio
    async def test_agentic_loop_retry_on_empty_result(self, tmp_path):
        """LLM sees null/empty result and retries with different params."""
        call_count = 0

        async def retry_execute(tool_name, arguments, namespace=None, timeout=None):
            nonlocal call_count
            call_count += 1
            if arguments.get("name") == "Leavenheath, Suffolk":
                return FakeToolCallResult(tool_name=tool_name, result={"results": None})
            # Simpler name works
            return FakeToolCallResult(
                tool_name=tool_name, result={"results": [{"lat": 52.0}]}
            )

        tm = FakeToolManager()
        tm.execute_tool = retry_execute
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        mm = self._make_model_manager(
            [
                # Turn 1: LLM tries full name
                self._tool_call_response("geocode", {"name": "Leavenheath, Suffolk"}),
                # Turn 2: LLM sees null results, tries simpler name
                self._tool_call_response("geocode", {"name": "Leavenheath"}),
                # Turn 3: LLM sees good result, signals done
                self._text_response("Found coordinates."),
            ]
        )

        runner = PlanRunner(context, model_manager=mm, enable_guards=False)

        plan = {
            "id": "test",
            "title": "Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Geocode",
                    "tool": "geocode",
                    "args": {"name": "Leavenheath, Suffolk"},
                    "result_variable": "geo",
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        assert result.success
        assert result.variables["geo"] == {"results": [{"lat": 52.0}]}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_agentic_loop_max_turns_exhausted_with_result(self, tmp_path):
        """When max turns exhausted but we have a result, return success."""
        tm = FakeToolManager(results={"tool_a": "data"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        # LLM keeps calling tools until turns run out (never sends text)
        mm = self._make_model_manager(
            [
                self._tool_call_response("tool_a", {"x": 1}),
                self._tool_call_response("tool_a", {"x": 2}),
                self._tool_call_response("tool_a", {"x": 3}),
            ]
        )

        runner = PlanRunner(
            context, model_manager=mm, enable_guards=False, max_step_retries=2
        )

        plan = {
            "id": "test",
            "title": "Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Do thing",
                    "tool": "tool_a",
                    "args": {"x": 1},
                    "result_variable": "out",
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        # Should succeed because we got a result
        assert result.success
        assert result.variables["out"] == "data"

    @pytest.mark.asyncio
    async def test_agentic_loop_max_turns_exhausted_no_result(self, tmp_path):
        """When max turns exhausted with no success, return failure."""
        tm = FailingToolManager({"tool_a"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        mm = self._make_model_manager(
            [
                self._tool_call_response("tool_a", {"x": 1}),
                self._tool_call_response("tool_a", {"x": 2}),
                self._tool_call_response("tool_a", {"x": 3}),
            ]
        )

        runner = PlanRunner(
            context, model_manager=mm, enable_guards=False, max_step_retries=2
        )

        plan = {
            "id": "test",
            "title": "Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Do thing",
                    "tool": "tool_a",
                    "args": {"x": 1},
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        assert not result.success

    @pytest.mark.asyncio
    async def test_replanned_flag_on_result(self, tmp_path):
        """PlanExecutionResult supports replanned flag."""
        result = PlanExecutionResult(
            plan_id="test",
            plan_title="Test",
            success=True,
            replanned=True,
        )
        assert result.replanned is True

    @pytest.mark.asyncio
    async def test_model_manager_protocol(self):
        """ModelManagerProtocol is satisfied by objects with get_client()."""
        mm = MagicMock()
        mm.get_client = MagicMock(return_value=AsyncMock())
        assert isinstance(mm, ModelManagerProtocol)


# ── Tests: Serialize Variables ───────────────────────────────────────────────


class TestSerializeVariables:
    """Test _serialize_variables helper."""

    def test_short_values_preserved(self):
        result = _serialize_variables({"x": "hello", "n": 42})
        assert result == {"x": "hello", "n": 42}

    def test_long_string_truncated(self):
        long_str = "a" * 2000
        result = _serialize_variables({"data": long_str})
        assert result["data"].endswith("... [truncated]")
        assert len(result["data"]) < 1100

    def test_large_dict_summarized(self):
        big_dict = {f"key_{i}": f"val_{i}" for i in range(200)}
        result = _serialize_variables({"config": big_dict})
        assert "dict" in result["config"]

    def test_small_dict_preserved(self):
        small_dict = {"a": 1, "b": 2}
        result = _serialize_variables({"config": small_dict})
        assert result["config"] == {"a": 1, "b": 2}


# ── Tests: Summarize Variables ───────────────────────────────────────────────


class TestSummarizeVariables:
    """Test _summarize_variables helper."""

    def test_empty_context(self):
        assert _summarize_variables({}) == "none"

    def test_single_variable(self):
        result = _summarize_variables({"geo": {"lat": 52.0}})
        assert "${geo}" in result
        assert "52.0" in result

    def test_multiple_variables(self):
        result = _summarize_variables({"a": 1, "b": "hello"})
        assert "${a}" in result
        assert "${b}" in result
        assert "1" in result
        assert "hello" in result

    def test_long_value_truncated(self):
        long_val = {"data": "x" * 1000}
        result = _summarize_variables({"big": long_val})
        assert "..." in result


# ── Tests: Maybe Await ───────────────────────────────────────────────────────


class TestMaybeAwait:
    """Test _maybe_await helper for sync/async callback support."""

    @pytest.mark.asyncio
    async def test_sync_value_returned(self):
        result = await _maybe_await(42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_none_returned(self):
        result = await _maybe_await(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_coroutine_awaited(self):
        async def async_fn():
            return "async_result"

        result = await _maybe_await(async_fn())
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_sync_callback_result(self):
        def sync_callback(x):
            return x * 2

        result = await _maybe_await(sync_callback(5))
        assert result == 10

    @pytest.mark.asyncio
    async def test_async_callback_result(self):
        async def async_callback(x):
            return x * 2

        result = await _maybe_await(async_callback(5))
        assert result == 10


# ── Tests: Extract Tool Call ──────────────────────────────────────────────────


class TestExtractToolCall:
    """Test _extract_tool_call with different response formats."""

    def test_chuk_llm_native_format_with_tool_calls(self):
        """chuk_llm returns {"response": null, "tool_calls": [...], "usage": {...}}."""
        response = {
            "response": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "geocode_location",
                        "arguments": '{"name": "London"}',
                    },
                }
            ],
            "usage": {"prompt_tokens": 53, "completion_tokens": 15},
        }
        result = _extract_tool_call(response)
        assert result is not None
        assert result["name"] == "geocode_location"
        assert result["args"] == {"name": "London"}

    def test_chuk_llm_native_format_text_response(self):
        """chuk_llm text response: {"response": "text", "tool_calls": null}."""
        response = {
            "response": "The temperature is 12.3 degrees.",
            "tool_calls": None,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
        result = _extract_tool_call(response)
        assert result is None

    def test_chuk_llm_native_format_empty_tool_calls(self):
        """chuk_llm with empty tool_calls list."""
        response = {
            "response": "Done",
            "tool_calls": [],
            "usage": {},
        }
        result = _extract_tool_call(response)
        assert result is None

    def test_openai_format_with_tool_calls(self):
        """OpenAI-style: {"choices": [{"message": {"tool_calls": [...]}}]}."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"lat": 52.0}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        result = _extract_tool_call(response)
        assert result is not None
        assert result["name"] == "get_weather"
        assert result["args"] == {"lat": 52.0}

    def test_openai_format_text_response(self):
        """OpenAI-style text response: no tool_calls in message."""
        response = {"choices": [{"message": {"content": "Hello!"}}]}
        result = _extract_tool_call(response)
        assert result is None

    def test_none_response(self):
        assert _extract_tool_call(None) is None

    def test_empty_dict(self):
        assert _extract_tool_call({}) is None

    def test_dict_args_not_string(self):
        """Arguments already parsed as dict (not JSON string)."""
        response = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "arguments": {"key": "value"},
                    },
                }
            ],
        }
        result = _extract_tool_call(response)
        assert result is not None
        assert result["args"] == {"key": "value"}

    def test_invalid_json_arguments(self):
        """Malformed JSON in arguments defaults to empty dict."""
        response = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "arguments": "not valid json{",
                    },
                }
            ],
        }
        result = _extract_tool_call(response)
        assert result is not None
        assert result["name"] == "test_tool"
        assert result["args"] == {}


class TestParseToolCallEntry:
    """Test _parse_tool_call_entry with dict and object formats."""

    def test_dict_entry(self):
        tc = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "my_tool", "arguments": '{"x": 1}'},
        }
        result = _parse_tool_call_entry(tc)
        assert result == {"name": "my_tool", "args": {"x": 1}}

    def test_object_entry(self):
        class FuncObj:
            name = "my_tool"
            arguments = '{"x": 1}'

        class TCObj:
            function = FuncObj()

        result = _parse_tool_call_entry(TCObj())
        assert result == {"name": "my_tool", "args": {"x": 1}}

    def test_no_function_returns_none(self):
        result = _parse_tool_call_entry("not a tool call")
        assert result is None

    def test_empty_name_returns_none(self):
        tc = {"function": {"name": "", "arguments": "{}"}}
        result = _parse_tool_call_entry(tc)
        assert result is None


# ── Tests: Error Paths & Edge Cases ──────────────────────────────────────────


class TestAgenticLoopErrorPaths:
    """Error paths in the agentic LLM loop.

    Covers: RuntimeError guard, LLM exceptions mid-loop, tool callbacks
    with async/sync variants, and turn-0 text fallback.
    """

    def _make_model_manager(self, responses: list[dict]) -> MagicMock:
        client = AsyncMock()
        client.create_completion = AsyncMock(side_effect=responses)
        mm = MagicMock(spec=ModelManagerProtocol)
        mm.get_client.return_value = client
        return mm

    def _tool_call_response(self, name: str, args: dict) -> dict:
        return {
            "response": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 15},
        }

    def _text_response(self, text: str = "Step complete.") -> dict:
        return {
            "response": text,
            "tool_calls": None,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    @pytest.mark.asyncio
    async def test_missing_model_manager_raises_runtime_error(self, tmp_path):
        """_execute_step_with_llm raises RuntimeError without model_manager."""
        tm = FakeToolManager(results={"tool_a": "ok"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")
        runner = PlanRunner(context, enable_guards=False)  # No model_manager

        with pytest.raises(RuntimeError, match="requires model_manager"):
            await runner._execute_step_with_llm(
                step={"index": "1", "title": "X", "tool": "tool_a", "args": {}},
                var_context={},
                step_index="1",
                step_title="X",
                hint_tool="tool_a",
                hint_args={},
            )

    @pytest.mark.asyncio
    async def test_llm_exception_mid_loop_retries(self, tmp_path):
        """If the LLM raises an exception on one turn, the loop retries."""
        tm = FakeToolManager(results={"tool_a": "result_data"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        client = AsyncMock()
        client.create_completion = AsyncMock(
            side_effect=[
                # Turn 1: LLM raises an exception
                RuntimeError("LLM connection failed"),
                # Turn 2: LLM succeeds with a tool call
                self._tool_call_response("tool_a", {"x": 1}),
                # Turn 3: LLM signals done
                self._text_response("Done"),
            ]
        )
        mm = MagicMock(spec=ModelManagerProtocol)
        mm.get_client.return_value = client

        runner = PlanRunner(context, model_manager=mm, enable_guards=False)

        plan = {
            "id": "exc-test",
            "title": "Exception Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Do thing",
                    "tool": "tool_a",
                    "args": {"x": 1},
                    "result_variable": "out",
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        assert result.success
        assert result.variables["out"] == "result_data"

    @pytest.mark.asyncio
    async def test_turn_0_text_falls_back_to_static(self, tmp_path):
        """If LLM responds with text on turn 0, fall back to static execution."""
        tm = FakeToolManager(results={"tool_a": "static_result"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        mm = self._make_model_manager(
            [
                self._text_response("I don't need to call a tool"),
            ]
        )

        runner = PlanRunner(context, model_manager=mm, enable_guards=False)

        plan = {
            "id": "fallback-test",
            "title": "Fallback Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Do thing",
                    "tool": "tool_a",
                    "args": {"key": "val"},
                    "result_variable": "out",
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        assert result.success
        assert result.variables["out"] == "static_result"
        # Should have used static execution, so one tool call
        assert len(tm.calls) == 1
        assert tm.calls[0] == ("tool_a", {"key": "val"})

    @pytest.mark.asyncio
    async def test_tool_callbacks_fire_in_agentic_loop(self, tmp_path):
        """on_tool_start / on_tool_complete fire for each tool call in loop."""
        tm = FakeToolManager(results={"tool_a": "ok"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        mm = self._make_model_manager(
            [
                self._tool_call_response("tool_a", {"x": 1}),
                self._tool_call_response("tool_a", {"x": 2}),
                self._text_response("Done"),
            ]
        )

        tool_starts: list[tuple[str, dict]] = []
        tool_completes: list[tuple[str, bool]] = []

        runner = PlanRunner(
            context,
            model_manager=mm,
            enable_guards=False,
            on_tool_start=lambda name, args: tool_starts.append((name, args)),
            on_tool_complete=lambda name, result, ok, elapsed: tool_completes.append(
                (name, ok)
            ),
        )

        plan = {
            "id": "cb-test",
            "title": "Callback Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Do thing",
                    "tool": "tool_a",
                    "args": {"x": 1},
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        assert result.success
        assert len(tool_starts) == 2
        assert tool_starts[0] == ("tool_a", {"x": 1})
        assert tool_starts[1] == ("tool_a", {"x": 2})
        assert len(tool_completes) == 2
        assert all(ok for _, ok in tool_completes)

    @pytest.mark.asyncio
    async def test_async_tool_callbacks(self, tmp_path):
        """Async tool callbacks are properly awaited."""
        tm = FakeToolManager(results={"tool_a": "ok"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        mm = self._make_model_manager(
            [
                self._tool_call_response("tool_a", {"x": 1}),
                self._text_response("Done"),
            ]
        )

        started = []
        completed = []

        async def on_start(name, args):
            started.append(name)

        async def on_complete(name, result, ok, elapsed):
            completed.append((name, ok))

        runner = PlanRunner(
            context,
            model_manager=mm,
            enable_guards=False,
            on_tool_start=on_start,
            on_tool_complete=on_complete,
        )

        plan = {
            "id": "async-cb-test",
            "title": "Async Callback Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Do thing",
                    "tool": "tool_a",
                    "args": {"x": 1},
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        assert result.success
        assert started == ["tool_a"]
        assert completed == [("tool_a", True)]

    @pytest.mark.asyncio
    async def test_text_on_later_turn_without_result_fails(self, tmp_path):
        """LLM text on turn > 0 but no prior success → step failure."""
        tm = FailingToolManager({"tool_a"})
        context = PlanningContext(tm, plans_dir=tmp_path / "plans")

        mm = self._make_model_manager(
            [
                # Turn 1: tool call fails
                self._tool_call_response("tool_a", {"x": 1}),
                # Turn 2: LLM gives up with text (no good result stored)
                self._text_response("I could not complete this step."),
            ]
        )

        runner = PlanRunner(context, model_manager=mm, enable_guards=False)

        plan = {
            "id": "fail-test",
            "title": "Fail Test",
            "steps": [
                {
                    "index": "1",
                    "title": "Do thing",
                    "tool": "tool_a",
                    "args": {"x": 1},
                },
            ],
        }
        result = await runner.execute_plan(plan, checkpoint=False)

        assert not result.success
        assert result.steps[0].error is not None
