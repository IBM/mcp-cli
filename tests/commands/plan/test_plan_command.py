# tests/commands/plan/test_plan_command.py
"""Tests for PlanCommand — plan CRUD via the unified command interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_cli.commands.base import CommandMode
from mcp_cli.commands.plan.plan import PlanCommand, _planning_context_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeToolInfo:
    name: str


class FakeToolManager:
    """Minimal ToolManager stub for PlanCommand tests."""

    def __init__(self, tool_names: list[str] | None = None):
        self._tool_names = tool_names or ["read_file", "write_file", "search_code"]

    async def get_all_tools(self) -> list[FakeToolInfo]:
        return [FakeToolInfo(name=n) for n in self._tool_names]

    async def get_adapted_tools_for_llm(self, provider: str) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": n, "description": f"Tool: {n}"}}
            for n in self._tool_names
        ]


SAMPLE_PLAN = {
    "title": "Test Plan",
    "steps": [
        {
            "title": "Read file",
            "tool": "read_file",
            "args": {"path": "/tmp/test.py"},
            "depends_on": [],
            "result_variable": "file_content",
        },
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cmd():
    return PlanCommand()


@pytest.fixture
def tool_manager():
    return FakeToolManager()


@pytest.fixture(autouse=True)
def _clear_context_cache():
    """Clear the module-level planning context cache between tests."""
    _planning_context_cache.clear()
    yield
    _planning_context_cache.clear()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestPlanCommandProperties:
    def test_name(self, cmd):
        assert cmd.name == "plan"

    def test_aliases(self, cmd):
        assert "plans" in cmd.aliases

    def test_description(self, cmd):
        assert "plan" in cmd.description.lower()

    def test_help_text(self, cmd):
        text = cmd.help_text
        assert "/plan" in text
        assert "create" in text
        assert "list" in text
        assert "show" in text
        assert "run" in text
        assert "delete" in text
        assert "resume" in text

    def test_parameters(self, cmd):
        names = {p.name for p in cmd.parameters}
        assert "action" in names
        assert "plan_id_or_description" in names

    def test_modes(self, cmd):
        assert cmd.modes == CommandMode.ALL


# ---------------------------------------------------------------------------
# execute() — no tool_manager
# ---------------------------------------------------------------------------


class TestPlanCommandNoToolManager:
    @pytest.mark.asyncio
    async def test_no_tool_manager_returns_error(self, cmd):
        result = await cmd.execute(args="list")
        assert result.success is False
        assert "Tool manager not available" in result.error


# ---------------------------------------------------------------------------
# LIST action
# ---------------------------------------------------------------------------


class TestPlanListAction:
    @pytest.mark.asyncio
    async def test_list_empty(self, cmd, tool_manager, tmp_path):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.list_plans = AsyncMock(return_value=[])
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                result = await cmd.execute(args="list", tool_manager=tool_manager)
        assert result.success is True
        assert "No saved plans" in result.output

    @pytest.mark.asyncio
    async def test_list_with_plans(self, cmd, tool_manager):
        plans = [
            {
                "id": "abc12345-full-uuid",
                "title": "Test Plan",
                "steps": [{"title": "s1"}],
            },
        ]
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.list_plans = AsyncMock(return_value=plans)
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                with patch(
                    "mcp_cli.commands.plan.plan.format_table",
                    create=True,
                    return_value=MagicMock(),
                ):
                    result = await cmd.execute(args="list", tool_manager=tool_manager)
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 1

    @pytest.mark.asyncio
    async def test_default_action_is_list(self, cmd, tool_manager):
        """No args defaults to list."""
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.list_plans = AsyncMock(return_value=[])
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                result = await cmd.execute(args="", tool_manager=tool_manager)
        assert result.success is True


# ---------------------------------------------------------------------------
# SHOW action
# ---------------------------------------------------------------------------


class TestPlanShowAction:
    @pytest.mark.asyncio
    async def test_show_no_id(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="show", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan ID required" in result.error

    @pytest.mark.asyncio
    async def test_show_not_found(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=None)
            mock_ctx.return_value = ctx
            result = await cmd.execute(
                args="show nonexistent", tool_manager=tool_manager
            )
        assert result.success is False
        assert "Plan not found" in result.error

    @pytest.mark.asyncio
    async def test_show_found(self, cmd, tool_manager):
        plan_data = {
            "title": "My Plan",
            "steps": [{"title": "s1", "tool": "read_file"}],
        }
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan._display_plan"):
                result = await cmd.execute(
                    args="show abc123", tool_manager=tool_manager
                )
        assert result.success is True
        assert result.data == plan_data


# ---------------------------------------------------------------------------
# DELETE action
# ---------------------------------------------------------------------------


class TestPlanDeleteAction:
    @pytest.mark.asyncio
    async def test_delete_no_id(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="delete", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan ID required" in result.error

    @pytest.mark.asyncio
    async def test_delete_found(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.delete_plan = AsyncMock(return_value=True)
            mock_ctx.return_value = ctx
            with patch("chuk_term.ui.output") as mock_out:
                result = await cmd.execute(
                    args="delete abc123", tool_manager=tool_manager
                )
        assert result.success is True
        mock_out.success.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.delete_plan = AsyncMock(return_value=False)
            mock_ctx.return_value = ctx
            result = await cmd.execute(args="delete abc123", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan not found" in result.error


# ---------------------------------------------------------------------------
# RUN action
# ---------------------------------------------------------------------------


class TestPlanRunAction:
    @pytest.mark.asyncio
    async def test_run_no_id(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="run", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan ID required" in result.error

    @pytest.mark.asyncio
    async def test_run_not_found(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=None)
            mock_ctx.return_value = ctx
            result = await cmd.execute(args="run abc123", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan not found" in result.error

    @pytest.mark.asyncio
    async def test_run_dry_run_flag(self, cmd, tool_manager):
        """--dry-run flag should be detected."""
        plan_data = {"title": "My Plan", "steps": []}
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_ctx.return_value = ctx

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.steps = []
            mock_result.total_duration = 0.1

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.execute_plan = AsyncMock(return_value=mock_result)
                mock_runner_cls.return_value = mock_runner
                with patch("chuk_term.ui.output"):
                    result = await cmd.execute(
                        args="run abc123 --dry-run", tool_manager=tool_manager
                    )

        assert result.success is True
        # Verify dry_run=True was passed
        call_kwargs = mock_runner.execute_plan.call_args
        assert call_kwargs[1].get("dry_run") is True


# ---------------------------------------------------------------------------
# RESUME action
# ---------------------------------------------------------------------------


class TestPlanResumeAction:
    @pytest.mark.asyncio
    async def test_resume_no_id(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="resume", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan ID required" in result.error

    @pytest.mark.asyncio
    async def test_resume_plan_not_found(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=None)
            mock_ctx.return_value = ctx
            result = await cmd.execute(args="resume abc123", tool_manager=tool_manager)
        assert result.success is False
        assert "Plan not found" in result.error

    @pytest.mark.asyncio
    async def test_resume_no_checkpoint(self, cmd, tool_manager):
        plan_data = {"title": "My Plan", "steps": []}
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_ctx.return_value = ctx
            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = MagicMock()
                mock_runner.load_checkpoint.return_value = None
                mock_runner_cls.return_value = mock_runner
                result = await cmd.execute(
                    args="resume abc123", tool_manager=tool_manager
                )
        assert result.success is False
        assert "No checkpoint found" in result.error


# ---------------------------------------------------------------------------
# CREATE action
# ---------------------------------------------------------------------------


class TestPlanCreateAction:
    @pytest.mark.asyncio
    async def test_create_no_description(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="create", tool_manager=tool_manager)
        assert result.success is False
        assert "Description required" in result.error

    @pytest.mark.asyncio
    async def test_create_no_tools(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.get_tool_names = AsyncMock(return_value=[])
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                result = await cmd.execute(
                    args="create do something", tool_manager=tool_manager
                )
        assert result.success is False
        assert "No tools available" in result.error


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------


class TestPlanUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action(self, cmd, tool_manager):
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            mock_ctx.return_value = AsyncMock()
            result = await cmd.execute(args="bogus", tool_manager=tool_manager)
        assert result.success is False
        assert "Unknown action" in result.error


# ---------------------------------------------------------------------------
# Args parsing — list vs string
# ---------------------------------------------------------------------------


class TestPlanArgsParsing:
    @pytest.mark.asyncio
    async def test_args_as_list(self, cmd, tool_manager):
        """Chat adapter passes args as a list."""
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.list_plans = AsyncMock(return_value=[])
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                result = await cmd.execute(args=["list"], tool_manager=tool_manager)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_args_as_string(self, cmd, tool_manager):
        """Interactive adapter passes args as a string."""
        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_ctx:
            ctx = AsyncMock()
            ctx.list_plans = AsyncMock(return_value=[])
            mock_ctx.return_value = ctx
            with patch("mcp_cli.commands.plan.plan.output", create=True):
                result = await cmd.execute(args="list", tool_manager=tool_manager)
        assert result.success is True


# ---------------------------------------------------------------------------
# _get_planning_context — lines 152-157
# ---------------------------------------------------------------------------


class TestGetPlanningContext:
    @pytest.mark.asyncio
    async def test_creates_planning_context(self, cmd, tool_manager):
        """First call creates a new PlanningContext and caches it."""
        with patch("mcp_cli.planning.context.PlanningContext") as mock_pc:
            mock_instance = MagicMock()
            mock_pc.return_value = mock_instance

            result = await cmd._get_planning_context(tool_manager)

            mock_pc.assert_called_once_with(tool_manager)
            assert result is mock_instance

    @pytest.mark.asyncio
    async def test_caches_planning_context(self, cmd, tool_manager):
        """Second call returns cached context, does not re-create."""
        with patch("mcp_cli.planning.context.PlanningContext") as mock_pc:
            mock_instance = MagicMock()
            mock_pc.return_value = mock_instance

            r1 = await cmd._get_planning_context(tool_manager)
            r2 = await cmd._get_planning_context(tool_manager)

            mock_pc.assert_called_once()  # Only created once
            assert r1 is r2


# ---------------------------------------------------------------------------
# _create_plan success path — lines 208-243
# ---------------------------------------------------------------------------


class TestPlanCreateSuccess:
    @pytest.mark.asyncio
    async def test_create_plan_success(self, cmd, tool_manager):
        """Full success path: agent generates plan, context saves it."""
        tool_catalog = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "properties": {
                            "path": {"type": "string", "description": "File path"}
                        },
                        "required": ["path"],
                    },
                },
            }
        ]
        plan_dict = {
            "title": "My Plan",
            "steps": [
                {
                    "title": "Read file",
                    "tool": "read_file",
                    "args": {"path": "/tmp/test.py"},
                    "depends_on": [],
                    "result_variable": "fc",
                }
            ],
        }

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_tool_catalog = AsyncMock(return_value=tool_catalog)
            ctx.save_plan_from_dict = AsyncMock(return_value="plan-abc-123")
            ctx.get_plan = AsyncMock(return_value=plan_dict)
            mock_get_ctx.return_value = ctx

            with patch("chuk_ai_planner.agents.plan_agent.PlanAgent") as mock_agent_cls:
                mock_agent = AsyncMock()
                mock_agent.plan = AsyncMock(return_value=plan_dict)
                mock_agent_cls.return_value = mock_agent

                with patch("chuk_term.ui.output"):
                    with patch("mcp_cli.commands.plan.plan._display_plan"):
                        result = await cmd.execute(
                            args="create read the config file",
                            tool_manager=tool_manager,
                        )

        assert result.success is True
        assert result.data["plan_id"] == "plan-abc-123"
        assert "1 steps" in result.output
        ctx.save_plan_from_dict.assert_awaited_once_with(plan_dict)

    @pytest.mark.asyncio
    async def test_create_plan_agent_returns_empty(self, cmd, tool_manager):
        """Agent returns empty plan → error."""
        tool_catalog = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"properties": {}, "required": []},
                },
            }
        ]

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_tool_catalog = AsyncMock(return_value=tool_catalog)
            mock_get_ctx.return_value = ctx

            with patch("chuk_ai_planner.agents.plan_agent.PlanAgent") as mock_agent_cls:
                mock_agent = AsyncMock()
                mock_agent.plan = AsyncMock(return_value={"steps": []})
                mock_agent_cls.return_value = mock_agent

                with patch("chuk_term.ui.output"):
                    result = await cmd.execute(
                        args="create do stuff", tool_manager=tool_manager
                    )

        assert result.success is False
        assert "Failed to generate" in result.error

    @pytest.mark.asyncio
    async def test_create_plan_agent_exception(self, cmd, tool_manager):
        """Agent raises exception → error."""
        tool_catalog = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"properties": {}, "required": []},
                },
            }
        ]

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_tool_catalog = AsyncMock(return_value=tool_catalog)
            mock_get_ctx.return_value = ctx

            with patch("chuk_ai_planner.agents.plan_agent.PlanAgent") as mock_agent_cls:
                mock_agent = AsyncMock()
                mock_agent.plan = AsyncMock(side_effect=RuntimeError("LLM failed"))
                mock_agent_cls.return_value = mock_agent

                with patch("chuk_term.ui.output"):
                    result = await cmd.execute(
                        args="create do stuff", tool_manager=tool_manager
                    )

        assert result.success is False
        assert "Plan creation failed" in result.error


# ---------------------------------------------------------------------------
# _run_plan callbacks, success, failure — lines 287-323
# ---------------------------------------------------------------------------


class TestPlanRunCallbacks:
    """Test _run_plan with step callbacks and success/failure paths."""

    def _make_exec_result(self, *, success=True, error=None):
        """Create a mock PlanExecutionResult."""
        r = MagicMock()
        r.success = success
        r.steps = [MagicMock()]
        r.total_duration = 1.5
        r.variables = {"v1": "val"}
        r.error = error
        return r

    @pytest.mark.asyncio
    async def test_run_success_with_callbacks(self, cmd, tool_manager):
        """Success path: result.success=True, output.success is called."""
        plan_data = {
            "title": "My Plan",
            "steps": [{"title": "s1", "tool": "read_file", "depends_on": []}],
        }
        exec_result = self._make_exec_result(success=True)

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_get_ctx.return_value = ctx

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.execute_plan = AsyncMock(return_value=exec_result)
                mock_runner_cls.return_value = mock_runner

                with patch("chuk_term.ui.output") as mock_output:
                    result = await cmd.execute(
                        args="run abc123", tool_manager=tool_manager
                    )

        assert result.success is True
        assert "completed" in result.output
        mock_output.success.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_failure(self, cmd, tool_manager):
        """Failure path: result.success=False, output.error is called."""
        plan_data = {
            "title": "My Plan",
            "steps": [{"title": "s1", "tool": "read_file", "depends_on": []}],
        }
        exec_result = self._make_exec_result(success=False, error="Step 1 timed out")

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_get_ctx.return_value = ctx

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.execute_plan = AsyncMock(return_value=exec_result)
                mock_runner_cls.return_value = mock_runner

                with patch("chuk_term.ui.output") as mock_output:
                    result = await cmd.execute(
                        args="run abc123", tool_manager=tool_manager
                    )

        assert result.success is False
        assert "failed" in result.output
        mock_output.error.assert_called_once()
        assert "Step 1 timed out" in mock_output.error.call_args[0][0]

    @pytest.mark.asyncio
    async def test_run_step_callbacks_invoked(self, cmd, tool_manager):
        """Verify on_step_start / on_step_complete callbacks are wired."""
        plan_data = {"title": "P", "steps": [{"title": "s1"}]}
        exec_result = self._make_exec_result(success=True)

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_get_ctx.return_value = ctx

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.execute_plan = AsyncMock(return_value=exec_result)
                mock_runner_cls.return_value = mock_runner

                with patch("chuk_term.ui.output"):
                    await cmd.execute(args="run abc123", tool_manager=tool_manager)

                # Check PlanRunner was constructed with callbacks
                init_kwargs = mock_runner_cls.call_args[1]
                assert callable(init_kwargs["on_step_start"])
                assert callable(init_kwargs["on_step_complete"])
                assert callable(init_kwargs["on_tool_start"])
                assert callable(init_kwargs["on_tool_complete"])

    @pytest.mark.asyncio
    async def test_on_step_start_callback(self, cmd, tool_manager):
        """Exercise the on_step_start callback."""
        plan_data = {"title": "P", "steps": []}
        exec_result = self._make_exec_result(success=True)

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_get_ctx.return_value = ctx

            captured_callbacks = {}

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.execute_plan = AsyncMock(return_value=exec_result)

                def capture_init(*args, **kwargs):
                    captured_callbacks.update(kwargs)
                    return mock_runner

                mock_runner_cls.side_effect = capture_init

                with patch("chuk_term.ui.output") as mock_output:
                    await cmd.execute(args="run abc123", tool_manager=tool_manager)

                    # Call the captured callback
                    captured_callbacks["on_step_start"]("1", "Read file", "read_file")
                    mock_output.info.assert_any_call("  Step 1: Read file")

    @pytest.mark.asyncio
    async def test_on_step_complete_failure_callback(self, cmd, tool_manager):
        """Exercise on_step_complete callback with a failed step."""
        plan_data = {"title": "P", "steps": []}
        exec_result = self._make_exec_result(success=True)

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_get_ctx.return_value = ctx

            captured_callbacks = {}

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.execute_plan = AsyncMock(return_value=exec_result)

                def capture_init(*args, **kwargs):
                    captured_callbacks.update(kwargs)
                    return mock_runner

                mock_runner_cls.side_effect = capture_init

                with patch("chuk_term.ui.output") as mock_output:
                    await cmd.execute(args="run abc123", tool_manager=tool_manager)

                    # Create a failed step result
                    step_result = MagicMock()
                    step_result.success = False
                    step_result.step_index = "2"
                    step_result.error = "timeout"
                    captured_callbacks["on_step_complete"](step_result)
                    mock_output.error.assert_any_call("  Step 2 failed: timeout")

    @pytest.mark.asyncio
    async def test_on_tool_start_with_display(self, cmd, tool_manager):
        """Exercise on_tool_start callback with a display manager."""
        plan_data = {"title": "P", "steps": []}
        exec_result = self._make_exec_result(success=True)

        mock_display = AsyncMock()
        mock_ui_manager = MagicMock()
        mock_ui_manager.display = mock_display

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_get_ctx.return_value = ctx

            captured_callbacks = {}

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.execute_plan = AsyncMock(return_value=exec_result)

                def capture_init(*args, **kwargs):
                    captured_callbacks.update(kwargs)
                    return mock_runner

                mock_runner_cls.side_effect = capture_init

                with patch("chuk_term.ui.output"):
                    await cmd.execute(
                        args="run abc123",
                        tool_manager=tool_manager,
                        ui_manager=mock_ui_manager,
                    )

                    # Call the async callback
                    await captured_callbacks["on_tool_start"](
                        "read_file", {"path": "/tmp"}
                    )
                    mock_display.start_tool_execution.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_tool_complete_with_display(self, cmd, tool_manager):
        """Exercise on_tool_complete callback with a display manager."""
        plan_data = {"title": "P", "steps": []}
        exec_result = self._make_exec_result(success=True)

        mock_display = AsyncMock()
        mock_ui_manager = MagicMock()
        mock_ui_manager.display = mock_display

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_get_ctx.return_value = ctx

            captured_callbacks = {}

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.execute_plan = AsyncMock(return_value=exec_result)

                def capture_init(*args, **kwargs):
                    captured_callbacks.update(kwargs)
                    return mock_runner

                mock_runner_cls.side_effect = capture_init

                with patch("chuk_term.ui.output"):
                    await cmd.execute(
                        args="run abc123",
                        tool_manager=tool_manager,
                        ui_manager=mock_ui_manager,
                    )

                    await captured_callbacks["on_tool_complete"](
                        "read_file", "ok", True, 0.5
                    )
                    mock_display.stop_tool_execution.assert_awaited_once_with(
                        "ok", True
                    )


# ---------------------------------------------------------------------------
# _resume_plan with checkpoint — lines 371-394
# ---------------------------------------------------------------------------


class TestPlanResumeWithCheckpoint:
    @pytest.mark.asyncio
    async def test_resume_success(self, cmd, tool_manager):
        """Resume with checkpoint found, execution succeeds."""
        plan_data = {
            "title": "My Plan",
            "steps": [
                {"index": "1", "title": "s1", "tool": "a", "depends_on": []},
                {"index": "2", "title": "s2", "tool": "b", "depends_on": ["1"]},
            ],
        }
        checkpoint = {
            "completed_steps": ["1"],
            "variables": {"v1": "data"},
        }
        exec_result = MagicMock()
        exec_result.success = True
        exec_result.error = None

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_get_ctx.return_value = ctx

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = MagicMock()
                mock_runner.load_checkpoint = MagicMock(return_value=checkpoint)
                mock_runner.execute_plan = AsyncMock(return_value=exec_result)
                mock_runner_cls.return_value = mock_runner

                with patch("chuk_term.ui.output") as mock_output:
                    result = await cmd.execute(
                        args="resume abc123", tool_manager=tool_manager
                    )

        assert result.success is True
        mock_output.success.assert_called_once()
        # Verify execute_plan was called with remaining steps only
        call_args = mock_runner.execute_plan.call_args
        plan_arg = call_args[0][0]
        assert len(plan_arg["steps"]) == 1  # Only step 2 remains
        assert call_args[1]["variables"] == {"v1": "data"}

    @pytest.mark.asyncio
    async def test_resume_failure(self, cmd, tool_manager):
        """Resume with checkpoint found, execution fails."""
        plan_data = {
            "title": "Fail Plan",
            "steps": [
                {"index": "1", "title": "s1", "tool": "a", "depends_on": []},
                {"index": "2", "title": "s2", "tool": "b", "depends_on": ["1"]},
            ],
        }
        checkpoint = {
            "completed_steps": ["1"],
            "variables": {},
        }
        exec_result = MagicMock()
        exec_result.success = False
        exec_result.error = "step 2 blew up"

        with patch(
            "mcp_cli.commands.plan.plan.PlanCommand._get_planning_context"
        ) as mock_get_ctx:
            ctx = AsyncMock()
            ctx.get_plan = AsyncMock(return_value=plan_data)
            mock_get_ctx.return_value = ctx

            with patch("mcp_cli.planning.executor.PlanRunner") as mock_runner_cls:
                mock_runner = MagicMock()
                mock_runner.load_checkpoint = MagicMock(return_value=checkpoint)
                mock_runner.execute_plan = AsyncMock(return_value=exec_result)
                mock_runner_cls.return_value = mock_runner

                with patch("chuk_term.ui.output") as mock_output:
                    result = await cmd.execute(
                        args="resume abc123", tool_manager=tool_manager
                    )

        assert result.success is False
        mock_output.error.assert_called_once()
        assert "step 2 blew up" in mock_output.error.call_args[0][0]


# ---------------------------------------------------------------------------
# _build_plan_system_prompt — lines 405-426
# ---------------------------------------------------------------------------


class TestBuildPlanSystemPrompt:
    def test_basic_prompt(self):
        from mcp_cli.commands.plan.plan import _build_plan_system_prompt

        catalog = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from disk",
                    "parameters": {
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to read",
                            }
                        },
                        "required": ["path"],
                    },
                },
            }
        ]
        prompt = _build_plan_system_prompt(catalog)

        assert "read_file" in prompt
        assert "Read a file from disk" in prompt
        assert "path: string (required)" in prompt
        assert "File path to read" in prompt
        assert "planning assistant" in prompt
        assert "JSON object" in prompt

    def test_tool_without_params(self):
        from mcp_cli.commands.plan.plan import _build_plan_system_prompt

        catalog = [
            {
                "type": "function",
                "function": {
                    "name": "no_args_tool",
                    "description": "Takes nothing",
                    "parameters": {"properties": {}, "required": []},
                },
            }
        ]
        prompt = _build_plan_system_prompt(catalog)

        assert "no_args_tool" in prompt
        assert "(no parameters)" in prompt

    def test_multiple_tools(self):
        from mcp_cli.commands.plan.plan import _build_plan_system_prompt

        catalog = [
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "description": "Tool A",
                    "parameters": {"properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_b",
                    "description": "Tool B",
                    "parameters": {
                        "properties": {
                            "x": {"type": "integer", "description": "A number"},
                        },
                        "required": [],
                    },
                },
            },
        ]
        prompt = _build_plan_system_prompt(catalog)

        assert "tool_a" in prompt
        assert "tool_b" in prompt
        assert "x: integer" in prompt

    def test_optional_param(self):
        from mcp_cli.commands.plan.plan import _build_plan_system_prompt

        catalog = [
            {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "desc",
                    "parameters": {
                        "properties": {
                            "opt": {"type": "string", "description": "optional arg"},
                        },
                        "required": [],  # Not required
                    },
                },
            }
        ]
        prompt = _build_plan_system_prompt(catalog)

        # Should NOT contain "(required)"
        assert "opt: string —" in prompt
        assert "(required)" not in prompt.split("opt")[1].split("\n")[0]


# ---------------------------------------------------------------------------
# _validate_step — lines 457-462
# ---------------------------------------------------------------------------


class TestValidateStep:
    def test_valid_step(self):
        from mcp_cli.commands.plan.plan import _validate_step

        step = {"tool": "read_file", "title": "Read it"}
        ok, msg = _validate_step(step, ["read_file", "write_file"])
        assert ok is True
        assert msg == ""

    def test_unknown_tool(self):
        from mcp_cli.commands.plan.plan import _validate_step

        step = {"tool": "hack_planet", "title": "Hack"}
        ok, msg = _validate_step(step, ["read_file", "write_file"])
        assert ok is False
        assert "Unknown tool" in msg
        assert "hack_planet" in msg

    def test_missing_title(self):
        from mcp_cli.commands.plan.plan import _validate_step

        step = {"tool": "read_file", "title": ""}
        ok, msg = _validate_step(step, ["read_file"])
        assert ok is False
        assert "title" in msg.lower()

    def test_missing_tool_key(self):
        from mcp_cli.commands.plan.plan import _validate_step

        step = {"title": "No tool"}
        ok, msg = _validate_step(step, ["read_file"])
        assert ok is False
        assert "Unknown tool" in msg


# ---------------------------------------------------------------------------
# _display_plan — lines 467-482
# ---------------------------------------------------------------------------


class TestDisplayPlan:
    def test_display_plan_basic(self):
        from mcp_cli.commands.plan.plan import _display_plan

        plan_data = {
            "title": "Test Plan",
            "steps": [
                {"title": "Read file", "tool": "read_file", "depends_on": []},
            ],
        }
        with patch("chuk_term.ui.output") as mock_output:
            with patch(
                "mcp_cli.planning.executor.render_plan_dag", return_value="DAG-VIZ"
            ):
                _display_plan(plan_data)

        # Check title and DAG were printed
        calls = [str(c) for c in mock_output.info.call_args_list]
        assert any("Test Plan" in c and "1 steps" in c for c in calls)
        assert any("DAG-VIZ" in c for c in calls)

    def test_display_plan_with_result_variables(self):
        from mcp_cli.commands.plan.plan import _display_plan

        plan_data = {
            "title": "Var Plan",
            "steps": [
                {
                    "title": "Read",
                    "tool": "read_file",
                    "depends_on": [],
                    "result_variable": "content",
                },
                {
                    "title": "Search",
                    "tool": "search_code",
                    "depends_on": ["1"],
                    "result_variable": "results",
                },
            ],
        }
        with patch("chuk_term.ui.output") as mock_output:
            with patch("mcp_cli.planning.executor.render_plan_dag", return_value="DAG"):
                _display_plan(plan_data)

        # Should display variables
        calls = [str(c) for c in mock_output.info.call_args_list]
        assert any("content" in c and "results" in c for c in calls)

    def test_display_plan_no_result_variables(self):
        from mcp_cli.commands.plan.plan import _display_plan

        plan_data = {
            "title": "No Vars",
            "steps": [
                {"title": "s1", "tool": "t1", "depends_on": []},
            ],
        }
        with patch("chuk_term.ui.output") as mock_output:
            with patch("mcp_cli.planning.executor.render_plan_dag", return_value="DAG"):
                _display_plan(plan_data)

        # "Variables:" should NOT appear (no result_variable keys)
        calls = " ".join(str(c) for c in mock_output.info.call_args_list)
        assert "Variables:" not in calls
