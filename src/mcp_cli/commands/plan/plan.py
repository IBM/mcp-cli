# src/mcp_cli/commands/plan/plan.py
"""Plan management command — create, list, show, run, delete, resume plans."""

from __future__ import annotations

import logging

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.config.enums import PlanAction

logger = logging.getLogger(__name__)

# Module-level cache: tool_manager id → PlanningContext
_planning_context_cache: dict[int, object] = {}


class PlanCommand(UnifiedCommand):
    """Manage execution plans."""

    @property
    def name(self) -> str:
        return "plan"

    @property
    def aliases(self) -> list[str]:
        return ["plans"]

    @property
    def description(self) -> str:
        return "Create, inspect, and execute plans"

    @property
    def help_text(self) -> str:
        return """
Manage execution plans — reproducible, inspectable tool call graphs.

Usage:
  /plan                          - List all saved plans
  /plan list                     - List all saved plans
  /plan create <description>     - Generate a plan from a description
  /plan show <id>                - Show plan details
  /plan run <id>                 - Execute a plan
  /plan run <id> --dry-run       - Trace without executing
  /plan delete <id>              - Delete a plan
  /plan resume <id>              - Resume an interrupted plan
"""

    @property
    def modes(self) -> CommandMode:
        return CommandMode.ALL

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="action",
                type=str,
                required=False,
                help="Action: create, list, show, run, delete, resume",
            ),
            CommandParameter(
                name="plan_id_or_description",
                type=str,
                required=False,
                help="Plan ID or description (for create)",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the plan command."""
        # Parse args — chat adapter passes a list, interactive passes a string
        args_val = kwargs.get("args", "")
        if isinstance(args_val, list):
            args_str = " ".join(str(a) for a in args_val)
        else:
            args_str = str(args_val).strip()
        parts = args_str.split(maxsplit=1)
        action = parts[0] if parts else PlanAction.LIST
        remainder = parts[1] if len(parts) > 1 else ""

        # Get tool_manager from kwargs
        tool_manager = kwargs.get("tool_manager")
        if not tool_manager:
            return CommandResult(
                success=False,
                error="Tool manager not available. Plans require an active session.",
            )

        # Lazy-create planning context
        planning_context = await self._get_planning_context(tool_manager)

        if action == PlanAction.LIST:
            return await self._list_plans(planning_context)

        elif action == PlanAction.CREATE:
            if not remainder:
                return CommandResult(
                    success=False,
                    error="Description required. Usage: /plan create <description>",
                )
            return await self._create_plan(planning_context, remainder, kwargs)

        elif action == PlanAction.SHOW:
            if not remainder:
                return CommandResult(
                    success=False,
                    error="Plan ID required. Usage: /plan show <id>",
                )
            return await self._show_plan(planning_context, remainder)

        elif action == PlanAction.RUN:
            if not remainder:
                return CommandResult(
                    success=False,
                    error="Plan ID required. Usage: /plan run <id>",
                )
            dry_run = "--dry-run" in remainder or "--simulate" in remainder
            plan_id = remainder.split()[0]
            return await self._run_plan(
                planning_context, plan_id, dry_run=dry_run, kwargs=kwargs
            )

        elif action == PlanAction.DELETE:
            if not remainder:
                return CommandResult(
                    success=False,
                    error="Plan ID required. Usage: /plan delete <id>",
                )
            return await self._delete_plan(planning_context, remainder.strip())

        elif action == PlanAction.RESUME:
            if not remainder:
                return CommandResult(
                    success=False,
                    error="Plan ID required. Usage: /plan resume <id>",
                )
            return await self._resume_plan(planning_context, remainder.strip())

        else:
            return CommandResult(
                success=False,
                error=f"Unknown action: {action}. Use create, list, show, run, delete, or resume.",
            )

    async def _get_planning_context(self, tool_manager):
        """Get or create PlanningContext."""
        from mcp_cli.planning.context import PlanningContext

        tm_id = id(tool_manager)
        if tm_id not in _planning_context_cache:
            _planning_context_cache[tm_id] = PlanningContext(tool_manager)
        return _planning_context_cache[tm_id]

    async def _list_plans(self, context) -> CommandResult:
        """List all saved plans."""
        from chuk_term.ui import output, format_table

        plans = await context.list_plans()
        if not plans:
            output.info("No saved plans. Use /plan create <description> to create one.")
            return CommandResult(success=True, output="No saved plans.")

        table_data = []
        for p in plans:
            step_count = len(p.get("steps", []))
            table_data.append(
                {
                    "ID": p.get("id", "?")[:8] + "...",
                    "Title": p.get("title", "Untitled")[:50],
                    "Steps": str(step_count),
                }
            )

        table = format_table(
            table_data,
            title="Saved Plans",
            columns=["ID", "Title", "Steps"],
        )
        output.print_table(table)
        return CommandResult(success=True, data=table_data)

    async def _create_plan(self, context, description: str, kwargs) -> CommandResult:
        """Generate a plan from a natural language description."""
        from chuk_term.ui import output

        output.info(f"Generating plan: {description}")

        try:
            # Get full tool catalog (with parameter schemas) for the system prompt
            tool_catalog = await context.get_tool_catalog()
            tool_names = [
                t.get("function", {}).get("name", "")
                for t in tool_catalog
                if t.get("function", {}).get("name")
            ]
            if not tool_names:
                return CommandResult(
                    success=False,
                    error="No tools available. Connect to MCP servers first.",
                )

            # Build system prompt with tool schemas
            system_prompt = _build_plan_system_prompt(tool_catalog)

            # Use PlanAgent to generate the plan
            from chuk_ai_planner.agents.plan_agent import PlanAgent

            agent = PlanAgent(
                system_prompt=system_prompt,
                validate_step=lambda step: _validate_step(step, tool_names),
                max_retries=2,
            )

            plan_dict = await agent.plan(description)

            if not plan_dict or not plan_dict.get("steps"):
                return CommandResult(
                    success=False,
                    error="Failed to generate a valid plan.",
                )

            # Save via PlanningContext (builds UniversalPlan + registers)
            plan_id = await context.save_plan_from_dict(plan_dict)

            # Display the saved plan (has patched 1-based dependencies)
            saved_plan = await context.get_plan(plan_id)
            _display_plan(saved_plan or plan_dict)

            output.success(f"Plan saved: {plan_id[:8]}...")
            return CommandResult(
                success=True,
                output=f"Plan '{plan_dict.get('title', 'Untitled')}' created with {len(plan_dict['steps'])} steps.",
                data={"plan_id": plan_id},
            )

        except Exception as e:
            logger.error("Plan creation failed: %s", e)
            return CommandResult(
                success=False,
                error=f"Plan creation failed: {e}",
            )

    async def _show_plan(self, context, plan_id: str) -> CommandResult:
        """Show details of a plan."""
        plan_data = await context.get_plan(plan_id)
        if not plan_data:
            return CommandResult(
                success=False,
                error=f"Plan not found: {plan_id}",
            )

        _display_plan(plan_data)
        return CommandResult(success=True, data=plan_data)

    async def _run_plan(
        self,
        context,
        plan_id: str,
        *,
        dry_run: bool = False,
        kwargs: dict | None = None,
    ) -> CommandResult:
        """Execute a plan."""
        from chuk_term.ui import output
        from mcp_cli.planning.executor import PlanRunner

        plan_data = await context.get_plan(plan_id)
        if not plan_data:
            return CommandResult(
                success=False,
                error=f"Plan not found: {plan_id}",
            )

        mode_label = "[DRY RUN] " if dry_run else ""
        output.info(f"{mode_label}Executing plan: {plan_data.get('title', 'Untitled')}")

        # Get display manager for tool execution rendering (matches regular chat display)
        ui_manager = (kwargs or {}).get("ui_manager")
        display = getattr(ui_manager, "display", None) if ui_manager else None

        def on_step_start(index, title, tool):
            output.info(f"  Step {index}: {title}")

        def on_step_complete(step_result):
            # Tool results are shown by on_tool_complete — step complete is a summary
            if not step_result.success:
                output.error(
                    f"  Step {step_result.step_index} failed: {step_result.error}"
                )

        async def on_tool_start(tool_name, arguments):
            if display:
                await display.start_tool_execution(tool_name, arguments)

        async def on_tool_complete(tool_name, result_text, success, elapsed):
            if display:
                await display.stop_tool_execution(result_text, success)

        # Get model_manager for LLM-driven execution
        model_manager = (kwargs or {}).get("model_manager")

        runner = PlanRunner(
            context,
            model_manager=model_manager,
            on_step_start=on_step_start,
            on_step_complete=on_step_complete,
            on_tool_start=on_tool_start,
            on_tool_complete=on_tool_complete,
        )

        result = await runner.execute_plan(plan_data, dry_run=dry_run)

        if result.success:
            output.success(
                f"Plan completed: {len(result.steps)} steps in {result.total_duration:.1f}s"
            )
        else:
            output.error(f"Plan failed: {result.error or 'unknown error'}")

        return CommandResult(
            success=result.success,
            output=f"{'[DRY RUN] ' if dry_run else ''}Plan {'completed' if result.success else 'failed'}",
            data={
                "plan_result": {
                    "success": result.success,
                    "steps": len(result.steps),
                    "duration": result.total_duration,
                    "variables": result.variables,
                }
            },
        )

    async def _delete_plan(self, context, plan_id: str) -> CommandResult:
        """Delete a plan."""
        from chuk_term.ui import output

        if await context.delete_plan(plan_id):
            output.success(f"Plan deleted: {plan_id}")
            return CommandResult(success=True, output=f"Deleted {plan_id}")
        return CommandResult(
            success=False,
            error=f"Plan not found: {plan_id}",
        )

    async def _resume_plan(self, context, plan_id: str) -> CommandResult:
        """Resume an interrupted plan."""
        from chuk_term.ui import output
        from mcp_cli.planning.executor import PlanRunner

        plan_data = await context.get_plan(plan_id)
        if not plan_data:
            return CommandResult(
                success=False,
                error=f"Plan not found: {plan_id}",
            )

        runner = PlanRunner(context)
        checkpoint = runner.load_checkpoint(plan_id)

        if not checkpoint:
            return CommandResult(
                success=False,
                error=f"No checkpoint found for plan {plan_id}. Use /plan run instead.",
            )

        completed = checkpoint.get("completed_steps", [])
        output.info(
            f"Resuming plan: {plan_data.get('title', 'Untitled')} "
            f"({len(completed)} steps already completed)"
        )

        # Filter out completed steps
        remaining_steps = [
            s for s in plan_data.get("steps", []) if s.get("index") not in completed
        ]
        plan_data = dict(plan_data)  # Don't mutate the original
        plan_data["steps"] = remaining_steps

        result = await runner.execute_plan(
            plan_data,
            variables=checkpoint.get("variables", {}),
        )

        if result.success:
            output.success("Plan resumed and completed successfully.")
        else:
            output.error(f"Plan resume failed: {result.error}")

        return CommandResult(success=result.success)


def _build_plan_system_prompt(tool_catalog: list[dict]) -> str:
    """Build the system prompt for LLM plan generation.

    Args:
        tool_catalog: Full tool definitions with schemas (OpenAI function format).
    """

    # Format each tool with its parameter schema
    tool_lines = []
    for tool in tool_catalog:
        func = tool.get("function", {})
        name = func.get("name", "?")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            req = " (required)" if pname in required else ""
            param_parts.append(f"      {pname}: {ptype}{req} — {pdesc}")

        params_str = "\n".join(param_parts) if param_parts else "      (no parameters)"
        tool_lines.append(f"  {name}: {desc}\n    Parameters:\n{params_str}")

    tools_text = "\n\n".join(tool_lines)

    return f"""You are a planning assistant. Given a task description, create a structured execution plan.

Available tools (with parameter schemas):

{tools_text}

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
- Use the EXACT parameter names shown in the tool schemas
- depends_on is a list of step indices (0-based) that must complete first
- result_variable stores the output for use in later steps as ${{var_name}}
- Keep plans focused — prefer fewer, targeted steps over many small ones
- Each step should have exactly one tool call"""


def _validate_step(step: dict, tool_names: list[str]) -> tuple[bool, str]:
    """Validate a plan step against available tools."""
    tool = step.get("tool", "")
    if tool not in tool_names:
        return False, f"Unknown tool: {tool}. Available: {', '.join(tool_names[:10])}"
    if not step.get("title"):
        return False, "Step must have a title"
    return True, ""


def _display_plan(plan_data: dict) -> None:
    """Display a plan in the terminal with DAG visualization."""
    from chuk_term.ui import output
    from mcp_cli.planning.executor import render_plan_dag

    title = plan_data.get("title", "Untitled Plan")
    steps = plan_data.get("steps", [])

    output.info(f"\nPlan: {title} ({len(steps)} steps)\n")

    # Render DAG
    dag = render_plan_dag(plan_data)
    output.info(dag)

    result_vars = [s.get("result_variable") for s in steps if s.get("result_variable")]
    if result_vars:
        output.info(f"\nVariables: {', '.join(result_vars)}")
    output.info("")
