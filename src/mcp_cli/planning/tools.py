# src/mcp_cli/planning/tools.py
"""Plan tool definitions and handler for LLM tool interception.

Provides internal tools that the model can call to autonomously create
and execute plans during conversation. These tools are intercepted in
tool_processor.py before MCP routing (same pattern as VM and memory tools).

Tools:
- plan_create: Generate a plan from a goal description
- plan_execute: Execute a previously created plan
- plan_create_and_execute: Generate and execute in one call (common case)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp_cli.planning.context import PlanningContext

logger = logging.getLogger(__name__)

# Tool names for interception in tool_processor.py
_PLAN_TOOL_NAMES = frozenset({"plan_create", "plan_execute", "plan_create_and_execute"})


def get_plan_tools_as_dicts() -> list[dict[str, Any]]:
    """Return OpenAI-format tool definitions for plan tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "plan_create_and_execute",
                "description": (
                    "Create and execute a multi-step plan to accomplish a complex goal. "
                    "Use this when a task requires multiple tool calls that depend on each other "
                    "(e.g., geocode a location then get weather for those coordinates). "
                    "The plan is generated from your goal description, then executed automatically. "
                    "Results from all steps are returned."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": (
                                "Natural language description of what to accomplish. "
                                "Be specific about the end result you want."
                            ),
                        },
                    },
                    "required": ["goal"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plan_create",
                "description": (
                    "Generate a multi-step execution plan from a goal description "
                    "without executing it. Returns the plan ID and step details. "
                    "Use plan_execute to run it later, or plan_create_and_execute "
                    "to do both at once."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "Natural language description of the goal.",
                        },
                    },
                    "required": ["goal"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plan_execute",
                "description": (
                    "Execute a previously created plan by its ID. "
                    "Returns the results from all executed steps."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan_id": {
                            "type": "string",
                            "description": "The plan ID returned by plan_create.",
                        },
                    },
                    "required": ["plan_id"],
                },
            },
        },
    ]


async def handle_plan_tool(
    tool_name: str,
    arguments: dict[str, Any],
    planning_context: PlanningContext,
    model_manager: Any | None = None,
    ui_manager: Any | None = None,
) -> str:
    """Execute a plan tool and return the result as a JSON string.

    Args:
        tool_name: One of plan_create, plan_execute, plan_create_and_execute.
        arguments: Tool arguments from the LLM.
        planning_context: PlanningContext with tool_manager and plan registry.
        model_manager: Optional ModelManager for LLM-driven step execution.
        ui_manager: Optional UI manager for per-step progress display.

    Returns:
        JSON string with the result (for insertion into conversation history).
    """
    if tool_name == "plan_create":
        return await _handle_plan_create(arguments, planning_context)

    if tool_name == "plan_execute":
        return await _handle_plan_execute(
            arguments, planning_context, model_manager, ui_manager
        )

    if tool_name == "plan_create_and_execute":
        return await _handle_plan_create_and_execute(
            arguments, planning_context, model_manager, ui_manager
        )

    return json.dumps({"error": f"Unknown plan tool: {tool_name}"})


async def _handle_plan_create(
    arguments: dict[str, Any],
    context: PlanningContext,
) -> str:
    """Generate a plan from a goal description."""
    goal = arguments.get("goal", "")
    if not goal:
        return json.dumps({"error": "Goal description is required."})

    try:
        plan_dict = await _generate_plan(context, goal)
        if not plan_dict or not plan_dict.get("steps"):
            return json.dumps({"error": "Failed to generate a valid plan."})

        plan_id = await context.save_plan_from_dict(plan_dict)

        return json.dumps(
            {
                "success": True,
                "plan_id": plan_id,
                "title": plan_dict.get("title", "Untitled"),
                "steps": [
                    {
                        "index": s.get("index", i + 1),
                        "title": s.get("title", ""),
                        "tool": s.get("tool", ""),
                    }
                    for i, s in enumerate(plan_dict.get("steps", []))
                ],
            }
        )

    except Exception as e:
        logger.error("Plan creation failed: %s", e)
        return json.dumps({"error": f"Plan creation failed: {e}"})


async def _handle_plan_execute(
    arguments: dict[str, Any],
    context: PlanningContext,
    model_manager: Any | None = None,
    ui_manager: Any | None = None,
) -> str:
    """Execute a previously created plan."""
    plan_id = arguments.get("plan_id", "")
    if not plan_id:
        return json.dumps({"error": "plan_id is required."})

    plan_data = await context.get_plan(plan_id)
    if not plan_data:
        return json.dumps({"error": f"Plan not found: {plan_id}"})

    return await _run_plan(context, plan_data, model_manager, ui_manager)


async def _handle_plan_create_and_execute(
    arguments: dict[str, Any],
    context: PlanningContext,
    model_manager: Any | None = None,
    ui_manager: Any | None = None,
) -> str:
    """Generate a plan and execute it immediately."""
    goal = arguments.get("goal", "")
    if not goal:
        return json.dumps({"error": "Goal description is required."})

    # Show plan generation phase
    if ui_manager:
        try:
            await ui_manager.start_tool_execution(
                "plan_create_and_execute", {"phase": "generating plan..."}
            )
        except Exception:
            pass

    try:
        plan_dict = await _generate_plan(context, goal)

        # Finish the generation spinner
        if ui_manager:
            try:
                steps = plan_dict.get("steps", []) if plan_dict else []
                title = plan_dict.get("title", "Untitled") if plan_dict else "?"
                await ui_manager.finish_tool_execution(
                    result=f"Plan generated: {title} ({len(steps)} steps)",
                    success=bool(plan_dict and steps),
                )
            except Exception:
                pass

        if not plan_dict or not plan_dict.get("steps"):
            return json.dumps({"error": "Failed to generate a valid plan."})

        # Save so it can be resumed if interrupted
        plan_id = await context.save_plan_from_dict(plan_dict)
        plan_data = await context.get_plan(plan_id)
        if not plan_data:
            return json.dumps({"error": "Failed to load saved plan."})

        return await _run_plan(context, plan_data, model_manager, ui_manager)

    except Exception as e:
        # Make sure spinner is stopped on error
        if ui_manager:
            try:
                await ui_manager.finish_tool_execution(result=str(e), success=False)
            except Exception:
                pass
        logger.error("Plan create-and-execute failed: %s", e)
        return json.dumps({"error": f"Plan create-and-execute failed: {e}"})


async def _generate_plan(
    context: PlanningContext,
    goal: str,
) -> dict[str, Any] | None:
    """Generate a plan dict from a goal description using PlanAgent."""
    from chuk_ai_planner.agents.plan_agent import PlanAgent

    tool_catalog = await context.get_tool_catalog()
    tool_names = [
        t.get("function", {}).get("name", "")
        for t in tool_catalog
        if t.get("function", {}).get("name")
    ]
    if not tool_names:
        return None

    system_prompt = _build_plan_system_prompt(tool_catalog)

    agent = PlanAgent(
        system_prompt=system_prompt,
        validate_step=lambda step: _validate_step(step, tool_names),
        max_retries=2,
    )

    result: dict[str, Any] | None = await agent.plan(goal)
    return result


async def _run_plan(
    context: PlanningContext,
    plan_data: dict[str, Any],
    model_manager: Any | None = None,
    ui_manager: Any | None = None,
) -> str:
    """Execute a plan and return JSON results."""
    from mcp_cli.planning.executor import PlanRunner

    # Build per-step callbacks that drive the UI manager
    async def on_tool_start(tool_name: str, arguments: dict) -> None:
        if ui_manager:
            try:
                await ui_manager.start_tool_execution(tool_name, arguments)
            except Exception:
                pass

    async def on_tool_complete(
        tool_name: str, result_text: str, success: bool, elapsed: float
    ) -> None:
        if ui_manager:
            try:
                await ui_manager.finish_tool_execution(
                    result=result_text, success=success
                )
            except Exception:
                pass

    runner = PlanRunner(
        context,
        model_manager=model_manager,
        enable_guards=False,
        on_tool_start=on_tool_start,
        on_tool_complete=on_tool_complete,
    )

    result = await runner.execute_plan(plan_data, checkpoint=False)

    response: dict[str, Any] = {
        "success": result.success,
        "plan_id": result.plan_id,
        "title": result.plan_title,
        "duration": round(result.total_duration, 2),
        "steps_completed": len([s for s in result.steps if s.success]),
        "steps_total": len(result.steps),
    }

    if result.error:
        response["error"] = result.error

    # Include variable results (the useful output)
    if result.variables:
        response["results"] = result.variables

    # Include per-step summaries
    response["steps"] = [
        {
            "index": s.step_index,
            "title": s.step_title,
            "tool": s.tool_name,
            "success": s.success,
            "error": s.error,
        }
        for s in result.steps
    ]

    return json.dumps(response, default=str)


def _build_plan_system_prompt(tool_catalog: list[dict[str, Any]]) -> str:
    """Build the system prompt for LLM plan generation.

    Reuses the same logic as plan.py's _build_plan_system_prompt.
    """
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


def _validate_step(step: dict[str, Any], tool_names: list[str]) -> tuple[bool, str]:
    """Validate a plan step against available tools."""
    tool = step.get("tool", "")
    if tool not in tool_names:
        return False, f"Unknown tool: {tool}. Available: {', '.join(tool_names[:10])}"
    if not step.get("title"):
        return False, "Step must have a title"
    return True, ""
