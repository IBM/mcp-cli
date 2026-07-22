# src/mcp_cli/planning/__init__.py
"""Plan-based execution for mcp-cli.

Integrates chuk-ai-planner's graph-based plan DSL with mcp-cli's
MCP tool execution layer. Plans make workflows reproducible,
inspectable, and resumable.

Key components:
- McpToolBackend: bridges planner to ToolManager with guard integration
- PlanningContext: state container for plan operations
- PlanRunner: orchestrates plan execution with parallel batches, checkpointing, re-planning
- render_plan_dag: ASCII DAG visualization for terminal display
- PlanExecutionResult / StepResult: structured execution results
"""

from mcp_cli.planning.backends import McpToolBackend
from mcp_cli.planning.context import PlanningContext
from mcp_cli.planning.executor import (
    PlanRunner,
    PlanExecutionResult,
    StepResult,
    render_plan_dag,
)

__all__ = [
    "McpToolBackend",
    "PlanningContext",
    "PlanRunner",
    "PlanExecutionResult",
    "StepResult",
    "render_plan_dag",
]
