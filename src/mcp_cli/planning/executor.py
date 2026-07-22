# src/mcp_cli/planning/executor.py
"""PlanRunner — orchestrates LLM-driven plan execution.

Executes plans step-by-step with:
- LLM-driven tool call generation with tool schemas
- Automatic retry on failure with LLM error correction
- Parallel batch execution for independent steps (topological batching)
- Progress callbacks for terminal/dashboard display
- Dry-run mode (trace without executing)
- Execution checkpointing and resume
- DAG visualization
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Coroutine
from typing import Any, Protocol, Union, runtime_checkable

from pydantic import BaseModel, Field

from chuk_ai_planner.execution.models import ToolExecutionRequest

from mcp_cli.chat.models import MessageRole
from mcp_cli.config.defaults import (
    DEFAULT_PLAN_CHECKPOINT_MAX_CHARS,
    DEFAULT_PLAN_DAG_TITLE_MAX_CHARS,
    DEFAULT_PLAN_MAX_CONCURRENCY,
    DEFAULT_PLAN_MAX_STEP_RETRIES,
    DEFAULT_PLAN_VARIABLE_SUMMARY_MAX_CHARS,
)
from mcp_cli.config.enums import PlanStatus
from mcp_cli.planning.backends import McpToolBackend
from mcp_cli.planning.context import PlanningContext

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelManagerProtocol(Protocol):
    """Minimal interface for model manager used by PlanRunner."""

    def get_client(
        self,
        provider: str | None = None,
        model: str | None = None,
    ) -> Any: ...


# Callback type aliases (sync or async)
ToolStartCallback = Callable[
    [str, dict[str, Any]],
    Union[None, Coroutine[Any, Any, None]],
]
"""Called before each tool execution: (tool_name, arguments) -> None."""

ToolCompleteCallback = Callable[
    [str, str, bool, float],
    Union[None, Coroutine[Any, Any, None]],
]
"""Called after each tool execution: (tool_name, result_text, success, elapsed) -> None."""


class StepResult(BaseModel):
    """Result of a single plan step execution."""

    step_index: str
    step_title: str
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class PlanExecutionResult(BaseModel):
    """Result of executing an entire plan."""

    plan_id: str
    plan_title: str
    success: bool
    steps: list[StepResult] = Field(default_factory=list)
    variables: dict[str, Any] = Field(default_factory=dict)
    total_duration: float = 0.0
    error: str | None = None
    replanned: bool = False

    model_config = {"arbitrary_types_allowed": True}


class PlanRunner:
    """Orchestrates LLM-driven plan execution with mcp-cli integration.

    Each step is executed by the LLM: given the step description and tool
    schemas, the LLM generates the actual tool call (with correct parameter
    names and values). If a tool fails, the error is fed back to the LLM
    which can retry with corrected arguments.

    Features:
    - LLM-driven tool call generation with tool schemas
    - Automatic retry on failure with LLM error correction
    - Parallel batch execution for independent steps (topological batching)
    - Progress callbacks for terminal/dashboard display
    - Dry-run mode (trace without executing)
    - Execution checkpointing and resume
    - DAG visualization
    """

    def __init__(
        self,
        context: PlanningContext,
        *,
        model_manager: ModelManagerProtocol | None = None,
        on_step_start: Callable[[str, str, str], None] | None = None,
        on_step_complete: Callable[[StepResult], None] | None = None,
        on_tool_start: ToolStartCallback | None = None,
        on_tool_complete: ToolCompleteCallback | None = None,
        enable_guards: bool = False,
        max_concurrency: int = DEFAULT_PLAN_MAX_CONCURRENCY,
        max_step_retries: int = DEFAULT_PLAN_MAX_STEP_RETRIES,
    ) -> None:
        """Initialize the plan runner.

        Args:
            context: PlanningContext with tool_manager and graph_store.
            model_manager: ModelManager for LLM-driven step execution.
                When provided, the LLM generates tool calls from step
                descriptions and can retry on failure with error feedback.
            on_step_start: Callback(step_index, step_title, tool_name) before each step.
            on_step_complete: Callback(StepResult) after each step.
            on_tool_start: Async callback(tool_name, arguments) before each tool call
                within a step. Called for each agentic loop tool invocation.
            on_tool_complete: Async callback(tool_name, result_str, success, elapsed)
                after each tool call. Called for each agentic loop tool invocation.
            enable_guards: If True, enforce guard checks during execution.
            max_concurrency: Maximum concurrent steps within a batch.
            max_step_retries: Maximum LLM retry attempts per step on failure.
        """
        self.context = context
        self._model_manager = model_manager
        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._on_tool_start = on_tool_start
        self._on_tool_complete = on_tool_complete
        self._max_concurrency = max_concurrency
        self._max_step_retries = max_step_retries

        # Create the MCP tool backend with guard integration
        self._backend = McpToolBackend(
            context.tool_manager,
            enable_guards=enable_guards,
        )

        # Tool catalog cache (lazy-loaded, protected by lock for parallel steps)
        self._tool_catalog: list[dict[str, Any]] | None = None
        self._tool_catalog_lock = asyncio.Lock()

    async def execute_plan(
        self,
        plan_data: dict[str, Any],
        *,
        variables: dict[str, Any] | None = None,
        dry_run: bool = False,
        checkpoint: bool = True,
    ) -> PlanExecutionResult:
        """Execute a plan with parallel batch execution.

        Steps are grouped into topological batches. Steps within a batch
        have no dependencies on each other and run concurrently. Batches
        execute sequentially to respect the dependency DAG.

        Args:
            plan_data: Plan dict (from PlanRegistry or plan generation).
            variables: Optional variable overrides for parameterized plans.
            dry_run: If True, trace without executing tools.
            checkpoint: If True, persist state after each batch.

        Returns:
            PlanExecutionResult with step results and final variables.
        """
        start_time = time.perf_counter()
        plan_id = plan_data.get("id", "unknown")
        plan_title = plan_data.get("title", "Untitled Plan")

        logger.info("Executing plan: %s (%s)", plan_title, plan_id)

        if dry_run:
            return await self._dry_run(plan_data, variables)

        try:
            # Build variable context
            var_context = dict(plan_data.get("variables", {}))
            if variables:
                var_context.update(variables)

            steps = plan_data.get("steps", [])
            if not steps:
                return PlanExecutionResult(
                    plan_id=plan_id,
                    plan_title=plan_title,
                    success=True,
                    total_duration=time.perf_counter() - start_time,
                )

            # Compute topological batches
            batches = _compute_batches(steps)
            logger.info(
                "Plan %s: %d steps in %d batches",
                plan_id,
                len(steps),
                len(batches),
            )

            all_step_results: list[StepResult] = []
            completed_indices: list[str] = []

            for batch_num, batch in enumerate(batches, 1):
                logger.debug(
                    "Batch %d/%d: %d steps",
                    batch_num,
                    len(batches),
                    len(batch),
                )

                if len(batch) == 1:
                    # Single step — execute directly (no gather overhead)
                    step = batch[0]
                    result = await self._execute_step(step, var_context)
                    all_step_results.append(result)

                    if result.success:
                        completed_indices.append(result.step_index)
                    else:
                        if checkpoint:
                            self._save_checkpoint(
                                plan_id,
                                completed_steps=completed_indices,
                                variables=var_context,
                                status=PlanStatus.FAILED,
                            )
                        return PlanExecutionResult(
                            plan_id=plan_id,
                            plan_title=plan_title,
                            success=False,
                            steps=all_step_results,
                            variables=var_context,
                            total_duration=time.perf_counter() - start_time,
                            error=f"Step {result.step_index} failed: {result.error}",
                        )
                else:
                    # Multiple independent steps — execute concurrently
                    batch_results = await self._execute_batch(batch, var_context)
                    all_step_results.extend(batch_results)

                    failed = [r for r in batch_results if not r.success]
                    if failed:
                        completed_indices.extend(
                            r.step_index for r in batch_results if r.success
                        )
                        if checkpoint:
                            self._save_checkpoint(
                                plan_id,
                                completed_steps=completed_indices,
                                variables=var_context,
                                status=PlanStatus.FAILED,
                            )
                        fail_msgs = "; ".join(
                            f"step {r.step_index}: {r.error}" for r in failed
                        )
                        return PlanExecutionResult(
                            plan_id=plan_id,
                            plan_title=plan_title,
                            success=False,
                            steps=all_step_results,
                            variables=var_context,
                            total_duration=time.perf_counter() - start_time,
                            error=f"Batch {batch_num} had failures: {fail_msgs}",
                        )

                    completed_indices.extend(r.step_index for r in batch_results)

                # Checkpoint after each batch
                if checkpoint:
                    self._save_checkpoint(
                        plan_id,
                        completed_steps=completed_indices,
                        variables=var_context,
                        status=PlanStatus.RUNNING,
                    )

            total_duration = time.perf_counter() - start_time

            # Final checkpoint
            if checkpoint:
                self._save_checkpoint(
                    plan_id,
                    completed_steps=completed_indices,
                    variables=var_context,
                    status=PlanStatus.COMPLETED,
                )

            return PlanExecutionResult(
                plan_id=plan_id,
                plan_title=plan_title,
                success=True,
                steps=all_step_results,
                variables=var_context,
                total_duration=total_duration,
            )

        except Exception as e:
            total_duration = time.perf_counter() - start_time
            logger.error("Plan execution failed: %s", e)
            return PlanExecutionResult(
                plan_id=plan_id,
                plan_title=plan_title,
                success=False,
                total_duration=total_duration,
                error=str(e),
            )

    async def _get_tool_catalog(self) -> list[dict[str, Any]]:
        """Get tool catalog, caching for the duration of the plan run.

        Uses an asyncio.Lock to prevent duplicate fetches when parallel
        batch steps hit the cache simultaneously.
        """
        async with self._tool_catalog_lock:
            if self._tool_catalog is None:
                self._tool_catalog = await self.context.get_tool_catalog()
            return self._tool_catalog

    async def _execute_step(
        self,
        step: dict[str, Any],
        var_context: dict[str, Any],
    ) -> StepResult:
        """Execute a single plan step using the LLM for tool call generation.

        When a model_manager is available, the LLM generates the tool call
        from the step description and tool schemas, then the tool is executed.
        If the tool fails, the error is fed back to the LLM for retry.

        Falls back to static arg execution when no model_manager is provided.
        """
        step_index = step.get("index", "?")
        step_title = step.get("title", "Untitled")
        tool_calls = step.get("tool_calls", [])
        hint_tool = tool_calls[0]["name"] if tool_calls else step.get("tool", "none")
        hint_args = (
            tool_calls[0].get("args", {}) if tool_calls else step.get("args", {})
        )

        if self._on_step_start:
            self._on_step_start(step_index, step_title, hint_tool)

        start_time = time.perf_counter()

        # LLM-driven execution (agentic loop with retry)
        if self._model_manager:
            logger.info(
                "Step %s: using agentic LLM execution (model_manager=%s)",
                step_index,
                type(self._model_manager).__name__,
            )
            step_result = await self._execute_step_with_llm(
                step, var_context, step_index, step_title, hint_tool, hint_args
            )
        else:
            logger.info(
                "Step %s: using static arg execution (no model_manager)", step_index
            )
            step_result = await self._execute_step_static(
                step, var_context, step_index, step_title, hint_tool, hint_args
            )

        step_result.duration = time.perf_counter() - start_time

        if self._on_step_complete:
            self._on_step_complete(step_result)

        return step_result

    async def _execute_step_static(
        self,
        step: dict[str, Any],
        var_context: dict[str, Any],
        step_index: str,
        step_title: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> StepResult:
        """Execute a step using the plan's static arguments (no LLM)."""
        resolved_args = _resolve_variables(args, var_context)

        try:
            if self._on_tool_start:
                await _maybe_await(self._on_tool_start(tool_name, resolved_args))

            tool_start = time.perf_counter()
            request = ToolExecutionRequest(
                tool_name=tool_name,
                args=resolved_args,
                step_id=f"step-{step_index}",
            )
            exec_result = await self._backend.execute_tool(request)
            tool_elapsed = time.perf_counter() - tool_start

            result_var = step.get("result_variable")
            if result_var and exec_result.success:
                var_context[result_var] = exec_result.result

            result_text = (
                json.dumps(exec_result.result, default=str)
                if exec_result.success
                else (exec_result.error or "Tool execution failed")
            )
            if self._on_tool_complete:
                await _maybe_await(
                    self._on_tool_complete(
                        tool_name, result_text, exec_result.success, tool_elapsed
                    )
                )

            return StepResult(
                step_index=step_index,
                step_title=step_title,
                tool_name=tool_name,
                success=exec_result.success,
                result=exec_result.result,
                error=exec_result.error,
            )
        except Exception as e:
            return StepResult(
                step_index=step_index,
                step_title=step_title,
                tool_name=tool_name,
                success=False,
                error=str(e),
            )

    async def _execute_step_with_llm(
        self,
        step: dict[str, Any],
        var_context: dict[str, Any],
        step_index: str,
        step_title: str,
        hint_tool: str,
        hint_args: dict[str, Any],
    ) -> StepResult:
        """Execute a step via a full agentic loop.

        The LLM drives the entire step execution:
        1. Sees the step description, tool schemas, and variable context
        2. Generates a tool call with correct parameters
        3. Sees the tool result (success OR failure)
        4. Evaluates the result — was it useful? Empty? Wrong?
        5. Decides: respond with final answer (step done) or call another tool

        The loop continues until the LLM responds with text (no tool call),
        indicating it considers the step complete, or max turns are exhausted.
        """
        tool_catalog = await self._get_tool_catalog()
        if self._model_manager is None:
            raise RuntimeError(
                "_execute_step_with_llm requires model_manager but none was provided"
            )
        client = self._model_manager.get_client()

        # Build variable summary for context
        var_summary = _summarize_variables(var_context) if var_context else "none"

        # Resolve any ${var} references in hint args for context
        resolved_hints = _resolve_variables(hint_args, var_context)
        hint_str = json.dumps(resolved_hints, default=str) if resolved_hints else "{}"

        result_var = step.get("result_variable")

        # Initial messages: instruct the LLM to act as an agent for this step
        messages: list[dict[str, Any]] = [
            {
                "role": MessageRole.SYSTEM,
                "content": (
                    "You are executing one step of a plan. Your job is to call "
                    "the appropriate tool(s) to accomplish the step goal.\n\n"
                    "Rules:\n"
                    "- Use the tool schemas to determine correct parameter names and types\n"
                    "- After each tool call, you will see the result\n"
                    "- If the result is empty, null, or unhelpful, try again with "
                    "different parameters (e.g. a simpler location name)\n"
                    "- If the tool returns an error, fix the parameters and retry\n"
                    "- When you have a satisfactory result, respond with a brief "
                    "text summary (no tool call) to complete the step\n"
                    "- Ensure numeric parameters are numbers, not strings\n\n"
                    f"Available variables from previous steps:\n{var_summary}"
                ),
            },
            {
                "role": MessageRole.USER,
                "content": (
                    f"Execute this plan step:\n"
                    f"  Step {step_index}: {step_title}\n"
                    f"  Suggested tool: {hint_tool}\n"
                    f"  Suggested args: {hint_str}\n\n"
                    f"Call the tool now."
                ),
            },
        ]

        last_error: str | None = None
        last_result: Any = None
        used_tool: str = hint_tool
        max_turns = 1 + self._max_step_retries

        for turn in range(max_turns):
            try:
                response = await client.create_completion(
                    messages=messages,
                    tools=tool_catalog,
                    stream=False,
                )

                tool_call = _extract_tool_call(response)

                if not tool_call:
                    # LLM responded with text (no tool call).
                    # On turn 0, it means the LLM chose not to call a tool — fall back.
                    if turn == 0:
                        logger.debug(
                            "Step %s: LLM did not generate tool call, "
                            "using static args",
                            step_index,
                        )
                        return await self._execute_step_static(
                            step,
                            var_context,
                            step_index,
                            step_title,
                            hint_tool,
                            hint_args,
                        )

                    # On later turns, the LLM is signaling step completion.
                    # Use the last successful result as the step output.
                    if last_result is not None:
                        if result_var:
                            var_context[result_var] = last_result
                        return StepResult(
                            step_index=step_index,
                            step_title=step_title,
                            tool_name=used_tool,
                            success=True,
                            result=last_result,
                        )
                    # No successful result yet — treat as failure
                    return StepResult(
                        step_index=step_index,
                        step_title=step_title,
                        tool_name=used_tool,
                        success=False,
                        error=last_error or "LLM ended step without a result",
                    )

                # LLM generated a tool call — execute it
                used_tool = tool_call["name"]
                tool_args = tool_call["args"]
                call_id = f"call_{step_index}_{turn}"

                # Notify UI: tool execution starting
                if self._on_tool_start:
                    await _maybe_await(self._on_tool_start(used_tool, tool_args))

                tool_start = time.perf_counter()
                request = ToolExecutionRequest(
                    tool_name=used_tool,
                    args=tool_args,
                    step_id=f"step-{step_index}",
                )
                exec_result = await self._backend.execute_tool(request)
                tool_elapsed = time.perf_counter() - tool_start

                # Build the assistant + tool result messages for the conversation
                messages.append(
                    {
                        "role": MessageRole.ASSISTANT,
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": used_tool,
                                    "arguments": json.dumps(tool_args, default=str),
                                },
                            }
                        ],
                    }
                )

                if exec_result.success:
                    last_result = exec_result.result
                    result_text = json.dumps(exec_result.result, default=str)
                    messages.append(
                        {
                            "role": MessageRole.TOOL,
                            "tool_call_id": call_id,
                            "content": result_text,
                        }
                    )
                    # Notify UI: tool execution completed
                    if self._on_tool_complete:
                        await _maybe_await(
                            self._on_tool_complete(
                                used_tool, result_text, True, tool_elapsed
                            )
                        )
                    logger.debug(
                        "Step %s turn %d: tool %s succeeded in %.2fs",
                        step_index,
                        turn + 1,
                        used_tool,
                        tool_elapsed,
                    )
                else:
                    last_error = exec_result.error or "Tool execution failed"
                    messages.append(
                        {
                            "role": MessageRole.TOOL,
                            "tool_call_id": call_id,
                            "content": f"Error: {last_error}",
                        }
                    )
                    # Notify UI: tool execution failed
                    if self._on_tool_complete:
                        await _maybe_await(
                            self._on_tool_complete(
                                used_tool, last_error, False, tool_elapsed
                            )
                        )
                    logger.info(
                        "Step %s turn %d: tool %s failed in %.2fs: %s",
                        step_index,
                        turn + 1,
                        used_tool,
                        tool_elapsed,
                        last_error,
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Step %s turn %d raised exception: %s",
                    step_index,
                    turn + 1,
                    last_error,
                )

        # Max turns exhausted — return whatever we have
        if last_result is not None:
            if result_var:
                var_context[result_var] = last_result
            return StepResult(
                step_index=step_index,
                step_title=step_title,
                tool_name=used_tool,
                success=True,
                result=last_result,
            )
        return StepResult(
            step_index=step_index,
            step_title=step_title,
            tool_name=used_tool,
            success=False,
            error=last_error or "All retry attempts failed",
        )

    async def _execute_batch(
        self,
        batch: list[dict[str, Any]],
        var_context: dict[str, Any],
    ) -> list[StepResult]:
        """Execute a batch of independent steps concurrently.

        Uses asyncio.Semaphore to limit concurrency.
        """
        sem = asyncio.Semaphore(self._max_concurrency)

        async def _run_with_sem(step: dict[str, Any]) -> StepResult:
            async with sem:
                return await self._execute_step(step, var_context)

        tasks = [asyncio.create_task(_run_with_sem(step)) for step in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        step_results: list[StepResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                step = batch[i]
                step_results.append(
                    StepResult(
                        step_index=step.get("index", "?"),
                        step_title=step.get("title", "Untitled"),
                        tool_name=step.get("tool", "none"),
                        success=False,
                        error=str(result),
                    )
                )
            else:
                step_results.append(result)
        return step_results

    async def _dry_run(
        self,
        plan_data: dict[str, Any],
        variables: dict[str, Any] | None = None,
    ) -> PlanExecutionResult:
        """Trace plan execution without running tools.

        Shows what each step would do, including resolved variable references
        and which steps run in parallel batches.
        """
        plan_id = plan_data.get("id", "unknown")
        plan_title = plan_data.get("title", "Untitled Plan")
        step_results = []
        var_context = dict(plan_data.get("variables", {}))
        if variables:
            var_context.update(variables)

        steps = plan_data.get("steps", [])
        batches = _compute_batches(steps)

        for batch in batches:
            for step in batch:
                step_index = step.get("index", "?")
                step_title = step.get("title", "Untitled")
                tool_calls = step.get("tool_calls", [])
                tool_name = (
                    tool_calls[0]["name"] if tool_calls else step.get("tool", "none")
                )

                if self._on_step_start:
                    self._on_step_start(step_index, step_title, tool_name)

                step_result = StepResult(
                    step_index=step_index,
                    step_title=step_title,
                    tool_name=tool_name,
                    success=True,
                    result="[dry-run: not executed]",
                )
                step_results.append(step_result)

                # Simulate variable binding
                result_var = step.get("result_variable")
                if result_var:
                    var_context[result_var] = f"<{tool_name} result>"

                if self._on_step_complete:
                    self._on_step_complete(step_result)

        return PlanExecutionResult(
            plan_id=plan_id,
            plan_title=plan_title,
            success=True,
            steps=step_results,
            variables=var_context,
        )

    def _save_checkpoint(
        self,
        plan_id: str,
        completed_steps: list[str],
        variables: dict[str, Any],
        status: PlanStatus,
    ) -> None:
        """Save execution checkpoint for resume support."""
        checkpoint_path = self.context.plans_dir / f"{plan_id}_state.json"
        checkpoint = {
            "plan_id": plan_id,
            "status": status,
            "completed_steps": completed_steps,
            "variables": _serialize_variables(variables),
        }

        try:
            checkpoint_path.write_text(
                json.dumps(checkpoint, indent=2, default=str),
                encoding="utf-8",
            )
            logger.debug("Saved checkpoint for plan %s: %s", plan_id, status)
        except Exception as e:
            logger.warning("Failed to save checkpoint for plan %s: %s", plan_id, e)

    def load_checkpoint(self, plan_id: str) -> dict[str, Any] | None:
        """Load execution checkpoint for resume."""
        checkpoint_path = self.context.plans_dir / f"{plan_id}_state.json"
        if not checkpoint_path.exists():
            return None

        try:
            data: dict[str, Any] = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            return data
        except Exception as e:
            logger.warning("Failed to load checkpoint for plan %s: %s", plan_id, e)
            return None


# ── Topological Batching ───────────────────────────────────────────────────


def _compute_batches(steps: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Compute parallel execution batches via topological sort.

    Groups steps into batches where all steps in a batch have their
    dependencies satisfied by previous batches. Steps within a batch
    can execute concurrently.

    Uses Kahn's BFS algorithm for topological sorting.

    Args:
        steps: List of step dicts with 'index' and 'depends_on' fields.

    Returns:
        List of batches, each batch is a list of step dicts.
    """
    if not steps:
        return []

    # Build index maps
    index_to_step: dict[str, dict[str, Any]] = {}
    for i, step in enumerate(steps):
        idx = str(step.get("index", str(i + 1)))
        step = dict(step)  # Don't mutate original
        step["index"] = idx
        index_to_step[idx] = step

    # Build dependency graph
    in_degree: dict[str, int] = {idx: 0 for idx in index_to_step}
    dependents: dict[str, list[str]] = {idx: [] for idx in index_to_step}

    for idx, step in index_to_step.items():
        deps = step.get("depends_on", [])
        for dep in deps:
            dep_str = str(dep)
            if dep_str in index_to_step:
                in_degree[idx] += 1
                dependents[dep_str].append(idx)

    # Kahn's BFS: find all ready nodes (in_degree == 0), emit as batch
    batches = []
    remaining = set(index_to_step.keys())

    while remaining:
        # Find all nodes with no unmet dependencies
        ready = [idx for idx in remaining if in_degree.get(idx, 0) == 0]

        if not ready:
            # Cycle detected — break tie by taking first remaining node
            logger.warning("Dependency cycle detected, forcing execution order")
            ready = [sorted(remaining)[0]]

        batch = [index_to_step[idx] for idx in sorted(ready)]
        batches.append(batch)

        # Remove processed nodes and update dependents
        for idx in ready:
            remaining.discard(idx)
            for dep_idx in dependents.get(idx, []):
                in_degree[dep_idx] = max(0, in_degree[dep_idx] - 1)

    return batches


# ── Variable Resolution ───────────────────────────────────────────────────


def _resolve_variables(
    args: dict[str, Any], variables: dict[str, Any]
) -> dict[str, Any]:
    """Resolve ${var} references in tool arguments.

    Supports:
    - ${variable} — direct replacement
    - ${variable.field} — nested dict access
    - Template strings: "prefix ${var} suffix"

    Args:
        args: Tool arguments dict (may contain ${var} references).
        variables: Current variable bindings.

    Returns:
        New dict with all resolvable references replaced.
    """
    resolved = {}
    for key, value in args.items():
        resolved[key] = _resolve_value(value, variables)
    return resolved


def _resolve_value(value: Any, variables: dict[str, Any]) -> Any:
    """Resolve a single value, recursing into dicts and lists."""
    if isinstance(value, str):
        return _resolve_string(value, variables)
    if isinstance(value, dict):
        return {k: _resolve_value(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, variables) for v in value]
    return value


def _resolve_string(value: str, variables: dict[str, Any]) -> Any:
    """Resolve ${var} references in a string value."""
    if not value or "${" not in value:
        return value

    # Single variable reference: "${var}" → return the value directly (preserves type)
    if value.startswith("${") and value.endswith("}") and value.count("${") == 1:
        var_path = value[2:-1]
        resolved = _resolve_path(var_path, variables)
        return resolved if resolved is not None else value

    # Template string: "text ${var} more" → string interpolation
    def replacer(match: re.Match) -> str:
        var_path = match.group(1)
        resolved = _resolve_path(var_path, variables)
        return str(resolved) if resolved is not None else match.group(0)

    return re.sub(r"\$\{([^}]+)}", replacer, value)


def _resolve_path(var_path: str, variables: dict[str, Any]) -> Any:
    """Resolve a dotted variable path like 'api.endpoint.port'."""
    parts = var_path.split(".")
    current: Any = variables
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


# ── Callback Helpers ──────────────────────────────────────────────────────


async def _maybe_await(result: Any) -> Any:
    """Await a result if it's a coroutine, otherwise return it directly.

    Allows callbacks to be either sync or async functions.
    """
    if asyncio.iscoroutine(result):
        return await result
    return result


# ── LLM Response Helpers ──────────────────────────────────────────────────


def _extract_tool_call(response: Any) -> dict[str, Any] | None:
    """Extract a tool call from an LLM completion response.

    Handles three response formats:
    1. chuk_llm native: {"response": ..., "tool_calls": [...], "usage": {...}}
    2. OpenAI-style dict: {"choices": [{"message": {"tool_calls": [...]}}]}
    3. Object-style: response.choices[0].message.tool_calls

    Returns:
        Dict with 'name' and 'args', or None if no tool call found.
    """
    if response is None:
        return None

    if isinstance(response, dict):
        # chuk_llm native format: top-level "tool_calls" key
        tool_calls = response.get("tool_calls")
        if tool_calls:
            return _parse_tool_call_entry(tool_calls[0])

        # OpenAI-style dict format: choices[0].message.tool_calls
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                return _parse_tool_call_entry(tool_calls[0])

        return None

    # Object-style response (e.g., Pydantic models)
    # Check for top-level tool_calls attribute (chuk_llm objects)
    if hasattr(response, "tool_calls") and response.tool_calls:
        return _parse_tool_call_entry(response.tool_calls[0])

    # OpenAI-style object: response.choices[0].message.tool_calls
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if message:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                return _parse_tool_call_entry(tool_calls[0])

    return None


def _parse_tool_call_entry(tc: Any) -> dict[str, Any] | None:
    """Parse a single tool call entry from either dict or object format.

    Args:
        tc: A tool call entry (dict or object with function attribute).

    Returns:
        Dict with 'name' and 'args', or None if parsing fails.
    """
    if isinstance(tc, dict):
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
    elif hasattr(tc, "function"):
        func = tc.function
        name = getattr(func, "name", "")
        args_str = getattr(func, "arguments", "{}")
    else:
        return None

    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except (json.JSONDecodeError, TypeError):
        args = {}

    return {"name": name, "args": args} if name else None


def _summarize_variables(var_context: dict[str, Any]) -> str:
    """Build a compact summary of current variables for the LLM."""
    if not var_context:
        return "none"
    lines = []
    for key, value in var_context.items():
        text = json.dumps(value, default=str)
        if len(text) > DEFAULT_PLAN_VARIABLE_SUMMARY_MAX_CHARS:
            text = text[:DEFAULT_PLAN_VARIABLE_SUMMARY_MAX_CHARS] + "..."
        lines.append(f"  ${{{key}}} = {text}")
    return "\n".join(lines)


# ── DAG Visualization ──────────────────────────────────────────────────────


def render_plan_dag(plan_data: dict[str, Any]) -> str:
    """Render a plan as an ASCII DAG for terminal display.

    Shows steps with their tools, dependencies, and execution status.
    Parallel steps (same batch) are shown with a parallel indicator.

    Args:
        plan_data: Plan dict with steps and dependencies.

    Returns:
        Multiline string with the DAG visualization.
    """
    steps = plan_data.get("steps", [])
    if not steps:
        return "  (empty plan)"

    # Compute batches for parallel indicators
    batches = _compute_batches(steps)
    step_to_batch: dict[str, int] = {}
    for batch_num, batch in enumerate(batches, 1):
        for step in batch:
            step_to_batch[step.get("index", "?")] = batch_num

    lines = []
    current_batch = 0

    for i, step in enumerate(steps):
        index = step.get("index", str(i + 1))
        title = step.get("title", "Untitled")[:DEFAULT_PLAN_DAG_TITLE_MAX_CHARS]
        tool_calls = step.get("tool_calls", [])
        tool_name = tool_calls[0]["name"] if tool_calls else step.get("tool", "?")
        depends_on = step.get("depends_on", [])

        # Status indicator
        status = step.get("_status", PlanStatus.PENDING)
        status_char = {
            PlanStatus.PENDING: "○",
            PlanStatus.RUNNING: "◉",
            PlanStatus.COMPLETED: "●",
            PlanStatus.FAILED: "✗",
        }.get(status, "○")

        # Batch separator for parallel groups
        batch_num = step_to_batch.get(str(index), 0)
        if batch_num != current_batch:
            if current_batch > 0:
                lines.append("")  # Blank line between batches
            current_batch = batch_num

        # Dependency arrows
        dep_str = ""
        if depends_on:
            dep_refs = ", ".join(str(d) for d in depends_on)
            dep_str = f"  ← after: {dep_refs}"

        # Parallel indicator
        batch_steps = batches[batch_num - 1] if batch_num > 0 else []
        parallel_marker = ""
        if len(batch_steps) > 1:
            parallel_marker = " ∥"

        lines.append(
            f"  {status_char} {index}. {title:<35} [{tool_name}]{dep_str}{parallel_marker}"
        )

    return "\n".join(lines)


# ── Serialization Helpers ──────────────────────────────────────────────────


def _serialize_variables(variables: dict[str, Any]) -> dict[str, Any]:
    """Make variables JSON-serializable.

    Truncates large values to prevent bloated checkpoint files.
    """
    result: dict[str, Any] = {}
    for key, value in variables.items():
        if isinstance(value, str) and len(value) > DEFAULT_PLAN_CHECKPOINT_MAX_CHARS:
            result[key] = value[:DEFAULT_PLAN_CHECKPOINT_MAX_CHARS] + "... [truncated]"
        elif isinstance(value, (dict, list)):
            serialized = json.dumps(value, default=str)
            if len(serialized) > DEFAULT_PLAN_CHECKPOINT_MAX_CHARS:
                result[key] = f"[{type(value).__name__}, {len(serialized)} chars]"
            else:
                result[key] = value
        else:
            result[key] = value
    return result
