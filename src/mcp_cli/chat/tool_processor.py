# mcp_cli/chat/tool_processor.py
"""
mcp_cli.chat.tool_processor

Simplified tool processor that delegates parallel execution to ToolManager.
Handles CLI-specific concerns: UI, conversation history, user confirmation.

Uses chuk-tool-processor's ToolCall/ToolResult models via ToolManager.
Uses Protocol-based interfaces for type safety.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from chuk_tool_processor import ToolCall as CTPToolCall
from chuk_tool_processor import ToolResult as CTPToolResult

from mcp_cli.chat.response_models import Message, MessageRole, ToolCall
from mcp_cli.chat.models import (
    ToolCallMetadata,
    ToolProcessorContext,
    UIManagerProtocol,
)
from mcp_cli.config.defaults import (
    DYNAMIC_TOOL_PROXY_NAME,
    DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES,
    TRANSPORT_ERROR_PATTERNS,
)
from chuk_ai_session_manager.guards import SoftBlockReason
from mcp_cli.chat.agent_tool_state import get_agent_tool_state
from chuk_tool_processor.discovery import get_search_engine
from mcp_cli.llm.content_models import ContentBlockType
from mcp_cli.agents.tools import _AGENT_TOOL_NAMES
from mcp_cli.memory.tools import _MEMORY_TOOL_NAMES
from mcp_cli.planning.tools import _PLAN_TOOL_NAMES
from mcp_cli.utils.preferences import get_preference_manager

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)

# VM tools handled locally via MemoryManager, not routed to MCP ToolManager
_VM_TOOL_NAMES = frozenset({"page_fault", "search_pages"})

# _AGENT_TOOL_NAMES imported from mcp_cli.agents.tools (single source of truth)
# _MEMORY_TOOL_NAMES imported from mcp_cli.memory.tools (single source of truth)
# _PLAN_TOOL_NAMES imported from mcp_cli.planning.tools (single source of truth)


class ToolProcessor:
    """
    Handle execution of tool calls returned by the LLM.

    Delegates parallel execution to ToolManager.stream_execute_tools(),
    handling only CLI-specific concerns: UI, conversation history, confirmation.

    Uses ToolProcessorContext protocol for type-safe context access.
    """

    def __init__(
        self,
        context: ToolProcessorContext,
        ui_manager: UIManagerProtocol,
        *,
        max_concurrency: int = 4,
    ) -> None:
        self.context = context
        self.ui_manager = ui_manager
        self.max_concurrency = max_concurrency

        # Tool manager for execution - access via protocol attribute
        self.tool_manager: ToolManager | None = context.tool_manager

        # Track transport failures for recovery detection
        self._transport_failures = 0
        self._consecutive_transport_failures = 0

        # Track state for callbacks
        self._call_metadata: dict[str, ToolCallMetadata] = {}
        self._cancelled = False

        # Track which tool_call_ids have received results (for orphan detection)
        self._result_ids_added: set[str] = set()

        # Track page_fault calls within a conversation to prevent re-fault loops
        self._faulted_page_ids: set[str] = set()

        # Give the context a back-pointer for Ctrl-C cancellation
        # Note: This is the one place we set an attribute on context
        context.tool_processor = self

    async def process_tool_calls(
        self,
        tool_calls: list[Any],
        name_mapping: dict[str, str] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        """
        Execute tool_calls in parallel using ToolManager.stream_execute_tools().

        Args:
            tool_calls: List of tool call objects from the LLM
            name_mapping: Mapping from LLM tool names to actual tool names
            reasoning_content: Optional reasoning content from the LLM
        """
        if not tool_calls:
            logger.warning("Empty tool_calls list received.")
            return

        if name_mapping is None:
            name_mapping = {}

        logger.info(
            f"Processing {len(tool_calls)} tool calls with {len(name_mapping)} name mappings"
        )

        # Reset state
        self._call_metadata.clear()
        self._cancelled = False
        self._result_ids_added = set()
        self._faulted_page_ids.clear()

        # Add assistant message with all tool calls BEFORE executing
        self._add_assistant_message_with_tool_calls(tool_calls, reasoning_content)

        # Convert LLM tool calls to CTP format and check confirmations
        ctp_calls: list[CTPToolCall] = []

        try:
            for idx, call in enumerate(tool_calls):
                if getattr(self.ui_manager, "interrupt_requested", False):
                    self._cancelled = True
                    break

                # Extract tool call details
                llm_tool_name, raw_arguments, call_id = self._extract_tool_call_info(
                    call, idx
                )

                # Map to execution name
                execution_tool_name = name_mapping.get(llm_tool_name, llm_tool_name)

                # Get display name - special handling for dynamic tool call_tool
                display_name = execution_tool_name
                display_arguments = raw_arguments

                # For dynamic tools, extract the actual tool name from call_tool
                if execution_tool_name == DYNAMIC_TOOL_PROXY_NAME:
                    # Parse arguments to get the real tool name
                    parsed_args = self._parse_arguments(raw_arguments)
                    if "tool_name" in parsed_args:
                        actual_tool = parsed_args["tool_name"]
                        # Show as "call_tool → actual_tool_name"
                        display_name = f"call_tool → {actual_tool}"
                        # Filter out tool_name from displayed args to reduce noise
                        display_arguments = {
                            k: v for k, v in parsed_args.items() if k != "tool_name"
                        }

                if hasattr(self.context, "get_display_name_for_tool"):
                    # Only apply name mapping if not already a dynamic tool
                    if not execution_tool_name.startswith(DYNAMIC_TOOL_PROXY_NAME):
                        display_name = self.context.get_display_name_for_tool(
                            execution_tool_name
                        )

                # Show tool call in UI
                try:
                    self.ui_manager.print_tool_call(display_name, display_arguments)
                except Exception as ui_exc:
                    logger.warning(f"UI display error (non-fatal): {ui_exc}")

                # Handle user confirmation
                server_url = self._get_server_url_for_tool(execution_tool_name)
                if self._should_confirm_tool(execution_tool_name, server_url):
                    confirmed = await self.ui_manager.do_confirm_tool_execution(
                        tool_name=display_name, arguments=raw_arguments
                    )
                    if not confirmed:
                        setattr(self.ui_manager, "interrupt_requested", True)
                        self._add_cancelled_tool_to_history(
                            llm_tool_name, call_id, raw_arguments
                        )
                        self._cancelled = True
                        break

                # Parse arguments
                arguments = self._parse_arguments(raw_arguments)

                # Resolve actual tool name for call_tool proxy
                actual_tool = execution_tool_name
                if execution_tool_name == DYNAMIC_TOOL_PROXY_NAME:
                    proxy_name = arguments.get("tool_name")
                    if proxy_name and isinstance(proxy_name, str):
                        actual_tool = proxy_name

                # ── VM tool interception ────────────────────────────────
                # page_fault and search_pages are internal VM operations,
                # handled by MemoryManager — not routed to MCP ToolManager.
                if actual_tool in _VM_TOOL_NAMES:
                    await self._handle_vm_tool(
                        actual_tool, arguments, llm_tool_name, call_id
                    )
                    continue

                # ── Memory scope tool interception ─────────────────────
                # remember, recall, forget are persistent memory ops,
                # handled locally — not routed to MCP ToolManager.
                if actual_tool in _MEMORY_TOOL_NAMES:
                    await self._handle_memory_tool(
                        actual_tool, arguments, llm_tool_name, call_id
                    )
                    continue

                # ── Plan tool interception ─────────────────────────────
                # plan_create, plan_execute, plan_create_and_execute are
                # internal planning ops — not routed to MCP ToolManager.
                if actual_tool in _PLAN_TOOL_NAMES:
                    await self._handle_plan_tool(
                        actual_tool, arguments, llm_tool_name, call_id
                    )
                    continue

                # ── Agent tool interception ────────────────────────────
                # agent_spawn, agent_stop, etc. are orchestration tools
                # handled by AgentManager — not routed to MCP ToolManager.
                if actual_tool in _AGENT_TOOL_NAMES:
                    await self._handle_agent_tool(
                        actual_tool, arguments, llm_tool_name, call_id
                    )
                    continue

                # DEBUG: Log exactly what the model sent for this tool call
                logger.info(f"TOOL CALL FROM MODEL: {llm_tool_name} id={call_id}")
                logger.info(f"  raw_arguments: {raw_arguments}")
                logger.info(f"  parsed_arguments: {arguments}")

                # Get actual tool name for checks (for call_tool, it's the inner tool)
                actual_tool_for_checks = execution_tool_name
                if (
                    execution_tool_name == DYNAMIC_TOOL_PROXY_NAME
                    and "tool_name" in arguments
                ):
                    actual_tool_for_checks = arguments["tool_name"]

                # GENERIC VALIDATION: Reject tool calls with None arguments
                # This catches cases where the model emits placeholders or incomplete calls
                none_args = [
                    k for k, v in arguments.items() if v is None and k != "tool_name"
                ]
                if none_args:
                    error_msg = (
                        f"INVALID_ARGS: Tool '{actual_tool_for_checks}' called with None values "
                        f"for: {', '.join(none_args)}. Please provide actual values."
                    )
                    logger.warning(error_msg)
                    self._add_tool_result_to_history(
                        llm_tool_name,
                        call_id,
                        f"**Error**: {error_msg}\n\nPlease retry with actual parameter values.",
                    )
                    continue

                # Check $vN references in arguments (dataflow validation)
                tool_state = get_agent_tool_state(
                    getattr(self.context, "agent_id", "default")
                )
                ref_check = tool_state.check_references(arguments)
                if not ref_check.valid:
                    logger.warning(
                        f"Missing references in {actual_tool_for_checks}: {ref_check.message}"
                    )
                    # Add error to history instead of executing
                    self._add_tool_result_to_history(
                        llm_tool_name,
                        call_id,
                        f"**Blocked**: {ref_check.message}\n\n"
                        f"{tool_state.format_bindings_for_model()}",
                    )
                    continue

                # Check for ungrounded calls (numeric args without $vN refs)
                # Skip discovery tools - they don't need grounded numeric inputs
                # Skip idempotent math tools - they should be allowed to compute with any literals
                # Use SoftBlock repair system: attempt rebind → symbolic fallback → ask user
                is_math_tool = tool_state.is_idempotent_math_tool(
                    actual_tool_for_checks
                )
                if (
                    not tool_state.is_discovery_tool(execution_tool_name)
                    and not is_math_tool
                ):
                    ungrounded_check = tool_state.check_ungrounded_call(
                        actual_tool_for_checks, arguments
                    )
                    if ungrounded_check.is_ungrounded:
                        # Log args for observability (important for debugging)
                        logger.info(
                            f"Ungrounded call to {actual_tool_for_checks} with args: {arguments}"
                        )

                        # Check if this tool should have auto-rebound applied
                        # Parameterized tools (normal_cdf, sqrt, etc.) should NOT be rebound
                        # because each call with different args has different semantics
                        if not tool_state.should_auto_rebound(actual_tool_for_checks):
                            # For parameterized tools, check preconditions first
                            # This blocks premature calls before any values are computed
                            precond_ok, precond_error = (
                                tool_state.check_tool_preconditions(
                                    actual_tool_for_checks, arguments
                                )
                            )
                            if not precond_ok:
                                logger.warning(
                                    f"Precondition failed for {actual_tool_for_checks}"
                                )
                                self._add_tool_result_to_history(
                                    llm_tool_name,
                                    call_id,
                                    f"**Blocked**: {precond_error}",
                                )
                                continue

                            # Preconditions met - log and allow execution
                            display_args = {
                                k: v for k, v in arguments.items() if k != "tool_name"
                            }
                            logger.info(
                                f"Allowing parameterized tool {actual_tool_for_checks} with args: {display_args}"
                            )
                            # Fall through to execution
                        else:
                            # For other tools, try to repair using SoftBlock system
                            should_proceed, repaired_args, fallback_response = (
                                tool_state.try_soft_block_repair(
                                    actual_tool_for_checks,
                                    arguments,
                                    SoftBlockReason.UNGROUNDED_ARGS,
                                )
                            )

                            if should_proceed and repaired_args:
                                # Rebind succeeded - use repaired arguments
                                logger.info(
                                    f"Auto-repaired ungrounded call to {actual_tool_for_checks}: "
                                    f"{arguments} -> {repaired_args}"
                                )
                                arguments = repaired_args
                            elif fallback_response:
                                # Symbolic fallback - return helpful response instead of blocking
                                # Show visible annotation for observability
                                logger.info(
                                    f"Symbolic fallback for {actual_tool_for_checks}"
                                )
                                self._add_tool_result_to_history(
                                    llm_tool_name, call_id, fallback_response
                                )
                                continue
                            else:
                                # All repairs failed - add error to history
                                logger.warning(
                                    f"Could not repair ungrounded call to {actual_tool_for_checks}"
                                )
                                self._add_tool_result_to_history(
                                    llm_tool_name,
                                    call_id,
                                    f"Cannot proceed with `{actual_tool_for_checks}`: "
                                    f"arguments require computed values.\n\n"
                                    f"{tool_state.format_bindings_for_model()}",
                                )
                                continue

                # Check per-tool call limit using the guard (handles exemptions for math/discovery)
                # per_tool_cap=0 means "disabled/unlimited" (see RuntimeLimits presets)
                per_tool_result = tool_state.check_per_tool_limit(
                    actual_tool_for_checks
                )
                if tool_state.limits.per_tool_cap > 0 and per_tool_result.blocked:
                    logger.warning(
                        f"Tool {actual_tool_for_checks} blocked by per-tool limit: {per_tool_result.reason}"
                    )
                    self._add_tool_result_to_history(
                        llm_tool_name,
                        call_id,
                        per_tool_result.reason or "Per-tool limit reached",
                    )
                    continue

                # Resolve $vN references in arguments (substitute actual values)
                resolved_arguments = tool_state.resolve_references(arguments)

                # Store metadata for callbacks
                self._call_metadata[call_id] = ToolCallMetadata(
                    llm_tool_name=llm_tool_name,
                    execution_tool_name=execution_tool_name,
                    display_name=display_name,
                    arguments=resolved_arguments,
                    raw_arguments=raw_arguments,
                )

                # Create CTP ToolCall with resolved arguments
                ctp_calls.append(
                    CTPToolCall(
                        id=call_id,
                        tool=execution_tool_name,
                        arguments=resolved_arguments,
                    )
                )

            if self._cancelled or not ctp_calls:
                return

            if self.tool_manager is None:
                raise RuntimeError("No tool manager available for tool execution")

            # Execute tools in parallel using ToolManager's streaming API
            async for result in self.tool_manager.stream_execute_tools(
                calls=ctp_calls,
                on_tool_start=self._on_tool_start,
                max_concurrency=self.max_concurrency,
            ):
                await self._on_tool_result(result)
                if self._cancelled:
                    break  # type: ignore[unreachable]

        except asyncio.CancelledError:
            pass
        finally:
            # SAFETY NET: Ensure every tool_call_id has a matching result.
            # This prevents OpenAI 400 errors from orphaned tool_call_ids.
            self._ensure_all_tool_results(tool_calls)
            await self._finish_tool_calls()

    def cancel_running_tasks(self) -> None:
        """Cancel running tool execution."""
        self._cancelled = True

    async def _on_tool_start(self, call: CTPToolCall) -> None:
        """Callback when a tool starts execution."""
        meta = self._call_metadata.get(call.id)
        display_name = meta.display_name if meta else call.tool
        arguments = meta.arguments if meta else call.arguments

        # For dynamic tools, enhance the display
        if call.tool == DYNAMIC_TOOL_PROXY_NAME and "tool_name" in arguments:
            actual_tool = arguments["tool_name"]
            display_name = f"{actual_tool}"  # Just show the actual tool name
            # Show only the tool's arguments, not tool_name
            arguments = {k: v for k, v in arguments.items() if k != "tool_name"}

        logger.info(f"Executing tool: {call.tool} with args: {arguments}")
        await self.ui_manager.start_tool_execution(display_name, arguments)

    async def _on_tool_result(self, result: CTPToolResult) -> None:
        """Callback when a tool completes.

        ENHANCED: Now includes value binding system for dataflow tracking.
        - Binds numeric results to $vN identifiers
        - Tracks per-tool call counts for anti-thrash
        - Caches results for state tracking
        """
        meta = self._call_metadata.get(result.id)
        llm_tool_name = meta.llm_tool_name if meta else result.tool
        execution_tool_name = meta.execution_tool_name if meta else result.tool
        arguments = meta.arguments if meta else {}

        # For dynamic tools, extract the actual tool name for better logging/caching
        actual_tool_name = execution_tool_name
        actual_arguments = arguments
        if execution_tool_name == DYNAMIC_TOOL_PROXY_NAME and "tool_name" in arguments:
            actual_tool_name = arguments["tool_name"]
            actual_arguments = {k: v for k, v in arguments.items() if k != "tool_name"}

        success = result.is_success
        logger.info(
            f"Tool result ({actual_tool_name}): success={success}, error='{result.error}'"
        )

        tool_state = get_agent_tool_state(getattr(self.context, "agent_id", "default"))
        value_binding = None

        # Cache successful results and create value bindings
        if success and result.result is not None:
            # Extract the actual value from MCP response structure
            actual_result = self._extract_result_value(result.result)

            # Cache result for dedup
            tool_state.cache_result(actual_tool_name, actual_arguments, actual_result)
            logger.debug(f"Cached result for {actual_tool_name}: {actual_result}")

            # Create value binding ($v1, $v2, etc.) for dataflow tracking
            # Only bind "execution" tool results (not discovery tools)
            if not tool_state.is_discovery_tool(execution_tool_name):
                value_binding = tool_state.bind_value(
                    actual_tool_name, actual_arguments, actual_result
                )
                logger.info(
                    f"Bound value ${value_binding.id} = {actual_result} from {actual_tool_name}"
                )

            # Record numeric results for runaway detection
            if isinstance(actual_result, (int, float)):
                tool_state.record_numeric_result(float(actual_result))

        # Increment tool call counter for budget tracking (with tool name for split budgets)
        tool_state.increment_tool_call(execution_tool_name)

        # Record tool use for session-aware search boosting
        # Successful tools get boosted in future search results
        search_engine = get_search_engine()
        search_engine.record_tool_use(actual_tool_name, success=success)

        # Track per-tool call count for anti-thrash
        if not tool_state.is_discovery_tool(execution_tool_name):
            per_tool_status = tool_state.track_tool_call(actual_tool_name)
            if per_tool_status.requires_justification:
                logger.warning(
                    f"Tool {actual_tool_name} called {per_tool_status.call_count} times"
                )

        # For discovery tools, register any tools found in results
        # Also use result shape to refine tool classification
        if tool_state.is_discovery_tool(execution_tool_name):
            tool_state.classify_by_result(execution_tool_name, result.result)
            self._register_discovered_tools(
                tool_state, execution_tool_name, result.result
            )

        # Track transport failures
        self._track_transport_failures(success, result.error)

        # Format content for history - include value binding info
        if success:
            content = self._format_tool_response(result.result)
            # Append value binding info so model sees the $vN reference
            if value_binding:
                content = f"{content}\n\n**RESULT: ${value_binding.id} = {value_binding.typed_value}**"
        else:
            content = f"Error: {result.error}"

        # Add to conversation history
        self._add_tool_result_to_history(llm_tool_name, result.id, content)

        # Store successful tool results as VM pages so they survive eviction
        if success:
            await self._store_tool_result_as_vm_page(actual_tool_name, content)

        # Finish UI display
        await self.ui_manager.finish_tool_execution(result=content, success=success)

        # Verbose mode display (lazy import to keep Core/UI separation)
        if hasattr(self.ui_manager, "verbose_mode") and self.ui_manager.verbose_mode:
            from mcp_cli.tools.models import ToolCallResult
            from mcp_cli.display import display_tool_call_result

            display_result = ToolCallResult(
                tool_name=result.tool,
                success=success,
                result=result.result if success else None,
                error=result.error if not success else None,
            )
            display_tool_call_result(display_result, self.ui_manager.console)

        # Check for MCP App UI — launch in browser if available
        if success and self.tool_manager:
            await self._check_and_launch_app(actual_tool_name, result.result)

        # Dashboard bridge — broadcast tool result to browser clients
        if bridge := getattr(self.context, "dashboard_bridge", None):
            server_name = getattr(self.context, "tool_to_server_map", {}).get(
                actual_tool_name, ""
            )
            duration_ms: int | None = None
            if (
                meta is not None
                and hasattr(meta, "start_time")
                and meta.start_time is not None
            ):
                import time as _time

                duration_ms = int((_time.monotonic() - meta.start_time) * 1000)
            _dash_result = (
                actual_result if success and result.result is not None else None
            )
            meta_ui = (
                getattr(result.result, "structuredContent", None)
                if (success and result.result is not None)
                else None
            )
            try:
                await bridge.on_tool_result(
                    tool_name=actual_tool_name,
                    server_name=server_name,
                    result=_dash_result,
                    success=success,
                    error=result.error if not success else None,
                    duration_ms=duration_ms,
                    meta_ui=meta_ui,
                    call_id=result.id,
                    arguments=actual_arguments,
                )
            except Exception as _bridge_exc:
                logger.debug("Dashboard bridge on_tool_result error: %s", _bridge_exc)

    # Maximum chars of page content to return from a single page_fault.
    # Prevents oversized pages from flooding the conversation context.
    _VM_MAX_PAGE_CONTENT_CHARS = 2000

    def _build_page_content_blocks(
        self,
        page: Any,
        page_content: Any,
        truncated: bool,
        was_compressed: bool,
        source_tier: Any,
    ) -> str | list[dict[str, Any]]:
        """Build tool result content, using multi-block for multimodal pages.

        Returns a JSON string for text/structured, or a list of content blocks
        when the page contains an image URL or data URI that a multimodal model
        can re-analyze.
        """
        modality = getattr(page, "modality", None)
        modality_val = getattr(modality, "value", str(modality)) if modality else "text"
        compression = getattr(page, "compression_level", None)
        comp_name = getattr(compression, "name", "FULL") if compression else "FULL"

        # IMAGE with URL or data URI → multi-block content
        if modality_val == "image" and isinstance(page_content, str):
            if page_content.startswith(("http://", "https://", "data:")):
                text_block = f"Page {page.page_id} (image, {comp_name}):"
                if truncated:
                    text_block += " [content truncated]"
                blocks: list[dict[str, Any]] = [
                    {"type": "text", "text": text_block},
                    {
                        "type": "image_url",
                        "image_url": {"url": page_content, "detail": "low"},
                    },
                ]
                return blocks

        # All other cases: JSON string response
        response: dict[str, Any] = {
            "success": True,
            "page_id": page.page_id,
            "content": page_content,
            "modality": modality_val,
            "compression": comp_name,
            "source_tier": str(source_tier) if source_tier else None,
            "was_compressed": was_compressed,
            "truncated": truncated,
        }

        # Hint for short pages
        if isinstance(page_content, str) and len(page_content) < 120:
            response["note"] = (
                "Very short content — this may be a user "
                "request. Check the manifest for the "
                "[assistant] response page and fault that."
            )

        # Hint for compressed content
        if comp_name in ("ABSTRACT", "REFERENCE"):
            response["note"] = (
                f"This is a {comp_name.lower()} summary. "
                f'Use page_fault("{page.page_id}", target_level=0) '
                "for full content."
            )

        return json.dumps(response)

    async def _handle_memory_tool(
        self,
        tool_name: str,
        arguments: dict,
        llm_tool_name: str,
        call_id: str,
    ) -> None:
        """Execute a memory scope tool (remember, recall, forget).

        Memory tools are persistent-memory operations that bypass the MCP
        ToolManager and all guard checks.
        """
        store = getattr(self.context, "memory_store", None)
        if not store:
            self._add_tool_result_to_history(
                llm_tool_name, call_id, "Memory scopes not available."
            )
            return

        logger.info("Memory tool %s called with args: %s", tool_name, arguments)

        # Show tool call in UI
        try:
            self.ui_manager.print_tool_call(tool_name, arguments)
            await self.ui_manager.start_tool_execution(tool_name, arguments)
        except Exception as e:
            logger.debug("UI error displaying memory tool call: %s", e)

        from mcp_cli.memory.tools import handle_memory_tool

        result_text = await handle_memory_tool(store, tool_name, arguments)

        # Mark system prompt dirty so memory changes appear next turn
        if tool_name in ("remember", "forget"):
            if hasattr(self.context, "_system_prompt_dirty"):
                self.context._system_prompt_dirty = True

        # Finish UI display
        try:
            await self.ui_manager.finish_tool_execution(
                result=result_text, success=True
            )
        except Exception as e:
            logger.debug("UI error finishing memory tool display: %s", e)

        self._add_tool_result_to_history(llm_tool_name, call_id, result_text)

    async def _handle_plan_tool(
        self,
        tool_name: str,
        arguments: dict,
        llm_tool_name: str,
        call_id: str,
    ) -> None:
        """Execute a plan tool (plan_create, plan_execute, plan_create_and_execute).

        Plan tools are internal operations that bypass the MCP ToolManager
        and all guard checks. They use PlanningContext to generate and
        execute multi-step plans.
        """
        if not getattr(self.context, "_enable_plan_tools", False):
            self._add_tool_result_to_history(
                llm_tool_name, call_id, "Plan tools are not enabled."
            )
            return

        logger.info("Plan tool %s called with args: %s", tool_name, arguments)

        from mcp_cli.planning.context import PlanningContext
        from mcp_cli.planning.tools import handle_plan_tool

        # Lazy-create PlanningContext (cached on context object)
        planning_context = getattr(self.context, "_planning_context", None)
        if planning_context is None:
            planning_context = PlanningContext(self.context.tool_manager)
            self.context._planning_context = planning_context

        # Get model_manager for LLM-driven step execution
        model_manager = getattr(self.context, "model_manager", None)

        # Broadcast plan start to dashboard
        bridge = getattr(self.context, "dashboard_bridge", None)
        plan_title = arguments.get("goal", "Plan")
        if bridge is not None:
            try:
                await bridge.on_plan_update(
                    plan_id=call_id,
                    title=plan_title,
                    steps=[],
                    status="running",
                )
            except Exception as _e:
                logger.debug("Dashboard plan start update error: %s", _e)

        # Pass the UI manager so handle_plan_tool can show step-by-step progress
        result_text = await handle_plan_tool(
            tool_name,
            arguments,
            planning_context,
            model_manager,
            ui_manager=self.ui_manager,
        )

        # Re-fetch bridge in case it changed during plan execution
        bridge = getattr(self.context, "dashboard_bridge", None)
        if bridge is not None:
            try:
                import json as _json

                plan_result = (
                    _json.loads(result_text) if isinstance(result_text, str) else {}
                )
                steps = plan_result.get("steps", [])
                await bridge.on_plan_update(
                    plan_id=plan_result.get("plan_id", call_id),
                    title=plan_result.get("title", plan_title),
                    steps=[
                        {
                            "index": s.get("index", i),
                            "title": s.get("title", ""),
                            "tool": s.get("tool", ""),
                            "status": "complete" if s.get("success") else "failed",
                            "error": s.get("error"),
                        }
                        for i, s in enumerate(steps)
                    ],
                    status="complete" if plan_result.get("success") else "failed",
                    error=plan_result.get("error"),
                )
            except Exception as _e:
                logger.debug("Dashboard plan update error: %s", _e)

        self._add_tool_result_to_history(llm_tool_name, call_id, result_text)

    async def _handle_agent_tool(
        self,
        tool_name: str,
        arguments: dict,
        llm_tool_name: str,
        call_id: str,
    ) -> None:
        """Execute an agent orchestration tool (agent_spawn, agent_stop, etc.).

        Agent tools are internal operations that bypass the MCP ToolManager
        and all guard checks.  They delegate to the AgentManager stored on
        the ChatContext.
        """
        agent_manager = getattr(self.context, "agent_manager", None)
        if agent_manager is None:
            self._add_tool_result_to_history(
                llm_tool_name, call_id, "Agent tools are not enabled."
            )
            return

        logger.info("Agent tool %s called with args: %s", tool_name, arguments)

        from mcp_cli.agents.tools import handle_agent_tool

        caller_id = getattr(self.context, "agent_id", "default")
        result_text = await handle_agent_tool(
            tool_name, arguments, agent_manager, caller_agent_id=caller_id
        )

        self._add_tool_result_to_history(llm_tool_name, call_id, result_text)

    async def _handle_vm_tool(
        self,
        tool_name: str,
        arguments: dict,
        llm_tool_name: str,
        call_id: str,
    ) -> None:
        """Execute a VM tool (page_fault or search_pages) via MemoryManager.

        VM tools are internal memory operations that bypass the MCP ToolManager
        and all guard checks (dataflow tracking, $vN references, per-tool limits).

        Includes:
        - UI display so tool calls are visible in the chat output
        - Loop prevention: refuses to re-fault the same page_id twice
        - Content truncation for oversized pages
        """
        vm = getattr(getattr(self.context, "session", None), "vm", None)
        if not vm:
            self._add_tool_result_to_history(
                llm_tool_name, call_id, "Error: VM not available."
            )
            return

        logger.info(f"VM tool {tool_name} called with args: {arguments}")

        # Show tool call in UI (so page_fault calls are visible)
        try:
            self.ui_manager.print_tool_call(tool_name, arguments)
            await self.ui_manager.start_tool_execution(tool_name, arguments)
        except Exception as e:
            logger.debug("UI error displaying VM tool call: %s", e)

        content: str | list[dict[str, Any]] = ""
        success = True
        try:
            if tool_name == "page_fault":
                page_id = arguments.get("page_id", "")

                # Loop prevention: don't re-fault the same page
                if page_id in self._faulted_page_ids:
                    content = json.dumps(
                        {
                            "success": True,
                            "already_loaded": True,
                            "page_id": page_id,
                            "message": (
                                "This page was already loaded earlier in the "
                                "conversation. The content is in a previous "
                                "tool result message — use that directly."
                            ),
                        }
                    )
                else:
                    result = await vm.handle_fault(
                        page_id=page_id,
                        target_level=arguments.get("target_level", 2),
                    )
                    if result.success and result.page:
                        self._faulted_page_ids.add(page_id)
                        page_content = result.page.content
                        truncated = False

                        # Truncate oversized page content
                        if (
                            isinstance(page_content, str)
                            and len(page_content) > self._VM_MAX_PAGE_CONTENT_CHARS
                        ):
                            page_content = (
                                page_content[: self._VM_MAX_PAGE_CONTENT_CHARS]
                                + f"\n\n[truncated — original was "
                                f"{len(result.page.content)} chars]"
                            )
                            truncated = True

                        content = self._build_page_content_blocks(
                            page=result.page,
                            page_content=page_content,
                            truncated=truncated,
                            was_compressed=result.was_compressed,
                            source_tier=result.source_tier,
                        )
                    else:
                        success = False
                        content = json.dumps(
                            {
                                "success": False,
                                "error": result.error or "Page not found",
                            }
                        )

            elif tool_name == "search_pages":
                result = await vm.search_pages(
                    query=arguments.get("query", ""),
                    modality=arguments.get("modality"),
                    limit=arguments.get("limit", 5),
                )
                content = result.to_json()

            else:
                success = False
                content = json.dumps({"error": f"Unknown VM tool: {tool_name}"})

            logger.info(f"VM tool {tool_name} completed: {content[:200]}")

        except Exception as exc:
            logger.error(f"VM tool {tool_name} failed: {exc}")
            success = False
            content = json.dumps({"success": False, "error": str(exc)})

        self._add_tool_result_to_history(llm_tool_name, call_id, content)

        # Finish UI display
        try:
            ui_result = content if isinstance(content, str) else json.dumps(content)
            await self.ui_manager.finish_tool_execution(
                result=ui_result, success=success
            )
        except Exception as e:
            logger.debug("UI error finishing VM tool display: %s", e)

    async def _store_tool_result_as_vm_page(self, tool_name: str, content: str) -> None:
        """Store a tool result as a VM page so it survives eviction.

        Without this, tool results (weather forecasts, geocoding data, etc.)
        exist only as raw session events and vanish when _vm_filter_events()
        evicts older turns.  Creating a VM page ensures the content appears
        in the manifest and can be recalled via page_fault.

        Active for all VM modes — in passive mode pages still participate
        in working set budget tracking and context packing.
        """
        vm = getattr(getattr(self.context, "session", None), "vm", None)
        if not vm:
            return

        try:
            from chuk_ai_session_manager.memory.models import PageType

            page = vm.create_page(
                content=content,
                page_type=PageType.ARTIFACT,
                importance=0.4,
                hint=f"{tool_name}: {content[:100]}",
            )
            await vm.add_to_working_set(page)
            logger.debug(f"Stored tool result as VM page: {page.page_id}")
        except Exception as exc:
            logger.debug(f"Could not store tool result as VM page: {exc}")

    async def _check_and_launch_app(self, tool_name: str, result: Any) -> None:
        """Check if a tool has an MCP Apps UI and launch/update it.

        Handles two cases per the MCP Apps spec:
        1. Tool has resourceUri — reuse an existing app with the same URI
           (multiple tools can share one UI), or launch a new one.
        2. Tool has no resourceUri but returns a ui_patch — route the
           patch to an already-running app so it can update in place.
        """
        if not self.tool_manager:
            return

        try:
            tool_info = await self.tool_manager.get_tool_by_name(tool_name)
            app_host = self.tool_manager.app_host

            # ── Case 1: tool declares a resourceUri ──────────────────────
            if tool_info and tool_info.has_app_ui:
                resource_uri = tool_info.app_resource_uri
                server_name = tool_info.namespace

                # Collect fallback viewUrl from definition and result meta
                view_url = getattr(
                    tool_info, "app_view_url", None
                ) or self._extract_result_view_url(result)

                # Reuse existing app — check by tool name, then by URI
                bridge = app_host.get_bridge(tool_name)
                if bridge is None and resource_uri:
                    bridge = app_host.get_bridge_by_uri(resource_uri)

                if bridge is not None:
                    logger.info(
                        "Pushing result to existing app (tool=%s, uri=%s)",
                        tool_name,
                        resource_uri,
                    )
                    await bridge.push_tool_result(result)
                    return

                # No running app for this URI — launch a new one
                has_dashboard = (
                    getattr(self.context, "dashboard_bridge", None) is not None
                )
                logger.info("Tool %s has MCP App UI at %s", tool_name, resource_uri)
                app_info = await app_host.launch_app(
                    tool_name=tool_name,
                    resource_uri=resource_uri,
                    server_name=server_name,
                    tool_result=result,
                    open_browser=not has_dashboard,
                    view_url=view_url,
                )
                logger.info("MCP App opened at %s", app_info.url)

                # Notify dashboard to embed the app as a panel
                if has_dashboard:
                    dash_bridge = getattr(self.context, "dashboard_bridge", None)
                    if dash_bridge is not None:
                        try:
                            await dash_bridge.on_app_launched(app_info)
                        except Exception as exc:
                            logger.debug("Dashboard on_app_launched error: %s", exc)
                return

            # ── Case 2: no resourceUri — route ui_patch to running app ───
            if self._result_contains_patch(result):
                bridge = app_host.get_any_ready_bridge()
                if bridge is not None:
                    logger.info("Routing ui_patch from %s to running app", tool_name)
                    await bridge.push_tool_result(result)

        except ImportError:
            logger.warning(
                "MCP Apps requires websockets. Install with: pip install mcp-cli[apps]"
            )
        except Exception as e:
            logger.error("Failed to launch MCP App for %s: %s", tool_name, e)

    @staticmethod
    def _result_contains_patch(result: Any) -> bool:
        """Check whether a tool result carries a ui_patch structuredContent."""
        try:
            # Unwrap middleware/ToolCallResult wrappers
            raw = result
            seen: set[int] = set()
            while hasattr(raw, "result") and not isinstance(raw, (dict, str)):
                rid = id(raw)
                if rid in seen:
                    break
                seen.add(rid)
                raw = raw.result

            # Check Pydantic model with structuredContent attr
            if not isinstance(raw, dict) and hasattr(raw, "structuredContent"):
                sc = raw.structuredContent
                if isinstance(sc, dict) and sc.get("type") == "ui_patch":
                    return True

            if isinstance(raw, dict):
                # Direct structuredContent field
                sc = raw.get("structuredContent")
                if isinstance(sc, dict) and sc.get("type") == "ui_patch":
                    return True

                # Recover from content text blocks (MCP backwards-compat)
                content = raw.get("content")
                if content is not None and hasattr(content, "content"):
                    content = content.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if isinstance(text, str) and '"ui_patch"' in text:
                                try:
                                    parsed = json.loads(text)
                                    if isinstance(parsed, dict):
                                        if parsed.get("type") == "ui_patch":
                                            return True
                                        psc = parsed.get("structuredContent")
                                        if (
                                            isinstance(psc, dict)
                                            and psc.get("type") == "ui_patch"
                                        ):
                                            return True
                                except (json.JSONDecodeError, TypeError):
                                    pass
        except Exception as e:
            logger.debug("Error checking UI result: %s", e)
        return False

    @staticmethod
    def _extract_result_view_url(result: Any) -> str | None:
        """Extract viewUrl from a tool result's meta.ui if present.

        The tool result (not the definition) may carry
        ``meta.ui.viewUrl`` — a direct HTTPS URL for the app's UI.
        """
        try:
            raw = result
            seen: set[int] = set()
            while hasattr(raw, "result") and not isinstance(raw, (dict, str)):
                rid = id(raw)
                if rid in seen:
                    break
                seen.add(rid)
                raw = raw.result

            meta: Any = None
            if isinstance(raw, dict):
                meta = raw.get("meta") or raw.get("_meta")
            elif hasattr(raw, "meta"):
                meta = raw.meta

            if isinstance(meta, dict):
                ui = meta.get("ui", {})
                if isinstance(ui, dict):
                    url = ui.get("viewUrl")
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        return url
        except Exception:
            pass
        return None

    def _track_transport_failures(self, success: bool, error: str | None) -> None:
        """Track transport failures for recovery detection."""
        if not success and error:
            error_lower = error.lower()
            if any(pat in error_lower for pat in TRANSPORT_ERROR_PATTERNS):
                self._transport_failures += 1
                self._consecutive_transport_failures += 1

                if (
                    self._consecutive_transport_failures
                    >= DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES
                ):
                    logger.warning(
                        f"Detected {self._consecutive_transport_failures} consecutive transport failures. "
                        "The connection may need to be restarted."
                    )
            else:
                self._consecutive_transport_failures = 0
        else:
            self._consecutive_transport_failures = 0

    def _ensure_all_tool_results(self, tool_calls: list[Any]) -> None:
        """Ensure every tool_call_id in the assistant message has a matching result.

        This is a safety net that prevents OpenAI 400 errors caused by orphaned
        tool_call_ids. If any tool_call_id is missing a result (due to guard
        exceptions, silent failures, or interrupted execution), a placeholder
        error result is added.
        """
        for idx, call in enumerate(tool_calls):
            llm_tool_name, _, call_id = self._extract_tool_call_info(call, idx)
            if call_id not in self._result_ids_added:
                logger.warning(
                    f"Missing tool result for {llm_tool_name} ({call_id}), "
                    "adding error placeholder"
                )
                self._add_tool_result_to_history(
                    llm_tool_name,
                    call_id,
                    "Tool execution was interrupted or failed to complete.",
                )

    async def _finish_tool_calls(self) -> None:
        """Signal UI that all tool calls are complete."""
        if hasattr(self.ui_manager, "finish_tool_calls") and callable(
            self.ui_manager.finish_tool_calls
        ):
            try:
                import asyncio

                if asyncio.iscoroutinefunction(self.ui_manager.finish_tool_calls):
                    await self.ui_manager.finish_tool_calls()
                else:
                    self.ui_manager.finish_tool_calls()
            except Exception:
                logger.debug("finish_tool_calls() raised", exc_info=True)

    def _extract_tool_call_info(self, tool_call: Any, idx: int) -> tuple[str, Any, str]:
        """Extract tool name, arguments, and call ID from a tool call."""
        llm_tool_name = "unknown_tool"
        raw_arguments: Any = {}
        call_id = f"call_{idx}"

        if isinstance(tool_call, ToolCall):
            llm_tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments
            call_id = tool_call.id
            # DEBUG: Log raw arguments from model
            logger.debug(
                f"RAW MODEL TOOL CALL: {llm_tool_name}, "
                f"raw_arguments type={type(raw_arguments).__name__}, "
                f"value={raw_arguments}"
            )
        elif isinstance(tool_call, dict) and "function" in tool_call:
            logger.warning(
                f"Received dict tool call instead of ToolCall model: {type(tool_call)}"
            )
            fn = tool_call["function"]
            llm_tool_name = fn.get("name", "unknown_tool")
            raw_arguments = fn.get("arguments", {})
            call_id = tool_call.get("id", call_id)
        else:
            logger.error(f"Unrecognized tool call format: {type(tool_call)}")

        # Validate
        if not llm_tool_name or llm_tool_name == "unknown_tool":
            logger.error(f"Tool name is empty or unknown in tool call: {tool_call}")
            llm_tool_name = f"unknown_tool_{idx}"

        return llm_tool_name, raw_arguments, call_id

    def _parse_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        """Parse raw arguments into a dictionary."""
        try:
            if isinstance(raw_arguments, str):
                if not raw_arguments.strip():
                    return {}
                parsed: dict[str, Any] = json.loads(raw_arguments)
                return parsed
            result: dict[str, Any] = raw_arguments or {}
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in arguments: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing arguments: {e}")
            return {}

    def _extract_result_value(self, result: Any) -> Any:
        """Extract the actual value from MCP response structures.

        MCP responses can be nested in various ways:
        1. Direct value (number, string)
        2. Dict with "content" containing MCP ToolResult with .content list
        3. Dict with "success"/"result" wrapper
        4. List of content blocks [{type: "text", text: "..."}]
        5. Object with .content attribute (MCP CallToolResult)
        6. String representation like "content=[{'type': 'text', 'text': '4.2426'}]"

        This normalizes all formats to extract the core value for binding.
        """
        if result is None:
            return None

        # Handle string "None" (bug in some MCP responses)
        if result == "None" or result == "null":
            return None

        # Handle MCP CallToolResult object (has .content attribute)
        if hasattr(result, "content") and isinstance(result.content, list):
            return self._extract_from_content_list(result.content)

        # Handle dict structures
        if isinstance(result, dict):
            # Case: {"content": <MCP ToolResult object>}
            if "content" in result:
                content = result["content"]
                # MCP ToolResult has a .content attribute that's a list
                if hasattr(content, "content"):
                    return self._extract_from_content_list(content.content)
                # Or it might be a direct list
                if isinstance(content, list):
                    return self._extract_from_content_list(content)
                # Or a string
                if isinstance(content, str):
                    return self._try_parse_number(content)

            # Case: {"success": true, "result": ...}
            if "success" in result and "result" in result:
                inner = result["result"]
                # Recurse if inner is not None/string "None"
                if inner is not None and inner != "None":
                    return self._extract_result_value(inner)
                return None

            # Case: {"isError": false, "content": ...} (MCP response wrapper)
            if "isError" in result:
                if result.get("isError"):
                    return result.get("error") or result.get("content")
                return self._extract_result_value(result.get("content"))

            # Case: {"text": "value"} direct
            if "text" in result and isinstance(result["text"], str):
                return self._try_parse_number(result["text"])

        # Handle list of content blocks directly
        if isinstance(result, list):
            return self._extract_from_content_list(result)

        # Handle string that might be a serialized structure
        if isinstance(result, str):
            # Check for "content=[...]" string pattern (MCP SDK repr)
            if result.startswith("content=["):
                return self._parse_content_repr(result)
            # Try to parse as number
            return self._try_parse_number(result)

        # Direct numeric values
        if isinstance(result, (int, float)):
            return result

        return result

    def _extract_from_content_list(self, content_list: list) -> Any:
        """Extract value from a list of MCP content blocks."""
        if not content_list:
            return None

        text_parts = []
        for block in content_list:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == ContentBlockType.TEXT.value or block_type == "text":
                    text = block.get("text", "")
                    if text:
                        text_parts.append(text)
            # Handle TextContent objects
            elif hasattr(block, "type") and hasattr(block, "text"):
                if block.type == "text":
                    text_parts.append(block.text)

        if not text_parts:
            return None

        # Join all text parts
        combined = "\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]
        return self._try_parse_number(combined)

    def _parse_content_repr(self, repr_str: str) -> Any:
        """Parse a string like "content=[{'type': 'text', 'text': '4.2426'}]"."""
        import re

        # Try to extract the text value using regex
        match = re.search(r"'text':\s*'([^']*)'", repr_str)
        if match:
            text = match.group(1)
            return self._try_parse_number(text)

        # Try another pattern for double quotes
        match = re.search(r'"text":\s*"([^"]*)"', repr_str)
        if match:
            text = match.group(1)
            return self._try_parse_number(text)

        return repr_str

    def _try_parse_number(self, text: str) -> Any:
        """Try to parse a string as a number, return original if not possible."""
        if not text or not isinstance(text, str):
            return text

        text = text.strip()

        # Handle "None" string
        if text in ("None", "null", ""):
            return None

        # Try float (handles integers too)
        try:
            return float(text)
        except (ValueError, TypeError):
            pass

        return text

    def _format_tool_response(self, result: Any) -> str:
        """Format tool response for conversation history."""
        if isinstance(result, dict):
            # Check for MCP response structure
            if "content" in result and hasattr(result["content"], "content"):
                tool_result_content = result["content"].content
                if isinstance(tool_result_content, list):
                    text_parts = []
                    for block in tool_result_content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == ContentBlockType.TEXT.value
                        ):
                            text_parts.append(block.get("text", ""))
                    if text_parts:
                        return "\n".join(text_parts)

            try:
                return json.dumps(result, indent=2)
            except (TypeError, ValueError):
                return str(result)
        elif isinstance(result, list):
            try:
                return json.dumps(result, indent=2)
            except (TypeError, ValueError):
                return str(result)
        return str(result)

    def _truncate_tool_result(self, content: str, max_chars: int) -> str:
        """Truncate tool result if it exceeds max_chars.

        Keeps head + tail with a truncation notice in the middle.
        max_chars <= 0 disables truncation.
        """
        if max_chars <= 0 or len(content) <= max_chars:
            return content

        head = max_chars * 2 // 3
        tail = max_chars // 6
        omitted = len(content) - head - tail
        notice = (
            f"\n\n--- TRUNCATED: {omitted:,} chars omitted "
            f"({len(content):,} total) ---\n\n"
        )
        truncated = content[:head] + notice + content[-tail:]
        logger.info(
            f"Truncated tool result from {len(content):,} to {len(truncated):,} chars"
        )
        return truncated

    def _add_assistant_message_with_tool_calls(
        self, tool_calls: list[Any], reasoning_content: str | None = None
    ) -> None:
        """Add assistant message with all tool calls to history."""
        try:
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
            )
            self.context.inject_tool_message(assistant_msg)
            logger.debug(
                f"Added assistant message with {len(tool_calls)} tool calls to history"
            )
        except Exception as e:
            logger.error(f"Error adding assistant message to history: {e}")

    def _add_tool_result_to_history(
        self,
        llm_tool_name: str,
        call_id: str,
        content: str | list[dict[str, Any]],
    ) -> None:
        """Add tool result to conversation history."""
        try:
            from mcp_cli.config.defaults import (
                DEFAULT_MAX_TOOL_RESULT_CHARS,
                DEFAULT_CONTEXT_NOTICES_ENABLED,
            )

            # Multi-block content (e.g. image_url blocks) — skip truncation
            if isinstance(content, list):
                tool_msg = Message(
                    role=MessageRole.TOOL,
                    name=llm_tool_name,
                    content=content,
                    tool_call_id=call_id,
                )
                self.context.inject_tool_message(tool_msg)
                self._result_ids_added.add(call_id)
                logger.debug(
                    f"Added multi-block tool result to history: {llm_tool_name}"
                )
                return

            original_len = len(content)
            content = self._truncate_tool_result(content, DEFAULT_MAX_TOOL_RESULT_CHARS)

            # Notify LLM about truncation
            if (
                len(content) < original_len
                and DEFAULT_CONTEXT_NOTICES_ENABLED
                and hasattr(self.context, "add_context_notice")
            ):
                self.context.add_context_notice(
                    f"Tool result from '{llm_tool_name}' was truncated from "
                    f"{original_len:,} to {DEFAULT_MAX_TOOL_RESULT_CHARS:,} chars. "
                    "Consider requesting less data (smaller range, fewer fields, pagination)."
                )

            tool_msg = Message(
                role=MessageRole.TOOL,
                name=llm_tool_name,
                content=content,
                tool_call_id=call_id,
            )
            self.context.inject_tool_message(tool_msg)
            self._result_ids_added.add(call_id)
            logger.debug(f"Added tool result to conversation history: {llm_tool_name}")
        except Exception as e:
            logger.error(f"Error updating conversation history: {e}")

    def _add_cancelled_tool_to_history(
        self, llm_tool_name: str, call_id: str, raw_arguments: Any
    ) -> None:
        """Add cancelled tool call to conversation history."""
        try:
            # User cancellation message
            self.context.inject_tool_message(
                Message(
                    role=MessageRole.USER,
                    content=f"Cancel {llm_tool_name} tool execution.",
                )
            )

            arg_json = (
                json.dumps(raw_arguments)
                if isinstance(raw_arguments, dict)
                else str(raw_arguments or {})
            )

            # Assistant acknowledgement with tool call
            self.context.inject_tool_message(
                Message(
                    role=MessageRole.ASSISTANT,
                    content="User cancelled tool execution.",
                    tool_calls=[
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": llm_tool_name,
                                "arguments": arg_json,
                            },
                        }
                    ],
                )
            )

            # Tool result
            self.context.inject_tool_message(
                Message(
                    role=MessageRole.TOOL,
                    name=llm_tool_name,
                    content="Tool execution cancelled by user.",
                    tool_call_id=call_id,
                )
            )
        except Exception as e:
            logger.error(f"Error adding cancelled tool to history: {e}")

    def _get_server_url_for_tool(self, tool_name: str) -> str | None:
        """Look up the server URL for a tool using cached context data."""
        try:
            # Get server namespace from tool_to_server_map
            tool_map = getattr(self.context, "tool_to_server_map", None)
            server_info_list = getattr(self.context, "server_info", None)
            if not tool_map or not server_info_list:
                return None

            namespace = tool_map.get(tool_name)
            if not namespace:
                return None

            # Find matching ServerInfo by namespace
            for server in server_info_list:
                if server.namespace == namespace or server.name == namespace:
                    url: str | None = server.url
                    return url
        except Exception as e:
            logger.debug(f"Could not resolve server URL for {tool_name}: {e}")
        return None

    def _should_confirm_tool(
        self, tool_name: str, server_url: str | None = None
    ) -> bool:
        """Check if tool requires user confirmation."""
        try:
            prefs = get_preference_manager()
            # Trusted domain bypass — skip confirmation entirely
            if server_url and prefs.is_trusted_domain(server_url):
                return False
            return prefs.should_confirm_tool(tool_name)
        except Exception as e:
            logger.warning(f"Error checking tool confirmation preference: {e}")
            return True

    def _register_discovered_tools(
        self,
        tool_state: Any,
        discovery_tool: str,
        result: Any,
    ) -> None:
        """Register tools found by discovery operations.

        Extracts tool names from search_tools, list_tools, or get_tool_schema results
        and registers them as discovered for split budget enforcement.

        Args:
            tool_state: The ToolStateManager instance
            discovery_tool: Name of the discovery tool (search_tools, list_tools, get_tool_schema)
            result: Raw result from the discovery tool
        """
        if result is None:
            return

        try:
            # Extract tool names from various result formats
            tool_names: list[str] = []

            # Handle string result (might be JSON)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    return

            # Handle list of tools (from search_tools or list_tools)
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        # Common keys for tool name
                        for key in ("name", "tool_name", "tool"):
                            if key in item:
                                tool_names.append(str(item[key]))
                                break
                    elif isinstance(item, str):
                        tool_names.append(item)

            # Handle dict result (from get_tool_schema or single tool)
            elif isinstance(result, dict):
                # Direct tool schema
                if "name" in result:
                    tool_names.append(str(result["name"]))
                # Nested tools list
                elif "tools" in result and isinstance(result["tools"], list):
                    for tool in result["tools"]:
                        if isinstance(tool, dict) and "name" in tool:
                            tool_names.append(str(tool["name"]))
                        elif isinstance(tool, str):
                            tool_names.append(tool)
                # Content wrapper
                elif "content" in result:
                    # Recursively extract from content
                    self._register_discovered_tools(
                        tool_state, discovery_tool, result["content"]
                    )
                    return

            # Register each discovered tool
            for name in tool_names:
                if name:
                    tool_state.register_discovered_tool(name)
                    logger.debug(f"Discovered tool via {discovery_tool}: {name}")

        except Exception as e:
            logger.warning(f"Error registering discovered tools: {e}")
