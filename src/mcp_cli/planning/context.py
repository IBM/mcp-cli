# src/mcp_cli/planning/context.py
"""PlanningContext — state container for plan operations.

Holds the graph store, plan registry, tool manager reference,
and provides convenience methods for plan CRUD and tool catalog access.

Wraps chuk-ai-planner's async PlanRegistry API, converting between
UniversalPlan objects and plain dicts for the command layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chuk_ai_planner.core.store.memory import InMemoryGraphStore
from chuk_ai_planner.core.planner.plan_registry import PlanRegistry
from chuk_ai_planner.core.planner.universal_plan import UniversalPlan

from mcp_cli.config.defaults import DEFAULT_PLANS_DIR

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)


class PlanningContext:
    """State container for plan operations.

    Centralizes access to the graph store, plan registry,
    and tool manager. Passed to the PlanRunner and PlanCommand.
    """

    def __init__(
        self,
        tool_manager: ToolManager,
        *,
        plans_dir: Path | None = None,
    ) -> None:
        """Initialize planning context.

        Args:
            tool_manager: The ToolManager for MCP tool execution and catalog.
            plans_dir: Directory for plan persistence. Defaults to ~/.mcp-cli/plans/
        """
        self.tool_manager = tool_manager
        self.plans_dir = plans_dir or Path(DEFAULT_PLANS_DIR).expanduser()
        self.graph_store = InMemoryGraphStore()
        self.plan_registry = PlanRegistry(str(self.plans_dir))

        # Ensure plans directory exists
        self.plans_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("PlanningContext initialized, plans_dir=%s", self.plans_dir)

    async def get_tool_catalog(self, provider: str = "openai") -> list[dict[str, Any]]:
        """Get the available tool catalog for LLM plan generation.

        Args:
            provider: LLM provider format for tool adaptation.

        Returns:
            List of tool definitions with name, description, and parameters.
        """
        try:
            result = await self.tool_manager.get_adapted_tools_for_llm(provider)
            # get_adapted_tools_for_llm returns (tools, namespace_map) tuple
            tools: list[dict[str, Any]] = (
                result[0] if isinstance(result, tuple) else result
            )
            return tools
        except Exception as e:
            logger.warning("Failed to get tool catalog: %s", e)
            return []

    async def get_tool_names(self) -> list[str]:
        """Get list of available tool names for plan validation.

        Returns:
            List of tool name strings.
        """
        try:
            all_tools = await self.tool_manager.get_all_tools()
            return [t.name for t in all_tools]
        except Exception as e:
            logger.warning("Failed to get tool names: %s", e)
            return []

    def _is_plan_id(self, registry_key: str) -> bool:
        """Check if a registry key is a real plan ID (not a checkpoint/state)."""
        return not registry_key.endswith("_state")

    def _plan_ids(self) -> set[str]:
        """Get all known plan IDs from both in-memory registry and disk.

        Excludes checkpoint/state files. Combines both sources because
        in-memory plans (created this session) may not be on disk yet,
        and disk plans (from previous sessions) may not be loaded yet.
        """
        known: set[str] = set()

        # In-memory registry keys
        for key in self.plan_registry.plans:
            if self._is_plan_id(key):
                known.add(key)

        # Plan files on disk ({plan_id}.json)
        if self.plans_dir.exists():
            for path in self.plans_dir.glob("*.json"):
                file_id = path.stem
                if self._is_plan_id(file_id):
                    known.add(file_id)

        return known

    async def list_plans(self) -> list[dict[str, Any]]:
        """List all saved plans as dicts.

        Returns:
            List of plan summaries with id, title, step count.
        """
        try:
            # Load all plans (populates the registry's in-memory cache)
            await self.plan_registry.get_all_plans()

            # Only return real plans, not checkpoint/state entries
            valid_ids = self._plan_ids()
            result = []
            for pid in sorted(valid_ids):
                plan = await self.plan_registry.get_plan(pid)
                if plan is not None:
                    plan_dict: dict[str, Any] = await plan.to_dict()
                    result.append(plan_dict)
            return result
        except Exception as e:
            logger.warning("Failed to list plans: %s", e)
            return []

    async def _resolve_plan_id(self, plan_id: str) -> str | None:
        """Resolve a full or prefix plan ID to the full UUID.

        Supports prefix matching: if the given ID is a prefix of exactly
        one plan's UUID, that plan is returned. This allows users to use
        the truncated IDs shown by ``/plan list``.

        Args:
            plan_id: Full UUID or unique prefix.

        Returns:
            Full plan UUID, or None if not found or ambiguous.
        """
        # Try exact match first (fast path)
        if self._is_plan_id(plan_id):
            plan = await self.plan_registry.get_plan(plan_id)
            if plan is not None:
                return plan_id

        # Ensure all plans are loaded, then prefix match
        await self.plan_registry.get_all_plans()
        valid_ids = self._plan_ids()
        matches = [pid for pid in valid_ids if pid.startswith(plan_id)]

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning(
                "Ambiguous plan prefix '%s' matches %d plans", plan_id, len(matches)
            )
        return None

    async def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        """Load a plan by ID (or unique prefix), returned as a dict.

        Reads from the saved JSON file directly to preserve depends_on
        fields that are lost through PlanRegistry's to_dict().

        Args:
            plan_id: The plan's UUID or a unique prefix.

        Returns:
            Plan dict or None if not found.
        """
        try:
            resolved_id = await self._resolve_plan_id(plan_id)
            if resolved_id is None:
                return None

            # Load directly from disk to preserve depends_on
            plan_path = self.plans_dir / f"{resolved_id}.json"
            if plan_path.exists():
                data: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
                return data

            # Fall back to registry (e.g., in-memory-only plans)
            plan = await self.plan_registry.get_plan(resolved_id)
            if plan is None:
                return None
            result: dict[str, Any] = await plan.to_dict()
            return result
        except Exception as e:
            logger.warning("Failed to load plan %s: %s", plan_id, e)
            return None

    async def get_plan_object(self, plan_id: str) -> UniversalPlan | None:
        """Load a plan by ID (or unique prefix) as a UniversalPlan object.

        Args:
            plan_id: The plan's UUID or a unique prefix.

        Returns:
            UniversalPlan or None if not found.
        """
        try:
            resolved_id = await self._resolve_plan_id(plan_id)
            if resolved_id is None:
                return None
            return await self.plan_registry.get_plan(resolved_id)
        except Exception as e:
            logger.warning("Failed to load plan %s: %s", plan_id, e)
            return None

    async def save_plan(self, plan: UniversalPlan) -> str:
        """Save a UniversalPlan to the registry.

        Args:
            plan: The plan to save.

        Returns:
            The plan ID.
        """
        plan_id: str = await self.plan_registry.register_plan(plan)
        return plan_id

    async def save_plan_from_dict(self, plan_dict: dict[str, Any]) -> str:
        """Build a UniversalPlan from a dict and save it.

        Saves the plan with dependency information preserved. The upstream
        PlanRegistry's to_dict() drops depends_on fields, so we overwrite
        the saved JSON with our enriched version that retains them.

        Args:
            plan_dict: Plan dict with title, steps, etc.

        Returns:
            The plan ID.
        """
        graph = InMemoryGraphStore()
        plan = await self._build_plan_from_dict(plan_dict, graph)
        plan_id: str = await self.plan_registry.register_plan(plan)

        # Re-save with depends_on preserved.
        # PlanRegistry.register_plan -> to_dict() drops depends_on,
        # so we patch the saved JSON to include the original dependencies.
        self._patch_saved_plan(plan_id, plan_dict)

        return plan_id

    def _patch_saved_plan(self, plan_id: str, original_dict: dict[str, Any]) -> None:
        """Patch the saved plan JSON to preserve depends_on from the original dict."""
        plan_path = self.plans_dir / f"{plan_id}.json"
        if not plan_path.exists():
            return

        try:
            saved: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
            original_steps = original_dict.get("steps", [])
            saved_steps = saved.get("steps", [])

            # Build mapping: 0-based position → actual saved step index.
            # The LLM generates 0-based depends_on but PlanRegistry
            # assigns 1-based string indices ("1", "2", ...).
            pos_to_index: dict[int, str] = {}
            for i, step in enumerate(saved_steps):
                pos_to_index[i] = str(step.get("index", str(i + 1)))

            for i, step in enumerate(saved_steps):
                if i < len(original_steps):
                    orig_step = original_steps[i]
                    if "depends_on" in orig_step:
                        # Convert 0-based positional refs to actual step indices
                        converted = []
                        for dep in orig_step["depends_on"]:
                            dep_int = int(dep) if isinstance(dep, (int, str)) else dep
                            if dep_int in pos_to_index:
                                converted.append(pos_to_index[dep_int])
                            else:
                                converted.append(str(dep))
                        step["depends_on"] = converted

            plan_path.write_text(
                json.dumps(saved, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to patch plan %s with dependencies: %s", plan_id, e)

    async def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan by ID (or unique prefix).

        Args:
            plan_id: The plan's UUID or a unique prefix.

        Returns:
            True if deleted, False if not found.
        """
        try:
            resolved_id = await self._resolve_plan_id(plan_id)
            if resolved_id is None:
                return False
            deleted: bool = self.plan_registry.delete_plan(resolved_id)
            return deleted
        except Exception as e:
            logger.warning("Failed to delete plan %s: %s", plan_id, e)
            return False

    async def _build_plan_from_dict(
        self, plan_dict: dict[str, Any], graph: InMemoryGraphStore
    ) -> UniversalPlan:
        """Build a UniversalPlan from a plan dict.

        Args:
            plan_dict: Dict with title, steps, tool_calls, dependencies.
            graph: Graph store to build the plan in.

        Returns:
            Constructed UniversalPlan.
        """
        plan = UniversalPlan(
            title=plan_dict.get("title", "Untitled"),
            description=plan_dict.get("description"),
            graph=graph,
            tags=plan_dict.get("tags"),
        )

        # Set variables if present
        for key, value in plan_dict.get("variables", {}).items():
            plan.set_variable(key, value)

        # Add each step
        for step in plan_dict.get("steps", []):
            tool_calls = step.get("tool_calls", [])
            # Support both tool_calls list and direct tool field
            if tool_calls:
                tc = tool_calls[0]
                tool = tc.get("name", "unknown")
                args = tc.get("args", {})
            else:
                tool = step.get("tool", "unknown")
                args = step.get("args", {})

            depends_on = step.get("depends_on", [])
            dep_indices = [str(d) for d in depends_on] if depends_on else []

            await plan.add_tool_step(
                title=step.get("title", "Untitled step"),
                tool=tool,
                args=args,
                depends_on=dep_indices,
                result_variable=step.get("result_variable"),
            )

        return plan
