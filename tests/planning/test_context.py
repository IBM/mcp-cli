# tests/planning/test_context.py
"""Tests for PlanningContext — state container with PlanRegistry round-trips."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import pytest

from mcp_cli.planning.context import PlanningContext


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class FakeToolInfo:
    """Minimal ToolInfo stub."""

    name: str


class FakeToolManager:
    """Minimal ToolManager stub for PlanningContext tests."""

    def __init__(self, tool_names: list[str] | None = None):
        self._tool_names = tool_names or ["read_file", "write_file", "search_code"]

    async def get_all_tools(self) -> list[FakeToolInfo]:
        return [FakeToolInfo(name=n) for n in self._tool_names]

    async def get_adapted_tools_for_llm(self, provider: str) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": n, "description": f"Tool: {n}"}}
            for n in self._tool_names
        ]


SAMPLE_PLAN_DICT = {
    "title": "Test Plan",
    "description": "A test plan for round-trip verification",
    "tags": ["test"],
    "variables": {"base_path": "/tmp"},
    "steps": [
        {
            "title": "Read file",
            "tool": "read_file",
            "args": {"path": "/tmp/test.py"},
            "depends_on": [],
            "result_variable": "file_content",
        },
        {
            "title": "Search code",
            "tool": "search_code",
            "args": {"query": "def main"},
            "depends_on": [0],
            "result_variable": "search_results",
        },
    ],
}


# ── Tests: Initialization ───────────────────────────────────────────────────


class TestPlanningContextInit:
    """Test PlanningContext initialization."""

    def test_default_plans_dir(self, tmp_path):
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")
        assert ctx.plans_dir.exists()

    def test_graph_store_created(self, tmp_path):
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")
        assert ctx.graph_store is not None

    def test_plan_registry_created(self, tmp_path):
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")
        assert ctx.plan_registry is not None


# ── Tests: Tool Catalog ─────────────────────────────────────────────────────


class TestPlanningContextToolCatalog:
    """Test tool catalog methods."""

    @pytest.mark.asyncio
    async def test_get_tool_names(self, tmp_path):
        tm = FakeToolManager(["alpha", "beta", "gamma"])
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")
        names = await ctx.get_tool_names()
        assert names == ["alpha", "beta", "gamma"]

    @pytest.mark.asyncio
    async def test_get_tool_catalog(self, tmp_path):
        tm = FakeToolManager(["read_file"])
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")
        catalog = await ctx.get_tool_catalog()
        assert len(catalog) == 1
        assert catalog[0]["function"]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_get_tool_names_handles_error(self, tmp_path):
        class BrokenToolManager:
            async def get_all_tools(self):
                raise RuntimeError("boom")

        ctx = PlanningContext(BrokenToolManager(), plans_dir=tmp_path / "plans")
        assert await ctx.get_tool_names() == []


# ── Tests: Plan CRUD ────────────────────────────────────────────────────────


class TestPlanningContextPlanCrud:
    """Test plan CRUD with real PlanRegistry round-trips."""

    @pytest.mark.asyncio
    async def test_list_plans_empty(self, tmp_path):
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")
        plans = await ctx.list_plans()
        assert plans == []

    @pytest.mark.asyncio
    async def test_get_plan_not_found(self, tmp_path):
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")
        plan = await ctx.get_plan("nonexistent-id")
        assert plan is None

    @pytest.mark.asyncio
    async def test_delete_plan_not_found(self, tmp_path):
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")
        assert await ctx.delete_plan("nonexistent-id") is False


# ── Tests: PlanRegistry Round-Trip ───────────────────────────────────────────


class TestPlanRegistryRoundTrip:
    """Verify that plans survive save → disk → load cycle."""

    @pytest.mark.asyncio
    async def test_save_and_load_plan(self, tmp_path):
        """Save a plan from dict, then load it back and verify contents."""
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        # Save
        plan_id = await ctx.save_plan_from_dict(SAMPLE_PLAN_DICT)
        assert plan_id is not None
        assert len(plan_id) > 0

        # Load back as dict
        loaded = await ctx.get_plan(plan_id)
        assert loaded is not None
        assert loaded["title"] == "Test Plan"
        assert len(loaded["steps"]) == 2

    @pytest.mark.asyncio
    async def test_save_and_list_plans(self, tmp_path):
        """Save a plan and verify it appears in the list."""
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_id = await ctx.save_plan_from_dict(SAMPLE_PLAN_DICT)

        plans = await ctx.list_plans()
        assert len(plans) >= 1

        plan_ids = [p.get("id") for p in plans]
        assert plan_id in plan_ids

    @pytest.mark.asyncio
    async def test_save_and_delete_plan(self, tmp_path):
        """Save a plan, then delete it, verify it's gone."""
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_id = await ctx.save_plan_from_dict(SAMPLE_PLAN_DICT)

        # Delete
        assert await ctx.delete_plan(plan_id) is True

        # Verify gone
        loaded = await ctx.get_plan(plan_id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_plan_persists_to_disk(self, tmp_path):
        """Plan JSON file should exist on disk after save."""
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_id = await ctx.save_plan_from_dict(SAMPLE_PLAN_DICT)

        json_path = tmp_path / "plans" / f"{plan_id}.json"
        assert json_path.exists()

        # Verify it's valid JSON
        import json

        data = json.loads(json_path.read_text())
        assert data["title"] == "Test Plan"

    @pytest.mark.asyncio
    async def test_fresh_context_loads_from_disk(self, tmp_path):
        """A new PlanningContext should discover plans saved by a previous one."""
        tm = FakeToolManager()

        # First context: save a plan
        ctx1 = PlanningContext(tm, plans_dir=tmp_path / "plans")
        plan_id = await ctx1.save_plan_from_dict(SAMPLE_PLAN_DICT)

        # Second context: should find the plan on disk
        ctx2 = PlanningContext(tm, plans_dir=tmp_path / "plans")
        loaded = await ctx2.get_plan(plan_id)
        assert loaded is not None
        assert loaded["title"] == "Test Plan"

    @pytest.mark.asyncio
    async def test_get_plan_object(self, tmp_path):
        """get_plan_object returns a UniversalPlan, not a dict."""
        from chuk_ai_planner.core.planner.universal_plan import UniversalPlan

        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_id = await ctx.save_plan_from_dict(SAMPLE_PLAN_DICT)

        plan_obj = await ctx.get_plan_object(plan_id)
        assert plan_obj is not None
        assert isinstance(plan_obj, UniversalPlan)
        assert plan_obj.title == "Test Plan"


# ── Tests: Prefix ID Resolution ────────────────────────────────────────────


class TestPlanPrefixResolution:
    """Test that get_plan, get_plan_object, and delete_plan support prefix IDs."""

    @pytest.mark.asyncio
    async def test_get_plan_by_prefix(self, tmp_path):
        """get_plan should resolve a unique prefix to the full plan ID."""
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_id = await ctx.save_plan_from_dict(SAMPLE_PLAN_DICT)

        # Use first 8 chars as prefix (like the UI shows)
        prefix = plan_id[:8]
        loaded = await ctx.get_plan(prefix)
        assert loaded is not None
        assert loaded["title"] == "Test Plan"

    @pytest.mark.asyncio
    async def test_get_plan_object_by_prefix(self, tmp_path):
        """get_plan_object should resolve a unique prefix."""
        from chuk_ai_planner.core.planner.universal_plan import UniversalPlan

        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_id = await ctx.save_plan_from_dict(SAMPLE_PLAN_DICT)
        prefix = plan_id[:8]

        plan_obj = await ctx.get_plan_object(prefix)
        assert plan_obj is not None
        assert isinstance(plan_obj, UniversalPlan)

    @pytest.mark.asyncio
    async def test_delete_plan_by_prefix(self, tmp_path):
        """delete_plan should resolve a unique prefix."""
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_id = await ctx.save_plan_from_dict(SAMPLE_PLAN_DICT)
        prefix = plan_id[:8]

        assert await ctx.delete_plan(prefix) is True
        assert await ctx.get_plan(plan_id) is None

    @pytest.mark.asyncio
    async def test_full_id_still_works(self, tmp_path):
        """Full plan IDs should still work as before."""
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_id = await ctx.save_plan_from_dict(SAMPLE_PLAN_DICT)
        loaded = await ctx.get_plan(plan_id)
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_nonexistent_prefix_returns_none(self, tmp_path):
        """A prefix that matches nothing should return None."""
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        loaded = await ctx.get_plan("zzzzzzz")
        assert loaded is None


# ── Tests: Build Plan from Dict ─────────────────────────────────────────────


class TestBuildPlanFromDict:
    """Test _build_plan_from_dict with different input formats."""

    @pytest.mark.asyncio
    async def test_build_with_tool_field(self, tmp_path):
        """Plans with 'tool' field (from PlanAgent) should build correctly."""
        from chuk_ai_planner.core.store.memory import InMemoryGraphStore

        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_dict = {
            "title": "Simple Plan",
            "steps": [
                {"title": "Step 1", "tool": "read_file", "args": {"path": "/tmp/x"}},
            ],
        }

        graph = InMemoryGraphStore()
        plan = await ctx._build_plan_from_dict(plan_dict, graph)
        assert plan.title == "Simple Plan"

    @pytest.mark.asyncio
    async def test_build_with_tool_calls_field(self, tmp_path):
        """Plans with 'tool_calls' field (from registry) should build correctly."""
        from chuk_ai_planner.core.store.memory import InMemoryGraphStore

        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_dict = {
            "title": "TC Plan",
            "steps": [
                {
                    "title": "Step 1",
                    "tool_calls": [
                        {"id": "tc-1", "name": "read_file", "args": {"path": "/tmp/x"}}
                    ],
                },
            ],
        }

        graph = InMemoryGraphStore()
        plan = await ctx._build_plan_from_dict(plan_dict, graph)
        assert plan.title == "TC Plan"

    @pytest.mark.asyncio
    async def test_build_with_variables(self, tmp_path):
        """Plan variables should be set on the UniversalPlan."""
        from chuk_ai_planner.core.store.memory import InMemoryGraphStore

        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_dict = {
            "title": "Var Plan",
            "variables": {"base_url": "http://localhost"},
            "steps": [
                {"title": "Fetch", "tool": "fetch", "args": {"url": "${base_url}"}},
            ],
        }

        graph = InMemoryGraphStore()
        plan = await ctx._build_plan_from_dict(plan_dict, graph)
        assert plan.variables.get("base_url") == "http://localhost"


# ── Tests: Dependency Preservation ────────────────────────────────────────


class TestDependencyPreservation:
    """Test that depends_on survives the save → disk → load cycle.

    UniversalPlan.to_dict() drops depends_on fields. PlanningContext
    patches the saved JSON to preserve them, and get_plan reads from
    disk directly to avoid the lossy to_dict() path.

    The LLM generates 0-based depends_on but PlanRegistry assigns
    1-based string indices. _patch_saved_plan converts 0→"1", 1→"2", etc.
    """

    @pytest.mark.asyncio
    async def test_depends_on_preserved_on_disk(self, tmp_path):
        """After save_plan_from_dict, the JSON file should contain depends_on
        with 0-based refs converted to match saved step indices."""
        import json

        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_dict = {
            "title": "Dependency Plan",
            "steps": [
                {
                    "title": "Step A",
                    "tool": "read_file",
                    "args": {"path": "/a"},
                    "depends_on": [],
                    "result_variable": "a_result",
                },
                {
                    "title": "Step B (depends on A)",
                    "tool": "search_code",
                    "args": {"query": "test"},
                    "depends_on": [0],
                    "result_variable": "b_result",
                },
            ],
        }

        plan_id = await ctx.save_plan_from_dict(plan_dict)

        # Read the JSON file directly
        json_path = tmp_path / "plans" / f"{plan_id}.json"
        data = json.loads(json_path.read_text())

        # Step 0 should have empty depends_on
        assert data["steps"][0].get("depends_on") == []
        # Step 1 should depend on step 1 (0-based 0 → 1-based "1")
        step1_deps = data["steps"][1].get("depends_on")
        assert step1_deps == ["1"]

    @pytest.mark.asyncio
    async def test_depends_on_survives_load(self, tmp_path):
        """get_plan should return plan dicts with converted depends_on intact."""
        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        plan_dict = {
            "title": "Chain Plan",
            "steps": [
                {
                    "title": "First",
                    "tool": "read_file",
                    "args": {},
                    "depends_on": [],
                },
                {
                    "title": "Second",
                    "tool": "write_file",
                    "args": {},
                    "depends_on": [0],
                },
                {
                    "title": "Third",
                    "tool": "search_code",
                    "args": {},
                    "depends_on": [0, 1],
                },
            ],
        }

        plan_id = await ctx.save_plan_from_dict(plan_dict)

        # Load through get_plan (should read from disk, preserving depends_on)
        loaded = await ctx.get_plan(plan_id)
        assert loaded is not None
        assert loaded["steps"][0].get("depends_on") == []
        # 0-based [0] → 1-based ["1"]
        assert loaded["steps"][1].get("depends_on") == ["1"]
        # 0-based [0, 1] → 1-based ["1", "2"]
        assert loaded["steps"][2].get("depends_on") == ["1", "2"]

    @pytest.mark.asyncio
    async def test_depends_on_survives_fresh_context(self, tmp_path):
        """A new PlanningContext should load depends_on from disk."""
        tm = FakeToolManager()

        # Save with first context
        ctx1 = PlanningContext(tm, plans_dir=tmp_path / "plans")
        plan_dict = {
            "title": "Cross-session Plan",
            "steps": [
                {"title": "A", "tool": "read_file", "args": {}, "depends_on": []},
                {"title": "B", "tool": "write_file", "args": {}, "depends_on": [0]},
            ],
        }
        plan_id = await ctx1.save_plan_from_dict(plan_dict)

        # Load with fresh context (simulating new session)
        ctx2 = PlanningContext(tm, plans_dir=tmp_path / "plans")
        loaded = await ctx2.get_plan(plan_id)
        assert loaded is not None
        # 0-based [0] → 1-based ["1"]
        assert loaded["steps"][1].get("depends_on") == ["1"]

    @pytest.mark.asyncio
    async def test_state_files_excluded_from_plan_ids(self, tmp_path):
        """Checkpoint _state.json files should not appear as plans."""
        import json

        tm = FakeToolManager()
        ctx = PlanningContext(tm, plans_dir=tmp_path / "plans")

        # Save a real plan
        plan_id = await ctx.save_plan_from_dict(SAMPLE_PLAN_DICT)

        # Create a fake state file (like PlanRunner saves)
        state_path = tmp_path / "plans" / f"{plan_id}_state.json"
        state_path.write_text(json.dumps({"completed_steps": [0]}))

        # _plan_ids should only return the real plan
        ids = ctx._plan_ids()
        assert plan_id in ids
        assert f"{plan_id}_state" not in ids

        # list_plans should not include the state file
        plans = await ctx.list_plans()
        plan_ids = [p.get("id") for p in plans]
        assert plan_id in plan_ids
        assert f"{plan_id}_state" not in plan_ids
