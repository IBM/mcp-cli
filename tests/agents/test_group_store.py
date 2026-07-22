# tests/agents/test_group_store.py
"""Unit tests for group save/restore."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from mcp_cli.agents.config import AgentConfig
from mcp_cli.agents.group_store import (
    list_groups,
    load_group_manifest,
    save_group,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_manager(*agent_configs):
    """Create a mock AgentManager with the given agent configs."""
    mgr = MagicMock()
    snapshots = {}
    statuses = []

    for cfg in agent_configs:
        ctx = MagicMock()
        ctx.agent_id = cfg.agent_id
        ctx.session_id = f"session-{cfg.agent_id}"
        ctx.conversation_history = []
        snapshots[cfg.agent_id] = {"config": cfg, "context": ctx}
        statuses.append(
            {
                "agent_id": cfg.agent_id,
                "name": cfg.name,
                "role": cfg.role,
                "status": "active",
            }
        )

    mgr.list_agents.return_value = statuses
    mgr.get_agent_snapshot.side_effect = lambda aid: snapshots.get(aid)
    return mgr


# ---------------------------------------------------------------------------
# TestSaveGroup
# ---------------------------------------------------------------------------


class TestSaveGroup:
    @pytest.mark.asyncio
    async def test_save_empty_group(self, tmp_path):
        mgr = _make_agent_manager()
        result = await save_group(mgr, description="empty", group_dir=tmp_path)
        assert result == tmp_path
        manifest = json.loads((tmp_path / "group.json").read_text())
        assert manifest["description"] == "empty"
        assert manifest["agents"] == []

    @pytest.mark.asyncio
    async def test_save_with_agents(self, tmp_path):
        cfg_a = AgentConfig(agent_id="a", name="Agent A", role="worker")
        cfg_b = AgentConfig(agent_id="b", name="Agent B", role="researcher")
        mgr = _make_agent_manager(cfg_a, cfg_b)

        result = await save_group(mgr, description="test group", group_dir=tmp_path)
        manifest = json.loads((result / "group.json").read_text())
        assert len(manifest["agents"]) == 2
        agent_ids = {a["agent_id"] for a in manifest["agents"]}
        assert agent_ids == {"a", "b"}

        # Check session files exist
        assert (tmp_path / "a" / "session.json").exists()
        assert (tmp_path / "b" / "session.json").exists()

    @pytest.mark.asyncio
    async def test_save_preserves_config_fields(self, tmp_path):
        cfg = AgentConfig(
            agent_id="x",
            name="X",
            role="coder",
            model="gpt-4",
            provider="openai",
            parent_agent_id="main",
        )
        mgr = _make_agent_manager(cfg)
        await save_group(mgr, group_dir=tmp_path)

        manifest = json.loads((tmp_path / "group.json").read_text())
        agent = manifest["agents"][0]
        assert agent["agent_id"] == "x"
        assert agent["role"] == "coder"
        assert agent["model"] == "gpt-4"
        assert agent["parent_agent_id"] == "main"


# ---------------------------------------------------------------------------
# TestLoadGroupManifest
# ---------------------------------------------------------------------------


class TestLoadGroupManifest:
    def test_load_valid(self, tmp_path):
        data = {
            "group_id": "g1",
            "created_at": "2026-01-01T00:00:00Z",
            "description": "test",
            "agents": [],
        }
        (tmp_path / "group.json").write_text(json.dumps(data))
        manifest = load_group_manifest(tmp_path)
        assert manifest["group_id"] == "g1"
        assert manifest["description"] == "test"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_group_manifest(tmp_path)


# ---------------------------------------------------------------------------
# TestListGroups
# ---------------------------------------------------------------------------


class TestListGroups:
    def test_empty_dir(self, tmp_path):
        assert list_groups(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert list_groups(tmp_path / "nope") == []

    def test_lists_groups(self, tmp_path):
        # Create two groups
        for name in ["group-1", "group-2"]:
            d = tmp_path / name
            d.mkdir()
            data = {
                "group_id": name,
                "description": f"desc-{name}",
                "created_at": "2026-01-01",
                "agents": [{"agent_id": "a"}],
            }
            (d / "group.json").write_text(json.dumps(data))

        groups = list_groups(tmp_path)
        assert len(groups) == 2
        ids = {g["group_id"] for g in groups}
        assert ids == {"group-1", "group-2"}
        for g in groups:
            assert g["agent_count"] == 1

    def test_skips_invalid(self, tmp_path):
        # Create one valid and one invalid group
        valid = tmp_path / "valid"
        valid.mkdir()
        (valid / "group.json").write_text(
            json.dumps({"group_id": "valid", "agents": []})
        )

        invalid = tmp_path / "invalid"
        invalid.mkdir()
        (invalid / "group.json").write_text("not json")

        groups = list_groups(tmp_path)
        assert len(groups) == 1
        assert groups[0]["group_id"] == "valid"
