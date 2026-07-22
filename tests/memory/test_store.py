# tests/memory/test_store.py
"""Tests for MemoryScopeStore."""

import pytest
from pathlib import Path

from mcp_cli.memory.models import MemoryScope
from mcp_cli.memory.store import MemoryScopeStore


@pytest.fixture
def store(tmp_path: Path) -> MemoryScopeStore:
    """Create a store with tmp dir."""
    return MemoryScopeStore(base_dir=tmp_path, workspace_dir="/test/project")


class TestRemember:
    def test_creates_entry(self, store: MemoryScopeStore):
        entry = store.remember(MemoryScope.GLOBAL, "framework", "pytest")
        assert entry.key == "framework"
        assert entry.content == "pytest"
        assert entry.created_at is not None
        assert entry.updated_at is not None

    def test_upserts_existing(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "db", "sqlite")
        entry = store.remember(MemoryScope.GLOBAL, "db", "postgres")
        assert entry.content == "postgres"
        # Should still be only one entry
        entries = store.list_entries(MemoryScope.GLOBAL)
        assert len(entries) == 1

    def test_workspace_and_global_separate(self, store: MemoryScopeStore):
        store.remember(MemoryScope.WORKSPACE, "key", "workspace_val")
        store.remember(MemoryScope.GLOBAL, "key", "global_val")

        ws = store.list_entries(MemoryScope.WORKSPACE)
        gl = store.list_entries(MemoryScope.GLOBAL)
        assert len(ws) == 1
        assert ws[0].content == "workspace_val"
        assert len(gl) == 1
        assert gl[0].content == "global_val"


class TestRecall:
    def test_all(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "a", "alpha")
        store.remember(MemoryScope.WORKSPACE, "b", "beta")

        entries = store.recall()
        assert len(entries) == 2

    def test_by_key(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "target", "found it")
        store.remember(MemoryScope.GLOBAL, "other", "not this")

        entries = store.recall(scope=MemoryScope.GLOBAL, key="target")
        assert len(entries) == 1
        assert entries[0].content == "found it"

    def test_by_query(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "lang", "Python is great")
        store.remember(MemoryScope.GLOBAL, "editor", "vim")

        entries = store.recall(scope=MemoryScope.GLOBAL, query="python")
        assert len(entries) == 1
        assert entries[0].key == "lang"

    def test_by_query_matches_key(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "python_version", "3.12")

        entries = store.recall(scope=MemoryScope.GLOBAL, query="python")
        assert len(entries) == 1

    def test_no_results(self, store: MemoryScopeStore):
        entries = store.recall(scope=MemoryScope.GLOBAL, key="nonexistent")
        assert entries == []

    def test_recall_all_scopes(self, store: MemoryScopeStore):
        store.remember(MemoryScope.WORKSPACE, "ws_key", "ws_val")
        store.remember(MemoryScope.GLOBAL, "gl_key", "gl_val")

        # No scope = search both
        entries = store.recall(query="val")
        assert len(entries) == 2


class TestForget:
    def test_existing(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "temp", "delete me")
        assert store.forget(MemoryScope.GLOBAL, "temp") is True
        assert store.list_entries(MemoryScope.GLOBAL) == []

    def test_nonexistent(self, store: MemoryScopeStore):
        assert store.forget(MemoryScope.GLOBAL, "nope") is False


class TestClear:
    def test_clear(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "a", "1")
        store.remember(MemoryScope.GLOBAL, "b", "2")

        count = store.clear(MemoryScope.GLOBAL)
        assert count == 2
        assert store.list_entries(MemoryScope.GLOBAL) == []


class TestFormatForSystemPrompt:
    def test_empty(self, store: MemoryScopeStore):
        assert store.format_for_system_prompt() == ""

    def test_with_entries(self, store: MemoryScopeStore):
        store.remember(MemoryScope.GLOBAL, "lang", "Python")
        store.remember(MemoryScope.WORKSPACE, "db", "PostgreSQL")

        prompt = store.format_for_system_prompt()
        assert "## Persistent Memory" in prompt
        assert "### Workspace Memories" in prompt
        assert "### Global Memories" in prompt
        assert "**lang**" in prompt
        assert "**db**" in prompt

    def test_truncation(self, tmp_path: Path):
        store = MemoryScopeStore(
            base_dir=tmp_path, workspace_dir="/test", max_prompt_chars=50
        )
        store.remember(MemoryScope.GLOBAL, "key", "x" * 200)

        prompt = store.format_for_system_prompt()
        assert len(prompt) <= 50
        assert prompt.endswith("...")


class TestWorkspaceScoping:
    def test_different_workspaces(self, tmp_path: Path):
        store_a = MemoryScopeStore(base_dir=tmp_path, workspace_dir="/project/a")
        store_b = MemoryScopeStore(base_dir=tmp_path, workspace_dir="/project/b")

        store_a.remember(MemoryScope.WORKSPACE, "framework", "django")
        store_b.remember(MemoryScope.WORKSPACE, "framework", "flask")

        assert store_a.list_entries(MemoryScope.WORKSPACE)[0].content == "django"
        assert store_b.list_entries(MemoryScope.WORKSPACE)[0].content == "flask"

    def test_global_shared(self, tmp_path: Path):
        store_a = MemoryScopeStore(base_dir=tmp_path, workspace_dir="/project/a")
        store_b = MemoryScopeStore(base_dir=tmp_path, workspace_dir="/project/b")

        store_a.remember(MemoryScope.GLOBAL, "editor", "vim")

        entries = store_b.list_entries(MemoryScope.GLOBAL)
        assert len(entries) == 1
        assert entries[0].content == "vim"


class TestMaxEntries:
    def test_evicts_oldest(self, tmp_path: Path):
        store = MemoryScopeStore(
            base_dir=tmp_path, workspace_dir="/test", max_entries=3
        )
        store.remember(MemoryScope.GLOBAL, "first", "1")
        store.remember(MemoryScope.GLOBAL, "second", "2")
        store.remember(MemoryScope.GLOBAL, "third", "3")

        # This should evict "first" (oldest by updated_at)
        store.remember(MemoryScope.GLOBAL, "fourth", "4")

        entries = store.list_entries(MemoryScope.GLOBAL)
        assert len(entries) == 3
        keys = {e.key for e in entries}
        assert "first" not in keys
        assert "fourth" in keys


class TestPersistence:
    def test_survives_reload(self, tmp_path: Path):
        store1 = MemoryScopeStore(base_dir=tmp_path, workspace_dir="/test")
        store1.remember(MemoryScope.GLOBAL, "persist", "data")

        # Create new store pointing to same dir
        store2 = MemoryScopeStore(base_dir=tmp_path, workspace_dir="/test")
        entries = store2.list_entries(MemoryScope.GLOBAL)
        assert len(entries) == 1
        assert entries[0].content == "data"
