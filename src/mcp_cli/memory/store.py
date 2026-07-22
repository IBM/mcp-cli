"""Persistent memory scope store with file-based storage."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from mcp_cli.config.defaults import (
    DEFAULT_MEMORY_BASE_DIR,
    DEFAULT_MEMORY_MAX_ENTRIES_PER_SCOPE,
    DEFAULT_MEMORY_MAX_PROMPT_CHARS,
)
from mcp_cli.memory.models import MemoryEntry, MemoryScope, MemoryScopeFile

logger = logging.getLogger(__name__)


class MemoryScopeStore:
    """Manages persistent workspace and global memories with file-based storage."""

    def __init__(
        self,
        base_dir: Path | None = None,
        workspace_dir: str | None = None,
        max_entries: int = DEFAULT_MEMORY_MAX_ENTRIES_PER_SCOPE,
        max_prompt_chars: int = DEFAULT_MEMORY_MAX_PROMPT_CHARS,
    ) -> None:
        self._base_dir = Path(base_dir or DEFAULT_MEMORY_BASE_DIR).expanduser()
        self._workspace_hash = hashlib.sha256(
            (workspace_dir or str(Path.cwd())).encode()
        ).hexdigest()[:16]
        self._max_entries = max_entries
        self._max_prompt_chars = max_prompt_chars
        self._lock = threading.Lock()

        # Ensure directories exist
        self._base_dir.mkdir(parents=True, exist_ok=True)
        (self._base_dir / "workspace").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public CRUD
    # ------------------------------------------------------------------

    def remember(self, scope: MemoryScope, key: str, content: str) -> MemoryEntry:
        """Store or update a memory entry (upsert by key)."""
        with self._lock:
            data = self._load(scope)
            now = datetime.now(timezone.utc)

            # Upsert
            for entry in data.entries:
                if entry.key == key:
                    entry.content = content
                    entry.updated_at = now
                    self._save(scope, data)
                    return entry

            # New entry — enforce max
            if len(data.entries) >= self._max_entries:
                # Evict oldest by updated_at
                data.entries.sort(key=lambda e: e.updated_at)
                data.entries.pop(0)

            entry = MemoryEntry(
                key=key, content=content, created_at=now, updated_at=now
            )
            data.entries.append(entry)
            self._save(scope, data)
            return entry

    def recall(
        self,
        scope: MemoryScope | None = None,
        key: str | None = None,
        query: str | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve memory entries by key, query, or list all."""
        scopes = [scope] if scope else [MemoryScope.WORKSPACE, MemoryScope.GLOBAL]
        results: list[MemoryEntry] = []

        with self._lock:
            for s in scopes:
                data = self._load(s)
                if key:
                    results.extend(e for e in data.entries if e.key == key)
                elif query:
                    q = query.lower()
                    results.extend(
                        e
                        for e in data.entries
                        if q in e.key.lower() or q in e.content.lower()
                    )
                else:
                    results.extend(data.entries)

        return results

    def forget(self, scope: MemoryScope, key: str) -> bool:
        """Remove a memory entry by key. Returns True if found and removed."""
        with self._lock:
            data = self._load(scope)
            original_len = len(data.entries)
            data.entries = [e for e in data.entries if e.key != key]

            if len(data.entries) < original_len:
                self._save(scope, data)
                return True
            return False

    def list_entries(self, scope: MemoryScope) -> list[MemoryEntry]:
        """List all entries in a scope."""
        with self._lock:
            return list(self._load(scope).entries)

    def clear(self, scope: MemoryScope) -> int:
        """Clear all entries in a scope. Returns count of removed entries."""
        with self._lock:
            data = self._load(scope)
            count = len(data.entries)
            data.entries = []
            self._save(scope, data)
            return count

    # ------------------------------------------------------------------
    # System prompt injection
    # ------------------------------------------------------------------

    def format_for_system_prompt(self) -> str:
        """Format all memories as a markdown section for the system prompt.

        Returns empty string if no entries exist.
        """
        sections: list[str] = []

        with self._lock:
            for scope in (MemoryScope.WORKSPACE, MemoryScope.GLOBAL):
                data = self._load(scope)
                if not data.entries:
                    continue
                lines = [f"### {scope.value.title()} Memories"]
                for entry in data.entries:
                    lines.append(f"- **{entry.key}**: {entry.content}")
                sections.append("\n".join(lines))

        if not sections:
            return ""

        result = "## Persistent Memory\n\n" + "\n\n".join(sections)

        # Truncate if exceeds budget
        if len(result) > self._max_prompt_chars:
            result = result[: self._max_prompt_chars - 3] + "..."

        return result

    # ------------------------------------------------------------------
    # Persistence (private)
    # ------------------------------------------------------------------

    def _scope_path(self, scope: MemoryScope) -> Path:
        """Return the file path for a scope."""
        if scope == MemoryScope.GLOBAL:
            return self._base_dir / "global.json"
        return self._base_dir / "workspace" / f"{self._workspace_hash}.json"

    def _load(self, scope: MemoryScope) -> MemoryScopeFile:
        """Load scope data from disk. Returns empty file if not found."""
        path = self._scope_path(scope)
        if not path.exists():
            return MemoryScopeFile(scope=scope)

        try:
            with open(path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    raw = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            result: MemoryScopeFile = MemoryScopeFile.model_validate(raw)
            return result
        except Exception as exc:
            logger.warning("Failed to load memory scope %s: %s", scope.value, exc)
            return MemoryScopeFile(scope=scope)

    def _save(self, scope: MemoryScope, data: MemoryScopeFile) -> None:
        """Save scope data to disk with file locking."""
        path = self._scope_path(scope)
        try:
            with open(path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data.model_dump(mode="json"), f, indent=2, default=str)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            logger.error("Failed to save memory scope %s: %s", scope.value, exc)
