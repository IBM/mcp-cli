# mcp_cli/chat/session_store.py
"""Session persistence — save and restore conversation sessions.

Pydantic-native. Sessions are stored as JSON files in a configurable directory.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mcp_cli.config.defaults import DEFAULT_AGENT_ID, DEFAULT_SESSIONS_DIR

logger = logging.getLogger(__name__)


class SessionMetadata(BaseModel):
    """Metadata for a saved session."""

    session_id: str
    agent_id: str = DEFAULT_AGENT_ID
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    provider: str = ""
    model: str = ""
    message_count: int = 0
    description: str = ""


class SessionData(BaseModel):
    """Complete session data for persistence."""

    metadata: SessionMetadata
    messages: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] | None = None


class SessionStore:
    """File-based session persistence.

    Stores sessions as JSON files in a configurable directory.
    """

    def __init__(
        self,
        sessions_dir: Path | None = None,
        agent_id: str = DEFAULT_AGENT_ID,
    ):
        if sessions_dir is None:
            sessions_dir = Path(DEFAULT_SESSIONS_DIR).expanduser()
        self.agent_id = agent_id
        # Agent-namespaced subdirectory
        self.sessions_dir = sessions_dir / agent_id
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        # Keep reference to root for backward-compat migration
        self._root_dir = sessions_dir

    def _session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        # Sanitize session_id to prevent path traversal
        safe_id = session_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.sessions_dir / f"{safe_id}.json"

    def save(self, data: SessionData) -> Path:
        """Save session data to disk.

        Args:
            data: Session data to save

        Returns:
            Path to the saved file
        """
        data.metadata.updated_at = datetime.now(timezone.utc).isoformat()
        data.metadata.message_count = len(data.messages)

        path = self._session_path(data.metadata.session_id)
        path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Session saved: {path}")
        return path

    def load(self, session_id: str) -> SessionData | None:
        """Load session data from disk.

        If the file isn't found in the agent-namespaced directory, checks the
        flat root directory for backward compatibility and auto-migrates.

        Args:
            session_id: Session ID to load

        Returns:
            SessionData or None if not found
        """
        path = self._session_path(session_id)
        if not path.exists():
            # Backward-compat: check flat root directory and migrate
            migrated = self._migrate_from_root(session_id)
            if migrated is None:
                logger.warning(f"Session not found: {session_id}")
                return None
            path = migrated

        try:
            raw = path.read_text(encoding="utf-8")
            data: SessionData = SessionData.model_validate_json(raw)
            return data
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def _migrate_from_root(self, session_id: str) -> Path | None:
        """Check flat root dir for a legacy session file and migrate it.

        Returns the new path if migration succeeded, None otherwise.
        """
        safe_id = session_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        legacy_path = self._root_dir / f"{safe_id}.json"
        if not legacy_path.exists() or not legacy_path.is_file():
            return None
        dest = self._session_path(session_id)
        try:
            import shutil

            shutil.move(str(legacy_path), str(dest))
            logger.info(f"Migrated session {session_id} from flat dir to {dest}")
            return dest
        except Exception as e:
            logger.warning(f"Failed to migrate session {session_id}: {e}")
            return None

    def list_sessions(self) -> list[SessionMetadata]:
        """List all saved sessions.

        Returns:
            List of session metadata, sorted by updated_at (newest first)
        """
        sessions: list[SessionMetadata] = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                raw = path.read_text(encoding="utf-8")
                data = SessionData.model_validate_json(raw)
                sessions.append(data.metadata)
            except Exception as e:
                logger.warning(f"Skipping corrupt session file {path}: {e}")

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a saved session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
