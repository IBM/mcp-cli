"""Pydantic models for persistent memory scopes."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class MemoryScope(str, Enum):
    """Available persistent memory scopes."""

    WORKSPACE = "workspace"
    GLOBAL = "global"


class MemoryEntry(BaseModel):
    """A single memory entry."""

    key: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryScopeFile(BaseModel):
    """On-disk representation of a memory scope."""

    scope: MemoryScope
    entries: list[MemoryEntry] = Field(default_factory=list)
